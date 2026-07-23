# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight-gradient GEMM with in-place accumulation (Apex-aligned).

Matches Apex ``fused_weight_gradient_mlp_cuda`` semantics used by Megatron
``LinearWithGradAccumulationAndAsyncCommunication`` when
``gradient_accumulation_fusion`` is enabled.

Each update performs ``main_grad += grad_output.T @ input`` (after collapsing
leading dimensions).

``wgrad_gemm_accum_fp32`` (including half/bf16 activations into fp32
``main_grad``) calls ``cublasGemmEx`` with the same layout / dtype / algo as
Apex.  ``wgrad_gemm_accum_fp16`` uses ``torch.addmm`` (cuBLAS) for same-dtype
accumulation.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
from functools import lru_cache

import torch

import flag_gems

logger = logging.getLogger(__name__)

# cublasOperation_t
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1

# cudaDataType (library_types.h)
_CUDA_R_32F = 0
_CUDA_R_16F = 2
_CUDA_R_16BF = 14

# cublasGemmAlgo_t — same as Apex CUBLAS_GEMM_DEFAULT_TENSOR_OP
_CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
_CUBLAS_STATUS_SUCCESS = 0


def _collapse_to_2d(input: torch.Tensor, grad_output: torch.Tensor):
    if input.dim() > 2:
        input_2d = input.reshape(-1, input.size(-1))
    else:
        input_2d = input

    if grad_output.dim() > 2:
        grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
    else:
        grad_output_2d = grad_output

    if input_2d.size(0) != grad_output_2d.size(0):
        raise RuntimeError(
            "input and grad_output must have the same number of rows after collapse"
        )

    return input_2d, grad_output_2d


def _validate_device(*tensors: torch.Tensor) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise RuntimeError("All tensors must be on the same device")
    device = devices.pop()
    if device.type != flag_gems.device:
        raise RuntimeError(
            f"Expected tensors on {flag_gems.device}, but got {device.type}"
        )


@lru_cache(None)
def _load_cublas() -> ctypes.CDLL:
    """Load libcublas from Torch / NVIDIA wheel paths, then system names."""
    candidates: list[str] = []
    torch_dir = os.path.dirname(torch.__file__)
    candidates.extend(glob.glob(os.path.join(torch_dir, "lib", "libcublas.so*")))
    candidates.extend(
        glob.glob(os.path.join(torch_dir, "lib", "**", "libcublas.so*"), recursive=True)
    )
    try:
        import nvidia.cublas.lib as cublas_pkg  # type: ignore

        pkg_dir = os.path.dirname(cublas_pkg.__file__)
        candidates.extend(glob.glob(os.path.join(pkg_dir, "libcublas.so*")))
    except Exception:
        pass
    candidates.extend(["libcublas.so.12", "libcublas.so.11", "libcublas.so"])

    seen: set[str] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise RuntimeError(
        "Unable to load libcublas; required for wgrad_gemm_accum_fp32 GemmEx path"
    )


def _blas_handle() -> int:
    if hasattr(torch.cuda, "current_blas_handle"):
        return int(torch.cuda.current_blas_handle())
    return int(torch._C._cuda_getCurrentBlasHandle())


def _cuda_dtype(tensor: torch.Tensor) -> int:
    if tensor.dtype == torch.float16:
        return _CUDA_R_16F
    if tensor.dtype == torch.bfloat16:
        return _CUDA_R_16BF
    if tensor.dtype == torch.float32:
        return _CUDA_R_32F
    raise RuntimeError(f"Unsupported dtype for cublasGemmEx wgrad: {tensor.dtype}")


def _cublas_wgrad_gemm_accum_fp32(
    input_2d: torch.Tensor,
    grad_output_2d: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """Apex ``wgrad_gemm_accum_fp32_cuda`` layout via ``cublasGemmEx``.

    Computes ``main_grad += grad_output.T @ input`` without materializing the
    transpose: ``OP_N(input)`` x ``OP_T(grad_output)`` into fp32 ``main_grad``.
    """
    if main_grad.dtype != torch.float32:
        raise RuntimeError("main_grad must be float32 for GemmEx fp32-accum path")
    if input_2d.dtype != grad_output_2d.dtype:
        raise RuntimeError(
            "input and grad_output dtype must match, "
            f"got {input_2d.dtype} vs {grad_output_2d.dtype}"
        )

    input_2d = input_2d.contiguous()
    grad_output_2d = grad_output_2d.contiguous()
    weight_is_main = main_grad.is_contiguous()
    weight = main_grad if weight_is_main else main_grad.contiguous()

    hidden_dim = int(input_2d.size(0))
    in_dim = int(input_2d.size(1))
    out_dim = int(grad_output_2d.size(1))
    if int(grad_output_2d.size(0)) != hidden_dim:
        raise RuntimeError("input/grad_output row mismatch after collapse")
    if tuple(weight.shape) != (out_dim, in_dim):
        raise RuntimeError(
            f"main_grad shape mismatch: expected ({out_dim}, {in_dim}), "
            f"got {tuple(weight.shape)}"
        )

    lib = _load_cublas()
    gemm_ex = lib.cublasGemmEx
    gemm_ex.restype = ctypes.c_int

    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(1.0)
    a_type = _cuda_dtype(input_2d)

    status = gemm_ex(
        ctypes.c_void_p(_blas_handle()),
        ctypes.c_int(_CUBLAS_OP_N),
        ctypes.c_int(_CUBLAS_OP_T),
        ctypes.c_int(in_dim),
        ctypes.c_int(out_dim),
        ctypes.c_int(hidden_dim),
        ctypes.byref(alpha),
        ctypes.c_void_p(input_2d.data_ptr()),
        ctypes.c_int(a_type),
        ctypes.c_int(in_dim),
        ctypes.c_void_p(grad_output_2d.data_ptr()),
        ctypes.c_int(a_type),
        ctypes.c_int(out_dim),
        ctypes.byref(beta),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_int(_CUDA_R_32F),
        ctypes.c_int(in_dim),
        ctypes.c_int(_CUDA_R_32F),
        ctypes.c_int(_CUBLAS_GEMM_DEFAULT_TENSOR_OP),
    )
    if status != _CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"cublasGemmEx failed with status {status}")

    if not weight_is_main:
        main_grad.copy_(weight)


def _matmul_operands(
    grad_output_2d: torch.Tensor, input_2d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grad_output.T, input)`` for same-dtype ``torch.addmm``.

    Densify first, then take a transpose view so contiguous / non-contiguous
    callers share one cuBLAS ``OP_T`` path.
    """
    if not grad_output_2d.is_contiguous():
        grad_output_2d = grad_output_2d.contiguous()
    if not input_2d.is_contiguous():
        input_2d = input_2d.contiguous()
    return grad_output_2d.t(), input_2d


def _fused_addmm_cublas(
    main_grad: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> None:
    """Same-dtype fused ``main_grad += mat1 @ mat2`` via PyTorch cuBLAS addmm."""
    torch.addmm(main_grad, mat1, mat2, beta=1, alpha=1, out=main_grad)


def _accum_wgrad(
    grad_output_2d: torch.Tensor,
    input_2d: torch.Tensor,
    main_grad: torch.Tensor,
    *,
    fp32_accum: bool,
) -> None:
    if fp32_accum:
        # Match Apex fused_weight_gradient path (half/bf16/fp32 -> fp32 C).
        _cublas_wgrad_gemm_accum_fp32(input_2d, grad_output_2d, main_grad)
        return

    grad_output_T, input_c = _matmul_operands(grad_output_2d, input_2d)
    _fused_addmm_cublas(main_grad, grad_output_T, input_c)


def wgrad_gemm_accum_fp32(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """Accumulate weight gradient into ``main_grad`` using fp32 storage."""
    logger.debug("GEMS WGRAD_GEMM_ACCUM_FP32")

    _validate_device(input, grad_output, main_grad)

    if main_grad.dtype != torch.float32:
        raise RuntimeError(
            "main_grad must be float32 for wgrad_gemm_accum_fp32, "
            f"but got {main_grad.dtype}"
        )
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise RuntimeError(
            "Unsupported input dtype for wgrad_gemm_accum_fp32: "
            f"{input.dtype}"
        )
    if grad_output.dtype != input.dtype:
        raise RuntimeError(
            "grad_output dtype must match input dtype, "
            f"but got {grad_output.dtype} vs {input.dtype}"
        )

    input_2d, grad_output_2d = _collapse_to_2d(input, grad_output)
    out_dim = grad_output_2d.size(-1)
    in_dim = input_2d.size(-1)
    if main_grad.shape != (out_dim, in_dim):
        raise RuntimeError(
            "main_grad shape mismatch: expected "
            f"({out_dim}, {in_dim}), got {tuple(main_grad.shape)}"
        )

    _accum_wgrad(
        grad_output_2d,
        input_2d,
        main_grad,
        fp32_accum=True,
    )


def wgrad_gemm_accum_fp16(
    input: torch.Tensor,
    grad_output: torch.Tensor,
    main_grad: torch.Tensor,
) -> None:
    """Accumulate weight gradient into ``main_grad`` using fp16/bf16 storage."""
    logger.debug("GEMS WGRAD_GEMM_ACCUM_FP16")

    _validate_device(input, grad_output, main_grad)

    if main_grad.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "main_grad must be float16 or bfloat16 for wgrad_gemm_accum_fp16, "
            f"but got {main_grad.dtype}"
        )
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "Unsupported input dtype for wgrad_gemm_accum_fp16: "
            f"{input.dtype}"
        )
    if grad_output.dtype != input.dtype:
        raise RuntimeError(
            "grad_output dtype must match input dtype, "
            f"but got {grad_output.dtype} vs {input.dtype}"
        )
    if main_grad.dtype != input.dtype:
        raise RuntimeError(
            "main_grad dtype must match input dtype, "
            f"but got {main_grad.dtype} vs {input.dtype}"
        )

    input_2d, grad_output_2d = _collapse_to_2d(input, grad_output)
    out_dim = grad_output_2d.size(-1)
    in_dim = input_2d.size(-1)
    if main_grad.shape != (out_dim, in_dim):
        raise RuntimeError(
            "main_grad shape mismatch: expected "
            f"({out_dim}, {in_dim}), got {tuple(main_grad.shape)}"
        )

    _accum_wgrad(
        grad_output_2d,
        input_2d,
        main_grad,
        fp32_accum=False,
    )
