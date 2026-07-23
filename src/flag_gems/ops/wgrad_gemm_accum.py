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
leading dimensions).  Half-precision activations with fp32 ``main_grad`` use
FlagGems ``addmm_dtype_out`` (fused fp32 accum).  Same-dtype paths use
``torch.addmm`` (cuBLAS) so layout can keep ``OP_T`` without materializing the
transpose, matching Apex's ``cublasGemmEx`` shape.
"""

import logging

import torch

import flag_gems

from .addmm import addmm_dtype_out

logger = logging.getLogger(__name__)


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


def _fused_addmm_cublas(
    main_grad: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> None:
    """Fused ``main_grad += mat1 @ mat2`` via cuBLAS (Apex-aligned).

    Apex calls ``cublasGemmEx`` with ``CUBLAS_GEMM_DEFAULT_TENSOR_OP`` and does
    not force TF32 off.  Use PyTorch ``addmm`` with default backend settings so
    numerics track the Apex extension on the same device.

    ``mat1`` may be a transpose view (``grad_output.t()``); cuBLAS can use
    ``OP_T`` without materializing a contiguous copy.
    """
    torch.addmm(main_grad, mat1, mat2, beta=1, alpha=1, out=main_grad)


def _matmul_operands(
    grad_output_2d: torch.Tensor, input_2d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(grad_output.T, input)`` for GEMM.

    Always densify first, then take a transpose *view*.  That way contiguous and
    non-contiguous callers share one cuBLAS ``OP_T`` path (bit-identical), instead
    of mixing ``OP_T`` with a materialized ``t().contiguous()`` which diverges in
    fp16/bf16.
    """
    if not grad_output_2d.is_contiguous():
        grad_output_2d = grad_output_2d.contiguous()
    if not input_2d.is_contiguous():
        input_2d = input_2d.contiguous()
    return grad_output_2d.t(), input_2d


def _accum_wgrad(
    grad_output_2d: torch.Tensor,
    input_2d: torch.Tensor,
    main_grad: torch.Tensor,
    *,
    fp32_accum: bool,
) -> None:
    grad_output_T, input_c = _matmul_operands(grad_output_2d, input_2d)

    if fp32_accum and input_c.dtype in (torch.float16, torch.bfloat16):
        # Half activations + fp32 main_grad: fused Triton addmm (no full fp32 cast).
        addmm_dtype_out(
            main_grad,
            grad_output_T,
            input_c,
            torch.float32,
            beta=1,
            alpha=1,
            out=main_grad,
        )
    else:
        # Same-dtype (fp32 input or fp16/bf16 accum): cuBLAS fused addmm.
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
