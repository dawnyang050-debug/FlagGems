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

import pytest
import torch

import flag_gems
from flag_gems.ops.wgrad_gemm_accum import wgrad_gemm_accum_fp16, wgrad_gemm_accum_fp32

from . import accuracy_utils as utils

try:
    import fused_weight_gradient_mlp_cuda as apex_wgrad

    HAS_APEX_WGRAD = True
except ImportError:
    HAS_APEX_WGRAD = False

WGRAD_SHAPES_2D = [
    (4, 16, 32),
    (8, 32, 64),
    (16, 64, 128),
]

WGRAD_SHAPES_3D = [
    (2, 4, 16, 32),
]

FP32_ACCUM_INPUT_DTYPES = [torch.float32, torch.float16]
if utils.bf16_is_supported:
    FP32_ACCUM_INPUT_DTYPES.append(torch.bfloat16)

# fp32 activations use cuBLAS tensor-op GEMM (Apex path); CPU fp64 matmul is not
# the right reference on TF32-capable GPUs.  Those cases are covered by vs_apex.
FP32_ACCUM_CPU_REF_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    FP32_ACCUM_CPU_REF_DTYPES.append(torch.bfloat16)
FP32_ACCUM_3D_APEX_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    FP32_ACCUM_3D_APEX_DTYPES.append(torch.bfloat16)

FP16_ACCUM_INPUT_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    FP16_ACCUM_INPUT_DTYPES.append(torch.bfloat16)

# Inner GEMM dimension K = collapsed batch size; scale atol like other BLAS tests.
DEFAULT_ATOL = 1e-4
TF32_OFF_ATOL = 1e-6


def _collapse_to_2d(input_tensor, grad_output):
    if input_tensor.dim() > 2:
        input_2d = input_tensor.view(-1, input_tensor.size(-1))
    else:
        input_2d = input_tensor
    if grad_output.dim() > 2:
        grad_output_2d = grad_output.view(-1, grad_output.size(-1))
    else:
        grad_output_2d = grad_output
    return input_2d, grad_output_2d


def _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, main_grad):
    """Independent CPU fp64 matmul, accumulated in fp32 (matches main_grad dtype)."""
    ref_input = input_tensor.detach().cpu().double()
    ref_grad_output = grad_output.detach().cpu().double()
    input_2d, grad_output_2d = _collapse_to_2d(ref_input, ref_grad_output)
    wgrad_fp32 = (grad_output_2d.t().contiguous() @ input_2d).float()
    main_grad_fp32 = main_grad.detach().cpu().float().clone()
    main_grad_fp32.add_(wgrad_fp32)
    main_grad.copy_(main_grad_fp32.to(device=main_grad.device, dtype=main_grad.dtype))


def _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, main_grad, dtype):
    """Independent CPU fp64 matmul reference, cast to half storage."""
    ref_input = input_tensor.detach().cpu().double()
    ref_grad_output = grad_output.detach().cpu().double()
    input_2d, grad_output_2d = _collapse_to_2d(ref_input, ref_grad_output)
    wgrad = grad_output_2d.t().contiguous() @ input_2d
    main_grad_cpu = main_grad.detach().cpu().clone()
    main_grad_cpu.add_(wgrad.to(dtype))
    main_grad.copy_(main_grad_cpu)


def _assert_vs_cpu_ref(res, ref, dtype, *, reduce_dim):
    # Independent CPU fp64 reference; always compare on CPU.
    utils.gems_assert_close(
        res.cpu(),
        ref.cpu(),
        dtype,
        reduce_dim=reduce_dim,
        atol=DEFAULT_ATOL,
    )


def _assert_vs_apex(res, ref, dtype, *, reduce_dim):
    """Apex is the deployment target; compare on device, strict tolerance."""
    utils.gems_assert_close(
        res, ref, dtype, reduce_dim=reduce_dim, atol=DEFAULT_ATOL
    )


def _with_seed(seed: int):
    """Set deterministic seed for reproducible coverage cases."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_with_tf32_disabled(fn):
    """Run function with TF32 disabled, then restore global flags."""
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return fn()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def _as_non_contiguous_2d(contiguous_2d: torch.Tensor) -> torch.Tensor:
    """Build a non-contiguous (B, F) view with identical values."""
    batch, feat = contiguous_2d.shape
    nc = torch.empty(
        feat,
        batch,
        dtype=contiguous_2d.dtype,
        device=contiguous_2d.device,
    ).transpose(0, 1)
    nc.copy_(contiguous_2d)
    assert not nc.is_contiguous()
    assert nc.shape == contiguous_2d.shape
    return nc


def _as_non_contiguous_3d(contiguous_3d: torch.Tensor) -> torch.Tensor:
    """Build a non-contiguous (D0, D1, F) view with identical values."""
    dim0, dim1, feat = contiguous_3d.shape
    nc = torch.empty(
        dim1,
        dim0,
        feat,
        dtype=contiguous_3d.dtype,
        device=contiguous_3d.device,
    ).transpose(0, 1)
    nc.copy_(contiguous_3d)
    assert not nc.is_contiguous()
    assert nc.shape == contiguous_3d.shape
    return nc


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_2d(batch, in_features, out_features, dtype):
    _with_seed(20260721)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    ref_main_grad = main_grad.clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main_grad)

    _assert_vs_cpu_ref(
        res_main_grad, ref_main_grad, torch.float32, reduce_dim=batch
    )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_3D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_3d(dim0, dim1, in_features, out_features, dtype):
    _with_seed(20260722)
    input_tensor = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    ref_main_grad = main_grad.clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main_grad)

    _assert_vs_cpu_ref(
        res_main_grad,
        ref_main_grad,
        torch.float32,
        reduce_dim=dim0 * dim1,
    )


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_2d(batch, in_features, out_features, dtype):
    _with_seed(20260723)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    ref_main_grad = utils.to_reference(main_grad, True).clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, ref_main_grad, dtype)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_main_grad)

    _assert_vs_cpu_ref(res_main_grad, ref_main_grad, dtype, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_accumulates_twice():
    """Verify += semantics across two micro-batch calls, not overwrite."""
    _with_seed(20260724)
    batch, in_features, out_features = 4, 16, 32
    dtype = torch.float16

    inp1 = torch.randn(batch, in_features, dtype=dtype, device=flag_gems.device)
    gout1 = torch.randn(batch, out_features, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(batch, in_features, dtype=dtype, device=flag_gems.device)
    gout2 = torch.randn(batch, out_features, dtype=dtype, device=flag_gems.device)

    base = torch.zeros(out_features, in_features, dtype=torch.float32, device="cpu")

    ref_main = base.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(inp1, gout1, ref_main)
    _ref_wgrad_gemm_accum_fp32_cpu(inp2, gout2, ref_main)

    res_main = torch.zeros(
        out_features, in_features, dtype=torch.float32, device=flag_gems.device
    )
    wgrad_gemm_accum_fp32(inp1, gout1, res_main)
    wgrad_gemm_accum_fp32(inp2, gout2, res_main)

    _assert_vs_cpu_ref(res_main, ref_main, torch.float32, reduce_dim=2 * batch)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_from_zero_main_grad():
    _with_seed(20260725)
    batch, in_features, out_features = 8, 32, 64
    input_tensor = torch.randn(
        (batch, in_features), dtype=torch.float16, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=torch.float16, device=flag_gems.device
    )

    ref_main = torch.zeros(out_features, in_features, dtype=torch.float32)
    res_main = torch.zeros(
        out_features, in_features, dtype=torch.float32, device=flag_gems.device
    )

    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main)

    _assert_vs_cpu_ref(res_main, ref_main, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_invalid_main_grad_shape():
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(4, 32, dtype=torch.float16, device=flag_gems.device)
    # Expected main_grad shape is (32, 16); use transposed (16, 32).
    main_grad = torch.zeros(16, 32, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="main_grad shape mismatch"):
        wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_invalid_main_grad_dtype():
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(4, 32, dtype=torch.float16, device=flag_gems.device)
    main_grad = torch.zeros(32, 16, dtype=torch.float16, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="main_grad must be float32"):
        wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not HAS_APEX_WGRAD,
    reason="Apex fused_weight_gradient_mlp_cuda not installed",
)
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex(batch, in_features, out_features, dtype):
    _with_seed(20260726)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex(gems_main_grad, apex_main_grad, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not HAS_APEX_WGRAD,
    reason="Apex fused_weight_gradient_mlp_cuda not installed",
)
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_3D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_3D_APEX_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_3d(
    dim0, dim1, in_features, out_features, dtype
):
    _with_seed(20260727)
    input_tensor = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex(
        gems_main_grad, apex_main_grad, torch.float32, reduce_dim=dim0 * dim1
    )


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not HAS_APEX_WGRAD,
    reason="Apex fused_weight_gradient_mlp_cuda not installed",
)
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex(batch, in_features, out_features, dtype):
    _with_seed(20260728)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex(gems_main_grad, apex_main_grad, dtype, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize(
    "batch, in_features, out_features",
    [
        (4, 3072, 4096),  # small batch, large hidden
        (257, 129, 257),  # non-aligned dimensions
        (1024, 64, 64),  # large K accumulation
    ],
)
def test_wgrad_gemm_accum_fp32_cpu_ref_strict_with_tf32_off(
    batch, in_features, out_features
):
    """Mathematical strictness check for fp32 inputs under full-fp32 GEMM."""
    _with_seed(20260729)
    input_tensor = torch.randn(
        (batch, in_features), dtype=torch.float32, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=torch.float32, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    ref_main_grad = main_grad.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main_grad)

    res_main_grad = main_grad.clone()
    _run_with_tf32_disabled(
        lambda: wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main_grad)
    )

    utils.gems_assert_close(
        res_main_grad.cpu(),
        ref_main_grad.cpu(),
        torch.float32,
        reduce_dim=batch,
        atol=TF32_OFF_ATOL,
    )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
@pytest.mark.parametrize("layout", ["input_nc", "grad_output_nc", "both_nc"])
def test_wgrad_gemm_accum_fp32_2d_non_contiguous(
    batch, in_features, out_features, dtype, layout
):
    """Non-contiguous 2D inputs must match contiguous results and CPU ref."""
    _with_seed(20260730)
    input_c = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output_c = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    input_tensor = input_c
    grad_output = grad_output_c
    if layout in ("input_nc", "both_nc"):
        input_tensor = _as_non_contiguous_2d(input_c)
    if layout in ("grad_output_nc", "both_nc"):
        grad_output = _as_non_contiguous_2d(grad_output_c)

    ref_main = main_grad_seed.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main)

    res_contig = main_grad_seed.clone()
    wgrad_gemm_accum_fp32(input_c, grad_output_c, res_contig)

    res_nc = main_grad_seed.clone()
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_nc)

    _assert_vs_cpu_ref(res_nc, ref_main, torch.float32, reduce_dim=batch)
    utils.gems_assert_close(
        res_nc,
        res_contig,
        torch.float32,
        reduce_dim=batch,
        atol=DEFAULT_ATOL,
    )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_3D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_3d_non_contiguous(
    dim0, dim1, in_features, out_features, dtype
):
    """Non-contiguous 3D inputs must match contiguous results and CPU ref."""
    _with_seed(20260731)
    input_c = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output_c = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    input_tensor = _as_non_contiguous_3d(input_c)
    grad_output = _as_non_contiguous_3d(grad_output_c)

    ref_main = main_grad_seed.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main)

    res_contig = main_grad_seed.clone()
    wgrad_gemm_accum_fp32(input_c, grad_output_c, res_contig)

    res_nc = main_grad_seed.clone()
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_nc)

    _assert_vs_cpu_ref(
        res_nc, ref_main, torch.float32, reduce_dim=dim0 * dim1
    )
    utils.gems_assert_close(
        res_nc,
        res_contig,
        torch.float32,
        reduce_dim=dim0 * dim1,
        atol=DEFAULT_ATOL,
    )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not HAS_APEX_WGRAD,
    reason="Apex fused_weight_gradient_mlp_cuda not installed",
)
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D[:1])
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
@pytest.mark.parametrize("layout", ["input_nc", "grad_output_nc", "both_nc"])
def test_wgrad_gemm_accum_fp32_vs_apex_non_contiguous(
    batch, in_features, out_features, dtype, layout
):
    """Non-contiguous inputs must match Apex on the same logical tensors."""
    _with_seed(20260732)
    input_c = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output_c = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    input_tensor = input_c
    grad_output = grad_output_c
    if layout in ("input_nc", "both_nc"):
        input_tensor = _as_non_contiguous_2d(input_c)
    if layout in ("grad_output_nc", "both_nc"):
        grad_output = _as_non_contiguous_2d(grad_output_c)

    apex_main = main_grad_seed.clone()
    gems_main = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main)

    _assert_vs_apex(gems_main, apex_main, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D[:1])
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
@pytest.mark.parametrize("layout", ["input_nc", "grad_output_nc", "both_nc"])
def test_wgrad_gemm_accum_fp16_2d_non_contiguous(
    batch, in_features, out_features, dtype, layout
):
    """fp16 accum path: non-contiguous inputs match contiguous and CPU ref."""
    _with_seed(20260733)
    input_c = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output_c = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    input_tensor = input_c
    grad_output = grad_output_c
    if layout in ("input_nc", "both_nc"):
        input_tensor = _as_non_contiguous_2d(input_c)
    if layout in ("grad_output_nc", "both_nc"):
        grad_output = _as_non_contiguous_2d(grad_output_c)

    ref_main = main_grad_seed.clone()
    _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, ref_main, dtype)

    res_contig = main_grad_seed.clone()
    wgrad_gemm_accum_fp16(input_c, grad_output_c, res_contig)

    res_nc = main_grad_seed.clone()
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_nc)

    _assert_vs_cpu_ref(res_nc, ref_main, dtype, reduce_dim=batch)
    utils.gems_assert_close(
        res_nc, res_contig, dtype, reduce_dim=batch, atol=DEFAULT_ATOL
    )
