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

FP16_ACCUM_INPUT_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    FP16_ACCUM_INPUT_DTYPES.append(torch.bfloat16)

# Inner GEMM dimension K = collapsed batch size; scale atol like other BLAS tests.
DEFAULT_ATOL = 1e-4


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
    """Independent CPU fp64 matmul reference (not GPU/Triton)."""
    ref_input = utils.to_reference(input_tensor, True).double()
    ref_grad_output = utils.to_reference(grad_output, True).double()
    input_2d, grad_output_2d = _collapse_to_2d(ref_input, ref_grad_output)
    wgrad = grad_output_2d.t().contiguous() @ input_2d
    main_grad.add_(wgrad.to(torch.float32))


def _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, main_grad, dtype):
    """Independent CPU fp64 matmul reference, cast to half storage."""
    ref_input = utils.to_reference(input_tensor, True).double()
    ref_grad_output = utils.to_reference(grad_output, True).double()
    input_2d, grad_output_2d = _collapse_to_2d(ref_input, ref_grad_output)
    wgrad = grad_output_2d.t().contiguous() @ input_2d
    main_grad.add_(wgrad.to(dtype))


def _assert_vs_cpu_ref(res, ref, dtype, *, reduce_dim):
    utils.gems_assert_close(
        res, ref, dtype, reduce_dim=reduce_dim, atol=DEFAULT_ATOL
    )


def _assert_vs_apex(res, ref, dtype, *, reduce_dim):
    """Apex is the deployment target; use strict default tolerance."""
    utils.gems_assert_close(
        res, ref, dtype, reduce_dim=reduce_dim, atol=DEFAULT_ATOL
    )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_2d(batch, in_features, out_features, dtype):
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    ref_main_grad = utils.to_reference(main_grad, True).clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main_grad)

    _assert_vs_cpu_ref(
        res_main_grad, ref_main_grad, torch.float32, reduce_dim=batch
    )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_3D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_3d(dim0, dim1, in_features, out_features, dtype):
    input_tensor = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    ref_main_grad = utils.to_reference(main_grad, True).clone()
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

    _assert_vs_cpu_ref(res_main, ref_main, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_from_zero_main_grad():
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
@pytest.mark.parametrize("dtype", [torch.float16])
def test_wgrad_gemm_accum_fp32_vs_apex_3d(
    dim0, dim1, in_features, out_features, dtype
):
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
