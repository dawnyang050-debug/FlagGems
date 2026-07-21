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


def _ref_wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad):
    """GPU torch matmul reference (closer to Triton mm than CPU ref)."""
    input_2d, grad_output_2d = _collapse_to_2d(input_tensor, grad_output)
    grad_output_T = grad_output_2d.t().contiguous()
    if input_2d.dtype in (torch.float16, torch.bfloat16):
        wgrad = grad_output_T.float() @ input_2d.float()
    else:
        wgrad = grad_output_T @ input_2d
    main_grad.add_(wgrad)


def _ref_wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad):
    input_2d, grad_output_2d = _collapse_to_2d(input_tensor, grad_output)
    wgrad = grad_output_2d.t().contiguous() @ input_2d
    main_grad.add_(wgrad)


def _assert_fp32_main_grad_close(res, ref, reduce_dim):
    utils.gems_assert_close(
        res, ref, torch.float32, reduce_dim=reduce_dim, atol=1e-3
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

    ref_main_grad = main_grad.clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp32(input_tensor, grad_output, ref_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main_grad)

    _assert_fp32_main_grad_close(res_main_grad, ref_main_grad, reduce_dim=batch)


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

    ref_main_grad = main_grad.clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp32(input_tensor, grad_output, ref_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main_grad)

    _assert_fp32_main_grad_close(
        res_main_grad, ref_main_grad, reduce_dim=dim0 * dim1
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

    ref_main_grad = main_grad.clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp16(input_tensor, grad_output, ref_main_grad)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_main_grad)

    atol = 1e-3 if dtype == torch.float16 else 0.02
    utils.gems_assert_close(
        res_main_grad, ref_main_grad, dtype, reduce_dim=batch, atol=atol
    )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not HAS_APEX_WGRAD,
    reason="Apex fused_weight_gradient_mlp_cuda not installed",
)
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D[:2])
@pytest.mark.parametrize("dtype", [torch.float16])
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

    _assert_fp32_main_grad_close(gems_main_grad, apex_main_grad, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not HAS_APEX_WGRAD,
    reason="Apex fused_weight_gradient_mlp_cuda not installed",
)
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D[:2])
@pytest.mark.parametrize("dtype", [torch.float16])
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

    utils.gems_assert_close(
        gems_main_grad, apex_main_grad, dtype, reduce_dim=batch, atol=1e-3
    )
