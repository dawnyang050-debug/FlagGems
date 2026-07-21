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

from . import base

try:
    import fused_weight_gradient_mlp_cuda as apex_wgrad

    HAS_APEX_WGRAD = True
except ImportError:
    HAS_APEX_WGRAD = False

WGRAD_GEMM_ACCUM_SHAPES = [
    (64, 512),
    (128, 1024),
    (256, 2048),
]


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


def _torch_ref_wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad_seed):
    main_grad = main_grad_seed.clone()
    input_2d, grad_output_2d = _collapse_to_2d(input_tensor, grad_output)
    main_grad.add_((grad_output_2d.t().contiguous().float() @ input_2d.float()))
    return main_grad


def _torch_ref_wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad_seed):
    main_grad = main_grad_seed.clone()
    input_2d, grad_output_2d = _collapse_to_2d(input_tensor, grad_output)
    main_grad.add_(grad_output_2d.t().contiguous() @ input_2d)
    return main_grad


def _apex_wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad_seed):
    main_grad = main_grad_seed.clone()
    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)
    return main_grad


def _apex_wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad_seed):
    main_grad = main_grad_seed.clone()
    apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad)
    return main_grad


def _gems_wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad_seed):
    main_grad = main_grad_seed.clone()
    flag_gems.wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)
    return main_grad


def _gems_wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad_seed):
    main_grad = main_grad_seed.clone()
    flag_gems.wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad)
    return main_grad


class WgradGemmAccumFp32Benchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = WGRAD_GEMM_ACCUM_SHAPES

    def get_input_iter(self, cur_dtype):
        for batch, in_features in self.shapes:
            out_features = in_features * 2
            input_tensor = torch.randn(
                batch, in_features, dtype=cur_dtype, device=self.device
            )
            grad_output = torch.randn(
                batch, out_features, dtype=cur_dtype, device=self.device
            )
            main_grad_seed = torch.randn(
                out_features, in_features, dtype=torch.float32, device=self.device
            )
            yield input_tensor, grad_output, main_grad_seed


class WgradGemmAccumFp16Benchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = WGRAD_GEMM_ACCUM_SHAPES

    def get_input_iter(self, cur_dtype):
        for batch, in_features in self.shapes:
            out_features = in_features * 2
            input_tensor = torch.randn(
                batch, in_features, dtype=cur_dtype, device=self.device
            )
            grad_output = torch.randn(
                batch, out_features, dtype=cur_dtype, device=self.device
            )
            main_grad_seed = torch.randn(
                out_features, in_features, dtype=cur_dtype, device=self.device
            )
            yield input_tensor, grad_output, main_grad_seed


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32():
    baseline = (
        _apex_wgrad_gemm_accum_fp32
        if HAS_APEX_WGRAD
        else _torch_ref_wgrad_gemm_accum_fp32
    )
    bench = WgradGemmAccumFp32Benchmark(
        op_name="wgrad_gemm_accum_fp32",
        torch_op=baseline,
        dtypes=[torch.float16, torch.float32],
    )
    bench.set_gems(_gems_wgrad_gemm_accum_fp32)
    bench.run()


@pytest.mark.wgrad_gemm_accum_fp16
def test_wgrad_gemm_accum_fp16():
    baseline = (
        _apex_wgrad_gemm_accum_fp16
        if HAS_APEX_WGRAD
        else _torch_ref_wgrad_gemm_accum_fp16
    )
    bench = WgradGemmAccumFp16Benchmark(
        op_name="wgrad_gemm_accum_fp16",
        torch_op=baseline,
        dtypes=[torch.float16],
    )
    bench.set_gems(_gems_wgrad_gemm_accum_fp16)
    bench.run()
