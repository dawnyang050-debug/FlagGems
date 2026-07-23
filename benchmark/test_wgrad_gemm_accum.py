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

# Table header defaults to "Torch Latency"; rename when baseline is Apex.
_BASELINE_LABEL = "Apex" if HAS_APEX_WGRAD else "PyTorch ref"

_BF16_OK = flag_gems.runtime.device.support_bf16

# fp32-accum API: half/bf16 activations into fp32 main_grad, plus fp32 activations.
_FP32_ACCUM_BENCH_DTYPES = [torch.float16, torch.float32]
if _BF16_OK:
    _FP32_ACCUM_BENCH_DTYPES.insert(1, torch.bfloat16)

# fp16-accum API: same-dtype half/bf16 main_grad.
_FP16_ACCUM_BENCH_DTYPES = [torch.float16]
if _BF16_OK:
    _FP16_ACCUM_BENCH_DTYPES.append(torch.bfloat16)

# (batch/K, in_features/N, out_features/M).
# GEMM is main_grad[M,N] += grad_output.T[M,K] @ input[K,N], same M/N/K as mm.
# Larger entries follow BlasBenchmark / mm in core_shapes.yaml:
#   [B, M, N, K] -> here (K, N, M).
WGRAD_GEMM_ACCUM_SHAPES = [
    # previous small shapes (kept for continuity; out was 2 * in)
    (64, 512, 1024),
    (128, 1024, 2048),
    (256, 2048, 4096),
    # from BlasBenchmark mm shapes
    (384, 384, 384),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    # larger K (closer to token-batch × hidden)
    (8192, 4096, 4096),
]


def _collapse_to_2d(input_tensor, grad_output):
    if input_tensor.dim() > 2:
        input_2d = input_tensor.reshape(-1, input_tensor.size(-1))
    else:
        input_2d = input_tensor
    if grad_output.dim() > 2:
        grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
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


def _run_with_baseline_header(bench):
    """Print baseline name and relabel the default 'Torch Latency' column."""
    print(f"[wgrad_gemm_accum] benchmark baseline: {_BASELINE_LABEL}")
    original_str = base.BenchmarkResult.__str__
    label = f"{_BASELINE_LABEL} Latency (ms)"

    def labeled_str(result):
        return (
            original_str(result)
            .replace("Torch Latency (ms)", label)
            .replace("Torch GBPS ", f"{_BASELINE_LABEL} GBPS ")
        )

    base.BenchmarkResult.__str__ = labeled_str
    try:
        bench.run()
    finally:
        base.BenchmarkResult.__str__ = original_str


class WgradGemmAccumFp32Benchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = WGRAD_GEMM_ACCUM_SHAPES

    def get_input_iter(self, cur_dtype):
        for batch, in_features, out_features in self.shapes:
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
        for batch, in_features, out_features in self.shapes:
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
        dtypes=_FP32_ACCUM_BENCH_DTYPES,
    )
    bench.set_gems(_gems_wgrad_gemm_accum_fp32)
    _run_with_baseline_header(bench)


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
        dtypes=_FP16_ACCUM_BENCH_DTYPES,
    )
    bench.set_gems(_gems_wgrad_gemm_accum_fp16)
    _run_with_baseline_header(bench)
