# wgrad_gemm_accum 完整交接文档

> **用途**：新对话把本文件丢给 Agent，即可无缝接续 yangyifei 在 FlagGems 上的 `wgrad_gemm_accum` 工作。  
> **日期**：2026-07-23（文档落盘）  
> **人物**：yangyifei（新人）；导师王震老师  
> **仓库**：本机 `E:\FlagGems`（fork）；跑卡 Docker `/workspace/FlagGems`  
> **分支**：`wgrad_gemm_accum_fp32/fp16`  
> **对话 transcript**：`C:\Users\悦\.cursor\projects\e-FlagGems\agent-transcripts\5f082a5a-eda2-4a85-a642-24a6441312eb\5f082a5a-eda2-4a85-a642-24a6441312eb.jsonl`  
> **注意**：本文件在 `YYF_Documents/`，**默认不进官方 PR**。

---

## 0. 新对话请先读这段 Prompt（可直接粘贴）

```text
请阅读 E:\FlagGems\YYF_Documents\模型训练算子\wgrad_gemm_accum_交接.md
（若路径不同，以用户给出的本交接文件为准）。

你是在接续 yangyifei 的 FlagGems 任务：对齐 Apex 的 wgrad_gemm_accum_fp32/fp16。
硬约束见文档第 7 节。始终中文。跑卡只用 GPU4、容器 yangyifei_docker。
给老师话术不提 push/PR。先根据文档状态快照确认「已完成 / 未完成」，再动手。
```

---

## 1. 身份 / 环境 / 双树

| 项 | 内容 |
|----|------|
| 用户 | yangyifei |
| 导师 | 王震老师 |
| 习惯 skill | `yangyifei-flaggems-habits`；任务发现 `wangzhen-task-discovery` |
| 本机 | Windows Cursor，`E:\FlagGems`；常无 CLI git / 无 Docker；用 GitHub Desktop Commit+Push |
| 跑卡 | JumpServer → 宿主机 `10.0.9.3` → 容器 **`yangyifei_docker`** → **`CUDA_VISIBLE_DEVICES=4` only** |
| 容器路径 | `/workspace/FlagGems`，venv：`source .venv/bin/activate` |
| 同步 | 本机改 → Desktop push → 容器 `git pull`；网页 Sync fork ≠ 容器已更新 |
| Apex | 系统包 `/usr/local/lib/python3.12/dist-packages/fused_weight_gradient_mlp_cuda*.so`；**不要**把整个 dist-packages 塞进 `PYTHONPATH` 抢在 venv torch 前面（会炸 nvshmem）。有需要时把 `.so` 拷进 venv site-packages |

### 进容器标准命令

```bash
docker start yangyifei_docker
docker exec -it yangyifei_docker bash
cd /workspace/FlagGems
source .venv/bin/activate
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=4
git checkout wgrad_gemm_accum_fp32/fp16
git pull --rebase origin wgrad_gemm_accum_fp32/fp16
pip install -e . --no-deps
```

---

## 2. 任务目标（标杆）

对齐 **Apex** `fused_weight_gradient_mlp_cuda`：

```python
wgrad_gemm_accum_fp32(input, grad_output, main_grad)  # -> None, inplace
wgrad_gemm_accum_fp16(input, grad_output, main_grad)  # -> None, inplace
```

| 参数 | 含义 |
|------|------|
| `input` | `(..., in_features)`，collapse → `(K, in)` |
| `grad_output` | `(..., out_features)`，collapse → `(K, out)` |
| `main_grad` | `(out, in)`，**原地** `+=` |

**数学**：`main_grad += grad_output.T @ input`

**调用链（训练侧）**：

```text
Megatron LinearWithGradAccumulationAndAsyncCommunication.backward
  → fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32/fp16
    → Apex cublasGemmEx（半精度读 + OP_T + fp32 C 累加）
```

**本阶段明确不做**（老师未点名前）：

- 改 Megatron / 接训练路径  
- async all-reduce / sequence parallel  
- 整条 Autograd Function 包装  

**老师最新口径（到 2026-07-23）**：性能对齐后可请示「先收 / 接训练 / 再补边界」；已让优化慢 case、加大 shape。

---

## 3. 当前状态快照（2026-07-23）

### 已完成

| 层 | 状态 |
|----|------|
| 算子实现 | `wgrad_gemm_accum_fp32/fp16`；fp32 accum 走 **ctypes→cublasGemmEx**（Apex 同布局）；fp16 accum 走 **torch.addmm** |
| 导出 | `ops/__init__.py`、`flag_gems.__init__` 已导出 |
| 正确性 | Docker GPU4：**约 114 passed**（含 vs Apex、non-contiguous、数值边界；后补了 fp16 vs Apex 边界） |
| 性能优化 | 半精度→fp32 从 ~0.2–0.9× 提到 **~1.0× Apex**（GemmEx） |
| Bench shape | 对齐 mm / BlasBenchmark 大 shape |
| Bench dtype | fp16 / **bf16** / fp32（设备 `support_bf16=True` 时） |
| 表头 | `baseline: Apex`，列名 `Apex Latency` |

### 未完成 / 可选后续

| 项 | 优先级 | 说明 |
|----|--------|------|
| 请示老师是否收口 / 接 Megatron | P0 | 球在老师 |
| ~~统一 `to_reference(..., True)`~~ | ~~P1~~ | **已改** `main_grad.clone()` |
| ~~空 batch / `main_grad` 非连续~~ | ~~P1~~ | **已补**（见 6.3） |
| ~~NaN/Inf 传播 vs Apex~~ | ~~P1~~ | **已补**；实现无改，测试对齐 Apex |
| ~~大 shape correctness + vs Apex~~ | ~~P1~~ | **已补** `WGRAD_SHAPES_LARGE_*`（含 K=8192） |
| ~~重复调用稳定性~~ | ~~P1~~ | **已补** fresh 200 / accum 200 / stress 1000 |
| ctypes GemmEx → 正式 CUDA 扩展 | P0 工程化 | 能跑且约 1×，但找 `.so`+魔数不够「产品级」 |
| 个人交接以外的正式 PR 节奏 | 听老师 | 话术里不提前说 push/PR |

### 性能结论（可对外）

复测 + bf16 补测后，相对 Apex：

| 路径 | Speedup |
|------|---------|
| fp16→fp32 accum | ~0.99–1.01× |
| bf16→fp32 accum | ~0.997–1.015× |
| fp32 输入 | ~1.00–1.10×（个别 shape 略快） |
| fp16 / bf16 accum | ~0.99–1.01× |

个别 0.992 属亚毫秒抖动，不是回退。

---

## 4. 关键文件路径

| 文件 | 作用 |
|------|------|
| `src/flag_gems/ops/wgrad_gemm_accum.py` | **内核实现（当前权威）** |
| `src/flag_gems/ops/__init__.py` | 导出算子名 |
| `src/flag_gems/__init__.py` | 顶层 API |
| `tests/test_wgrad_gemm_accum.py` | 正确性（CPU fp64 ref + vs Apex + 边界 + nc） |
| `benchmark/test_wgrad_gemm_accum.py` | 性能（baseline 优先 Apex） |
| 本交接 | `YYF_Documents/模型训练算子/wgrad_gemm_accum_交接.md` |

相关但非本任务核心：

- Apex 源：`NVIDIA/apex` → `csrc/megatron/fused_weight_gradient_dense_cuda.cu`  
- Megatron：`tensor_parallel/layers.py` 里 `gradient_accumulation_fusion`  

---

## 5. 当前实现细节（必须读懂再改）

### 5.1 分路径

```text
wgrad_gemm_accum_fp32  (main_grad 必须 fp32)
  └─ _accum_wgrad(fp32_accum=True)
       └─ _cublas_wgrad_gemm_accum_fp32(input_2d, grad_output_2d, main_grad)
            = Apex 同款 cublasGemmEx(
                OP_N(input), OP_T(grad_output),
                m=in_dim, n=out_dim, k=hidden,
                A/B = half|bf16|fp32, C = fp32,
                alpha=1, beta=1,
                algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP(99)
              )

wgrad_gemm_accum_fp16  (三者同为 fp16 或同为 bf16)
  └─ _accum_wgrad(fp32_accum=False)
       └─ densify → gout.t() view → torch.addmm(..., beta=1, out=main_grad)
```

### 5.2 Apex 布局（不要改错参数顺序）

Apex `wgrad_gemm_accum_fp32_cuda`：

```text
hidden = input.size(0)   # K
in_dim = input.size(1)
out_dim = d_weight.size(0)
cublasGemmEx(handle, OP_N, OP_T, in_dim, out_dim, hidden,
  &alpha, input, Atype, in_dim,
  d_output, Btype, out_dim,
  &beta, d_weight, CUDA_R_32F, in_dim,
  CUDA_R_32F, DEFAULT_TENSOR_OP)
```

语义等价：`d_weight += d_output.T @ input`。

### 5.3 ctypes 加载 libcublas

- `torch.cuda.current_blas_handle()` 取 handle  
- 在 torch/nvidia 包路径下找 `libcublas.so*`  
- 枚举：`CUDA_R_16F=2`, `CUDA_R_16BF=14`, `CUDA_R_32F=0`, algo=`99`  
- **风险**：依赖机器上的 `.so` 与枚举；长期可收成 cpp_extension，但当前 Docker H20 已验证正确性+性能  

### 5.4 Collapse

- 用 **`reshape(-1, last)`**，不用 `view`（non-contiguous 会挂；Apex stub 用 view，所以 vs Apex nc 时 Apex 喂 contiguous 等价数据）  

### 5.5 实现演进（避免新 Agent 走回头路）

| 版本 | 半精度→fp32 | 结果 |
|------|-------------|------|
| v1 | `addmm` inplace | 大面积错 |
| v2 | `mm`+`add_`（整表 cast fp32） | 正确，慢 ~0.23× |
| v3 | Triton `addmm_dtype_out` | 正确，~0.5–0.9× |
| **v4 当前** | **cublasGemmEx** | 正确，**~1×** |

同 dtype：先 mm+add → 后 `torch.addmm` → 已约 1×。

non-contiguous 坑：不能对连续走 `OP_T` view、对非连续走 `t().contiguous()` 两套数值路径（fp16 会对不齐）；应 **先 densify 再统一 `.t()` view**（fp16 accum）；fp32 accum 的 GemmEx 路径内部 `contiguous()` 输入即可。

---

## 6. 测试体系（`tests/test_wgrad_gemm_accum.py`）

### 6.1 Ref 原则（严禁「对着答案放宽容差」）

- **独立 CPU fp64** 算 matmul 再累加（`_ref_wgrad_gemm_accum_fp32_cpu` / `_fp16_cpu`）  
- **vs Apex** 为部署对齐目标  
- fp32 输入默认 **不** 与 CPU fp64 硬比（TF32）；另有关 TF32 严测 + vs Apex  
- 容差：仓库常规 `atol * reduce_dim` + dtype rtol；`large_1e3` 用量级感知 atol + `rtol=1e-4`（有注释）  

### 6.2 主要用例类别

- 2D/3D 形状；fp16/bf16/fp32 输入（按 API）  
- 两次累加、零起点、非法 shape/dtype  
- vs Apex 2D/3D  
- TF32-off 严格数学（极端 shape）  
- non-contiguous（2D/3D；vs Apex 时 Apex 用 contiguous）  
- 数值边界：zeros / large / small / mixed_signs  
  - fp32 accum：CPU + vs Apex  
  - **fp16 accum：CPU + vs Apex（2026-07-23 已补）**  
- fp16 large 用 scale=64 防 fp16 溢出（`_large_activation_scale`）  

### 6.3 已知小瑕疵（P1）

- ~~`test_wgrad_gemm_accum_fp16_2d` 的 `to_reference(..., True)`~~：**已改为 `main_grad.clone()`**（2026-07-23）  
- ~~空 batch / `main_grad` 非连续~~：**已补**（实现：`K==0` early return；fp16 accum 对非连续 `main_grad` densify 再 `copy_`；测试：`empty_batch` / `main_grad_non_contiguous`）  
- ~~NaN/Inf 传播~~：**已补 vs Apex 并容器验证**（`*_vs_apex_nan_inf`，35 passed；口径与 Apex/cuBLAS GEMM 一致传播）  
- ~~大 shape / 长序列 correctness~~：**已补并容器验证** `*_vs_apex_large_shape`（28 passed；含 `(8192,4096,4096)` 与 3D collapse）  
- ~~重复调用稳定性~~：**已补** `*_repeat_fresh`（200）/ `*_repeat_accum`（200）/ `*_repeat_stress`（1000，fp16→fp32）

### 6.4 跑正确性

```bash
pytest tests/test_wgrad_gemm_accum.py -v -s --tb=short 2>&1 | tee wgrad_acc.log
# 期望：全绿（有 Apex 时约 114+；无 Apex 会 skip vs_apex）
```

---

## 7. Benchmark（`benchmark/test_wgrad_gemm_accum.py`）

### 7.1 Shapes `(batch/K, in/N, out/M)`

```text
(64, 512, 1024)
(128, 1024, 2048)
(256, 2048, 4096)
(384, 384, 384)      # mm
(1024, 1024, 1024)
(2048, 2048, 2048)
(4096, 4096, 4096)
(8192, 4096, 4096)
```

必须 `set_shapes` 强制，否则会被 yaml 默认盖掉。

### 7.2 Dtypes

- fp32 API：`[fp16, bf16?, fp32]`（`support_bf16` 时插入 bf16）  
- fp16 API：`[fp16, bf16?]`  

### 7.3 Baseline

- 有 Apex → Apex；表头替换为 `Apex Latency`  
- 打印：`[wgrad_gemm_accum] benchmark baseline: Apex`  

### 7.4 历史数字（对比时勿混 shape 列表）

**优化前（仅小 shape，拼装 mm+add）**：

- fp16→fp32：~0.23–0.34×  
- fp16 accum：~0.30–0.42×  
- fp32 输入大者：~0.95×  

**加大 shape + Triton 融合后（GemmEx 前）**：

- fp16→fp32：~0.46–0.90×  
- 同 dtype：~1×  

**GemmEx + bf16 后（当前）**：

- 各路径 ~1×（见第 3 节）  

### 7.5 跑性能

```bash
python -c "import fused_weight_gradient_mlp_cuda; print('apex ok')"
pytest benchmark/test_wgrad_gemm_accum.py -v -s --tb=short 2>&1 | tee wgrad_bench.log
# 含 bf16 时整次约 50–60s；无 bf16 约 15–35s
```

---

## 8. 硬约束（Agent 必须遵守）

1. **始终中文**回复用户  
2. 给老师话术：**不提 push、不提 PR**  
3. 跑卡 **只用 GPU 4**；容器 `yangyifei_docker`  
4. **默认不改 Megatron**，除非老师明确允许  
5. `YYF_Documents/` **不进官方 PR**  
6. 正确性：**不能**为过测试乱放宽容差或改语义凑答案  
7. 本机无 Docker 时：给完整容器命令，等用户贴日志再下结论  
8. `PYTHONPATH` 不要把系统 `dist-packages` 整包 prepend（破坏 venv torch / nvshmem）  

---

## 9. 与老师沟通进度（避免重复汇报）

已汇报过、**新话术不要整段重复**的内容：

1. 接口/正确性初版、数值边界 100 passed  
2. 小 shape 首轮性能（0.23× 等）+ 表头曾误写成 Torch  
3. 加大 shape + 融合优化后：同 dtype ~1×，半精度→fp32 仍弱  
4. GemmEx 后半精度→fp32 ~1×；复测确认  
5. bf16 bench 覆盖  

**当前适合的请示**：

> 正确性全过；性能（含 bf16）相对 Apex 大约 1×；空 batch / main_grad 非连续 / NaN·Inf 传播（vs Apex）也已补。请问这块先收，还是接训练路径？

---

## 10. 常见坑速查

| 现象 | 原因 / 处理 |
|------|-------------|
| `libtorch_nvshmem.so: undefined symbol` | `PYTHONPATH` 抢了系统 torch → `unset PYTHONPATH`，Apex `.so` 拷进 venv |
| `pytest file not found` | 容器未 pull |
| bench 无 bf16 表 | 未 pull 到含 bf16 的 bench，或 `support_bf16=False`（H20 应为 True） |
| non-contiguous fp16 fail | 两套转置路径数值分叉 → 先 densify 再 `.t()` |
| 性能「又变慢」到 0.99× | 亚毫秒抖动，对比整表不要单看一行 |
| JumpServer websocket closed | 网页断了，容器进程可能还在 → `ps` / `nohup`+log |
| 本机无 docker/git | Desktop 推；容器 pull；Agent 勿假装已跑卡 |

---

## 11. 优化原理（给自己 / 给老师白话）

慢的不是公式，是路径：

1. 整表 cast fp32 + 分步 mm + add → 带宽和 kernel 次数爆炸  
2. Triton addmm 融合更好，但大 GEMM 仍难打过 cuBLAS Tensor Core  
3. Apex / 当前实现：一次 `cublasGemmEx`，半精度读、`OP_T` 不落地转置、fp32 累加进 `main_grad` → ~1×  

---

## 12. 建议下一步（新对话默认顺序）

1. 读本文件第 3、5、8 节确认状态  
2. 容器 `git status` / `git log -1` 确认与本机分支一致  
3. **本机改动 sync 到容器后**，先跑 `-k repeat`，再全量正确性  
4. **等老师回「收 / 接训练」**；未回前可选：GemmEx 工程化  
5. 若老师要求接 Megatron：先确认允许，再改调用点，**仍 GPU4 only**  

---

## 13. 关键代码锚点（当前实现摘要）

文件：`src/flag_gems/ops/wgrad_gemm_accum.py`

- `_collapse_to_2d`：reshape  
- `_load_cublas` / `_cublas_wgrad_gemm_accum_fp32`：GemmEx（非连续 `main_grad` densify + `copy_`）  
- `_matmul_operands` + `_fused_addmm_cublas`：同 dtype addmm（同上 nc 处理）  
- `_accum_wgrad`：`K==0` early return；`fp32_accum` 分流  

文件：`benchmark/test_wgrad_gemm_accum.py`

- `WGRAD_GEMM_ACCUM_SHAPES`  
- `_FP32_ACCUM_BENCH_DTYPES` / `_FP16_ACCUM_BENCH_DTYPES`  
- `_run_with_baseline_header`  

文件：`tests/test_wgrad_gemm_accum.py`

- `_make_numeric_boundary_tensors`（fp32 main_grad）  
- `_make_fp16_accum_boundary_tensors` + `test_*_fp16_vs_apex_numeric_boundaries`  

---

## 14. 文档维护

有下列变更时请更新本文件：

- 实现路径变更（尤其离开 ctypes）  
- 测试条数 / 新用例  
- 新一轮 bench 数字（写清 shape 列表，勿混比）  
- 老师新指示  

**最后更新**：2026-07-23 — GemmEx 性能对齐 + fp16 vs Apex 边界 + bf16 bench 完成；待老师收口或下发下一任务。
