# Ditto TP + PP integration

The working tree combines `TP_support` (25a629c74) and the code changes from
`feature/ditto-pp` (76a61e42a), preserving existing local edits. Benchmark result
JSON files and `.orig` backups from the PP branch are not part of the integration.
The integrated version is saved on `feature/ditto-tp-pp-graph` as a feature
commit based on `TP_support`; no merge commit was used.

## Behavior

Use `--tp-size T --pp-size P` together; a single-node configuration needs T × P
GPUs. TP partitions attention heads within each PP stage, and PP partitions
transformer layers. Weight names, per-layer head orders, auxiliary weights and
resident-head configuration retain global layer numbering. Cache arrays use
stage-local indices. Cross-layer prefetch wraps within each stage, and its
projection aliases are refreshed after TP replaces the linear modules.

Transfer statistics include both PP and TP rank suffixes. Empty PP stages and
incompatible incoming hidden-state shapes fail explicitly. Batched chunked full
attention accepts noncontiguous hidden-state slices from the preceding stage.

Existing restrictions on DP attention, context parallelism, multi-node TP and
replicated KV-head permutations remain. TP internal CUDA graphs are disabled by
default unless `DITTO_TP_ENABLE_CUDA_GRAPH=1` is explicitly set. The validation
in the original baseline used eager execution. See the Ditto Graph section below
for the subsequent capture/replay validation and fixes.

## Reproduce validation

Run from the repository root with this checkout's Python modules and native
kernel extension. The smoke test accepts `fullattn`, `--prompt-file`,
`--output-file` and `--ignore-eos`, and shuts down its engine after generation.

```bash
PYTHONPATH=python:sgl-kernel/python python3 -m pytest \
  test/ditto/test_pp_partition.py \
  test/ditto/speedup/test_build_tp_head_mapping_cost_model.py \
  test/ditto/speedup/test_build_tp_head_mapping_timeline.py \
  test/ditto/speedup/test_build_tp_head_mapping_robust_refine.py -q

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test/ditto/run_tp_pp_smoke.py \
  --model-path /jhe/Qwen2.5-7B-Instruct-1M \
  --output-dir /tmp/ditto-tp-pp-validation/full-matrix
```

The GPU runner compares greedy output token IDs for TP1/PP1, TP2/PP1,
TP1/PP2, TP2/PP2, and TP2/PP2 with a different KV-head permutation per layer.
Each case writes its log and output JSON; a mismatch or timeout fails the run.
Use an otherwise idle set of four GPUs. Full attention with TP1/PP1 requires the
whole model to fit on one GPU.

Hash offloading with TP2/PP2 (auxiliary data must match the model):

```bash
PYTHONPATH=python:sgl-kernel/python CUDA_VISIBLE_DEVICES=0,1,2,3 \
DITTO_RECORD_TRANSFER_STATS=1 \
DITTO_TRANSFER_STATS_FILE=/tmp/ditto-tp-pp-validation/offload-transfer.json \
python3 python/sglang/srt/models/ditto/minimal_selftest.py \
  --model-path /jhe/Llama-3-8B-Instruct-Gradient-1048k \
  --variant offloading --tp-size 2 --pp-size 2 \
  --prompt-file /tmp/ditto-tp-pp-validation/offload-prompt.txt \
  --output-file /tmp/ditto-tp-pp-validation/offload.json \
  --max-new-tokens 16 --ignore-eos \
  --max-tokens 4096 --max-total-tokens 2048 --gpu-memory-budget 4 \
  --rbits 256 --recent-budget 64 --num-skip-layers 1 \
  --aux-data-path /jhe/myTransformer/auxiliary/hash_weights/Llama-3-8B-Instruct-Gradient-1048k-256 \
  --attn-pattern-path /jhe/myTransformer/auxiliary/attn_pattern/Llama-3-8B-Instruct-Gradient-1048k
```

The tested prompt file contains `Paris is the capital of France. ` repeated
128 times, followed by `\nQuestion: What is the capital of France?\nAnswer:`.
Set `DITTO_TP_KV_HEAD_ORDER_FILE` to a JSON array of 32 layer permutations to
also exercise head balancing. The tested permutations are
`[(head + layer + 1) % 8 for head in range(8)]` for each layer.

## Original eager results (2026-09-20)

Environment: NVIDIA A40 GPUs, driver 570.133.20, Python 3.13.9,
PyTorch 2.9.1+cu128. Results and raw logs are under
`/tmp/ditto-tp-pp-validation/`. The parity matrix, batch outputs and summary
are also preserved in `test/ditto/results/eager-20260920/`.

- **26 unit tests passed**, including PP boundaries, local cache indexing,
  global-layer TP mappings, replicated KV heads, prefetch aliases, rank-specific
  filenames, and batched/ragged chunked prefill.
- **All five GPU full-attention configurations matched exactly**, generating
  `[12095, 13, 1084, 374, 7407, 304, 279, 10200]` for the default prompt.
- **Llama hash offloading TP2/PP2 passed** with a 909-token prompt, both with and
  without head permutations. The permuted run generated 16 tokens with EOS
  ignored (15 decode steps). All four ranks recorded actual transfers:

  | PP rank | TP rank | H2D bytes | D2H bytes |
  | --- | --- | --- | --- |
  | 0 | 0 | 6,922,240 | 199,680 |
  | 0 | 1 | 7,389,184 | 192,000 |
  | 1 | 0 | 2,207,232 | 92,160 |
  | 1 | 1 | 1,974,272 | 84,480 |

- Two unequal-length requests with internal prefill chunks of four tokens ran
  successfully twice on TP2/PP2; both requests matched the single-GPU reference
  exactly across both calls. See `batched-tp1-pp1.json` and `batched-tp2-pp2.json`.
- Python AST checks, modified shell syntax checks and `git diff --check` passed.

The existing `kvlib_cpu_gather.so` lacked the new `transfer_backend` argument.
For these runs, `py_cpu_gather_engine_v3.cc`, `cpu_gather_engine_v3.cc` and
`prefetch.cu` were rebuilt with `torch.utils.cpp_extension.load` for SM86 and
copied to `sgl-kernel/python/sgl_kernel/kvlib_cpu_gather.so`. The previous binary
is preserved at `/tmp/ditto-tp-pp-validation/kvlib_cpu_gather.previous.so`;
build output is in `gather-build.log` and `gather-build/` in that directory.
When moving to another environment, rebuild sgl-kernel using the repository's
installation instructions; the checked-in source is the portable change.

These tests verify functionality on the named models and eager configurations;
they do not measure long-context accuracy, throughput, other offloading
algorithms or quantization. Ditto Graph is covered by the follow-up below.

## Ditto Graph follow-up (2026-09-20–21)

本次补测使用 **Ditto 内部 CUDA Graph**：
`DITTO_TP_ENABLE_CUDA_GRAPH=1` + `--ditto-enable-cuda-graph`，外层 SGLang
保持 `--disable-cuda-graph`。测试脚本保存在
[`test/ditto/run_tp_pp_graph.py`](test/ditto/run_tp_pp_graph.py)，不依赖 `/tmp` 的输入文件。
脚本自动准备 prompt，验证每个 rank 的 capture/replay、真实 offloading 传输、
重复请求输出以及与同配置 eager 的逐 token 一致性。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test/ditto/run_tp_pp_graph.py \
  --model-path /jhe/Llama-3-8B-Instruct-Gradient-1048k \
  --output-dir test/ditto/results/new-llama-graph-run

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test/ditto/run_tp_pp_graph.py \
  --model-path /jhe/Qwen2.5-14B-Instruct-1M \
  --cases 2x2 --permute-heads \
  --output-dir test/ditto/results/new-qwen-graph-run
```

每次选择新的输出目录；默认 32 个输出 token、连续两次请求。
以脚本退出码 0 和最终 PASS 行作为整轮通过依据，不能只看中途生成的 summary。

实测暴露并修复了三类错误：

1. **capture 步未执行新图**：原实现只捕获而没有 replay，返回上一步输出。
   full attention 和 offloading 路径都在首次捕获后立即 replay；full attention
   同时使用 TP collective 的 graph capture 上下文。
2. **top-k 长度变化越界**：两个 hash backend 原先把索引截成 capture 当时的 k，
   后续动态 k 增大时会越界。Graph 路径现在保留固定容量，由设备上的长度控制有效值；
   prefetch 同样保留容量，并允许 k 从 0 增长。CUDA memcheck 曾直接定位到
   `_fwd_mix_kernel` 读取非法索引，不是单纯按日志推测。
3. **Graph replay 未更新 host 长度**：Python append 函数只在 warmup/capture 时运行，
   导致 host 长度和后续 top-k 预算停在旧值。现在 host 长度在每步解码结束时更新一次，
   不依赖统计开关。脚本同时检查长度每步递增，以及 Graph/eager 的长度、k 轨迹一致。

`test/ditto/test_cuda_graph_decode.py` 保存了 warmup/capture/replay 数值回归，
覆盖两个 hash backend、current/prefetch、k=0 与 k>0 的增长场景。
加上 host 长度、PP/head-mapping 测试共 **40 项通过**。

Qwen14B TP2/PP2 + 每层 head 重排对照通过，两个请求各输出 32 tokens。
四个 rank 各记录 warmup 1 次、capture 2 次、replay 59 次，没有 eager fallback。
结果在 `test/ditto/results/tp-pp-graph/qwen-final/summary.json`。
Llama 最终四种布局（1×1、2×1、1×2、2×2）全部通过，记录见
`test/ditto/results/tp-pp-graph/llama-complete/`。
最终汇总与测试代码校验值：
[`validation-final.json`](test/ditto/results/tp-pp-graph/validation-final.json)。
历史失败日志保留用于诊断，不算通过；以最终报告和每轮完整 PASS 为准。

最终代码的 Qwen14B TP2/PP2 定向 CUDA memcheck 退出码为 0，
`_fwd_mix_kernel` 的检查结果为 **ERROR SUMMARY: 0 errors**。
这里只检查了此前越界的混合 attention kernel，不代表全量 kernel 检查。
命令、环境、运行日志保存在 `test/ditto/results/tp-pp-graph/memcheck-final-command.json`、
`qwen-memcheck-final-driver.log` 和 `memcheck-final-*.log`。

短输入边界已验证 TP1/PP1、TP2/PP1：每次生成 32 tokens、连续两次请求，
最后一次请求的缓存长度从 336 到 367，预取 k 从 0 到 4；Graph/eager 输出一致。
记录在 `test/ditto/results/tp-pp-graph/llama-zero-k-fixed/`。复现时添加：

```bash
--cases 1x1,2x1 --prompt-repetitions 46
```

其他已保留脚本：

- `test/ditto/run_tp_pp_batch_smoke.py`：原先临时的 ragged batch/重复调用验证，增加了模型、输出和 reference 参数。
- `scripts/build_ditto_gather.py`：原先临时的 gather/prefetch 增量编译命令；`--install` 前先备份旧二进制。
- `test/ditto/run_tp_pp_smoke.py`：原有五种 full-attention eager 配置对照。

脚本分类见 [`test/ditto/README.md`](test/ditto/README.md)，目录清理判断和已归档
文件见 [`DIRECTORY_AUDIT.md`](DIRECTORY_AUDIT.md)。本次没有做吞吐量或长上下文
准确率结论，也没有验证外层 SGLang Graph 与 Ditto Graph 同时开启。
