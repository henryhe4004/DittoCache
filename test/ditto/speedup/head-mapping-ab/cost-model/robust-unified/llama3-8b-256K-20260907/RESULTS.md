# Llama 3 8B 256K RSCTRO Validation

Date: 2026-09-07

## Configuration

- Model: `Llama-3-8B-Instruct-Gradient-1048k`
- Hardware: 2 x NVIDIA A40, GPU 0/1, NV4, NUMA node 0
- TP2, PP1, batch size 1, sequence 256000, top-k ratio 0.10
- Layer-0-only residency, SGLang CUDA graph off, Ditto CUDA graph on
- Fixed 51 completion tokens with `ignore_eos=true`
- Profile: warmup 1, epoch 1, transfer recorder and head masks enabled
- Performance: warmup 1, epoch 3, transfer recorder disabled

Candidate and linear were run in fresh processes. Their manifests record the
same model, configuration, data, resident placement, topology, and sgl-kernel
binary hash. Only the candidate manifest has an active mapping.

## Prerequisite Fixes

The first attempt could not JIT FlashInfer RMSNorm because the venv `bin`
directory was not in `PATH`; `run.failed-missing-ninja.log` preserves it. The
unified runner now prepends the selected Python environment's `bin` directory.

The second attempt reached decode and exposed a real Llama TP2 kernel gap:
`ham_dist.cu` supported Qwen's 20 local query heads but not Llama's 16. The
`HEAD_SWITCH` gained a `NumHead=16` specialization, the A40 `sm100/common_ops`
target was rebuilt, and a direct 16-query-head/4-KV-head CUDA call passed. The
old and new binary SHA256 values are:

```text
old 0dfb5662193936e5ce79b98123c27d26333ffc905bc56b9d550eed7169f6bb85
new 062cefe4deeb3316d9d422d54eeafbcb0451a4c0166cd45b6221dc74ed3fc981
```

`run.failed-numhead16.log` preserves the original abort. A subsequent valid
run produced only 19 steps because Llama emitted EOS early; it is retained in
`short-eos-19steps/` and excluded. The final profile uses `ignore_eos=true`.

## Independent Profile

Both rank files contain 51 decode records. Dropping the first 10 leaves 41
samples. The loader inferred 32 layers, 8 global KV heads, and 512 bytes per
token/head. Mean H2D traffic was 52.51 MB/step on rank 0 and 55.07 MB/step on
rank 1.

With the Qwen-derived default constraints (`delta=4%`, `B=5`), RSCTRO selected
zero layers. An exploratory relaxation to `delta=0`, `B=1` selected only layer
8. It changed the linear order from `[0,1,2,3 | 4,5,6,7]` to
`[0,1,3,4 | 2,5,6,7]`. Its transfer proxy predicted 22.99% improvement for
that layer but only 7.27 us absolute saving. Odd and first-half splits improved
12.70%; even and second-half splits were tied at 0%.

## Paired Result

| Variant | Epoch latency (ms) | Mean latency (ms) | Stdev (ms) | Mean TPS |
| --- | --- | ---: | ---: | ---: |
| Layer-8 candidate | 22.803, 29.126, 28.910 | 26.946 | 3.590 | 37.592 |
| Linear | 22.846, 28.196, 28.839 | 26.627 | 3.290 | 37.971 |

- Latency improvement: **-1.20%** (candidate is 0.319 ms slower)
- Throughput improvement: **-1.00%**
- Epoch ranges overlap; the acceptance rule fails.
- Decision: **reject**, retain linear mapping for this Llama 3 case.

The Qwen 256K speedup therefore does not transfer to Llama 3 8B. RSCTRO does
transfer as a decision procedure: it produced a tightly constrained candidate
and rejected it through the same predeclared end-to-end rule. Mapping benefit
depends on the model's observed head-miss structure and absolute transferable
work, not only sequence length.

## Artifacts

- `profile-linear/`: final profile, both rank transfer JSON/CSV files, manifest,
  successful log, and all excluded failed/short attempts
- `candidate-threshold4-nochange.json`: default conservative no-op result
- `candidate.json`: exploratory one-layer candidate
- `paired/candidate/` and `paired/linear/`: manifests, raw logs, result JSON
- `comparison.json`: exact paired statistics and rejection decision
- `generate.log` and `summarize.log`: optimizer and summarizer output

## Reproduction

Run the unified pipeline inside `jhe_sglang_lite_tp_bench`. Use a new output
directory to preserve the retained run above.

```bash
cd /jhe/sglang-litecache
env \
  MODEL_NAME=Llama-3-8B-Instruct-Gradient-1048k \
  MODEL_PATH=/jhe/Llama-3-8B-Instruct-Gradient-1048k \
  CONFIG_FILE=$PWD/test/ditto/config/hata_offloading/Llama-3-8B-Instruct-Gradient-1048k-256K.yaml \
  DATA_FILE=/jhe/myTransformer/speedup/data/RULER-Llama-3-8B-Instruct-Gradient-1048k-256K.jsonl \
  EXPERIMENT_DIR=$PWD/test/ditto/speedup/head-mapping-ab/cost-model/robust-unified/llama3-reproduction \
  SEQ_LEN=256000 MAX_TOTAL_TOKENS=262144 \
  RESIDENT_FILE=$PWD/test/ditto/speedup/head-mapping-ab/cost-model/robust-unified/llama3-8b-layer0-only.json \
  MAX_CHANGES=1 MIN_SPLIT_IMPROVEMENT_PCT=0 \
  bash test/ditto/speedup/run_robust_mapping_experiment.sh
```
