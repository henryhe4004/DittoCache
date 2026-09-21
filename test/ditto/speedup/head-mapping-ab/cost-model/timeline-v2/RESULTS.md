# 256K Mapping Optimization Results

Date: 2026-09-07

## Configuration

- Container: `jhe_sglang_lite_tp_bench`
- Repository commit: `25a629c7421601463d757c0375f4843dcde8cc30`
- Python 3.13.15, PyTorch 2.9.1+cu128
- 2 x NVIDIA A40: GPU 0/1, NV4, NUMA node 0, CPU cores 0-95
- Qwen2.5-14B-Instruct-1M, TP2, PP1, batch size 1
- Sequence 256000, input tokens 255860, top-k ratio 0.10
- Layer-0-only residency (`l25-none.json`)
- SGLang CUDA graph off, Ditto CUDA graph on
- Warmup 1, measured epochs 3, 50 decode steps per epoch
- Transfer recording disabled during performance measurements

Each paired experiment ran the candidate first and a fresh linear process
immediately afterward. Each variant directory contains a manifest, raw log, and
result JSON. `coordinator.log` records the entire outer run.

## Independent Profile

The optimization used a separately collected 256K linear profile, not a 128K
extrapolation. Both TP ranks contain 51 decode records; the optimizer drops the
first 10 and uses 41 steps. Mean H2D volume was 349.24 MB/step on rank 0 and
391.62 MB/step on rank 1.

## Attempt 1: Full-Timeline Search

The initial 48-layer beam-search candidate changed 37 layers and predicted
-8.76% transfer-only regret. End-to-end validation rejected it:

| Variant | Epoch latency (ms) | Mean latency (ms) | Mean TPS |
| --- | --- | ---: | ---: |
| Full-timeline 256K | 51.778, 51.937, 51.399 | 51.705 | 19.341 |
| Paired linear | 50.229, 51.259, 52.667 | 51.385 | 19.468 |

Result: latency +0.62%, TPS -0.65%. The uncalibrated timeline proxy overfit and
did not rank this candidate correctly.

Artifacts: `perf-256K-paired-20260907/` and its `comparison.json`.

## Attempt 2: Robust Trust-Region Refinement

The second candidate starts from the previous multi-length mapping and uses the
independent 256K profile to reconsider every layer. A layer is eligible only if
its proposed partition improves all four profile splits: odd steps, even steps,
first half, and second half. The five strongest eligible layers were selected:

| Layer | Full-profile improvement | Minimum split improvement | Proxy saving |
| ---: | ---: | ---: | ---: |
| 21 | 26.86% | 20.45% | 102.93 us |
| 44 | 14.14% | 11.73% | 153.77 us |
| 19 | 10.68% | 7.00% | 65.29 us |
| 43 | 8.93% | 4.53% | 88.48 us |
| 20 | 6.55% | 4.06% | 50.79 us |

The selected changes predict 461.26 us total byte-critical saving. The exact
orders, split scores, source profile paths, and reference mapping are embedded
in `per-profile/robust-refine-256K-top5.json`.

## Accepted Result

| Variant | Epoch latency (ms) | Mean latency (ms) | Stdev (ms) | Mean TPS |
| --- | --- | ---: | ---: | ---: |
| Robust top-5 | 48.999, 49.264, 49.006 | 49.090 | 0.151 | 20.371 |
| Paired linear | 52.718, 50.174, 52.704 | 51.865 | 1.465 | 19.291 |

- Mean latency improvement: **5.35%** (`-2.776 ms/step`)
- Mean throughput improvement: **5.60%**
- All three candidate epochs were faster than the fastest linear epoch
  (`49.264 ms < 50.174 ms`).
- Mean prefill differed by only -0.005 s, so the decode comparison was not
  accompanied by a material prefill change.
- Against the earlier 256K linear mean (50.288 ms, 19.887 TPS), the candidate
  remains +2.38% in latency and +2.44% in TPS.

The sample count is only three epochs. A conservative difference-of-means 95%
interval using a df=2 t critical value is [-6.434, +0.882] ms because the new
linear run has high variance. The observed epoch ranges are nevertheless fully
separated, and the result remains positive against the earlier linear control.

Decision: accept `robust-refine-256K-top5.json` as the best measured 256K
candidate, while retaining linear as the rollback and requiring revalidation if
topology, batch size, top-k ratio, residency, or model changes.

Artifacts: `perf-256K-robust-top5-paired-20260907/` and its `comparison.json`.

## Reproduction

Run these commands inside `jhe_sglang_lite_tp_bench` from
`/jhe/sglang-litecache`. The first command is the equivalent invocation used to
collect the standalone linear profile; the retained result and transfer files
are under `profile-linear-256K-20260907/`.

```bash
env \
  PYTHON_BIN="$PWD/.venv-py313/bin/python" \
  MODEL_PATH="$PWD/../Qwen2.5-14B-Instruct-1M" \
  CONFIG_ROOT="$PWD/test/ditto/config/hata_offloading" \
  DATA_ROOT="$PWD/../myTransformer/speedup/data" \
  LOG_DIR="$PWD/test/ditto/speedup/head-mapping-ab/cost-model/profile-linear-256K-20260907/results" \
  TRANSFER_STATS_DIR="$PWD/test/ditto/speedup/head-mapping-ab/cost-model/profile-linear-256K-20260907/transfer" \
  METHODS=offloading SEQ_LIST=256000 BSZ=1 TOPK=0.10 DECODE_STEPS=50 \
  WARMUP=1 EPOCH=1 CPUSET=0-95 CUDA_DEVICE=0,1 TP_SIZE=2 PP_SIZE=1 \
  MAX_TOTAL_TOKENS=262144 MEM_FRACTION_STATIC=0.92 \
  SGLANG_CUDA_GRAPH=0 DITTO_CUDA_GRAPH=1 DITTO_TP_ENABLE_CUDA_GRAPH=1 \
  DITTO_RESIDENT_HEADS_FILE="$PWD/test/ditto/speedup/head-mapping-ab/resident-placement/l25-none.json" \
  RECORD_TRANSFER_STATS=1 RECORD_HEAD_MASKS=1 \
  bash test/ditto/speedup/run_n2n_qwen_offloading_seqlen.sh
```

Generate the robust trust-region candidate from only that 256K profile:

```bash
.venv-py313/bin/python \
  test/ditto/speedup/build_tp_head_mapping_robust_refine.py \
  --transfer-json test/ditto/speedup/head-mapping-ab/cost-model/profile-linear-256K-20260907/transfer/qwen2.5-14b-1m-offloading-bsz1-seq256K-topk0.10-transfer.tp00.json \
  --transfer-json test/ditto/speedup/head-mapping-ab/cost-model/profile-linear-256K-20260907/transfer/qwen2.5-14b-1m-offloading-bsz1-seq256K-topk0.10-transfer.tp01.json \
  --reference-mapping test/ditto/speedup/head-mapping-ab/cost-model/unified-byte-critical-multilen.json \
  --resident-heads-file test/ditto/speedup/head-mapping-ab/resident-placement/l25-none.json \
  --max-changes 5 --min-split-improvement-pct 4 \
  --output test/ditto/speedup/head-mapping-ab/cost-model/timeline-v2/per-profile/robust-refine-256K-top5.json
```

Run the candidate followed by a fresh linear control, then summarize:

```bash
EXPERIMENT_DIR="$PWD/test/ditto/speedup/head-mapping-ab/cost-model/timeline-v2/perf-256K-robust-top5-paired-20260907" \
MAPPING_FILE="$PWD/test/ditto/speedup/head-mapping-ab/cost-model/timeline-v2/per-profile/robust-refine-256K-top5.json" \
CANDIDATE_NAME=robust-top5 \
  bash test/ditto/speedup/head-mapping-ab/cost-model/timeline-v2/run_paired_256k.sh

.venv-py313/bin/python \
  test/ditto/speedup/head-mapping-ab/cost-model/timeline-v2/summarize_paired_256k.py \
  --experiment-dir test/ditto/speedup/head-mapping-ab/cost-model/timeline-v2/perf-256K-robust-top5-paired-20260907 \
  --candidate robust-top5
```
