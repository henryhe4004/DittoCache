# Ditto Speedup Bench (Migrated)

This directory migrates `internal prototype/speedup` into SGLang's Ditto test flow.

## Core script

- `n2n_offloading.py`
  - Runs warmup + benchmark loops in one process.
  - Supports internal prototype-style config files (`.yaml` and `.json`).
  - Works with methods like `offloading`, `offloading-hash`, `hash-offloading`, `flashattn`.
  - Writes per-run JSON metrics via `--result-json`.

## Batch size and graph controls

The benchmark accepts `--batch_size N`; `--skip-invalid-batch` is an optional
legacy switch that skips Ditto batches larger than one. It is off by default.
Actual supported batch sizes depend on the model/cache configuration.

For Ditto internal CUDA graphs use `--ditto-enable-cuda-graph` together with
`DITTO_TP_ENABLE_CUDA_GRAPH=1` when TP is enabled. Keep `--disable-cuda-graph`
for the outer SGLang graph in the validated configuration. The saved
[`../run_tp_pp_graph.py`](../run_tp_pp_graph.py) checks real capture/replay and
compares graph outputs with eager outputs. See [the script index](../README.md).

New latency runs default to `test/ditto/results/latency/`. Previous latency and
September 2 RULER runs are preserved in `../archive/history/`.

## Run scripts

Qwen:
- `run_n2n_qwen_ablation_bsz.sh`
- `run_n2n_qwen_offloading_bsz.sh`
- `run_n2n_qwen_offloading_seqlen.sh`
- `run_n2n_qwen_fullattn_bsz.sh`
- `run_n2n_qwen_fullattn_seqlen.sh`

Llama:
- `run_n2n_llama_offloading_bsz.sh`
- `run_n2n_llama_offloading_seqlen.sh`
- `run_n2n_llama_fullattn_bsz.sh`
- `run_n2n_llama_fullattn_seqlen.sh`

## Offline ablation

`run_n2n_qwen_ablation_bsz.sh` runs the full-attention reference and the
cumulative LiteCache stages in one offline sweep:

```text
b0_memcpy -> b1_gdr -> b2_prefetch -> b3_qsac_fixed
          -> b4_cudagraph -> b5_adaptive -> b6_resident
```

| Stage | Transfer | Fetch schedule | Reuse | Threshold | Ditto graph | Resident heads |
| --- | --- | --- | --- | --- | --- | --- |
| `b0_memcpy` | CUDA memcpy | same-layer demand | always gather | fixed 0.8 | off | none |
| `b1_gdr` | GDR | same-layer demand | always gather | fixed 0.8 | off | none |
| `b2_prefetch` | GDR | cross-layer prefetch | always gather | fixed 0.8 | off | none |
| `b3_qsac_fixed` | GDR | cross-layer prefetch | QSAC | fixed 0.8 | off | none |
| `b4_cudagraph` | GDR | cross-layer prefetch | QSAC | fixed 0.8 | on | none |
| `b5_adaptive` | GDR | cross-layer prefetch | QSAC | profile-adaptive | on | none |
| `b6_resident` | GDR | cross-layer prefetch | QSAC | profile-adaptive | on | profile-selected heads (`skip=0`, `overlap=0`) |

`profile-adaptive` means a different threshold is derived for each head from
the offline attention profile during cache initialization. It is not an online,
per-token threshold update.

Select a subset without starting a server:

```bash
STAGES="fullattn b0_memcpy b1_gdr" \
BSZ_LIST="1 2 4" CUDA_DEVICE=1 \
bash run_n2n_qwen_ablation_bsz.sh
```

The script auto-detects the `/jhe` model and RULER data used in this workspace.
Override them, or the fixed KV-cache GPU budget, when running elsewhere:

```bash
MODEL_PATH=/path/to/model DATA_ROOT=/path/to/ruler/data \
ATTN_PATTERN_PATH=/path/to/attention/profile \
GPU_MEMORY_BUDGET=16 bash run_n2n_qwen_ablation_bsz.sh
```

Every result JSON records the resolved ablation configuration in
`runtime_meta.ablation_config`.

## Export CSV

- `export_speedup_csv.py`
  - Aggregates benchmark JSON files into one CSV table.
  - Default input: `logs-perf-from32k/*.json`.
  - Default output: `logs-perf-from32k/summary.csv`.

Example:

```bash
cd test/ditto/speedup
python3 export_speedup_csv.py --input-dir logs-perf-from32k --output-csv logs-perf-from32k/summary.csv
```

`run_n2n_qwen_offloading_perf_from32k.sh` now exports CSV automatically after all runs.
- Disable auto-export: `EXPORT_CSV=0`
- Change output path: `CSV_SUMMARY=/path/to/your.csv`

## Example

```bash
cd test/ditto/speedup
bash run_n2n_qwen_offloading_seqlen.sh
```

Override any setting with environment variables, e.g.:

```bash
METHODS="offloading-hash" CUDA_DEVICE=1 SEQ_LIST="8000 16000" bash run_n2n_qwen_offloading_seqlen.sh
```
