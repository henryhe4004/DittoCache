# Ditto Speedup Bench (Migrated)

This directory migrates `internal prototype/speedup` into SGLang's Ditto test flow.

## Core script

- `n2n_offloading.py`
  - Runs warmup + benchmark loops in one process.
  - Supports internal prototype-style config files (`.yaml` and `.json`).
  - Works with methods like `offloading`, `offloading-hash`, `hash-offloading`, `flashattn`.
  - Writes per-run JSON metrics via `--result-json`.

## Batch-size limitation

SGLang Ditto bridge currently only supports `batch_size=1`.

- If `method` resolves to Ditto and `batch_size != 1`, the run is skipped by default.
- This keeps sweep scripts running without aborting the full loop.
- Use `--strict-invalid-batch` to make it fail instead.

## Run scripts

Qwen:
- `run_n2n_qwen_offloading_bsz.sh`
- `run_n2n_qwen_offloading_seqlen.sh`
- `run_n2n_qwen_fullattn_bsz.sh`
- `run_n2n_qwen_fullattn_seqlen.sh`

Llama:
- `run_n2n_llama_offloading_bsz.sh`
- `run_n2n_llama_offloading_seqlen.sh`
- `run_n2n_llama_fullattn_bsz.sh`
- `run_n2n_llama_fullattn_seqlen.sh`

## Export CSV

- `export_speedup_csv.py`
  - Aggregates benchmark JSON files into one CSV table.
  - Default input: `logs-perf-from32k/*.json`.
  - Default output: `logs-perf-from32k/summary.csv`.

Example:

```bash
cd /speedup
python3 export_speedup_csv.py --input-dir logs-perf-from32k --output-csv logs-perf-from32k/summary.csv
```

`run_n2n_qwen_offloading_perf_from32k.sh` now exports CSV automatically after all runs.
- Disable auto-export: `EXPORT_CSV=0`
- Change output path: `CSV_SUMMARY=/path/to/your.csv`

## Example

```bash
cd /speedup
bash run_n2n_qwen_offloading_seqlen.sh
```

Override any setting with environment variables, e.g.:

```bash
METHODS="offloading-hash" CUDA_DEVICE=1 SEQ_LIST="8000 16000" bash run_n2n_qwen_offloading_seqlen.sh
```

## B0-B6 ablation

`run_ablation_b0_b6.sh` runs the seven cumulative speedup groups. It samples
prompts from `/workspace/jhe/8K`; for `64x8K` it selects
`../config/hata_offloading/Qwen2.5-14B-Instruct-1M-512K.yaml`.

Run only the `64x8K` ablation in B0 -> B6 order:

```bash
cd /workspace/jhe/sglang-litecache/test/ditto/speedup
OUTPUT_DIR=/workspace/jhe/sglang-litecache/test/ditto/speedup/results/ablation_64x8k_20260923 \
PYTHON_BIN=/workspace/jhe/sglang-litecache/.venv-py313/bin/python \
ABLATION_BATCH_SIZES=64 \
ABLATION_PROFILES="B0 B1 B2 B3 B4 B5 B6" \
WATCHDOG_TIMEOUT=600 \
bash ./run_ablation_b0_b6.sh \
  --model /workspace/jhe/Qwen2.5-14B-Insturct-1M
```

Authoritative profile order:

| Profile | Transfer | Communication | Layer prefetch | QSAC / similarity | Ditto graph | Threshold switch | Resident cache |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `B0` | CUDA memcpy | off | off, always gather | off | `1.0/1.0` | off | off |
| `B1` | GDRCopy | off | off, always gather | off | `1.0/1.0` | off | off |
| `B2` | GDRCopy | on, cross-layer | on, cross-layer | off | `1.0/1.0` | off | off |
| `B3` | GDRCopy | on | on, cross-layer | on | `1.0/1.0` | off | off |
| `B4` | GDRCopy | on | on | on | fixed `0.8/0.8` | off | off |
| `B5` | GDRCopy | on | on | on | profile adaptive, current `-1.0/0.8` from the `512K` YAML | off | off |
| `B6` | GDRCopy | on | on | on | profile adaptive, current `-1.0/0.8` from the `512K` YAML | on | off |

Per-profile outputs are written to:

```text
results/ablation_64x8k_20260923/64x8K/B0.json
results/ablation_64x8k_20260923/64x8K/B1.json
results/ablation_64x8k_20260923/64x8K/B2.json
results/ablation_64x8k_20260923/64x8K/B3.json
results/ablation_64x8k_20260923/64x8K/B4.json
results/ablation_64x8k_20260923/64x8K/B5.json
results/ablation_64x8k_20260923/64x8K/B6.json
results/ablation_64x8k_20260923/summary.csv
```

## 64x8K clean rerun, drop first 20 decode steps

Run directory:

```text
results/ablation_64x8k_clean_drop20_20260923_rerun
```

This rerun uses the current script mapping, isolated process teardown between cases, and `decode ms/step` averaged after dropping the first 20 decode steps. The `512K` YAML adaptive thresholds are `-1.0/0.8`. GPU checks before each case reported no running compute process and only 4MiB baseline memory.

Actual profile deltas used by the script:

| Profile | Main delta | Actual key flags |
| --- | --- | --- |
| `B0` | CUDA memcpy baseline | prefetch off, similarity off, Ditto graph off, adaptive off, resident off, `1.0/1.0` |
| `B1` | GDRCopy | prefetch off, similarity off, Ditto graph off, adaptive off, resident off, `1.0/1.0` |
| `B2` | Layer prefetch / cross-layer communication | prefetch on, similarity off, Ditto graph off, adaptive off, resident off, `1.0/1.0` |
| `B3` | Ditto CUDA graph | prefetch on, similarity off, Ditto graph on, adaptive off, resident off, `1.0/1.0` |
| `B4` | QSAC / similarity + fixed threshold | prefetch on, similarity on, Ditto graph on, adaptive off, resident off, `0.8/0.8` |
| `B5` | Profile adaptive threshold | prefetch on, similarity on, Ditto graph on, adaptive on, resident off, `-1.0/0.8` |
| `B6` | Resident cache | prefetch on, similarity on, Ditto graph on, adaptive on, resident on, `-1.0/0.8` |

Results:

| Profile | decode ms/step | prefill s | elapsed s | decode tok/s | Trend note |
| --- | ---: | ---: | ---: | ---: | --- |
| `B0` | 1242.080 | 5.707 | 42.969 | 51.526 | baseline |
| `B1` | 920.564 | 5.063 | 32.680 | 69.523 | faster than `B0` |
| `B2` | 2806.151 | 6.095 | 90.280 | 22.807 | slower than `B1`; GPU was clean before/after, likely config/path overhead |
| `B3` | 290.280 | 5.379 | 14.088 | 220.477 | faster than `B2` |
| `B4` | 162.663 | 5.646 | 10.526 | 393.451 | faster than `B3` |
| `B5` | 655.565 | 5.473 | 25.140 | 97.626 | slower than `B4`; config confirmed adaptive `-1.0/0.8` |
| `B6` | 954.211 | 6.071 | 34.697 | 67.071 | slower than `B5`; warmup was 65.523 ms/step but epoch result regressed |

Generated files:

```text
results/ablation_64x8k_clean_drop20_20260923_rerun/64x8K/B0.json
results/ablation_64x8k_clean_drop20_20260923_rerun/64x8K/B1.json
results/ablation_64x8k_clean_drop20_20260923_rerun/64x8K/B2.json
results/ablation_64x8k_clean_drop20_20260923_rerun/64x8K/B3.json
results/ablation_64x8k_clean_drop20_20260923_rerun/64x8K/B4.json
results/ablation_64x8k_clean_drop20_20260923_rerun/64x8K/B5.json
results/ablation_64x8k_clean_drop20_20260923_rerun/64x8K/B6.json
results/ablation_64x8k_clean_drop20_20260923_rerun/summary.csv
```


Run status on 2026-09-23 UTC:

- Dry run and profile unit tests passed before launch.
- First live attempt failed before `B0.json` because stale GPU memory caused CUDA OOM; after the stale process/memory was cleared, the command above was rerun.
- Rerun completed successfully with exit code 0. All `B0` through `B6` JSON files and `summary.csv` were written under `results/ablation_64x8k_20260923`.

Final 64x8K metrics from the old run are retained only as raw records. They were measured with the previous profile mapping, previous `0.0/0.8` adaptive lower threshold, and drop-first-10 decode averaging, so they should not be used as the final ablation table.

Key generated files:

```text
results/ablation_64x8k_20260923/64x8K/B0.json
results/ablation_64x8k_20260923/64x8K/B1.json
results/ablation_64x8k_20260923/64x8K/B2.json
results/ablation_64x8k_20260923/64x8K/B3.json
results/ablation_64x8k_20260923/64x8K/B4.json
results/ablation_64x8k_20260923/64x8K/B5.json
results/ablation_64x8k_20260923/64x8K/B6.json
results/ablation_64x8k_20260923/summary.csv
```
