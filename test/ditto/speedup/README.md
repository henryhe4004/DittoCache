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
