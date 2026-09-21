# 256K Timeline Mapping Paired Benchmark

This directory records the end-to-end validation of the mapping generated only
from the independent 256K profile.

## Fixed Configuration

- Model: Qwen2.5-14B-Instruct-1M
- Sequence: 256000 (`255860` input tokens)
- TP2, PP1, batch size 1, top-k ratio 0.10
- GPU 0/1, NV4, NUMA node 0, CPU cores 0-95
- SGLang global CUDA graph off; Ditto CUDA graph on
- Explicit layer-0-only resident placement (`l25-none.json`)
- 50 decode steps, warmup 1, measured epochs 3
- Transfer recorder off during performance measurement

## Procedure

`../run_paired_256k.sh` runs the 256K-specific mapping first and the linear
control immediately afterward. Each variant directory contains the original
runner log, result JSON, and a manifest with software, topology, hashes, and
environment settings. `coordinator.log` is the complete outer command output.

The comparison must use decode latency/TPS from the paired result JSON files.
Profile-run latency is excluded because synchronous transfer recording changes
decode performance.

## Result

The full-timeline candidate was rejected: 51.705 ms and 19.341 TPS versus
51.385 ms and 19.468 TPS for paired linear. This is -0.65% throughput. See
`comparison.json` for the complete statistics.
