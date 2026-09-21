# 256K Robust Top-5 Refinement Paired Benchmark

This experiment validates `robust-refine-256K-top5.json`, a trust-region
candidate derived from the independent 256K profile. It starts from the prior
multi-length mapping and changes only layers 21, 44, 19, 43, and 20. Each
change improves the byte-critical objective on all four 256K profile splits:
odd steps, even steps, first half, and second half.

The candidate and linear control both use TP2, PP1, batch size 1, top-k 0.10,
GPU 0/1, CPU cores 0-95, layer-0-only residency, Ditto CUDA graph enabled,
one warmup, and three measured epochs. The candidate runs first, followed
immediately by linear. The transfer recorder is disabled for both.

The manifests, raw logs, result JSON files, and full coordinator output are
retained in this directory.

## Result

The robust top-5 candidate was accepted: 49.090 ms and 20.371 TPS versus
51.865 ms and 19.291 TPS for paired linear. This is 5.35% lower latency and
5.60% higher throughput. All three candidate epochs were faster than the
fastest linear epoch. See `comparison.json` for complete statistics.
