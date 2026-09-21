# TP2 KV-Head Mapping Cost Model Report

Date: 2026-09-06

## Scope

- Model: Qwen2.5-14B-Instruct-1M
- GPU: 2 x NVIDIA A40 on an NV4 pair, TP2, PP1
- Workload: batch size 1, top-k ratio 0.10, 50 decode steps
- Measurement: warmup 1, epoch 3
- CUDA graph: SGLang graph off, Ditto internal graph on
- Residency: all layer-0 heads resident; every head in layers 1-47 offloaded

All throughput values below are decode throughput. They exclude prefill.

## Existing 128K Mapping Across Sequence Lengths

The 4K-64K mapped values were rerun in
`../multilen-rerun-20260906`. The 128K mapped value reuses the existing
formal `l25-none` result, as requested.

| Sequence | Linear latency (ms) | Mapped latency (ms) | Linear TPS | Mapped TPS | TPS change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 31.9827 | 32.0374 | 31.2672 | 31.2135 | -0.17% |
| 8K | 32.2726 | 32.2441 | 30.9864 | 31.0134 | +0.09% |
| 16K | 32.6132 | 32.6709 | 30.6626 | 30.6083 | -0.18% |
| 32K | 34.5482 | 34.5862 | 28.9451 | 28.9132 | -0.11% |
| 64K | 34.8260 | 34.8960 | 28.7142 | 28.6566 | -0.20% |
| 128K | 40.6323 | 39.6077 | 24.6110 | 25.2479 | +2.59% |

The old 8K result with 40-48 ms decode epochs was an outlier. The rerun
epochs were 32.2384, 32.2585, and 32.2353 ms.

The 128K-derived mapping is neutral within about 0.2% at 4K-64K. Its useful
gain starts at 128K.

## Model

The previous aggregate generator assumed heuristic resident heads while the
best formal run used explicit layer-0-only residency. For example, at layer
25 it modeled rank misses as 91/90 after treating head 5 as resident, but the
actual placement makes the loads 123/90. The new model reads the explicit
resident placement and removes this mismatch.

For every profile step `s`, layer `l`, and rank `r`, the model estimates:

```text
T(l,r,s) = H2D_bytes / H2D_bandwidth
         + D2H_bytes / D2H_bandwidth
         + active_head_count * per_head_cost
         + transfer_launch_costs
```

Its primary term is the TP critical rank, `max(T(l,0,s), T(l,1,s))`. The
objective blends observed stepwise critical cost with aggregate rank cost,
then adds tail, rank-imbalance, and worst-profile penalties. This shrinkage
is important: the 128K-only mapping overfit the recorded steps and was slower
in the formal benchmark.

The optimizer exhaustively evaluates all 70 oriented 4+4 partitions for each
layer. Layer 0 remains unchanged. Rank orientation is selected to balance
cumulative predicted rank load when both links have the same bandwidth.

Implementation: `../../build_tp_head_mapping_cost_model.py`

## Training Data And Offline Validation

Linear profiles were collected at 4K, 8K, 16K, 32K, 64K, and 128K. Each
profile has both TP ranks and 51 decode steps. The first 10 steps are dropped,
leaving 246 samples across six sequence buckets. KV size was inferred from
the recorded bytes as 512 bytes per token per global KV head.

The multi-length mapping reduced the modeled layer-summed objective from
8748.1 us for the current mapping to 8371.1 us. This is a data-path proxy,
not an end-to-end latency prediction.

Leave-one-sequence-out predicted changes against the current mapping:

| Held-out bucket | Predicted change |
| ---: | ---: |
| 4K | +1.11% |
| 8K | +1.04% |
| 16K | +0.62% |
| 32K | -0.82% |
| 64K | +0.03% |
| 128K | +1.05% |

Transfer recording synchronously writes JSON at every step and substantially
slows decode. Profile-run latency is therefore not used as a performance
measurement.

## End-To-End Validation

`unified-byte-critical-multilen.json` is the multi-length model output. The
128K current value below is a same-container paired control collected directly
before the new-mapping comparison.

| Sequence | Linear TPS | Current TPS | Cost-model TPS | vs linear | vs current |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 31.2672 | 31.2135 | 31.2624 | -0.02% | +0.16% |
| 8K | 30.9864 | 31.0134 | 31.0177 | +0.10% | +0.01% |
| 16K | 30.6626 | 30.6083 | 30.6397 | -0.08% | +0.10% |
| 32K | 28.9451 | 28.9132 | 28.8776 | -0.23% | -0.12% |
| 64K | 28.7142 | 28.6566 | 28.5312 | -0.64% | -0.44% |
| 128K | 24.6110 | 25.2204 | 25.5134 | +3.67% | +1.16% |

The new 128K epochs were 39.1799, 39.2076, and 39.1976 ms. All three are
faster than the fastest paired-current epoch, 39.4530 ms. Mean prefill was
effectively identical for the paired runs: 135.3665 s new versus 135.3762 s
current.

## Bucket Ablations

| Candidate | Latency (ms) | TPS | vs linear | Decision |
| --- | ---: | ---: | ---: | --- |
| 32K-only | 34.5371 | 28.9544 | +0.03% | No material gain |
| 64K-only | 34.8463 | 28.6975 | -0.06% | Reject |
| 128K-only | 40.3296 | 24.7958 | +0.75% | Reject: profile overfit |
| Multi-length at 128K | 39.1950 | 25.5134 | +3.67% | Accept for 128K |

## 256K Extrapolation

The same TP2, batch-size-1, top-k-0.10, warmup-1, epoch-3, and explicit
layer-0-only residency setup was also tested at 256K. The cost model was not
trained on a 256K profile, so this is an out-of-range extrapolation test.

| Mapping | Latency (ms) | TPS | TPS vs linear | Latency stdev (ms) |
| --- | ---: | ---: | ---: | ---: |
| Linear | 50.2877 | 19.8867 | 0.00% | 0.471 |
| Existing 128K mapping | 50.7429 | 19.7076 | -0.90% | 0.293 |
| Multi-length cost model | 50.1981 | 19.9214 | +0.17% | 0.251 |

The cost-model mapping is 1.08% faster than the existing 128K mapping, but
only 0.17% faster than linear. Its 0.090 ms mean advantage over linear is
smaller than the observed epoch variation, so it should be treated as tied
with linear rather than as a validated improvement. Mean prefill latency was
also effectively identical: 494.591 s linear, 494.587 s existing mapping,
and 494.395 s cost model.

## Recommended Selection Policy

- Up to 64K: keep the linear mapping. Measured differences are too small to
  justify a more complex static order.
- At 128K: use `unified-byte-critical-multilen.json` with `l25-none.json`.
- At 256K: use `timeline-v2/per-profile/robust-refine-256K-top5.json` after the
  paired validation documented in `timeline-v2/RESULTS.md`; keep linear as the
  rollback. Do not use the existing 128K mapping, which regresses by 0.90%.
- Treat sequence bucket, residency policy, batch size, top-k ratio, and GPU
  topology as model inputs. Do not reuse the 128K mapping blindly when any of
  these changes.

The result supports a bucketed mapping selector, not one universal mapping.
For a new workload, collect per-rank head masks, generate a candidate, and
retain it only after paired end-to-end validation against the current choice.

## Full-Timeline Optimizer V2

`../../build_tp_head_mapping_timeline.py` implements the next optimizer without
replacing the original layerwise cost model. It carries the following state for
every recorded decode step and TP rank while advancing across all 48 layers:

```text
layer completion time
transfer-engine available time
next-layer prefetch launch time
```

For a candidate layer partition, the transition schedules its H2D transfer from
the launch point produced by the previous layer, waits only for the portion that
is still incomplete when the layer consumes KV, schedules the one-token D2H
append, and advances both ranks to the next TP synchronization boundary. A beam
search retains the best full prefixes; full-timeline coordinate descent then
refines the completed mappings. This makes consecutive-layer rank orientation
part of the optimization instead of a post-hoc cumulative-load heuristic.

Each sequence profile is normalized against its own linear-reference timeline:

```text
regret(profile, mapping) =
    (timeline(mapping) - timeline(linear)) / timeline(linear)
```

Profile regrets are averaged with equal weight and a worst-profile penalty. The
CLI can also write one independently optimized mapping per input profile with
`--per-profile-output-dir`.

The current outputs are under `timeline-v2/`. With beam width 128 and one local
refinement pass, the uncalibrated transfer-only proxy reports:

| Candidate | Normalized proxy regret vs linear |
| --- | ---: |
| Unified 4K-128K | -3.88% |
| 4K-only | -5.68% |
| 8K-only | -7.89% |
| 16K-only | -7.73% |
| 32K-only | -2.54% |
| 64K-only | -8.15% |
| 128K-only | -7.80% |
| 256K-only | -8.76% |

These are candidate-generation scores, not measured speedups. No compute-window
calibration was supplied, so `pre_wait_us`, `post_launch_us`, and `tp_sync_us`
are zero. `--timing-json` accepts a scalar or 48-element list for each field,
with optional per-sequence overrides:

```json
{
  "default": {
    "pre_wait_us": 0.0,
    "post_launch_us": 0.0,
    "tp_sync_us": 0.0
  },
  "profiles": {
    "128K": {
      "pre_wait_us": 0.0,
      "post_launch_us": 0.0,
      "tp_sync_us": 0.0
    }
  }
}
```

In a real calibration file, each list must contain exactly 48 measured values.
The generated mappings must not be promoted until paired end-to-end validation
shows a repeatable improvement.

### Independent 256K Profile

The 256K candidate is not an extrapolation. A separate linear profile was
collected on 2026-09-07 with the same TP2, batch-size-1, top-k-0.10, Ditto CUDA
graph, and explicit layer-0-only residency setup. Both rank files contain 51
decode steps; dropping the first 10 leaves 41 combined samples. Mean recorded
H2D volume per step was 349.24 MB on rank 0 and 391.62 MB on rank 1.

The resulting `timeline-v2/per-profile/timeline-256K.json` changes 37 of 48
layer partitions. Its uncalibrated transfer-only timeline decreases from
21.016 ms for linear to 19.116 ms, or -8.76% normalized proxy regret.

### 256K End-To-End Refinement

Paired validation rejected the full-timeline 256K-only candidate: 51.705 ms
versus 51.385 ms linear (-0.65% TPS). The transfer-only proxy had overfit the
profile.

A trust-region refinement then kept the previous multi-length mapping and
changed only five layers (21, 44, 19, 43, and 20) whose proposed partitions
improved every odd/even and first/second-half split of the independent 256K
profile. In the second paired run it achieved 49.090 ms and 20.371 TPS versus
51.865 ms and 19.291 TPS for linear: **5.35% lower latency and 5.60% higher
throughput**. All three candidate epochs were faster than the fastest paired
linear epoch. Full methodology, caveats, manifests, logs, and JSON outputs are
in `timeline-v2/RESULTS.md` and
`timeline-v2/perf-256K-robust-top5-paired-20260907/`.
