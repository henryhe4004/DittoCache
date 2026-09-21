# Robust Split-Constrained Trust-Region Optimization

Robust Split-Constrained Trust-Region Optimization (RSCTRO) is the unified
candidate-generation and acceptance procedure used for TP2 KV-head mapping. It
is model-independent: layer count, global KV-head count, and KV bytes per
token/head are inferred from the recorded transfer profile.

## Inputs

- One or more independently collected linear transfer profiles with per-head
  masks for both TP ranks.
- A reference mapping, or the linear partition when no mapping is supplied.
- The exact resident-head placement used by the benchmark.
- A minimum split improvement `delta` and a maximum changed-layer budget `B`.

## Optimization

For every optimizable layer, enumerate each unique TP2 half partition. Score
the candidate and reference on four temporal views of every profile: odd
steps, even steps, first half, and second half. The score estimates the
byte-critical TP rank and includes tail and rank-imbalance penalties.

```text
gain(layer, partition, split) =
    100 * (reference_cost - candidate_cost) / reference_cost

robust_gain(layer, partition) = min_split gain(layer, partition, split)
```

A layer is eligible only when `robust_gain >= delta`. Eligible changes are
ordered by robust gain, then full-profile gain. Apply at most `B` changes to
the reference mapping. `B` defines a trust region around the reference and
limits accumulated model error and unmodeled cross-layer interactions.

## Acceptance

The optimizer generates candidates; it does not establish a speedup. Run the
candidate and a fresh linear control in separate processes with identical
model, data, topology, residency, and runtime switches. Accept only when:

```text
candidate mean decode latency < linear mean decode latency
and
max(candidate epoch latency) < min(linear epoch latency)
```

Otherwise retain the reference or linear mapping. Re-profile and revalidate
when the model, sequence bucket, batch size, top-k ratio, resident placement,
or GPU topology changes.

## Implementation

- Optimizer: `../../build_tp_head_mapping_robust_refine.py`
- End-to-end runner: `../../run_robust_mapping_experiment.sh`
- Acceptance summary: `../../summarize_robust_mapping_experiment.py`
