#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_tp_head_mapping_cost_model import (
    CostConfig,
    StepSample,
    choose_partition,
    load_reference_orders,
    load_resident_heads,
    load_step_samples,
    score_partition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine a reference TP2 head mapping only where every profile shows "
            "consistent improvements across multiple step splits."
        )
    )
    parser.add_argument("--transfer-json", action="append", required=True)
    parser.add_argument(
        "--reference-mapping",
        help="Mapping to refine. Omit to refine the linear TP partition.",
    )
    parser.add_argument(
        "--resident-heads-file",
        help="Explicit resident placement. Omit when all optimized layers offload.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-steps", type=int, default=10)
    parser.add_argument("--skip-layers", type=int, default=1)
    parser.add_argument("--max-changes", type=int, default=5)
    parser.add_argument("--min-split-improvement-pct", type=float, default=4.0)
    parser.add_argument("--h2d-gbps", type=float, default=22.0)
    parser.add_argument("--d2h-gbps", type=float, default=22.0)
    parser.add_argument("--h2d-launch-us", type=float, default=2.0)
    parser.add_argument("--d2h-launch-us", type=float, default=1.0)
    parser.add_argument("--active-head-us", type=float, default=0.5)
    return parser.parse_args()


def _same_unoriented_partition(first: list[int], second: list[int]) -> bool:
    local_heads = len(first) // 2
    first_groups = {
        frozenset(first[:local_heads]),
        frozenset(first[local_heads:]),
    }
    second_groups = {
        frozenset(second[:local_heads]),
        frozenset(second[local_heads:]),
    }
    return first_groups == second_groups


def _improvement_pct(reference: float, candidate: float) -> float:
    if reference <= 0.0:
        return 0.0
    return 100.0 * (reference - candidate) / reference


def _build_splits(samples: list[StepSample]) -> dict[str, list[StepSample]]:
    profiles = sorted({sample.profile for sample in samples})
    splits = {}
    for profile in profiles:
        profile_samples = [sample for sample in samples if sample.profile == profile]
        midpoint = len(profile_samples) // 2
        prefix = "" if len(profiles) == 1 else f"{Path(profile).name}:"
        profile_splits = {
            "even_steps": profile_samples[::2],
            "odd_steps": profile_samples[1::2],
            "first_half": profile_samples[:midpoint],
            "second_half": profile_samples[midpoint:],
        }
        for split_name, split_samples in profile_splits.items():
            if not split_samples:
                raise ValueError(
                    f"Profile {profile} needs at least four post-skip samples"
                )
            splits[f"{prefix}{split_name}"] = split_samples
    return splits


def main() -> None:
    args = parse_args()
    if args.skip_steps < 0 or args.skip_layers < 0 or args.max_changes < 0:
        raise ValueError("skip counts and max changes must be non-negative")
    if args.min_split_improvement_pct < 0.0:
        raise ValueError("minimum split improvement must be non-negative")

    config = CostConfig(
        h2d_gbps=(args.h2d_gbps, args.h2d_gbps),
        d2h_gbps=(args.d2h_gbps, args.d2h_gbps),
        h2d_launch_us=args.h2d_launch_us,
        d2h_launch_us=args.d2h_launch_us,
        active_head_us=args.active_head_us,
        shrinkage=0.5,
        tail_quantile=0.95,
        tail_weight=0.10,
        imbalance_weight=0.05,
        worst_profile_weight=0.25,
    )
    samples, num_layers, num_heads, per_token_head_bytes = load_step_samples(
        args.transfer_json, args.skip_steps
    )
    references = load_reference_orders(
        args.reference_mapping, num_layers, num_heads
    )
    residents = load_resident_heads(args.resident_heads_file, num_layers)
    splits = _build_splits(samples)

    candidates = []
    for layer_idx in range(args.skip_layers, num_layers):
        reference = references[layer_idx]
        local_heads = num_heads // 2
        reference_groups = (
            tuple(reference[:local_heads]),
            tuple(reference[local_heads:]),
        )
        rank0, rank1, candidate_metrics = choose_partition(
            layer_idx,
            samples,
            residents[layer_idx],
            per_token_head_bytes,
            config,
            reference,
        )
        candidate_order = list(rank0 + rank1)
        if _same_unoriented_partition(reference, candidate_order):
            continue
        reference_metrics = score_partition(
            layer_idx,
            reference_groups,
            samples,
            residents[layer_idx],
            per_token_head_bytes,
            config,
        )
        full_improvement = _improvement_pct(
            reference_metrics["objective_us"], candidate_metrics["objective_us"]
        )
        split_improvements = {}
        for split_name, split_samples in splits.items():
            split_reference = score_partition(
                layer_idx,
                reference_groups,
                split_samples,
                residents[layer_idx],
                per_token_head_bytes,
                config,
            )
            split_candidate = score_partition(
                layer_idx,
                (rank0, rank1),
                split_samples,
                residents[layer_idx],
                per_token_head_bytes,
                config,
            )
            split_improvements[split_name] = _improvement_pct(
                split_reference["objective_us"], split_candidate["objective_us"]
            )
        minimum_split = min(split_improvements.values())
        candidates.append(
            {
                "layer_idx": layer_idx,
                "reference_order": reference,
                "candidate_order": candidate_order,
                "full_improvement_pct": full_improvement,
                "minimum_split_improvement_pct": minimum_split,
                "split_improvement_pct": split_improvements,
                "predicted_saving_us": (
                    reference_metrics["objective_us"]
                    - candidate_metrics["objective_us"]
                ),
                "eligible": minimum_split >= args.min_split_improvement_pct,
            }
        )

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    eligible.sort(
        key=lambda candidate: (
            -candidate["minimum_split_improvement_pct"],
            -candidate["full_improvement_pct"],
            candidate["layer_idx"],
        )
    )
    selected = eligible[: args.max_changes]
    orders = [order[:] for order in references]
    for candidate in selected:
        orders[candidate["layer_idx"]] = candidate["candidate_order"]

    payload = {
        "orders": orders,
        "metadata": {
            "objective": "robust split-constrained trust-region refinement",
            "transfer_json": args.transfer_json,
            "reference_mapping": (
                str(Path(args.reference_mapping).resolve())
                if args.reference_mapping
                else None
            ),
            "resident_heads_file": (
                str(Path(args.resident_heads_file).resolve())
                if args.resident_heads_file
                else None
            ),
            "profile_steps": len(samples),
            "skip_steps": args.skip_steps,
            "skip_layers": args.skip_layers,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "per_token_head_bytes": per_token_head_bytes,
            "split_names": list(splits),
            "min_split_improvement_pct": args.min_split_improvement_pct,
            "max_changes": args.max_changes,
            "selected_layers": [candidate["layer_idx"] for candidate in selected],
            "predicted_saving_us": sum(
                candidate["predicted_saving_us"] for candidate in selected
            ),
        },
        "selected": selected,
        "all_candidates": sorted(candidates, key=lambda item: item["layer_idx"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(selected)}-layer robust refinement to {output}; "
        f"layers={payload['metadata']['selected_layers']}, "
        f"predicted saving={payload['metadata']['predicted_saving_us']:.3f} us"
    )


if __name__ == "__main__":
    main()
