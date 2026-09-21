#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


T_CRITICAL_DF2_95 = 4.3026527


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a robust-mapping candidate/linear paired benchmark."
    )
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--linear-result", required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def variant_stats(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    latency = [float(value) for value in payload["epoch_decode_latency_ms_per_step"]]
    if len(latency) < 3:
        raise ValueError(f"{path} needs at least three measured epochs")
    return {
        "result": str(path.resolve()),
        "epoch_decode_latency_ms": latency,
        "mean_decode_latency_ms": statistics.mean(latency),
        "median_decode_latency_ms": statistics.median(latency),
        "stdev_decode_latency_ms": statistics.stdev(latency),
        "mean_decode_tps": float(payload["avg_decode_tokens_per_s"]),
        "mean_prefill_latency_s": float(payload["avg_prefill_latency_s"]),
        "runtime_meta": payload["runtime_meta"],
        "controls": {
            key: payload[key]
            for key in (
                "config_file",
                "data",
                "batch_size",
                "max_seq_len",
                "warmup",
                "epoch",
                "input_tokens",
            )
        },
    }


def main() -> None:
    args = parse_args()
    candidate = variant_stats(Path(args.candidate_result))
    linear = variant_stats(Path(args.linear_result))
    if candidate["controls"] != linear["controls"]:
        raise ValueError("Candidate and linear benchmark controls differ")
    if candidate["runtime_meta"] != linear["runtime_meta"]:
        raise ValueError("Candidate and linear runtime metadata differ")
    latency_delta = (
        candidate["mean_decode_latency_ms"] - linear["mean_decode_latency_ms"]
    )
    standard_error = math.sqrt(
        candidate["stdev_decode_latency_ms"] ** 2
        / len(candidate["epoch_decode_latency_ms"])
        + linear["stdev_decode_latency_ms"] ** 2
        / len(linear["epoch_decode_latency_ms"])
    )
    conservative_half_width = T_CRITICAL_DF2_95 * standard_error
    comparison = {
        "candidate": args.candidate_name,
        "candidate_stats": candidate,
        "linear_stats": linear,
        "candidate_minus_linear_latency_ms": latency_delta,
        "latency_improvement_pct": (
            -100.0 * latency_delta / linear["mean_decode_latency_ms"]
        ),
        "tps_improvement_pct": 100.0
        * (candidate["mean_decode_tps"] / linear["mean_decode_tps"] - 1.0),
        "prefill_delta_s": (
            candidate["mean_prefill_latency_s"]
            - linear["mean_prefill_latency_s"]
        ),
        "all_candidate_epochs_faster_than_all_linear_epochs": (
            max(candidate["epoch_decode_latency_ms"])
            < min(linear["epoch_decode_latency_ms"])
        ),
        "difference_of_means_standard_error_ms": standard_error,
        "conservative_df2_95pct_latency_delta_ci_ms": [
            latency_delta - conservative_half_width,
            latency_delta + conservative_half_width,
        ],
    }
    comparison["decision"] = (
        "accept"
        if comparison["latency_improvement_pct"] > 0.0
        and comparison["all_candidate_epochs_faster_than_all_linear_epochs"]
        else "reject"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
