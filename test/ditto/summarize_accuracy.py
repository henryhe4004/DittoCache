#!/usr/bin/env python3

import json
import os
import statistics
import sys


def flatten_scores(raw):
    flat = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            vals = [float(x) for x in value.values()]
            flat[key] = sum(vals) / len(vals) if vals else 0.0
        else:
            flat[key] = float(value)
    return flat


def main():
    if len(sys.argv) != 5:
        print(
            "usage: summarize_accuracy.py RESULT_JSON BASELINE_RESULT MAX_AVG_DROP MAX_SINGLE_DROP",
            file=sys.stderr,
        )
        return 2

    result_path, baseline_path, max_avg_drop, max_single_drop = sys.argv[1:]
    max_avg_drop = float(max_avg_drop)
    max_single_drop = float(max_single_drop)

    with open(result_path, "r", encoding="utf-8") as f:
        result = flatten_scores(json.load(f))

    vals = list(result.values())
    avg = sum(vals) / len(vals) if vals else 0.0
    med = statistics.median(vals) if vals else 0.0
    print(f"[SUMMARY] task_count={len(vals)} avg_score={avg:.2f} median_score={med:.2f}")

    if not baseline_path:
        print("[SUMMARY] no baseline comparison. set BASELINE_RESULT=/path/to/result.json to compare.")
        return 0

    if not os.path.exists(baseline_path):
        print(f"[ERROR] baseline not found: {baseline_path}")
        return 2

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = flatten_scores(json.load(f))

    common = sorted(set(result).intersection(baseline))
    if not common:
        print("[ERROR] no overlapping tasks with baseline")
        return 2

    drops = {task: baseline[task] - result[task] for task in common}
    avg_drop = sum(drops.values()) / len(drops)
    worst_task = max(drops, key=drops.get)
    worst_drop = drops[worst_task]

    print(
        "[COMPARE] "
        f"tasks={len(common)} avg_drop={avg_drop:.2f} "
        f"worst_drop={worst_drop:.2f}({worst_task}) "
        f"thresholds(avg<={max_avg_drop}, worst<={max_single_drop})"
    )

    if avg_drop <= max_avg_drop and worst_drop <= max_single_drop:
        print("[PASS] accuracy drop is within tolerance.")
        return 0

    print("[FAIL] accuracy drop is larger than tolerance.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
