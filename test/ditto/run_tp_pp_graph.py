#!/usr/bin/env python3
"""Validate hash-offloading Ditto graphs against eager execution, per TP/PP layout.

The outer SGLang CUDA graph stays disabled. Each graph rank must log capture
and replay; a silently disabled Ditto graph fails this test. Artifacts include
commands, logs, token IDs, per-rank transfer statistics and summary.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def graph_paths(log: str) -> dict[str, Counter]:
    paths = {}
    for line in log.splitlines():
        match = re.search(r"Ditto decode graph path=(\w+)", line)
        if match is None:
            continue
        pp, tp = re.search(r"\bPP(\d+)\b", line), re.search(r"\bTP(\d+)\b", line)
        rank = f"pp{pp[1] if pp else 0}.tp{tp[1] if tp else 0}"
        paths.setdefault(rank, Counter())[match[1]] += 1
    return paths


def run_case(command, env, directory, timeout):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "command.json").write_text(json.dumps({
        "argv": command,
        "env": {k: v for k, v in env.items() if k.startswith("DITTO_") or k in
                ("CUDA_VISIBLE_DEVICES", "PYTHONPATH")},
    }, indent=2) + "\n")
    with (directory / "run.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if status:
        raise RuntimeError(f"Exited {status}; see {directory / 'run.log'}")
    return json.loads((directory / "output.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--aux-root", type=Path, default=Path("/jhe/myTransformer/auxiliary"))
    parser.add_argument("--aux-data-path", type=Path)
    parser.add_argument("--attn-pattern-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default="1x1,2x1,1x2,2x2", help="TPxPP layouts")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--prompt-repetitions", type=int, default=128)
    parser.add_argument("--permute-heads", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.max_new_tokens < 5 or args.repeat < 1 or args.prompt_repetitions < 1:
        parser.error("need >=5 output tokens, >=1 repeat and >=1 prompt repetitions")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = args.model_path.resolve()
    aux = (args.aux_data_path or args.aux_root / "hash_weights" / f"{model.name}-256").resolve()
    pattern = (args.attn_pattern_path or args.aux_root / "attn_pattern" / model.name).resolve()
    for path in (model / "config.json", aux / "hash_weight_layer_00.pt",
                 pattern / "heads_cosine_similarity.csv", pattern / "k_heads_importance.tsv",
                 pattern / "q_heads_importance.tsv"):
        if not path.is_file():
            parser.error(f"Missing required asset: {path}")
    layouts = []
    for case in args.cases.split(","):
        match = re.fullmatch(r"([1-9]\d*)x([1-9]\d*)", case)
        if match is None:
            parser.error(f"Invalid TPxPP layout: {case}")
        layouts.append(tuple(map(int, match.groups())))
    for tp, pp in layouts:
        for mode in ("eager", "ditto_graph"):
            if (output_dir / f"tp{tp}-pp{pp}-{mode}").exists():
                parser.error("Output cases already exist; choose a new --output-dir")
    prompt = output_dir / "prompt.txt"
    prompt.write_text("Paris is the capital of France. " * args.prompt_repetitions +
                      "\nQuestion: What is the capital of France?\nAnswer:")
    env = os.environ.copy()
    # Isolate the controlled comparison from inherited Ditto debug/ablation flags.
    for key in list(env):
        if key.startswith("DITTO_"):
            del env[key]
    env.update({"DITTO_TP_ENABLE_CUDA_GRAPH": "1", "DITTO_CUDA_GRAPH_DEBUG": "1",
                "DITTO_RECORD_TRANSFER_STATS": "1",
                "PYTHONPATH": os.pathsep.join([str(ROOT / "python"),
                                              str(ROOT / "sgl-kernel/python"),
                                              env.get("PYTHONPATH", "")])})
    if args.permute_heads:
        config = json.loads((model / "config.json").read_text())
        heads = config.get("num_key_value_heads", config["num_attention_heads"])
        orders = [[(h + layer + 1) % heads for h in range(heads)]
                  for layer in range(config["num_hidden_layers"])]
        mapping = output_dir / "head-orders.json"
        mapping.write_text(json.dumps(orders))
        env["DITTO_TP_KV_HEAD_ORDER_FILE"] = str(mapping)
    summary = []
    for tp, pp in layouts:
        baseline = None
        baseline_metadata = {}
        expected_ranks = {f"pp{p}.tp{t}" for p in range(pp) for t in range(tp)}
        for mode in ("eager", "ditto_graph"):
            directory = output_dir / f"tp{tp}-pp{pp}-{mode}"
            case_env = dict(env, DITTO_TRANSFER_STATS_FILE=str(directory / "transfer.json"))
            command = [sys.executable, str(ROOT / "python/sglang/srt/models/ditto/minimal_selftest.py"),
                       "--model-path", str(model), "--variant", "offloading",
                       "--tp-size", str(tp), "--pp-size", str(pp),
                       "--prompt-file", str(prompt), "--output-file", str(directory / "output.json"),
                       "--max-new-tokens", str(args.max_new_tokens), "--ignore-eos",
                       "--repeat", str(args.repeat), "--max-tokens", "4096",
                       "--max-total-tokens", "2048", "--gpu-memory-budget", "4",
                       "--rbits", "256", "--recent-budget", "64", "--num-skip-layers", "1",
                       "--aux-data-path", str(aux), "--attn-pattern-path", str(pattern),
                       "--disable-cuda-graph",
                       "--ditto-enable-cuda-graph" if mode == "ditto_graph" else "--ditto-disable-cuda-graph"]
            print(f"Running {directory.name}", flush=True)
            output = run_case(command, case_env, directory, args.timeout)
            runs = output["runs"] if args.repeat > 1 else [output]
            ids = [run["output_ids"] for run in runs]
            if len(ids) != args.repeat or any(len(x) != args.max_new_tokens for x in ids):
                raise AssertionError(f"Incomplete generation in {directory}")
            if any(x != ids[0] for x in ids):
                raise AssertionError(f"Repeated requests diverged in {directory}")
            paths = graph_paths((directory / "run.log").read_text())
            if set(paths) != expected_ranks:
                raise AssertionError(f"Missing graph path evidence: {paths}; expected {expected_ranks}")
            for rank, counts in paths.items():
                if mode == "ditto_graph":
                    if counts["capture"] < args.repeat or counts["replay"] < args.repeat or counts["fallback_eager"]:
                        raise AssertionError(f"Graph did not capture/replay on {rank}: {counts}")
                elif counts["fallback_eager"] < 1:
                    raise AssertionError(f"Eager baseline missing on {rank}: {counts}")
            if mode == "eager":
                baseline = ids
            elif ids != baseline:
                raise AssertionError(f"Ditto graph differs from eager for TP{tp}/PP{pp}: {ids} != {baseline}")
            stats = sorted(directory.glob("transfer*.json"))
            if len(stats) != tp * pp:
                raise AssertionError(f"Expected {tp * pp} transfer files; got {stats}")
            transfer_summary = []
            for path in stats:
                data = json.loads(path.read_text())
                if not data["enabled"] or data["total_h2d_bytes"] <= 0 or data["total_d2h_bytes"] <= 0:
                    raise AssertionError(f"No offloading transfer on {path}")
                lengths = [step["seq_len"] for step in data["steps"]]
                if any(b != a + 1 for a, b in zip(lengths, lengths[1:])):
                    raise AssertionError(f"Decode cache length did not advance: {path}: {lengths}")
                metadata = [(step["seq_len"], step["prefetch_k"]) for step in data["steps"]]
                if mode == "eager":
                    baseline_metadata[path.name] = metadata
                elif metadata != baseline_metadata[path.name]:
                    raise AssertionError(f"Graph cache metadata differs from eager: {path}")
                ks = [step["prefetch_k"] for step in data["steps"]]
                transfer_summary.append({"file": path.name,
                                        "seq_len_start": lengths[0], "seq_len_end": lengths[-1],
                                        "prefetch_k_min": min(ks), "prefetch_k_max": max(ks),
                                        **{key: data[key] for key in
                                           ("decode_steps", "total_h2d_bytes", "total_d2h_bytes")}})
            summary.append({"case": directory.name, "passed": True, "output_ids": ids,
                            "graph_paths": paths, "transfers": transfer_summary})
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"PASS {directory.name}: {dict(paths)}", flush=True)
    print(f"PASS: all {len(layouts)} layouts match eager; actual capture/replay verified.", flush=True)


if __name__ == "__main__":
    main()
