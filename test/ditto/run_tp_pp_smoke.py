"""Compare greedy full-attention outputs across single GPU, TP, PP and TP+PP.

Example (four visible GPUs required):
    python test/ditto/run_tp_pp_smoke.py --model-path /path/to/qwen-or-llama \
        --output-dir /tmp/ditto-tp-pp
"""
from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "python"), str(root / "sgl-kernel/python"), env.get("PYTHONPATH", "")]
    )
    # Full attention must be invariant to each layer's KV-head permutation.
    config = json.loads((args.model_path / "config.json").read_text())
    heads = config.get("num_key_value_heads", config["num_attention_heads"])
    orders = [
        [(i + layer + 1) % heads for i in range(heads)]
        for layer in range(config["num_hidden_layers"])
    ]
    mapping_path = output_dir / "head-orders.json"
    mapping_path.write_text(json.dumps(orders))
    baseline = None
    results = []
    for tp, pp, permute in [(1, 1, False), (2, 1, False), (1, 2, False),
                            (2, 2, False), (2, 2, True)]:
        name = f"tp{tp}-pp{pp}" + ("-permuted" if permute else "")
        output_path = output_dir / f"{name}.json"
        log_path = output_dir / f"{name}.log"
        case_env = env.copy()
        case_env.pop("DITTO_TP_KV_HEAD_ORDER_FILE", None)
        if permute:
            case_env["DITTO_TP_KV_HEAD_ORDER_FILE"] = str(mapping_path)
        cmd = [
            sys.executable, str(root / "python/sglang/srt/models/ditto/minimal_selftest.py"),
            "--model-path", str(args.model_path.resolve()), "--variant", "fullattn",
            "--tp-size", str(tp), "--pp-size", str(pp), "--prompt", args.prompt,
            "--max-new-tokens", str(args.max_new_tokens), "--max-tokens", "4096",
            "--gpu-memory-budget", "4", "--output-file", str(output_path),
        ]
        print(f"Running {name}; log: {log_path}", flush=True)
        # Give every case its own process group so a timeout cleans up workers.
        with log_path.open("w") as log:
            proc = subprocess.Popen(cmd, cwd=root, env=case_env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = proc.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise
        if status:
            raise RuntimeError(f"{name} exited {status}; see {log_path}")
        result = json.loads(output_path.read_text())
        ids = result["output_ids"]
        if not ids:
            raise AssertionError(f"{name} returned no output tokens")
        if baseline is None:
            baseline = ids
        if ids != baseline:
            raise AssertionError(f"{name} differs from single-GPU baseline: {ids} != {baseline}")
        results.append({"case": name, "output_ids": ids, "text": result["text"]})
        print(f"PASS {name}: {result['text']!r}", flush=True)
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"All {len(results)} configurations match exactly.", flush=True)


if __name__ == "__main__":
    main()
