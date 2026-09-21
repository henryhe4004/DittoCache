#!/usr/bin/env python3
"""Test ragged batched prefill and repeated requests for one TP/PP layout."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "sgl-kernel/python")]
# Multiprocessing workers must inherit the same source checkout.
os.environ["PYTHONPATH"] = os.pathsep.join(sys.path[:2] + [os.environ.get("PYTHONPATH", "")])

from sglang import Engine
from transformers import AutoConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    model_type = AutoConfig.from_pretrained(args.model_path).model_type
    architectures = {"qwen2": "DittoQwen2ForCausalLM", "llama": "DittoLlamaForCausalLM"}
    if model_type not in architectures:
        parser.error(f"Unsupported model type: {model_type}")
    override = {"architectures": [architectures[model_type]], "ditto_variant": "fullattn",
                "custom_config": {"enable_cuda_graph": False, "chunk_prefill_size": 4,
                                  "kvcache_manager_config": {"max_tokens": 4096,
                                                            "max_batch_size": 2,
                                                            "gpu_memory_budget": 4}}}
    engine = Engine(model_path=args.model_path, tp_size=args.tp_size, pp_size=args.pp_size,
                    pp_max_micro_batch_size=2, json_model_override_args=json.dumps(override),
                    disable_cuda_graph=True, disable_overlap_schedule=True,
                    chunked_prefill_size=-1, disable_radix_cache=True,
                    max_running_requests=2, max_total_tokens=512, random_seed=42)
    try:
        prompts = ["The capital of France is", "Name a city in France. The capital of France is"]
        runs = [engine.generate(prompts, {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True})
                for _ in range(2)]
        ids = [[r["output_ids"] for r in run] for run in runs]
        assert ids[0] == ids[1], ids
        if args.reference:
            reference = json.loads(args.reference.read_text())
            expected = [[r["output_ids"] for r in run] for run in reference]
            assert ids == expected, (ids, expected)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(runs, indent=2) + "\n")
        print("PASS", ids)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
