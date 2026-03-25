from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig

from sglang import Engine


def build_args():
    parser = argparse.ArgumentParser(description="LiteCache minimal smoke test.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello LiteCache")
    parser.add_argument("--variant", type=str, default="loki", choices=["loki", "hash", "infinigen", "quest"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-budget", type=float, default=16.0)
    parser.add_argument("--token-budget", type=float, default=0.2)
    parser.add_argument("--sink-budget", type=int, default=4)
    parser.add_argument("--recent-budget", type=int, default=128)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--rbits", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--aux-data-path", type=str, default=None)
    parser.add_argument("--attn-pattern-path", type=str, default="")
    parser.add_argument("--num-omp-threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=0.92)
    parser.add_argument("--max-total-tokens", type=int, default=512)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument(
        "--disable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_true",
        default=True,
        help="Disable CUDA graph capture (recommended for LiteCache bring-up debug).",
    )
    parser.add_argument(
        "--enable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_false",
        help="Enable CUDA graph capture (may improve perf after bring-up is stable).",
    )
    parser.add_argument(
        "--litecache-custom-config-path",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON file that directly provides model override args "
            "(e.g. minimal_custom_config_template.json). If omitted, this script "
            "builds a LiteCache override from CLI flags."
        ),
    )
    parser.add_argument(
        "--disable-dual-chunk-config",
        action="store_true",
        default=False,
        help=(
            "Deprecated compatibility flag. Dual-chunk override injection is now "
            "disabled by default and this flag is ignored."
        ),
    )
    return parser.parse_args()


def resolve_litecache_architecture(model_path: str) -> str:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    archs = [str(x).lower() for x in (getattr(cfg, "architectures", None) or [])]

    if "qwen2" in model_type or any("qwen2" in a for a in archs):
        return "LiteCacheQwen2ForCausalLM"
    if "llama" in model_type or any("llama" in a for a in archs):
        return "LiteCacheLlamaForCausalLM"
    return "LiteCacheLlamaForCausalLM"


def build_litecache_override(args) -> dict:
    architecture = resolve_litecache_architecture(args.model_path)
    return {
        "architectures": [architecture],
        "litecache_variant": args.variant,
        "custom_config": {
            "enable_cuda_graph": False,
            "new_config": True,
            "is_profiling": False,
            "num_channels": args.num_channels,
            "rbits": args.rbits,
            "block_size": args.block_size,
            "aux_data_path": args.aux_data_path,
            "kvcache_manager_config": {
                "max_tokens": args.max_tokens,
                "max_batch_size": args.max_batch_size,
                "gpu_memory_budget": args.gpu_memory_budget,
            },
            "sparse_attention_config": {
                "token_budget": args.token_budget,
                "sink_budget": args.sink_budget,
                "recent_budget": args.recent_budget,
            },
            "offload_config": {
                "attn_pattern_path": args.attn_pattern_path,
                "reuse_threshold_upper": 0.95,
                "reuse_threshold_lower": 0.7,
                "decay_p": 2.0,
                "cosine_padding": 0.02,
                "num_skip_layers": 0,
                "num_overlapped_heads": 0,
                "num_omp_threads": args.num_omp_threads,
            },
        },
    }


def main():
    args = build_args()
    torch = __import__("torch")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in current environment. "
            "Run this script inside a GPU-enabled container/session "
            "(e.g. your `jhe_sglang_lite` docker with NVIDIA runtime)."
        )

    if args.litecache_custom_config_path:
        cfg_path = Path(args.litecache_custom_config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"LiteCache custom config file not found: {cfg_path}")
        model_override = json.loads(cfg_path.read_text())
    else:
        model_override = build_litecache_override(args)

    if args.disable_dual_chunk_config:
        print(
            "[litecache] --disable-dual-chunk-config is deprecated and ignored. "
            "No dual_chunk_attention_config override is injected."
        )

    engine = Engine(
        model_path=args.model_path,
        model_impl="auto",
        trust_remote_code=True,
        log_level="info",
        device=args.device,
        attention_backend=args.attention_backend,
        json_model_override_args=json.dumps(model_override),
        # Minimize SGLang runtime features for bring-up.
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        enable_mixed_chunk=False,
        schedule_policy="fcfs",
        # Minimize SGLang pre-allocated pools to avoid OOM during bring-up.
        mem_fraction_static=args.mem_fraction_static,
        max_total_tokens=args.max_total_tokens,
        max_running_requests=args.max_running_requests,
        page_size=args.page_size,
        disable_cuda_graph=args.disable_cuda_graph,
    )

    print(
        "[litecache] active override: "
        f"architectures={model_override.get('architectures')}, "
        f"variant={model_override.get('litecache_variant', args.variant)}"
    )

    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
    }
    out = engine.generate(prompt=args.prompt, sampling_params=sampling_params)
    print(out)


if __name__ == "__main__":
    main()
