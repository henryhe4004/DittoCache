from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional


def _get(obj: Any, key: str, default: Any):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def ensure_litecache_custom_config(custom_config: Optional[Any], hf_config: Optional[Any] = None):
    """
    Normalize LiteCache custom config into an object with all required attributes.
    Accepts dict/object/None and fills missing fields with safe defaults.
    """
    if hf_config is not None:
        default_max_tokens = int(
            getattr(hf_config, "max_position_embeddings", 4096)
            or getattr(hf_config, "sliding_window", 4096)
            or 4096
        )
    else:
        default_max_tokens = 4096

    src = custom_config
    kmc = _get(src, "kvcache_manager_config", None)
    sac = _get(src, "sparse_attention_config", None)
    ofc = _get(src, "offload_config", None)

    normalized = _ns(
        enable_cuda_graph=bool(_get(src, "enable_cuda_graph", False)),
        new_config=bool(_get(src, "new_config", False)),
        is_profiling=bool(_get(src, "is_profiling", False)),
        chunk_prefill_size=int(_get(src, "chunk_prefill_size", 0)),
        num_channels=int(_get(src, "num_channels", 32)),
        rbits=int(_get(src, "rbits", 32)),
        block_size=int(_get(src, "block_size", 64)),
        aux_data_path=_get(src, "aux_data_path", None),
        kvcache_manager_config=_ns(
            max_tokens=int(_get(kmc, "max_tokens", default_max_tokens)),
            max_batch_size=int(_get(kmc, "max_batch_size", 1)),
            gpu_memory_budget=float(_get(kmc, "gpu_memory_budget", 16.0)),
        ),
        sparse_attention_config=_ns(
            token_budget=float(_get(sac, "token_budget", 0.2)),
            sink_budget=int(_get(sac, "sink_budget", 4)),
            recent_budget=int(_get(sac, "recent_budget", 128)),
            selective_start_len=int(_get(sac, "selective_start_len", 0)),
        ),
        offload_config=_ns(
            attn_pattern_path=str(_get(ofc, "attn_pattern_path", "")),
            reuse_threshold_upper=float(_get(ofc, "reuse_threshold_upper", 0.95)),
            reuse_threshold_lower=float(_get(ofc, "reuse_threshold_lower", 0.7)),
            decay_p=float(_get(ofc, "decay_p", 2.0)),
            cosine_padding=float(_get(ofc, "cosine_padding", 0.02)),
            # <=0 means disabled: do not force gather refresh by reuse count.
            max_reuse_count=int(_get(ofc, "max_reuse_count", 0)),
            num_skip_layers=int(_get(ofc, "num_skip_layers", 0)),
            num_overlapped_heads=int(_get(ofc, "num_overlapped_heads", 0)),
            num_omp_threads=int(_get(ofc, "num_omp_threads", 4)),
        ),
    )
    return normalized
