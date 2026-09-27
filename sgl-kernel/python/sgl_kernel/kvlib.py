from __future__ import annotations

from typing import Optional

import torch


def hamming_score_norm(
    key_code: torch.Tensor,
    query_code: torch.Tensor,
    key_norm: torch.Tensor,
    rbit: int,
    seq_len: int,
    sink: int = 0,
    recent: int = 0,
    use_key_norm: bool = True,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_hamming_score_norm(
        key_code, query_code, key_norm, rbit, seq_len, sink, recent, use_key_norm
    )


def hamming_score(
    key_code: torch.Tensor,
    query_code: torch.Tensor,
    rbit: int,
    seq_len: int,
    sink: int = 0,
    recent: int = 0,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_hamming_score(key_code, query_code, rbit, seq_len, sink, recent)


def hamming_score_head_mask(
    key_code: torch.Tensor,
    query_code: torch.Tensor,
    head_mask: torch.Tensor,
    rbit: int,
    seq_len: int,
    sink: int = 0,
    recent: int = 0,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_hamming_score_head_mask(
        key_code, query_code, head_mask, rbit, seq_len, sink, recent
    )


def static_hamming_score_mask(
    key_codes: torch.Tensor,
    query_code: torch.Tensor,
    mask: torch.Tensor,
    score: torch.Tensor,
    seqlen: torch.Tensor,
    rbit: int,
    max_value: float,
    min_value: float,
    sink: int = 0,
    recent: int = 0,
    skip_sink: int = 0,
    skip_recent: int = 0,
) -> None:
    torch.ops.sgl_kernel.kvlib_static_hamming_score_mask(
        key_codes,
        query_code,
        mask,
        score,
        seqlen,
        rbit,
        max_value,
        min_value,
        sink,
        recent,
        skip_sink,
        skip_recent,
    )


def batch_topk(data: torch.Tensor, k: int, largest: bool) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_batch_topk(data, k, largest)


def batch_topk_masked(
    data: torch.Tensor,
    bh_mask: torch.Tensor,
    out_index: torch.Tensor,
    out_values: torch.Tensor,
    real_len: torch.Tensor,
    real_k: torch.Tensor,
    largest: bool,
) -> None:
    torch.ops.sgl_kernel.kvlib_batch_topk_masked(data, bh_mask, out_index, out_values, real_len, real_k, largest)


def kvcache_append(kv_cache: torch.Tensor, key: torch.Tensor, value: torch.Tensor, insert_pos: int) -> None:
    torch.ops.sgl_kernel.kvlib_kvcache_append(kv_cache, key, value, insert_pos)


def kvcache_append_head_sparse(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_ids: torch.Tensor,
    insert_pos: int,
) -> None:
    torch.ops.sgl_kernel.kvlib_kvcache_append_head_sparse(kv_cache, key, value, head_ids, insert_pos)


def kvcache_append2(dst_kv: torch.Tensor, src_kv: torch.Tensor, dst_pos: int, src_pos: int) -> None:
    torch.ops.sgl_kernel.kvlib_kvcache_append2(dst_kv, src_kv, dst_pos, src_pos)


def kvcache_append_tensor_pos(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    insert_pos: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.kvlib_kvcache_append_tensor_pos(kv_cache, key, value, insert_pos)


def kvcache_append_tensor_pos_head_sparse(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_ids: torch.Tensor,
    insert_pos: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.kvlib_kvcache_append_tensor_pos_head_sparse(kv_cache, key, value, head_ids, insert_pos)


def real_indices_and_launch_prefetch(
    indices: torch.Tensor,
    gpu_gather_mask: torch.Tensor,
    output: torch.Tensor,
    gather_flag: torch.Tensor,
    cpu_ready_mask: torch.Tensor,
    cache_seq_len: int,
    batch_size: int,
    num_heads: int,
    layer_idx: int,
) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "real_indices_and_launch_prefetch"
    ):
        _kvlib_cpu_gather.real_indices_and_launch_prefetch(
            indices,
            gpu_gather_mask,
            output,
            gather_flag,
            cpu_ready_mask,
            cache_seq_len,
            batch_size,
            num_heads,
            layer_idx,
        )
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.real_indices_and_launch_prefetch is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def static_launch_prefetch(
    gpu_indices: torch.Tensor,
    gpu_gather_mask: torch.Tensor,
    gpu_index_length: torch.Tensor,
    cpu_indices: torch.Tensor,
    cpu_gather_flag: torch.Tensor,
    cpu_ready_mask: torch.Tensor,
    batch_size: int,
    max_cache_seqlen: int,
    num_heads: int,
    layer_idx: int,
) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "static_launch_prefetch"
    ):
        _kvlib_cpu_gather.static_launch_prefetch(
            gpu_indices,
            gpu_gather_mask,
            gpu_index_length,
            cpu_indices,
            cpu_gather_flag,
            cpu_ready_mask,
            batch_size,
            max_cache_seqlen,
            num_heads,
            layer_idx,
        )
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.static_launch_prefetch is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def decode_append_offload_wait(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    gpu_kv_buffer: torch.Tensor,
    cpu_kv_cache: torch.Tensor,
    gpu_append_pos: int,
    cpu_append_pos: int,
    ready_flags: torch.Tensor,
    cpu_head_ids: torch.Tensor,
) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "decode_append_offload_wait"
    ):
        _kvlib_cpu_gather.decode_append_offload_wait(
            key_states,
            value_states,
            gpu_kv_buffer,
            cpu_kv_cache,
            gpu_append_pos,
            cpu_append_pos,
            ready_flags,
            cpu_head_ids,
        )
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.decode_append_offload_wait is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def decode_append_offload_tensor_pos_wait(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    gpu_kv_buffer: torch.Tensor,
    cpu_kv_cache: torch.Tensor,
    gpu_append_pos: torch.Tensor,
    cpu_append_pos: torch.Tensor,
    ready_flags: torch.Tensor,
    cpu_head_ids: torch.Tensor,
) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "decode_append_offload_tensor_pos_wait"
    ):
        _kvlib_cpu_gather.decode_append_offload_tensor_pos_wait(
            key_states,
            value_states,
            gpu_kv_buffer,
            cpu_kv_cache,
            gpu_append_pos,
            cpu_append_pos,
            ready_flags,
            cpu_head_ids,
        )
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.decode_append_offload_tensor_pos_wait is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def wait_kv_data(ready_flags: torch.Tensor, batch_size: int, num_heads: int) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "wait_kv_data"
    ):
        _kvlib_cpu_gather.wait_kv_data(ready_flags, batch_size, num_heads)
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.wait_kv_data is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def gather_gpu_kvcache(
    indices: torch.Tensor,
    src_key: torch.Tensor,
    src_value: torch.Tensor,
    dst_key: torch.Tensor,
    dst_value: torch.Tensor,
    head_ids: torch.Tensor,
    sink_recent_budget: int,
) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "gather_gpu_kvcache"
    ):
        _kvlib_cpu_gather.gather_gpu_kvcache(
            indices,
            src_key,
            src_value,
            dst_key,
            dst_value,
            head_ids,
            sink_recent_budget,
        )
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.gather_gpu_kvcache is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def block_id_to_token_id(
    block_idx: torch.Tensor, block_size: int, num_sink: int, num_recent: int, seq_length: int
) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_block_id_to_token_id(block_idx, block_size, num_sink, num_recent, seq_length)


def block_id_to_token_id_head_mask(
    block_idx: torch.Tensor,
    block_size: int,
    num_sink: int,
    num_recent: int,
    seq_length: int,
    head_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_block_id_to_token_id_head_mask(
        block_idx, block_size, num_sink, num_recent, seq_length, head_mask
    )


def create_tensor(size, dtype: int) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_create_tensor(size, dtype)


def create_cpu_gather_engine_v3(
    num_omp_threads: int,
    cpu_kv_data,
    gpu_kv_buffer,
    dst_head_index,
    num_gpu_heads,
    cpu_indices_buffer: torch.Tensor,
    launch_flag: torch.Tensor,
    ready_flags,
    max_batch_size: int,
    sink_recent_budget: int,
    num_heads: int,
    head_dim: int,
    debug: bool = False,
) -> int:
    return torch.ops.sgl_kernel.kvlib_create_cpu_gather_engine_v3(
        num_omp_threads,
        cpu_kv_data,
        gpu_kv_buffer,
        dst_head_index,
        num_gpu_heads,
        cpu_indices_buffer,
        launch_flag,
        ready_flags,
        max_batch_size,
        sink_recent_budget,
        num_heads,
        head_dim,
        debug,
    )


__all__ = [
    "hamming_score_norm",
    "hamming_score",
    "hamming_score_head_mask",
    "static_hamming_score_mask",
    "batch_topk",
    "batch_topk_masked",
    "kvcache_append",
    "kvcache_append_head_sparse",
    "kvcache_append2",
    "kvcache_append_tensor_pos",
    "kvcache_append_tensor_pos_head_sparse",
    "real_indices_and_launch_prefetch",
    "static_launch_prefetch",
    "decode_append_offload_wait",
    "decode_append_offload_tensor_pos_wait",
    "wait_kv_data",
    "gather_gpu_kvcache",
    "block_id_to_token_id",
    "block_id_to_token_id_head_mask",
    "create_tensor",
    "create_cpu_gather_engine_v3",
]

"""
KVLib operators for sglang (from myTransformer).
Includes: hamming score, topk, kvcache append, prefetch/offload, gather, block_id, hash encode, check_reuse.
"""

import torch


def hamming_score_norm(
    key_code: torch.Tensor,
    query_code: torch.Tensor,
    key_norm: torch.Tensor,
    rbit: int,
    seq_len: int,
    sink: int = 0,
    recent: int = 0,
    use_key_norm: bool = False,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_hamming_score_norm.default(
        key_code, query_code, key_norm, rbit, seq_len, sink, recent, use_key_norm
    )


def hamming_score(
    key_code: torch.Tensor,
    query_code: torch.Tensor,
    rbit: int,
    seq_len: int,
    sink: int = 0,
    recent: int = 0,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_hamming_score.default(
        key_code, query_code, rbit, seq_len, sink, recent
    )


def hamming_score_head_mask(
    key_code: torch.Tensor,
    query_code: torch.Tensor,
    head_mask: torch.Tensor,
    rbit: int,
    seq_len: int,
    sink: int = 0,
    recent: int = 0,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_hamming_score_head_mask.default(
        key_code, query_code, head_mask, rbit, seq_len, sink, recent
    )


def static_hamming_score_mask(
    key_codes: torch.Tensor,
    query_code: torch.Tensor,
    mask: torch.Tensor,
    score: torch.Tensor,
    seqlen: torch.Tensor,
    rbit: int,
    max_value: float,
    min_value: float,
    sink: int,
    recent: int,
    skip_sink: int,
    skip_recent: int,
) -> None:
    torch.ops.sgl_kernel.kvlib_static_hamming_score_mask.default(
        key_codes, query_code, mask, score, seqlen,
        rbit, max_value, min_value, sink, recent, skip_sink, skip_recent
    )


def batch_topk(data: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    """Batch top-k indices. data: [B, H, S], returns [B, H, k] int32."""
    return torch.ops.sgl_kernel.kvlib_batch_topk.default(data, k, largest)


def batch_topk_masked(
    data: torch.Tensor,
    bh_mask: torch.Tensor,
    out_index: torch.Tensor,
    out_values: torch.Tensor,
    real_len: torch.Tensor,
    real_k: torch.Tensor,
    largest: bool = True,
) -> None:
    """Masked batch top-k (requires RAFT when built from myTransformer)."""
    torch.ops.sgl_kernel.kvlib_batch_topk_masked.default(
        data, bh_mask, out_index, out_values, real_len, real_k, largest
    )


def kvcache_append(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    insert_pos: int,
) -> None:
    torch.ops.sgl_kernel.kvlib_kvcache_append.default(kv_cache, key, value, insert_pos)


def kvcache_append_head_sparse(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_ids: torch.Tensor,
    insert_pos: int,
) -> None:
    torch.ops.sgl_kernel.kvlib_kvcache_append_head_sparse.default(
        kv_cache, key, value, head_ids, insert_pos
    )


def kvcache_append2(
    dst_kv: torch.Tensor,
    src_kv: torch.Tensor,
    dst_pos: int,
    src_pos: int,
) -> None:
    torch.ops.sgl_kernel.kvlib_kvcache_append2.default(dst_kv, src_kv, dst_pos, src_pos)


def kvcache_append_tensor_pos(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    insert_pos: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.kvlib_kvcache_append_tensor_pos.default(
        kv_cache, key, value, insert_pos
    )


def kvcache_append_tensor_pos_head_sparse(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_ids: torch.Tensor,
    insert_pos: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.kvlib_kvcache_append_tensor_pos_head_sparse.default(
        kv_cache, key, value, head_ids, insert_pos
    )


def real_indices_and_launch_prefetch(
    indices: torch.Tensor,
    gpu_gather_mask: torch.Tensor,
    output: torch.Tensor,
    gather_flag: torch.Tensor,
    cpu_ready_mask: torch.Tensor,
    cache_seq_len: int,
    batch_size: int,
    num_heads: int,
    layer_idx: int,
) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "real_indices_and_launch_prefetch"
    ):
        _kvlib_cpu_gather.real_indices_and_launch_prefetch(
            indices,
            gpu_gather_mask,
            output,
            gather_flag,
            cpu_ready_mask,
            cache_seq_len,
            batch_size,
            num_heads,
            layer_idx,
        )
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.real_indices_and_launch_prefetch is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def static_launch_prefetch(
    gpu_indices: torch.Tensor,
    gpu_gather_mask: torch.Tensor,
    gpu_index_length: torch.Tensor,
    cpu_indices: torch.Tensor,
    cpu_gather_flag: torch.Tensor,
    cpu_ready_mask: torch.Tensor,
    batch_size: int,
    max_cache_seqlen: int,
    num_heads: int,
    layer_idx: int,
) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "static_launch_prefetch"
    ):
        _kvlib_cpu_gather.static_launch_prefetch(
            gpu_indices,
            gpu_gather_mask,
            gpu_index_length,
            cpu_indices,
            cpu_gather_flag,
            cpu_ready_mask,
            batch_size,
            max_cache_seqlen,
            num_heads,
            layer_idx,
        )
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.static_launch_prefetch is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def decode_append_offload_wait(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    gpu_kv_buffer: torch.Tensor,
    cpu_kv_cache: torch.Tensor,
    gpu_append_pos: int,
    cpu_append_pos: int,
    ready_flags: torch.Tensor,
    cpu_head_ids: torch.Tensor,
) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "decode_append_offload_wait"
    ):
        _kvlib_cpu_gather.decode_append_offload_wait(
            key_states,
            value_states,
            gpu_kv_buffer,
            cpu_kv_cache,
            gpu_append_pos,
            cpu_append_pos,
            ready_flags,
            cpu_head_ids,
        )
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.decode_append_offload_wait is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def decode_append_offload_tensor_pos_wait(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    gpu_kv_buffer: torch.Tensor,
    cpu_kv_cache: torch.Tensor,
    gpu_append_pos: torch.Tensor,
    cpu_append_pos: torch.Tensor,
    ready_flags: torch.Tensor,
    cpu_head_ids: torch.Tensor,
) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "decode_append_offload_tensor_pos_wait"
    ):
        _kvlib_cpu_gather.decode_append_offload_tensor_pos_wait(
            key_states,
            value_states,
            gpu_kv_buffer,
            cpu_kv_cache,
            gpu_append_pos,
            cpu_append_pos,
            ready_flags,
            cpu_head_ids,
        )
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.decode_append_offload_tensor_pos_wait is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def wait_kv_data(ready_flags: torch.Tensor, batch_size: int, num_heads: int) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "wait_kv_data"
    ):
        _kvlib_cpu_gather.wait_kv_data(ready_flags, batch_size, num_heads)
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.wait_kv_data is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def gather_gpu_kvcache(
    indices: torch.Tensor,
    src_key: torch.Tensor,
    src_value: torch.Tensor,
    dst_key: torch.Tensor,
    dst_value: torch.Tensor,
    head_ids: torch.Tensor,
    sink_recent_budget: int,
) -> None:
    if "_kvlib_cpu_gather" in globals() and _kvlib_cpu_gather is not None and hasattr(
        _kvlib_cpu_gather, "gather_gpu_kvcache"
    ):
        _kvlib_cpu_gather.gather_gpu_kvcache(
            indices,
            src_key,
            src_value,
            dst_key,
            dst_value,
            head_ids,
            sink_recent_budget,
        )
    else:
        raise RuntimeError(
            "kvlib_cpu_gather.gather_gpu_kvcache is not available; "
            "ensure sgl-kernel was built with kvlib_cpu_gather target."
        )


def block_id_to_token_id(
    block_idx: torch.Tensor,
    block_size: int,
    num_sink: int,
    num_recent: int,
    seq_length: int,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_block_id_to_token_id.default(
        block_idx, block_size, num_sink, num_recent, seq_length
    )


def block_id_to_token_id_head_mask(
    block_idx: torch.Tensor,
    block_size: int,
    num_sink: int,
    num_recent: int,
    seq_length: int,
    head_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.kvlib_block_id_to_token_id_head_mask.default(
        block_idx, block_size, num_sink, num_recent, seq_length, head_mask
    )


# -----------------------------------------------------------------------------
# create_tensor: allocate pinned host memory (for offload). dtype 16=fp16, 32=fp32.
# -----------------------------------------------------------------------------


def create_tensor(size: list, dtype: int) -> torch.Tensor:
    """Allocate pinned host tensor. size: shape, dtype: 16 (fp16) or 32 (fp32)."""
    return torch.ops.sgl_kernel.kvlib_create_tensor.default(size, dtype)


# -----------------------------------------------------------------------------
# CPUGatherEngineV3: CPU-side gather engine for offload, via dedicated module.
# -----------------------------------------------------------------------------

try:
    from sgl_kernel import kvlib_cpu_gather as _kvlib_cpu_gather
except ImportError:
    _kvlib_cpu_gather = None


class CPUGatherEngineV3:
    """Python wrapper for CPUGatherEngineV3 (offload) backed by kvlib_cpu_gather."""

    def __init__(
        self,
        num_omp_threads: int,
        cpu_kv_data: list,
        gpu_kv_buffer: list,
        dst_head_index: list,
        num_gpu_heads: list,
        cpu_indices_buffer: torch.Tensor,
        launch_flag: torch.Tensor,
        ready_flags: list,
        max_batch_size: int,
        sink_recent_budget: int,
        num_heads: int,
        head_dim: int,
        debug: bool = False,
        transfer_backend: str = "gdrcopy",
    ):
        if _kvlib_cpu_gather is None:
            raise RuntimeError(
                "kvlib_cpu_gather extension is not available; "
                "ensure sgl-kernel was built with gdrapi and kvlib_cpu_gather target."
            )
        self._impl = _kvlib_cpu_gather.CPUGatherEngineV3(
            num_omp_threads,
            cpu_kv_data,
            gpu_kv_buffer,
            dst_head_index,
            num_gpu_heads,
            cpu_indices_buffer,
            launch_flag,
            ready_flags,
            max_batch_size,
            sink_recent_budget,
            num_heads,
            head_dim,
            debug,
            transfer_backend,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Rely on C++ destructor for cleanup.
        self._impl = None


def flash_index_decode(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    gather_idx: torch.Tensor,
    gather_lens_or_scale,
    scale: Optional[float] = None,
):
    if scale is None:
        scale = gather_lens_or_scale
        return torch.ops.sgl_kernel.kvlib_flash_index_decode(
            query_states, key_states, value_states, gather_idx, scale
        )
    gather_lens = gather_lens_or_scale
    return torch.ops.sgl_kernel.kvlib_flash_index_decode_varlen(
        query_states, key_states, value_states, gather_idx, gather_lens, scale
    )


def flash_index_decode_legacy(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    gather_idx: torch.Tensor,
    scale: float,
):
    return torch.ops.sgl_kernel.kvlib_flash_index_decode(
        query_states, key_states, value_states, gather_idx, scale
    )


def flash_mixed_decode(
    query_states: torch.Tensor,
    cached_keys: torch.Tensor,
    cached_values: torch.Tensor,
    top_index: torch.Tensor,
    buffer_keys: torch.Tensor,
    buffer_values: torch.Tensor,
    k_head_mask: torch.Tensor,
    k_head_index: torch.Tensor,
    real_seq_len,
    scale: float,
):
    if isinstance(real_seq_len, torch.Tensor):
        return torch.ops.sgl_kernel.kvlib_flash_mixed_decode_varlen(
            query_states,
            cached_keys,
            cached_values,
            top_index,
            buffer_keys,
            buffer_values,
            k_head_mask,
            k_head_index,
            real_seq_len,
            scale,
        )
    return torch.ops.sgl_kernel.kvlib_flash_mixed_decode(
        query_states,
        cached_keys,
        cached_values,
        top_index,
        buffer_keys,
        buffer_values,
        k_head_mask,
        k_head_index,
        real_seq_len,
        scale,
    )


def flash_decode(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scale: float,
    real_seq_len: int,
):
    return torch.ops.sgl_kernel.kvlib_flash_decode(
        query_states,
        key_states,
        value_states,
        scale,
        real_seq_len,
    )
