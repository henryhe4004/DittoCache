import torch
import triton
import triton.language as tl


@triton.jit
def _check_reuse_with_importance(
    curr_query,
    prev_query,
    gpu_head_mask,
    out_mask,
    threshold,
    importance,
    query_cache_valid,
    query_b_stride,
    query_h_stride,
    NUM_HEAD: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    qcache_valid = tl.load(query_cache_valid)
    batch_id = tl.program_id(0)
    head_kv_id = tl.program_id(1)
    query_offset = batch_id * query_b_stride + head_kv_id * KV_GROUP * query_h_stride
    curr_query_ptr = tl.make_block_ptr(
        base=curr_query + query_offset,
        shape=(KV_GROUP, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM),
        order=(1, 0),
    )

    prev_query_ptr = tl.make_block_ptr(
        base=prev_query + query_offset,
        shape=(KV_GROUP, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM),
        order=(1, 0),
    )

    if qcache_valid:
        out_mask_ptr = tl.make_block_ptr(
            base=out_mask + batch_id * NUM_KV_HEAD + head_kv_id,
            shape=(1, 1),
            strides=(1, 1),
            offsets=(0, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )

        threshold_ptr = tl.make_block_ptr(
            base=threshold + head_kv_id,
            shape=(1, 1),
            strides=(1, 1),
            offsets=(0, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )

        importance_ptr = tl.make_block_ptr(
            base=importance + head_kv_id * KV_GROUP,
            shape=(KV_GROUP, 1),
            strides=(1, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, 1),
            order=(1, 0),
        )

        gpu_head_mask_ptr = tl.make_block_ptr(
            base=gpu_head_mask + head_kv_id,
            shape=(1, 1),
            strides=(1, 1),
            offsets=(0, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )

        curr_q_data = tl.load(curr_query_ptr,
                            boundary_check=(1, 0),
                            padding_option="nan")
        prev_q_data = tl.load(prev_query_ptr,
                            boundary_check=(1, 0),
                            padding_option="nan")

        norm1 = tl.sqrt(tl.sum(tl.cast(curr_q_data * curr_q_data, tl.float32),
                            1)).reshape(BLOCK_SIZE_M, 1)
        norm2 = tl.sqrt(tl.sum(tl.cast(prev_q_data * prev_q_data, tl.float32),
                            1)).reshape(BLOCK_SIZE_M, 1)
        dot = curr_q_data * prev_q_data
        cos = tl.sum(dot / (norm1 * norm2 + 1e-8), 1)

        # weighted avg
        q_importance = tl.load(importance_ptr,
                            boundary_check=(1, 0),
                            padding_option="zero")
        mask = tl.arange(0, BLOCK_SIZE_M) >= KV_GROUP
        q_importance = q_importance.reshape(BLOCK_SIZE_M)
        result = q_importance / cos
        result = tl.where(mask, 0, result)
        result = tl.sum(result, axis=0, keep_dims=True)
        result = 1 / result
        result = result.reshape(1, 1)
        cos = tl.cast(result, tl.float16)

        threshold_val = tl.load(threshold_ptr,
                                boundary_check=(1, 0),
                                padding_option="zero")
        gpu_head_mask_val = tl.load(gpu_head_mask_ptr,
                                    boundary_check=(1, 0),
                                    padding_option="zero")
        gather_flag = (cos < threshold_val) & (gpu_head_mask_val == 0)
        tl.store(out_mask_ptr,
                tl.cast(gather_flag, tl.int8),
                boundary_check=(1, 0))

        gather_mask = tl.broadcast_to(gather_flag, (BLOCK_SIZE_M, 1))
        valid_mask = (tl.arange(0, BLOCK_SIZE_M) < KV_GROUP)[:, None]

        q_store_offset = (query_offset + tl.arange(0, BLOCK_SIZE_M) *
                        query_h_stride)[:, None] + tl.arange(0,
                                                            HEAD_DIM)[None, :]
        q_store_mask = gather_mask & valid_mask
        tl.store(prev_query + q_store_offset, curr_q_data, mask=q_store_mask)

    else:
        curr_q_data = tl.load(curr_query_ptr,
                            boundary_check=(1, 0))
        tl.store(prev_query_ptr, curr_q_data, boundary_check=(1, 0))


def check_reuse_with_importance(curr_query: torch.Tensor,
                                prev_query: torch.Tensor,
                                gpu_head_mask: torch.Tensor,
                                q_head_importance: torch.Tensor,
                                out_mask: torch.Tensor,
                                threshold: torch.Tensor,
                                query_cache_valid: torch.Tensor):
    B, _, H, D = curr_query.shape
    _, HKV = out_mask.shape
    G = H // HKV
    BLOCK_SIZE_M = triton.next_power_of_2(G)

    grid = lambda args: (
        B,
        HKV,
        1,
    )
    _check_reuse_with_importance[grid](
        curr_query,
        prev_query,
        gpu_head_mask,
        out_mask,
        threshold,
        q_head_importance,
        query_cache_valid,
        curr_query.stride(0),
        curr_query.stride(2),
        H,
        HKV,
        H // HKV,
        D,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )


@triton.jit
def _check_reuse_and_update_query_head_threshold_with_gpu_head(
    curr_query,
    prev_query,
    gpu_head_mask,
    out_mask,
    threshold,
    query_cache_valid,
    query_b_stride,
    query_h_stride,
    NUM_HEAD: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    qcache_valid = tl.load(query_cache_valid)
    batch_id = tl.program_id(0)
    head_kv_id = tl.program_id(1)
    query_offset = batch_id * query_b_stride + head_kv_id * KV_GROUP * query_h_stride
    curr_query_ptr = tl.make_block_ptr(
        base=curr_query + query_offset,
        shape=(KV_GROUP, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM),
        order=(1, 0),
    )
    prev_query_ptr = tl.make_block_ptr(
        base=prev_query + query_offset,
        shape=(KV_GROUP, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM),
        order=(1, 0),
    )

    if qcache_valid:
        out_mask_ptr = tl.make_block_ptr(
            base=out_mask + batch_id * NUM_KV_HEAD + head_kv_id,
            shape=(1, 1),
            strides=(1, 1),
            offsets=(0, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )

        threshold_ptr = tl.make_block_ptr(
            base=threshold + head_kv_id,
            shape=(1, 1),
            strides=(1, 1),
            offsets=(0, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )

        gpu_head_mask_ptr = tl.make_block_ptr(
            base=gpu_head_mask + head_kv_id,
            shape=(1, 1),
            strides=(1, 1),
            offsets=(0, 0),
            block_shape=(1, 1),
            order=(1, 0),
        )

        curr_q_data = tl.load(curr_query_ptr,
                            boundary_check=(1, 0),
                            padding_option="nan")
        prev_q_data = tl.load(prev_query_ptr,
                            boundary_check=(1, 0),
                            padding_option="nan")

        norm1 = tl.sqrt(tl.sum(tl.cast(curr_q_data * curr_q_data, tl.float32),
                            1)).reshape(BLOCK_SIZE_M, 1)
        norm2 = tl.sqrt(tl.sum(tl.cast(prev_q_data * prev_q_data, tl.float32),
                            1)).reshape(BLOCK_SIZE_M, 1)
        dot = curr_q_data * prev_q_data
        cos = tl.sum(dot / (norm1 * norm2 + 1e-8), 1)

        mask = tl.arange(0, BLOCK_SIZE_M) >= KV_GROUP

        cos = tl.where(mask, 1, cos).reshape(BLOCK_SIZE_M, 1)
        cos = tl.min(cos, axis=0, keep_dims=True)

        cos = tl.cast(cos, tl.float16)

        threshold_val = tl.load(threshold_ptr,
                                boundary_check=(1, 0),
                                padding_option="zero")
        gpu_head_mask_val = tl.load(gpu_head_mask_ptr,
                                    boundary_check=(1, 0),
                                    padding_option="zero")
        gather_flag = (cos < threshold_val) & (gpu_head_mask_val == 0)
        tl.store(out_mask_ptr,
                tl.cast(gather_flag, tl.int8),
                boundary_check=(1, 0))

        gather_mask = tl.broadcast_to(gather_flag, (BLOCK_SIZE_M, 1))
        valid_mask = (tl.arange(0, BLOCK_SIZE_M) < KV_GROUP)[:, None]

        q_store_offset = (query_offset + tl.arange(0, BLOCK_SIZE_M) *
                        query_h_stride)[:, None] + tl.arange(0,
                                                            HEAD_DIM)[None, :]
        q_store_mask = gather_mask & valid_mask
        tl.store(prev_query + q_store_offset, curr_q_data, mask=q_store_mask)

    else:
        curr_q_data = tl.load(curr_query_ptr,
                            boundary_check=(1, 0))
        tl.store(prev_query_ptr, curr_q_data, boundary_check=(1, 0))


def check_reuse_head_threshold_with_gpu_head(curr_query: torch.Tensor,
                                             prev_query: torch.Tensor,
                                             gpu_head_mask: torch.Tensor,
                                             out_mask: torch.Tensor,
                                             threshold: torch.Tensor,
                                             query_cache_valid: torch.Tensor):
    B, _, H, D = curr_query.shape
    _, HKV = out_mask.shape
    G = H // HKV
    BLOCK_SIZE_M = triton.next_power_of_2(G)

    grid = lambda args: (
        B,
        HKV,
        1,
    )
    _check_reuse_and_update_query_head_threshold_with_gpu_head[grid](
        curr_query,
        prev_query,
        gpu_head_mask,
        out_mask,
        threshold,
        query_cache_valid,
        curr_query.stride(0),
        curr_query.stride(2),
        H,
        HKV,
        H // HKV,
        D,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
