from typing import Tuple
import torch
import math

import triton
import triton.language as tl


PI = math.pi


@triton.jit
def acos(x):
    negate = x < 0
    negate = tl.cast(negate, tl.float16)
    x = tl.abs(x)
    ret = -0.0187293
    ret = ret * x
    ret = ret + 0.0742610
    ret = ret * x
    ret = ret - 0.2121144
    ret = ret * x
    ret = ret + 1.5707288
    ret = ret * tl.sqrt(1.0 - x)
    ret = ret - 2 * negate * ret
    return negate * 3.14159265358979 + ret


@triton.jit
def _check_reuse_and_update_query(
    curr_query,
    prev_query,
    mask,
    threshold,
    query_b_stride,
    query_h_stride,
    NUM_HEAD: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
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

    mask_ptr = tl.make_block_ptr(
        base=mask + batch_id * NUM_KV_HEAD + head_kv_id,
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

    mask_vec = tl.arange(0, BLOCK_SIZE_M) >= KV_GROUP
    cos = tl.where(mask_vec, 1, cos).reshape(BLOCK_SIZE_M, 1)
    cos = tl.min(cos, axis=0, keep_dims=True)
    cos = tl.cast(cos, tl.float16)

    gather_flag = cos < threshold
    tl.store(mask_ptr, tl.cast(gather_flag, tl.int8), boundary_check=(1, 0))

    gather_mask = tl.broadcast_to(gather_flag, (BLOCK_SIZE_M, 1))
    valid_mask = (tl.arange(0, BLOCK_SIZE_M) < KV_GROUP)[:, None]

    q_store_offset = (query_offset + tl.arange(0, BLOCK_SIZE_M) *
                      query_h_stride)[:, None] + tl.arange(0,
                                                           HEAD_DIM)[None, :]
    q_store_mask = gather_mask & valid_mask
    tl.store(prev_query + q_store_offset, curr_q_data, mask=q_store_mask)


def check_reuse(curr_query: torch.Tensor, prev_query: torch.Tensor,
                mask: torch.Tensor, threshold: int):
    with torch.cuda.device(curr_query.device):
        B, _, H, D = curr_query.shape
        _, HKV = mask.shape
        G = H // HKV
        BLOCK_SIZE_M = 2**math.ceil(math.log2(G))

        grid = lambda args: (
            B,
            HKV,
            1,
        )
        _check_reuse_and_update_query[grid](
            curr_query,
            prev_query,
            mask,
            threshold,
            curr_query.stride(0),
            curr_query.stride(2),
            H,
            HKV,
            H // HKV,
            D,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
        )


@triton.jit
def _check_reuse_and_update_query_head_threshold(
    curr_query,
    prev_query,
    mask,
    threshold,
    query_b_stride,
    query_h_stride,
    NUM_HEAD: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_kv_id = tl.program_id(1)

    # mask_offset = batch_id * NUM_KV_HEAD + head_kv_id

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

    mask_ptr = tl.make_block_ptr(
        base=mask + batch_id * NUM_KV_HEAD + head_kv_id,
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
    gather_flag = cos < threshold_val
    tl.store(mask_ptr, tl.cast(gather_flag, tl.int8), boundary_check=(1, 0))

    gather_mask = tl.broadcast_to(gather_flag, (BLOCK_SIZE_M, 1))
    valid_mask = (tl.arange(0, BLOCK_SIZE_M) < KV_GROUP)[:, None]

    q_store_offset = (query_offset + tl.arange(0, BLOCK_SIZE_M) *
                      query_h_stride)[:, None] + tl.arange(0,
                                                           HEAD_DIM)[None, :]
    q_store_mask = gather_mask & valid_mask
    tl.store(prev_query + q_store_offset, curr_q_data, mask=q_store_mask)


def check_reuse_head_threshold(curr_query: torch.Tensor,
                               prev_query: torch.Tensor, mask: torch.Tensor,
                               threshold: torch.Tensor):
    with torch.cuda.device(curr_query.device):
        B, _, H, D = curr_query.shape
        _, HKV = mask.shape
        G = H // HKV
        BLOCK_SIZE_M = 2**math.ceil(math.log2(G))

        grid = lambda args: (
            B,
            HKV,
            1,
        )
        _check_reuse_and_update_query_head_threshold[grid](
            curr_query,
            prev_query,
            mask,
            threshold,
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
    query_b_stride,
    query_h_stride,
    NUM_HEAD: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_kv_id = tl.program_id(1)

    # mask_offset = batch_id * NUM_KV_HEAD + head_kv_id

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

    # min
    cos = tl.where(mask, 1, cos).reshape(BLOCK_SIZE_M, 1)
    cos = tl.min(cos, axis=0, keep_dims=True)

    # # avg
    # cos = tl.where(mask, 0, cos).reshape(BLOCK_SIZE_M, 1)
    # cos = tl.sum(cos, axis=0, keep_dims=True) / KV_GROUP

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


def check_reuse_head_threshold_with_gpu_head(curr_query: torch.Tensor,
                                             prev_query: torch.Tensor,
                                             gpu_head_mask: torch.Tensor,
                                             out_mask: torch.Tensor,
                                             threshold: torch.Tensor):
    with torch.cuda.device(curr_query.device):
        B, _, H, D = curr_query.shape
        _, HKV = out_mask.shape
        G = H // HKV
        BLOCK_SIZE_M = 2**math.ceil(math.log2(G))

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
            curr_query.stride(0),
            curr_query.stride(2),
            H,
            HKV,
            H // HKV,
            D,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
        )


@triton.jit
def _check_reuse_with_importance(
    curr_query,
    prev_query,
    gpu_head_mask,
    out_mask,
    threshold,
    importance,
    query_b_stride,
    query_h_stride,
    NUM_HEAD: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_kv_id = tl.program_id(1)

    # mask_offset = batch_id * NUM_KV_HEAD + head_kv_id

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

    # q_importance = tl.where(mask, )
    # cos = tl.where(mask, 0, cos).reshape(BLOCK_SIZE_M, 1)
    # tl.static_print("shape", q_importance.shape)
    # cos = tl.sum(cos * q_importance, axis=0, keep_dims=True)

    # cos = tl.cast(cos, tl.float16)

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


def check_reuse_with_importance(curr_query: torch.Tensor,
                                prev_query: torch.Tensor,
                                gpu_head_mask: torch.Tensor,
                                q_head_importance: torch.Tensor,
                                out_mask: torch.Tensor,
                                threshold: torch.Tensor):
    with torch.cuda.device(curr_query.device):
        B, _, H, D = curr_query.shape
        _, HKV = out_mask.shape
        G = H // HKV
        BLOCK_SIZE_M = 2**math.ceil(math.log2(G))

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
            curr_query.stride(0),
            curr_query.stride(2),
            H,
            HKV,
            H // HKV,
            D,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
        )


@triton.jit
def _check_reuse_with_importance2(
    curr_query,
    prev_query,
    gpu_head_mask,
    out_mask,
    threshold,
    importance,
    query_b_stride,
    query_h_stride,
    NUM_HEAD: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_kv_id = tl.program_id(1)

    # mask_offset = batch_id * NUM_KV_HEAD + head_kv_id

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

    q_importance = tl.load(importance_ptr,
                           boundary_check=(1, 0),
                           padding_option="zero")
    q_importance = tl.clamp(1.2 * q_importance, 0.8, 1.0)
    mask = (tl.arange(0, BLOCK_SIZE_M) >= KV_GROUP).reshape(BLOCK_SIZE_M, 1)
    arccos = acos(cos).reshape(BLOCK_SIZE_M, 1)
    q_importance = tl.where(mask, 0, q_importance)
    arccos = arccos * q_importance
    cos = tl.cos(arccos)
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


def check_reuse_with_importance2(curr_query: torch.Tensor,
                                 prev_query: torch.Tensor,
                                 gpu_head_mask: torch.Tensor,
                                 q_head_importance: torch.Tensor,
                                 out_mask: torch.Tensor,
                                 threshold: torch.Tensor):
    with torch.cuda.device(curr_query.device):
        B, _, H, D = curr_query.shape
        _, HKV = out_mask.shape
        G = H // HKV
        BLOCK_SIZE_M = 2**math.ceil(math.log2(G))

        grid = lambda args: (
            B,
            HKV,
            1,
        )
        _check_reuse_with_importance2[grid](
            curr_query,
            prev_query,
            gpu_head_mask,
            out_mask,
            threshold,
            q_head_importance,
            curr_query.stride(0),
            curr_query.stride(2),
            H,
            HKV,
            H // HKV,
            D,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
        )


@triton.jit
def _check_reuse_qhead_threshold(
    curr_query,
    prev_query,
    gpu_head_mask,
    out_mask,
    threshold,
    query_b_stride,
    query_h_stride,
    NUM_HEAD: tl.constexpr,
    NUM_KV_HEAD: tl.constexpr,
    KV_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_kv_id = tl.program_id(1)

    # mask_offset = batch_id * NUM_KV_HEAD + head_kv_id

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

    out_mask_ptr = tl.make_block_ptr(
        base=out_mask + batch_id * NUM_KV_HEAD + head_kv_id,
        shape=(1, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(1, 1),
        order=(1, 0),
    )

    threshold_ptr = tl.make_block_ptr(
        base=threshold + head_kv_id * KV_GROUP,
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

    mask = tl.arange(0, BLOCK_SIZE_M) >= KV_GROUP
    mask = mask.reshape(BLOCK_SIZE_M, 1)
    cos = tl.cast(cos, tl.float16).reshape(BLOCK_SIZE_M, 1)
    threshold_val = tl.load(threshold_ptr,
                            boundary_check=(1, 0),
                            padding_option="zero")
    q_gather_flag = cos < threshold_val
    q_gather_cnts = tl.cast(q_gather_flag, tl.float16)
    q_gather_cnts = tl.where(q_gather_flag, mask, 0)
    q_gather_cnts = tl.sum(q_gather_flag, axis=0, keep_dims=True)
    gpu_head_mask_val = tl.load(gpu_head_mask_ptr,
                                boundary_check=(1, 0),
                                padding_option="zero")
    gather_flag = (q_gather_cnts > 0) & (gpu_head_mask_val == 0)
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


def check_reuse_qhead_threshold(curr_query: torch.Tensor,
                                prev_query: torch.Tensor,
                                gpu_head_mask: torch.Tensor,
                                out_mask: torch.Tensor,
                                threshold: torch.Tensor):
    with torch.cuda.device(curr_query.device):
        B, _, H, D = curr_query.shape
        _, HKV = out_mask.shape
        G = H // HKV
        BLOCK_SIZE_M = 2**math.ceil(math.log2(G))

        grid = lambda args: (
            B,
            HKV,
            1,
        )
        _check_reuse_qhead_threshold[grid](
            curr_query,
            prev_query,
            gpu_head_mask,
            out_mask,
            threshold,
            curr_query.stride(0),
            curr_query.stride(2),
            H,
            HKV,
            H // HKV,
            D,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
        )

