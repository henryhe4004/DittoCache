from typing import Tuple
import torch
import math

import triton
import triton.language as tl


configs = [
    triton.Config({"BLOCK_M": BM}, num_stages=s, num_warps=w)
    for BM in [16, 32, 64, 128] for s in ([1, 2, 4]) for w in [4, 8]
]


@triton.jit
def _quest_score_kernel(
    q_ptr,
    k_max_ptr,
    k_min_ptr,
    o_ptr,
    q_b_stride,
    q_h_stride,
    q_s_stride,
    k_b_stride,
    k_h_stride,
    k_s_stride,
    o_b_stride,
    o_h_stride,
    o_s_stride,
    SCALE,
    SEQ_LEN,
    FP16_OUTPUT: tl.constexpr,
    GQA_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_id = tl.program_id(0)
    hkv_id = tl.program_id(1)
    h_id = hkv_id * GQA_SIZE
    s_id = tl.program_id(2) * BLOCK_N

    q_offset = b_id * q_b_stride + h_id * q_h_stride
    k_offset = b_id * k_b_stride + hkv_id * k_h_stride
    o_offset = b_id * o_b_stride + hkv_id * o_h_stride

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(GQA_SIZE, HEAD_DIM),
        strides=(q_h_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_max_block_ptr = tl.make_block_ptr(
        base=k_max_ptr + k_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(1, k_s_stride),
        offsets=(0, s_id),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    k_min_block_ptr = tl.make_block_ptr(
        base=k_min_ptr + k_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(1, k_s_stride),
        offsets=(0, s_id),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + o_offset,
        shape=(1, SEQ_LEN),
        strides=(o_h_stride, o_s_stride),
        offsets=(0, s_id),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )

    q_data = tl.load(q_block_ptr, boundary_check=(1, 0),
                     padding_option="zero")  # (BLOCK_M, HEAD_DIM)
    q_mask = q_data >= 0

    k_max_data = tl.load(k_max_block_ptr,
                         boundary_check=(0, 1),
                         padding_option="zero")
    k_min_data = tl.load(k_min_block_ptr,
                         boundary_check=(0, 1),
                         padding_option="zero")
    acc = (tl.dot(q_data * q_mask, k_max_data) +
           tl.dot(q_data * ~q_mask, k_min_data)) / SCALE  # (BLOCK_M, BLOCK_N)
    acc = tl.sum(acc, 0, keep_dims=True)  # (1, BLOCK_N)

    if FP16_OUTPUT:
        acc = tl.cast(acc, tl.float16)
        tl.store(o_block_ptr, acc, boundary_check=(1, 0))
    else:
        acc = tl.cast(acc, tl.bfloat16)
        tl.store(o_block_ptr, acc, boundary_check=(1, 0))


@triton.jit
def _quest_score_head_mask_kernel(
    q_ptr,
    k_max_ptr,
    k_min_ptr,
    o_ptr,
    head_mask_ptr,
    q_b_stride,
    q_h_stride,
    q_s_stride,
    k_b_stride,
    k_h_stride,
    k_s_stride,
    o_b_stride,
    o_h_stride,
    o_s_stride,
    SCALE,
    SEQ_LEN,
    FP16_OUTPUT: tl.constexpr,
    GQA_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    NUM_KV_HEADS = NUM_HEADS // GQA_SIZE

    b_id = tl.program_id(0)
    hkv_id = tl.program_id(1)
    h_id = hkv_id * GQA_SIZE
    s_id = tl.program_id(2) * BLOCK_N

    head_mask = tl.load(head_mask_ptr + b_id * NUM_KV_HEADS + hkv_id)
    if head_mask:

        q_offset = b_id * q_b_stride + h_id * q_h_stride
        k_offset = b_id * k_b_stride + hkv_id * k_h_stride
        o_offset = b_id * o_b_stride + hkv_id * o_h_stride

        q_block_ptr = tl.make_block_ptr(
            base=q_ptr + q_offset,
            shape=(GQA_SIZE, HEAD_DIM),
            strides=(q_h_stride, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        k_max_block_ptr = tl.make_block_ptr(
            base=k_max_ptr + k_offset,
            shape=(HEAD_DIM, SEQ_LEN),
            strides=(1, k_s_stride),
            offsets=(0, s_id),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        k_min_block_ptr = tl.make_block_ptr(
            base=k_min_ptr + k_offset,
            shape=(HEAD_DIM, SEQ_LEN),
            strides=(1, k_s_stride),
            offsets=(0, s_id),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        o_block_ptr = tl.make_block_ptr(
            base=o_ptr + o_offset,
            shape=(1, SEQ_LEN),
            strides=(o_h_stride, o_s_stride),
            offsets=(0, s_id),
            block_shape=(1, BLOCK_N),
            order=(1, 0),
        )

        q_data = tl.load(q_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")  # (BLOCK_M, HEAD_DIM)
        q_mask = q_data >= 0

        k_max_data = tl.load(k_max_block_ptr,
                             boundary_check=(0, 1),
                             padding_option="zero")
        k_min_data = tl.load(k_min_block_ptr,
                             boundary_check=(0, 1),
                             padding_option="zero")
        acc = (tl.dot(q_data * q_mask, k_max_data) + tl.dot(
            q_data * ~q_mask, k_min_data)) / SCALE  # (BLOCK_M, BLOCK_N)
        acc = tl.sum(acc, 0, keep_dims=True)  # (1, BLOCK_N)

        if FP16_OUTPUT:
            acc = tl.cast(acc, tl.float16)
            tl.store(o_block_ptr, acc, boundary_check=(1, 0))
        else:
            acc = tl.cast(acc, tl.bfloat16)
            tl.store(o_block_ptr, acc, boundary_check=(1, 0))


def quest_score(query, key_max, key_min, block_num, head_mask=None):
    with torch.cuda.device(query.device):
        BSZ, _, NUM_KV_HEADS, HEAD_DIM = key_max.shape
        NUM_HEADS = query.shape[2]
        GQA_SIZE = NUM_HEADS // NUM_KV_HEADS

        extra_kern_args = {}

        BLOCK_M = triton.cdiv(GQA_SIZE, 16) * 16
        BLOCK_N = 128
        grid = lambda args: (
            BSZ,
            NUM_KV_HEADS,
            triton.cdiv(block_num, BLOCK_N),
        )
        SCALE = math.sqrt(HEAD_DIM)
        out = torch.zeros((BSZ, NUM_KV_HEADS, block_num),
                          device=query.device,
                          dtype=query.dtype)

        if head_mask is None:
            _quest_score_kernel[grid](
                query,
                key_max,
                key_min,
                out,
                query.stride(0),
                query.stride(2),
                query.stride(1),
                key_max.stride(0),
                key_max.stride(2),
                key_max.stride(1),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                SCALE,
                block_num,
                query.dtype == torch.float16,
                GQA_SIZE,
                NUM_HEADS,
                HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                **extra_kern_args,
            )
        else:
            _quest_score_head_mask_kernel[grid](
                query,
                key_max,
                key_min,
                out,
                head_mask,
                query.stride(0),
                query.stride(2),
                query.stride(1),
                key_max.stride(0),
                key_max.stride(2),
                key_max.stride(1),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                SCALE,
                block_num,
                query.dtype == torch.float16,
                GQA_SIZE,
                NUM_HEADS,
                HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                **extra_kern_args,
            )

        return out

