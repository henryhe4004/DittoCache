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
def _prefill_loki_encode(
    data_ptr,
    data_stride0,
    pca_weights_ptr,
    output_code_output_ptr,
    output_code_stride0,
    SEQ,
    BSZ,
    NUM_HEAD: tl.constexpr,
    NUM_CHANNELS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_N: tl.constexpr = 32

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    DataPtr = data_ptr + batch_id * data_stride0 + head_id * HEAD_DIM
    OutputCodePtr = output_code_output_ptr + batch_id * output_code_stride0 + head_id * NUM_CHANNELS
    PCAWeightPtr = pca_weights_ptr + head_id * HEAD_DIM * HEAD_DIM

    Data_block_ptr = tl.make_block_ptr(
        base=DataPtr,
        shape=(SEQ, HEAD_DIM),
        strides=(NUM_HEAD * HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    OutputCode_block_ptr = tl.make_block_ptr(
        base=OutputCodePtr,
        shape=(SEQ, NUM_CHANNELS),
        strides=(NUM_HEAD * NUM_CHANNELS, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    PCAWeights_ptr = tl.make_block_ptr(
        base=PCAWeightPtr,
        shape=(HEAD_DIM, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )

    data = tl.load(Data_block_ptr,
                   boundary_check=(1, 0),
                   padding_option="zero")  # [BLOCK_M, HEAD_DIM]

    for start_n in range(0, NUM_CHANNELS, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        weights = tl.load(PCAWeights_ptr)  # [HEAD_DIM, BLOCK_N]
        acc = tl.dot(data, weights)  # [BLOCK_M, BLOCK_N]
        acc = tl.cast(acc, data.dtype)
        tl.store(OutputCode_block_ptr, acc, boundary_check=(1, 0))

        PCAWeights_ptr = tl.advance(PCAWeights_ptr, (0, BLOCK_N))
        OutputCode_block_ptr = tl.advance(OutputCode_block_ptr, (0, BLOCK_N))


def prefill_loki_encode(data: torch.Tensor, pca_weights: torch.Tensor,
                        data_code_output: torch.Tensor,
                        num_channels: int) -> None:
    """
    data: [bsz, seq, num_head, head_dim]
    pca_weights: [num_head, head_dim, head_dim]
    data_code_output: [bsz, xx, num_head, num_channels] (xx > seq)
    num_channels: <= head_dim
    """

    with torch.cuda.device(data.device):
        BSZ, SEQ, NUM_HEAD, HEAD_DIM = data.shape
        assert HEAD_DIM in {16, 32, 64, 128, 256}
        BLOCK_M = 128

        grid = lambda args: (
            triton.cdiv(SEQ, BLOCK_M),
            BSZ,
            NUM_HEAD,
        )
        _prefill_loki_encode[grid](
            data,
            data.stride(0),
            pca_weights,
            data_code_output,
            data_code_output.stride(0),
            SEQ,
            BSZ,
            NUM_HEAD,
            num_channels,
            HEAD_DIM,
            BLOCK_M=BLOCK_M,
            num_stages=2,
            num_warps=4,
        )


@triton.jit
def _decode_loki_encode_k(
    key_data_ptr,
    key_data_stride0,
    pca_weight_ptr,
    key_code_output_ptr,
    key_code_output_stride0,
    CUR_SEQ,
    BSZ,
    KV_HEAD,
    NUM_CHANNELS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_id = tl.program_id(1) // KV_HEAD
    head_id = tl.program_id(1) % KV_HEAD
    start_n = tl.program_id(2)

    Weight_ptr = tl.make_block_ptr(
        base=pca_weight_ptr + head_id * HEAD_DIM * HEAD_DIM,
        shape=(HEAD_DIM, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )
    weight = tl.load(Weight_ptr)  # [HEAD_DIM, BLOCK_N]

    K_data_block_ptr = tl.make_block_ptr(
        base=key_data_ptr + batch_id * key_data_stride0 + head_id * HEAD_DIM,
        shape=(1, HEAD_DIM),
        strides=(KV_HEAD * HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    k_data = tl.load(K_data_block_ptr,
                     boundary_check=(1, 0),
                     padding_option="zero")  # [BLOCK_M, HEAD_DIM]

    k_acc = tl.dot(k_data, weight)  # [BLOCK_M, BLOCK_N]
    k_acc = tl.cast(k_acc, k_data.dtype)

    K_output_block_ptr = tl.make_block_ptr(
        base=key_code_output_ptr + batch_id * key_code_output_stride0 +
        CUR_SEQ * KV_HEAD * NUM_CHANNELS + head_id * NUM_CHANNELS,
        shape=(1, NUM_CHANNELS),
        strides=(KV_HEAD * NUM_CHANNELS, 1),
        offsets=(start_m * BLOCK_M, start_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(K_output_block_ptr, k_acc, boundary_check=(1, 0))


def decode_loki_encode_k(key_data: torch.Tensor, key_code_output: torch.Tensor,
                         pca_weights: torch.Tensor, num_channels: int,
                         cur_seq: int):

    with torch.cuda.device(key_data.device):
        BSZ, SEQ, NUM_KV_HEAD, HEAD_DIM = key_data.shape
        BLOCK_M = 16
        BLOCK_N = 32

        grid = lambda args: (
            triton.cdiv(SEQ, BLOCK_M),
            BSZ * NUM_KV_HEAD,
            triton.cdiv(num_channels, BLOCK_N),
        )

        _decode_loki_encode_k[grid](
            key_data,
            key_data.stride(0),
            pca_weights,
            key_code_output,
            key_code_output.stride(0),
            cur_seq,
            BSZ,
            NUM_KV_HEAD,
            num_channels,
            HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=1,
        )


@triton.jit
def _decode_loki_encode_qk(
    key_data_ptr,
    key_data_stride0,
    query_data_ptr,
    query_data_stride0,
    key_pca_weight_ptr,
    query_pca_weight_ptr,
    key_code_output_ptr,
    key_code_output_stride0,
    query_code_output_ptr,
    query_code_output_stride0,
    CUR_SEQ,
    BSZ,
    KV_HEAD,
    Q_HEAD,
    NUM_CHANNELS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    KV_GROUP = Q_HEAD // KV_HEAD
    SLICE = tl.cdiv(KV_GROUP, BLOCK_M)

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1) // KV_HEAD
    head_id = tl.program_id(1) % KV_HEAD
    start_n = tl.program_id(2)

    if start_m < SLICE:
        QWeight_ptr = tl.make_block_ptr(
            base=query_pca_weight_ptr + head_id * HEAD_DIM * HEAD_DIM,
            shape=(HEAD_DIM, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(0, start_n * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(1, 0),
        )
        q_weight = tl.load(QWeight_ptr)

        Q_data_block_ptr = tl.make_block_ptr(
            base=query_data_ptr + batch_id * query_data_stride0 +
            head_id * KV_GROUP * HEAD_DIM,
            shape=(KV_GROUP, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        q_data = tl.load(Q_data_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")

        q_acc = tl.dot(q_data, q_weight)
        q_acc = tl.cast(q_acc, q_data.dtype)

        Q_output_block_ptr = tl.make_block_ptr(
            base=query_code_output_ptr + batch_id * query_code_output_stride0 +
            head_id * KV_GROUP * NUM_CHANNELS,
            shape=(KV_GROUP, NUM_CHANNELS),
            strides=(NUM_CHANNELS, 1),
            offsets=(start_m * BLOCK_M, start_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(Q_output_block_ptr, q_acc, boundary_check=(1, 0))

    else:
        start_m = start_m - SLICE

        K_weight_ptr = tl.make_block_ptr(
            base=key_pca_weight_ptr + head_id * HEAD_DIM * HEAD_DIM,
            shape=(HEAD_DIM, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(0, start_n * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(1, 0),
        )
        k_weight = tl.load(K_weight_ptr)

        K_data_block_ptr = tl.make_block_ptr(
            base=key_data_ptr + batch_id * key_data_stride0 +
            head_id * HEAD_DIM,
            shape=(1, HEAD_DIM),
            strides=(KV_HEAD * HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        k_data = tl.load(K_data_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")

        k_acc = tl.dot(k_data, k_weight)
        k_acc = tl.cast(k_acc, k_data.dtype)

        K_output_block_ptr = tl.make_block_ptr(
            base=key_code_output_ptr + batch_id * key_code_output_stride0 +
            CUR_SEQ * KV_HEAD * NUM_CHANNELS + head_id * NUM_CHANNELS,
            shape=(1, NUM_CHANNELS),
            strides=(KV_HEAD * NUM_CHANNELS, 1),
            offsets=(start_m * BLOCK_M, start_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(K_output_block_ptr, k_acc, boundary_check=(1, 0))


def decode_loki_encode_qk(
        key_data: torch.Tensor, key_code_output: torch.Tensor,
        key_pca_weights: torch.Tensor, query_data: torch.Tensor,
        query_code_output: torch.Tensor, query_pca_weights: torch.Tensor,
        num_channels: int, cur_seq: int):

    with torch.cuda.device(key_data.device):
        BSZ, SEQ, NUM_KV_HEAD, HEAD_DIM = key_data.shape
        NUM_HEAD = query_data.shape[2]
        BLOCK_M = 16
        BLOCK_N = 32
        KV_GROUP = NUM_HEAD // NUM_KV_HEAD

        grid = lambda args: (
            triton.cdiv(KV_GROUP * SEQ, BLOCK_M) + triton.cdiv(SEQ, BLOCK_M),
            BSZ * NUM_KV_HEAD,
            triton.cdiv(num_channels, BLOCK_N),
        )

        _decode_loki_encode_qk[grid](
            key_data,
            key_data.stride(0),
            query_data,
            query_data.stride(0),
            key_pca_weights,
            query_pca_weights,
            key_code_output,
            key_code_output.stride(0),
            query_code_output,
            query_code_output.stride(0),
            cur_seq,
            BSZ,
            NUM_KV_HEAD,
            NUM_HEAD,
            num_channels,
            HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=1,
        )


@triton.jit
def _decode_loki_encode_qqk(
    key_data_ptr,
    key_data_stride0,
    query_data_ptr,
    query_data_stride0,
    query2_data_ptr,
    query2_data_stride0,
    key_pca_weight_ptr,
    query_pca_weight_ptr,
    query2_pca_weight_ptr,
    key_code_output_ptr,
    key_code_output_stride0,
    query_code_output_ptr,
    query_code_output_stride0,
    query2_code_output_ptr,
    query2_code_output_stride0,
    CUR_SEQ,
    BSZ,
    KV_HEAD,
    Q_HEAD,
    NUM_CHANNELS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    KV_GROUP = Q_HEAD // KV_HEAD

    SLICE = tl.cdiv(KV_GROUP, BLOCK_M)
    SLICE2 = SLICE + tl.cdiv(KV_GROUP, BLOCK_M)

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1) // KV_HEAD
    head_id = tl.program_id(1) % KV_HEAD
    start_n = tl.program_id(2)

    if start_m < SLICE:
        QWeight_ptr = tl.make_block_ptr(
            base=query_pca_weight_ptr + head_id * HEAD_DIM * HEAD_DIM,
            shape=(HEAD_DIM, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(0, start_n * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(1, 0),
        )
        q_weight = tl.load(QWeight_ptr)

        Q_data_block_ptr = tl.make_block_ptr(
            base=query_data_ptr + batch_id * query_data_stride0 +
            head_id * KV_GROUP * HEAD_DIM,
            shape=(KV_GROUP, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        q_data = tl.load(Q_data_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")

        q_acc = tl.dot(q_data, q_weight)
        q_acc = tl.cast(q_acc, q_data.dtype)

        Q_output_block_ptr = tl.make_block_ptr(
            base=query_code_output_ptr + batch_id * query_code_output_stride0 +
            head_id * KV_GROUP * NUM_CHANNELS,
            shape=(KV_GROUP, NUM_CHANNELS),
            strides=(NUM_CHANNELS, 1),
            offsets=(start_m * BLOCK_M, start_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(Q_output_block_ptr, q_acc, boundary_check=(1, 0))

    elif start_m < SLICE2:
        start_m = start_m - SLICE
        QWeight_ptr2 = tl.make_block_ptr(
            base=query2_pca_weight_ptr + head_id * HEAD_DIM * HEAD_DIM,
            shape=(HEAD_DIM, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(0, start_n * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(1, 0),
        )
        q_weight2 = tl.load(QWeight_ptr2)

        Q_data_block_ptr2 = tl.make_block_ptr(
            base=query2_data_ptr + batch_id * query2_data_stride0 +
            head_id * KV_GROUP * HEAD_DIM,
            shape=(KV_GROUP, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        q_data2 = tl.load(Q_data_block_ptr2,
                          boundary_check=(1, 0),
                          padding_option="zero")

        q_acc2 = tl.dot(q_data2, q_weight2)
        q_acc2 = tl.cast(q_acc2, q_data2.dtype)

        Q_output_block_ptr2 = tl.make_block_ptr(
            base=query2_code_output_ptr +
            batch_id * query2_code_output_stride0 +
            head_id * KV_GROUP * NUM_CHANNELS,
            shape=(KV_GROUP, NUM_CHANNELS),
            strides=(NUM_CHANNELS, 1),
            offsets=(start_m * BLOCK_M, start_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(Q_output_block_ptr2, q_acc2, boundary_check=(1, 0))

    if start_m >= SLICE2:
        start_m = start_m - SLICE2

        K_weight_ptr = tl.make_block_ptr(
            base=key_pca_weight_ptr + head_id * HEAD_DIM * HEAD_DIM,
            shape=(HEAD_DIM, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(0, start_n * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(1, 0),
        )
        k_weight = tl.load(K_weight_ptr)

        K_data_block_ptr = tl.make_block_ptr(
            base=key_data_ptr + batch_id * key_data_stride0 +
            head_id * HEAD_DIM,
            shape=(1, HEAD_DIM),
            strides=(KV_HEAD * HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        k_data = tl.load(K_data_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")

        k_acc = tl.dot(k_data, k_weight)
        k_acc = tl.cast(k_acc, k_data.dtype)

        K_output_block_ptr = tl.make_block_ptr(
            base=key_code_output_ptr + batch_id * key_code_output_stride0 +
            CUR_SEQ * KV_HEAD * NUM_CHANNELS + head_id * NUM_CHANNELS,
            shape=(1, NUM_CHANNELS),
            strides=(KV_HEAD * NUM_CHANNELS, 1),
            offsets=(start_m * BLOCK_M, start_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(K_output_block_ptr, k_acc, boundary_check=(1, 0))


def decode_loki_encode_qqk(
        key_data: torch.Tensor, key_code_output: torch.Tensor,
        key_pca_weights: torch.Tensor, query_data: torch.Tensor,
        query_code_output: torch.Tensor, query_pca_weights: torch.Tensor,
        query_data2: torch.Tensor, query_code_output2: torch.Tensor,
        query_pca_weights2: torch.Tensor, num_channels: int, cur_seq: int):

    with torch.cuda.device(key_data.device):
        BSZ, SEQ, NUM_KV_HEAD, HEAD_DIM = key_data.shape
        NUM_HEAD = query_data.shape[2]
        BLOCK_M = 16
        BLOCK_N = 32
        KV_GROUP = NUM_HEAD // NUM_KV_HEAD

        grid = lambda args: (
            triton.cdiv(KV_GROUP * SEQ, BLOCK_M) + triton.cdiv(
                KV_GROUP * SEQ, BLOCK_M) + triton.cdiv(SEQ, BLOCK_M),
            BSZ * NUM_KV_HEAD,
            triton.cdiv(num_channels, BLOCK_N),
        )

        _decode_loki_encode_qqk[grid](
            key_data,
            key_data.stride(0),
            query_data,
            query_data.stride(0),
            query_data2,
            query_data2.stride(0),
            key_pca_weights,
            query_pca_weights,
            query_pca_weights2,
            key_code_output,
            key_code_output.stride(0),
            query_code_output,
            query_code_output.stride(0),
            query_code_output2,
            query_code_output2.stride(0),
            cur_seq,
            BSZ,
            NUM_KV_HEAD,
            NUM_HEAD,
            num_channels,
            HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=1,
        )


@triton.jit
def _loki_score_kernel(
    q_ptr,
    k_ptr,
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
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_offset,
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

    k_data = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
    acc = tl.dot(q_data, k_data) / SCALE  # (BLOCK_M, BLOCK_N)
    acc = tl.sum(acc, 0, keep_dims=True)

    if FP16_OUTPUT:
        acc = tl.cast(acc, tl.float16)
        tl.store(o_block_ptr, acc, boundary_check=(1, 0))
    else:
        acc = tl.cast(acc, tl.bfloat16)
        tl.store(o_block_ptr, acc, boundary_check=(1, 0))


@triton.jit
def _loki_score_head_mask_kernel(
    q_ptr,
    k_ptr,
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
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + k_offset,
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

        k_data = tl.load(k_block_ptr,
                         boundary_check=(0, 1),
                         padding_option="zero")
        acc = tl.dot(q_data, k_data) / SCALE  # (BLOCK_M, BLOCK_N)
        acc = tl.sum(acc, 0, keep_dims=True)

        if FP16_OUTPUT:
            acc = tl.cast(acc, tl.float16)
            tl.store(o_block_ptr, acc, boundary_check=(1, 0))
        else:
            acc = tl.cast(acc, tl.bfloat16)
            tl.store(o_block_ptr, acc, boundary_check=(1, 0))


def loki_score(query, key, seq_len, head_mask=None):
    with torch.cuda.device(query.device):
        BSZ, _, NUM_KV_HEADS, HEAD_DIM = key.shape
        NUM_HEADS = query.shape[2]
        GQA_SIZE = NUM_HEADS // NUM_KV_HEADS

        extra_kern_args = {}

        # PASS 1: compute acc, max, sumexp
        BLOCK_M = triton.cdiv(GQA_SIZE, 16) * 16
        BLOCK_N = 512
        grid = lambda args: (
            BSZ,
            NUM_KV_HEADS,
            triton.cdiv(seq_len, BLOCK_N),
        )
        SCALE = math.sqrt(HEAD_DIM)
        out = torch.zeros((BSZ, NUM_KV_HEADS, seq_len),
                          device=query.device,
                          dtype=query.dtype)

        if head_mask is None:
            _loki_score_kernel[grid](
                query,
                key,
                out,
                query.stride(0),
                query.stride(2),
                query.stride(1),
                key.stride(0),
                key.stride(2),
                key.stride(1),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                SCALE,
                seq_len,
                query.dtype == torch.float16,
                GQA_SIZE,
                NUM_HEADS,
                HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                **extra_kern_args,
            )
        else:
            _loki_score_head_mask_kernel[grid](
                query,
                key,
                out,
                head_mask,
                query.stride(0),
                query.stride(2),
                query.stride(1),
                key.stride(0),
                key.stride(2),
                key.stride(1),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                SCALE,
                seq_len,
                query.dtype == torch.float16,
                GQA_SIZE,
                NUM_HEADS,
                HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                **extra_kern_args,
            )

        return out

