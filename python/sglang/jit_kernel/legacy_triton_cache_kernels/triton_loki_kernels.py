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

