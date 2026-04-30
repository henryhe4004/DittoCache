import triton
import triton.language as tl


@triton.jit
def _hash_encode_append_prefill(
    data_ptr,
    data_stride0,
    output_code_output_ptr,
    output_code_stride0,
    output_code_stride1,
    hash_weights_ptr,
    seq_len_ptr,
    packbit_tensor_ptr,
    SEQ,
    BSZ,
    NUM_HEAD: tl.constexpr,
    RBIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    CHUNK_SIZE: tl.constexpr = 32
    BLOCK_N: tl.constexpr = 32
    NUM_CHUNK: tl.constexpr = RBIT // 32

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    cur_batch_seq_len = tl.load(seq_len_ptr)

    DataPtr = data_ptr + batch_id * data_stride0 + head_id * HEAD_DIM
    OutputCodePtr = (
        output_code_output_ptr + batch_id * output_code_stride0 +
        cur_batch_seq_len * output_code_stride1 +
        head_id * NUM_CHUNK
    )
    HashWeightPtr = head_id * (HEAD_DIM * RBIT)

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
        shape=(SEQ, NUM_CHUNK),
        strides=(output_code_stride1, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    HashWeights_ptr = tl.make_block_ptr(
        base=hash_weights_ptr + HashWeightPtr,
        shape=(HEAD_DIM, RBIT),
        strides=(RBIT, 1),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )

    # load K
    data = tl.load(Data_block_ptr,
                   boundary_check=(1, 0),
                   padding_option="zero")  # [BLOCK_M, HEAD_DIM]

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    for start_n in range(0, RBIT, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        weights = tl.load(HashWeights_ptr)  # [HEAD_DIM, BLOCK_N]
        acc = tl.dot(data, weights) > 0  # [BLOCK_M, BLOCK_N]
        acc = acc.to(packbit_tensor.type.element_ty)
        acc = acc * packbit_tensor
        acc = tl.sum(acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(OutputCode_block_ptr, acc, boundary_check=(1, 0))

        # move on
        HashWeights_ptr = tl.advance(HashWeights_ptr, (0, BLOCK_N))
        OutputCode_block_ptr = tl.advance(OutputCode_block_ptr, (0, 1))


def hash_encode_append_prefill(
    key_data,               # [b, s, h, d]
    code_cache,             # [b, smax, h, num_chunk]
    hash_weights,           # [h, d, rbit]
    seq_len_tensor,         # [1]
    packbit_aux_tensor,     # [32]
) -> None:
    RBIT = hash_weights.shape[-1]
    assert RBIT % 32 == 0

    BSZ, SEQ, NUM_HEAD, HEAD_DIM = key_data.shape
    assert HEAD_DIM in {16, 32, 64, 128, 256}

    BLOCK_M = 128

    grid = lambda args: (
        triton.cdiv(SEQ, BLOCK_M),
        BSZ,
        NUM_HEAD,
    )
    _hash_encode_append_prefill[grid](
        key_data,
        key_data.stride(0),
        code_cache,
        code_cache.stride(0),
        code_cache.stride(1),
        hash_weights,
        seq_len_tensor,
        packbit_aux_tensor,
        SEQ,
        BSZ,
        NUM_HEAD,
        RBIT,
        HEAD_DIM,
        BLOCK_M=BLOCK_M,
        num_stages=2,
        num_warps=4,
    )
