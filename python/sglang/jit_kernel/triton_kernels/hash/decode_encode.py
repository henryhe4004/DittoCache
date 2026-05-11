import torch
import triton
import triton.language as tl


@triton.jit
def _hash_encode_append_decode_qqk(
    key_data_ptr,
    key_data_stride0,
    query_data_ptr,
    query_data_stride0,
    query2_data_ptr,
    query2_data_stride0,
    key_hash_weight_ptr,
    query_hash_weight_ptr,
    query2_hash_weight_ptr,
    packbit_tensor_ptr,
    key_code_output_ptr,
    key_code_output_stride0,
    query_code_output_ptr,
    query_code_output_stride0,
    query2_code_output_ptr,
    query2_code_output_stride0,
    seqlen_ptr,
    BSZ,
    KV_HEAD,
    Q_HEAD,
    RBIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    CHUNK_SIZE: tl.constexpr = 32
    NUM_CHUNK: tl.constexpr = RBIT // 32
    BLOCK_N: tl.constexpr = CHUNK_SIZE
    KV_GROUP = Q_HEAD // KV_HEAD
    SLICE = tl.cdiv(KV_GROUP, BLOCK_M)
    SLICE2 = SLICE + tl.cdiv(KV_GROUP, BLOCK_M)

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1) // KV_HEAD
    head_id = tl.program_id(1) % KV_HEAD
    start_n = tl.program_id(2)

    cur_k_len = tl.load(seqlen_ptr + batch_id)

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))
    if start_m < SLICE:
        Q_weight_ptr = tl.make_block_ptr(
            base=query_hash_weight_ptr + head_id * HEAD_DIM * RBIT,
            shape=(HEAD_DIM, RBIT),
            strides=(RBIT, 1),
            offsets=(0, start_n * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(1, 0),
        )
        q_weight = tl.load(Q_weight_ptr)  # [HEAD_DIM, BLOCK_N]

        # do operator for query
        Q_data_block_ptr = tl.make_block_ptr(
            base=query_data_ptr + batch_id * query_data_stride0 +
            head_id * KV_GROUP * HEAD_DIM,
            shape=(KV_GROUP, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        Q_output_block_ptr = tl.make_block_ptr(
            base=query_code_output_ptr + batch_id * query_code_output_stride0 +
            head_id * KV_GROUP * NUM_CHUNK,
            shape=(KV_GROUP, NUM_CHUNK),
            strides=(NUM_CHUNK, 1),
            offsets=(start_m * BLOCK_M, start_n),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )

        q_data = tl.load(Q_data_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")

        q_acc = tl.dot(q_data, q_weight) > 0  # [BLOCK_M, BLOCK_N]
        q_acc = q_acc.to(packbit_tensor.type.element_ty)
        q_acc = q_acc * packbit_tensor
        q_acc = tl.sum(q_acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(Q_output_block_ptr, q_acc, boundary_check=(1, 0))

    elif start_m < SLICE2:
        start_m = start_m - SLICE
        Q_weight_ptr2 = tl.make_block_ptr(
            base=query2_hash_weight_ptr + head_id * HEAD_DIM * RBIT,
            shape=(HEAD_DIM, RBIT),
            strides=(RBIT, 1),
            offsets=(0, start_n * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(1, 0),
        )
        q_weight2 = tl.load(Q_weight_ptr2)  # [HEAD_DIM, BLOCK_N]

        # do operator for query
        Q_data_block_ptr2 = tl.make_block_ptr(
            base=query2_data_ptr + batch_id * query2_data_stride0 +
            head_id * KV_GROUP * HEAD_DIM,
            shape=(KV_GROUP, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        Q_output_block_ptr2 = tl.make_block_ptr(
            base=query2_code_output_ptr +
            batch_id * query2_code_output_stride0 +
            head_id * KV_GROUP * NUM_CHUNK,
            shape=(KV_GROUP, NUM_CHUNK),
            strides=(NUM_CHUNK, 1),
            offsets=(start_m * BLOCK_M, start_n),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )

        q_data2 = tl.load(Q_data_block_ptr2,
                          boundary_check=(1, 0),
                          padding_option="zero")
        q_acc2 = tl.dot(q_data2, q_weight2) > 0  # [BLOCK_M, BLOCK_N]
        q_acc2 = q_acc2.to(packbit_tensor.type.element_ty)
        q_acc2 = q_acc2 * packbit_tensor
        q_acc2 = tl.sum(q_acc2, axis=1).reshape(BLOCK_M, 1)
        tl.store(Q_output_block_ptr2, q_acc2, boundary_check=(1, 0))

    if start_m >= SLICE2:
        K_weight_ptr = tl.make_block_ptr(
            base=key_hash_weight_ptr + head_id * HEAD_DIM * RBIT,
            shape=(HEAD_DIM, RBIT),
            strides=(RBIT, 1),
            offsets=(0, start_n * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(1, 0),
        )
        k_weight = tl.load(K_weight_ptr)  # [HEAD_DIM, BLOCK_N]

        start_m = start_m - SLICE2

        K_data_block_ptr = tl.make_block_ptr(
            base=key_data_ptr + batch_id * key_data_stride0 +
            head_id * HEAD_DIM,
            shape=(1, HEAD_DIM),
            strides=(KV_HEAD * HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        K_output_block_ptr = tl.make_block_ptr(
            base=key_code_output_ptr + batch_id * key_code_output_stride0 +
            cur_k_len * KV_HEAD * NUM_CHUNK + head_id * NUM_CHUNK,
            shape=(1, NUM_CHUNK),
            strides=(KV_HEAD * NUM_CHUNK, 1),
            offsets=(start_m * BLOCK_M, start_n),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )

        k_data = tl.load(K_data_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")  # [BLOCK_M, HEAD_DIM]

        k_acc = tl.dot(k_data, k_weight) > 0  # [BLOCK_M, BLOCK_N]
        k_acc = k_acc.to(packbit_tensor.type.element_ty)
        k_acc = k_acc * packbit_tensor
        k_acc = tl.sum(k_acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(K_output_block_ptr, k_acc, boundary_check=(1, 0))


def hash_encode_append_decode_qqk(
    key_data: torch.Tensor,
    key_code_output: torch.Tensor,
    key_hash_weights: torch.Tensor,
    query_data: torch.Tensor,
    query_code_output: torch.Tensor,
    query_hash_weights: torch.Tensor,
    query_data2: torch.Tensor,
    query_code_output2: torch.Tensor,
    query_hash_weights2: torch.Tensor,
    packbit_aux_tensor: torch.Tensor,
    seqlen_tensor: torch.Tensor,
):

    RBIT = key_hash_weights.shape[-1]
    NUM_CHUNK = RBIT // 32
    BSZ, SEQ, NUM_KV_HEAD, HEAD_DIM = key_data.shape
    NUM_HEAD = query_data.shape[2]
    assert SEQ == 1
    KV_GROUP = NUM_HEAD // NUM_KV_HEAD
    BLOCK_M = 16

    grid = lambda args: (
        triton.cdiv(KV_GROUP, BLOCK_M) + triton.cdiv(
            KV_GROUP, BLOCK_M) + 1,
        BSZ * NUM_KV_HEAD,
        NUM_CHUNK,
    )

    _hash_encode_append_decode_qqk[grid](
        key_data,
        key_data.stride(0),
        query_data,
        query_data.stride(0),
        query_data2,
        query_data2.stride(0),
        key_hash_weights,
        query_hash_weights,
        query_hash_weights2,
        packbit_aux_tensor,
        key_code_output,
        key_code_output.stride(0),
        query_code_output,
        query_code_output.stride(0),
        query_code_output2,
        query_code_output2.stride(0),
        seqlen_tensor,
        BSZ,
        NUM_KV_HEAD,
        NUM_HEAD,
        RBIT,
        HEAD_DIM,
        BLOCK_M=BLOCK_M,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _hash_encode_append_decode_qk(
    key_data_ptr,
    key_data_stride0,
    query_data_ptr,
    query_data_stride0,
    key_hash_weight_ptr,
    query_hash_weight_ptr,
    packbit_tensor_ptr,
    key_code_output_ptr,
    key_code_output_stride0,
    query_code_output_ptr,
    query_code_output_stride0,
    seqlen_ptr,
    BSZ,
    KV_HEAD,
    Q_HEAD,
    RBIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    CHUNK_SIZE: tl.constexpr = 32
    NUM_CHUNK: tl.constexpr = RBIT // 32
    BLOCK_N: tl.constexpr = CHUNK_SIZE

    KV_GROUP = Q_HEAD // KV_HEAD

    SLICE = tl.cdiv(KV_GROUP, BLOCK_M)

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1) // KV_HEAD
    head_id = tl.program_id(1) % KV_HEAD
    start_n = tl.program_id(2)

    cur_k_len = tl.load(seqlen_ptr + batch_id)

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    if start_m < SLICE:
        Q_weight_ptr = tl.make_block_ptr(
            base=query_hash_weight_ptr + head_id * HEAD_DIM * RBIT,
            shape=(HEAD_DIM, RBIT),
            strides=(RBIT, 1),
            offsets=(0, start_n * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(1, 0),
        )
        q_weight = tl.load(Q_weight_ptr)  # [HEAD_DIM, BLOCK_N]

        # do operator for query
        Q_data_block_ptr = tl.make_block_ptr(
            base=query_data_ptr + batch_id * query_data_stride0 +
            head_id * KV_GROUP * HEAD_DIM,
            shape=(KV_GROUP, HEAD_DIM),
            strides=(HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        Q_output_block_ptr = tl.make_block_ptr(
            base=query_code_output_ptr + batch_id * query_code_output_stride0 +
            head_id * KV_GROUP * NUM_CHUNK,
            shape=(KV_GROUP, NUM_CHUNK),
            strides=(NUM_CHUNK, 1),
            offsets=(start_m * BLOCK_M, start_n),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )

        q_data = tl.load(Q_data_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")

        q_acc = tl.dot(q_data, q_weight) > 0  # [BLOCK_M, BLOCK_N]
        q_acc = q_acc.to(packbit_tensor.type.element_ty)
        q_acc = q_acc * packbit_tensor
        q_acc = tl.sum(q_acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(Q_output_block_ptr, q_acc, boundary_check=(1, 0))

    else:
        K_weight_ptr = tl.make_block_ptr(
            base=key_hash_weight_ptr + head_id * HEAD_DIM * RBIT,
            shape=(HEAD_DIM, RBIT),
            strides=(RBIT, 1),
            offsets=(0, start_n * BLOCK_N),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(1, 0),
        )
        k_weight = tl.load(K_weight_ptr)  # [HEAD_DIM, BLOCK_N]

        start_m = start_m - SLICE

        K_data_block_ptr = tl.make_block_ptr(
            base=key_data_ptr + batch_id * key_data_stride0 +
            head_id * HEAD_DIM,
            shape=(1, HEAD_DIM),
            strides=(KV_HEAD * HEAD_DIM, 1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        K_output_block_ptr = tl.make_block_ptr(
            base=key_code_output_ptr + batch_id * key_code_output_stride0 +
            cur_k_len * KV_HEAD * NUM_CHUNK + head_id * NUM_CHUNK,
            shape=(1, NUM_CHUNK),
            strides=(KV_HEAD * NUM_CHUNK, 1),
            offsets=(start_m * BLOCK_M, start_n),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )

        k_data = tl.load(K_data_block_ptr,
                         boundary_check=(1, 0),
                         padding_option="zero")  # [BLOCK_M, HEAD_DIM]

        k_acc = tl.dot(k_data, k_weight) > 0  # [BLOCK_M, BLOCK_N]
        k_acc = k_acc.to(packbit_tensor.type.element_ty)
        k_acc = k_acc * packbit_tensor
        k_acc = tl.sum(k_acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(K_output_block_ptr, k_acc, boundary_check=(1, 0))


def hash_encode_append_decode_qk(
        key_data: torch.Tensor, key_code_output: torch.Tensor,
        key_hash_weights: torch.Tensor, query_data: torch.Tensor,
        query_code_output: torch.Tensor, query_hash_weights: torch.Tensor,
        packbit_aux_tensor: torch.Tensor, seqlen_tensor: torch.Tensor):

    assert key_data.is_contiguous()

    RBIT = key_hash_weights.shape[-1]
    assert query_hash_weights.shape[-1] == RBIT
    assert RBIT % 32 == 0

    NUM_CHUNK = RBIT // 32

    BSZ, SEQ, NUM_KV_HEAD, HEAD_DIM = key_data.shape
    NUM_HEAD = query_data.shape[2]

    assert SEQ == 1

    KV_GROUP = NUM_HEAD // NUM_KV_HEAD

    BLOCK_M = 16

    grid = lambda args: (
        triton.cdiv(KV_GROUP * SEQ, BLOCK_M) + triton.cdiv(SEQ, BLOCK_M),
        BSZ * NUM_KV_HEAD,
        NUM_CHUNK,
    )

    _hash_encode_append_decode_qk[grid](
        key_data,
        key_data.stride(0),
        query_data,
        query_data.stride(0),
        key_hash_weights,
        query_hash_weights,
        packbit_aux_tensor,
        key_code_output,
        key_code_output.stride(0),
        query_code_output,
        query_code_output.stride(0),
        seqlen_tensor,
        BSZ,
        NUM_KV_HEAD,
        NUM_HEAD,
        RBIT,
        HEAD_DIM,
        BLOCK_M=BLOCK_M,
        num_warps=4,
        num_stages=1,
    )



@triton.jit
def _hash_encode_append_decode_k(
    key_data_ptr,
    key_data_stride0,
    hash_weight_ptr,
    packbit_tensor_ptr,
    key_code_output_ptr,
    key_code_output_stride0,
    seqlen_ptr,
    BSZ,
    KV_HEAD,
    RBIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    CHUNK_SIZE: tl.constexpr = 32
    NUM_CHUNK: tl.constexpr = RBIT // 32
    BLOCK_N: tl.constexpr = CHUNK_SIZE

    start_m = tl.program_id(0)
    batch_id = tl.program_id(1) // KV_HEAD
    head_id = tl.program_id(1) % KV_HEAD
    start_n = tl.program_id(2)

    cur_k_len = tl.load(seqlen_ptr + batch_id)

    Weight_ptr = tl.make_block_ptr(
        base=hash_weight_ptr + head_id * HEAD_DIM * RBIT,
        shape=(HEAD_DIM, RBIT),
        strides=(RBIT, 1),
        offsets=(0, start_n * BLOCK_N),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(1, 0),
    )

    # load pack tensor
    packbit_tensor = tl.load(packbit_tensor_ptr + tl.arange(0, CHUNK_SIZE))

    weight = tl.load(Weight_ptr)  # [HEAD_DIM, BLOCK_N]

    K_data_block_ptr = tl.make_block_ptr(
        base=key_data_ptr + batch_id * key_data_stride0 + head_id * HEAD_DIM,
        shape=(1, HEAD_DIM),
        strides=(KV_HEAD * HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    K_output_block_ptr = tl.make_block_ptr(
        base=key_code_output_ptr + batch_id * key_code_output_stride0 +
        cur_k_len * KV_HEAD * NUM_CHUNK + head_id * NUM_CHUNK,
        shape=(1, NUM_CHUNK),
        strides=(KV_HEAD * NUM_CHUNK, 1),
        offsets=(start_m * BLOCK_M, start_n),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )

    k_data = tl.load(K_data_block_ptr,
                     boundary_check=(1, 0),
                     padding_option="zero")  # [BLOCK_M, HEAD_DIM]

    k_acc = tl.dot(k_data, weight) > 0  # [BLOCK_M, BLOCK_N]
    k_acc = k_acc.to(packbit_tensor.type.element_ty)
    k_acc = k_acc * packbit_tensor
    k_acc = tl.sum(k_acc, axis=1).reshape(BLOCK_M, 1)
    tl.store(K_output_block_ptr, k_acc, boundary_check=(1, 0))


def hash_encode_append_decode_k(key_data: torch.Tensor,
                               key_code_output: torch.Tensor,
                               hash_weights: torch.Tensor,
                               packbit_aux_tensor: torch.Tensor,
                               seqlen_tensor: torch.Tensor,):
    RBIT = hash_weights.shape[-1]
    assert RBIT % 32 == 0

    NUM_CHUNK = RBIT // 32

    BSZ, SEQ, NUM_KV_HEAD, HEAD_DIM = key_data.shape

    assert SEQ == 1

    BLOCK_M = 16

    grid = lambda args: (
        triton.cdiv(SEQ, BLOCK_M),
        BSZ * NUM_KV_HEAD,
        NUM_CHUNK,
    )

    _hash_encode_append_decode_k[grid](
        key_data,
        key_data.stride(0),
        hash_weights,
        packbit_aux_tensor,
        key_code_output,
        key_code_output.stride(0),
        seqlen_tensor,
        BSZ,
        NUM_KV_HEAD,
        RBIT,
        HEAD_DIM,
        BLOCK_M=BLOCK_M,
        num_warps=4,
        num_stages=1,
    )
