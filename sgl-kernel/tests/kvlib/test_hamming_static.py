import torch
from functools import partial
import triton
import triton.language as tl
import sgl_kernel.kvlib as capi

torch.cuda.set_device(6)
torch.manual_seed(42)

@triton.jit
def _hash_encode(
    data_ptr,
    hash_weight_ptr,
    packbit_tensor_ptr,
    output_ptr,
    TOTAL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RBIT: tl.constexpr,
    NUM_CHUNK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    Data_block_ptr = tl.make_block_ptr(
        base=data_ptr,
        shape=(TOTAL_HEAD, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    Output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(TOTAL_HEAD, NUM_CHUNK),
        strides=(NUM_CHUNK, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0),
    )
    Weight_ptr = tl.make_block_ptr(
        base=hash_weight_ptr,
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
        weight = tl.load(Weight_ptr)  # [HEAD_DIM, BLOCK_N]
        acc = tl.dot(data, weight) > 0  # [BLOCK_M, BLOCK_N]
        acc = acc.to(packbit_tensor.type.element_ty)
        acc = acc * packbit_tensor
        acc = tl.sum(acc, axis=1).reshape(BLOCK_M, 1)
        tl.store(Output_block_ptr, acc, boundary_check=(1, 0))

        # move on
        Weight_ptr = tl.advance(Weight_ptr, (0, BLOCK_N))
        Output_block_ptr = tl.advance(Output_block_ptr, (0, 1))


# @torch.compile(fullgraph=True)
def hash_encode(
        data: torch.Tensor, hash_weight: torch.Tensor,
        packbit_aux_tensor: torch.tensor):
    with torch.cuda.device(data.device):
        assert data.is_contiguous()

        RBIT = hash_weight.shape[1]
        assert RBIT % 32 == 0

        output_dtype = torch.int32
        num_chunk = RBIT // 32
        chunk_size = 32

        if len(data.shape) == 4:
            BSZ, SEQ, NUM_HEAD, HEAD_DIM = data.shape
            output_shape = (BSZ, SEQ, NUM_HEAD, num_chunk)
            TOTAL_HEAD = BSZ * SEQ * NUM_HEAD
        elif len(data.shape) == 3:
            BSZ, NUM_HEAD, HEAD_DIM = data.shape
            output_shape = (BSZ, NUM_HEAD, num_chunk)
            TOTAL_HEAD = BSZ * NUM_HEAD
        else:
            TOTAL_HEAD, HEAD_DIM = data.shape
            output_shape = (TOTAL_HEAD, num_chunk)

        assert HEAD_DIM in {16, 32, 64, 128, 256}

        assert packbit_aux_tensor.numel() == chunk_size

        output = torch.empty(output_shape,
                             dtype=output_dtype,
                             device=data.device)

        extra_kern_args = {}
        BLOCK_M = 16
        grid = lambda args: (
            triton.cdiv(TOTAL_HEAD, BLOCK_M),
            1,
            1,
        )
        _hash_encode[grid](
            data,
            hash_weight,
            packbit_aux_tensor,
            output,
            TOTAL_HEAD,
            HEAD_DIM,
            RBIT,
            num_chunk,
            chunk_size,
            BLOCK_N=chunk_size,
            BLOCK_M=BLOCK_M,
        )
        return output


def bench(func):
    import time
    import numpy as np

    for i in range(5):
        func()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        func()
    torch.cuda.synchronize()
    t1 = time.time()
    latency = (t1 - t0) / 100
    print(latency * 1000)
    return latency


def torch_hamming_distance(key, query, hash_weight):
    h = query.shape[2]
    b, s, hk, _ = key.shape
    gqa = h // hk
    rbit = hash_weight.shape[-1]
    key = torch.matmul(key, hash_weight) > 0
    query = torch.matmul(query, hash_weight) > 0
    key = key.view(b, s, hk, 1, rbit).expand(-1, -1, -1, gqa,
                                             -1).reshape(b, s, h, rbit)
    hamming_distance = query.to(torch.float16) - key.to(torch.float16)
    hamming_distance = hamming_distance.abs().sum(dim=-1)
    return hamming_distance


def encode(key, query, hash_weight):
    packbit_aux_tensor = torch.pow(
        2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))
    key_code = hash_encode(key, hash_weight, packbit_aux_tensor)
    query_code = hash_encode(query, hash_weight, packbit_aux_tensor)
    return key_code, query_code

if __name__ == "__main__":
    b = 16
    rbit = 256
    h = 28
    hk = 4
    gqa = h // hk
    s = 4000
    key = torch.randn(b, s, hk, 128).to(torch.bfloat16).cuda()
    query = torch.randn(b, 1, h, 128).to(torch.bfloat16).cuda()

    mask = torch.zeros((b * hk, ), dtype=torch.bool, device='cuda')
    index = torch.arange(0, b * hk, 2, device='cuda')
    mask[index] = True

    real_s = s // 2
    real_s_tensor = torch.tensor([real_s], dtype=torch.int32, device='cuda')

    hash_weight = torch.normal(
        0,
        2,
        size=(128, rbit),
        device=key.device,
        dtype=key.dtype,
    )
    torch_output = torch_hamming_distance(key[:, :real_s, ...], query, hash_weight)
    torch_output = torch_output.transpose(1, 2).view(b, hk, gqa, -1).sum(2)
    torch_output = torch_output.reshape(-1, real_s)

    key_code, query_code = encode(key, query, hash_weight)
    my_output = torch.zeros((b * hk, s), dtype=torch.float16, device='cuda')
    torch.cuda.synchronize()

    capi.static_hamming_score_mask(
        key_code,
        query_code,
        mask,
        my_output,
        real_s_tensor,
        rbit,
        float(torch.finfo(torch.float16).max),
        0.0,
        0,
        0,
        0,
        0,
    )
    print(torch_output)
    print(my_output[..., :real_s])
    print((my_output[mask, :real_s] - torch_output[mask, :]).abs().max())

    test_key_code = key_code[:, :real_s].contiguous()
    bench(
        partial(
            capi.hamming_score,
            test_key_code,
            query_code,
            rbit,
            s,
            0,
            0,
        )
    )

    print("#gather heads", mask.sum().item())
    bench(
        partial(
            capi.static_hamming_score_mask,
            key_code,
            query_code,
            mask,
            my_output,
            real_s_tensor,
            rbit,
            float(torch.finfo(torch.float16).max),
            0.0,
            0,
            0,
            0,
            0,
        )
    )

    mask[:] = True
    print("#gather heads", mask.sum().item())
    bench(
        partial(
            capi.static_hamming_score_mask,
            key_code,
            query_code,
            mask,
            my_output,
            real_s_tensor,
            rbit,
            float(torch.finfo(torch.float16).max),
            0.0,
            0,
            0,
            0,
            0,
        )
    )

    capi.static_hamming_score_mask(
        key_code,
        query_code,
        mask,
        my_output,
        real_s_tensor,
        rbit,
        float(torch.finfo(torch.float16).max),
        0.0,
        4,
        64,
        0,
        0,
    )
    print(my_output[..., -64:])
    print(my_output[..., :4])

    capi.static_hamming_score_mask(
        key_code,
        query_code,
        mask,
        my_output,
        real_s_tensor,
        rbit,
        float(torch.finfo(torch.float16).max),
        0.0,
        0,
        0,
        4,
        64,
    )
    print(my_output[..., -64:])
    print(my_output[..., :4])
