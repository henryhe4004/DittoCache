from myTransformer.cache.kernels.triton_hash_encode_new import (
    prefill_multi_hash_encode,
    decode_multi_hash_encode_qk,
    decode_multi_hash_encode_qqk,
    decode_multi_hash_encode_k,
    decode_multi_hash_encode_q,
)
import torch
from functools import partial


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
    print((t1 - t0) * 1000 / 100)


def matmul(key_states, hash_weight):
    return torch.matmul(key_states, hash_weight)


def torch_hash_encode(data: torch.Tensor, hash_weight: torch.Tensor,
                      packbit_aux_tensor: torch.Tensor):
    output_dtype = torch.int32
    chunk_size = 32

    RBIT = hash_weight.shape[-1]
    chunk_num = RBIT // chunk_size

    BSZ, SEQ, HEAD, HEAD_DIM = data.shape
    output_shape = (BSZ, SEQ, HEAD, chunk_num, chunk_size)

    key_code = torch.einsum("bshd,hdr->bshr", data, hash_weight) > 0
    packbit_key_code = key_code.reshape(*output_shape)
    packbit_key_code = packbit_key_code * packbit_aux_tensor
    packbit_key_code = packbit_key_code.sum(dim=-1, dtype=output_dtype)

    return packbit_key_code


def torch_gqa_hash_encode(data: torch.Tensor, hash_weight: torch.Tensor,
                          packbit_aux_tensor: torch.Tensor):
    output_dtype = torch.int32
    chunk_size = 32

    RBIT = hash_weight.shape[-1]
    chunk_num = RBIT // chunk_size

    BSZ, SEQ, HEAD, HEAD_DIM = data.shape
    HEADKV = hash_weight.shape[0]
    output_shape = (BSZ, SEQ, HEAD, chunk_num, chunk_size)
    data = data.view(BSZ, SEQ, HEADKV, -1, HEAD_DIM)

    key_code = torch.einsum("bshgd,hdr->bshgr", data, hash_weight) > 0
    packbit_key_code = key_code.reshape(*output_shape)
    packbit_key_code = packbit_key_code * packbit_aux_tensor
    packbit_key_code = packbit_key_code.sum(dim=-1, dtype=output_dtype)

    return packbit_key_code


torch.cuda.set_device(7)
torch.manual_seed(42)
device = "cuda"
dtype = torch.float16

BSZ = 128
HEAD = 4
HEAD_DIM = 128
RBIT = 128
hash_weight = torch.normal(
    0,
    2,
    size=(HEAD, HEAD_DIM, RBIT),
    device=device,
    dtype=dtype,
)
packbit_aux_tensor = torch.pow(
    2, torch.arange(0, 32, 1, dtype=torch.int32, device=device))

print("Prefill")
SEQ = 4000
key_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
                         dtype=torch.float16,
                         device=device)
triton_output = torch.empty((BSZ, SEQ * 2, HEAD, int(RBIT / 32)),
                            dtype=torch.int32,
                            device=device)
torch_output = torch_hash_encode(key_states, hash_weight, packbit_aux_tensor)
prefill_multi_hash_encode(key_states, hash_weight, triton_output,
                          packbit_aux_tensor)
assert (torch_output == triton_output[:, :SEQ, :, :]).all()
bench(partial(torch_hash_encode, key_states, hash_weight, packbit_aux_tensor))
bench(
    partial(prefill_multi_hash_encode, key_states, hash_weight, triton_output,
            packbit_aux_tensor))

print("Decode k")
SEQ = 1
LEN = 8000
key_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
                         dtype=torch.float16,
                         device=device)
triton_output = torch.empty((BSZ, LEN * 2, HEAD, int(RBIT / 32)),
                            dtype=torch.int32,
                            device=device)
torch_output = torch_hash_encode(key_states, hash_weight, packbit_aux_tensor)
decode_multi_hash_encode_k(key_states, triton_output, hash_weight,
                           packbit_aux_tensor, LEN)
assert (torch_output == triton_output[:, LEN:LEN + 1, :, :]).all()
bench(partial(torch_hash_encode, key_states, hash_weight, packbit_aux_tensor))
bench(
    partial(decode_multi_hash_encode_k, key_states, triton_output, hash_weight,
            packbit_aux_tensor, LEN))

print("Decode q")
HEADQ = 28
query_states = torch.randn((BSZ, 1, HEADQ, HEAD_DIM),
                           dtype=torch.float16,
                           device=device)
triton_output = torch.empty((BSZ, 1, HEADQ, int(RBIT / 32)),
                            dtype=torch.int32,
                            device=device)
torch_output = torch_gqa_hash_encode(query_states, hash_weight,
                                     packbit_aux_tensor)
decode_multi_hash_encode_q(query_states, triton_output, hash_weight,
                           packbit_aux_tensor)
assert (torch_output == triton_output).all()
bench(
    partial(torch_gqa_hash_encode, query_states, hash_weight,
            packbit_aux_tensor))
bench(
    partial(decode_multi_hash_encode_q, query_states, triton_output,
            hash_weight, packbit_aux_tensor))

print("Decode qk")
HEADQ = 32
SEQ = 1
LEN = 8000
q_hash_weight = torch.normal(
    0,
    2,
    size=(HEAD, HEAD_DIM, RBIT),
    device=device,
    dtype=dtype,
)
query_states = torch.randn((BSZ, SEQ, HEADQ, HEAD_DIM),
                           dtype=torch.float16,
                           device=device)
key_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
                         dtype=torch.float16,
                         device=device)
triton_q_output = torch.empty((BSZ, 1, HEADQ, int(RBIT / 32)),
                              dtype=torch.int32,
                              device=device)
triton_k_output = torch.empty((BSZ, LEN * 2, HEAD, int(RBIT / 32)),
                              dtype=torch.int32,
                              device=device)
torch_q_output = torch_gqa_hash_encode(query_states, q_hash_weight,
                                       packbit_aux_tensor)
torch_k_output = torch_hash_encode(key_states, hash_weight, packbit_aux_tensor)
decode_multi_hash_encode_qk(key_states, triton_k_output, hash_weight,
                            query_states, triton_q_output, q_hash_weight,
                            packbit_aux_tensor, LEN)
assert (torch_k_output == triton_k_output[:, LEN:LEN + 1, :, :]).all()
assert (torch_q_output == triton_q_output).all()
bench(
    partial(torch_gqa_hash_encode, query_states, q_hash_weight,
            packbit_aux_tensor))
bench(
    partial(decode_multi_hash_encode_qk, key_states, triton_k_output,
            hash_weight, query_states, triton_q_output, q_hash_weight,
            packbit_aux_tensor, LEN))

print("Decode qqk")
HEADQ = 32
SEQ = 1
LEN = 8000
q_hash_weight = torch.normal(
    0,
    2,
    size=(HEAD, HEAD_DIM, RBIT),
    device=device,
    dtype=dtype,
)
q_hash_weight2 = torch.normal(
    0,
    2,
    size=(HEAD, HEAD_DIM, RBIT),
    device=device,
    dtype=dtype,
)
query_states = torch.randn((BSZ, SEQ, HEADQ, HEAD_DIM),
                           dtype=torch.float16,
                           device=device)
query_states2 = torch.randn((BSZ, SEQ, HEADQ, HEAD_DIM),
                            dtype=torch.float16,
                            device=device)
key_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
                         dtype=torch.float16,
                         device=device)
triton_q_output = torch.zeros((BSZ, 1, HEADQ, int(RBIT / 32)),
                              dtype=torch.int32,
                              device=device)
triton_q_output2 = torch.zeros((BSZ, 1, HEADQ, int(RBIT / 32)),
                               dtype=torch.int32,
                               device=device)
triton_k_output = torch.zeros((BSZ, LEN * 2, HEAD, int(RBIT / 32)),
                              dtype=torch.int32,
                              device=device)
torch_q_output = torch_gqa_hash_encode(query_states, q_hash_weight,
                                       packbit_aux_tensor)
torch_q_output2 = torch_gqa_hash_encode(query_states2, q_hash_weight2,
                                        packbit_aux_tensor)
torch_k_output = torch_hash_encode(key_states, hash_weight, packbit_aux_tensor)
decode_multi_hash_encode_qqk(key_states, triton_k_output, hash_weight,
                             query_states, triton_q_output, q_hash_weight,
                             query_states2, triton_q_output2, q_hash_weight2,
                             packbit_aux_tensor, LEN)
assert (torch_k_output == triton_k_output[:, LEN:LEN + 1, :, :]).all()
assert (torch_q_output == triton_q_output).all()
assert (torch_q_output2 == triton_q_output2).all()
bench(
    partial(torch_gqa_hash_encode, query_states, q_hash_weight,
            packbit_aux_tensor))
bench(
    partial(decode_multi_hash_encode_qqk, key_states, triton_k_output,
            hash_weight, query_states, triton_q_output, q_hash_weight,
            query_states2, triton_q_output2, q_hash_weight2,
            packbit_aux_tensor, LEN))
