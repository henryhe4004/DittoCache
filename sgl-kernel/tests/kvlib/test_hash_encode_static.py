from sglang.jit_kernel.triton_kernels.hash.decode_encode import (
    hash_encode_append_decode_k,
    hash_encode_append_decode_qk,
    hash_encode_append_decode_qqk,
)
import torch
from functools import partial
import random


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


def test_encode_k():
    bsz = 16
    num_kv_heads = 8
    head_dim = 128
    rbit = 256
    hash_dim = rbit // 32
    dtype = torch.bfloat16
    for max_seqlen in [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        curr_seqlen = random.randint(0, max_seqlen - 1)
        print(f"Test {max_seqlen=}, {curr_seqlen=}......", end="")
        hash_weight = torch.normal(0,
                                   2,
                                   size=(num_kv_heads, head_dim, rbit),
                                   dtype=dtype,
                                   device="cuda")
        packbit_aux_tensor = torch.pow(
            2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))
        key_states = torch.randn((bsz, 1, num_kv_heads, head_dim),
                                 dtype=dtype,
                                 device="cuda")
        cache_length = torch.tensor([curr_seqlen],
                                    dtype=torch.int32,
                                    device="cuda")

        target_cache = torch.zeros((bsz, max_seqlen, num_kv_heads, hash_dim),
                                   dtype=torch.int32,
                                   device="cuda")

        hash_encode_append_decode_k(
            key_states,
            target_cache,
            hash_weight,
            packbit_aux_tensor,
            cache_length,
        )
        torch_encode_k = torch_hash_encode(key_states, hash_weight,
                                           packbit_aux_tensor)
        assert (torch_encode_k == target_cache[:, curr_seqlen:curr_seqlen +
                                               1, :, :]).all()
        print("Passed")


def test_encode_qk():
    bsz = 16
    num_kv_heads = 8
    num_heads = 32
    head_dim = 128
    rbit = 256
    hash_dim = rbit // 32
    dtype = torch.bfloat16
    for max_seqlen in [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        curr_seqlen = random.randint(0, max_seqlen - 1)
        print(f"Test {max_seqlen=}, {curr_seqlen=}......", end="")
        hash_weight = torch.normal(0,
                                   2,
                                   size=(num_kv_heads, head_dim, rbit),
                                   dtype=dtype,
                                   device="cuda")
        packbit_aux_tensor = torch.pow(
            2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))
        key_states = torch.randn((bsz, 1, num_kv_heads, head_dim),
                                 dtype=dtype,
                                 device="cuda")
        query_states = torch.randn((bsz, 1, num_heads, head_dim),
                                   dtype=dtype,
                                   device="cuda")
        cache_length = torch.tensor([curr_seqlen],
                                    dtype=torch.int32,
                                    device="cuda")

        target_cache = torch.zeros((bsz, max_seqlen, num_kv_heads, hash_dim),
                                   dtype=torch.int32,
                                   device="cuda")
        output_query = torch.zeros((bsz, 1, num_heads, hash_dim),
                                   dtype=torch.int32,
                                   device="cuda")
        hash_encode_append_decode_qk(
            key_states,
            target_cache,
            hash_weight,
            query_states,
            output_query,
            hash_weight,
            packbit_aux_tensor,
            cache_length,
        )
        torch_encode_k = torch_hash_encode(key_states, hash_weight,
                                           packbit_aux_tensor)
        torch_encode_q = torch_gqa_hash_encode(query_states, hash_weight,
                                               packbit_aux_tensor)
        assert (torch_encode_k == target_cache[:, curr_seqlen:curr_seqlen +
                                               1, :, :]).all()
        assert (torch_encode_q == output_query).all()
        print("Passed")


def test_encode_qqk():
    bsz = 16
    num_kv_heads = 8
    num_heads = 32
    head_dim = 128
    rbit = 256
    hash_dim = rbit // 32
    dtype = torch.bfloat16
    for max_seqlen in [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        curr_seqlen = random.randint(0, max_seqlen - 1)
        print(f"Test {max_seqlen=}, {curr_seqlen=}......", end="")
        hash_weight = torch.normal(0,
                                   2,
                                   size=(num_kv_heads, head_dim, rbit),
                                   dtype=dtype,
                                   device="cuda")
        prefetch_hash_weight = torch.normal(0,
                                            2,
                                            size=(num_kv_heads, head_dim, rbit),
                                            dtype=dtype,
                                            device="cuda")
        packbit_aux_tensor = torch.pow(
            2, torch.arange(0, 32, 1, dtype=torch.int32, device="cuda"))
        key_states = torch.randn((bsz, 1, num_kv_heads, head_dim),
                                 dtype=dtype,
                                 device="cuda")
        query_states = torch.randn((bsz, 1, num_heads, head_dim),
                                   dtype=dtype,
                                   device="cuda")
        prefetch_query_states = torch.randn((bsz, 1, num_heads, head_dim),
                                            dtype=dtype,
                                            device="cuda")
        cache_length = torch.tensor([curr_seqlen],
                                    dtype=torch.int32,
                                    device="cuda")

        target_cache = torch.zeros((bsz, max_seqlen, num_kv_heads, hash_dim),
                                   dtype=torch.int32,
                                   device="cuda")
        output_query = torch.zeros((bsz, 1, num_heads, hash_dim),
                                   dtype=torch.int32,
                                   device="cuda")
        output_prefetch_query = torch.zeros((bsz, 1, num_heads, hash_dim),
                                            dtype=torch.int32,
                                            device="cuda")
        hash_encode_append_decode_qqk(
            key_states,
            target_cache,
            hash_weight,
            query_states,
            output_query,
            hash_weight,
            prefetch_query_states,
            output_prefetch_query,
            prefetch_hash_weight,
            packbit_aux_tensor,
            cache_length,
        )
        torch_encode_k = torch_hash_encode(key_states, hash_weight,
                                           packbit_aux_tensor)
        torch_encode_q = torch_gqa_hash_encode(query_states, hash_weight,
                                               packbit_aux_tensor)
        torch_encode_q_prefetch = torch_gqa_hash_encode(prefetch_query_states, prefetch_hash_weight, packbit_aux_tensor)
        assert (torch_encode_k == target_cache[:, curr_seqlen:curr_seqlen +
                                               1, :, :]).all()
        assert (torch_encode_q == output_query).all()
        assert (torch_encode_q_prefetch == output_prefetch_query).all()
        print("Passed")


if __name__ == "__main__":
    print("-" * 40)
    print("Test encode k......")
    test_encode_k()
    print("-" * 40)
    print("Test encode qk......")
    test_encode_qk()
    print("-" * 40)
    print("Test encode qqk......")
    test_encode_qqk()
