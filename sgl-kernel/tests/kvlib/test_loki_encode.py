from myTransformer.cache.kernels.triton_loki_kernels import (
    prefill_loki_encode,
    decode_loki_encode_k,
    decode_loki_encode_qk,
    decode_loki_encode_qqk,
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


def torch_loki_encode(key: torch.Tensor, weight: torch.Tensor,
                      num_channels: int):
    encode = torch.einsum("bshd,hdr->bshr", key, weight[..., :num_channels])
    return encode


def torch_gqa_loki_encode(query: torch.Tensor, weight: torch.Tensor,
                          num_channels: int):
    b, s, h, d = query.shape
    hk = weight.shape[0]
    query = query.view(b, s, hk, -1, d)
    encode = torch.einsum("bshgd,hdr->bshgr", query,
                          weight[..., :num_channels])
    return encode.reshape(b, s, h, num_channels)


torch.cuda.set_device(0)
torch.manual_seed(42)
device = "cuda"
dtype = torch.float16

BSZ = 32
HEAD = 4
HEADQ = 28
HEAD_DIM = 128
NUM_CHANNELS = 32
pca_weight1 = torch.randn((HEAD, HEAD_DIM, HEAD_DIM),
                          dtype=dtype,
                          device=device)
pca_weight2 = torch.randn((HEAD, HEAD_DIM, HEAD_DIM),
                          dtype=dtype,
                          device=device)
pca_weight3 = torch.randn((HEAD, HEAD_DIM, HEAD_DIM),
                          dtype=dtype,
                          device=device)
print("Prefill")
SEQ = 8000
key_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
                         dtype=dtype,
                         device=device)
triton_output = torch.empty((BSZ, SEQ * 2, HEAD, NUM_CHANNELS),
                            dtype=dtype,
                            device=device)
torch_output = torch_loki_encode(key_states, pca_weight1, NUM_CHANNELS)
prefill_loki_encode(key_states, pca_weight1, triton_output, NUM_CHANNELS)
assert torch.equal(torch_output, triton_output[:, :SEQ, :, :])
bench(partial(torch_loki_encode, key_states, pca_weight1, NUM_CHANNELS))
bench(
    partial(prefill_loki_encode, key_states, pca_weight1, triton_output,
            NUM_CHANNELS))

print("Decode k")
SEQ = 1
LEN = 8000
key_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
                         dtype=dtype,
                         device=device)
triton_output = torch.empty((BSZ, LEN * 2, HEAD, NUM_CHANNELS),
                            dtype=dtype,
                            device=device)
torch_output = torch_loki_encode(key_states, pca_weight1, NUM_CHANNELS)
decode_loki_encode_k(key_states, triton_output, pca_weight1, NUM_CHANNELS, LEN)
assert torch.equal(torch_output, triton_output[:, LEN:LEN + 1, :, :])
bench(partial(torch_loki_encode, key_states, pca_weight1, NUM_CHANNELS))
bench(
    partial(decode_loki_encode_k, key_states, triton_output, pca_weight1,
            NUM_CHANNELS, LEN))

print("Decode qk")
SEQ = 1
LEN = 8000
query_states = torch.randn((BSZ, SEQ, HEADQ, HEAD_DIM),
                           dtype=dtype,
                           device=device)
key_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
                         dtype=dtype,
                         device=device)
triton_q_output = torch.empty((BSZ, 1, HEADQ, NUM_CHANNELS),
                              dtype=dtype,
                              device=device)
triton_k_output = torch.empty((BSZ, LEN * 2, HEAD, NUM_CHANNELS),
                              dtype=dtype,
                              device=device)
torch_q_output = torch_gqa_loki_encode(query_states, pca_weight2, NUM_CHANNELS)
torch_k_output = torch_loki_encode(key_states, pca_weight1, NUM_CHANNELS)
decode_loki_encode_qk(key_states, triton_k_output, pca_weight1, query_states,
                      triton_q_output, pca_weight2, NUM_CHANNELS, LEN)
assert torch.equal(torch_k_output, triton_k_output[:, LEN:LEN + 1, :, :])
assert torch.equal(torch_q_output, triton_q_output)
bench(partial(torch_gqa_loki_encode, query_states, pca_weight2, NUM_CHANNELS))
bench(partial(torch_loki_encode, key_states, pca_weight1, NUM_CHANNELS))
bench(
    partial(decode_loki_encode_qk, key_states, triton_k_output, pca_weight1,
            query_states, triton_q_output, pca_weight2, NUM_CHANNELS, LEN))

print("Decode qqk")
SEQ = 1
LEN = 8000
query_states = torch.randn((BSZ, SEQ, HEADQ, HEAD_DIM),
                           dtype=dtype,
                           device=device)
query_states2 = torch.randn((BSZ, SEQ, HEADQ, HEAD_DIM),
                            dtype=dtype,
                            device=device)
key_states = torch.randn((BSZ, SEQ, HEAD, HEAD_DIM),
                         dtype=dtype,
                         device=device)
triton_q_output = torch.zeros((BSZ, 1, HEADQ, NUM_CHANNELS),
                              dtype=dtype,
                              device=device)
triton_q_output2 = torch.zeros((BSZ, 1, HEADQ, NUM_CHANNELS),
                               dtype=dtype,
                               device=device)
triton_k_output = torch.zeros((BSZ, LEN * 2, HEAD, NUM_CHANNELS),
                              dtype=dtype,
                              device=device)
torch_q_output = torch_gqa_loki_encode(query_states, pca_weight2, NUM_CHANNELS)
torch_q_output2 = torch_gqa_loki_encode(query_states2, pca_weight3,
                                        NUM_CHANNELS)
torch_k_output = torch_loki_encode(key_states, pca_weight1, NUM_CHANNELS)
decode_loki_encode_qqk(key_states, triton_k_output, pca_weight1, query_states,
                       triton_q_output, pca_weight2, query_states2,
                       triton_q_output2, pca_weight3, NUM_CHANNELS, LEN)
assert torch.equal(torch_k_output, triton_k_output[:, LEN:LEN + 1, :, :])
assert torch.equal(torch_q_output, triton_q_output)
assert torch.equal(torch_q_output2, triton_q_output2)
bench(partial(torch_gqa_loki_encode, query_states, pca_weight2, NUM_CHANNELS))
bench(partial(torch_gqa_loki_encode, query_states2, pca_weight3, NUM_CHANNELS))
bench(partial(torch_loki_encode, key_states, pca_weight1, NUM_CHANNELS))
bench(
    partial(decode_loki_encode_qqk, key_states, triton_k_output, pca_weight1,
            query_states, triton_q_output, pca_weight2, query_states2,
            triton_q_output2, pca_weight3, NUM_CHANNELS, LEN))
