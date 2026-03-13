import torch
import math
from functools import partial
from myTransformer.cache.kernels.triton_loki_kernels import (
    loki_score, )


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
    
    time_cost = (t1 - t0) * 1000 / 100
    return time_cost


def torch_loki_score(query_states: torch.Tensor, key_states: torch.Tensor,
                     seq_len: int):
    key_states = key_states[:, :seq_len, ...]
    b, s, hk, d = key_states.shape
    hq = query_states.shape[2]
    query_states = query_states.view(b, hk, -1, d)
    score = torch.einsum("bhgd,bshd->bhgs", query_states, key_states)
    score = score / math.sqrt(d)
    score = score.sum(dim=2).to(query_states.dtype)
    return score


torch.cuda.set_device(0)
torch.manual_seed(42)
device = "cuda"
dtype = torch.float16
B = 64
HQ = 32
HK = 8
SEQ = 8000
D = 32

query_states = torch.randn((B, 1, HQ, D), dtype=dtype, device=device)
key_states = torch.randn((B, SEQ * 2, HK, D), dtype=dtype, device=device)
head_mask = torch.zeros((B * HK, ), dtype=torch.bool, device=device)
head_id = torch.arange(0, B * HK, 40, device=device)
head_mask[head_id] = True

torch_out = torch_loki_score(query_states, key_states, SEQ)
triton_out = loki_score(query_states, key_states, SEQ, head_mask)

len = triton_out.shape[-1]

torch_out = torch_out.view(-1, len)
torch_out_ture = torch_out[head_mask]

triton_out = triton_out.view(-1, len)
triton_out_ture = triton_out[head_mask]

print((torch_out_ture - triton_out_ture).abs().max())

bench(partial(torch_loki_score, query_states, key_states, SEQ))
time_cost = bench(partial(loki_score, query_states, key_states, SEQ, head_mask))


print("bandwidth",
      key_states.numel() * key_states.element_size() * (head_mask.sum() /
                                                           head_mask.numel()) /
      1024 / 1024 / 1024 / (time_cost) * 1000)