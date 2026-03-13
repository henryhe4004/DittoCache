import torch
import math
from functools import partial
from myTransformer.cache.kernels.triton_quest_kernels import (
    quest_score, )


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
    
    cost_time = (t1 - t0) * 1000 / 100
    
    print(cost_time)
    
    return cost_time


def torch_quest_score(
    query_states: torch.Tensor,
    key_states_max: torch.Tensor,
    key_states_min: torch.Tensor,
    seq_len: int,
):
    b, _, hk, d = key_states_max.shape
    hq = query_states.shape[2]
    query_states = query_states.view(b, hk, -1, d)

    query_states_pos = torch.where(query_states >= 0, query_states, 0)
    query_states_neg = torch.where(query_states < 0, query_states, 0)

    score = torch.einsum("bhgd,bshd->bhgs", query_states_pos,
                         key_states_max[:, :seq_len, ...]) + torch.einsum(
                             "bhgd,bshd->bhgs", query_states_neg,
                             key_states_min[:, :seq_len, ...])
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
D = 128
BS = 8
NB = SEQ // BS

query_states = torch.randn((B, 1, HQ, D), dtype=dtype, device=device)
key_states = torch.randn((B, SEQ * 2, HK, D), dtype=dtype, device=device)
key_states = key_states.view(B, -1, BS, HK, D)
key_states_max = torch.max(key_states, dim=2).values
key_states_min = torch.min(key_states, dim=2).values
head_mask = torch.zeros((B * HK, ), dtype=torch.bool, device=device)
head_id = torch.arange(0, B * HK, 1, device=device)
head_mask[head_id] = True

torch_out = torch_quest_score(query_states, key_states_max, key_states_min, NB)

triton_out = quest_score(query_states, key_states_max, key_states_min, NB,
                         head_mask)
len = triton_out.shape[-1]

torch_out = torch_out.view(-1, len)
torch_out_ture = torch_out[head_mask]

triton_out = triton_out.view(-1, len)
triton_out_ture = triton_out[head_mask]

print((torch_out_ture - triton_out_ture).abs().max())

bench(
    partial(torch_quest_score, query_states, key_states_max, key_states_min,
            NB))
cost_time = bench(
    partial(quest_score, query_states, key_states_max, key_states_min, NB,
            head_mask))

# print(key_states_max.shape, key_states_max.dtype) # 0.24 GB

print("bandwith", 2 * key_states_max.numel() * key_states.element_size() / 1024 / 1024 / 1024 / (cost_time / 1000))