from transformers.models.llama.modeling_llama import repeat_kv
import torch
from functools import partial
import math
from flash_attn import flash_attn_func


def flash_attnention(q, k, v, scale):
    attn, lse, _ = flash_attn_func(q,
                                   k,
                                   v,
                                   softmax_scale=scale,
                                   return_attn_probs=True)
    return attn, lse


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
    print((t1 - t0) * 1000 * 1000 / 100)


torch.cuda.set_device(0)
torch.manual_seed(42)

batch_size = 32
num_heads = 32
num_kv_heads = 8
head_dim = 128
seq_len = 400

print(f"Batchsize {batch_size}, Seqlen {seq_len}")

query_states = torch.randn((batch_size, 1, num_heads, head_dim),
                           dtype=torch.float16,
                           device=torch.device("cuda"))
key_states = torch.randn((batch_size, seq_len, num_kv_heads, head_dim),
                         dtype=torch.float16,
                         device=torch.device("cuda"))
value_states = torch.randn((batch_size, seq_len, num_kv_heads, head_dim),
                           dtype=torch.float16,
                           device=torch.device("cuda"))
scale = 1 / math.sqrt(head_dim)

print(query_states.shape)
print(key_states.shape)

bench((partial(flash_attnention, query_states, key_states, value_states,
               scale)))
