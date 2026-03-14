import torch
import math
from functools import partial
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.jit_kernel.triton_kernels.attention import decode_attention_fwd_grouped_split


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


hq, hkv, d = 32, 8, 128
max_tokens= 1024000
b_lst = [1, 8, 16, 64]
for b in b_lst:
    max_s = max_tokens // b
    s_lst = [1000]
    while s_lst[-1] * 2 <= max_s:
        s_lst.append(s_lst[-1] * 2)
    for s in s_lst:
        print("-" * 40)
        print(f"Test batch size = {b}, seq len = {s}......")
        q = torch.randn(b, 1, hq, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, int(s * 1.5), hkv, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(b, int(s * 1.5), hkv, d, device="cuda", dtype=torch.bfloat16)

        scale = 1 / math.sqrt(d)

        ref_out = torch.empty_like(q)
        ref_out = flash_attn_with_kvcache(
            q,
            k[:, :s],
            v[:, :s],
            causal=True,
        )

        max_kv_splits = 20
        real_out = torch.empty_like(q)
        intermediate_attn_logits = torch.zeros(
            (b, hq, max_kv_splits, d), device="cuda", dtype=torch.bfloat16
        )
        intermediate_attn_lse = torch.zeros((b, hq, max_kv_splits), device="cuda", dtype=torch.bfloat16)
        kv_seqlen = torch.full((1,), s, device="cuda", dtype=torch.int32)
        num_kv_splits = torch.full((1,), 8, device="cuda", dtype=torch.int32)
        torch.cuda.synchronize()
        decode_attention_fwd_grouped_split(
            q,
            k,
            v,
            real_out,
            intermediate_attn_logits,
            intermediate_attn_lse,
            kv_seqlen,
            num_kv_splits,
            max_kv_splits,
            scale,
        )
        print(f"[Accuracy] max diff = {(real_out - ref_out).abs().max().item()}")
        print(f"[Performance] FA2 kernel:", end=" ")
        bench(
            partial(flash_attn_with_kvcache, q, k[:, :s], v[:, :s], causal=True))
        print(f"[Performance] our kernel:", end=" ")
        bench(
            partial(decode_attention_fwd_grouped_split,
                q,
                k,
                v,
                real_out,
                intermediate_attn_logits,
                intermediate_attn_lse,
                kv_seqlen,
                num_kv_splits,
                max_kv_splits,
                scale,
            )
        )
