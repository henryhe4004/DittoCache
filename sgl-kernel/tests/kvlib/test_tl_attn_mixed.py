import torch
import math
from functools import partial
from flash_attn import flash_attn_with_kvcache
from myTransformer.kernels.attention import decode_mixed_attention_fwd_grouped


def bench(func):
    import time
    import numpy as np

    for i in range(5):
        func()

    torch.cuda.synchronize()

    # Graph capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(100):
            func()
            
    g.replay()
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    g.replay()
    torch.cuda.synchronize()
    t1 = time.time()
    print((t1 - t0) * 1000 / 100)


def test_full_gpu_cached():
    hq, hkv, d = 32, 8, 128
    max_tokens= 1024000
    b_lst = [1, 8, 16, 64]
    for b in b_lst:
        max_s = max_tokens // b
        s_lst = [1000]
        while s_lst[-1] * 2 <= max_s:
            s_lst.append(s_lst[-1] * 2)
        for s in s_lst:
            maxs = int(s * 1.5)
            maxk = int(maxs * 0.1)
            k = int(s * 0.1)
            print("-" * 40)
            print(f"Test batch size = {b}, seq len = {s}, topk = {k}......")

            q = torch.randn(b, 1, hq, d, device="cuda", dtype=torch.bfloat16)
            kcache = torch.randn(b, maxs, hkv, d, device="cuda", dtype=torch.bfloat16)
            vcache = torch.randn(b, maxs, hkv, d, device="cuda", dtype=torch.bfloat16)

            topk_index_buffer = torch.full((b, hkv, maxk), -1, device="cuda", dtype=torch.int32)
            topk_index_length = torch.tensor([k], device="cuda", dtype=torch.int32)
            topk_index = torch.randint(0, s, (b, hkv, k), device="cuda", dtype=torch.int64)
            topk_index_buffer[:, :, :k] = topk_index.int()

            cached_mask = torch.ones((hkv, ), device="cuda", dtype=torch.bool)
            mixed_hids = torch.full((hkv, ), -1, device="cuda", dtype=torch.int32)
            cached_num = int(torch.sum(cached_mask).item())
            bufferd_num = hkv - cached_num
            mixed_hids[cached_mask] = torch.arange(0, cached_num, device="cuda", dtype=torch.int32)
            mixed_hids[~cached_mask] = torch.arange(0, bufferd_num, device="cuda", dtype=torch.int32)

            scale = 1 / math.sqrt(d)

            # reference
            ref_out = torch.empty_like(q)
            ref_index = topk_index.transpose(-1, -2).reshape(b, k, hkv, 1).expand(-1, -1, -1, d)
            ref_input_k = torch.gather(kcache, 1, ref_index)
            ref_input_v = torch.gather(vcache, 1, ref_index)
            ref_out = flash_attn_with_kvcache(
                q,
                ref_input_k,
                ref_input_v,
                causal=True,
            )

            real_out = torch.empty_like(q)        
            decode_mixed_attention_fwd_grouped(
                q,
                real_out,
                kcache,
                vcache,
                topk_index_buffer,
                topk_index_length,
                None,
                None,
                None,
                cached_mask,
                mixed_hids,
                scale,
            )
            print(f"[Accuracy] max diff = {(real_out - ref_out).abs().max().item()}")
            print(f"[Performance] FA2 kernel:", end=" ")
            bench(
                partial(flash_attn_with_kvcache, q, ref_input_k, ref_input_v, causal=True))
            print(f"[Performance] our kernel:", end=" ")
            bench(
                partial(
                    decode_mixed_attention_fwd_grouped,
                    q,
                    real_out,
                    kcache,
                    vcache,
                    topk_index_buffer,
                    topk_index_length,
                    None,
                    None,
                    None,
                    cached_mask,
                    mixed_hids,
                    scale,
                )
            )


def test_full_gpu_buffered():
    hq, hkv, d = 32, 8, 128
    max_tokens= 1024000
    b_lst = [1, 8, 16, 64]
    for b in b_lst:
        max_s = max_tokens // b
        s_lst = [1000]
        while s_lst[-1] * 2 <= max_s:
            s_lst.append(s_lst[-1] * 2)
        for s in s_lst:
            maxs = int(s * 1.5)
            maxk = int(maxs * 0.1)
            k = int(s * 0.1)
            print("-" * 40)
            print(f"Test batch size = {b}, seq len = {s}, topk = {k}......")

            q = torch.randn(b, 1, hq, d, device="cuda", dtype=torch.bfloat16)

            kbuffer = torch.randn(b, maxk, hkv, d, device="cuda", dtype=torch.bfloat16)
            vbuffer = torch.randn(b, maxk, hkv, d, device="cuda", dtype=torch.bfloat16)
            buffer_length = torch.tensor([k], device="cuda", dtype=torch.int32)

            cached_mask = torch.zeros((hkv, ), device="cuda", dtype=torch.bool)
            mixed_hids = torch.full((hkv, ), -1, device="cuda", dtype=torch.int32)
            cached_num = int(torch.sum(cached_mask).item())
            bufferd_num = hkv - cached_num
            mixed_hids[cached_mask] = torch.arange(0, cached_num, device="cuda", dtype=torch.int32)
            mixed_hids[~cached_mask] = torch.arange(0, bufferd_num, device="cuda", dtype=torch.int32)

            scale = 1 / math.sqrt(d)

            # reference
            ref_out = torch.empty_like(q)
            ref_out = flash_attn_with_kvcache(
                q,
                kbuffer[:, :k],
                vbuffer[:, :k],
                causal=True,
            )

            real_out = torch.empty_like(q)
            decode_mixed_attention_fwd_grouped(
                q,
                real_out,
                None,
                None,
                None,
                None,
                kbuffer,
                vbuffer,
                buffer_length,
                cached_mask,
                mixed_hids,
                scale,
            )
            print(f"[Accuracy] max diff = {(real_out - ref_out).abs().max().item()}")
            print(f"[Performance] FA2 kernel:", end=" ")
            bench(
                partial(flash_attn_with_kvcache, q, kbuffer[:, :k], vbuffer[:, :k], causal=True))
            print(f"[Performance] our kernel:", end=" ")
            bench(
                partial(
                    decode_mixed_attention_fwd_grouped,
                    q,
                    real_out,
                    None,
                    None,
                    None,
                    None,
                    kbuffer,
                    vbuffer,
                    buffer_length,
                    cached_mask,
                    mixed_hids,
                    scale,
                )
            )


def test_mixed():
    hq, hkv, d = 32, 8, 128
    max_tokens= 1024000
    b_lst = [1, 8, 16, 64]
    for b in b_lst:
        max_s = max_tokens // b
        s_lst = [1000]
        while s_lst[-1] * 2 <= max_s:
            s_lst.append(s_lst[-1] * 2)
        for s in s_lst:
            maxs = int(s * 1.5)
            maxk = int(maxs * 0.1)
            k = int(s * 0.1)
            print("-" * 40)
            print(f"Test batch size = {b}, seq len = {s}, topk = {k}......")

            cached_num = hkv // 2
            buffered_num = hkv - cached_num
            cached_mask = torch.zeros((hkv, ), device="cuda", dtype=torch.bool)
            cached_hids = torch.randperm(hkv, device="cuda", dtype=torch.int64)[:cached_num].sort().values
            cached_mask[cached_hids] = True
            mixed_hids = torch.full((hkv, ), -1, device="cuda", dtype=torch.int32)
            mixed_hids[cached_mask] = torch.arange(0, cached_num, device="cuda", dtype=torch.int32)
            mixed_hids[~cached_mask] = torch.arange(0, buffered_num, device="cuda", dtype=torch.int32)

            q = torch.randn(b, 1, hq, d, device="cuda", dtype=torch.bfloat16)
            kcache = torch.randn(b, maxs, cached_num, d, device="cuda", dtype=torch.bfloat16)
            vcache = torch.randn(b, maxs, cached_num, d, device="cuda", dtype=torch.bfloat16)
            kbuffer = torch.randn(b, maxk, buffered_num, d, device="cuda", dtype=torch.bfloat16)
            vbuffer = torch.randn(b, maxk, buffered_num, d, device="cuda", dtype=torch.bfloat16)

            topk_index_buffer = torch.full((b, hkv, maxk), -1, device="cuda", dtype=torch.int32)
            topk_index_length = torch.tensor([k], device="cuda", dtype=torch.int32)
            topk_index = torch.randint(0, s, (b, hkv, k), device="cuda", dtype=torch.int64)
            topk_index_buffer[:, cached_mask, :k] = topk_index.int()[:, cached_mask, :]

            buffer_length = torch.tensor([k], device="cuda", dtype=torch.int32)

            scale = 1 / math.sqrt(d)

            # reference
            ref_out = torch.empty_like(q)
            ref_index = topk_index.transpose(-1, -2).reshape(b, k, hkv, 1).expand(-1, -1, -1, d)
            ref_input_k = torch.zeros(b, k, hkv, d, device="cuda", dtype=torch.bfloat16)
            ref_input_v = torch.zeros(b, k, hkv, d, device="cuda", dtype=torch.bfloat16)
            ref_cached_k = torch.gather(kcache, 1, ref_index[:, :, cached_mask, :])
            ref_cached_v = torch.gather(vcache, 1, ref_index[:, :, cached_mask, :])
            ref_input_k[:, :, cached_mask, :] = ref_cached_k
            ref_input_v[:, :, cached_mask, :] = ref_cached_v
            ref_input_k[:, :, ~cached_mask, :] = kbuffer[:, :k, ...]
            ref_input_v[:, :, ~cached_mask, :] = vbuffer[:, :k, ...]
            ref_out = flash_attn_with_kvcache(
                q,
                ref_input_k,
                ref_input_v,
                causal=True,
            )

            real_out = torch.empty_like(q)
            decode_mixed_attention_fwd_grouped(
                q,
                real_out,
                kcache,
                vcache,
                topk_index_buffer,
                topk_index_length,
                kbuffer,
                vbuffer,
                buffer_length,
                cached_mask,
                mixed_hids,
                scale,
            )
            print(f"[Accuracy] max diff = {(real_out - ref_out).abs().max().item()}")
            print(f"[Performance] FA2 kernel:", end=" ")
            bench(
                partial(flash_attn_with_kvcache, q, ref_input_k, ref_input_v, causal=True))
            print(f"[Performance] our kernel:", end=" ")
            bench(
                partial(
                    decode_mixed_attention_fwd_grouped,
                    q,
                    real_out,
                    kcache,
                    vcache,
                    topk_index_buffer,
                    topk_index_length,
                    kbuffer,
                    vbuffer,
                    buffer_length,
                    cached_mask,
                    mixed_hids,
                    scale,
                )
            )


if __name__ == "__main__":
    
    torch.cuda.set_device(7)
    
    print("-" * 40)
    print("Test all cached......")
    test_full_gpu_cached()

    print("-" * 40)
    print("Test all buffered......")
    test_full_gpu_buffered()
    
    print("-" * 40)
    print("Test all mixed......")
    test_mixed()
