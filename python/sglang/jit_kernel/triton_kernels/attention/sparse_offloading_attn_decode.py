import triton
import triton.language as tl
import math


@triton.jit
def _fwd_mix_kernel(
    Q, # [b, 1, h, d]
    Attn_Out, # [b, 1, h, d]
    stride_qbs,
    stride_qh,
    #
    k_cache,  # [b, max_seq_len, hkv_gpu, d]
    v_cache,  # [b, max_seq_len, hkv_gpu, d]
    topk_index,  # [b, **hkv**, topk_budget]
    topk_index_count,  # [1, ]
    stride_cacheb,
    stride_caches,
    stride_cacheh,
    stride_topk_indexb,
    stride_topk_indexh,
    #
    k_buffer,  # [b, max_buffer_len, hkv_cpu, d]
    v_buffer,  # [b, max_buffer_len, hkv_cpu, d]
    buffer_valid_count,  # [1, ]
    stride_bufb,
    stride_bufs,
    stride_bufh,
    #
    mask,  # [hkv]
    mixed_head_ids,  # [hkv]
    #
    sm_scale,  # scalar
    #
    KV_GROUP_NUM: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    #
    BLOCK_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HGROUP: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(KV_GROUP_NUM, BLOCK_HGROUP)

    if BLOCK_HGROUP < KV_GROUP_NUM:
        VALID_BLOCK_HGROUP: tl.constexpr = BLOCK_HGROUP
    else:
        VALID_BLOCK_HGROUP: tl.constexpr = KV_GROUP_NUM

    # is_on_gpu=True for topk, False for offloaded
    is_on_gpu = tl.load(mask + cur_kv_head)
    real_k_head_id = tl.load(mixed_head_ids + cur_kv_head)

    cur_head = cur_head_id * VALID_BLOCK_HGROUP + tl.arange(
        0, BLOCK_HGROUP)  # [16, ]
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_HGROUP
    mask_h = mask_h & (cur_head < Q_HEAD_NUM)

    offs_d = tl.arange(0, BLOCK_DIM)
    mask_d = offs_d < HEAD_DIM

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[
        None, :]  # [16, BLOCK_DIM]

    e_max = tl.zeros([BLOCK_HGROUP], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_HGROUP], dtype=tl.float32)
    acc = tl.zeros([BLOCK_HGROUP, BLOCK_DIM], dtype=tl.float32)

    q = tl.load(Q + offs_q,
                mask=(mask_h[:, None]) & (mask_d[None, :]),
                other=0.0)

    if is_on_gpu:
        if topk_index is not None:
            K_ptr = k_cache
            V_ptr = v_cache

            cur_batch_seq_len = tl.load(topk_index_count)
            seq_len_end = tl.cdiv(cur_batch_seq_len, BLOCK_SEQ) * BLOCK_SEQ

            topk_index_ptr = (topk_index + cur_batch * stride_topk_indexb + cur_kv_head * stride_topk_indexh)

            for start in range(0, seq_len_end, BLOCK_SEQ):
                tl.multiple_of(start, BLOCK_SEQ)

                offs_topk_indices = start + tl.arange(0, BLOCK_SEQ)
                topk_indices = tl.load(topk_index_ptr + offs_topk_indices,
                    mask=(offs_topk_indices < cur_batch_seq_len)) # [BLOCK_SEQ, ]

                offs_buf_k = (cur_batch * stride_cacheb +
                            topk_indices[None, :] * stride_caches +
                            real_k_head_id * stride_cacheh + offs_d[:, None]
                            )  # [BLOCK_DIM, BLOCK_SEQ]
                k = tl.load(
                    K_ptr + offs_buf_k,
                    mask=(offs_topk_indices[None, :] < cur_batch_seq_len) &
                    (mask_d[:, None]),
                    other=0.0,
                )
                qk = tl.dot(q, k.to(q.dtype))  # [BLOCK_HGROUP, BLOCK_SEQ]
                qk *= sm_scale
                qk = tl.where(
                    mask_h[:, None] &
                    (offs_topk_indices[None, :] < cur_batch_seq_len), qk,
                    float("-inf"))

                offs_buf_v = (cur_batch * stride_cacheb +
                            topk_indices[:, None] * stride_caches +
                            real_k_head_id * stride_cacheh + offs_d[None, :]
                            )  # [BLOCK_SEQ, BLOCK_DIM]
                v = tl.load(
                    V_ptr + offs_buf_v,
                    mask=(offs_topk_indices[:, None] < cur_batch_seq_len) &
                    (mask_d[None, :]),
                    other=0.0,
                )

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                acc *= re_scale[:, None]
                acc += tl.dot(p.to(v.dtype), v)

                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

    else:
        if buffer_valid_count is not None:
            K_ptr = k_buffer
            V_ptr = v_buffer

            cur_batch_seq_len = tl.load(buffer_valid_count)
            seq_len_end = tl.cdiv(cur_batch_seq_len, BLOCK_SEQ) * BLOCK_SEQ

            for start_n in range(0, seq_len_end, BLOCK_SEQ):
                tl.multiple_of(start_n, BLOCK_SEQ)
                offs_n = start_n + tl.arange(0, BLOCK_SEQ)
                offs_buf_k = (cur_batch * stride_bufb +
                            offs_n[None, :] * stride_bufs +
                            real_k_head_id * stride_bufh + offs_d[:, None]
                            )  # [BLOCK_DIM, BLOCK_SEQ]
                k = tl.load(
                    K_ptr + offs_buf_k,
                    mask=(offs_n[None, :] < cur_batch_seq_len) & (mask_d[:, None]),
                    other=0.0,
                )
                qk = tl.dot(q, k.to(q.dtype))  # [BLOCK_HGROUP, BLOCK_SEQ]
                qk *= sm_scale
                qk = tl.where(
                    mask_h[:, None] & (offs_n[None, :] < cur_batch_seq_len), qk,
                    float("-inf"))

                offs_buf_v = (cur_batch * stride_bufb +
                            offs_n[:, None] * stride_bufs +
                            real_k_head_id * stride_bufh + offs_d[None, :]
                            )  # [BLOCK_SEQ, BLOCK_DIM]
                v = tl.load(
                    V_ptr + offs_buf_v,
                    mask=(offs_n[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
                    other=0.0,
                )

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                acc *= re_scale[:, None]
                acc += tl.dot(p.to(v.dtype), v)

                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

    tl.store(
        Attn_Out + offs_q,
        acc / e_sum[:, None],
        mask=(mask_h[:, None]) & (mask_d[None, :]),
    )


def decode_mixed_attention_fwd_grouped(
    q,  # [b, 1, h, d]
    attn_out,  # [b, 1, h, d]
    #
    k_cache,  # [b, max_seq, hkv_gpu, d]
    v_cache,  # [b, max_seq, hkv_gpu, d]
    topk_index,  # [b, **hkv**, topk_budget]
    topk_index_count,  # [1,]
    #
    k_buffer,  # [b, topk_budget, hkv_offloaded, d]
    v_buffer,  # [b, topk_budget, hkv_offloaded, d]
    buffer_valid_count,  # [1,]
    #
    mask,  # [hkv]
    mixed_head_ids,  # [hkv]
    #
    sm_scale,  # scalar  
):
    HEAD_DIM = q.shape[-1]
    BLOCK_DIM = triton.next_power_of_2(HEAD_DIM)

    batch, Q_HEAD_NUM = q.shape[0], q.shape[2]
    K_HEAD_NUM = mask.shape[0]
    KV_GROUP_NUM = Q_HEAD_NUM // K_HEAD_NUM
    BLOCK_SEQ = 32
    BLOCK_HGROUP = 16
    grid = (
        batch,
        triton.cdiv(Q_HEAD_NUM, min(BLOCK_HGROUP, KV_GROUP_NUM)),
        1,
    )

    if k_cache is not None:
        stride_cacheb = k_cache.stride(0)  # b
        stride_caches = k_cache.stride(1)  # s
        stride_cacheh = k_cache.stride(2)  # hkv
        stride_topk_indexb = topk_index.stride(0)  # b
        stride_topk_indexh = topk_index.stride(1)  # hkv
    else:
        stride_cacheb = -9999
        stride_caches = -9999
        stride_cacheh = -9999
        stride_topk_indexb = -9999
        stride_topk_indexh = -9999

    if k_buffer is not None:
        stride_bufb = k_buffer.stride(0)  # b
        stride_bufs = k_buffer.stride(1)  # s
        stride_bufh = k_buffer.stride(2)  # hkv
    else:
        stride_bufb = -9999
        stride_bufs = -9999
        stride_bufh = -9999

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    num_stages = 4
    _fwd_mix_kernel[grid](
        q,
        attn_out,
        q.stride(0),
        q.stride(2),
        #
        k_cache,
        v_cache,
        topk_index,
        topk_index_count,
        stride_cacheb,  # b
        stride_caches,  # s
        stride_cacheh,  # h
        stride_topk_indexb,  # b
        stride_topk_indexh,  # h
        #
        k_buffer,
        v_buffer,
        buffer_valid_count,
        stride_bufb,  # b
        stride_bufs,  # s
        stride_bufh,  # h
        #
        mask,
        mixed_head_ids,
        #
        sm_scale,      
        #
        KV_GROUP_NUM,
        Q_HEAD_NUM,
        HEAD_DIM,
        #
        BLOCK_DIM,
        BLOCK_SEQ,
        BLOCK_HGROUP,
        num_stages=num_stages,
    )

