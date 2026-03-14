import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel_stage2(
    Mid_o,
    Mid_lse,
    O,
    #
    mask,  # [hkv]
    mixed_head_ids,  # [hkv],
    topk_index_count, # [1]
    buffer_valid_count, # [1]
    #
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    HEAD_DIM: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    k_head_id = cur_head // GQA_GROUP_SIZE

    # cur_batch_seq_len = tl.load(kv_seqlen)
    on_gpu = tl.load(mask + k_head_id)
    real_k_head_id = tl.load(mixed_head_ids + k_head_id)

    cur_batch_seq_len = -1
    if (on_gpu == 1):
        if topk_index_count is not None:
            cur_batch_seq_len = tl.load(topk_index_count)
    else:
        if buffer_valid_count is not None:
            cur_batch_seq_len = tl.load(buffer_valid_count)

    kv_splits = tl.load(num_kv_splits)

    offs_d = tl.arange(0, BLOCK_DIM)
    mask_d = offs_d < HEAD_DIM

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DIM], dtype=tl.float32)

    offs_mo = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d  # [BLOCK_DIM, ]
    offs_mlse = (cur_batch * stride_mid_ob +
                 cur_head * stride_mid_oh) // HEAD_DIM
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), BLOCK_SEQ) * BLOCK_SEQ)

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                                  cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_o + offs_mo + split_kv_id * stride_mid_os,
                         mask=mask_d,
                         other=0.0)  # [BLOCK_DIM, ]
            tlogic = tl.load(Mid_lse + offs_mlse +
                             split_kv_id * stride_mid_os // HEAD_DIM)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
        logits,  # [b, h, split, d]
        lse,  # [b, h, split]
        o,  # [b, 1, h, d]
        mask,  # [hkv]
        mixed_head_ids,  # [hkv],
        topk_index_count,  # [1]
        buffer_valid_count,  # [1]
        num_kv_splits,  # [1, ]
        max_kv_splits,  # const
):
    batch, HEAD_NUM, HEAD_DIM = o.shape[0], o.shape[2], o.shape[3]
    BLOCK_DIM = triton.next_power_of_2(HEAD_DIM)
    BLOCK_SEQ = 32
    K_HEAD_NUM = mask.shape[0]

    grid = (batch, HEAD_NUM)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        #
        mask,  # [hkv]
        mixed_head_ids,  # [hkv],
        topk_index_count,
        buffer_valid_count,
        num_kv_splits,
        #
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        #
        o.stride(0),
        o.stride(2),
        #
        HEAD_DIM=HEAD_DIM,
        GQA_GROUP_SIZE=HEAD_NUM // K_HEAD_NUM,
        MAX_KV_SPLITS=max_kv_splits,
        #
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DIM=BLOCK_DIM,
        #
        num_warps=4,
        num_stages=4,
    )


@triton.jit
def _fwd_mix_split_kernel(
    Q,
    Attn_Out,
    Attn_LSE,
    k_cache,
    v_cache,
    topk_index,  # [b, hkv_gpu, topk_budget]
    topk_index_count,  # [1]
    k_buffer,
    v_buffer,
    buffer_valid_count,  # [1]
    mask,  # [hkv]
    mixed_head_ids,  # [hkv]
    num_kv_splits,  # [1]
    sm_scale,  # scalar
    #
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    #
    stride_qbs,
    stride_qh,
    # for topk
    stride_cacheb,
    stride_caches,
    stride_cacheh,
    stride_topk_indexb,
    stride_topk_indexh,
    # for offloaded
    stride_bufb,
    stride_bufs,
    stride_bufh,
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
    split_kv_id = tl.program_id(2)

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

    kv_splits = tl.load(num_kv_splits)

    # get metadata
    cur_batch_seq_len = -1
    if is_on_gpu:
        if topk_index_count is not None:
            cur_batch_seq_len = tl.load(topk_index_count)

    else:
        if buffer_valid_count is not None:
            # for offload
            cur_batch_seq_len = tl.load(buffer_valid_count)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), BLOCK_SEQ) * BLOCK_SEQ)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                              cur_batch_seq_len)

    if split_kv_end > split_kv_start:

        q = tl.load(Q + offs_q,
                    mask=(mask_h[:, None]) & (mask_d[None, :]),
                    other=0.0)

        if is_on_gpu:
            if topk_index_count is not None:

                K_ptr = k_cache
                V_ptr = v_cache

                seq_len_end = tl.cdiv(cur_batch_seq_len, BLOCK_SEQ) * BLOCK_SEQ

                # prefetch
                topk_ptrs = topk_index + cur_batch * stride_topk_indexb + cur_kv_head * stride_topk_indexh

                for start in range(0, seq_len_end, BLOCK_SEQ):
                    tl.multiple_of(start, BLOCK_SEQ)

                    offs_topk_indices = start + tl.arange(0, BLOCK_SEQ)
                    topk_indices = tl.load(topk_ptrs + offs_topk_indices,
                                        mask=(offs_topk_indices
                                                < cur_batch_seq_len))

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

                for start_n in range(split_kv_start, split_kv_end, BLOCK_SEQ):
                    tl.multiple_of(start_n, BLOCK_SEQ)
                    offs_n = start_n + tl.arange(0, BLOCK_SEQ)
                    offs_buf_k = (cur_batch * stride_bufb +
                                offs_n[None, :] * stride_bufs +
                                real_k_head_id * stride_bufh + offs_d[:, None]
                                )  # [BLOCK_DIM, BLOCK_SEQ]
                    k = tl.load(
                        K_ptr + offs_buf_k,
                        mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                        other=0.0,
                    )
                    qk = tl.dot(q, k.to(q.dtype))  # [BLOCK_HGROUP, BLOCK_SEQ]
                    qk *= sm_scale
                    qk = tl.where(
                        mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk,
                        float("-inf"))

                    offs_buf_v = (cur_batch * stride_bufb +
                                offs_n[:, None] * stride_bufs +
                                real_k_head_id * stride_bufh + offs_d[None, :]
                                )  # [BLOCK_SEQ, BLOCK_DIM]
                    v = tl.load(
                        V_ptr + offs_buf_v,
                        mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                        other=0.0,
                    )

                    n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                    re_scale = tl.exp(e_max - n_e_max)
                    p = tl.exp(qk - n_e_max[:, None])
                    acc *= re_scale[:, None]
                    acc += tl.dot(p.to(v.dtype), v)

                    e_sum = e_sum * re_scale + tl.sum(p, 1)
                    e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob +
                      cur_head[:, None] * stride_mid_oh +
                      split_kv_id * stride_mid_os + offs_d[None, :]
                      )  # [BLOCK_HGROUP, BLOCK_DIM]
        tl.store(
            Attn_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_d[None, :]),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh +
                        split_kv_id * stride_mid_os) // HEAD_DIM  # [16, ]
        tl.store(
            Attn_LSE + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_mixed_split_attn_stage1(
        q,  # [b, 1, h, d]
        att_out,  # [b, h, split, d]
        att_lse,  # [b, h, split]
        k_cache,  # [b, max_seq, hkv_gpu, d]
        v_cache,  # [b, max_seq, hkv_gpu, d]
        topk_index,  # [b, hkv_gpu, topk_budget]
        topk_index_count,  # [1]
        k_buffer,  # [b, topk_budget, hkv_offloaded, d]
        v_buffer,  # [b, topk_budget, hkv_offloaded, d]
        buffer_valid_count,  # [1]
        mask,  # [hkv]
        mixed_head_ids,  # [hkv]
        num_kv_splits,  # [1]
        max_kv_splits,  # const
        sm_scale=None,  # scalar  
):
    HEAD_DIM = q.shape[-1]
    BLOCK_DIM = triton.next_power_of_2(HEAD_DIM)

    batch, Q_HEAD_NUM = q.shape[0], q.shape[2]
    K_HEAD_NUM = mask.shape[0]
    KV_GROUP_NUM = Q_HEAD_NUM // K_HEAD_NUM
    BLOCK_SEQ = 32
    BLOCK_HGROUP = 16

    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(Q_HEAD_NUM, min(BLOCK_HGROUP, KV_GROUP_NUM)),
        MAX_KV_SPLITS,
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
        sm_scale = 1.0 / HEAD_DIM**0.5

    _fwd_mix_split_kernel[grid](
        q,
        att_out,
        att_lse,
        k_cache,
        v_cache,
        topk_index,
        topk_index_count,
        k_buffer,
        v_buffer,
        buffer_valid_count,
        mask,
        mixed_head_ids,
        num_kv_splits,
        sm_scale,
        #
        att_out.stride(0),  # b
        att_out.stride(1),  # h
        att_out.stride(2),  # split
        #
        q.stride(0),  # bs = b (s = 1)
        q.stride(2),  # h
        #
        stride_cacheb,  # b
        stride_caches,  # s
        stride_cacheh,  # hkv
        stride_topk_indexb,
        stride_topk_indexh,
        #
        stride_bufb,  # b
        stride_bufs,  # s
        stride_bufh,  # hkv
        #
        KV_GROUP_NUM,
        Q_HEAD_NUM,
        HEAD_DIM,
        BLOCK_DIM,
        BLOCK_SEQ,
        BLOCK_HGROUP,
        num_stages=4,
        num_warps=4,
    )


def decode_mixed_split_attention_fwd_grouped(
        q,  # [b, 1, h, d]
        att_out,
        intermediate_attn_logits,  # [b, h, split, d]
        intermediate_attn_lse,  # [b, h, split]
        k_cache,  # [b, max_seq, hkv_gpu, d]
        v_cache,  # [b, max_seq, hkv_gpu, d]
        topk_index,  # [b, hkv_gpu, topk_budget]
        topk_index_count,  # [1]
        k_buffer,  # [b, topk_budget, hkv_offloaded, d]
        v_buffer,  # [b, topk_budget, hkv_offloaded, d]
        buffer_valid_count,  # [1]
        mask,  # [hkv]
        mixed_head_ids,  # [hkv]
        num_kv_splits,  # [1]
        max_kv_splits,  # const
        sm_scale=None,  # scalar  
):
    _decode_mixed_split_attn_stage1(
        q,  # [b, 1, h, d]
        intermediate_attn_logits,  # [b, h, split, d]
        intermediate_attn_lse,  # [b, h, split]
        k_cache,  # [b, max_seq, hkv_gpu, d]
        v_cache,  # [b, max_seq, hkv_gpu, d]
        topk_index,  # [b, hkv_gpu, topk_budget]
        topk_index_count,  # [1]
        k_buffer,  # [b, topk_budget, hkv_offloaded, d]
        v_buffer,  # [b, topk_budget, hkv_offloaded, d]
        buffer_valid_count,  # [1]
        mask,  # [hkv]
        mixed_head_ids,  # [hkv]
        num_kv_splits,  # [1]
        max_kv_splits,  # const
        sm_scale,  # scalar  
    )
    _decode_softmax_reducev_fwd(
        intermediate_attn_logits,
        intermediate_attn_lse,
        att_out,
        mask,  # [hkv]
        mixed_head_ids,  # [hkv],
        topk_index_count,
        buffer_valid_count,
        num_kv_splits,
        max_kv_splits,
    )
