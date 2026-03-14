import triton
import triton.language as tl


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_buffer,
    V_buffer,
    Att_Out,
    Att_Lse,
    kv_seq_len,
    num_kv_splits,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_bufb,
    stride_bufs,
    stride_bufh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    KV_GROUP_NUM: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
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
    cur_head = cur_head_id * VALID_BLOCK_HGROUP + tl.arange(0, BLOCK_HGROUP) # [16, ]
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_HGROUP
    mask_h = mask_h & (cur_head < Q_HEAD_NUM)

    offs_d = tl.arange(0, BLOCK_DIM)
    mask_d = offs_d < HEAD_DIM

    cur_batch_seq_len = tl.load(kv_seq_len)
    kv_splits = tl.load(num_kv_splits)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :] # [16, BLOCK_DIM]
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), BLOCK_SEQ) * BLOCK_SEQ
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_HGROUP], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_HGROUP], dtype=tl.float32)
    acc = tl.zeros([BLOCK_HGROUP, BLOCK_DIM], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_SEQ):
            offs_n = start_n + tl.arange(0, BLOCK_SEQ)
            offs_buf_k = (
                cur_batch * stride_bufb
                + offs_n[None, :] * stride_bufs
                + cur_kv_head * stride_bufh
                + offs_d[:, None]
            ) # [BLOCK_DIM, BLOCK_SEQ]
            k = tl.load(
                K_buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype)) # [BLOCK_HGROUP, BLOCK_SEQ]
            qk *= sm_scale
            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                cur_batch * stride_bufb
                + offs_n[:, None] * stride_bufs
                + cur_kv_head * stride_bufh
                + offs_d[None, :]
            ) # [BLOCK_SEQ, BLOCK_DIM]
            v = tl.load(
                V_buffer + offs_buf_v,
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

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_d[None, :]
        ) # [BLOCK_HGROUP, BLOCK_DIM]
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_d[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // HEAD_DIM # [16, ]
        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,               # [b, 1, hq, d]
    k_buffer,        # [b, s, h, d]
    v_buffer,        # [b, s, h, d]
    att_out,         # [b, hq, split, d]
    att_lse,         # [b, hq, split]
    kv_seq_len,      # [1]
    num_kv_splits,   # [1]
    max_kv_splits,   # const
    sm_scale,        # const
):
    HEAD_DIM = k_buffer.shape[-1]
    BLOCK_DIM = triton.next_power_of_2(HEAD_DIM)

    batch, Q_HEAD_NUM = q.shape[0], q.shape[2]
    KV_GROUP_NUM = q.shape[2] // k_buffer.shape[2]
    BLOCK_SEQ = 32
    BLOCK_HGROUP = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(Q_HEAD_NUM, min(BLOCK_HGROUP, KV_GROUP_NUM)),
        MAX_KV_SPLITS,
    )

    num_stages = 2
    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        att_out,
        att_lse,
        #
        kv_seq_len,
        num_kv_splits,
        sm_scale,
        #
        q.stride(0),        # bs = b (s = 1)
        q.stride(2),        # h
        #
        k_buffer.stride(0), # b
        k_buffer.stride(1), # s
        k_buffer.stride(2), # h
        #
        att_out.stride(0),  # b
        att_out.stride(1),  # h
        att_out.stride(2),  # split
        #
        KV_GROUP_NUM=KV_GROUP_NUM,
        Q_HEAD_NUM=Q_HEAD_NUM,
        HEAD_DIM=HEAD_DIM,
        #
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_HGROUP=BLOCK_HGROUP,
        #
        num_warps=4,
        num_stages=num_stages,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_o,
    Mid_lse,
    O,
    kv_seqlen,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    HEAD_DIM: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_seqlen)
    kv_splits = tl.load(num_kv_splits)

    offs_d = tl.arange(0, BLOCK_DIM)
    mask_d = offs_d < HEAD_DIM

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DIM], dtype=tl.float32)

    offs_mo = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d # [BLOCK_DIM, ]
    offs_mlse = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // HEAD_DIM
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), BLOCK_SEQ) * BLOCK_SEQ
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_o + offs_mo + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            ) # [BLOCK_DIM, ]
            tlogic = tl.load(Mid_lse + offs_mlse + split_kv_id * stride_mid_os // HEAD_DIM)
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
    logits,           # [b, h, split, d]
    lse,              # [b, h, split]
    o,                # [b, 1, h, d]
    kv_seqlen,        # [1, ]
    num_kv_splits,    # [1, ]
    max_kv_splits,    # const
):
    batch, HEAD_NUM, HEAD_DIM = o.shape[0], o.shape[2], o.shape[3]
    BLOCK_DIM = triton.next_power_of_2(HEAD_DIM)
    BLOCK_SEQ = 32

    grid = (batch, HEAD_NUM)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        #
        kv_seqlen,
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
        MAX_KV_SPLITS=max_kv_splits,
        #
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DIM=BLOCK_DIM,
        #
        num_warps=4,
        num_stages=2,
    )


def decode_attention_fwd_grouped_split(
    q,
    k_buffer,
    v_buffer,
    attn_output,
    intermediate_attn_logits,
    intermediate_attn_lse,
    kv_seqlen,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        intermediate_attn_logits,
        intermediate_attn_lse,
        kv_seqlen,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
    )
    _decode_softmax_reducev_fwd(
        intermediate_attn_logits,
        intermediate_attn_lse,
        attn_output,
        kv_seqlen,
        num_kv_splits,
        max_kv_splits,
    )
