import math
import triton
import triton.language as tl


@triton.jit
def _fwd_grouped_kernel(
    Q,
    K_buffer,
    V_buffer,
    Att_Out,
    #
    kv_seq_len,
    sm_scale,
    #
    stride_qbs,
    stride_qh,
    #
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

    if BLOCK_HGROUP < KV_GROUP_NUM:
        VALID_BLOCK_HGROUP: tl.constexpr = BLOCK_HGROUP
    else:
        VALID_BLOCK_HGROUP: tl.constexpr = KV_GROUP_NUM
    cur_head = cur_head_id * VALID_BLOCK_HGROUP + tl.arange(0, BLOCK_HGROUP) # [16, ]
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_HGROUP
    mask_h = mask_h & (cur_head < Q_HEAD_NUM)

    offs_d = tl.arange(0, BLOCK_DIM)
    mask_d = offs_d < HEAD_DIM

    cur_batch_seq_len = tl.load(kv_seq_len + cur_batch)
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :] # [16, BLOCK_DIM]

    e_max = tl.zeros([BLOCK_HGROUP], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_HGROUP], dtype=tl.float32)
    acc = tl.zeros([BLOCK_HGROUP, BLOCK_DIM], dtype=tl.float32)

    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
    seq_len_end = tl.cdiv(cur_batch_seq_len, BLOCK_SEQ) * BLOCK_SEQ

    for start_n in range(0, seq_len_end, BLOCK_SEQ):
        tl.multiple_of(start_n, BLOCK_SEQ)
        offs_n = start_n + tl.arange(0, BLOCK_SEQ)
        offs_buf_k = (
            cur_batch * stride_bufb
            + offs_n[None, :] * stride_bufs
            + cur_kv_head * stride_bufh
            + offs_d[:, None]
        ) # [BLOCK_DIM, BLOCK_SEQ]
        k = tl.load(
            K_buffer + offs_buf_k,
            mask=(offs_n[None, :] < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )
        qk = tl.dot(q, k.to(q.dtype)) # [BLOCK_HGROUP, BLOCK_SEQ]
        qk *= sm_scale
        qk = tl.where(
            mask_h[:, None] & (offs_n[None, :] < cur_batch_seq_len), qk, float("-inf")
        )

        offs_buf_v = (
            cur_batch * stride_bufb
            + offs_n[:, None] * stride_bufs
            + cur_kv_head * stride_bufh
            + offs_d[None, :]
        ) # [BLOCK_SEQ, BLOCK_DIM]
        v = tl.load(
            V_buffer + offs_buf_v,
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
        Att_Out + offs_q,
        acc / e_sum[:, None],
        mask=(mask_h[:, None]) & (mask_d[None, :]),
    )


def decode_attention_fwd_grouped(
    q,               # [b, 1, hq, d]
    k_buffer,        # [b, s, h, d]
    v_buffer,        # [b, s, h, d]
    attn_output,     # [b, 1, hq, d]
    kv_seq_len,      # [b]
    sm_scale,        # const
):
    HEAD_DIM = k_buffer.shape[-1]
    BLOCK_DIM = triton.next_power_of_2(HEAD_DIM)

    batch, Q_HEAD_NUM = q.shape[0], q.shape[2]
    KV_GROUP_NUM = q.shape[2] // k_buffer.shape[2]
    BLOCK_SEQ = 32
    BLOCK_HGROUP = 16
    grid = (
        batch,
        triton.cdiv(Q_HEAD_NUM, min(BLOCK_HGROUP, KV_GROUP_NUM)),
        1,
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    num_stages = 2
    _fwd_grouped_kernel[grid](
        q,
        k_buffer,
        v_buffer,
        attn_output,
        #
        kv_seq_len,
        sm_scale,
        #
        q.stride(0),        # bs = b (s = 1)
        q.stride(2),        # h
        #
        k_buffer.stride(0), # b
        k_buffer.stride(1), # s
        k_buffer.stride(2), # h
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
