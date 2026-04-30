# test_cuda_fa.py 报错分析（不改算子）

## 测试在做什么

| 变量 | 形状 | 含义 |
|------|------|------|
| `query_states` | (32, **1**, **32**, 128) | batch=32，**seq_q=1**（decode），**32 个 q head**，head_dim=128 |
| `key_states`   | (32, **400**, **8**, 128) | batch=32，**seq_k=400**，**8 个 kv head**（GQA），head_dim=128 |
| `value_states` | (32, 400, 8, 128) | 同 key |

即：**decode 场景**（每样本 1 个 query 对 400 个 key/value） + **GQA**（32 q heads，8 kv heads，8|32）。

测试调用：

```python
flash_attn_func(q, k, v, softmax_scale=scale, return_attn_probs=True)
```

## 为什么会报错

报错来自底层：`RuntimeError: Number of heads in key/value must divide number of heads in query`。

- **从 head 数看**：8 的确整除 32，按理满足 “key/value 的 head 数整除 query 的 head 数”。
- **真正的问题**：测试的 **tensor 形状和用法** 与当前 `flash_attn_func` 的 **使用约定不一致**。

当前 `flash_attn_func` 的实现（用 `flash_attn_varlen_func` 做包装时）是按「**q 和 k 的 seq 长度相同**」来写的：

- 用 `seq_len = q.shape[1]` 作为「统一」的 seq 长；
- 构造 `cu_seqlens_q` / `cu_seqlens_k` 时都按这一个 `seq_len`；
- 把 k、v 按 `batch * seq_len` 展平。

于是对这份测试数据会变成：

- `seq_len = q.shape[1] = 1`；
- `cu_seqlens_q = [0,1,2,...,32]`，`cu_seqlens_k = [0,1,2,...,32]`；
- `q_flat = (32, 32, 128)`（32 个 q token，32 head）✓；
- `k_flat = k.reshape(32*1, -1, 128) = (32, 8, 128)`（只取了 k 的 **第 1 个** 时间步，共 32 个 token，8 head）。

也就是说：

1. **语义错误**：测试本意是「1 个 query 对 400 个 key」，但传入底层时变成了「1 个 query 对 1 个 key」，k/v 的 400 个时间步被 reshape 方式截掉了。
2. **varlen 与 head 检查**：底层 varlen 接口会根据 `cu_seqlens` 和实际 tensor 形状推断「每个序列的 head 数 / token 数」。这里 `cu_seqlens` 和真实的 (seq_q=1, seq_k=400) 不一致，且 k 被错误地展平成 (32, 8, 128)，容易触发内部对 head 数或 layout 的校验，从而报 “Number of heads in key/value must divide number of heads in query”（在错误的形状/解释下，整除关系可能被判成不满足）。

## 结论（只分析测试、不改算子）

- **测试的假设**：`flash_attn_func` 支持  
  - q 与 k **不同 seq 长度**（这里是 1 vs 400），  
  - 以及 **GQA**（q 32 head，k/v 8 head）。
- **当前算子假设**：包装层按「q 和 k **相同 seq 长度**」构造 varlen，没有区分 `seq_len_q` 和 `seq_len_k`。
- **矛盾点**：测试是 **decode + GQA**，当前实现是 **等长 + 同一套 cu_seqlens**，二者不匹配，导致传参错误并触发底层 head 数检查报错。

因此，报错根因是：**测试用例的用法（decode、q/k 不同长、GQA）与当前 `flash_attn_func` 的“等长 + 当前 cu_seqlens 构造方式”的约定不一致**；要消除报错，要么让算子支持「q/k 不同长 + GQA」，要么**改测试**以符合当前算子约定（例如改成 prefill：q/k 同长、同 head，或先对 k/v 做 repeat_kv 再调用等）。
