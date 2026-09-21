# Llama TP2 Kernel Enablement

The Llama 3 8B TP2 decode path has 16 local query heads and 4 local KV heads.
The original `HEAD_SWITCH` in `sgl-kernel/csrc/kvlib/ham_dist.cu` supported 20
query heads for Qwen TP2 but omitted 16, causing a fatal abort.

The fix adds the `NumHead=16` branch. The existing `NumKVHead=4` and
`KVGroup=4` branches already cover the rest of the Llama shape.

The first full editable build attempts were discarded because a fresh CMake
configuration tried to fetch CUTLASS over an unavailable network and some
in-tree dependency directories were incomplete. Configuration was completed
with the repository's prior successful dependency snapshot, and only the A40
runtime target was built:

```bash
export PATH=/jhe/sglang-litecache/.venv-py313/bin:$PATH
cmake --build /jhe/sglang-litecache/sgl-kernel/build \
  --target common_ops_sm100_build --config Release -j 8
```

The resulting `sgl-kernel/build/sm100/common_ops.abi3.so` was installed into
the editable source package at
`sgl-kernel/python/sgl_kernel/sm100/common_ops.abi3.so`. The old binary remains
beside it as `common_ops.abi3.so.before-numhead16`.

```text
old SHA256 0dfb5662193936e5ce79b98123c27d26333ffc905bc56b9d550eed7169f6bb85
new SHA256 062cefe4deeb3316d9d422d54eeafbcb0451a4c0166cd45b6221dc74ed3fc981
```

A direct CUDA smoke test invoked `static_hamming_score_mask` with shapes
`key=[1,128,4,8]` and `query=[1,1,16,8]`, synchronized successfully, and
verified finite output. The subsequent full 256K Llama profile and paired
benchmark exercised the same specialization without kernel errors.
