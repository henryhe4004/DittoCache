import torch
import time
import numpy as np
import sgl_kernel.kvlib as capi

B = 16
H = 8
S = 4000
K = 512
R = 32
D = 128

device = "cuda"
dtype = torch.float16

key_states = torch.zeros((B, 1, H, D), dtype=dtype, device=device)
value_states = torch.zeros((B, 1, H, D), dtype=dtype, device=device)
cpu_head_index = torch.randperm(H, device=device)[:H // 2].sort().values.long()

cpu_cache = torch.zeros((2, B, S, H, D),
                        dtype=dtype,
                        device="cpu",
                        pin_memory=True)
gpu_buffer = torch.zeros((2, B, K + R, H // 2, D), dtype=dtype, device="cuda")
ready_flags = torch.ones((B * H, ), dtype=bool, device="cpu", pin_memory=True)
gpu_insert_pos = K
cpu_insert_pos = 800

capi.decode_append_offload_wait(
    key_states,
    value_states,
    gpu_buffer,
    cpu_cache,
    gpu_insert_pos,
    cpu_insert_pos,
    ready_flags,
    cpu_head_index,
)

assert torch.equal(
    cpu_cache[0, :, cpu_insert_pos:cpu_insert_pos + 1, :, :].cuda(),
    key_states)
assert torch.equal(
    cpu_cache[1, :, cpu_insert_pos:cpu_insert_pos + 1, :, :].cuda(),
    value_states)
assert torch.equal(gpu_buffer[0, :, gpu_insert_pos:gpu_insert_pos + 1, :, :],
                   key_states[:, :, cpu_head_index, :])
assert torch.equal(gpu_buffer[1, :, gpu_insert_pos:gpu_insert_pos + 1, :, :],
                   value_states[:, :, cpu_head_index, :])

# warmup
for i in range(5):
    gpu_insert_pos = K
    cpu_insert_pos = 800 + i
    key_states = torch.zeros((B, 1, H, D), dtype=dtype, device=device)
    value_states = torch.zeros((B, 1, H, D), dtype=dtype, device=device)
    ready_flags = torch.ones((B * H, ), dtype=bool, device=device)
    capi.decode_append_offload_wait(
        key_states,
        value_states,
        gpu_buffer,
        cpu_cache,
        gpu_insert_pos,
        cpu_insert_pos,
        ready_flags,
        cpu_head_index,
    )
    gpu_insert_pos += 1
    if gpu_insert_pos == K + R:
        gpu_insert_pos = K

duration = 0
for i in range(100):
    gpu_insert_pos = K
    cpu_insert_pos = 800 + i
    key_states = torch.zeros((B, 1, H, D), dtype=dtype, device=device)
    value_states = torch.zeros((B, 1, H, D), dtype=dtype, device=device)
    ready_flags = torch.ones((B * H, ), dtype=bool, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    capi.decode_append_offload_wait(
        key_states,
        value_states,
        gpu_buffer,
        cpu_cache,
        gpu_insert_pos,
        cpu_insert_pos,
        ready_flags,
        cpu_head_index,
    )
    torch.cuda.synchronize()
    t1 = time.time()
    duration += t1 - t0
    gpu_insert_pos += 1
    if gpu_insert_pos == K + R:
        gpu_insert_pos = K

print(duration * 1000 / 100)
