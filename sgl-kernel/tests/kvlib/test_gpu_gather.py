import torch
import sgl_kernel.kvlib as capi
import time
from functools import partial


def bench(func):
    for i in range(5):
        func()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        func()
    torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) / 100


def torch_gather(key, value, idx, head_ids):
    b, _, h, d = key.shape
    s = idx.shape[-1]
    idx = idx[:, head_ids, :]
    idx = idx.transpose(-1, -2).view(b, s, h, 1).long().expand(b, s, h, d)
    key = torch.gather(key, dim=1, index=idx)
    value = torch.gather(value, dim=1, index=idx)
    return key, value


B = 1
H = 4
HG = 2
S = 128000
D = 128
K = int(S * 0.1)
R = 69

key_states = torch.randn((B, S, HG, D), dtype=torch.float16, device="cuda")
value_states = torch.randn((B, S, HG, D), dtype=torch.float16, device="cuda")
key_buffer = torch.zeros((B, H, R + 2 * K, D),
                         dtype=torch.float16,
                         device="cuda")
value_buffer = torch.zeros((B, H, R + 2 * K, D),
                           dtype=torch.float16,
                           device="cuda")
head_ids = torch.randperm(H)[:HG].long().cuda()

gpu_indices = torch.randint(0, S, (B, H, K), dtype=torch.int32, device="cuda")
torch_key, torch_value = torch_gather(key_states, value_states, gpu_indices,
                                      head_ids)
capi.gather_gpu_kvcache(
    gpu_indices, key_states, value_states, key_buffer, value_buffer, head_ids, R
)

torch_key = torch_key.transpose(1, 2)
torch_value = torch_value.transpose(1, 2)

assert torch.equal(key_buffer[:, head_ids][:, :, R:R + K, :], torch_key)
assert torch.equal(value_buffer[:, head_ids][:, :, R:R + K, :], torch_value)

ours_time = bench(
    partial(capi.gather_gpu_kvcache, gpu_indices, key_states,
            value_states, key_buffer, value_buffer, head_ids, R))
datasize = gpu_indices.numel() * D * 2 * 2 / 1024 / 1024 / 1024
bandwidth = datasize / ours_time
print(f"Ours time: {ours_time * 1000:.3f} ms, bandwidth: {bandwidth:.3f} GB/s")
