import torch
import sgl_kernel.kvlib as capi
from functools import partial


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


def torch_real_indices(topk_indices, gather_hid, h, bstrd, sstrd, hstrd):
    # result = torch.empty_like(topk_indices, device="cpu", dtype=torch.int64)
    boffset = gather_hid // h * bstrd
    hoffset = gather_hid % h * hstrd
    soffset = topk_indices[gather_hid, :] * sstrd
    result = boffset[:, None] + hoffset[:, None] + soffset
    return result.cpu()


B = 1
H = 4
K = 12800
D = 128
S = 128000
GH = B * H // 2

layer_idx = 17
indices = torch.randint(0, S, (
    B * H,
    K,
), device="cuda", dtype=torch.int32)
real_indices = torch.full((B * H, 2 * K),
                          -1,
                          device="cpu",
                          dtype=torch.int64,
                          pin_memory=True)
launch_flag = torch.full((3, ),
                         -1,
                         device="cpu",
                         dtype=torch.int32,
                         pin_memory=True)
cpu_mask = torch.zeros((B * H, ),
                       dtype=torch.bool,
                       device="cpu",
                       pin_memory=True)
gpu_mask = torch.zeros((B * H, ), dtype=torch.bool, device='cuda')
index = torch.arange(0, B * H, 2, device='cuda')
gpu_mask[index] = True
gpu_hid = torch.nonzero(gpu_mask).view(-1)

capi.real_indices_and_launch_prefetch(
    indices, gpu_mask, real_indices, launch_flag, cpu_mask, S, B, H, layer_idx
)
torch_output = torch_real_indices(indices, gpu_hid, H, S * H, H, 1)
assert torch.equal(torch_output, real_indices[gpu_mask.cpu(), :K])
print(cpu_mask)
print(launch_flag)

bench(partial(torch_real_indices, indices, gpu_hid, H, S * H, H, 1))

gpu_mask[:] = True
print("#gather heads", gpu_mask.sum().item())
bench(
    partial(capi.real_indices_and_launch_prefetch, indices, gpu_mask, real_indices, launch_flag, cpu_mask, S, B, H, layer_idx))

gpu_mask[:] = False
gpu_mask[index] = True
print("#gather heads", gpu_mask.sum().item())
bench(
    partial(capi.real_indices_and_launch_prefetch, indices, gpu_mask, real_indices, launch_flag, cpu_mask, S, B, H, layer_idx))
