import torch
import time
import os
import sgl_kernel.kvlib as capi
from sglang.litecache.kvcache_offloading import create_aligned_cuda_tensor

B = 16
S = 128000
K = int(S * 0.1)
R = 69
H = 8
D = 128
layer = 2
num_iters = int(os.getenv("KVLIB_GDR_ITERS", "200"))


def torch_gather(cpu_data, real_indices):
    # cpu_data: [2, b, s, h, d]
    # cpu_indices: [b * h, k]
    d = cpu_data.shape[-1]
    h = real_indices.shape[0]
    cpu_data = cpu_data.view(2, -1, d)
    gathered_data = cpu_data[:, real_indices.view(-1), :]
    return gathered_data.view(2, h, -1, d)


def torch_real_indices(topk_indices, gather_hid, h, bstrd, sstrd, hstrd):
    # result = torch.empty_like(topk_indices, device="cpu", dtype=torch.int64)
    boffset = gather_hid // h * bstrd
    hoffset = gather_hid % h * hstrd
    soffset = topk_indices[gather_hid, :] * sstrd
    result = boffset[:, None] + hoffset[:, None] + soffset
    return result.cpu()


device = "cuda"
dtype = torch.float16
pagesize = 65536

cpu_data = [
    torch.randn((2, B, S, H, D), dtype=dtype, device="cpu", pin_memory=True)
    for _ in range(layer)
]
cpu_indices_buffer = torch.full((B * H, K),
                                -1,
                                dtype=torch.int64,
                                device="cpu",
                                pin_memory=True)
launch_flags = torch.full((6 + B * H, ),
                          -1,
                          dtype=torch.int32,
                          device="cpu",
                          pin_memory=True)
launch_flags[3] = S
launch_flags[4] = R + K
launch_flags[5] = K
launch_flags[6:] = K
ready_flags = [
    torch.full((B * H, ), 0, dtype=torch.bool, device="cpu", pin_memory=True)
    for _ in range(layer)
]

# num_gpu_buffer_heads = torch.randint(1, H, (layer, ), device="cpu").tolist()
num_gpu_buffer_heads = [H // 2 for _ in range(layer)]
mixed_head_index = []
gpu_head_mask = []
gpu_buffer_raw = []
gpu_buffer = []
for l in range(layer):
    mask = torch.zeros((H, ), dtype=torch.bool, device=device)
    hid = torch.sort(torch.randperm(H)[:H // 2]).values.cuda()
    mask[hid] = True
    gpu_head_mask.append(mask)

    mixed_index = torch.zeros((H, ), dtype=torch.int64, device="cpu")
    mixed_index[mask] = torch.arange(0, H // 2, device="cpu")
    mixed_index[~mask] = torch.arange(0, H // 2, device="cpu")
    mixed_head_index.append(mixed_index.int())

    raw, aligned = create_aligned_cuda_tensor(
        2 * B * (R + K) * num_gpu_buffer_heads[l] * D, dtype, device, pagesize)
    gpu_buffer_raw.append(raw)
    gpu_buffer.append(aligned)
torch.cuda.synchronize()

cpu_gather_engine = capi.CPUGatherEngineV3(4,
                                                         cpu_data,
                                                         gpu_buffer,
                                                         mixed_head_index,
                                                         num_gpu_buffer_heads,
                                                         cpu_indices_buffer,
                                                         launch_flags,
                                                         ready_flags,
                                                         B,
                                                         R,
                                                         H,
                                                         D,
                                                         debug=False)

for _ in range(num_iters):
    for layer_idx in range(layer):
        gpu_gather_mask = torch.zeros((B * H, ),
                                      dtype=torch.bool,
                                      device='cuda')
        gpu_gather_hid = torch.sort(
            torch.randperm(gpu_gather_mask.numel())[:gpu_gather_mask.numel() //
                                                    2]).values.cuda()
        gpu_gather_mask[gpu_gather_hid] = True
        gpu_gather_mask = gpu_gather_mask.view(B, H)
        gpu_gather_mask[:, gpu_head_mask[layer_idx]] = False
        gpu_gather_mask = gpu_gather_mask.view(-1)
        gpu_gather_hid = torch.nonzero(gpu_gather_mask).view(-1)

        gpu_indices = torch.randint(0,
                                    S, (B * H, K),
                                    dtype=torch.int32,
                                    device=device)

        launch_flags[3] = S
        launch_flags[4] = R + K
        launch_flags[5] = K
        launch_flags[6:] = K

        torch.cuda.synchronize()
        tic = time.perf_counter()
        capi.real_indices_and_launch_prefetch(
            gpu_indices, gpu_gather_mask, cpu_indices_buffer, launch_flags,
            ready_flags[layer_idx], S, B, H, layer_idx)
        capi.wait_kv_data(ready_flags[layer_idx], B, H)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        duration = toc - tic

        moved_bytes = gpu_gather_hid.numel() * K * 2 * D * dtype.itemsize
        print(
            f"Layer {layer_idx} Equivalent Bandwidth: {moved_bytes / 1024 / 1024 / 1024 / duration} GB/s"
        )
