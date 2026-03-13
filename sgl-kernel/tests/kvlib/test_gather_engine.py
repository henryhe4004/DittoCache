import time
import torch
import sgl_kernel.kvlib as capi
from myTransformer.cache.kvcache_offloading import create_aligned_cuda_tensor


GPU_PAGE_SIZE = 65536


def torch_real_indices(topk_indices, gather_hid, h, bstrd, sstrd, hstrd):
    # result = torch.empty_like(topk_indices, device="cpu", dtype=torch.int64)
    boffset = gather_hid // h * bstrd
    hoffset = gather_hid % h * hstrd
    soffset = topk_indices[gather_hid, :] * sstrd
    result = boffset[:, None] + hoffset[:, None] + soffset
    return result.cpu()


def test_gather_engine():
    b = 4
    h = 8
    h_gpu = 0
    h_reuse = 0
    d = 128
    num_layers = 2
    dtype = torch.bfloat16
    device = "cuda"
    sink_recent_pad = 69

    for maxs in [32000]:
        maxk = int(maxs * 0.1)
        s = maxs // 2
        k = maxk // 2
        maxfetch = maxk - sink_recent_pad
        fetch = k - sink_recent_pad
        print(f"Test s={s} k+sink-recent={k} fetch_k={fetch}......")

        # CPU data
        cpu_indices_buffer = torch.full(
            (b, h, maxfetch),
            -1,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        gather_engine_metadta = torch.tensor(
            [-1, -1, b, maxs, maxk, maxfetch],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        ready_flags = [torch.zeros((b, h), dtype=torch.bool, device="cpu", pin_memory=True)
            for _ in range(num_layers)]
        cpu_data = [
            torch.randn(
                (2, b, maxs, h, d),
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            ) for _ in range(num_layers)
        ]

        # GPU data
        gpu_indices_buffer = torch.full(
            (b, h, maxk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        gpu_topk_length = torch.tensor([fetch], dtype=torch.int32, device=device)
        
        num_gpu_heads = []
        num_cpu_heads = []
        mixed_head_index = []
        gpu_head_mask = []
        gpu_hids = []
        cpu_hids = []
        gpu_buffer_raw = []
        gpu_buffer = []
        torch_outputs = []
        for l in range(num_layers):
            gpu_mask = torch.zeros((h, ), dtype=torch.bool, device=device)
            gpu_hid = torch.randperm(h)[:h_gpu].cuda()
            gpu_mask[gpu_hid] = True

            gpu_head_mask.append(gpu_mask)
            num_gpu_heads.append(gpu_hid.shape[0])
            num_cpu_heads.append(h - gpu_hid.shape[0])
            gpu_hids.append(torch.nonzero(gpu_mask).squeeze())
            cpu_hids.append(torch.nonzero(~gpu_mask).squeeze())
            
            mixed_index = torch.zeros((h, ), dtype=torch.int64, device="cpu")
            mixed_index[gpu_mask] = torch.arange(0, num_gpu_heads[l], device="cpu")
            mixed_index[~gpu_mask] = torch.arange(0, h - num_gpu_heads[l], device="cpu")
            mixed_head_index.append(mixed_index.int())
            
            raw, aligned = create_aligned_cuda_tensor(
                2 * b * maxk * num_cpu_heads[l] * d,
                dtype=dtype,
                device=device,
                pagesize=GPU_PAGE_SIZE,
            )
            gpu_buffer_raw.append(raw)
            aligned = aligned.view(2, b, maxk, num_cpu_heads[l], d)
            gpu_buffer.append(aligned)

            torch_outputs.append(torch.zeros((2, b, fetch, num_cpu_heads[l], d), dtype=dtype, device="cpu"))
        
        torch.cuda.synchronize()
        cpu_gather_engine = capi.CPUGatherEngineV3(
            16,
            cpu_data,
            gpu_buffer,
            mixed_head_index,
            num_cpu_heads,
            cpu_indices_buffer,
            gather_engine_metadta,
            ready_flags,
            b,
            sink_recent_pad,
            h,
            d,
            debug=False,
        )
        gpu_gather_mask = torch.ones((b, h), dtype=torch.bool, device=device)

        for i in range(10):
            for l in range(num_layers):
                gpu_gather_mask = gpu_gather_mask.view(b, h)
                gpu_gather_mask[:, cpu_hids[l]] = True
                gpu_gather_mask = gpu_gather_mask.view(-1)
                if h_reuse > 0:
                    reuse_idx = torch.randperm(num_cpu_heads[l] * b)[:h_reuse * b]
                    b_cpu_hids = cpu_hids[l].repeat(b).view(b, num_cpu_heads[l])
                    b_cpu_hids = b_cpu_hids + torch.arange(0, b, device=device)[:, None] * h
                    reuse_hid = b_cpu_hids.view(-1)[reuse_idx]
                    gpu_gather_mask[reuse_hid] = False

                raw_topk_index = torch.randint(0, s, (b, h, fetch), device="cuda", dtype=torch.int32)

                # prepare indices for prefetch
                gpu_indices_buffer = gpu_indices_buffer.view(-1, maxk)
                gpu_indices_buffer[gpu_gather_mask, :fetch] = raw_topk_index.view(-1, fetch)[gpu_gather_mask, :]

                # torch_real_indices
                torch.cuda.synchronize()

                capi.static_launch_prefetch(
                    gpu_indices_buffer,
                    gpu_gather_mask,
                    gpu_topk_length,
                    cpu_indices_buffer,
                    gather_engine_metadta,
                    ready_flags[l],
                    b, maxs, h, l,
                )

                torch.cuda.synchronize()
                tic = time.time()
                capi.wait_kv_data(ready_flags[l], b, h)
                torch.cuda.synchronize()
                toc = time.time()

                torch_real_indices_output = torch_real_indices(
                    raw_topk_index.view(-1, fetch).cpu(),
                    gpu_gather_mask.nonzero().view(-1).cpu(),
                    h,
                    cpu_data[l].stride(1) // d,
                    cpu_data[l].stride(2) // d,
                    cpu_data[l].stride(3) // d,
                )

                # torch gather
                for i in range(b):
                    for j in range(h):
                        if not gpu_gather_mask[i * h + j]:
                            continue
                        indices = raw_topk_index[i, j, :fetch].cpu().long()
                        torch_outputs[l][:, i, :, j, :] = cpu_data[l][:, i, indices, j, :]

                # 检查 real indices 是否计算正确
                assert torch.equal(cpu_indices_buffer.view(-1, cpu_indices_buffer.shape[-1])[gpu_gather_mask.cpu(), :fetch], torch_real_indices_output)

                # 检查 gather 结果是否正确
                for bid in range(b):
                    for hid in range(h):
                        if gpu_head_mask[l][hid]:
                            continue
                        if not gpu_gather_mask[bid * h + hid]:
                            continue
                        real_h = mixed_head_index[l][hid]
                        for s in range(fetch):
                            my_out = gpu_buffer[l][:, bid, sink_recent_pad + s, real_h, :].cpu()
                            torch_out = torch_outputs[l][:, bid, s, real_h, :]
                            assert torch.equal(my_out, torch_out), f"bid={bid}, hid={real_h}, s={s}, my_out={my_out[..., :5]}, torch_out={torch_out[..., :5]}"

                transfer_data_size = 2 * gpu_gather_mask.sum().item() * fetch * d * dtype.itemsize
                print(
                    f"Layer {l} Equivalent Bandwidth: {transfer_data_size / 1024**3 / (toc - tic)} GB/s"
                )

        del cpu_gather_engine

if __name__ == "__main__":
    test_gather_engine()
