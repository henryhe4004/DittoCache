import os
import time

import torch
from torch.utils.cpp_extension import load_inline

from sglang.litecache.kvcache_offloading import create_aligned_cuda_tensor


B = 16
S = 8000
K = int(S * 0.1)
R = 69
H = 8
D = 128
layer = 2

device = "cuda"
dtype = torch.float16
pagesize = 65536
num_iters = int(os.getenv("KVLIB_COPYENGINE_ITERS", "200"))
verify = os.getenv("KVLIB_COPYENGINE_VERIFY", "0") == "1"


copyengine_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void head_sparse_copyengine(
    torch::Tensor cpu_data,
    torch::Tensor gpu_buffer,
    torch::Tensor real_indices,
    torch::Tensor gather_hids,
    torch::Tensor mixed_head_index,
    int64_t sink_recent_budget,
    int64_t num_heads,
    int64_t head_dim,
    int64_t num_gpu_buffer_heads) {
  TORCH_CHECK(cpu_data.device().is_cpu(), "cpu_data must be on CPU");
  TORCH_CHECK(real_indices.device().is_cpu(), "real_indices must be on CPU");
  TORCH_CHECK(gather_hids.device().is_cpu(), "gather_hids must be on CPU");
  TORCH_CHECK(mixed_head_index.device().is_cpu(), "mixed_head_index must be on CPU");
  TORCH_CHECK(gpu_buffer.is_cuda(), "gpu_buffer must be on CUDA");
  TORCH_CHECK(cpu_data.is_contiguous(), "cpu_data must be contiguous");
  TORCH_CHECK(gpu_buffer.is_contiguous(), "gpu_buffer must be contiguous");
  TORCH_CHECK(real_indices.is_contiguous(), "real_indices must be contiguous");
  TORCH_CHECK(gather_hids.is_contiguous(), "gather_hids must be contiguous");
  TORCH_CHECK(mixed_head_index.is_contiguous(), "mixed_head_index must be contiguous");

  const int64_t num_gather_heads = gather_hids.size(0);
  const int64_t gather_length = real_indices.size(1);
  const size_t vector_bytes = head_dim * cpu_data.element_size();
  const int64_t gpu_cache_len = sink_recent_budget + gather_length;
  const size_t gpu_key_bytes =
      num_gather_heads == 0
          ? 0
          : (gpu_buffer.numel() * gpu_buffer.element_size()) / 2;

  const char* cpu_key_base = static_cast<const char*>(cpu_data.data_ptr());
  const char* cpu_value_base =
      cpu_key_base + (cpu_data.numel() * cpu_data.element_size()) / 2;
  char* gpu_key_base = static_cast<char*>(gpu_buffer.data_ptr());
  char* gpu_value_base = gpu_key_base + gpu_key_bytes;

  const int64_t* indices = real_indices.data_ptr<int64_t>();
  const int64_t* hids = gather_hids.data_ptr<int64_t>();
  const int32_t* dst_head_index = mixed_head_index.data_ptr<int32_t>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (int64_t i = 0; i < num_gather_heads; ++i) {
    const int64_t total_hid = hids[i];
    const int64_t bid = total_hid / num_heads;
    const int64_t hid = total_hid % num_heads;
    const int64_t dst_hid = dst_head_index[hid];

    for (int64_t k = 0; k < gather_length; ++k) {
      const int64_t src_vec_id = indices[i * gather_length + k];
      const size_t src_offset = src_vec_id * vector_bytes;
      const size_t dst_vec_id =
          ((bid * gpu_cache_len + sink_recent_budget + k) *
               num_gpu_buffer_heads +
           dst_hid);
      const size_t dst_offset = dst_vec_id * vector_bytes;

      cudaError_t err = cudaMemcpyAsync(
          gpu_key_base + dst_offset,
          cpu_key_base + src_offset,
          vector_bytes,
          cudaMemcpyHostToDevice,
          stream);
      TORCH_CHECK(err == cudaSuccess, "key cudaMemcpyAsync failed: ",
                  cudaGetErrorString(err));

      err = cudaMemcpyAsync(
          gpu_value_base + dst_offset,
          cpu_value_base + src_offset,
          vector_bytes,
          cudaMemcpyHostToDevice,
          stream);
      TORCH_CHECK(err == cudaSuccess, "value cudaMemcpyAsync failed: ",
                  cudaGetErrorString(err));
    }
  }
}

"""


copyengine = load_inline(
    name="kvlib_head_sparse_copyengine_ext",
    cpp_sources=copyengine_src,
    functions=["head_sparse_copyengine"],
    with_cuda=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


def torch_gather(cpu_data, real_indices):
    d = cpu_data.shape[-1]
    h = real_indices.shape[0]
    cpu_data = cpu_data.view(2, -1, d)
    gathered_data = cpu_data[:, real_indices.view(-1), :]
    return gathered_data.view(2, h, -1, d)


def torch_real_indices(topk_indices, gather_hid, h, bstrd, sstrd, hstrd):
    boffset = gather_hid // h * bstrd
    hoffset = gather_hid % h * hstrd
    soffset = topk_indices[gather_hid, :] * sstrd
    return (boffset[:, None] + hoffset[:, None] + soffset).cpu().contiguous()


cpu_data = [
    torch.randn((2, B, S, H, D), dtype=dtype, device="cpu", pin_memory=True)
    for _ in range(layer)
]

num_gpu_buffer_heads = [H // 2 for _ in range(layer)]
mixed_head_index = []
gpu_head_mask = []
gpu_buffer_raw = []
gpu_buffer = []

for l in range(layer):
    mask = torch.zeros((H,), dtype=torch.bool, device=device)
    hid = torch.sort(torch.randperm(H)[: H // 2]).values.cuda()
    mask[hid] = True
    gpu_head_mask.append(mask)

    mixed_index = torch.zeros((H,), dtype=torch.int64, device="cpu")
    mixed_index[mask.cpu()] = torch.arange(0, H // 2, device="cpu")
    mixed_index[~mask.cpu()] = torch.arange(0, H // 2, device="cpu")
    mixed_head_index.append(mixed_index.int().contiguous())

    raw, aligned = create_aligned_cuda_tensor(
        2 * B * (R + K) * num_gpu_buffer_heads[l] * D,
        dtype,
        device,
        pagesize,
    )
    gpu_buffer_raw.append(raw)
    gpu_buffer.append(aligned)

torch.cuda.synchronize()

for _ in range(num_iters):
    for layer_idx in range(layer):
        gpu_gather_mask = torch.zeros((B * H,), dtype=torch.bool, device=device)
        gpu_gather_hid = torch.sort(
            torch.randperm(gpu_gather_mask.numel(), device=device)[
                : gpu_gather_mask.numel() // 2
            ]
        ).values
        gpu_gather_mask[gpu_gather_hid] = True
        gpu_gather_mask = gpu_gather_mask.view(B, H)
        gpu_gather_mask[:, gpu_head_mask[layer_idx]] = False
        gpu_gather_mask = gpu_gather_mask.view(-1)
        gpu_gather_hid = torch.nonzero(gpu_gather_mask).view(-1)

        gpu_indices = torch.randint(
            0,
            S,
            (B * H, K),
            dtype=torch.int32,
            device=device,
        )
        real_indices = torch_real_indices(
            gpu_indices, gpu_gather_hid, H, S * H, H, 1
        )
        gather_hid_cpu = gpu_gather_hid.cpu().long().contiguous()

        torch.cuda.synchronize()
        tic = time.time()
        copyengine.head_sparse_copyengine(
            cpu_data[layer_idx],
            gpu_buffer[layer_idx],
            real_indices,
            gather_hid_cpu,
            mixed_head_index[layer_idx],
            R,
            H,
            D,
            num_gpu_buffer_heads[layer_idx],
        )
        torch.cuda.synchronize()
        toc = time.time()
        duration = toc - tic

        moved_bytes = real_indices.numel() * 2 * D * dtype.itemsize
        print(
            f"Layer {layer_idx} CopyEngine cudaMemcpyAsync Equivalent Bandwidth: "
            f"{moved_bytes / 1024 / 1024 / 1024 / duration} GB/s"
        )

        if verify:
            torch_output = torch_gather(cpu_data[layer_idx], real_indices)
            my_gpu_output = gpu_buffer[layer_idx].view(
                2, B, R + K, H // 2, D
            )[:, :, R:]
            for i in range(torch_output.shape[1]):
                total_hid = gather_hid_cpu[i].item()
                b, h = total_hid // H, total_hid % H
                torch.testing.assert_close(
                    my_gpu_output[:, b, :, mixed_head_index[layer_idx][h], :].cpu(),
                    torch_output[:, i, :, :],
                )
