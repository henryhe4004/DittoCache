#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include "cp_async.cuh"
#include "operator.h"

namespace kvlib {

template <typename T>
__global__ void CUDARealIndicesAndLaunchPrefetching(
    T* __restrict__ indices, bool* __restrict__ gpu_gather_mask,
    volatile int64_t* __restrict__ real_indices,
    int32_t* __restrict__ gather_flag,
    volatile bool* __restrict__ cpu_ready_mask, int64_t num_gather_heads,
    int64_t batch_size, int64_t num_head, int64_t cache_seq_len,
    int64_t src_len, int64_t dst_len, int64_t layer_idx) {
  const int head_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  // compute real indices and copy to cpu
  if (gpu_gather_mask[head_idx]) {
    T* src_ptr = indices + head_idx * src_len;
    // int64_t* dst_ptr = real_indices + head_idx * dst_len;
    const int32_t batch_id = head_idx / num_head;
    const int32_t head_id = head_idx % num_head;
    for (int i = tid; i < src_len; i += num_threads) {
      int64_t data =
          batch_id * num_head * cache_seq_len + head_id + src_ptr[i] * num_head;
      *(real_indices + head_idx * dst_len + i) = data;
      // __threadfence_system();
    }
  }

  // copy gather mask to cpu
  const int copy_mask_idx = blockIdx.x * blockDim.x + tid;
  if (copy_mask_idx < num_gather_heads) {
    cpu_ready_mask[copy_mask_idx] = !gpu_gather_mask[copy_mask_idx];
  }

  // copy metadata to cpu
  if (tid == 0 && head_idx == 0) {
    gather_flag[1] = src_len;
    gather_flag[2] = batch_size;
  }

  __threadfence_system();

  // // __nanosleep(1);

  // // notify CPU to launch prefetch
  // if (tid == 0 && head_idx == 0) {
  //   gather_flag[0] = layer_idx;
  // }
}

__global__ void SetReadyKernel(int* ready_flag, int src_len, int batch_size,
                               int layer_idx) {
  // ready_flag[1] = src_len;
  // ready_flag[2] = batch_size;
  // __threadfence_system();
  ready_flag[0] = layer_idx;
}

void RealInndicesAndLaunchPrefetching(torch::Tensor& indices,
                                      torch::Tensor& gpu_gather_mask,
                                      torch::Tensor& output,
                                      torch::Tensor& gather_flag,
                                      torch::Tensor& cpu_ready_mask,
                                      int64_t cache_seq_len, int64_t batch_size,
                                      int64_t num_heads, int64_t layer_idx) {
  // indices: [batchsize * num_head, k - 1], int32
  // gpu_gather_mask: [batchsize * num_head], bool
  // output: [batchsize * num_head, k], int64
  // gather_flag: [3, ], int32, 0 is layeridx, 1 is k, 2 is batchsize * num_head
  // cpu_ready_mask: [batchsize * num_head], bool

  int64_t num_gather_heads = indices.size(0);
  int64_t src_len = indices.size(1);
  int64_t dst_len = output.size(1);

  auto device = indices.device();
  int32_t device_id = device.index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  constexpr int num_threads = 128;
  dim3 block(num_threads);
  dim3 grid(num_gather_heads);
  CUDARealIndicesAndLaunchPrefetching<int32_t><<<grid, block, 0, stream>>>(
      indices.data_ptr<int32_t>(), gpu_gather_mask.data_ptr<bool>(),
      output.data_ptr<int64_t>(), gather_flag.data_ptr<int32_t>(),
      cpu_ready_mask.data_ptr<bool>(), num_gather_heads, batch_size, num_heads,
      cache_seq_len, src_len, dst_len, layer_idx);

  SetReadyKernel<<<1, 1>>>(gather_flag.data_ptr<int32_t>(), src_len, batch_size,
                           layer_idx);
}

template <typename T>
__global__ void CUDABlcokIdx2TokenIdxKernel(T* __restrict__ block_idx,
                                            T* __restrict__ token_idx,
                                            int block_size, int num_sink,
                                            int num_recent, int dst_length,
                                            int seq_length, int block_length) {
  int head_id = blockIdx.y;
  int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
  T* dst_ptr = token_idx + head_id * dst_length + seq_id;
  if (seq_id < dst_length) {
    if (seq_id < num_sink) {
      // sink
      *dst_ptr = seq_id;
    } else if (seq_id < dst_length - num_recent) {
      // middle
      seq_id -= num_sink;
      int block_id = seq_id / block_size;
      int in_block_seq_id = seq_id % block_size;
      T* src_ptr = block_idx + head_id * block_length + block_id;
      *dst_ptr = (*src_ptr) * block_size + in_block_seq_id + num_sink;
    } else {
      // recent
      *dst_ptr = seq_length - dst_length + seq_id;
    }
  } else {
    return;
  }
}

torch::Tensor BlockIdx2TokenIdx(torch::Tensor& block_idx, int64_t block_size,
                                int64_t num_sink, int64_t num_recent,
                                int64_t seq_length) {
  // block_idx: [b, h, nblocks], int32
  // output [b, h, num_sink + nblocks * block_size + num_recent], int32
  int64_t num_heads = block_idx.size(0) * block_idx.size(1);
  int64_t src_len = block_idx.size(2);
  int64_t dst_len = num_sink + src_len * block_size + num_recent;
  torch::Tensor token_idx = torch::empty(
      {block_idx.size(0), block_idx.size(1), dst_len}, block_idx.options());

  auto device = block_idx.device();
  int32_t device_id = device.index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  constexpr int num_threads = 128;
  int num_blocks = (dst_len + num_threads - 1) / num_threads;
  dim3 block(num_threads);
  dim3 grid(num_blocks, num_heads);
  CUDABlcokIdx2TokenIdxKernel<int32_t><<<grid, block, 0, stream>>>(
      block_idx.data_ptr<int32_t>(), token_idx.data_ptr<int32_t>(), block_size,
      num_sink, num_recent, dst_len, seq_length, src_len);

  return token_idx;
}

template <typename T>
__global__ void CUDABlcokIdx2TokenIdxHeadMaskKernel(
    T* __restrict__ block_idx, T* __restrict__ token_idx,
    bool* __restrict__ head_mask, int block_size, int num_sink, int num_recent,
    int dst_length, int seq_length, int block_length) {
  int head_id = blockIdx.y;
  if (head_mask[head_id]) {
    int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    T* dst_ptr = token_idx + head_id * dst_length + seq_id;
    if (seq_id < dst_length) {
      if (seq_id < num_sink) {
        // sink
        *dst_ptr = seq_id;
      } else if (seq_id < dst_length - num_recent) {
        // middle
        seq_id -= num_sink;
        int block_id = seq_id / block_size;
        int in_block_seq_id = seq_id % block_size;
        T* src_ptr = block_idx + head_id * block_length + block_id;
        *dst_ptr = (*src_ptr) * block_size + in_block_seq_id + num_sink;
      } else {
        // recent
        *dst_ptr = seq_length - dst_length + seq_id;
      }
    } else {
      return;
    }
  } else {
    return;
  }
}

torch::Tensor BlockIdx2TokenIdxHeadMask(torch::Tensor& block_idx,
                                        int64_t block_size, int64_t num_sink,
                                        int64_t num_recent, int64_t seq_length,
                                        torch::Tensor& head_mask) {
  // block_idx: [b, h, nblocks], int32
  // output [b, h, num_sink + nblocks * block_size + num_recent], int32
  int64_t num_heads = block_idx.size(0) * block_idx.size(1);
  int64_t src_len = block_idx.size(2);
  int64_t dst_len = num_sink + src_len * block_size + num_recent;
  torch::Tensor token_idx = torch::empty(
      {block_idx.size(0), block_idx.size(1), dst_len}, block_idx.options());

  auto device = block_idx.device();
  int32_t device_id = device.index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  constexpr int num_threads = 128;
  int num_blocks = (dst_len + num_threads - 1) / num_threads;
  dim3 block(num_threads);
  dim3 grid(num_blocks, num_heads);
  CUDABlcokIdx2TokenIdxHeadMaskKernel<int32_t><<<grid, block, 0, stream>>>(
      block_idx.data_ptr<int32_t>(), token_idx.data_ptr<int32_t>(),
      head_mask.data_ptr<bool>(), block_size, num_sink, num_recent, dst_len,
      seq_length, src_len);

  return token_idx;
}

template <typename T>
__global__ void CUDAAppendOffloadWait(
    T* __restrict__ key, T* __restrict__ value, T* __restrict__ gpu_dst,
    T* __restrict__ cpu_dst, int64_t* __restrict__ cpu_head_ids,
    const int32_t gpu_pos, const int32_t cpu_pos, volatile bool* ready_flags,
    const int32_t batch_size, const int32_t num_heads,
    const int32_t num_cpu_heads, const int32_t head_dim,
    const int64_t gpu_t_stride, const int64_t gpu_b_stride,
    const int64_t gpu_h_stride, const int64_t gpu_s_stride,
    const int64_t cpu_t_stride, const int64_t cpu_b_stride,
    const int64_t cpu_h_stride, const int64_t cpu_s_stride) {
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  int bsz_id = blockIdx.x;
  int head_id = blockIdx.y;

  T* input_ptr = nullptr;
  T* output_ptr = nullptr;

  if (head_id >= num_cpu_heads) {  // cpu
    head_id -= num_cpu_heads;
    int64_t src_offset = bsz_id * num_heads * head_dim + head_id * head_dim;

    if (warp_id == 0) {  // key
      int64_t dst_offset = bsz_id * cpu_b_stride + cpu_pos * cpu_s_stride +
                           head_id * cpu_h_stride;
      input_ptr = key + src_offset;
      output_ptr = cpu_dst + dst_offset;
    }

    else {  // value
      int64_t dst_offset = cpu_t_stride + bsz_id * cpu_b_stride +
                           cpu_pos * cpu_s_stride + head_id * cpu_h_stride;
      input_ptr = value + src_offset;
      output_ptr = cpu_dst + dst_offset;
    }
  }

  else {  // gpu
    int64_t input_head_id = cpu_head_ids[head_id];
    int64_t src_offset =
        bsz_id * num_heads * head_dim + input_head_id * head_dim;

    if (warp_id == 0) {  // key
      int64_t dst_offset = bsz_id * gpu_b_stride + gpu_pos * gpu_s_stride +
                           head_id * gpu_h_stride;
      input_ptr = key + src_offset;
      output_ptr = gpu_dst + dst_offset;
    }

    else {  // value
      int64_t dst_offset = gpu_t_stride + bsz_id * gpu_b_stride +
                           gpu_pos * gpu_s_stride + head_id * gpu_h_stride;
      input_ptr = value + src_offset;
      output_ptr = gpu_dst + dst_offset;
    }
  }

  for (int i = lane_id; i < head_dim; i += 32) {
    output_ptr[i] = input_ptr[i];
  }

  int total_hid = threadIdx.x + blockIdx.x * blockDim.x;
  if (total_hid < batch_size * num_heads && blockIdx.y == 0) {
    while (!ready_flags[total_hid]) {
      __nanosleep(1 * 1000);
    }
    // ready_flags[total_hid] = false;
  } else {
    return;
  }
}

void AppendOffloadWait(torch::Tensor& key_states, torch::Tensor& value_states,
                       torch::Tensor& gpu_kv_buffer,
                       torch::Tensor& cpu_kv_cache, int32_t gpu_append_pos,
                       int32_t cpu_append_pos, torch::Tensor& ready_flags,
                       torch::Tensor& cpu_head_ids) {
  // shape for gpu_cache is (2, bsz, num_kv_head, gpu_len, head_dim)
  // shape for cpu_cache is (2, bsz, cpu_len, num_kv_head, head_dim)
  // shape for key and value is (bsz, 1, num_kv_head, head_dim)

  int32_t bsz = key_states.size(0);
  int32_t num_heads = key_states.size(2);
  int32_t num_cpu_heads = cpu_head_ids.size(0);
  int32_t head_dim = key_states.size(3);

  int64_t gpu_t_stride = gpu_kv_buffer.stride(0);
  int64_t gpu_b_stride = gpu_kv_buffer.stride(1);
  int64_t gpu_h_stride = gpu_kv_buffer.stride(3);
  int64_t gpu_s_stride = gpu_kv_buffer.stride(2);

  int64_t cpu_t_stride = cpu_kv_cache.stride(0);
  int64_t cpu_b_stride = cpu_kv_cache.stride(1);
  int64_t cpu_h_stride = cpu_kv_cache.stride(3);
  int64_t cpu_s_stride = cpu_kv_cache.stride(2);

  constexpr int num_threads = 32 * 2;
  dim3 blk(num_threads);
  dim3 grid(bsz, num_cpu_heads + num_heads);
  auto device = key_states.device();
  int32_t device_id = device.index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  CUDAAppendOffloadWait<half><<<grid, blk, 0, stream>>>(
      (half*)key_states.data_ptr<at::Half>(),
      (half*)value_states.data_ptr<at::Half>(),
      (half*)gpu_kv_buffer.data_ptr<at::Half>(),
      (half*)cpu_kv_cache.data_ptr<at::Half>(),
      cpu_head_ids.data_ptr<int64_t>(), gpu_append_pos, cpu_append_pos,
      ready_flags.data_ptr<bool>(), bsz, num_heads, num_cpu_heads, head_dim,
      gpu_t_stride, gpu_b_stride, gpu_h_stride, gpu_s_stride, cpu_t_stride,
      cpu_b_stride, cpu_h_stride, cpu_s_stride);
}

template <typename T>
__global__ void CUDAAppendOffloadTensorPosWait(
    T* __restrict__ key, T* __restrict__ value, T* __restrict__ gpu_dst,
    T* __restrict__ cpu_dst, int64_t* __restrict__ cpu_head_ids,
    int32_t* __restrict__ gpu_pos,
    int32_t* __restrict__ cpu_pos,
    volatile bool* ready_flags,
    const int32_t batch_size, const int32_t num_heads,
    const int32_t num_cpu_heads, const int32_t head_dim,
    const int64_t gpu_t_stride, const int64_t gpu_b_stride,
    const int64_t gpu_h_stride, const int64_t gpu_s_stride,
    const int64_t cpu_t_stride, const int64_t cpu_b_stride,
    const int64_t cpu_h_stride, const int64_t cpu_s_stride) {
  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  int bsz_id = blockIdx.x;
  int head_id = blockIdx.y;

  T* input_ptr = nullptr;
  T* output_ptr = nullptr;

  if (head_id >= num_cpu_heads) {  // cpu
    head_id -= num_cpu_heads;
    int64_t src_offset = bsz_id * num_heads * head_dim + head_id * head_dim;

    if (warp_id == 0) {  // key
      int64_t dst_offset = bsz_id * cpu_b_stride + (*cpu_pos) * cpu_s_stride +
                           head_id * cpu_h_stride;
      input_ptr = key + src_offset;
      output_ptr = cpu_dst + dst_offset;
    }

    else {  // value
      int64_t dst_offset = cpu_t_stride + bsz_id * cpu_b_stride +
                           (*cpu_pos) * cpu_s_stride + head_id * cpu_h_stride;
      input_ptr = value + src_offset;
      output_ptr = cpu_dst + dst_offset;
    }
  }

  else {  // gpu
    int64_t input_head_id = cpu_head_ids[head_id];
    int64_t src_offset =
        bsz_id * num_heads * head_dim + input_head_id * head_dim;

    if (warp_id == 0) {  // key
      int64_t dst_offset = bsz_id * gpu_b_stride + (*gpu_pos) * gpu_s_stride +
                           head_id * gpu_h_stride;
      input_ptr = key + src_offset;
      output_ptr = gpu_dst + dst_offset;
    }

    else {  // value
      int64_t dst_offset = gpu_t_stride + bsz_id * gpu_b_stride +
                           (*gpu_pos) * gpu_s_stride + head_id * gpu_h_stride;
      input_ptr = value + src_offset;
      output_ptr = gpu_dst + dst_offset;
    }
  }

  for (int i = lane_id; i < head_dim; i += 32) {
    output_ptr[i] = input_ptr[i];
  }

  int total_hid = threadIdx.x + blockIdx.x * blockDim.x;
  if (total_hid < batch_size * num_heads && blockIdx.y == 0) {
    while (!ready_flags[total_hid]) {
      __nanosleep(1 * 1000);
    }
    // ready_flags[total_hid] = false;
  } else {
    return;
  }
}

void AppendOffloadTensorPosAndWait(
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& gpu_kv_buffer,
  torch::Tensor& cpu_kv_cache,
  torch::Tensor& gpu_append_pos,
  torch::Tensor& cpu_append_pos,
  torch::Tensor& ready_flags,
  torch::Tensor& cpu_head_ids) {
  // shape for gpu_cache is (2, bsz, num_kv_head, gpu_len, head_dim)
  // shape for cpu_cache is (2, bsz, cpu_len, num_kv_head, head_dim)
  // shape for key and value is (bsz, 1, num_kv_head, head_dim)

  int32_t bsz = key_states.size(0);
  int32_t num_heads = key_states.size(2);
  int32_t num_cpu_heads = cpu_head_ids.size(0);
  int32_t head_dim = key_states.size(3);

  int64_t gpu_t_stride = gpu_kv_buffer.stride(0);
  int64_t gpu_b_stride = gpu_kv_buffer.stride(1);
  int64_t gpu_h_stride = gpu_kv_buffer.stride(3);
  int64_t gpu_s_stride = gpu_kv_buffer.stride(2);

  int64_t cpu_t_stride = cpu_kv_cache.stride(0);
  int64_t cpu_b_stride = cpu_kv_cache.stride(1);
  int64_t cpu_h_stride = cpu_kv_cache.stride(3);
  int64_t cpu_s_stride = cpu_kv_cache.stride(2);

  constexpr int num_threads = 32 * 2;
  dim3 blk(num_threads);
  dim3 grid(bsz, num_cpu_heads + num_heads);
  auto device = key_states.device();
  int32_t device_id = device.index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  auto scalar_type = key_states.scalar_type();
  if (scalar_type == at::ScalarType::Half) {
    CUDAAppendOffloadTensorPosWait<at::Half><<<grid, blk, 0, stream>>>(
        key_states.data_ptr<at::Half>(),
        value_states.data_ptr<at::Half>(),
        gpu_kv_buffer.data_ptr<at::Half>(),
        cpu_kv_cache.data_ptr<at::Half>(),
        cpu_head_ids.data_ptr<int64_t>(),
        gpu_append_pos.data_ptr<int32_t>(),
        cpu_append_pos.data_ptr<int32_t>(),
        ready_flags.data_ptr<bool>(), bsz, num_heads, num_cpu_heads, head_dim,
        gpu_t_stride, gpu_b_stride, gpu_h_stride, gpu_s_stride, cpu_t_stride,
        cpu_b_stride, cpu_h_stride, cpu_s_stride);
  } else if (scalar_type == at::ScalarType::BFloat16) {
    CUDAAppendOffloadTensorPosWait<at::BFloat16><<<grid, blk, 0, stream>>>(
        key_states.data_ptr<at::BFloat16>(),
        value_states.data_ptr<at::BFloat16>(),
        gpu_kv_buffer.data_ptr<at::BFloat16>(),
        cpu_kv_cache.data_ptr<at::BFloat16>(),
        cpu_head_ids.data_ptr<int64_t>(),
        gpu_append_pos.data_ptr<int32_t>(),
        cpu_append_pos.data_ptr<int32_t>(),
        ready_flags.data_ptr<bool>(), bsz, num_heads, num_cpu_heads, head_dim,
        gpu_t_stride, gpu_b_stride, gpu_h_stride, gpu_s_stride, cpu_t_stride,
        cpu_b_stride, cpu_h_stride, cpu_s_stride);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type: ", scalar_type);
  }
}

__global__ void CUDAWaitKVData(volatile bool* ready_flags,
                               const int32_t batch_size,
                               const int32_t num_heads) {
  int total_hid = threadIdx.x + blockIdx.x * blockDim.x;
  if (total_hid < batch_size * num_heads) {
    while (!ready_flags[total_hid]) {
      __nanosleep(1 * 1000);
    }
    // ready_flags[total_hid] = false; // bug here?
  } else {
    return;
  }
}

void WaitKVData(torch::Tensor& ready_flags, int64_t batch_size,
                int64_t num_heads) {
  constexpr int num_threads = 256;
  int num_blocks = (batch_size * num_heads + num_threads - 1) / num_threads;
  dim3 blk(num_threads);
  dim3 grid(num_blocks);
  int32_t device_id = torch::cuda::current_device();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  CUDAWaitKVData<<<grid, blk, 0, stream>>>(ready_flags.data_ptr<bool>(),
                                           batch_size, num_heads);
}

template <typename T, typename IdT, const int32_t kElemPerThread>
__global__ void GatherGPUKVCacheKernel(
    IdT* __restrict__ indices, T* __restrict__ src_key,
    T* __restrict__ src_value, T* __restrict__ dst_key,
    T* __restrict__ dst_value, int64_t* __restrict__ head_ids,
    const int32_t gather_length, const int32_t sink_recent_budget,
    const int32_t head_dim, const int64_t idx_bsz_stride,
    const int64_t idx_head_stride, const int64_t src_bsz_stride,
    const int64_t src_head_stride, const int64_t src_seq_stride,
    const int64_t dst_bsz_stride, const int64_t dst_head_stride,
    const int64_t dst_seq_stride) {
  uint32_t tid = threadIdx.x;
  uint32_t sid = blockIdx.x;
  uint32_t hidx = blockIdx.y;
  uint32_t bid = blockIdx.z;
  uint32_t hid = head_ids[hidx];

  if (sid < gather_length) {
    uint32_t offset = tid * kElemPerThread;
    uint32_t data_type = offset / head_dim;
    uint32_t col_offset = offset % head_dim;
    IdT token_id =
        *(indices + bid * idx_bsz_stride + hid * idx_head_stride + sid);

    T* dst_ptr;
    T* src_ptr;
    if (data_type == 0) {
      src_ptr = src_key + bid * src_bsz_stride + hidx * src_head_stride +
                token_id * src_seq_stride + col_offset;
      dst_ptr = dst_key + bid * dst_bsz_stride + hid * dst_head_stride +
                (sink_recent_budget + sid) * dst_seq_stride + col_offset;
    } else {
      src_ptr = src_value + bid * src_bsz_stride + hidx * src_head_stride +
                token_id * src_seq_stride + col_offset;
      dst_ptr = dst_value + bid * dst_bsz_stride + hid * dst_head_stride +
                (sink_recent_budget + sid) * dst_seq_stride + col_offset;
    }

#pragma unroll
    for (uint32_t i = 0; i < kElemPerThread; i += 1) {
      dst_ptr[i] = src_ptr[i];
    }
  } else {
    return;
  }
}

void GatherGPUKVCache(torch::Tensor& indices, torch::Tensor& src_key,
                      torch::Tensor& src_value, torch::Tensor& dst_key,
                      torch::Tensor& dst_value, torch::Tensor& head_ids,
                      int64_t sink_recent_budget) {
  // indices: [batch_size * num_head, k], int32
  // src_key/value: [batch_size, seqlen, num_gather_heads, head_dim], half
  // dst_key/value: [batch_size, num_heads, k + sink_recent, head_dim], half
  // head_ids: [num_gather_heads, ], int64

  int64_t batch_size = src_key.size(0);
  int64_t num_gather_heads = head_ids.size(0);
  int64_t gather_len = indices.size(2);
  int64_t head_dim = src_key.size(3);

  auto device = src_key.device();
  int32_t device_id = device.index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  constexpr int32_t thread_per_block = 32;
  dim3 block(thread_per_block);
  dim3 grid(gather_len, num_gather_heads, batch_size);
  GatherGPUKVCacheKernel<at::Half, int32_t, 8><<<grid, block, 0, stream>>>(
      indices.data_ptr<int32_t>(), src_key.data_ptr<at::Half>(),
      src_value.data_ptr<at::Half>(), dst_key.data_ptr<at::Half>(),
      dst_value.data_ptr<at::Half>(), head_ids.data_ptr<int64_t>(), gather_len,
      sink_recent_budget, head_dim, indices.stride(0), indices.stride(1),
      src_key.stride(0), src_key.stride(2), src_key.stride(1),
      dst_key.stride(0), dst_key.stride(1), dst_key.stride(2));
}

template <typename T>
__global__ void LaunchPrefetchingKernel(
    T* __restrict__ gpu_indices,
    bool* __restrict__ gpu_gather_mask,
    int32_t* __restrict__ gpu_index_length,
    volatile int64_t* __restrict__ cpu_indices,
    int32_t* __restrict__ cpu_gather_flag,
    volatile bool* __restrict__ cpu_ready_mask,
    int64_t num_total_heads,
    int64_t batch_size, int64_t num_head, int64_t cache_seq_len,
    int64_t src_len, int64_t dst_len, int64_t layer_idx) {

  const int head_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  // compute real indices and copy to cpu
  if (gpu_gather_mask[head_idx]) {
    T* src_ptr = gpu_indices + head_idx * src_len;
    const int32_t batch_id = head_idx / num_head;
    const int32_t head_id = head_idx % num_head;
    for (int i = tid; i < *gpu_index_length; i += num_threads) {
      int64_t data =
          batch_id * num_head * cache_seq_len + head_id + src_ptr[i] * num_head;
      *(cpu_indices + head_idx * dst_len + i) = data;
      // __threadfence_system();
    }
  }

  // copy gather mask to cpu
  const int copy_mask_idx = blockIdx.x * blockDim.x + tid;
  if (copy_mask_idx < num_total_heads) {
    cpu_ready_mask[copy_mask_idx] = !gpu_gather_mask[copy_mask_idx];
  }

  // copy metadata to cpu
  if (tid == 0 && head_idx == 0) {
    cpu_gather_flag[1] = *gpu_index_length;
    cpu_gather_flag[2] = batch_size;
  }

  __threadfence_system();
}

__global__ void SetReadyKernel2(int* gather_flag, int layer_idx) {
  *gather_flag = layer_idx;
}

void StaticLaunchPrefetching(torch::Tensor& gpu_indices,  // [b * h, max_topk + sink + recent + 1]
                             torch::Tensor& gpu_gather_mask,  // [b * h]
                             torch::Tensor& gpu_index_length,  // [1, ]
                             torch::Tensor& cpu_indices,  // [b * h, max_topk]
                             torch::Tensor& cpu_gather_flag,  // [6, ]
                             torch::Tensor& cpu_ready_mask,  // [b * h]
                             int64_t batch_size, int64_t max_cache_seqlen,
                             int64_t num_heads, int64_t layer_idx) {
  int64_t num_total_heads = gpu_gather_mask.size(0);
  int64_t src_len = gpu_indices.sizes().back();
  int64_t dst_len = cpu_indices.sizes().back();

  auto device = gpu_indices.device();
  int32_t device_id = device.index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  constexpr int num_threads = 128;
  dim3 block(num_threads);
  dim3 grid(num_total_heads);
  LaunchPrefetchingKernel<int32_t><<<grid, block, 0, stream>>>(
      gpu_indices.data_ptr<int32_t>(), gpu_gather_mask.data_ptr<bool>(),
      gpu_index_length.data_ptr<int32_t>(), cpu_indices.data_ptr<int64_t>(),
      cpu_gather_flag.data_ptr<int32_t>(), cpu_ready_mask.data_ptr<bool>(),
      num_total_heads, batch_size, num_heads, max_cache_seqlen, src_len, dst_len, layer_idx);
  SetReadyKernel2<<<1, 1, 0, stream>>>(cpu_gather_flag.data_ptr<int32_t>(), layer_idx);
}

}  // namespace kvlib