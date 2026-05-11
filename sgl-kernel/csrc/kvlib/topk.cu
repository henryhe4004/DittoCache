#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <raft/matrix/detail/select_radix.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include "operator.h"
#include "topk/select_radix_masked.cuh"

namespace kvlib {

class my_custom_resource : public rmm::mr::device_memory_resource {
  /* implement do_allocate and do_deallocate */
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) {
    stream = (cudaStream_t)stream;
    return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(bytes,
                                                                  stream);
  }

  void do_deallocate(void* ptr, std::size_t bytes,
                     rmm::cuda_stream_view stream = rmm::cuda_stream_view{}) {
    stream = (cudaStream_t)stream;
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
  }

  // Get free and available memory for memory resource
  std::pair<std::size_t, std::size_t> do_get_mem_info(
      rmm::cuda_stream_view stream) const noexcept override {
    return std::make_pair(0, 0);
  }

  bool supports_streams() const noexcept override { return true; }

  bool supports_get_mem_info() const noexcept override { return false; }
};

torch::Tensor TopkCUDA(torch::Tensor& data, int32_t k, bool largest) {
  // note for data, its shape must be [batch_size, num_head, seq_len]
  // may need transpose before this function call
  CHECK(data.device().is_cuda() && data.is_contiguous());
  int32_t batch_size = data.size(0);
  int32_t num_head = data.size(1);
  int32_t seq_len = data.size(2);

  int32_t total_batch_size = batch_size * num_head;

  auto device = data.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  torch::Tensor topk_values =
      torch::empty({batch_size, num_head, k},
                   torch::TensorOptions().dtype(data.dtype()).device(device));
  torch::Tensor topk_indices = torch::empty({batch_size, num_head, k}, options);

  my_custom_resource my_mr;

  if (data.dtype() == torch::kFloat32) {
    raft::matrix::detail::select::radix::select_k<float, int32_t, 11, 512>(
        data.data_ptr<float>(), static_cast<int32_t*>(nullptr),
        total_batch_size, seq_len, k, topk_values.data_ptr<float>(),
        topk_indices.data_ptr<int32_t>(), !largest, true, stream, &my_mr);
  } else {
    raft::matrix::detail::select::radix::select_k<half, int32_t, 11, 512>(
        (half*)data.data_ptr<at::Half>(), static_cast<int32_t*>(nullptr),
        total_batch_size, seq_len, k, (half*)topk_values.data_ptr<at::Half>(),
        topk_indices.data_ptr<int32_t>(), !largest, true, stream, &my_mr);
  }

  return topk_indices;
}

void TopkMaskedCUDA(
  torch::Tensor& data,
  torch::Tensor& bh_mask,
  torch::Tensor& out_index,
  torch::Tensor& out_values,
  torch::Tensor& real_len,
  torch::Tensor& real_k,
  bool largest
) {
  int32_t batch_size = data.size(0);
  int32_t num_head = data.size(1);
  int32_t max_len = data.size(2);
  int32_t max_k = out_index.sizes().back();
  int32_t total_batch_size = batch_size * num_head;
  bool real_len_is_scalar = real_len.numel() == 1;
  bool real_k_is_scalar = real_k.numel() == 1;
  TORCH_CHECK(real_len_is_scalar || real_len.numel() >= total_batch_size,
              "real_len must have 1 element or at least batch_size*num_head elements, got ",
              real_len.numel(), " for total rows ", total_batch_size);
  TORCH_CHECK(real_k_is_scalar || real_k.numel() >= total_batch_size,
              "real_k must have 1 element or at least batch_size*num_head elements, got ",
              real_k.numel(), " for total rows ", total_batch_size);

  auto device = data.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  my_custom_resource my_mr;

  auto scalar_type = data.scalar_type();
  if (scalar_type == at::ScalarType::Half) {
    raft::matrix::detail::select::radix::select_k_masked<half, int32_t, bool,
                                                         11, 512>(
        (half*)data.data_ptr<at::Half>(), static_cast<int32_t*>(nullptr),
        bh_mask.data_ptr<bool>(), real_len.data_ptr<int32_t>(),
        real_k.data_ptr<int32_t>(),
        total_batch_size, max_len, max_k,
        (half*)out_values.data_ptr<at::Half>(),
        out_index.data_ptr<int32_t>(), !largest, true,
        real_len_is_scalar, real_k_is_scalar, stream, &my_mr);
  } else if (scalar_type == at::ScalarType::Float) {
    raft::matrix::detail::select::radix::select_k_masked<float, int32_t, bool,
                                                         11, 512>(
        data.data_ptr<float>(), static_cast<int32_t*>(nullptr),
        bh_mask.data_ptr<bool>(), real_len.data_ptr<int32_t>(),
        real_k.data_ptr<int32_t>(),
        total_batch_size, max_len, max_k,
        out_values.data_ptr<float>(),
        out_index.data_ptr<int32_t>(), !largest, true,
        real_len_is_scalar, real_k_is_scalar, stream, &my_mr);
  } else {
    TORCH_CHECK(false, "Top-k kernel unsupported scalar type: ", scalar_type);
  }
}

}  // namespace kvlib
