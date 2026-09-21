#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>
#include <nvtx3/nvToolsExt.h>
#include <omp.h>
#include <pthread.h>
#include <chrono>
#include <iostream>
#include "cpu_gather_engine.h"

#define PREFETCH_DISTANCE 5

namespace kvlib {

namespace {

void check_cuda_or_abort(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return;
  }
  std::cerr << operation << " failed: " << cudaGetErrorString(status)
            << std::endl;
  std::abort();
}

}  // namespace

void CPUGatherEngineV3::_work_loop() {
  pthread_t thId = pthread_self();
  pthread_attr_t thAttr;
  int policy = 0;
  int max_prio_for_policy = 0;

  pthread_attr_init(&thAttr);
  pthread_attr_getschedpolicy(&thAttr, &policy);
  max_prio_for_policy = sched_get_priority_max(policy);

  pthread_setschedprio(thId, max_prio_for_policy);
  pthread_attr_destroy(&thAttr);

  const size_t vector_size = this->_head_dim * sizeof(uint16_t);
  const size_t onetoken_key_size = this->_num_heads * vector_size;

  while (!_stop_requested.load(std::memory_order_acquire)) {
    if (*_launch_flag >= 0) {
      const int layer_idx = *_launch_flag;
      const int gather_length = *(_launch_flag + 1);
      const int curr_batch_size = *(_launch_flag + 2);
      const int curr_max_cache_seq_length = *(_launch_flag + 3);
      const int curr_max_cache_buffer_length = *(_launch_flag + 4);
      const int curr_max_indices_buffer_length = *(_launch_flag + 5);
      // shape for cpu_kv_data is [2, Batch, Seq, Head, HeadDim]
      const size_t cpu_key_size = curr_batch_size * curr_max_cache_seq_length * onetoken_key_size;
      char *key_tensor_ptr = this->_cpu_kv_data[layer_idx];
      char *value_tensor_ptr = key_tensor_ptr + cpu_key_size;

      const size_t gpu_buffer_key_size = curr_batch_size * curr_max_cache_buffer_length *
                                         _gpu_buffer_head_num[layer_idx] * vector_size;

      char *output_key_tensor_ptr =
          _transfer_backend == TransferBackend::kGdr
              ? (char *)this->_user_space_gpu_kv_buffer_mapped[layer_idx]
              : (char *)this->_cuda_staging_buffers[layer_idx];
      char *output_value_tensor_ptr =
          output_key_tensor_ptr + gpu_buffer_key_size;

      int num_gather_heads = 0;
      for (int i = 0; i < curr_batch_size * _num_heads; i += 1) {
        if (!_ready_flags[layer_idx][i]) {
          _gather_hids[num_gather_heads++] = i;
        }
      }

      _num_total_requests += curr_batch_size * _num_heads;
      _num_processed_requests += num_gather_heads;
      if (num_gather_heads == 0) {
        *_launch_flag = -1;
        continue;
      }
      if (gather_length <= 0) {
        *_launch_flag = -1;
        std::atomic_thread_fence(std::memory_order_release);
        for (int i = 0; i < num_gather_heads; ++i) {
          this->_ready_flags[layer_idx][_gather_hids[i]] = true;
        }
        continue;
      }

      const int num_threads = this->_num_omp_threads;
      const int selected_numel = num_gather_heads * gather_length;
      const volatile int32_t *per_head_lengths = _launch_flag + 6;

      // 预计算常用值
      const int num_heads = this->_num_heads;
      const int num_gpu_buffer_heads = this->_gpu_buffer_head_num[layer_idx];
      const size_t gpu_onetoken_size = num_gpu_buffer_heads * vector_size;
      const size_t gpu_sink_recent_offset = this->_sink_recent_budget * gpu_onetoken_size;
      const size_t gpu_batch_stride = curr_max_cache_buffer_length * gpu_onetoken_size;
      const int chunk_size = (selected_numel + num_threads - 1) / num_threads;

#pragma omp parallel num_threads(num_threads)
      {
        const int tid = omp_get_thread_num();
        int begin = tid * chunk_size;
        int end = std::min(selected_numel, begin + chunk_size);

        int start_sid = begin % gather_length;
        int tot_hid_idx = begin / gather_length;
        int remaining = end - begin;
        int j = 0;

        // 预计算指针
        int64_t *indices_base = this->_cpu_indices_buffer;
        const char *key_base = key_tensor_ptr;
        const char *value_base = value_tensor_ptr;
        char *out_key_base = output_key_tensor_ptr;
        char *out_value_base = output_value_tensor_ptr;

        while (j < remaining) {
          int total_hid = _gather_hids[tot_hid_idx];
          int bid = total_hid / num_heads;
          int hid = total_hid % num_heads;
          int dst_hid = this->_dst_head_index[layer_idx][hid];
          int head_gather_length = per_head_lengths[total_hid];
          if (head_gather_length < 0) {
            head_gather_length = 0;
          }
          if (head_gather_length > gather_length) {
            head_gather_length = gather_length;
          }

          size_t cur_dst_offset = bid * gpu_batch_stride +
                                  dst_hid * vector_size +
                                  gpu_sink_recent_offset;

          int64_t *thread_indices_ptr =
              indices_base + total_hid * curr_max_indices_buffer_length;
          char *dst_key_ptr = out_key_base + cur_dst_offset;
          char *dst_value_ptr = out_value_base + cur_dst_offset;

          // 确定当前块的起始位置和长度
          int sid = (j == 0) ? start_sid : 0;
          int slots_remaining =
              std::min(gather_length - sid, remaining - j);
          int block_remaining =
              sid < head_gather_length
                  ? std::min(head_gather_length - sid, slots_remaining)
                  : 0;
          const int end_k = block_remaining;

          // 处理当前块
          for (int k = 0; k < end_k; ++k) {
            int64_t index = thread_indices_ptr[sid + k];
            size_t src_offset = index * vector_size;
            size_t dst_offset = (sid + k) * gpu_onetoken_size;

            // PREFETCH：提前加载后续访问的 key/value 数据
            if (k + PREFETCH_DISTANCE < end_k) {
              int64_t next_index =
                  thread_indices_ptr[sid + k + PREFETCH_DISTANCE];
              size_t next_src_offset = next_index * vector_size;
              _mm_prefetch(key_base + next_src_offset, _MM_HINT_NTA);
              _mm_prefetch(value_base + next_src_offset, _MM_HINT_NTA);
            }

            memcpy(dst_key_ptr + dst_offset, key_base + src_offset,
                   vector_size);
            memcpy(dst_value_ptr + dst_offset, value_base + src_offset,
                   vector_size);
          }

          j += slots_remaining;
          ++tot_hid_idx;
        }
        _mm_sfence();

      }

      if (_transfer_backend == TransferBackend::kCudaMemcpy) {
        check_cuda_or_abort(
            cudaSetDevice(_gpu_device_ids[layer_idx]), "cudaSetDevice");
        cudaStream_t stream = _cuda_copy_streams[layer_idx];
        char* gpu_key_base = _gpu_kv_buffer[layer_idx];
        char* gpu_value_base = gpu_key_base + gpu_buffer_key_size;
        char* staging_key_base =
            (char*)_cuda_staging_buffers[layer_idx];
        char* staging_value_base =
            staging_key_base + gpu_buffer_key_size;

        for (int i = 0; i < num_gather_heads; ++i) {
          int total_hid = _gather_hids[i];
          int bid = total_hid / num_heads;
          int hid = total_hid % num_heads;
          int dst_hid = _dst_head_index[layer_idx][hid];
          int head_gather_length = per_head_lengths[total_hid];
          head_gather_length =
              std::max(0, std::min(head_gather_length, gather_length));
          if (head_gather_length == 0) {
            continue;
          }
          size_t dst_offset = bid * gpu_batch_stride +
                              dst_hid * vector_size +
                              gpu_sink_recent_offset;
          check_cuda_or_abort(
              cudaMemcpy2DAsync(
                  gpu_key_base + dst_offset,
                  gpu_onetoken_size,
                  staging_key_base + dst_offset,
                  gpu_onetoken_size,
                  vector_size,
                  head_gather_length,
                  cudaMemcpyHostToDevice,
                  stream),
              "cudaMemcpy2DAsync(key)");
          check_cuda_or_abort(
              cudaMemcpy2DAsync(
                  gpu_value_base + dst_offset,
                  gpu_onetoken_size,
                  staging_value_base + dst_offset,
                  gpu_onetoken_size,
                  vector_size,
                  head_gather_length,
                  cudaMemcpyHostToDevice,
                  stream),
              "cudaMemcpy2DAsync(value)");
        }
        check_cuda_or_abort(
            cudaStreamSynchronize(stream), "cudaStreamSynchronize");
      }

      // Release the mailbox before publishing completion. The next GPU layer
      // cannot enqueue another request until it observes these ready flags.
      *_launch_flag = -1;
      std::atomic_thread_fence(std::memory_order_release);
      for (int i = 0; i < num_gather_heads; ++i) {
        int total_hid = _gather_hids[i];
        this->_ready_flags[layer_idx][total_hid] = true;
      }

    } else if (*_launch_flag == -2) {
      // exit signal
      break;
    }
  }
}

CPUGatherEngineV3::CPUGatherEngineV3(
    int64_t num_omp_threads,
    std::vector<std::optional<torch::Tensor>> &cpu_kv_data,
    std::vector<std::optional<torch::Tensor>> &gpu_kv_buffer,
    std::vector<std::optional<torch::Tensor>> &dst_head_index,
    std::vector<int64_t> &num_gpu_buffer_heads,
    torch::Tensor &cpu_indices_buffer,
    torch::Tensor &launch_flag,
    std::vector<std::optional<torch::Tensor>> &ready_flags,
    int64_t max_batch_size,
    int64_t sink_recent_budget,
    int64_t num_heads,
    int64_t head_dim,
    bool debug,
    const std::string& transfer_backend)
    : _max_batch_size(max_batch_size),
      _sink_recent_budget(sink_recent_budget),
      _num_heads(num_heads),
      _head_dim(head_dim),
      _num_omp_threads(num_omp_threads),
      // _launch_flag(launch_flag.data_ptr<int32_t>()),
      _launch_flag(reinterpret_cast<volatile int32_t*>(launch_flag.data_ptr<int32_t>())),
      _cpu_indices_buffer(cpu_indices_buffer.data_ptr<int64_t>()),
      _debug(debug) {

  if (transfer_backend == "gdr" || transfer_backend == "gdrcopy") {
    _transfer_backend = TransferBackend::kGdr;
  } else if (transfer_backend == "cuda_memcpy" || transfer_backend == "memcpy") {
    _transfer_backend = TransferBackend::kCudaMemcpy;
  } else {
    TORCH_CHECK(
        false,
        "Unsupported CPUGatherEngineV3 transfer_backend=",
        transfer_backend);
  }

  _total_num_heads = _max_batch_size * _num_heads;

  // setup ready_flags for each layer
  // 0 for not ready, 1 for ready
  // it is setup by CPUGatherEngine to launch successive GPU kernels
  for (auto &tensor : ready_flags) {
    bool *ptr =
        tensor.has_value() ? (bool *)tensor.value().data_ptr() : nullptr;
    _ready_flags.emplace_back(ptr);
  }

  // setup cpu_kv_data ptr
  for (auto &tensor : cpu_kv_data) {
    char *ptr =
        tensor.has_value() ? (char *)tensor.value().data_ptr() : nullptr;
    _cpu_kv_data.emplace_back(ptr);
  }

  // setup dst_head_index ptr
  for (auto &tensor : dst_head_index) {
    int *ptr = tensor.has_value() ? (int *)tensor.value().data_ptr() : nullptr;
    _dst_head_index.emplace_back(ptr);
  }

  for (auto &num : num_gpu_buffer_heads) {
    _gpu_buffer_head_num.emplace_back(num);
  }

  // Set up either GDR BAR mappings or pinned staging plus CUDA copy streams.
  if (_transfer_backend == TransferBackend::kGdr) {
    _g = gdr_open();
    TORCH_CHECK(_g != nullptr, "gdr_open failed");
  }
  for (size_t layer_idx = 0; layer_idx < gpu_kv_buffer.size(); ++layer_idx) {
    auto &tensor = gpu_kv_buffer[layer_idx];
    if (tensor.has_value()) {
      size_t this_layer_gpu_buffer_size =
          tensor.value().numel() * sizeof(uint16_t);
      char *d_ptr = (char *)tensor.value().data_ptr();
      _gpu_kv_buffer.emplace_back(d_ptr);
      int device_id = tensor.value().get_device();
      _gpu_device_ids.emplace_back(device_id);
      if (_transfer_backend == TransferBackend::kGdr) {
        gdr_mh_t handler;
        const int pin_status = gdr_pin_buffer(
            _g,
            (unsigned long)_gpu_kv_buffer.back(),
            this_layer_gpu_buffer_size,
            0,
            0,
            &handler);
        TORCH_CHECK(
            pin_status == 0,
            "gdr_pin_buffer failed at layer ",
            layer_idx,
            ": device=",
            device_id,
            ", address=",
            static_cast<void*>(d_ptr),
            ", alignment_offset=",
            reinterpret_cast<uintptr_t>(d_ptr) % GPU_PAGE_SIZE,
            ", size=",
            this_layer_gpu_buffer_size,
            ", status=",
            pin_status,
            " (",
            std::strerror(pin_status),
            ")");
        void *mapped_gpu_ptr;
        const int map_status = gdr_map(
            _g,
            handler,
            &mapped_gpu_ptr,
            this_layer_gpu_buffer_size);
        TORCH_CHECK(
            map_status == 0,
            "gdr_map failed at layer ",
            layer_idx,
            ": device=",
            device_id,
            ", address=",
            static_cast<void*>(d_ptr),
            ", alignment_offset=",
            reinterpret_cast<uintptr_t>(d_ptr) % GPU_PAGE_SIZE,
            ", size=",
            this_layer_gpu_buffer_size,
            ", status=",
            map_status,
            " (",
            std::strerror(map_status),
            ")");
        gdr_info_t info;
        const int info_status = gdr_get_info(_g, handler, &info);
        TORCH_CHECK(
            info_status == 0,
            "gdr_get_info failed at layer ",
            layer_idx,
            ": status=",
            info_status,
            " (",
            std::strerror(info_status),
            ")");

        void *user_space_ptr =
            (char *)mapped_gpu_ptr + ((uintptr_t)d_ptr & (info.page_size - 1));

        _gpu_kv_buffer_mapped.emplace_back(mapped_gpu_ptr);
        _gdr_handlers.emplace_back(handler);
        _user_space_gpu_kv_buffer_mapped.emplace_back(user_space_ptr);
        _cuda_staging_buffers.emplace_back(nullptr);
        _cuda_copy_streams.emplace_back(nullptr);
      } else {
        TORCH_CHECK(
            cudaSetDevice(device_id) == cudaSuccess,
            "cudaSetDevice failed while constructing memcpy backend");
        void* staging_ptr = nullptr;
        TORCH_CHECK(
            cudaMallocHost(&staging_ptr, this_layer_gpu_buffer_size) ==
                cudaSuccess,
            "cudaMallocHost failed for memcpy staging buffer");
        cudaStream_t stream = nullptr;
        TORCH_CHECK(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
                cudaSuccess,
            "cudaStreamCreateWithFlags failed for memcpy backend");
        _gpu_kv_buffer_mapped.emplace_back(nullptr);
        _gdr_handlers.emplace_back(std::nullopt);
        _user_space_gpu_kv_buffer_mapped.emplace_back(nullptr);
        _cuda_staging_buffers.emplace_back(staging_ptr);
        _cuda_copy_streams.emplace_back(stream);
      }
      _gpu_buffer_size.emplace_back(this_layer_gpu_buffer_size);
    } else {
      _gpu_kv_buffer.emplace_back(nullptr);
      _gpu_device_ids.emplace_back(-1);
      _gpu_kv_buffer_mapped.emplace_back(nullptr);
      _gdr_handlers.emplace_back(std::nullopt);
      _user_space_gpu_kv_buffer_mapped.emplace_back(nullptr);
      _cuda_staging_buffers.emplace_back(nullptr);
      _cuda_copy_streams.emplace_back(nullptr);
      _gpu_buffer_size.emplace_back(0);
    }
  }

  _gather_hids = std::vector<int>(_total_num_heads);
  _worker = std::thread(&CPUGatherEngineV3::_work_loop, this);
}

CPUGatherEngineV3::~CPUGatherEngineV3() {
  _stop_requested.store(true, std::memory_order_release);
  *_launch_flag = -2;

  int64_t num_reused_requests = _num_total_requests - _num_processed_requests;
  double hit_ratio =
      _num_total_requests == 0
          ? 0.0
          : (double)num_reused_requests / (double)_num_total_requests;
  std::cout << "Access num: " << _num_total_requests
            << " Hit num: " << num_reused_requests
            << " Hit ratio: " << hit_ratio << std::endl;

  if (_worker.joinable()) {
    _worker.join();
  }

  if (_transfer_backend == TransferBackend::kGdr) {
    for (size_t i = 0; i < _gdr_handlers.size(); i += 1) {
      if (_gdr_handlers[i].has_value()) {
        gdr_unmap(_g, _gdr_handlers[i].value(), _gpu_kv_buffer_mapped[i],
                  _gpu_buffer_size[i]);
        gdr_unpin_buffer(_g, _gdr_handlers[i].value());
      }
    }
    if (_g != nullptr) {
      gdr_close(_g);
    }
  } else {
    for (size_t i = 0; i < _cuda_staging_buffers.size(); ++i) {
      if (_gpu_device_ids[i] >= 0) {
        cudaSetDevice(_gpu_device_ids[i]);
      }
      if (_cuda_copy_streams[i] != nullptr) {
        cudaStreamDestroy(_cuda_copy_streams[i]);
      }
      if (_cuda_staging_buffers[i] != nullptr) {
        cudaFreeHost(_cuda_staging_buffers[i]);
      }
    }
  }
}

}  // namespace kvlib
