#include <immintrin.h>
#include <nvtx3/nvToolsExt.h>
#include <omp.h>
#include <pthread.h>
#include <chrono>
#include <iostream>
#include "cpu_gather_engine.h"

#define PREFETCH_DISTANCE 5

namespace kvlib {

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

  while (true) {
    if (*_launch_flag >= 0) {
      const int layer_idx = *_launch_flag;
      const int gather_length = *(_launch_flag + 1);
      const int curr_batch_size = *(_launch_flag + 2);
      const int curr_max_cache_seq_length = *(_launch_flag + 3);
      const int curr_max_cache_buffer_length = *(_launch_flag + 4);
      const int curr_max_indices_buffer_length = *(_launch_flag + 5);
      *_launch_flag = -1;

      // shape for cpu_kv_data is [2, Batch, Seq, Head, HeadDim]
      const size_t cpu_key_size = curr_batch_size * curr_max_cache_seq_length * onetoken_key_size;
      char *key_tensor_ptr = this->_cpu_kv_data[layer_idx];
      char *value_tensor_ptr = key_tensor_ptr + cpu_key_size;

      const size_t gpu_buffer_key_size = curr_batch_size * curr_max_cache_buffer_length *
                                         _gpu_buffer_head_num[layer_idx] * vector_size;

      char *output_key_tensor_ptr =
          (char *)this->_user_space_gpu_kv_buffer_mapped[layer_idx];
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
      if (num_gather_heads == 0) continue;

      const int num_threads = this->_num_omp_threads;
      const int selected_numel = num_gather_heads * gather_length;

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

          size_t cur_dst_offset = bid * gpu_batch_stride +
                                  dst_hid * vector_size +
                                  gpu_sink_recent_offset;

          int64_t *thread_indices_ptr =
              indices_base + total_hid * curr_max_indices_buffer_length;
          char *dst_key_ptr = out_key_base + cur_dst_offset;
          char *dst_value_ptr = out_value_base + cur_dst_offset;

          // 确定当前块的起始位置和长度
          int sid = (j == 0) ? start_sid : 0;
          int block_remaining = std::min(gather_length - sid, remaining - j);
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

          j += block_remaining;
          ++tot_hid_idx;
        }
        _mm_sfence();

#pragma omp barrier
        for (int i = tid; i < num_gather_heads; i += num_threads) {
          int total_hid = _gather_hids[i];
          this->_ready_flags[layer_idx][total_hid] = true;
          // if (this->_debug) {
          //   printf("ready_flags[%d][%d] = true\n", layer_idx, total_hid);
          // }
        }
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
    bool debug)
    : _max_batch_size(max_batch_size),
      _sink_recent_budget(sink_recent_budget),
      _num_heads(num_heads),
      _head_dim(head_dim),
      _num_omp_threads(num_omp_threads),
      // _launch_flag(launch_flag.data_ptr<int32_t>()),
      _launch_flag(reinterpret_cast<volatile int32_t*>(launch_flag.data_ptr<int32_t>())),
      _cpu_indices_buffer(cpu_indices_buffer.data_ptr<int64_t>()),
      _debug(debug) {

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

  // init gdr and setup gpu_kv_buffer
  _g = gdr_open();
  for (auto &tensor : gpu_kv_buffer) {
    if (tensor.has_value()) {
      size_t this_layer_gpu_buffer_size =
          tensor.value().numel() * sizeof(uint16_t);
      char *d_ptr = (char *)tensor.value().data_ptr();
      _gpu_kv_buffer.emplace_back(d_ptr);
      gdr_mh_t handler;
      gdr_pin_buffer(_g, (unsigned long)_gpu_kv_buffer.back(),
                     this_layer_gpu_buffer_size, 0, 0, &handler);
      void *mapped_gpu_ptr;
      gdr_map(_g, handler, &mapped_gpu_ptr, this_layer_gpu_buffer_size);
      gdr_info_t info;
      gdr_get_info(_g, handler, &info);

      void *user_space_ptr =
          (char *)mapped_gpu_ptr + ((uintptr_t)d_ptr & (info.page_size - 1));

      _gpu_kv_buffer_mapped.emplace_back(mapped_gpu_ptr);
      _gdr_handlers.emplace_back(handler);
      _user_space_gpu_kv_buffer_mapped.emplace_back(user_space_ptr);
      _gpu_buffer_size.emplace_back(this_layer_gpu_buffer_size);
    } else {
      _gpu_kv_buffer.emplace_back(nullptr);
      _gpu_kv_buffer_mapped.emplace_back(nullptr);
      _gdr_handlers.emplace_back(std::nullopt);
      _user_space_gpu_kv_buffer_mapped.emplace_back(nullptr);
      _gpu_buffer_size.emplace_back(0);
    }
  }

  _gather_hids = std::vector<int>(_total_num_heads);
  _worker = std::thread(&CPUGatherEngineV3::_work_loop, this);
}

CPUGatherEngineV3::~CPUGatherEngineV3() {
  *_launch_flag = -2;

  int64_t num_reused_requests = _num_total_requests - _num_processed_requests;
  double hit_ratio = (double)num_reused_requests / (double)_num_total_requests;
  std::cout << "Access num: " << _num_total_requests
            << " Hit num: " << num_reused_requests
            << " Hit ratio: " << hit_ratio << std::endl;

  if (_worker.joinable()) {
    _worker.join();
  }

  for (size_t i = 0; i < _gdr_handlers.size(); i += 1) {
    if (_gdr_handlers[i].has_value()) {
      gdr_unmap(_g, _gdr_handlers[i].value(), _gpu_kv_buffer_mapped[i],
                _gpu_buffer_size[i]);
      gdr_unpin_buffer(_g, _gdr_handlers[i].value());
    }
  }
  gdr_close(_g);
}

}  // namespace kvlib