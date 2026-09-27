#pragma once
#include <cuda_runtime_api.h>
#include <gdrapi.h>
#include <torch/script.h>

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace kvlib {

class CPUGatherEngineV3 {
 private:
  int32_t _max_batch_size = 0;
  int32_t _sink_recent_budget = 0;
  int32_t _num_heads = 0;
  int32_t _head_dim = 0;

  int32_t _total_num_heads = 0;
  std::vector<size_t> _gpu_buffer_size = {};
  std::vector<size_t> _gpu_buffer_head_num = {};

  int64_t _num_omp_threads = 0;

  // int32_t* __restrict__ _launch_flag = nullptr;  // (3, )
  volatile int32_t* __restrict__ _launch_flag = nullptr;  // (3, )
  std::vector<bool*> _ready_flags = {};
  std::vector<int> _gather_hids = {};
  std::vector<int> _dst_offset = {};

  std::vector<char*> _cpu_kv_data = {};
  std::vector<char*> _gpu_kv_buffer = {};
  std::vector<int*> _dst_head_index = {};

  std::vector<void*> _gpu_kv_buffer_mapped = {};
  std::vector<void*> _user_space_gpu_kv_buffer_mapped = {};
  int64_t* __restrict__ _cpu_indices_buffer = nullptr;

  bool _use_gdrcopy = true;
  std::string _transfer_backend = "gdrcopy";
  int _cuda_device = -1;
  cudaStream_t _memcpy_stream = nullptr;
  char* _memcpy_staging = nullptr;
  size_t _memcpy_staging_capacity = 0;

  gdr_t _g = nullptr;
  std::vector<std::optional<gdr_mh_t>> _gdr_handlers = {};

  std::thread _worker;

  bool _debug;

  void _work_loop();

  int64_t _num_total_requests = 0;
  int64_t _num_processed_requests = 0;
  
 public:
  CPUGatherEngineV3(
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
    std::string transfer_backend = "gdrcopy");
  ~CPUGatherEngineV3();
};

}  // namespace kvlib