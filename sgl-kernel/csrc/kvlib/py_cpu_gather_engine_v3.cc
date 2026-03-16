// Lightweight pybind-style wrapper for CPUGatherEngineV3.
// This exposes a Python-visible class without going through torch.library dispatch.

#include <torch/extension.h>

#include "kvlib/cpu_gather_engine.h"

namespace {

static std::vector<std::optional<torch::Tensor>> tensor_list_from_vector(
    const std::vector<torch::Tensor>& list) {
  std::vector<std::optional<torch::Tensor>> out;
  out.reserve(list.size());
  for (const torch::Tensor& t : list) {
    out.push_back(t.defined() && t.numel() > 0 ? std::optional<torch::Tensor>(t)
                                               : std::nullopt);
  }
  return out;
}

struct PyCPUGatherEngineV3 {
  std::unique_ptr<kvlib::CPUGatherEngineV3> engine;

  PyCPUGatherEngineV3(
      int64_t num_omp_threads,
      const std::vector<torch::Tensor>& cpu_kv_data,
      const std::vector<torch::Tensor>& gpu_kv_buffer,
      const std::vector<torch::Tensor>& dst_head_index,
      const std::vector<int64_t>& num_gpu_heads,
      torch::Tensor cpu_indices_buffer,
      torch::Tensor launch_flag,
      const std::vector<torch::Tensor>& ready_flags,
      int64_t max_batch_size,
      int64_t sink_recent_budget,
      int64_t num_heads,
      int64_t head_dim,
      bool debug) {
    auto cpu_kv_opt = tensor_list_from_vector(cpu_kv_data);
    auto gpu_kv_opt = tensor_list_from_vector(gpu_kv_buffer);
    auto dst_head_opt = tensor_list_from_vector(dst_head_index);
    auto ready_opt = tensor_list_from_vector(ready_flags);
    std::vector<int64_t> num_gpu_vec(num_gpu_heads.begin(), num_gpu_heads.end());

    engine = std::make_unique<kvlib::CPUGatherEngineV3>(
        num_omp_threads,
        cpu_kv_opt,
        gpu_kv_opt,
        dst_head_opt,
        num_gpu_vec,
        cpu_indices_buffer,
        launch_flag,
        ready_opt,
        max_batch_size,
        sink_recent_budget,
        num_heads,
        head_dim,
        debug);
  }
};

}  // namespace

PYBIND11_MODULE(kvlib_cpu_gather, m) {
  pybind11::class_<PyCPUGatherEngineV3>(m, "CPUGatherEngineV3")
      .def(pybind11::init<
           int64_t,
           const std::vector<torch::Tensor>&,
           const std::vector<torch::Tensor>&,
           const std::vector<torch::Tensor>&,
           const std::vector<int64_t>&,
           torch::Tensor,
           torch::Tensor,
           const std::vector<torch::Tensor>&,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           bool>(),
           pybind11::arg("num_omp_threads"),
           pybind11::arg("cpu_kv_data"),
           pybind11::arg("gpu_kv_buffer"),
           pybind11::arg("dst_head_index"),
           pybind11::arg("num_gpu_heads"),
           pybind11::arg("cpu_indices_buffer"),
           pybind11::arg("launch_flag"),
           pybind11::arg("ready_flags"),
           pybind11::arg("max_batch_size"),
           pybind11::arg("sink_recent_budget"),
           pybind11::arg("num_heads"),
           pybind11::arg("head_dim"),
           pybind11::arg("debug") = false);
}


