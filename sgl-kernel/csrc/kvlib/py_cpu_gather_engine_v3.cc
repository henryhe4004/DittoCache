// Lightweight pybind-style wrappers for KVLib offload features.
// These expose Python-visible APIs without going through torch.library
// dispatch, matching the original myTransformer.capi behaviour.

#include <torch/extension.h>

#include "kvlib/cpu_gather_engine.h"
#include "kvlib/operator.h"

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
      bool debug,
      const std::string& transfer_backend) {
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
        debug,
        transfer_backend);
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
           bool,
           const std::string&>(),
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
           pybind11::arg("debug") = false,
           pybind11::arg("transfer_backend") = "gdr");

  // Offload-related helpers mirroring kvlib CUDA kernels.
  m.def(
      "real_indices_and_launch_prefetch",
      [](torch::Tensor indices,
         torch::Tensor gpu_gather_mask,
         torch::Tensor output,
         torch::Tensor gather_flag,
         torch::Tensor cpu_ready_mask,
         int64_t cache_seq_len,
         int64_t batch_size,
         int64_t num_heads,
         int64_t layer_idx) {
        kvlib::RealInndicesAndLaunchPrefetching(
            indices,
            gpu_gather_mask,
            output,
            gather_flag,
            cpu_ready_mask,
            cache_seq_len,
            batch_size,
            num_heads,
            layer_idx);
      },
      pybind11::arg("indices"),
      pybind11::arg("gpu_gather_mask"),
      pybind11::arg("output"),
      pybind11::arg("gather_flag"),
      pybind11::arg("cpu_ready_mask"),
      pybind11::arg("cache_seq_len"),
      pybind11::arg("batch_size"),
      pybind11::arg("num_heads"),
      pybind11::arg("layer_idx"));

  m.def(
      "static_launch_prefetch",
      [](torch::Tensor gpu_indices,
         torch::Tensor gpu_gather_mask,
         torch::Tensor gpu_index_length,
         torch::Tensor cpu_indices,
         torch::Tensor cpu_gather_flag,
         torch::Tensor cpu_ready_mask,
         int64_t batch_size,
         int64_t max_cache_seqlen,
         int64_t num_heads,
         int64_t layer_idx) {
        kvlib::StaticLaunchPrefetching(
            gpu_indices,
            gpu_gather_mask,
            gpu_index_length,
            cpu_indices,
            cpu_gather_flag,
            cpu_ready_mask,
            batch_size,
            max_cache_seqlen,
            num_heads,
            layer_idx);
      },
      pybind11::arg("gpu_indices"),
      pybind11::arg("gpu_gather_mask"),
      pybind11::arg("gpu_index_length"),
      pybind11::arg("cpu_indices"),
      pybind11::arg("cpu_gather_flag"),
      pybind11::arg("cpu_ready_mask"),
      pybind11::arg("batch_size"),
      pybind11::arg("max_cache_seqlen"),
      pybind11::arg("num_heads"),
      pybind11::arg("layer_idx"));

  m.def(
      "wait_kv_data",
      [](torch::Tensor ready_flags, int64_t batch_size, int64_t num_heads) {
        kvlib::WaitKVData(ready_flags, batch_size, num_heads);
      },
      pybind11::arg("ready_flags"),
      pybind11::arg("batch_size"),
      pybind11::arg("num_heads"));

  m.def(
      "decode_append_offload_wait",
      [](torch::Tensor key_states,
         torch::Tensor value_states,
         torch::Tensor gpu_kv_buffer,
         torch::Tensor cpu_kv_cache,
         int64_t gpu_append_pos,
         int64_t cpu_append_pos,
         torch::Tensor ready_flags,
         torch::Tensor cpu_head_ids) {
        kvlib::AppendOffloadWait(
            key_states,
            value_states,
            gpu_kv_buffer,
            cpu_kv_cache,
            static_cast<int32_t>(gpu_append_pos),
            static_cast<int32_t>(cpu_append_pos),
            ready_flags,
            cpu_head_ids);
      },
      pybind11::arg("key_states"),
      pybind11::arg("value_states"),
      pybind11::arg("gpu_kv_buffer"),
      pybind11::arg("cpu_kv_cache"),
      pybind11::arg("gpu_append_pos"),
      pybind11::arg("cpu_append_pos"),
      pybind11::arg("ready_flags"),
      pybind11::arg("cpu_head_ids"));

  m.def(
      "decode_append_offload_tensor_pos_wait",
      [](torch::Tensor key_states,
         torch::Tensor value_states,
         torch::Tensor gpu_kv_buffer,
         torch::Tensor cpu_kv_cache,
         torch::Tensor gpu_append_pos,
         torch::Tensor cpu_append_pos,
         torch::Tensor ready_flags,
         torch::Tensor cpu_head_ids) {
        kvlib::AppendOffloadTensorPosAndWait(
            key_states,
            value_states,
            gpu_kv_buffer,
            cpu_kv_cache,
            gpu_append_pos,
            cpu_append_pos,
            ready_flags,
            cpu_head_ids);
      },
      pybind11::arg("key_states"),
      pybind11::arg("value_states"),
      pybind11::arg("gpu_kv_buffer"),
      pybind11::arg("cpu_kv_cache"),
      pybind11::arg("gpu_append_pos"),
      pybind11::arg("cpu_append_pos"),
      pybind11::arg("ready_flags"),
      pybind11::arg("cpu_head_ids"));

  m.def(
      "gather_gpu_kvcache",
      [](torch::Tensor indices,
         torch::Tensor src_key,
         torch::Tensor src_value,
         torch::Tensor dst_key,
         torch::Tensor dst_value,
         torch::Tensor head_ids,
         int64_t sink_recent_budget) {
        kvlib::GatherGPUKVCache(
            indices,
            src_key,
            src_value,
            dst_key,
            dst_value,
            head_ids,
            sink_recent_budget);
      },
      pybind11::arg("indices"),
      pybind11::arg("src_key"),
      pybind11::arg("src_value"),
      pybind11::arg("dst_key"),
      pybind11::arg("dst_value"),
      pybind11::arg("head_ids"),
      pybind11::arg("sink_recent_budget"));
}

