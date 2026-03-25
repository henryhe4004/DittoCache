/* Copyright 2025 SGLang Team. KVLib operator bindings for sgl_kernel (from myTransformer). */

#include <torch/all.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

#if defined(KVLIB_GDR_AVAILABLE)
#include "cpu_gather_engine.h"
#endif
#include "operator.h"
#include "cuda-attn/flash_api.h"

namespace {

#if !defined(KVLIB_RAFT_AVAILABLE)
// Fallback when RAFT is not linked: batch_topk via torch::topk, batch_topk_masked throws.
torch::Tensor batch_topk_impl(torch::Tensor data, int64_t k, bool largest) {
  TORCH_CHECK(data.device().is_cuda() && data.is_contiguous());
  auto data_flat = data.reshape({-1, data.size(-1)});
  auto result = torch::topk(data_flat, k, -1, largest);
  return std::get<1>(result)
      .reshape({data.size(0), data.size(1), k})
      .to(torch::kInt32);
}

void batch_topk_masked_impl(torch::Tensor data, torch::Tensor bh_mask,
                            torch::Tensor out_index, torch::Tensor out_values,
                            torch::Tensor real_len, torch::Tensor real_k, bool largest) {
  (void)data;
  (void)bh_mask;
  (void)out_index;
  (void)out_values;
  (void)real_len;
  (void)real_k;
  (void)largest;
  TORCH_CHECK(false,
              "kvlib_batch_topk_masked requires RAFT. Build with -DSGL_KERNEL_USE_RAFT=ON and RAFT/RMM available.");
}
#endif

// create_tensor: pinned host memory, shape from size[], dtype 16 (fp16) or 32 (fp32)
torch::Tensor create_tensor_impl(c10::IntArrayRef size, int64_t dtype) {
  int64_t num_elements = 1;
  for (int64_t d : size) {
    num_elements *= d;
  }
  void* buf = nullptr;
  if (dtype == 16) {
    TORCH_CHECK(cudaMallocHost(&buf, num_elements * sizeof(c10::Half)) == cudaSuccess);
    auto t = torch::from_blob(buf, {num_elements}, torch::kFloat16);
    return t.reshape(size);
  } else {
    TORCH_CHECK(dtype == 32, "create_tensor dtype must be 16 or 32");
    TORCH_CHECK(cudaMallocHost(&buf, num_elements * sizeof(float)) == cudaSuccess);
    auto t = torch::from_blob(buf, {num_elements}, torch::kFloat32);
    return t.reshape(size);
  }
}

#if defined(KVLIB_GDR_AVAILABLE)
// CPUGatherEngineV3 handle map (for offload)
static std::atomic<int64_t> g_engine_next_handle{1};
static std::unordered_map<int64_t, std::unique_ptr<kvlib::CPUGatherEngineV3>> g_engine_map;
static std::mutex g_engine_mutex;

static std::vector<std::optional<torch::Tensor>> tensor_list_from_list(
    const c10::List<torch::Tensor>& list) {
  std::vector<std::optional<torch::Tensor>> out;
  out.reserve(list.size());
  for (const torch::Tensor& t : list) {
    out.push_back(t.defined() && t.numel() > 0 ? std::optional<torch::Tensor>(t)
                                               : std::nullopt);
  }
  return out;
}

int64_t create_cpu_gather_engine_v3_impl(
    int64_t num_omp_threads,
    const c10::List<torch::Tensor>& cpu_kv_data,
    const c10::List<torch::Tensor>& gpu_kv_buffer,
    const c10::List<torch::Tensor>& dst_head_index,
    c10::IntArrayRef num_gpu_heads,
    const torch::Tensor& cpu_indices_buffer,
    const torch::Tensor& launch_flag,
    const c10::List<torch::Tensor>& ready_flags,
    int64_t max_batch_size,
    int64_t sink_recent_budget,
    int64_t num_heads,
    int64_t head_dim,
    bool debug) {
  std::vector<std::optional<torch::Tensor>> cpu_kv = tensor_list_from_list(cpu_kv_data);
  std::vector<std::optional<torch::Tensor>> gpu_kv = tensor_list_from_list(gpu_kv_buffer);
  std::vector<std::optional<torch::Tensor>> dst_head = tensor_list_from_list(dst_head_index);
  std::vector<std::optional<torch::Tensor>> ready = tensor_list_from_list(ready_flags);
  std::vector<int64_t> num_gpu_heads_vec(num_gpu_heads.begin(), num_gpu_heads.end());
  std::unique_ptr<kvlib::CPUGatherEngineV3> engine = std::make_unique<kvlib::CPUGatherEngineV3>(
      num_omp_threads, cpu_kv, gpu_kv, dst_head, num_gpu_heads_vec,
      const_cast<torch::Tensor&>(cpu_indices_buffer),
      const_cast<torch::Tensor&>(launch_flag), ready,
      max_batch_size, sink_recent_budget, num_heads, head_dim, debug);
  int64_t h = g_engine_next_handle++;
  std::lock_guard<std::mutex> lock(g_engine_mutex);
  g_engine_map[h] = std::move(engine);
  return h;
}

#else
int64_t create_cpu_gather_engine_v3_impl(
    int64_t, const c10::List<torch::Tensor>&, const c10::List<torch::Tensor>&,
    const c10::List<torch::Tensor>&, c10::IntArrayRef, const torch::Tensor&,
    const torch::Tensor&, const c10::List<torch::Tensor>&,
    int64_t, int64_t, int64_t, int64_t, bool) {
  TORCH_CHECK(false, "CPUGatherEngineV3 requires gdrapi. Install libgdrapi and rebuild.");
  return -1;
}
#endif

}  // namespace

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  // Hamming / score
  m.def("kvlib_hamming_score_norm(Tensor key_code, Tensor query_code, Tensor key_norm, int rbit, "
        "int seq_len, int sink, int recent, bool use_key_norm) -> Tensor");
  m.def("kvlib_hamming_score(Tensor key_code, Tensor query_code, int rbit, int seq_len, "
        "int sink, int recent) -> Tensor");
  m.def("kvlib_hamming_score_head_mask(Tensor key_code, Tensor query_code, Tensor head_mask, "
        "int rbit, int seq_len, int sink, int recent) -> Tensor");
  m.def("kvlib_static_hamming_score_mask(Tensor key_codes, Tensor query_code, Tensor mask, "
        "Tensor score, Tensor seqlen, int rbit, float max_value, float min_value, "
        "int sink, int recent, int skip_sink, int skip_recent) -> ()");
  // Topk
  m.def("kvlib_batch_topk(Tensor data, int k, bool largest) -> Tensor");
  m.def("kvlib_batch_topk_masked(Tensor data, Tensor bh_mask, Tensor out_index, Tensor out_values, "
        "Tensor real_len, Tensor real_k, bool largest) -> ()");
  // Flash-attention decode
  m.def("kvlib_flash_index_decode(Tensor query_states, Tensor key_states, Tensor value_states, "
        "Tensor gather_idx, float scale) -> Tensor[]");
  m.def("kvlib_flash_mixed_decode(Tensor query_states, Tensor cached_keys, Tensor cached_values, "
        "Tensor top_index, Tensor buffer_keys, Tensor buffer_values, Tensor k_head_mask, "
        "Tensor k_head_index, int real_seq_len, float scale) -> Tensor[]");
  m.def("kvlib_flash_decode(Tensor query_states, Tensor key_states, Tensor value_states, "
        "float scale, int real_seq_len) -> Tensor[]");
  // KVCache append
  m.def("kvlib_kvcache_append(Tensor kv_cache, Tensor key, Tensor value, int insert_pos) -> ()");
  m.def("kvlib_kvcache_append_head_sparse(Tensor kv_cache, Tensor key, Tensor value, "
        "Tensor head_ids, int insert_pos) -> ()");
  m.def("kvlib_kvcache_append2(Tensor dst_kv, Tensor src_kv, int dst_pos, int src_pos) -> ()");
  m.def("kvlib_kvcache_append_tensor_pos(Tensor kv_cache, Tensor key, Tensor value, "
        "Tensor insert_pos) -> ()");
  m.def("kvlib_kvcache_append_tensor_pos_head_sparse(Tensor kv_cache, Tensor key, Tensor value, "
        "Tensor head_ids, Tensor insert_pos) -> ()");
  // Prefetch / offload
  m.def("kvlib_real_indices_and_launch_prefetch(Tensor indices, Tensor gpu_gather_mask, "
        "Tensor output, Tensor gather_flag, Tensor cpu_ready_mask, int cache_seq_len, "
        "int batch_size, int num_heads, int layer_idx) -> ()");
  m.def("kvlib_static_launch_prefetch(Tensor gpu_indices, Tensor gpu_gather_mask, "
        "Tensor gpu_index_length, Tensor cpu_indices, Tensor cpu_gather_flag, "
        "Tensor cpu_ready_mask, int batch_size, int max_cache_seqlen, int num_heads, "
        "int layer_idx) -> ()");
  m.def("kvlib_decode_append_offload_wait(Tensor key_states, Tensor value_states, "
        "Tensor gpu_kv_buffer, Tensor cpu_kv_cache, int gpu_append_pos, int cpu_append_pos, "
        "Tensor ready_flags, Tensor cpu_head_ids) -> ()");
  m.def("kvlib_decode_append_offload_tensor_pos_wait(Tensor key_states, Tensor value_states, "
        "Tensor gpu_kv_buffer, Tensor cpu_kv_cache, Tensor gpu_append_pos, Tensor cpu_append_pos, "
        "Tensor ready_flags, Tensor cpu_head_ids) -> ()");
  m.def("kvlib_wait_kv_data(Tensor ready_flags, int batch_size, int num_heads) -> ()");
  m.def("kvlib_gather_gpu_kvcache(Tensor indices, Tensor src_key, Tensor src_value, "
        "Tensor dst_key, Tensor dst_value, Tensor head_ids, int sink_recent_budget) -> ()");
  // Cache reuse / block id
  m.def("kvlib_block_id_to_token_id(Tensor block_idx, int block_size, int num_sink, "
        "int num_recent, int seq_length) -> Tensor");
  m.def("kvlib_block_id_to_token_id_head_mask(Tensor block_idx, int block_size, int num_sink, "
        "int num_recent, int seq_length, Tensor head_mask) -> Tensor");
  // create_tensor: pinned host tensor (for offload). dtype 16=fp16, 32=fp32.
  m.def("kvlib_create_tensor(int[] size, int dtype) -> Tensor");
  // CPUGatherEngineV3 handle API (offload)
  m.def("kvlib_create_cpu_gather_engine_v3(int num_omp_threads, Tensor[] cpu_kv_data, "
        "Tensor[] gpu_kv_buffer, Tensor[] dst_head_index, int[] num_gpu_heads, "
        "Tensor cpu_indices_buffer, Tensor launch_flag, Tensor[] ready_flags, "
        "int max_batch_size, int sink_recent_budget, int num_heads, int head_dim, bool debug) -> int");
}

TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("kvlib_hamming_score_norm", [](torch::Tensor key_code, torch::Tensor query_code,
                                         torch::Tensor key_norm, int64_t rbit, int64_t seq_len,
                                         int64_t sink, int64_t recent, bool use_key_norm) {
    return kvlib::HammingScoreNormCUDA(key_code, query_code, key_norm, static_cast<int32_t>(rbit),
                                       static_cast<int32_t>(seq_len), static_cast<int32_t>(sink),
                                       static_cast<int32_t>(recent), use_key_norm);
  });
  m.impl("kvlib_hamming_score", [](torch::Tensor key_code, torch::Tensor query_code, int64_t rbit,
                                   int64_t seq_len, int64_t sink, int64_t recent) {
    return kvlib::HammingScoreCUDA(key_code, query_code, static_cast<int32_t>(rbit),
                                   static_cast<int32_t>(seq_len), static_cast<int32_t>(sink),
                                   static_cast<int32_t>(recent));
  });
  m.impl("kvlib_hamming_score_head_mask",
         [](torch::Tensor key_code, torch::Tensor query_code, torch::Tensor head_mask,
            int64_t rbit, int64_t seq_len, int64_t sink, int64_t recent) {
           return kvlib::HammingScoreHeadMaskCUDA(
               key_code, query_code, head_mask, static_cast<int32_t>(rbit),
               static_cast<int32_t>(seq_len), static_cast<int32_t>(sink),
               static_cast<int32_t>(recent));
         });
  m.impl("kvlib_static_hamming_score_mask",
         [](torch::Tensor key_codes,
            torch::Tensor query_code,
            torch::Tensor mask,
            torch::Tensor score,
            torch::Tensor seqlen,
            int64_t rbit,
            double max_value,
            double min_value,
            int64_t sink,
            int64_t recent,
            int64_t skip_sink,
            int64_t skip_recent) {
           kvlib::StaticHammingScoreMaskCUDA(
               key_codes,
               query_code,
               mask,
               score,
               seqlen,
               static_cast<int32_t>(rbit),
               static_cast<float>(max_value),
               static_cast<float>(min_value),
               static_cast<int32_t>(sink),
               static_cast<int32_t>(recent),
               static_cast<int32_t>(skip_sink),
               static_cast<int32_t>(skip_recent));
         });
  m.impl("kvlib_flash_index_decode",
         [](torch::Tensor query_states,
            torch::Tensor key_states,
            torch::Tensor value_states,
            torch::Tensor gather_idx,
            double scale) {
           return kvlib::mha_index_decode_fwd(
               query_states,
               key_states,
               value_states,
               gather_idx,
               static_cast<float>(scale));
         });
  m.impl("kvlib_flash_mixed_decode",
         [](torch::Tensor query_states,
            torch::Tensor cached_keys,
            torch::Tensor cached_values,
            torch::Tensor top_index,
            torch::Tensor buffer_keys,
            torch::Tensor buffer_values,
            torch::Tensor k_head_mask,
            torch::Tensor k_head_index,
            int64_t real_seq_len,
            double scale) {
           return kvlib::mha_mixed_decode_fwd(
               query_states,
               cached_keys,
               cached_values,
               top_index,
               buffer_keys,
               buffer_values,
               k_head_mask,
               k_head_index,
               static_cast<int>(real_seq_len),
               static_cast<float>(scale));
         });
  m.impl("kvlib_flash_decode",
         [](torch::Tensor query_states,
            torch::Tensor key_states,
            torch::Tensor value_states,
            double scale,
            int64_t real_seq_len) {
           return kvlib::mha_decode_fwd(
               query_states,
               key_states,
               value_states,
               static_cast<float>(scale),
               static_cast<int32_t>(real_seq_len));
         });
#if defined(KVLIB_RAFT_AVAILABLE)
  m.impl("kvlib_batch_topk",
         [](torch::Tensor data, int64_t k, bool largest) {
           return kvlib::TopkCUDA(data, static_cast<int32_t>(k), largest);
         });
  m.impl("kvlib_batch_topk_masked", &kvlib::TopkMaskedCUDA);
#else
  m.impl("kvlib_batch_topk", batch_topk_impl);
  m.impl("kvlib_batch_topk_masked", batch_topk_masked_impl);
#endif
  m.impl("kvlib_kvcache_append",
         [](torch::Tensor kv_cache,
            torch::Tensor key,
            torch::Tensor value,
            int64_t insert_pos) {
           kvlib::KVCacheAppend(
               kv_cache,
               key,
               value,
               static_cast<int32_t>(insert_pos));
         });
  m.impl("kvlib_kvcache_append_head_sparse",
         [](torch::Tensor kv_cache,
            torch::Tensor key,
            torch::Tensor value,
            torch::Tensor head_ids,
            int64_t insert_pos) {
           kvlib::KVCacheAppendHeadSparse(
               kv_cache,
               key,
               value,
               head_ids,
               static_cast<int32_t>(insert_pos));
         });
  m.impl("kvlib_kvcache_append2",
         [](torch::Tensor dst_kv,
            torch::Tensor src_kv,
            int64_t dst_pos,
            int64_t src_pos) {
           kvlib::KVCacheAppend2(
               dst_kv,
               src_kv,
               static_cast<int32_t>(dst_pos),
               static_cast<int32_t>(src_pos));
         });
  m.impl("kvlib_kvcache_append_tensor_pos",
         [](torch::Tensor kv_cache,
            torch::Tensor key,
            torch::Tensor value,
            torch::Tensor insert_pos) {
           kvlib::KVCacheAppendTensorPos(
               kv_cache,
               key,
               value,
               insert_pos);
         });
  m.impl("kvlib_kvcache_append_tensor_pos_head_sparse",
         [](torch::Tensor kv_cache,
            torch::Tensor key,
            torch::Tensor value,
            torch::Tensor head_ids,
            torch::Tensor insert_pos) {
           kvlib::KVCacheAppendTensorPosHeadSparse(
               kv_cache,
               key,
               value,
               head_ids,
               insert_pos);
         });
  m.impl("kvlib_real_indices_and_launch_prefetch",
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
         });
  m.impl("kvlib_static_launch_prefetch",
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
         });
  m.impl("kvlib_decode_append_offload_wait",
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
         });
  m.impl("kvlib_decode_append_offload_tensor_pos_wait",
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
         });
  m.impl("kvlib_wait_kv_data",
         [](torch::Tensor ready_flags,
            int64_t batch_size,
            int64_t num_heads) {
           kvlib::WaitKVData(
               ready_flags,
               batch_size,
               num_heads);
         });
  m.impl("kvlib_gather_gpu_kvcache",
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
         });
  m.impl("kvlib_block_id_to_token_id",
         [](torch::Tensor block_idx,
            int64_t block_size,
            int64_t num_sink,
            int64_t num_recent,
            int64_t seq_length) {
           return kvlib::BlockIdx2TokenIdx(
               block_idx,
               block_size,
               num_sink,
               num_recent,
               seq_length);
         });
  m.impl("kvlib_block_id_to_token_id_head_mask",
         [](torch::Tensor block_idx,
            int64_t block_size,
            int64_t num_sink,
            int64_t num_recent,
            int64_t seq_length,
            torch::Tensor head_mask) {
           return kvlib::BlockIdx2TokenIdxHeadMask(
               block_idx,
               block_size,
               num_sink,
               num_recent,
               seq_length,
               head_mask);
         });
}

TORCH_LIBRARY_IMPL(sgl_kernel, CPU, m) {
  m.impl("kvlib_create_tensor", create_tensor_impl);
  m.impl("kvlib_create_cpu_gather_engine_v3", create_cpu_gather_engine_v3_impl);
}
