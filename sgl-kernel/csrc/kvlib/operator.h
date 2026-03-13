#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

namespace kvlib {

torch::Tensor HammingScoreNormCUDA(torch::Tensor& key_codes,
                                   torch::Tensor& query_code,
                                   torch::Tensor& key_norms, int32_t rbit,
                                   int32_t seq_len, int32_t sink,
                                   int32_t recent, bool use_key_norm);
torch::Tensor HammingScoreCUDA(torch::Tensor& key_codes,
                               torch::Tensor& query_code, int32_t rbit,
                               int32_t seq_len, int32_t sink, int32_t recent);
torch::Tensor HammingScoreHeadMaskCUDA(torch::Tensor& key_codes,
                                       torch::Tensor& query_code,
                                       torch::Tensor& mask, int32_t rbit,
                                       int32_t seq_len, int32_t sink,
                                       int32_t recent);
torch::Tensor TopkCUDA(torch::Tensor& data, int32_t k, bool largest);
torch::Tensor combine_attention(torch::Tensor attn1, torch::Tensor lse1,
                                torch::Tensor attn2, torch::Tensor lse2);
void KVCacheAppend(torch::Tensor kv_cache_tensor, torch::Tensor key_tensor,
                   torch::Tensor value_tensor, int32_t insert_pos);
void KVCacheAppend2(torch::Tensor dst_kv_cache_tensor,
                    torch::Tensor src_kv_cache_tensor, int32_t dst_pos,
                    int32_t src_pos);
void KVCacheAppendTensorPos(torch::Tensor kv_cache_tensor, torch::Tensor key_tensor,
                            torch::Tensor value_tensor, torch::Tensor insert_pos);
void KVCacheAppendTensorPosHeadSparse(
  torch::Tensor kv_cache_tensor, torch::Tensor key_tensor,
  torch::Tensor value_tensor, torch::Tensor head_ids,
  torch::Tensor insert_pos);
void KVCacheAppendHeadSparse(torch::Tensor kv_cache_tensor,
                             torch::Tensor key_tensor,
                             torch::Tensor value_tensor, torch::Tensor head_ids,
                             int32_t insert_pos);
void RealInndicesAndLaunchPrefetching(torch::Tensor& indices,
                                      torch::Tensor& gpu_gather_mask,
                                      torch::Tensor& output,
                                      torch::Tensor& gather_flag,
                                      torch::Tensor& cpu_ready_mask,
                                      int64_t cache_seq_len, int64_t batch_size,
                                      int64_t num_heads, int64_t layer_idx);
void AppendOffloadWait(torch::Tensor& key_states, torch::Tensor& value_states,
                       torch::Tensor& gpu_kv_buffer,
                       torch::Tensor& cpu_kv_cache, int32_t gpu_append_pos,
                       int32_t cpu_append_pos, torch::Tensor& ready_flags,
                       torch::Tensor& cpu_head_ids);
void AppendOffloadTensorPosAndWait(
  torch::Tensor& key_states,
  torch::Tensor& value_states,
  torch::Tensor& gpu_kv_buffer,
  torch::Tensor& cpu_kv_cache,
  torch::Tensor& gpu_append_pos,
  torch::Tensor& cpu_append_pos,
  torch::Tensor& ready_flags,
  torch::Tensor& cpu_head_ids);
void TopkMaskedCUDA(torch::Tensor& data, torch::Tensor& bh_mask,
                    torch::Tensor& out_index, torch::Tensor& out_values,
                    torch::Tensor& real_len, torch::Tensor& real_k,
                    bool largest);
void WaitKVData(torch::Tensor& ready_flags, int64_t batch_size,
                int64_t num_heads);
void GatherGPUKVCache(torch::Tensor& indices, torch::Tensor& src_key,
                      torch::Tensor& src_value, torch::Tensor& dst_key,
                      torch::Tensor& dst_value, torch::Tensor& head_ids,
                      int64_t sink_recent_budget);
torch::Tensor BlockIdx2TokenIdx(torch::Tensor& block_idx, int64_t block_size,
                                int64_t num_sink, int64_t num_recent,
                                int64_t seq_length);
torch::Tensor BlockIdx2TokenIdxHeadMask(torch::Tensor& block_idx,
                                        int64_t block_size, int64_t num_sink,
                                        int64_t num_recent, int64_t seq_length,
                                        torch::Tensor& head_mask);
void StaticHammingScoreMaskCUDA(
  torch::Tensor& key_codes,
  torch::Tensor& query_code,
  torch::Tensor& mask,
  torch::Tensor& score,
  torch::Tensor& seqlen,
  int32_t rbit,
  float max_value,
  float min_value,
  int32_t sink,
  int32_t recent,
  int32_t skip_sink,
  int32_t skip_recent
);
void StaticLaunchPrefetching(torch::Tensor& gpu_indices,  // [b * h, max_topk + sink + recent + 1]
                             torch::Tensor& gpu_gather_mask,  // [b * h]
                             torch::Tensor& gpu_index_length,  // [1, ]
                             torch::Tensor& cpu_indices,  // [b * h, max_topk]
                             torch::Tensor& cpu_gather_flag,  // [6, ]
                             torch::Tensor& cpu_ready_mask,  // [b * h]
                             int64_t batch_size, int64_t max_cache_seqlen,
                             int64_t num_heads, int64_t layer_idx);

}  // namespace kvlib