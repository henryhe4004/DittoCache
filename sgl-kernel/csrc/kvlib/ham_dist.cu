#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include "cp_async.cuh"
#include "operator.h"

#define HEAD_SWITCH(val, NumHead, ...)        \
  do {                                        \
    if ((val) == 32) {                        \
      constexpr int NumHead = 32;             \
      { __VA_ARGS__ }                         \
    } else if ((val) == 40) {                 \
      constexpr int NumHead = 40;             \
      { __VA_ARGS__ }                         \
    } else if ((val) == 28) {                 \
      constexpr int NumHead = 28;             \
      { __VA_ARGS__ }                         \
    } else if ((val) == 8) {                  \
      constexpr int NumHead = 8;              \
      { __VA_ARGS__ }                         \
    } else if ((val) == 4) {                  \
      constexpr int NumHead = 4;              \
      { __VA_ARGS__ }                         \
    } else if ((val) == 2) {                  \
      constexpr int NumHead = 2;              \
      { __VA_ARGS__ }                         \
    } else {                                  \
      LOG(FATAL) << "NumHead is not support"; \
    }                                         \
  } while (0);

#define KVHEAD_SWITCH(val, NumKVHead, ...)      \
  do {                                          \
    if ((val) == 32) {                          \
      constexpr int NumKVHead = 32;             \
      { __VA_ARGS__ }                           \
    } else if ((val) == 8) {                    \
      constexpr int NumKVHead = 8;              \
      { __VA_ARGS__ }                           \
    } else if ((val) == 4) {                    \
      constexpr int NumKVHead = 4;              \
      { __VA_ARGS__ }                           \
    } else if ((val) == 2) {                    \
      constexpr int NumKVHead = 2;              \
      { __VA_ARGS__ }                           \
    } else {                                    \
      LOG(FATAL) << "NumKVHead is not support"; \
    }                                           \
  } while (0);

#define KVGROUP_SWITCH(val, KVGroup, ...)     \
  do {                                        \
    if ((val) == 1) {                         \
      constexpr int KVGroup = 1;              \
      { __VA_ARGS__ }                         \
    } else if ((val) == 4) {                  \
      constexpr int KVGroup = 4;              \
      { __VA_ARGS__ }                         \
    } else if ((val) == 5) {                  \
      constexpr int KVGroup = 5;              \
      { __VA_ARGS__ }                         \
    } else if ((val) == 7) {                  \
      constexpr int KVGroup = 7;              \
      { __VA_ARGS__ }                         \
    } else if ((val) == 8) {                  \
      constexpr int KVGroup = 8;              \
      { __VA_ARGS__ }                         \
    } else if ((val) == 16) {                 \
      constexpr int KVGroup = 16;             \
      { __VA_ARGS__ }                         \
    } else {                                  \
      LOG(FATAL) << "KVGroup is not support"; \
    }                                         \
  } while (0);

#define NUMCHUNK_SWITCH(val, NumChunk, ...)    \
  do {                                         \
    if ((val) == 16) {                         \
      constexpr int NumChunk = 16;             \
      { __VA_ARGS__ }                          \
    } else if ((val) == 8) {                   \
      constexpr int NumChunk = 8;              \
      { __VA_ARGS__ }                          \
    } else if ((val) == 4) {                   \
      constexpr int NumChunk = 4;              \
      { __VA_ARGS__ }                          \
    } else if ((val) == 3) {                   \
      constexpr int NumChunk = 3;              \
      { __VA_ARGS__ }                          \
    } else if ((val) == 2) {                   \
      constexpr int NumChunk = 2;              \
      { __VA_ARGS__ }                          \
    } else if ((val) == 1) {                   \
      constexpr int NumChunk = 1;              \
      { __VA_ARGS__ }                          \
    } else {                                   \
      LOG(FATAL) << "NumChunk is not support"; \
    }                                          \
  } while (0);

namespace kvlib {

template <typename T, bool USE_INT64, int32_t NumThreads, int32_t ELEMS,
          int32_t NumHead, int32_t NumKVHead, int32_t NumChunk>
__global__ void HammingScoreNormKernel(
    void* __restrict__ keys_ptr, void* __restrict__ query_ptr,
    half* __restrict__ output_ptr, int32_t BSZ, int32_t seq,
    half* __restrict__ key_norms, int32_t batch_key_code_stride,
    int32_t batch_key_norms_stride, bool USE_KEY_NORM, int32_t SINK,
    int32_t RECENT) {
  assert(NumThreads == blockDim.x);
  constexpr int32_t BLOCK_M = (ELEMS / NumChunk / NumKVHead);
  constexpr int32_t KVGroup = NumHead / NumKVHead;

  const half RBIT = (half)((int32_t)sizeof(T) * 8 * NumChunk);

  const int32_t tid = threadIdx.x;
  const int32_t batch_id = blockIdx.y;

  const int32_t block_id = blockIdx.x;
  const int32_t block_start_elem = block_id * ELEMS;
  const int32_t block_start_m = block_id * BLOCK_M;
  const int32_t left_m = min(BLOCK_M, seq - block_start_m);
  const int32_t left_elem = left_m * NumKVHead * NumChunk;

  extern __shared__ int32_t smem[];
  int32_t* smem_popc = smem;
  T* smem_query_code = (T*)(smem_popc + NumThreads);
  half* smem_key_norm = (half*)(smem_query_code + NumHead * NumChunk);

  T* q_ptr = (T*)query_ptr + batch_id * 1 * NumHead * NumChunk;
// load q to smem
#pragma unroll
  for (int i = tid; i < NumHead * NumChunk; i += NumThreads) {
    smem_query_code[i] = q_ptr[i];
  }

  if (USE_KEY_NORM) {
    half* key_norm_ptr = (half*)key_norms + batch_id * batch_key_norms_stride +
                         block_start_m * NumKVHead;

#pragma unroll
    for (int i = (NumThreads - 1 - tid); i < left_m * NumKVHead;
         i += NumThreads) {
      smem_key_norm[i] = key_norm_ptr[i];
    }
  }
  __syncthreads();

  int32_t score = 0;
  if (tid < left_elem) {
    T* k_ptr =
        (T*)keys_ptr + batch_id * batch_key_code_stride + block_start_elem;

    const int32_t kv_head_id = (tid / NumChunk) % NumKVHead;
    const int32_t kv_chunk_id = tid % NumChunk;

    T key = k_ptr[tid];
    T* query = smem_query_code + kv_head_id * KVGroup * NumChunk + kv_chunk_id;

#pragma unroll
    for (int i = 0; i < KVGroup; i++) {
      T q = *(query + i * NumChunk);
      T tmp = key ^ q;
      score += USE_INT64 ? __popcll(tmp) : __popc(tmp);
    }
  }

  smem_popc[tid] = score;
  __syncthreads();

  // write results to global memory
  if (tid < NumKVHead * BLOCK_M) {
    int kv_head_id = tid / BLOCK_M;
    int m_id = tid % BLOCK_M;
    bool is_sink_or_recent = (m_id + block_start_m) < SINK ||
                             (m_id + block_start_m) >= (seq - RECENT);

    if (m_id < left_m) {
      // transpose
      half* o_ptr =
          (half*)output_ptr + batch_id * NumKVHead * seq + block_start_m;
      half* _o_ptr = o_ptr + kv_head_id * seq + m_id;
      half sum = (half)(0.);

#pragma unroll
      for (int h = 0; h < NumChunk; h += 1) {
        sum +=
            (half)(smem_popc[(m_id * NumKVHead + kv_head_id) * NumChunk + h]);
      }
      if (USE_KEY_NORM) {
        sum = half(KVGroup) - half(2.) * sum / RBIT;
        sum = sum * smem_key_norm[m_id * NumKVHead + kv_head_id];
        *_o_ptr = is_sink_or_recent ? half(65504) : sum;
      } else {
        *_o_ptr = is_sink_or_recent ? half(0.) : sum;
      }
    }
  }
}

torch::Tensor HammingScoreNormCUDA(torch::Tensor& key_codes,
                                   torch::Tensor& query_code,
                                   torch::Tensor& key_norms, int32_t rbit,
                                   int32_t seq_len, int32_t sink,
                                   int32_t recent, bool use_key_norm) {
  // shape for key_codes is (BATCH_SIZE, seq, #NUM_K_HEAD, num_chunk) and dtype
  // is int32 shape for query_code is (BATCH_SIZE, 1, #NUM_HEAD, num_chunk) and
  // dtype is int32 shape for key_norms is (BATCH_SIZE, XXX, #NUM_K_HEAD), dtype
  // is float16

  // shape for output is (BATCH_SIZE, #NUM_KV_HEAD, seq) [transpose head and seq
  // here] and dtype is float16

  int32_t bsz = key_codes.size(0);
  int32_t num_kv_head = key_codes.size(2);
  int32_t num_chunk = key_codes.size(3);

  int32_t num_head = query_code.size(2);

  int32_t batch_key_norm_stride = key_norms.stride(0);
  int32_t batch_key_code_stride = key_codes.stride(0);

  auto device = key_codes.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);
  torch::Tensor output = torch::empty({bsz, num_kv_head, seq_len}, options);

  HEAD_SWITCH(num_head, NumHead, {
    KVHEAD_SWITCH(num_kv_head, NumKVHead, {
      NUMCHUNK_SWITCH(num_chunk, NumChunk, {
        constexpr int32_t NumThreads = 512;
        size_t shm_size = 0;
        shm_size += NumThreads * sizeof(int32_t);          // for popc results
        shm_size += NumHead * NumChunk * sizeof(int32_t);  // for query_code

        if (NumChunk % 2 == 0) {
          // convert int32 to int64
          constexpr int32_t HalfNumChunk = NumChunk / 2 < 1 ? 1 : NumChunk / 2;
          constexpr int32_t NumTokens = NumThreads / (NumKVHead * HalfNumChunk);
          constexpr int32_t ELEMS_PER_BLOCK =
              NumKVHead * HalfNumChunk * NumTokens;

          shm_size += NumTokens * NumKVHead * sizeof(half);  // for key norm

          dim3 blks(NumThreads);
          dim3 grids(
              (seq_len * NumKVHead * HalfNumChunk + ELEMS_PER_BLOCK - 1) /
                  ELEMS_PER_BLOCK,
              bsz);

          HammingScoreNormKernel<int64_t, true, NumThreads, ELEMS_PER_BLOCK,
                                 NumHead, NumKVHead, HalfNumChunk>
              <<<grids, blks, shm_size, stream>>>(
                  key_codes.data_ptr(), query_code.data_ptr(),
                  (half*)(output.data_ptr<at::Half>()), bsz, seq_len,
                  (half*)(key_norms.data_ptr<at::Half>()),
                  (batch_key_code_stride / 2), batch_key_norm_stride,
                  use_key_norm, sink, recent);

        } else {
          constexpr int32_t NumTokens = NumThreads / (NumKVHead * NumChunk);
          constexpr int32_t ELEMS_PER_BLOCK = NumKVHead * NumChunk * NumTokens;

          shm_size += NumTokens * NumKVHead * sizeof(half);  // for key norm

          dim3 blks(NumThreads);
          dim3 grids((seq_len * NumKVHead * NumChunk + ELEMS_PER_BLOCK - 1) /
                         ELEMS_PER_BLOCK,
                     bsz);

          HammingScoreNormKernel<int32_t, false, NumThreads, ELEMS_PER_BLOCK,
                                 NumHead, NumKVHead, NumChunk>
              <<<grids, blks, shm_size, stream>>>(
                  key_codes.data_ptr(), query_code.data_ptr(),
                  (half*)(output.data_ptr<at::Half>()), bsz, seq_len,
                  (half*)(key_norms.data_ptr<at::Half>()),
                  batch_key_code_stride, batch_key_norm_stride, use_key_norm,
                  sink, recent);
        }
      });
    });
  });

  return output;
}

template <typename T, bool USE_INT64, int32_t NumThreads, int32_t ELEMS,
          int32_t NumHead, int32_t NumKVHead, int32_t NumChunk>
__global__ void HammingScoreKernel(void* __restrict__ keys_ptr,
                                   void* __restrict__ query_ptr,
                                   half* __restrict__ output_ptr, int32_t BSZ,
                                   int32_t seq, int32_t batch_key_code_stride,
                                   int32_t sink, int32_t recent) {
  assert(NumThreads == blockDim.x);
  constexpr int32_t BLOCK_M = (ELEMS / NumChunk / NumKVHead);
  // constexpr int32_t ELEMS_IN_TOKEN = NumChunk * NumKVHead;
  constexpr int32_t KVGroup = NumHead / NumKVHead;

  const half RBIT = (half)((int32_t)sizeof(T) * 8 * NumChunk);

  const int32_t tid = threadIdx.x;
  const int32_t batch_id = blockIdx.y;

  const int32_t block_id = blockIdx.x;
  const int32_t block_start_elem = block_id * ELEMS;
  const int32_t block_start_m = block_id * BLOCK_M;
  const int32_t left_m = min(BLOCK_M, seq - block_start_m);
  const int32_t left_elem = left_m * NumKVHead * NumChunk;

  extern __shared__ int32_t smem[];
  int32_t* smem_popc = smem;
  T* smem_query_code = (T*)(smem_popc + NumThreads);

  T* q_ptr = (T*)query_ptr + batch_id * 1 * NumHead * NumChunk;
// load q to smem
#pragma unroll
  for (int i = tid; i < NumHead * NumChunk; i += NumThreads) {
    smem_query_code[i] = q_ptr[i];
  }

  __syncthreads();

  int32_t score = 0;
  if (tid < left_elem) {
    T* k_ptr =
        (T*)keys_ptr + batch_id * batch_key_code_stride + block_start_elem;

    const int32_t kv_head_id = (tid / NumChunk) % NumKVHead;
    const int32_t kv_chunk_id = tid % NumChunk;

    T key = k_ptr[tid];
    T* query = smem_query_code + kv_head_id * KVGroup * NumChunk + kv_chunk_id;

#pragma unroll
    for (int i = 0; i < KVGroup; i++) {
      T q = *(query + i * NumChunk);
      T tmp = key ^ q;
      score += USE_INT64 ? __popcll(tmp) : __popc(tmp);
    }
  }

  smem_popc[tid] = score;
  __syncthreads();

  // write results to global memory
  if (tid < NumKVHead * BLOCK_M) {
    int kv_head_id = tid / BLOCK_M;
    int m_id = tid % BLOCK_M;
    

    if (m_id < left_m) {
      // transpose
      half* o_ptr =
          (half*)output_ptr + batch_id * NumKVHead * seq + block_start_m;
      half* _o_ptr = o_ptr + kv_head_id * seq + m_id;
      half sum = (half)(0.);
      bool is_sink_or_recent = (m_id + block_start_m) < sink ||
                             (m_id + block_start_m) >= (seq - recent);

#pragma unroll
      for (int h = 0; h < NumChunk; h += 1) {
        sum +=
            (half)(smem_popc[(m_id * NumKVHead + kv_head_id) * NumChunk + h]);
      }
      *_o_ptr = is_sink_or_recent ? half(0.) : sum;
    }
  }
}

torch::Tensor HammingScoreCUDA(torch::Tensor& key_codes,
                               torch::Tensor& query_code, int32_t rbit,
                               int32_t seq_len, int32_t sink, int32_t recent) {
  // shape for key_codes is (BATCH_SIZE, seq, #NUM_K_HEAD, num_chunk) and dtype
  // is int32
  // shape for query_code is (BATCH_SIZE, 1, #NUM_HEAD, num_chunk)
  // shape for output is (BATCH_SIZE, #NUM_KV_HEAD, seq) [transpose head and seq
  // here] and dtype is float16

  int32_t bsz = key_codes.size(0);
  int32_t num_kv_head = key_codes.size(2);
  int32_t num_chunk = key_codes.size(3);

  int32_t num_head = query_code.size(2);

  int32_t batch_key_code_stride = key_codes.stride(0);

  // int32_t kv_group = num_head / num_kv_head;

  auto device = key_codes.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);
  torch::Tensor output = torch::empty({bsz, num_kv_head, seq_len}, options);

  HEAD_SWITCH(num_head, NumHead, {
    KVHEAD_SWITCH(num_kv_head, NumKVHead, {
      NUMCHUNK_SWITCH(num_chunk, NumChunk, {
        constexpr int32_t NumThreads = 512;
        size_t shm_size = 0;
        shm_size += NumThreads * sizeof(int32_t);          // for popc results
        shm_size += NumHead * NumChunk * sizeof(int32_t);  // for query_code

        if (NumChunk % 2 == 0) {
          // convert int32 to int64
          constexpr int32_t HalfNumChunk = NumChunk / 2 < 1 ? 1 : NumChunk / 2;
          constexpr int32_t NumTokens = NumThreads / (NumKVHead * HalfNumChunk);
          constexpr int32_t ELEMS_PER_BLOCK =
              NumKVHead * HalfNumChunk * NumTokens;

          dim3 blks(NumThreads);
          dim3 grids(
              (seq_len * NumKVHead * HalfNumChunk + ELEMS_PER_BLOCK - 1) /
                  ELEMS_PER_BLOCK,
              bsz);

          HammingScoreKernel<int64_t, true, NumThreads, ELEMS_PER_BLOCK,
                             NumHead, NumKVHead, HalfNumChunk>
              <<<grids, blks, shm_size, stream>>>(
                  key_codes.data_ptr(), query_code.data_ptr(),
                  (half*)(output.data_ptr<at::Half>()), bsz, seq_len,
                  (batch_key_code_stride / 2), sink, recent);

        } else {
          constexpr int32_t NumTokens = NumThreads / (NumKVHead * NumChunk);
          constexpr int32_t ELEMS_PER_BLOCK = NumKVHead * NumChunk * NumTokens;

          dim3 blks(NumThreads);
          dim3 grids((seq_len * NumKVHead * NumChunk + ELEMS_PER_BLOCK - 1) /
                         ELEMS_PER_BLOCK,
                     bsz);

          HammingScoreKernel<int32_t, false, NumThreads, ELEMS_PER_BLOCK,
                             NumHead, NumKVHead, NumChunk>
              <<<grids, blks, shm_size, stream>>>(
                  key_codes.data_ptr(), query_code.data_ptr(),
                  (half*)(output.data_ptr<at::Half>()), bsz, seq_len,
                  batch_key_code_stride, sink, recent);
        }
      });
    });
  });

  return output;
}

template <typename T, bool USE_INT64, int32_t NumThreads,
          int32_t ELEMS_PER_BLOCK, int32_t NumHead, int32_t NumKVHead,
          int32_t KVGroup, int32_t NumChunk>
__global__ void HammingScoreHeadMaskKernel(
    void* __restrict__ keys_ptr, void* __restrict__ query_ptr,
    half* __restrict__ output_ptr, bool* __restrict__ mask_ptr, int32_t seq,
    int64_t key_code_batch_stride, int64_t key_code_seq_stride,
    int64_t key_code_head_stride, int32_t sink, int32_t recent) {
  assert(NumThreads == blockDim.x);
  constexpr int32_t BLOCK_SEQ = (ELEMS_PER_BLOCK / NumChunk);

  const int32_t head_idx = blockIdx.y;
  if (*(mask_ptr + head_idx)) {
    const int32_t tid = threadIdx.x;
    const int32_t batch_id = head_idx / NumKVHead;
    const int32_t kv_head_id = head_idx % NumKVHead;

    const int32_t block_id = blockIdx.x;
    const int32_t block_start_seq = block_id * BLOCK_SEQ;
    const int32_t left_seq = min(BLOCK_SEQ, seq - block_start_seq);
    const int32_t left_elem = left_seq * NumChunk;

    extern __shared__ int32_t smem[];
    int32_t* smem_popc = smem;
    T* smem_query_code = (T*)(smem_popc + NumThreads);

    // load q (of this gqa group) to smem
    T* q_ptr = (T*)query_ptr + batch_id * NumHead * NumChunk +
               kv_head_id * KVGroup * NumChunk;
#pragma unroll
    for (int i = tid; i < KVGroup * NumChunk; i += NumThreads) {
      smem_query_code[i] = q_ptr[i];
    }
    __syncthreads();

    // compute score
    int32_t score = 0;
    if (tid < left_elem) {
      const int32_t seq_id = block_start_seq + tid / NumChunk;
      const int32_t chunk_id = tid % NumChunk;
      T* k_ptr = (T*)keys_ptr + batch_id * key_code_batch_stride +
                 seq_id * key_code_seq_stride +
                 kv_head_id * key_code_head_stride + chunk_id;
      T* q_smem_ptr = smem_query_code + chunk_id;
      T key = *k_ptr;
#pragma unroll
      for (int i = 0; i < KVGroup; i++) {
        T q = *(q_smem_ptr + i * NumChunk);
        T tmp = key ^ q;
        score += USE_INT64 ? __popcll(tmp) : __popc(tmp);
      }
    }
    smem_popc[tid] = score;
    __syncthreads();

    // write results to global memory
    if (tid < left_seq) {
      // transpose
      bool is_sink_or_recent = (block_start_seq + tid) < sink ||
                               (block_start_seq + tid) >= (seq - recent);
      half* o_ptr =
          (half*)output_ptr + head_idx * seq + (block_start_seq + tid);
      half sum = (half)(0.);

#pragma unroll
      for (int h = 0; h < NumChunk; h += 1) {
        sum += (half)(smem_popc[tid * NumChunk + h]);
      }
      *o_ptr = is_sink_or_recent ? half(0.) : sum;
    }
  } else {
    return;
  }
}

torch::Tensor HammingScoreHeadMaskCUDA(torch::Tensor& key_codes,
                                       torch::Tensor& query_code,
                                       torch::Tensor& mask, int32_t rbit,
                                       int32_t seq_len, int32_t sink,
                                       int32_t recent) {
  // shape for key_codes is (BATCH_SIZE, seq, #NUM_K_HEAD, num_chunk) and dtype
  // is int32
  // shape for query_code is (BATCH_SIZE, 1, #NUM_HEAD, num_chunk)
  // shape for output is (BATCH_SIZE, #NUM_KV_HEAD, seq) [transpose head and seq
  // here] and dtype is float16

  int32_t bsz = key_codes.size(0);
  int32_t num_kv_head = key_codes.size(2);
  int32_t num_chunk = key_codes.size(3);
  int32_t num_head = query_code.size(2);
  int32_t gqa_size = num_head / num_kv_head;
  int32_t total_num_heads = bsz * num_kv_head;

  auto device = key_codes.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);
  torch::Tensor output = torch::empty({bsz, num_kv_head, seq_len}, options);

  HEAD_SWITCH(num_head, NumHead, {
    KVHEAD_SWITCH(num_kv_head, NumKVHead, {
      KVGROUP_SWITCH(gqa_size, KVGroup, {
        NUMCHUNK_SWITCH(num_chunk, NumChunk, {
          constexpr int32_t NumThreads = 256;
          size_t shm_size = 0;
          shm_size += NumThreads * sizeof(int32_t);          // for popc results
          shm_size += KVGroup * NumChunk * sizeof(int32_t);  // for query_code

          if (NumChunk % 2 == 0) {
            // convert int32 to int64
            constexpr int32_t HalfNumChunk =
                NumChunk / 2 < 1 ? 1 : NumChunk / 2;
            constexpr int32_t NumTokens = NumThreads / HalfNumChunk;
            constexpr int32_t ELEMS_PER_BLOCK = HalfNumChunk * NumTokens;
            dim3 blks(NumThreads);
            dim3 grids((seq_len * HalfNumChunk + ELEMS_PER_BLOCK - 1) /
                           ELEMS_PER_BLOCK,
                       total_num_heads);
            HammingScoreHeadMaskKernel<int64_t, true, NumThreads,
                                       ELEMS_PER_BLOCK, NumHead, NumKVHead,
                                       KVGroup, HalfNumChunk>
                <<<grids, blks, shm_size, stream>>>(
                    key_codes.data_ptr(), query_code.data_ptr(),
                    (half*)(output.data_ptr<at::Half>()), mask.data_ptr<bool>(),
                    seq_len, key_codes.stride(0) / 2, key_codes.stride(1) / 2,
                    key_codes.stride(2) / 2, sink, recent);

          } else {
            constexpr int32_t NumTokens = NumThreads / NumChunk;
            constexpr int32_t ELEMS_PER_BLOCK = NumChunk * NumTokens;
            dim3 blks(NumThreads);
            dim3 grids(
                (seq_len * NumChunk + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK,
                total_num_heads);
            HammingScoreHeadMaskKernel<int32_t, false, NumThreads,
                                       ELEMS_PER_BLOCK, NumHead, NumKVHead,
                                       KVGroup, NumChunk>
                <<<grids, blks, shm_size, stream>>>(
                    key_codes.data_ptr(), query_code.data_ptr(),
                    (half*)(output.data_ptr<at::Half>()), mask.data_ptr<bool>(),
                    seq_len, key_codes.stride(0), key_codes.stride(1),
                    key_codes.stride(2), sink, recent);
          }
        });
      });
    });
  });

  return output;
}

template <typename T, bool USE_INT64, int32_t NumThreads,
          int32_t ELEMS_PER_BLOCK, int32_t NumHead, int32_t NumKVHead,
          int32_t KVGroup, int32_t NumChunk>
__global__ void StaticHammingScoreMaskKernel(
    void* __restrict__ keys_ptr, void* __restrict__ query_ptr,
    half* __restrict__ output_ptr, bool* __restrict__ mask_ptr,
    int32_t* __restrict__ seqlen_ptr,
    half max_value,
    half min_value,
    int64_t key_code_batch_stride,
    int64_t key_code_seq_stride,
    int64_t key_code_head_stride,
    int32_t max_seqlen,
    int32_t sink,
    int32_t recent,
    int32_t skip_sink,
    int32_t skip_recent) {
  assert(NumThreads == blockDim.x);
  constexpr int32_t BLOCK_SEQ = (ELEMS_PER_BLOCK / NumChunk);
  const int32_t head_idx = blockIdx.y;
  if (*(mask_ptr + head_idx)) {
    const int32_t tid = threadIdx.x;
    const int32_t batch_id = head_idx / NumKVHead;
    const int32_t kv_head_id = head_idx % NumKVHead;

    const int32_t block_id = blockIdx.x;
    const int32_t block_start_seq = block_id * BLOCK_SEQ;
    const int32_t left_seq = min(BLOCK_SEQ, *seqlen_ptr - block_start_seq);
    const int32_t left_elem = left_seq * NumChunk;

    extern __shared__ int32_t smem[];
    int32_t* smem_popc = smem;
    T* smem_query_code = (T*)(smem_popc + NumThreads);

    // load q (of this gqa group) to smem
    T* q_ptr = (T*)query_ptr + batch_id * NumHead * NumChunk +
               kv_head_id * KVGroup * NumChunk;
#pragma unroll
    for (int i = tid; i < KVGroup * NumChunk; i += NumThreads) {
      smem_query_code[i] = q_ptr[i];
    }
    __syncthreads();

    // compute score
    int32_t score = 0;
    if (tid < left_elem) {
      const int32_t seq_id = block_start_seq + tid / NumChunk;
      const int32_t chunk_id = tid % NumChunk;
      T* k_ptr = (T*)keys_ptr + batch_id * key_code_batch_stride +
                 seq_id * key_code_seq_stride +
                 kv_head_id * key_code_head_stride + chunk_id;
      T* q_smem_ptr = smem_query_code + chunk_id;
      T key = *k_ptr;
#pragma unroll
      for (int i = 0; i < KVGroup; i++) {
        T q = *(q_smem_ptr + i * NumChunk);
        T tmp = key ^ q;
        score += USE_INT64 ? __popcll(tmp) : __popc(tmp);
      }
    }
    smem_popc[tid] = score;
    __syncthreads();

    // write results to global memory
    if (tid < left_seq) {
      // transpose
      bool is_sink_or_recent = (block_start_seq + tid) < sink ||
                               (block_start_seq + tid) >= (*seqlen_ptr - recent);
      bool is_skip_sink_or_recent = (block_start_seq + tid) < skip_sink ||
                                    (block_start_seq + tid) >= (*seqlen_ptr - skip_recent);
      half* o_ptr =
          (half*)output_ptr + head_idx * max_seqlen + (block_start_seq + tid);
      half sum = (half)(0.);

#pragma unroll
      for (int h = 0; h < NumChunk; h += 1) {
        sum += (half)(smem_popc[tid * NumChunk + h]);
      }
      if (is_sink_or_recent) {
        *o_ptr = min_value;
      } else if (is_skip_sink_or_recent) {
        *o_ptr = max_value;
      } else {
        *o_ptr = sum;
      }
    }
  } else {
    return;
  }
}

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
) {
  int32_t bsz = key_codes.size(0);
  int32_t max_seqlen = key_codes.size(1);
  int32_t num_kv_head = key_codes.size(2);
  int32_t num_chunk = key_codes.size(3);
  int32_t num_head = query_code.size(2);
  int32_t gqa_size = num_head / num_kv_head;
  int32_t total_num_heads = bsz * num_kv_head;

  auto device = key_codes.device();
  int32_t device_id = device.index();
  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id);

  HEAD_SWITCH(num_head, NumHead, {
    KVHEAD_SWITCH(num_kv_head, NumKVHead, {
      KVGROUP_SWITCH(gqa_size, KVGroup, {
        NUMCHUNK_SWITCH(num_chunk, NumChunk, {
          constexpr int32_t NumThreads = 256;
          size_t shm_size = 0;
          shm_size += NumThreads * sizeof(int32_t);          // for popc results
          shm_size += KVGroup * NumChunk * sizeof(int32_t);  // for query_code

          if (NumChunk % 2 == 0) {
            // convert int32 to int64
            constexpr int32_t HalfNumChunk = NumChunk / 2 < 1 ? 1 : NumChunk / 2;
            constexpr int32_t NumTokens = NumThreads / HalfNumChunk;
            constexpr int32_t ELEMS_PER_BLOCK = HalfNumChunk * NumTokens;
            dim3 blks(NumThreads);
            dim3 grids((max_seqlen * HalfNumChunk + ELEMS_PER_BLOCK - 1) /
                        ELEMS_PER_BLOCK, total_num_heads);
            StaticHammingScoreMaskKernel<int64_t, true, NumThreads,
                                       ELEMS_PER_BLOCK, NumHead, NumKVHead,
                                       KVGroup, HalfNumChunk>
                <<<grids, blks, shm_size, stream>>>(
                    key_codes.data_ptr(), query_code.data_ptr(),
                    (half*)(score.data_ptr<at::Half>()), mask.data_ptr<bool>(),
                    seqlen.data_ptr<int32_t>(), max_value, min_value,
                    key_codes.stride(0) / 2, key_codes.stride(1) / 2,
                    key_codes.stride(2) / 2, max_seqlen,
                    sink, recent, skip_sink, skip_recent);

          } else {
            constexpr int32_t NumTokens = NumThreads / NumChunk;
            constexpr int32_t ELEMS_PER_BLOCK = NumChunk * NumTokens;
            dim3 blks(NumThreads);
            dim3 grids((max_seqlen * NumChunk + ELEMS_PER_BLOCK - 1) /
                        ELEMS_PER_BLOCK, total_num_heads);
            StaticHammingScoreMaskKernel<int32_t, false, NumThreads,
                                       ELEMS_PER_BLOCK, NumHead, NumKVHead,
                                       KVGroup, NumChunk>
                <<<grids, blks, shm_size, stream>>>(
                    key_codes.data_ptr(), query_code.data_ptr(),
                    (half*)(score.data_ptr<at::Half>()), mask.data_ptr<bool>(),
                    seqlen.data_ptr<int32_t>(), max_value, min_value,
                    key_codes.stride(0), key_codes.stride(1),
                    key_codes.stride(2), max_seqlen,
                    sink, recent, skip_sink, skip_recent);
          }
        });
      });
    });
  });
}

}  // namespace kvlib