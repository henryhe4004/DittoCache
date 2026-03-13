/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>

#include <raft/matrix/detail/select_radix.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

namespace raft::matrix::detail::select::radix {
namespace impl {

/**
 *
 * It is expected to call this kernel multiple times (passes), in each pass we
 * process a radix, going from the most significant towards the least
 * significant bits (MSD).
 *
 * Conceptually, each pass consists of 4 steps:
 *
 * 1. Calculate histogram
 *      First, transform bits into a digit, the value of which is in the range
 *      [0, 2^{BITS_PER_PASS}-1]. Then count the frequency of each digit value
 * and the result is a histogram. That is, histogram[i] contains the count of
 * inputs having value i.
 *
 * 2. Scan the histogram
 *      Inclusive prefix sum is computed for the histogram. After this step,
 * histogram[i] contains the count of inputs having value <= i.
 *
 * 3. Find the bucket j of the histogram that the max_k-th value falls into
 *
 * 4. Filtering
 *      Input elements whose digit value <j are the top-max_k elements. We put them
 * into the result array out. The number of such elements is histogram[j-1].
 * Since the max_k-th value must be in the bucket j, we write all elements in bucket
 * j into a intermediate buffer out_buf. For the next pass, these elements are
 * used as input, and we would like to find the (max_k - histogram[j-1])-th value
 * among them. That is, the max_k in the next pass is set to (max_k - histogram[j-1]).
 *
 * In the implementation, the filtering step is delayed to the next pass so the
 * filtering and histogram computation are fused. In this way, inputs are read
 * once rather than twice.
 *
 * During the filtering step, we won't write candidates (elements in bucket j)
 * to `out_buf` if the number of candidates is larger than the length of
 * `out_buf` (this could happen when the leading bits of input values are almost
 * the same). And then in the next pass, inputs are read from `in` rather than
 * from `in_buf`. The benefit is that we can save the cost of writing candidates
 * and their indices.
 */
template <typename T, typename IdxT, typename MaskT, int BitsPerPass,
          int BlockSize, bool fused_last_filter>
__global__ void radix_masked_kernel(
    const T* in, const IdxT* in_idx, const MaskT* batch_mask,
    const IdxT *real_len, const IdxT *real_k,
    const T* in_buf, const IdxT* in_idx_buf,
    T* out_buf, IdxT* out_idx_buf, T* out,
    IdxT* out_idx, Counter<T, IdxT>* counters, IdxT* histograms,
    const IdxT max_len, const IdxT max_k,
    const bool select_min, const int pass) {
  const size_t batch_id = blockIdx.y;

  if (*(batch_mask + batch_id)) {
    auto counter = counters + batch_id;
    IdxT current_k;
    IdxT previous_len;
    IdxT current_len;
    if (pass == 0) {
      current_k = *real_k;
      previous_len = *real_len;
      // Need to do this so setting counter->previous_len for the next pass is
      // correct. This value is meaningless for pass 0, but it's fine because
      // pass 0 won't be the last pass in this implementation so pass 0 won't
      // hit the "if (pass == num_passes - 1)" branch. Maybe it's better to
      // reload counter->previous_len and use it rather than current_len in
      // last_filter()
      current_len = *real_len;
    } else {
      current_k = counter->k;
      current_len = counter->len;
      previous_len = counter->previous_len;
    }
    if (current_len == 0) {
      return;
    }

    // When max_k=max_len, early_stop will be true at pass 0. It means
    // filter_and_histogram() should handle correctly the case that pass=0 and
    // early_stop=true. However, this special case of max_k=max_len is handled in other
    // way in select_k() so such case is not possible here.
    const bool early_stop = (current_len == current_k);
    const IdxT buf_len = calc_buf_len<T>(max_len);

    // "previous_len > buf_len" means previous pass skips writing buffer
    if (pass == 0 || pass == 1 || previous_len > buf_len) {
      in_buf = in + batch_id * max_len;
      in_idx_buf = in_idx ? (in_idx + batch_id * max_len) : nullptr;
      previous_len = *real_len;
    } else {
      in_buf += batch_id * buf_len;
      in_idx_buf += batch_id * buf_len;
    }
    // "current_len > buf_len" means current pass will skip writing buffer
    if (pass == 0 || current_len > buf_len) {
      out_buf = nullptr;
      out_idx_buf = nullptr;
    } else {
      out_buf += batch_id * buf_len;
      out_idx_buf += batch_id * buf_len;
    }
    out += batch_id * max_k;
    out_idx += batch_id * max_k;

    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    auto histogram = histograms + batch_id * num_buckets;

    filter_and_histogram<T, IdxT, BitsPerPass>(
        in_buf, in_idx_buf, out_buf, out_idx_buf, out, out_idx, previous_len,
        counter, histogram, select_min, pass, early_stop);
    __threadfence();

    bool isLastBlock = false;
    if (threadIdx.x == 0) {
      unsigned int finished =
          atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
      isLastBlock = (finished == (gridDim.x - 1));
    }

    if (__syncthreads_or(isLastBlock)) {
      if (early_stop) {
        if (threadIdx.x == 0) {
          // `last_filter_kernel()` requires setting previous_len
          counter->previous_len = 0;
          counter->len = 0;
        }
        return;
      }

      scan<IdxT, BitsPerPass, BlockSize>(histogram);
      __syncthreads();
      choose_bucket<T, IdxT, BitsPerPass>(counter, histogram, current_k, pass);
      __syncthreads();

      constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
      // reset for next pass
      if (pass != num_passes - 1) {
        for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
          histogram[i] = 0;
        }
      }
      if (threadIdx.x == 0) {
        // `last_filter_kernel()` requires setting previous_len even in the last
        // pass
        counter->previous_len = current_len;
        // not necessary for the last pass, but put it here anyway
        counter->filter_cnt = 0;
      }

      if constexpr (fused_last_filter) {
        if (pass == num_passes - 1) {
          last_filter<T, IdxT, BitsPerPass>(
              out_buf ? out_buf : in_buf,
              out_idx_buf ? out_idx_buf : in_idx_buf, out, out_idx,
              out_buf ? current_len : *real_len, *real_k, counter, select_min, pass);
        }
      }
    }
  } else {
    return;
  }
}

template <typename T, typename IdxT, typename MaskT, int BitsPerPass,
          int BlockSize>
void radix_masked_topk(const T* in, const IdxT* in_idx, const MaskT* batch_mask,
                       const IdxT* real_len, const IdxT* real_k,
                       int batch_size, IdxT max_len, IdxT max_k, T* out, IdxT* out_idx,
                       bool select_min, bool fused_last_filter,
                       unsigned grid_dim, int sm_cnt,
                       rmm::cuda_stream_view stream,
                       rmm::mr::device_memory_resource* mr) {
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();

  auto kernel =
      radix_masked_kernel<T, IdxT, MaskT, BitsPerPass, BlockSize, false>;
  const size_t max_chunk_size =
      calc_chunk_size<T, IdxT, BlockSize>(batch_size, max_len, sm_cnt, kernel);
  if (max_chunk_size != static_cast<size_t>(batch_size)) {
    grid_dim = calc_grid_dim<T, IdxT, BitsPerPass, BlockSize>(max_chunk_size,
                                                              max_len, sm_cnt);
  }
  const IdxT buf_len = calc_buf_len<T>(max_len);

  size_t req_aux =
      max_chunk_size * (sizeof(Counter<T, IdxT>) + num_buckets * sizeof(IdxT));
  size_t req_buf = max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT));
  size_t mem_req =
      req_aux + req_buf + 256 * 6;  // might need extra memory for alignment

  auto pool_guard = raft::get_pool_memory_resource(mr, mem_req);
  if (pool_guard) {
    RAFT_LOG_DEBUG(
        "radix::select_k: using pool memory resource with initial size %zu "
        "bytes",
        pool_guard->pool_size());
  }

  rmm::device_uvector<Counter<T, IdxT>> counters(max_chunk_size, stream, mr);
  rmm::device_uvector<IdxT> histograms(max_chunk_size * num_buckets, stream,
                                       mr);
  rmm::device_uvector<T> buf1(max_chunk_size * buf_len, stream, mr);
  rmm::device_uvector<IdxT> idx_buf1(max_chunk_size * buf_len, stream, mr);
  rmm::device_uvector<T> buf2(max_chunk_size * buf_len, stream, mr);
  rmm::device_uvector<IdxT> idx_buf2(max_chunk_size * buf_len, stream, mr);
  for (size_t offset = 0; offset < static_cast<size_t>(batch_size);
       offset += max_chunk_size) {
    int chunk_size = std::min(max_chunk_size, batch_size - offset);
    RAFT_CUDA_TRY(cudaMemsetAsync(counters.data(), 0,
                                  counters.size() * sizeof(Counter<T, IdxT>),
                                  stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(histograms.data(), 0,
                                  histograms.size() * sizeof(IdxT), stream));

    const T* chunk_in = in + offset * max_len;
    const IdxT* chunk_in_idx = in_idx ? (in_idx + offset * max_len) : nullptr;
    const MaskT* chunk_batch_mask = batch_mask + offset;
    T* chunk_out = out + offset * max_k;
    IdxT* chunk_out_idx = out_idx + offset * max_k;

    const T* in_buf = nullptr;
    const IdxT* in_idx_buf = nullptr;
    T* out_buf = nullptr;
    IdxT* out_idx_buf = nullptr;

    dim3 blocks(grid_dim, chunk_size);
    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();

    for (int pass = 0; pass < num_passes; ++pass) {
      set_buf_pointers(chunk_in, chunk_in_idx, buf1.data(), idx_buf1.data(),
                       buf2.data(), idx_buf2.data(), pass, in_buf, in_idx_buf,
                       out_buf, out_idx_buf);

      if (fused_last_filter && pass == num_passes - 1) {
        kernel =
            radix_masked_kernel<T, IdxT, MaskT, BitsPerPass, BlockSize, true>;
      }

      kernel<<<blocks, BlockSize, 0, stream>>>(
          chunk_in, chunk_in_idx, chunk_batch_mask,
          real_len, real_k, in_buf, in_idx_buf, out_buf,
          out_idx_buf, chunk_out, chunk_out_idx, counters.data(),
          histograms.data(), max_len, max_k, select_min, pass);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }
}

template <typename T, typename IdxT, typename MaskT, int BitsPerPass,
          int BlockSize>
__global__ void radix_masked_topk_one_block_kernel(
    const T* in, const IdxT* in_idx, const MaskT* batch_mask,
    const IdxT *real_len, const IdxT *real_k,
    const IdxT max_len, const IdxT max_k, T* out,
    IdxT* out_idx, const bool select_min, T* buf1,
    IdxT* idx_buf1, T* buf2, IdxT* idx_buf2) {
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ Counter<T, IdxT> counter;
  __shared__ IdxT histogram[num_buckets];

  if (threadIdx.x == 0) {
    counter.k = *real_k;
    counter.len = *real_len;
    counter.previous_len = *real_len;
    counter.kth_value_bits = 0;
    counter.out_cnt = 0;
    counter.out_back_cnt = 0;
  }
  __syncthreads();

  const size_t batch_id =
      blockIdx.x;  // size_t to avoid multiplication overflow

  if (*(batch_mask + batch_id)) {
    in += batch_id * max_len;
    if (in_idx) {
      in_idx += batch_id * max_len;
    }
    out += batch_id * max_k;
    out_idx += batch_id * max_k;
    buf1 += batch_id * max_len;
    idx_buf1 += batch_id * max_len;
    buf2 += batch_id * max_len;
    idx_buf2 += batch_id * max_len;
    const T* in_buf = nullptr;
    const IdxT* in_idx_buf = nullptr;
    T* out_buf = nullptr;
    IdxT* out_idx_buf = nullptr;

    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
    for (int pass = 0; pass < num_passes; ++pass) {
      set_buf_pointers(in, in_idx, buf1, idx_buf1, buf2, idx_buf2, pass, in_buf,
                       in_idx_buf, out_buf, out_idx_buf);

      IdxT current_len = counter.len;
      IdxT current_k = counter.k;

      filter_and_histogram_for_one_block<T, IdxT, BitsPerPass>(
          in_buf, in_idx_buf, out_buf, out_idx_buf, out, out_idx, &counter,
          histogram, select_min, pass);
      __syncthreads();

      scan<IdxT, BitsPerPass, BlockSize>(histogram);
      __syncthreads();

      choose_bucket<T, IdxT, BitsPerPass>(&counter, histogram, current_k, pass);
      if (threadIdx.x == 0) {
        counter.previous_len = current_len;
      }
      __syncthreads();

      if (counter.len == counter.k || pass == num_passes - 1) {
        last_filter<T, IdxT, BitsPerPass>(
            pass == 0 ? in : out_buf, pass == 0 ? in_idx : out_idx_buf, out,
            out_idx, current_len, *real_k, &counter, select_min, pass);
        break;
      }
    }
  } else {
    return;
  }
}

// radix_topk() might use multiple thread blocks for one row of a batch. In
// contrast, the following one-block version uses single thread block for one
// row of a batch, so intermediate data, like counters and global histograms,
// can be kept in shared memory and cheap sync operations can be used. It's used
// when max_len is relatively small or when the number of blocks per row calculated
// by `calc_grid_dim()` is 1.
template <typename T, typename IdxT, typename MaskT, int BitsPerPass,
          int BlockSize>
void radix_masked_topk_one_block(
  const T* in, const IdxT* in_idx,
  const MaskT* batch_mask,
  const IdxT* real_len, const IdxT* real_k,
  int batch_size, IdxT max_len, IdxT max_k,
  T* out, IdxT* out_idx,
  bool select_min, int sm_cnt,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr
) {
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);

  auto kernel = radix_masked_topk_one_block_kernel<T, IdxT, MaskT, BitsPerPass,
                                                   BlockSize>;
  const size_t max_chunk_size =
      calc_chunk_size<T, IdxT, BlockSize>(batch_size, max_len, sm_cnt, kernel);

  auto pool_guard = raft::get_pool_memory_resource(
      mr,
      max_chunk_size * max_len * 2 * (sizeof(T) + sizeof(IdxT)) +
          256 * 4  // might need extra memory for alignment
  );
  if (pool_guard) {
    RAFT_LOG_DEBUG(
        "radix::select_k: using pool memory resource with initial size %zu "
        "bytes",
        pool_guard->pool_size());
  }

  rmm::device_uvector<T> buf1(max_len * max_chunk_size, stream, mr);
  rmm::device_uvector<IdxT> idx_buf1(max_len * max_chunk_size, stream, mr);
  rmm::device_uvector<T> buf2(max_len * max_chunk_size, stream, mr);
  rmm::device_uvector<IdxT> idx_buf2(max_len * max_chunk_size, stream, mr);

  for (size_t offset = 0; offset < static_cast<size_t>(batch_size);
       offset += max_chunk_size) {
    int chunk_size = std::min(max_chunk_size, batch_size - offset);
    kernel<<<chunk_size, BlockSize, 0, stream>>>(
        in + offset * max_len, in_idx ? (in_idx + offset * max_len) : nullptr,
        batch_mask + offset, real_len, real_k,
        max_len, max_k, out + offset * max_k, out_idx + offset * max_k,
        select_min, buf1.data(), idx_buf1.data(), buf2.data(), idx_buf2.data());
  }
}

}  // namespace impl

/**
 * Select max_k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_keys` as a row-major matrix with max_len
 * columns and batch_size rows, then this function selects max_k smallest/largest
 * values in each row and fills in the row-major matrix `out` of size
 * (batch_size, max_k).
 *
 * Note, the output is NOT sorted within the groups of `max_k` selected elements.
 *
 * @tparam T
 *   the type of the keys (what is being compared).
 * @tparam IdxT
 *   the index type (what is being selected together with the keys).
 * @tparam BitsPerPass
 *   The size of the radix;
 *   it affects the number of passes and number of buckets.
 * @tparam BlockSize
 *   Number of threads in a kernel thread block.
 *
 * @param[in] in
 *   contiguous device array of inputs of size (max_len * batch_size);
 *   these are compared and selected.
 * @param[in] in_idx
 *   contiguous device array of inputs of size (max_len * batch_size);
 *   typically, these are indices of the corresponding in_keys.
 * @param batch_size
 *   number of input rows, i.e. the batch size.
 * @param max_len
 *   length of a single input array (row); also sometimes referred as n_cols.
 *   Invariant: max_len >= max_k.
 * @param max_k
 *   the number of outputs to select in each input row.
 * @param[out] out
 *   contiguous device array of outputs of size (max_k * batch_size);
 *   the max_k smallest/largest values from each row of the `in_keys`.
 * @param[out] out_idx
 *   contiguous device array of outputs of size (max_k * batch_size);
 *   the payload selected together with `out`.
 * @param select_min
 *   whether to select max_k smallest (true) or largest (false) keys.
 * @param fused_last_filter
 *   when it's true, the last filter is fused into the kernel in the last pass
 * and only one thread block will do the filtering; when false, a standalone
 * filter kernel with multiple thread blocks is called. The later case is
 * preferable when leading bits of input data are almost the same. That is, when
 * the value range of input data is narrow. In such case, there could be a large
 * number of inputs for the last filter, hence using multiple thread blocks is
 * beneficial.
 * @param stream
 * @param mr an optional memory resource to use across the calls (you can
 * provide a large enough memory pool here to avoid memory allocations within
 * the call).
 */
template <typename T, typename IdxT, typename MaskT, int BitsPerPass,
          int BlockSize>
void select_k_masked(
  const T* in,
  const IdxT* in_idx,
  const MaskT* batch_mask,
  const IdxT* real_len,
  const IdxT* real_k,
  int batch_size,
  IdxT max_len,
  IdxT max_k,
  T* out,
  IdxT* out_idx,
  bool select_min,
  bool fused_last_filter,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = nullptr
) {
  int sm_cnt;
  {
    int dev;
    RAFT_CUDA_TRY(cudaGetDevice(&dev));
    RAFT_CUDA_TRY(
        cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev));
  }

  constexpr int items_per_thread = 32;

  if (max_len <= BlockSize * items_per_thread) {
    impl::radix_masked_topk_one_block<T, IdxT, MaskT, BitsPerPass, BlockSize>(
        in, in_idx, batch_mask, real_len, real_k,
        batch_size, max_len, max_k, out, out_idx, select_min,
        sm_cnt, stream, mr);
  } else {
    unsigned grid_dim = impl::calc_grid_dim<T, IdxT, BitsPerPass, BlockSize>(
        batch_size, max_len, sm_cnt);
    if (grid_dim == 1) {
      impl::radix_masked_topk_one_block<T, IdxT, MaskT, BitsPerPass, BlockSize>(
          in, in_idx, batch_mask, real_len, real_k,
          batch_size, max_len, max_k, out, out_idx, select_min,
          sm_cnt, stream, mr);
    } else {
      impl::radix_masked_topk<T, IdxT, MaskT, BitsPerPass, BlockSize>(
          in, in_idx, batch_mask, real_len, real_k,
          batch_size, max_len, max_k, out, out_idx, select_min,
          fused_last_filter, grid_dim, sm_cnt, stream, mr);
    }
  }
}

}  // namespace raft::matrix::detail::select::radix
