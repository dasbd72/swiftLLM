#ifdef USE_CUDA
#include "cpu_kvcache_mgmt.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include "thread_pool.h"

struct ckm_data_t {
  int num_seqs;
  int block_size;
  torch::Tensor k;
  torch::Tensor v;
  torch::Tensor k_cache;
  torch::Tensor v_cache;
  torch::Tensor block_table;
  torch::Tensor seq_ids;
  torch::Tensor seq_lens;
};

/**
 * Helper function to check shape and location of ckm_data_t
 *
 * k, v, k_cache, v_cache should be on cpu
 * block_table, seq_ids, seq_lens should be on cpu
 **/
void _check_ckm_data(ckm_data_t& data) {
  // Get variables
  torch::Tensor k = data.k;
  torch::Tensor v = data.v;
  torch::Tensor k_cache = data.k_cache;
  torch::Tensor v_cache = data.v_cache;
  torch::Tensor block_table = data.block_table;
  torch::Tensor seq_ids = data.seq_ids;
  torch::Tensor seq_lens = data.seq_lens;

  // Check shapes
  int num_seqs = k.size(0);
  if (v.size(0) != num_seqs) {
    throw std::runtime_error("v.size(0) != num_seqs");
  }
  if (seq_ids.size(0) != num_seqs) {
    throw std::runtime_error("seq_ids.size(0) != num_seqs");
  }
  if (seq_lens.size(0) != num_seqs) {
    throw std::runtime_error("seq_lens.size(0) != num_seqs");
  }

  // Check device type
  if (k.device().type() != torch::kCPU) {
    throw std::runtime_error("k should be on cpu");
  }
  if (v.device().type() != torch::kCPU) {
    throw std::runtime_error("v should be on cpu");
  }
  if (k_cache.device().type() != torch::kCPU) {
    throw std::runtime_error("k_cache should be on cpu");
  }
  if (v_cache.device().type() != torch::kCPU) {
    throw std::runtime_error("v_cache should be on cpu");
  }
  if (block_table.device().type() != torch::kCPU) {
    throw std::runtime_error("block_table should be on cpu");
  }
  if (seq_ids.device().type() != torch::kCPU) {
    throw std::runtime_error("seq_ids should be on cpu");
  }
  if (seq_lens.device().type() != torch::kCPU) {
    throw std::runtime_error("seq_lens should be on cpu");
  }
}

/**
 * Helper function for running a single sequence of kvcache management
 **/
void _cpu_store_kvcache_decode_seq(int seq_i, ckm_data_t& data) {
  // Make sure each thread runs in inference mode
  c10::InferenceMode guard;
  // Get variables
  int block_size = data.block_size;
  torch::Tensor k = data.k;
  torch::Tensor v = data.v;
  torch::Tensor k_cache = data.k_cache;
  torch::Tensor v_cache = data.v_cache;
  torch::Tensor block_table = data.block_table;
  torch::Tensor seq_ids = data.seq_ids;
  torch::Tensor seq_lens = data.seq_lens;

  // 1. Get the data for the current sequence.
  auto my_k = k.select(0, seq_i);  // [num_kv_heads, head_dim]
  auto my_v = v.select(0, seq_i);  // [num_kv_heads, head_dim]
  int64_t my_seq_id = seq_ids[seq_i].item<int64_t>();
  int64_t my_seq_len = seq_lens[seq_i].item<int64_t>();
  auto my_block_table = block_table.select(0, my_seq_id);

  // 2. Calculate the block index and offset.
  int64_t my_block_id = (my_seq_len - 1) / block_size;
  int64_t my_block_offset = (my_seq_len - 1) % block_size;
  int64_t my_block_index = my_block_table[my_block_id].item<int64_t>();

  // 3. Copy the key and value to the cache.
  k_cache.index_put_({my_block_index, torch::indexing::Slice(), my_block_offset,
                      torch::indexing::Slice()},
                     my_k);
  v_cache.index_put_({my_block_index, torch::indexing::Slice(), my_block_offset,
                      torch::indexing::Slice()},
                     my_v);
}

/**
 * Helper function for running all sequences of kvcache management
 **/
void _cpu_store_kvcache_decode_seqs(ckm_data_t& data) {
  // Get number of sequences
  int num_seqs = data.num_seqs;
  // Run each sequence
  for (int seq_i = 0; seq_i < num_seqs; ++seq_i) {
    _cpu_store_kvcache_decode_seq(seq_i, data);
  }
}

/**
 * Helper function for parallel execution of kvcache management across
 *sequences.
 **/
void _cpu_store_kvcache_decode(ckm_data_t& data) {
  // Parallelizing copy is not so useful
  // int num_seqs = data.num_seqs;
  // std::vector<std::future<void>> tasks(num_seqs);
  // for (int seq_i = 0; seq_i < num_seqs; ++seq_i) {
  //     tasks[seq_i] = worker_pool.enqueue(_cpu_store_kvcache_decode_seq,
  //     seq_i, data);
  // }
  // for (int seq_i = 0; seq_i < num_seqs; ++seq_i) {
  //     tasks[seq_i].get();
  // }
  int num_seqs = data.num_seqs;
  std::future<void> task;
  task = worker_pool.enqueue(_cpu_store_kvcache_decode_seqs, data);
  task.get();
}

/**
 * Stores the key/value cache to paged attention for decode sequences on cpu
 * @param k Key tensor [num_decode_tokens, num_kv_heads, head_dim]
 * @param v Value tensor [num_decode_tokens, num_kv_heads, head_dim]
 * @param k_cache Key cache tensor [num_blocks, num_kv_heads, block_size,
 *head_dim]
 * @param v_cache Value cache tensor [num_blocks, num_kv_heads, block_size,
 *head_dim]
 * @param block_table Block table tensor [num_decoding_seqs,
 *max_num_blocks_per_seq]
 * @param seq_ids Sequence IDs [num_decoding_seqs]
 * @param seq_lens Sequence lengths tensor [num_decoding_seqs]
 *
 * This function is executed by the single-threaded serializer_pool, ensuring
 * that only one kvcache management operation runs at a time. It then uses the
 * worker_pool to parallelize the computation across sequences.
 **/
void cpu_store_kvcache_decode(torch::Tensor k, torch::Tensor v,
                              torch::Tensor k_cache, torch::Tensor v_cache,
                              torch::Tensor block_table, torch::Tensor seq_ids,
                              torch::Tensor seq_lens) {
  // Build ckm_data_t
  int num_seqs = k.size(0);
  int block_size = k_cache.size(2);
  ckm_data_t* data = new ckm_data_t{
      num_seqs, block_size,  k,       v,        k_cache,
      v_cache,  block_table, seq_ids, seq_lens,
  };
  _check_ckm_data(*data);

  // Current stream
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::cuda::CUDAStream default_stream = c10::cuda::getDefaultCUDAStream();
  if (stream == default_stream) {
    // If we are on the default stream, we can run the function directly
    _cpu_store_kvcache_decode(*data);
    delete data;  // Clean up the memory after use
  } else {
    cudaError_t err = cudaLaunchHostFunc(
        stream.stream(),
        [](void* arg) {
          ckm_data_t* data_ptr = static_cast<ckm_data_t*>(arg);
          _cpu_store_kvcache_decode(*data_ptr);
          delete data_ptr;  // Clean up the memory after use
        },
        data);
    if (err != cudaSuccess) {
      cudaGetLastError();  // Clear the error state
      throw std::runtime_error("Failed to launch CUDA host function: " +
                               std::string(cudaGetErrorString(err)));
    }
  }
}
#endif  // USE_CUDA
