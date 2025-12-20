#include "cpu_paged_attention.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include "c10/util/Half.h"
#include "cpu_paged_attention_ispc.h"
#include "defines.h"
#include "thread_pool.h"

struct cpa_data_t {
  int num_seqs;
  int num_q_heads;
  int num_kv_heads;
  int block_size;
  int head_dim;
  int num_heads_per_kv;
  int elem_per_block;
  int block_table_width;
  torch::Tensor q;
  torch::Tensor k_cache;
  torch::Tensor v_cache;
  torch::Tensor block_table;
  double softmax_scale;
  torch::Tensor seq_ids;
  torch::Tensor seq_lens;
  torch::Tensor o;
};

/**
 * Helper function to check shape and location of cpa_data_t
 *
 * q, k_cache, v_cache, o should be on cpu
 * block_table, seq_ids, seq_lens should be on cpu
 **/
void _check_cpa_data(cpa_data_t& data) {
  // Get variables
  int num_seqs = data.num_seqs;
  int num_q_heads = data.num_q_heads;
  int num_kv_heads = data.num_kv_heads;
  int block_size = data.block_size;
  int head_dim = data.head_dim;
  torch::Tensor q = data.q;
  torch::Tensor k_cache = data.k_cache;
  torch::Tensor v_cache = data.v_cache;
  torch::Tensor block_table = data.block_table;
  torch::Tensor seq_ids = data.seq_ids;
  torch::Tensor seq_lens = data.seq_lens;
  torch::Tensor o = data.o;

  // Check shapes
  if (seq_ids.size(0) != num_seqs) {
    throw std::invalid_argument(
        "expected seq_ids to have " + std::to_string(num_seqs) +
        " sequences, but got " + std::to_string(seq_ids.size(0)));
  }
  if (seq_lens.size(0) != num_seqs) {
    throw std::invalid_argument(
        "expected seq_lens to have " + std::to_string(num_seqs) +
        " sequences, but got " + std::to_string(seq_lens.size(0)));
  }
  if (o.size(0) != num_seqs) {
    throw std::invalid_argument(
        "expected o to have " + std::to_string(num_seqs) +
        " sequences, but got " + std::to_string(o.size(0)));
  }

  // Check device type
  if (q.device().type() != torch::kCPU) {
    throw std::invalid_argument("expected q to be on CPU, but got " +
                                q.device().str());
  }
  if (k_cache.device().type() != torch::kCPU) {
    throw std::invalid_argument("expected k_cache to be on CPU, but got " +
                                k_cache.device().str());
  }
  if (v_cache.device().type() != torch::kCPU) {
    throw std::invalid_argument("expected v_cache to be on CPU, but got " +
                                v_cache.device().str());
  }
  if (o.device().type() != torch::kCPU) {
    throw std::invalid_argument("expected o to be on CPU, but got " +
                                o.device().str());
  }
  if (block_table.device().type() != torch::kCPU) {
    throw std::invalid_argument("expected block_table to be on CPU, but got " +
                                block_table.device().str());
  }
  if (seq_ids.device().type() != torch::kCPU) {
    throw std::invalid_argument("expected seq_ids to be on CPU, but got " +
                                seq_ids.device().str());
  }
  if (seq_lens.device().type() != torch::kCPU) {
    throw std::invalid_argument("expected seq_lens to be on CPU, but got " +
                                seq_lens.device().str());
  }

  // Check data types
  if (q.scalar_type() != torch::kHalf) {
    throw std::invalid_argument("expected q to be of type Half, but got " +
                                std::to_string(int(q.scalar_type())));
  }
  if (k_cache.scalar_type() != torch::kHalf) {
    throw std::invalid_argument(
        "expected k_cache to be of type Half, but got " +
        std::to_string(int(k_cache.scalar_type())));
  }
  if (v_cache.scalar_type() != torch::kHalf) {
    throw std::invalid_argument(
        "expected v_cache to be of type Half, but got " +
        std::to_string(int(v_cache.scalar_type())));
  }
  if (o.scalar_type() != torch::kHalf) {
    throw std::invalid_argument("expected o to be of type Half, but got " +
                                std::to_string(int(o.scalar_type())));
  }
  if (block_table.scalar_type() != torch::kInt) {
    throw std::invalid_argument(
        "expected block_table to be of type Int, but got " +
        std::to_string(int(block_table.scalar_type())));
  }
  if (seq_ids.scalar_type() != torch::kInt) {
    throw std::invalid_argument("expected seq_ids to be of type Int, but got " +
                                std::to_string(int(seq_ids.scalar_type())));
  }
  if (seq_lens.scalar_type() != torch::kInt) {
    throw std::invalid_argument(
        "expected seq_lens to be of type Int, but got " +
        std::to_string(int(seq_lens.scalar_type())));
  }
}

#if defined(TORCH_CPU_PAGED_ATTENTION)

/**
 * Helper function for running a single sequence of paged attention
 **/
void _cpu_paged_attention_seq(int seq_i, cpa_data_t& data) {
  // Make sure each thread runs in inference mode
  c10::InferenceMode guard;
  // Get variables
  int block_size = data.block_size;
  int num_heads_per_kv = data.num_heads_per_kv;
  torch::Tensor q = data.q;
  torch::Tensor k_cache = data.k_cache;
  torch::Tensor v_cache = data.v_cache;
  torch::Tensor block_table = data.block_table;
  double softmax_scale = data.softmax_scale;
  torch::Tensor seq_ids = data.seq_ids;
  torch::Tensor seq_lens = data.seq_lens;
  torch::Tensor o = data.o;

  // 1. Get the data for the current sequence.
  auto my_q = q.select(0, seq_i);  // [num_q_heads, head_dim]
  int64_t my_seq_id = seq_ids[seq_i].item<int64_t>();
  int64_t my_seq_len = seq_lens[seq_i].item<int64_t>();
  auto my_block_table = block_table.select(0, my_seq_id);

  // 2. Reconstruct the key and value caches for this sequence without loops.
  // Create indices for every token in the sequence's history.
  auto token_indices = torch::arange(my_seq_len, torch::kCPU);

  // Find which block and offset each token belongs to.
  auto block_indices = (token_indices / block_size).to(torch::kLong);
  auto block_offsets =
      (token_indices % block_size).to(torch::kLong).to(torch::kCPU);
  // Get the physical block numbers from the block table.
  auto physical_blocks = my_block_table.index({block_indices}).to(torch::kCPU);

  // Gather the key and value vectors from the cache in one operation.
  auto my_k =
      k_cache.index({physical_blocks, torch::indexing::Slice(),
                     block_offsets});  // [seq_len, num_kv_heads, head_dim]
  auto my_v =
      v_cache.index({physical_blocks, torch::indexing::Slice(),
                     block_offsets});  // [seq_len, num_kv_heads, head_dim]

  // 3. Vectorized Attention Calculation (for all heads at once).
  // Expand K and V to match the number of Q heads for Grouped-Query Attention.
  if (num_heads_per_kv > 1) {
    my_k = my_k.repeat_interleave(num_heads_per_kv, 1).to(torch::kFloat32);
    my_v = my_v.repeat_interleave(num_heads_per_kv, 1).to(torch::kFloat32);
  }

  // Transpose K and V for matrix multiplication.
  my_k = my_k.transpose(0, 1);  // [num_q_heads, seq_len, head_dim]
  my_v = my_v.transpose(0, 1);  // [num_q_heads, seq_len, head_dim]

  // Calculate scores, apply softmax, and get the final output vector.
  auto attn_scores = torch::matmul(my_q.unsqueeze(1), my_k.transpose(1, 2));
  attn_scores = attn_scores * softmax_scale;
  attn_scores = attn_scores.squeeze(1);
  auto attn_probs = torch::softmax(attn_scores, 1);
  auto o_seq = torch::matmul(attn_probs.unsqueeze(1), my_v);
  o_seq = o_seq.squeeze(1);
  o.select(0, seq_i) = o_seq.view({-1}).to(o.dtype());
}

/**
 * Helper function for parallel execution of paged attention across sequences.
 **/
void _cpu_paged_attention(cpa_data_t& data) {
  int num_seqs = data.num_seqs;
  std::vector<std::future<void>> tasks(num_seqs);
  for (int seq_i = 0; seq_i < num_seqs; ++seq_i) {
    tasks[seq_i] = worker_pool.enqueue(_cpu_paged_attention_seq, seq_i, data);
  }
  for (int seq_i = 0; seq_i < num_seqs; ++seq_i) {
    tasks[seq_i].get();
  }
}

#elif defined(ISPC_CPU_PAGED_ATTENTION)

/**
 * Helper function for running a single sequence of paged attention
 **/
void _cpu_paged_attention_seq(int seq_i, cpa_data_t& data) {
  // Get variables
  int num_seqs = data.num_seqs;          // Number of sequences
  int num_q_heads = data.num_q_heads;    // Number of query heads
  int num_kv_heads = data.num_kv_heads;  // Number of key-value heads
  int block_size = data.block_size;      // Size of each block
  int head_dim = data.head_dim;          // Dimension of each head
  int num_heads_per_kv =
      data.num_heads_per_kv;  // Number of heads per key-value pair
  int elem_per_block = data.elem_per_block;  // Number of elements per block
  int block_table_width = data.block_table_width;  // Width of the block table
  torch::Tensor q = data.q;  // (num_seqs * num_q_heads * head_dim)
  torch::Tensor k_cache =
      data.k_cache;  // (num_blocks * num_kv_heads * block_size * head_dim)
  torch::Tensor v_cache =
      data.v_cache;  // (num_blocks * num_kv_heads * block_size * head_dim)
  torch::Tensor block_table =
      data.block_table;  // (num_blocks * block_table_width)
  double softmax_scale = data.softmax_scale;  // Scaling factor for softmax
  torch::Tensor seq_ids = data.seq_ids;       // (num_seqs)
  torch::Tensor seq_lens = data.seq_lens;     // (num_seqs)
  torch::Tensor o = data.o;                   // (num_seqs * hidden_size)

  // Extract pointers from tensors
  __fp16* q_ptr = (__fp16*)q.data_ptr<torch::Half>();
  __fp16* k_cache_ptr = (__fp16*)k_cache.data_ptr<torch::Half>();
  __fp16* v_cache_ptr = (__fp16*)v_cache.data_ptr<torch::Half>();
  int32_t* block_table_ptr = (int32_t*)block_table.data_ptr<int32_t>();
  int32_t* seq_ids_ptr = (int32_t*)seq_ids.data_ptr<int32_t>();
  int32_t* seq_lens_ptr = (int32_t*)seq_lens.data_ptr<int32_t>();
  __fp16* o_ptr = (__fp16*)o.data_ptr<torch::Half>();

  // Get pointer of the current sequence
  int32_t my_seq_id = seq_ids_ptr[seq_i];
  int32_t my_seq_len = seq_lens_ptr[seq_i];
  __fp16* my_q_ptr =
      q_ptr + seq_i * num_q_heads * head_dim;  // (num_q_heads * head_dim)
  int32_t* my_block_table_ptr =
      block_table_ptr + my_seq_id * block_table_width;  // (block_table_width)
  __fp16* my_o_ptr = o_ptr + seq_i * num_q_heads * head_dim;  // (hidden_size)
  float* my_a_score =
      new float[num_q_heads * my_seq_len];   // (num_q_heads * my_seq_len)
  float* my_a_sum = new float[num_q_heads];  // (num_q_heads)

  bool is_llama_2_7b = num_q_heads == LLAMA_2_7B_NUM_Q_HEADS &&
                       num_kv_heads == LLAMA_2_7B_NUM_KV_HEADS &&
                       head_dim == LLAMA_2_7B_HEAD_DIM &&
                       num_heads_per_kv == LLAMA_2_7B_NUM_HEADS_PER_KV;
  if (is_llama_2_7b && block_size == LLAMA_2_7B_4_BLOCK_SIZE) {
    ispc::attn_one_seq_llama_2_7b_4(
        my_seq_len, softmax_scale, my_q_ptr, k_cache_ptr, v_cache_ptr,
        my_block_table_ptr, my_a_score, my_o_ptr, my_a_sum);
  } else if (is_llama_2_7b && block_size == LLAMA_2_7B_8_BLOCK_SIZE) {
    ispc::attn_one_seq_llama_2_7b_8(
        my_seq_len, softmax_scale, my_q_ptr, k_cache_ptr, v_cache_ptr,
        my_block_table_ptr, my_a_score, my_o_ptr, my_a_sum);
  } else if (is_llama_2_7b && block_size == LLAMA_2_7B_16_BLOCK_SIZE) {
    ispc::attn_one_seq_llama_2_7b_16(
        my_seq_len, softmax_scale, my_q_ptr, k_cache_ptr, v_cache_ptr,
        my_block_table_ptr, my_a_score, my_o_ptr, my_a_sum);
  } else if (is_llama_2_7b && block_size == LLAMA_2_7B_32_BLOCK_SIZE) {
    ispc::attn_one_seq_llama_2_7b_32(
        my_seq_len, softmax_scale, my_q_ptr, k_cache_ptr, v_cache_ptr,
        my_block_table_ptr, my_a_score, my_o_ptr, my_a_sum);
  } else {
    // Call ISPC function for custom configuration
    ispc::attn_one_seq(num_q_heads, num_kv_heads, block_size, head_dim,
                       num_heads_per_kv, elem_per_block, my_seq_len,
                       softmax_scale, my_q_ptr, k_cache_ptr, v_cache_ptr,
                       my_block_table_ptr, my_a_score, my_o_ptr, my_a_sum);
  }

  // Clean up
  delete[] my_a_score;
  delete[] my_a_sum;
}

/**
 * Helper function for parallel execution of paged attention across sequences.
 **/
void _cpu_paged_attention(cpa_data_t& data) {
  int num_seqs = data.num_seqs;
  std::vector<std::future<void>> tasks(num_seqs);
  for (int seq_i = 0; seq_i < num_seqs; ++seq_i) {
    tasks[seq_i] = worker_pool.enqueue(_cpu_paged_attention_seq, seq_i, data);
  }
  for (int seq_i = 0; seq_i < num_seqs; ++seq_i) {
    tasks[seq_i].get();
  }
}

#endif  // *_CPU_PAGED_ATTENTION

/**
 * Computes paged attention on cpu
 * @param q Query tensor [num_decoding_seqs, num_q_heads, head_dim]
 * @param k_cache Key cache tensor [num_blocks, num_kv_heads, block_size,
 *head_dim]
 * @param v_cache Value cache tensor [num_blocks, num_kv_heads, block_size,
 *head_dim]
 * @param block_table Block table tensor [num_decoding_seqs,
 *max_num_blocks_per_seq]
 * @param softmax_scale Scaling factor for softmax
 * @param seq_ids Sequence IDs [num_decoding_seqs]
 * @param seq_lens Sequence lengths tensor [num_decoding_seqs]
 * @param o Output tensor [num_decoding_seqs, hidden_size]
 *
 * This function is executed by the single-threaded serializer_pool, ensuring
 * that only one paged attention operation runs at a time. It then uses the
 * worker_pool to parallelize the computation across sequences.
 **/
void cpu_paged_attention(torch::Tensor q, torch::Tensor k_cache,
                         torch::Tensor v_cache, torch::Tensor block_table,
                         double softmax_scale, torch::Tensor seq_ids,
                         torch::Tensor seq_lens, torch::Tensor o) {
  // Build cpa_data_t
  int num_seqs = q.size(0);
  int num_q_heads = q.size(1);
  int num_kv_heads = k_cache.size(1);
  int block_size = k_cache.size(2);
  int head_dim = k_cache.size(3);
  int num_heads_per_kv = num_q_heads / num_kv_heads;
  int elem_per_block = num_kv_heads * block_size * head_dim;
  int block_table_width = block_table.size(1);
  cpa_data_t* data = new cpa_data_t{
      num_seqs,
      num_q_heads,
      num_kv_heads,
      block_size,
      head_dim,
      num_heads_per_kv,
      elem_per_block,
      block_table_width,
      q,
      k_cache,
      v_cache,
      block_table,
      softmax_scale,
      seq_ids,
      seq_lens,
      o,
  };
  _check_cpa_data(*data);

  // Current stream
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::cuda::CUDAStream default_stream = c10::cuda::getDefaultCUDAStream();
  if (stream == default_stream) {
    // If we are on the default stream, run the CPU paged attention directly
    _cpu_paged_attention(*data);
    delete data;  // Clean up the memory after use
  } else {
    cudaError_t err = cudaLaunchHostFunc(
        stream.stream(),
        [](void* arg) {
          cpa_data_t* data_ptr = static_cast<cpa_data_t*>(arg);
          _cpu_paged_attention(*data_ptr);
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
