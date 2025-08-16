#include "defines.h"

inline void qk_product(const uniform int num_q_heads,
                       const uniform int num_kv_heads,
                       const uniform int block_size, const uniform int head_dim,
                       const uniform int num_heads_per_kv,
                       const uniform int elem_per_block,
                       const uniform int seq_len, const uniform float16 q[],
                       const uniform float16 k_cache[],
                       uniform float attn_score[],
                       const uniform int block_table[]) {
  uniform int imax = seq_len / block_size + 1;
  for (uniform int i = 0; i < imax; i++) {
    const uniform float16* k =
        k_cache + (1ll * block_table[i] * elem_per_block);
    for (uniform int j = 0; j < num_kv_heads; j++) {
      uniform int q_off = j * num_heads_per_kv * head_dim;
      uniform int k_off = j * block_size * head_dim;
      uniform int a_off = i * block_size * num_q_heads + j * num_heads_per_kv;
      uniform int tmax = min(block_size, seq_len - i * block_size);
      uniform int t;
      for (t = 0; t < tmax - K_TILE_WIDTH + 1; t += K_TILE_WIDTH) {
        float sum[MAX_NUM_HEADS_PER_KV][K_TILE_WIDTH];
        for (uniform int h = 0; h < num_heads_per_kv; h++) {
          for (uniform int g = 0; g < K_TILE_WIDTH; g++) {
            sum[h][g] = 0;
          }
        }
        foreach (l = 0 ... head_dim) {
          for (uniform int g = 0; g < K_TILE_WIDTH; g++) {
            float k_val = k[k_off + g * head_dim + l];
            for (uniform int h = 0; h < num_heads_per_kv; h++) {
              sum[h][g] += q[q_off + h * head_dim + l] * k_val;
            }
          }
        }
        for (uniform int h = 0; h < num_heads_per_kv; h++) {
          for (uniform int g = 0; g < K_TILE_WIDTH; g++) {
            attn_score[a_off + (t + g) * num_q_heads + h] =
                reduce_add(sum[h][g]);
          }
        }
        k_off += head_dim * K_TILE_WIDTH;
      }
      for (; t < tmax; t++) {
        float sum[MAX_NUM_HEADS_PER_KV];
        for (uniform int h = 0; h < num_heads_per_kv; h++) {
          sum[h] = 0;
        }
        foreach (l = 0 ... head_dim) {
          float k_val = k[k_off + l];
          for (uniform int h = 0; h < num_heads_per_kv; h++) {
            sum[h] += q[q_off + h * head_dim + l] * k_val;
          }
        }
        for (uniform int h = 0; h < num_heads_per_kv; h++) {
          attn_score[a_off + t * num_q_heads + h] = reduce_add(sum[h]);
        }
        k_off += head_dim;
      }
    }
  }
}

template <int num_q_heads, int num_kv_heads, int block_size, int head_dim,
          int num_heads_per_kv, int elem_per_block>
void qk_product_template(const uniform int seq_len, const uniform float16 q[],
                         const uniform float16 k_cache[],
                         uniform float attn_score[],
                         const uniform int block_table[]) {
  qk_product(num_q_heads, num_kv_heads, block_size, head_dim, num_heads_per_kv,
             elem_per_block, seq_len, q, k_cache, attn_score, block_table);
}

inline void av_product(const uniform int num_q_heads,
                       const uniform int num_kv_heads,
                       const uniform int block_size, const uniform int head_dim,
                       const uniform int num_heads_per_kv,
                       const uniform int elem_per_block,
                       const uniform int seq_len,
                       const uniform float attn_score[],
                       const uniform float16 v_cache[], uniform float16 o[],
                       const uniform int block_table[]) {
  uniform int imax = seq_len / block_size + 1;
  memset(o, 0, num_q_heads * head_dim * sizeof(uniform float16));
  for (uniform int i = 0; i < imax; i++) {
    const uniform float16* uniform v =
        v_cache + (1ll * block_table[i] * elem_per_block);
    for (uniform int j = 0; j < num_kv_heads; j++) {
      uniform int o_off = j * num_heads_per_kv * head_dim;
      uniform int v_off = j * block_size * head_dim;
      uniform int tmax = min(block_size, seq_len - i * block_size);
      for (uniform int t = 0; t < tmax; t++) {
        uniform int a_off =
            (i * block_size + t) * num_q_heads + j * num_heads_per_kv;
        foreach (l = 0 ... head_dim) {
          float16 v_val = v[v_off + l];
          for (uniform int h = 0; h < num_heads_per_kv; h++) {
            o[o_off + h * head_dim + l] += v_val * attn_score[a_off + h];
          }
        }
        v_off += head_dim;
      }
    }
  }
}

template <int num_q_heads, int num_kv_heads, int block_size, int head_dim,
          int num_heads_per_kv, int elem_per_block>
void av_product_template(const uniform int seq_len,
                         const uniform float attn_score[],
                         const uniform float16 v_cache[], uniform float16 o[],
                         const uniform int block_table[]) {
  av_product(num_q_heads, num_kv_heads, block_size, head_dim, num_heads_per_kv,
             elem_per_block, seq_len, attn_score, v_cache, o, block_table);
}

inline void softmax(const uniform int num_q_heads, const uniform int seq_len,
                    const uniform float softmax_scale,
                    uniform float attn_score[], uniform float attn_sum[]) {
  uniform float attn_max[MAX_NUM_Q_HEADS];

  foreach (h = 0 ... num_q_heads) {
    attn_max[h] = -1e20;
    for (uniform int i = 0; i < seq_len; i++) {
      attn_score[i * num_q_heads + h] *= softmax_scale;
      attn_max[h] = max(attn_max[h], attn_score[i * num_q_heads + h]);
    }
  }

  foreach (h = 0 ... num_q_heads) {
    attn_sum[h] = 0;
    for (uniform int i = 0; i < seq_len; i++) {
      attn_score[i * num_q_heads + h] =
          exp(attn_score[i * num_q_heads + h] - attn_max[h]);
      attn_sum[h] += attn_score[i * num_q_heads + h];
    }
  }

  foreach (h = 0 ... num_q_heads) {
    for (uniform int i = 0; i < seq_len; i++) {
      attn_score[i * num_q_heads + h] /= attn_sum[h];
    }
    attn_sum[h] = log(attn_sum[h]) + attn_max[h];
  }
}

template <uniform int num_q_heads>
void softmax_template(const uniform int seq_len,
                      const uniform float softmax_scale,
                      uniform float attn_score[], uniform float attn_sum[]) {
  softmax(num_q_heads, seq_len, softmax_scale, attn_score, attn_sum);
}

export void attn_one_seq(
    const uniform int num_q_heads, const uniform int num_kv_heads,
    const uniform int block_size, const uniform int head_dim,
    const uniform int num_heads_per_kv, const uniform int elem_per_block,
    const uniform int seq_len, const uniform float softmax_scale,
    const uniform float16 q[], const uniform float16 k_cache[],
    const uniform float16 v_cache[], const uniform int block_table[],
    uniform float attn_score[], uniform float16 o[], uniform float attn_sum[]) {
  qk_product(num_q_heads, num_kv_heads, block_size, head_dim, num_heads_per_kv,
             elem_per_block, seq_len, q, k_cache, attn_score, block_table);
  softmax(num_q_heads, seq_len, softmax_scale, attn_score, attn_sum);
  av_product(num_q_heads, num_kv_heads, block_size, head_dim, num_heads_per_kv,
             elem_per_block, seq_len, attn_score, v_cache, o, block_table);
}

export void attn_one_seq_llama_2_7b_4(
    const uniform int seq_len, const uniform float softmax_scale,
    const uniform float16 q[], const uniform float16 k_cache[],
    const uniform float16 v_cache[], const uniform int block_table[],
    uniform float attn_score[], uniform float16 o[], uniform float attn_sum[]) {
  qk_product_template<LLAMA_2_7B_NUM_Q_HEADS, LLAMA_2_7B_NUM_KV_HEADS,
                      LLAMA_2_7B_4_BLOCK_SIZE, LLAMA_2_7B_HEAD_DIM,
                      LLAMA_2_7B_NUM_HEADS_PER_KV, LLAMA_2_7B_4_ELEM_PER_BLOCK>(
      seq_len, q, k_cache, attn_score, block_table);
  softmax_template<LLAMA_2_7B_NUM_Q_HEADS>(seq_len, softmax_scale, attn_score,
                                           attn_sum);
  av_product_template<LLAMA_2_7B_NUM_Q_HEADS, LLAMA_2_7B_NUM_KV_HEADS,
                      LLAMA_2_7B_4_BLOCK_SIZE, LLAMA_2_7B_HEAD_DIM,
                      LLAMA_2_7B_NUM_HEADS_PER_KV, LLAMA_2_7B_4_ELEM_PER_BLOCK>(
      seq_len, attn_score, v_cache, o, block_table);
}

export void attn_one_seq_llama_2_7b_8(
    const uniform int seq_len, const uniform float softmax_scale,
    const uniform float16 q[], const uniform float16 k_cache[],
    const uniform float16 v_cache[], const uniform int block_table[],
    uniform float attn_score[], uniform float16 o[], uniform float attn_sum[]) {
  qk_product_template<LLAMA_2_7B_NUM_Q_HEADS, LLAMA_2_7B_NUM_KV_HEADS,
                      LLAMA_2_7B_8_BLOCK_SIZE, LLAMA_2_7B_HEAD_DIM,
                      LLAMA_2_7B_NUM_HEADS_PER_KV, LLAMA_2_7B_8_ELEM_PER_BLOCK>(
      seq_len, q, k_cache, attn_score, block_table);
  softmax_template<LLAMA_2_7B_NUM_Q_HEADS>(seq_len, softmax_scale, attn_score,
                                           attn_sum);
  av_product_template<LLAMA_2_7B_NUM_Q_HEADS, LLAMA_2_7B_NUM_KV_HEADS,
                      LLAMA_2_7B_8_BLOCK_SIZE, LLAMA_2_7B_HEAD_DIM,
                      LLAMA_2_7B_NUM_HEADS_PER_KV, LLAMA_2_7B_8_ELEM_PER_BLOCK>(
      seq_len, attn_score, v_cache, o, block_table);
}

export void attn_one_seq_llama_2_7b_16(
    const uniform int seq_len, const uniform float softmax_scale,
    const uniform float16 q[], const uniform float16 k_cache[],
    const uniform float16 v_cache[], const uniform int block_table[],
    uniform float attn_score[], uniform float16 o[], uniform float attn_sum[]) {
  qk_product_template<LLAMA_2_7B_NUM_Q_HEADS, LLAMA_2_7B_NUM_KV_HEADS,
                      LLAMA_2_7B_16_BLOCK_SIZE, LLAMA_2_7B_HEAD_DIM,
                      LLAMA_2_7B_NUM_HEADS_PER_KV,
                      LLAMA_2_7B_16_ELEM_PER_BLOCK>(seq_len, q, k_cache,
                                                    attn_score, block_table);
  softmax_template<LLAMA_2_7B_NUM_Q_HEADS>(seq_len, softmax_scale, attn_score,
                                           attn_sum);
  av_product_template<LLAMA_2_7B_NUM_Q_HEADS, LLAMA_2_7B_NUM_KV_HEADS,
                      LLAMA_2_7B_16_BLOCK_SIZE, LLAMA_2_7B_HEAD_DIM,
                      LLAMA_2_7B_NUM_HEADS_PER_KV,
                      LLAMA_2_7B_16_ELEM_PER_BLOCK>(seq_len, attn_score,
                                                    v_cache, o, block_table);
}

export void attn_one_seq_llama_2_7b_32(
    const uniform int seq_len, const uniform float softmax_scale,
    const uniform float16 q[], const uniform float16 k_cache[],
    const uniform float16 v_cache[], const uniform int block_table[],
    uniform float attn_score[], uniform float16 o[], uniform float attn_sum[]) {
  qk_product_template<LLAMA_2_7B_NUM_Q_HEADS, LLAMA_2_7B_NUM_KV_HEADS,
                      LLAMA_2_7B_32_BLOCK_SIZE, LLAMA_2_7B_HEAD_DIM,
                      LLAMA_2_7B_NUM_HEADS_PER_KV,
                      LLAMA_2_7B_32_ELEM_PER_BLOCK>(seq_len, q, k_cache,
                                                    attn_score, block_table);
  softmax_template<LLAMA_2_7B_NUM_Q_HEADS>(seq_len, softmax_scale, attn_score,
                                           attn_sum);
  av_product_template<LLAMA_2_7B_NUM_Q_HEADS, LLAMA_2_7B_NUM_KV_HEADS,
                      LLAMA_2_7B_32_BLOCK_SIZE, LLAMA_2_7B_HEAD_DIM,
                      LLAMA_2_7B_NUM_HEADS_PER_KV,
                      LLAMA_2_7B_32_ELEM_PER_BLOCK>(seq_len, attn_score,
                                                    v_cache, o, block_table);
}
