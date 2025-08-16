#pragma once

#include <torch/extension.h>

void cpu_paged_attention(torch::Tensor q, torch::Tensor k_cache,
                         torch::Tensor v_cache, torch::Tensor block_table,
                         double softmax_scale, torch::Tensor seq_ids,
                         torch::Tensor seq_lens, torch::Tensor o);
