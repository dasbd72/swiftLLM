#pragma once

#include <torch/torch.h>

void cpu_store_kvcache_decode(torch::Tensor k, torch::Tensor v,
                              torch::Tensor k_cache, torch::Tensor v_cache,
                              torch::Tensor block_table, torch::Tensor seq_ids,
                              torch::Tensor seq_lens);
