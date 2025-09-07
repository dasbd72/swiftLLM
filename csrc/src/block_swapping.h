#ifdef USE_CUDA
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

void swap_blocks(const std::vector<int64_t> &source_block_ids,
                 const std::vector<int64_t> &target_block_ids,
                 const bool is_swap_in,

                 torch::Tensor k_cache, torch::Tensor v_cache,
                 torch::Tensor k_swap, torch::Tensor v_swap);
#endif  // USE_CUDA