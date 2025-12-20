#include <torch/extension.h>

#include "block_swapping.h"
#include "cpu_kvcache_mgmt.h"
#include "cpu_paged_attention.h"

PYBIND11_MODULE(swiftllm_c, m) {
  m.def("swap_blocks", &swap_blocks);
  m.def("cpu_paged_attention", &cpu_paged_attention);
  m.def("cpu_store_kvcache_decode", &cpu_store_kvcache_decode);
}
