import torch
from absl.testing import absltest

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.kernels.kvcache_mgmt import (
    store_kvcache_decode,
    store_kvcache_prefill,
)


class KvCacheMgmtTest(absltest.TestCase):
    """Tests for KV cache management kernels."""

    def setUp(self):
        """Set up common mock configurations for tests."""
        super().setUp()
        torch.manual_seed(42)

        model_config_dict = {
            "_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.32.0.dev0",
            "use_cache": True,
            "vocab_size": 32000,
        }
        self.model_config = LlamaModelConfig(model_config=model_config_dict)

        self.engine_config = EngineConfig(
            model_path="mock_model_path",
            use_dummy=False,
            block_size=16,
            gpu_mem_utilization=0.99,
            num_cpu_blocks=65536,
            max_seqs_in_block_table=64,
            max_blocks_per_seq=19,
            max_batch_size=32,
            max_tokens_in_batch=133 * 32,
            weight_device="cuda",
            profile_scheduling_strategy="gpu",
            max_micro_batch_size=16,
        )

    def test_store_kvcache_prefill_raises_not_implemented_on_cpu(self):
        """Tests that prefill raises NotImplementedError on CPU."""
        num_prefill_seqs = 4
        seq_lens_list = [10, 20, 30, 40]
        num_prefill_tokens = sum(seq_lens_list)
        num_blocks = 256

        k = torch.randn(
            (
                num_prefill_tokens,
                self.model_config.num_kv_heads,
                self.model_config.head_dim,
            ),
            dtype=torch.float16,
            device="cpu",
        )
        v = torch.randn_like(k)
        k_cache = torch.zeros(
            (
                num_blocks,
                self.model_config.num_kv_heads,
                self.engine_config.block_size,
                self.model_config.head_dim,
            ),
            dtype=torch.float16,
            device="cpu",
        )
        v_cache = torch.zeros_like(k_cache)
        block_table = torch.randint(
            0,
            num_blocks,
            (
                self.engine_config.max_seqs_in_block_table,
                self.engine_config.max_blocks_per_seq,
            ),
            dtype=torch.int32,
            device="cpu",
        )
        seq_ids = torch.arange(
            0, num_prefill_seqs, dtype=torch.int32, device="cpu"
        )
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cpu")
        seq_start_locs = torch.cumsum(
            torch.tensor([0] + seq_lens_list[:-1]), dim=0
        ).to(dtype=torch.int32, device="cpu")

        with self.assertRaises(NotImplementedError):
            store_kvcache_prefill(
                k,
                v,
                k_cache,
                v_cache,
                block_table,
                seq_ids,
                seq_start_locs,
                seq_lens,
                self.model_config,
                self.engine_config,
            )

    def test_store_kvcache_decode(self):
        """Tests decode kernel by comparing GPU and CPU outputs."""
        num_decoding_seqs = 16
        num_blocks = 128

        # Create base tensors on GPU
        k_gpu = torch.randn(
            (
                num_decoding_seqs,
                self.model_config.num_kv_heads,
                self.model_config.head_dim,
            ),
            dtype=torch.float16,
            device="cuda",
        )
        v_gpu = torch.randn_like(k_gpu)
        k_cache_gpu = torch.zeros(
            (
                num_blocks,
                self.model_config.num_kv_heads,
                self.engine_config.block_size,
                self.model_config.head_dim,
            ),
            dtype=torch.float16,
            device="cuda",
        )
        v_cache_gpu = torch.zeros_like(k_cache_gpu)
        block_table_gpu = torch.randint(
            0,
            num_blocks,
            (
                self.engine_config.max_seqs_in_block_table,
                self.engine_config.max_blocks_per_seq,
            ),
            dtype=torch.int32,
            device="cuda",
        )
        seq_ids_gpu = torch.arange(
            0, num_decoding_seqs, dtype=torch.int32, device="cuda"
        )
        seq_lens_gpu = torch.randint(
            1,
            self.engine_config.max_blocks_per_seq
            * self.engine_config.block_size,
            (num_decoding_seqs,),
            dtype=torch.int32,
            device="cuda",
        )

        # Create CPU tensor copies
        k_cpu = k_gpu.cpu()
        v_cpu = v_gpu.cpu()
        k_cache_cpu = torch.zeros_like(k_cache_gpu, device="cpu")
        v_cache_cpu = torch.zeros_like(v_cache_gpu, device="cpu")
        block_table_cpu = block_table_gpu.cpu()
        seq_ids_cpu = seq_ids_gpu.cpu()
        seq_lens_cpu = seq_lens_gpu.cpu()

        # Execute on GPU
        store_kvcache_decode(
            k_gpu,
            v_gpu,
            k_cache_gpu,
            v_cache_gpu,
            block_table_gpu,
            seq_ids_gpu,
            seq_lens_gpu,
            self.model_config,
            self.engine_config,
        )

        # Execute on CPU
        store_kvcache_decode(
            k_cpu,
            v_cpu,
            k_cache_cpu,
            v_cache_cpu,
            block_table_cpu,
            seq_ids_cpu,
            seq_lens_cpu,
            self.model_config,
            self.engine_config,
        )

        # Compare results
        self.assertTrue(
            torch.allclose(
                k_cache_cpu, k_cache_gpu.cpu(), atol=1e-4, rtol=1e-4
            ),
            "k_cache mismatch between CPU and GPU implementations.",
        )
        self.assertTrue(
            torch.allclose(
                v_cache_cpu, v_cache_gpu.cpu(), atol=1e-4, rtol=1e-4
            ),
            "v_cache mismatch between CPU and GPU implementations.",
        )


if __name__ == "__main__":
    absltest.main()
