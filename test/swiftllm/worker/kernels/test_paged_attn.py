"""Tests for the paged attention kernel."""

from typing import Any, Dict
from unittest import mock

import torch
from absl.testing import absltest, parameterized

from swiftllm.worker.kernels.paged_attn import paged_attention


class PagedAttnTest(parameterized.TestCase):
    """Test case for the paged attention kernel."""

    def _get_mock_parameters(self) -> Dict[str, Any]:
        """Generate mock parameters for testing."""
        torch.manual_seed(42)  # For reproducibility
        num_hidden_layers = 32
        num_q_heads = 32
        num_kv_heads = 32
        head_dim = 128
        hidden_size = num_q_heads * head_dim  # 4096
        block_size = 16
        max_blocks_per_seq = 64
        batch_size = 8

        mock_model_config = mock.Mock()
        mock_model_config.num_hidden_layers = num_hidden_layers
        mock_model_config.num_q_heads = num_q_heads
        mock_model_config.num_kv_heads = num_kv_heads
        mock_model_config.head_dim = head_dim
        mock_model_config.hidden_size = hidden_size
        mock_engine_config = mock.Mock()
        mock_engine_config.block_size = block_size
        mock_engine_config.max_blocks_per_seq = max_blocks_per_seq

        q = torch.randn(
            (batch_size, num_q_heads, head_dim),
            dtype=torch.float16,
            device="cuda",
        )
        num_blocks = batch_size * max_blocks_per_seq
        k_cache = torch.randn(
            (num_blocks, num_kv_heads, block_size, head_dim),
            dtype=torch.float16,
            device="cuda",
        )
        v_cache = torch.randn_like(k_cache, dtype=torch.float16, device="cuda")
        block_table = torch.randint(
            0,
            num_blocks,
            (batch_size, max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        seq_ids = torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
        seq_lens = torch.randint(
            1,
            max_blocks_per_seq * block_size,
            (batch_size,),
            dtype=torch.int32,
            device="cuda",
        )
        softmax_scale = 0.125
        o = torch.zeros(
            (batch_size, hidden_size),
            dtype=torch.float16,
            device="cuda",
        )
        seq_block_size = 32
        num_seq_blocks = (
            seq_lens.max().item() + seq_block_size - 1
        ) // seq_block_size

        return {
            "model_config": mock_model_config,
            "engine_config": mock_engine_config,
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "block_table": block_table,
            "seq_block_size": seq_block_size,
            "num_seq_blocks": num_seq_blocks,
            "softmax_scale": softmax_scale,
            "seq_ids": seq_ids,
            "seq_lens": seq_lens,
            "o": o,
        }

    def test_paged_attention(self):
        """Test the paged attention kernel."""
        parameters: Dict[str, Any] = self._get_mock_parameters()
        model_config = parameters["model_config"]
        engine_config = parameters["engine_config"]
        q_gpu = parameters["q"]
        k_cache_gpu = parameters["k_cache"]
        v_cache_gpu = parameters["v_cache"]
        block_table_gpu = parameters["block_table"]
        seq_ids_gpu = parameters["seq_ids"]
        seq_lens_gpu = parameters["seq_lens"]
        o_gpu = parameters["o"]

        # Prepare CPU tensors for comparison
        q_cpu = q_gpu.cpu().pin_memory()
        k_cache_cpu = k_cache_gpu.cpu().pin_memory()
        v_cache_cpu = v_cache_gpu.cpu().pin_memory()
        block_table_cpu = block_table_gpu.cpu()
        seq_ids_cpu = seq_ids_gpu.cpu()
        seq_lens_cpu = seq_lens_gpu.cpu()
        o_cpu = o_gpu.cpu().pin_memory()

        # Common parameters
        seq_block_size = parameters["seq_block_size"]
        num_seq_blocks = parameters["num_seq_blocks"]
        softmax_scale = parameters["softmax_scale"]

        # Run on GPU
        paged_attention(
            q_gpu,
            k_cache_gpu,
            v_cache_gpu,
            block_table_gpu,
            seq_block_size,
            num_seq_blocks,
            softmax_scale,
            seq_ids_gpu,
            seq_lens_gpu,
            model_config,
            engine_config,
            o_gpu,
        )

        # Run on CPU
        paged_attention(
            q_cpu,
            k_cache_cpu,
            v_cache_cpu,
            block_table_cpu,
            seq_block_size,
            num_seq_blocks,
            softmax_scale,
            seq_ids_cpu,
            seq_lens_cpu,
            model_config,
            engine_config,
            o_cpu,
        )

        # Compare outputs
        self.assertTrue(
            torch.allclose(o_gpu.cpu(), o_cpu, atol=5e-2, rtol=5e-2),
            "Paged attention output mismatch between GPU and CPU",
        )


if __name__ == "__main__":
    absltest.main()
