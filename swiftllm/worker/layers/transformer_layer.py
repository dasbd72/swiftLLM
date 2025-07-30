import torch
import vllm_flash_attn

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.kernels.kvcache_mgmt import (
    store_kvcache_decode,
    store_kvcache_prefill,
)
from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.kernels.paged_attn import paged_attention
from swiftllm.worker.kernels.prefill_attn import prefill_attention
from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace
from swiftllm.worker.weight import LlamaTransformerLayerWeight


class LlamaTransformerLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        weight_device: str,
        layer_id: int,
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.weight_device = weight_device
        self.weight_names = [
            "attn_norm",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "ffn_norm",
            "up_gate_proj",
            "down_proj",
        ]
        self.weight_num_chunks = None
        self.weight_chunk_size = None
        self.weight_num_params = 0
        for name in self.weight_names:
            weight_attr: torch.Tensor = getattr(weight, name)
            self.weight_num_params += weight_attr.numel()
        self.weight_cpu = {name: None for name in self.weight_names}
        if weight_device == "cpu":
            for name in self.weight_names:
                self.weight_cpu[name] = getattr(weight, name).pin_memory()
                setattr(weight, name, None)
        self.layer_id = layer_id

    def weight_to_gpu(self):
        """
        Load weights to GPU if they are on CPU
        """
        if self.weight_device == "cpu":
            for name in self.weight_names:
                setattr(
                    self.weight,
                    name,
                    self.weight_cpu[name].to("cuda", non_blocking=True),
                )

    def weight_to_gpu_chunked_init(self, num_chunks: int):
        """
        Initialize chunk loading settings
        """
        if self.weight_device == "cpu":
            self.weight_num_chunks = num_chunks
            self.weight_chunk_size = (
                self.weight_num_params + num_chunks - 1
            ) // num_chunks
            for name in self.weight_names:
                cpu_weight_attr: torch.Tensor = self.weight_cpu[name]
                data = torch.empty_like(
                    cpu_weight_attr, dtype=cpu_weight_attr.dtype, device="cuda"
                )
                setattr(self.weight, name, data)

    def weight_to_gpu_chunked(self, chunk_id: int):
        """
        Load weights to GPU in chunks if they are on CPU
        """
        if self.weight_device == "cpu":
            weight_num_params_scanned = 0
            weight_num_params_start = chunk_id * self.weight_chunk_size
            weight_num_params_end = min(
                (chunk_id + 1) * self.weight_chunk_size,
                self.weight_num_params,
            )
            for name in self.weight_names:
                cpu_weight_attr: torch.Tensor = self.weight_cpu[name]
                gpu_weight_attr: torch.Tensor = getattr(self.weight, name)
                weight_num_params_this_attr = cpu_weight_attr.numel()
                if (
                    weight_num_params_scanned + weight_num_params_this_attr
                    > weight_num_params_start
                ):
                    # This attribute has some params in this chunk
                    start = max(
                        0, weight_num_params_start - weight_num_params_scanned
                    )
                    end = min(
                        weight_num_params_this_attr,
                        weight_num_params_end - weight_num_params_scanned,
                    )
                    if start < end:
                        gpu_weight_attr.view(-1)[start:end].copy_(
                            cpu_weight_attr.view(-1)[start:end],
                            non_blocking=True,
                        )
                weight_num_params_scanned += weight_num_params_this_attr

    def weight_gpu_free(self):
        """
        Free weights if we have a copy on CPU
        """
        if self.weight_device == "cpu":
            for name in self.weight_names:
                setattr(self.weight, name, None)

    def prefill(
        self,
        input_embds: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf: torch.Tensor,  # [num_tokens, hidden_size]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: float,
        seq_ids: torch.Tensor,
        seq_start_locs: torch.Tensor,
        seq_lens: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        ignore_kvcache: bool,
    ) -> torch.Tensor:
        """
        Prefill phase of the transformer layer.

        Args:
            input_embds: The input embeddings tensor of shape [num_tokens, hidden_size].
            residual_buf: The residual buffer tensor of shape [num_tokens, hidden_size].
            k_cache: The key cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim].
            v_cache: The value cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim].
            block_table: The block table tensor of shape [num_seqs, max_blocks_per_seq].
            softmax_scale: The scale factor for softmax.
            seq_ids: The sequence IDs tensor of shape [num_seqs].
            seq_start_locs: The sequence start locations tensor of shape [num_seqs].
            seq_lens: The sequence lengths tensor of shape [num_seqs].
            position_cos: The cosine values for rotary embedding of shape [num_tokens, head_dim//2].
            position_sin: The sine values for rotary embedding of shape [num_tokens, head_dim//2].
            ignore_kvcache: Whether to ignore the key-value cache.
        Returns:
            ffn_out: The output of the feed-forward network after attention, of shape [num_tokens, hidden_size].
        """

        # (fused) Add last layer's residual, and perform RMSNorm
        # Before: input_embds is the output of the last FFN block, and residual_buf
        #         is the residual to be added to input_embds
        # After: input_embds will be RMSNorm(input_embds + residual_buf), and
        #        residual_buf will be input_embds + residual_buf (which will be
        #        used as the residual after the attention block)
        fused_add_rmsnorm_inplace(
            input_embds,
            residual_buf,
            self.weight.attn_norm,
            self.model_config.rms_norm_eps,
        )

        # Calculate QKV
        q = linear(
            input_embds, self.weight.q_proj
        )  # [num_tokens, hidden_size]
        k = linear(
            input_embds, self.weight.k_proj
        )  # [num_tokens, num_kv_heads*head_dim]
        v = linear(
            input_embds, self.weight.v_proj
        )  # [num_tokens, num_kv_heads*head_dim]
        q = q.view(
            -1, self.model_config.num_q_heads, self.model_config.head_dim
        )  # [num_tokens, num_q_heads, head_dim]
        k = k.view(
            -1, self.model_config.num_kv_heads, self.model_config.head_dim
        )  # [num_tokens, num_kv_heads, head_dim]
        v = v.view(
            -1, self.model_config.num_kv_heads, self.model_config.head_dim
        )  # [num_tokens, num_kv_heads, head_dim]

        # Rotary emb
        rotary_embedding_inplace(q, k, position_cos, position_sin)

        if not ignore_kvcache:
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

        # Attention
        o = input_embds  # [num_tokens, hidden_size]
        if torch.cuda.get_device_capability() >= (8, 0):
            # Here the performance of vLLM's flash attention is better than us,
            # so use vllm_flash_attn
            seq_start_locs_with_end = torch.cat(
                (
                    seq_start_locs,
                    torch.tensor(
                        [input_embds.shape[0]],
                        dtype=torch.int32,
                        device=seq_start_locs.device,
                    ),
                )
            )
            max_seq_len = seq_lens.max().item()
            o[:, :] = vllm_flash_attn.flash_attn_varlen_func(
                q,
                k,
                v,
                seq_start_locs_with_end,
                seq_start_locs_with_end,
                max_seq_len,
                max_seq_len,
                softmax_scale=softmax_scale,
                causal=True,
            ).reshape(-1, self.model_config.hidden_size)
        else:
            # Switch to prefill_attention since V100 cannot use flash attention
            prefill_attention(
                q,
                k,
                v,
                o,
                softmax_scale,
                seq_ids,
                seq_start_locs,
                seq_lens,
                self.model_config,
            )

        # Output GEMM
        o = linear(o, self.weight.o_proj)  # [num_tokens, hidden_size]

        # residual & FFN norm
        fused_add_rmsnorm_inplace(
            o,
            residual_buf,
            self.weight.ffn_norm,
            self.model_config.rms_norm_eps,
        )
        q = None
        k = None
        v = None

        # FFN
        up_gate_proj = linear(o, self.weight.up_gate_proj)
        silu_and_mul_inplace(up_gate_proj)
        ffn_out = linear(
            up_gate_proj[:, : self.model_config.ffn_inter_dim],
            self.weight.down_proj,
        )

        return ffn_out

    def decode_pre_attn(
        self,
        input_embds: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf: torch.Tensor,  # [num_tokens, hidden_size]
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode phase of the transformer layer, before attention.

        Args:
            input_embds: The input embeddings tensor of shape [num_tokens, hidden_size].
            residual_buf: The residual buffer tensor of shape [num_tokens, hidden_size].
            position_cos: The cosine values for rotary embedding of shape [num_tokens, head_dim//2].
            position_sin: The sine values for rotary embedding of shape [num_tokens, head_dim//2].
        Returns:
            q: The query tensor of shape [num_tokens, num_q_heads, head_dim].
            k: The key tensor of shape [num_tokens, num_kv_heads, head_dim].
            v: The value tensor of shape [num_tokens, num_kv_heads, head_dim].
        """

        # (fused) Add last layer's residual, and perform RMSNorm
        # Before: input_embds is the output of the last FFN block, and residual_buf
        #         is the residual to be added to input_embds
        # After: input_embds will be RMSNorm(input_embds + residual_buf), and
        #        residual_buf will be input_embds + residual_buf (which will be
        #        used as the residual after the attention block)
        fused_add_rmsnorm_inplace(
            input_embds,
            residual_buf,
            self.weight.attn_norm,
            self.model_config.rms_norm_eps,
        )

        # Calculate QKV
        q = linear(
            input_embds, self.weight.q_proj
        )  # [num_tokens, hidden_size]
        k = linear(
            input_embds, self.weight.k_proj
        )  # [num_tokens, num_kv_heads*head_dim]
        v = linear(
            input_embds, self.weight.v_proj
        )  # [num_tokens, num_kv_heads*head_dim]
        q = q.view(
            -1, self.model_config.num_q_heads, self.model_config.head_dim
        )  # [num_tokens, num_q_heads, head_dim]
        k = k.view(
            -1, self.model_config.num_kv_heads, self.model_config.head_dim
        )  # [num_tokens, num_kv_heads, head_dim]
        v = v.view(
            -1, self.model_config.num_kv_heads, self.model_config.head_dim
        )  # [num_tokens, num_kv_heads, head_dim]

        # Rotary emb
        rotary_embedding_inplace(q, k, position_cos, position_sin)

        return q, k, v

    def decode_store_kvcache(
        self,
        k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_ids: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """
        Decode phase of the transformer layer, to store key-value cache.

        Args:
            k: The key tensor of shape [num_tokens, num_kv_heads, head_dim].
            v: The value tensor of shape [num_tokens, num_kv_heads, head_dim].
            k_cache: The key cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim].
            v_cache: The value cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim].
            block_table: The block table tensor of shape [num_seqs, max_blocks_per_seq].
            seq_ids: The sequence IDs tensor of shape [num_seqs].
            seq_lens: The sequence lengths tensor of shape [num_seqs].
        """
        store_kvcache_decode(
            k,
            v,
            k_cache,
            v_cache,
            block_table,
            seq_ids,
            seq_lens,
            self.model_config,
            self.engine_config,
        )

    def decode_attn(
        self,
        q: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_block_size: int,
        num_seq_blocks: int,
        softmax_scale: float,
        seq_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        o: torch.Tensor,  # [num_tokens, hidden_size]
    ) -> torch.Tensor:
        """
        Decode phase of the attention mechanism.

        Args:
            q: The query tensor of shape [num_tokens, num_q_heads, head_dim].
            k_cache: The key cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim].
            v_cache: The value cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim].
            block_table: The block table tensor of shape [num_seqs, max_blocks_per_seq].
            seq_block_size: The block size of the sequence.
            num_seq_blocks: The number of blocks in the sequence.
            softmax_scale: The scale factor for softmax.
            seq_ids: The sequence IDs tensor of shape [num_seqs].
            seq_lens: The sequence lengths tensor of shape [num_seqs].
            o: The output of the attention mechanism, of shape [num_tokens, hidden_size].
        """
        paged_attention(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_block_size,
            num_seq_blocks,
            softmax_scale,
            seq_ids,
            seq_lens,
            self.model_config,
            self.engine_config,
            o,
        )

    def decode_post_attn(
        self,
        o: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf: torch.Tensor,  # [num_tokens, hidden_size]
    ) -> torch.Tensor:
        """
        Decode phase of the transformer layer, after attention.

        Args:
            o: The output tensor from the attention mechanism of shape [num_tokens, hidden_size].
            residual_buf: The residual buffer tensor of shape [num_tokens, hidden_size].
        Returns:
            ffn_out: The output of the feed-forward network after attention, of shape [num_tokens, hidden_size].
        """

        # Output GEMM
        o = linear(o, self.weight.o_proj)  # [num_tokens, hidden_size]

        # residual & FFN norm
        fused_add_rmsnorm_inplace(
            o,
            residual_buf,
            self.weight.ffn_norm,
            self.model_config.rms_norm_eps,
        )

        # FFN
        up_gate_proj = linear(o, self.weight.up_gate_proj)
        silu_and_mul_inplace(up_gate_proj)
        ffn_out = linear(
            up_gate_proj[:, : self.model_config.ffn_inter_dim],
            self.weight.down_proj,
        )
        return ffn_out

    def decode(
        self,
        input_embds: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf: torch.Tensor,  # [num_tokens, hidden_size]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_block_size: int,
        num_seq_blocks: int,
        softmax_scale: float,
        seq_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        ignore_kvcache: bool,
    ) -> torch.Tensor:
        """
        Decode phase of the transformer layer.

        Args:
            input_embds: The input embeddings tensor of shape [num_tokens, hidden_size].
            residual_buf: The residual buffer tensor of shape [num_tokens, hidden_size].
            k_cache: The key cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim].
            v_cache: The value cache tensor of shape [num_blocks, num_kv_heads, block_size, head_dim].
            block_table: The block table tensor of shape [num_seqs, max_blocks_per_seq].
            seq_block_size: The block size of the sequence.
            num_seq_blocks: The number of blocks in the sequence.
            softmax_scale: The scale factor for softmax.
            seq_ids: The sequence IDs tensor of shape [num_seqs].
            seq_lens: The sequence lengths tensor of shape [num_seqs].
            position_cos: The cosine values for rotary embedding of shape [num_tokens, head_dim//2].
            position_sin: The sine values for rotary embedding of shape [num_tokens, head_dim//2].
            ignore_kvcache: Whether to ignore the key-value cache.
        Returns:
            ffn_out: The output of the feed-forward network after attention, of shape [num_tokens, hidden_size].
        """
        # Decode pre-attention
        q, k, v = self.decode_pre_attn(
            input_embds,
            residual_buf,
            position_cos,
            position_sin,
        )

        # Store key-value cache
        if not ignore_kvcache:
            self.decode_store_kvcache(
                k,
                v,
                k_cache,
                v_cache,
                block_table,
                seq_ids,
                seq_lens,
            )

        # Decode attention
        o = torch.empty(
            q.shape[0],
            self.model_config.hidden_size,
            dtype=q.dtype,
            device=q.device,
        )
        self.decode_attn(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_block_size,
            num_seq_blocks,
            softmax_scale,
            seq_ids,
            seq_lens,
            o,
        )

        # Decode post-attention
        ffn_out = self.decode_post_attn(o, residual_buf)

        return ffn_out
