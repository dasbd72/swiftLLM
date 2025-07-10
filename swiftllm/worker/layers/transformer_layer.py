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
        layer_id: int,
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.layer_id = layer_id

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
            input_embds: The input embeddings tensor of shape [num_total_tokens, hidden_size].
            residual_buf: The residual buffer tensor of shape [num_total_tokens, hidden_size].
            k_cache: The key cache tensor of shape [num_blocks, num_layers, num_kv_heads, block_size, head_dim].
            v_cache: The value cache tensor of shape [num_blocks, num_layers, num_kv_heads, block_size, head_dim].
            block_table: The block table tensor of shape [*, max_blocks_per_seq].
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
        )  # [num_total_tokens, hidden_size]
        k = linear(
            input_embds, self.weight.k_proj
        )  # [num_total_tokens, num_kv_heads*head_dim]
        v = linear(
            input_embds, self.weight.v_proj
        )  # [num_total_tokens, num_kv_heads*head_dim]
        q = q.view(
            -1, self.model_config.num_q_heads, self.model_config.head_dim
        )  # [num_total_tokens, num_q_heads, head_dim]
        k = k.view(
            -1, self.model_config.num_kv_heads, self.model_config.head_dim
        )  # [num_total_tokens, num_kv_heads, head_dim]
        v = v.view(
            -1, self.model_config.num_kv_heads, self.model_config.head_dim
        )  # [num_total_tokens, num_kv_heads, head_dim]

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
                self.layer_id,
            )

        # Attention
        o = input_embds  # [num_total_tokens, hidden_size]
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
        o = linear(o, self.weight.o_proj)  # [num_total_tokens, hidden_size]

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
            input_embds: The input embeddings tensor of shape [num_total_tokens, hidden_size].
            residual_buf: The residual buffer tensor of shape [num_total_tokens, hidden_size].
            k_cache: The key cache tensor of shape [num_blocks, num_layers, num_kv_heads, block_size, head_dim].
            v_cache: The value cache tensor of shape [num_blocks, num_layers, num_kv_heads, block_size, head_dim].
            block_table: The block table tensor of shape [*, max_blocks_per_seq].
            seq_block_size: The size of each sequence block.
            num_seq_blocks: The number of sequence blocks, which is equal to ceil(max_seq_len / seq_block_size).
            softmax_scale: The scale factor for softmax.
            seq_ids: The sequence IDs tensor of shape [num_seqs].
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
        )  # [num_total_tokens, hidden_size]
        k = linear(
            input_embds, self.weight.k_proj
        )  # [num_total_tokens, num_kv_heads*head_dim]
        v = linear(
            input_embds, self.weight.v_proj
        )  # [num_total_tokens, num_kv_heads*head_dim]
        q = q.view(
            -1, self.model_config.num_q_heads, self.model_config.head_dim
        )  # [num_total_tokens, num_q_heads, head_dim]
        k = k.view(
            -1, self.model_config.num_kv_heads, self.model_config.head_dim
        )  # [num_total_tokens, num_kv_heads, head_dim]
        v = v.view(
            -1, self.model_config.num_kv_heads, self.model_config.head_dim
        )  # [num_total_tokens, num_kv_heads, head_dim]

        # Rotary emb
        rotary_embedding_inplace(q, k, position_cos, position_sin)

        if not ignore_kvcache:
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
                self.layer_id,
            )

        # Attention
        o = input_embds  # [num_total_tokens, hidden_size]
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
            self.layer_id,
            o,
        )

        # Output GEMM
        o = linear(o, self.weight.o_proj)  # [num_total_tokens, hidden_size]

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
