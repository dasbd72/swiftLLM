import torch
import triton
import triton.language as tl

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.utils import cdiv


@triton.jit
def _fwd_kvcache_mgmt_prefill_kernel(
    k_cache: torch.Tensor,  # [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,  # [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,  # [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,  # [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    seq_ids: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_start_locs: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_lens: torch.Tensor,  # [num_prefill_seqs], contiguous
    cur_layer: int,
    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_prefill_seqs, cdiv(max_prefill_len, block_size)]
    my_batch_id = tl.program_id(0)
    my_block_id = tl.program_id(1)
    my_seq_len = tl.load(prefill_seq_lens + my_batch_id)
    my_seq_start_loc = tl.load(prefill_seq_start_locs + my_batch_id)
    if my_block_id * block_size >= my_seq_len:
        return

    my_token_range = (
        tl.arange(0, block_size).to(tl.int64)
        + my_block_id * block_size
        + my_seq_start_loc
    )
    my_seq_id = tl.load(seq_ids + my_batch_id)
    my_block_index = tl.load(
        block_table + my_seq_id * max_blocks_per_seq + my_block_id
    ).to(tl.int64)

    offs_kv = (
        (my_token_range * num_kv_heads * head_dim).to(tl.int64)[:, None, None]
        + (tl.arange(0, num_kv_heads) * head_dim)[None, :, None]
        + tl.arange(0, head_dim)[None, None, :]
    )
    offs_kvcache = (
        (my_block_index * num_layers + cur_layer)
        * num_kv_heads
        * block_size
        * head_dim
        + (tl.arange(0, num_kv_heads) * block_size * head_dim)[None, :, None]
        + (tl.arange(0, block_size) * head_dim)[:, None, None]
        + tl.arange(0, head_dim)[None, None, :]
    )

    mask = (my_token_range < my_seq_len + my_seq_start_loc)[:, None, None]
    tl.store(
        k_cache + offs_kvcache, tl.load(k + offs_kv, mask=mask), mask=mask
    )
    tl.store(
        v_cache + offs_kvcache, tl.load(v + offs_kv, mask=mask), mask=mask
    )


@triton.jit
def _fwd_kvcache_mgmt_decoding_kernel(
    k_cache: torch.Tensor,  # [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,  # [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,  # [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,  # [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    decoding_seq_ids: torch.Tensor,  # [num_decoding_seqs], contiguous
    decoding_seq_lens: torch.Tensor,  # [num_decoding_seqs], contiguous
    cur_layer: int,
    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_decoding_seqs]
    my_batch_id = tl.program_id(0).to(tl.int64)
    my_seq_id = tl.load(decoding_seq_ids + my_batch_id)
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    my_block_id = (my_seq_len - 1) // block_size
    my_block_offset = (my_seq_len - 1) % block_size
    my_block_index = tl.load(
        block_table + my_seq_id * max_blocks_per_seq + my_block_id
    ).to(tl.int64)

    offs_kv = (
        my_batch_id * num_kv_heads * head_dim
        + (tl.arange(0, num_kv_heads) * head_dim)[:, None]
        + tl.arange(0, head_dim)[None, :]
    )
    offs_kvcache = (
        (my_block_index * num_layers + cur_layer)
        * num_kv_heads
        * block_size
        * head_dim
        + (tl.arange(0, num_kv_heads) * block_size * head_dim)[:, None]
        + my_block_offset * head_dim
        + tl.arange(0, head_dim)[None, :]
    )

    tl.store(k_cache + offs_kvcache, tl.load(k + offs_kv))
    tl.store(v_cache + offs_kvcache, tl.load(v + offs_kv))


def store_kvcache_prefill(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_ids: torch.Tensor,
    seq_start_locs: torch.Tensor,
    seq_lens: torch.Tensor,
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    cur_layer: int,
):
    """
    Store the key/value cache to paged attention for prefill sequences.

    Args:
        k: The key tensor of shape [num_prefill_tokens, num_kv_heads, head_dim].
        v: The value tensor of shape [num_prefill_tokens, num_kv_heads, head_dim].
        k_cache: The key cache tensor of shape [num_blocks, num_layers, num_kv_heads, block_size, head_dim].
        v_cache: The value cache tensor of shape [num_blocks, num_layers, num_kv_heads, block_size, head_dim].
        block_table: The block table tensor of shape [*, max_blocks_per_seq].
        seq_ids: The sequence IDs tensor of shape [num_seqs].
        seq_start_locs: The sequence start locations tensor of shape [num_seqs].
        seq_lens: The sequence lengths tensor of shape [num_seqs].
        model_config: The model configuration.
        engine_config: The engine configuration.
        cur_layer: The current layer index.
    """
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert seq_ids.is_contiguous()
    assert seq_lens.is_contiguous()

    num_seqs = seq_ids.shape[0]
    max_seq_len = seq_lens.max().item()
    grid = (
        num_seqs,
        cdiv(max_seq_len, engine_config.block_size),
    )
    _fwd_kvcache_mgmt_prefill_kernel[grid](
        k_cache,
        v_cache,
        k,
        v,
        block_table,
        seq_ids,
        seq_start_locs,
        seq_lens,
        cur_layer,
        model_config.num_layers,
        model_config.num_kv_heads,
        engine_config.block_size,
        model_config.head_dim,
        engine_config.max_blocks_per_seq,
    )


def store_kvcache_decode(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    cur_layer: int,
):
    """
    Store the key/value cache to paged attention for decode sequences.

    Args:
        k: The key tensor of shape [num_decode_tokens, num_kv_heads, head_dim].
        v: The value tensor of shape [num_decode_tokens, num_kv_heads, head_dim].
        k_cache: The key cache tensor of shape [num_blocks, num_layers, num_kv_heads, block_size, head_dim].
        v_cache: The value cache tensor of shape [num_blocks, num_layers, num_kv_heads, block_size, head_dim].
        block_table: The block table tensor of shape [*, max_blocks_per_seq].
        seq_ids: The sequence IDs tensor of shape [num_seqs].
        seq_lens: The sequence lengths tensor of shape [num_seqs].
        model_config: The model configuration.
        engine_config: The engine configuration.
        cur_layer: The current layer index.
    """
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert seq_ids.is_contiguous()
    assert seq_lens.is_contiguous()

    num_seqs = seq_ids.shape[0]
    grid = (num_seqs,)
    _fwd_kvcache_mgmt_decoding_kernel[grid](
        k_cache,
        v_cache,
        k,
        v,
        block_table,
        seq_ids,
        seq_lens,
        cur_layer,
        model_config.num_layers,
        model_config.num_kv_heads,
        engine_config.block_size,
        model_config.head_dim,
        engine_config.max_blocks_per_seq,
    )
