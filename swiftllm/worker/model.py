import itertools
import math
from dataclasses import dataclass

import swiftllm_c
import torch

from swiftllm.engine_config import EngineConfig, SchedulingStrategy
from swiftllm.model_config import LlamaModelConfig
from swiftllm.utils import GB
from swiftllm.worker.block_manager import BlockManager
from swiftllm.worker.weight import load_weights

from .layers.post_layer import LlamaPostLayer
from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer


@dataclass
class _PrefillArguments:
    """Class to hold arguments for the _prefill method.
    This is created to make managing micro-batch arguments easier.
    """

    input_ids: torch.Tensor
    batch_size: int
    softmax_scale: float
    seq_ids: torch.Tensor
    seq_start_locs: torch.Tensor
    seq_lens: torch.Tensor
    position_cos: torch.Tensor
    position_sin: torch.Tensor
    ignore_kvcache: bool


@dataclass
class _DecodeArguments:
    """Class to hold arguments for the _decode method.
    This is created to make managing micro-batch arguments easier."""

    input_ids: torch.Tensor
    batch_size: int
    seq_block_size: int
    num_seq_blocks: int
    softmax_scale: float
    seq_ids: torch.Tensor
    seq_lens: torch.Tensor
    position_cos: torch.Tensor
    position_sin: torch.Tensor
    ignore_kvcache: bool


class LlamaModel:
    """
    LlamaModel - A Llama model that can be used for inference.

    This class also acts as a "worker" that resides on a particular GPU, waiting
    for the control plane (the scheduler) to send commands.

    To initialize, please:
    - call __init__()
    - call load_weights()
    - call profile_num_blocks() on one worker
    - call init_kvcache_and_swap()
    """

    @torch.inference_mode()
    def __init__(self, engine_config: EngineConfig):
        """
        Initialize the LlamaModel.
        """
        self.engine_config = engine_config

        # Load model config
        self.model_config = LlamaModelConfig.load_from_model_path(
            engine_config.model_path
        )

        # Weight and RoPE cache
        self.weight = None
        self._cos_cached = self._sin_cached = None

        # Layers
        self.pre_layer = None
        self.transformer_layers = None
        self.post_layer = None

        # KV Cache
        self.num_blocks = None
        self.k_cache = self.v_cache = None
        self.k_swap = self.v_swap = None

        # Block manager
        self.cpu_block_manager = self.gpu_block_manager = None

        # Forward streams
        self.htod_stream = torch.cuda.Stream()

    @torch.inference_mode()
    def load_weights(self):
        """
        Load weights and initialize layers
        """
        # Load weights
        self.weight = load_weights(
            self.model_config,
            torch.float16,
            self.engine_config.model_path,
            self.engine_config.use_dummy,
            device=self.engine_config.weight_device,
        )

        # Initialize rotary embeddings
        self._init_to_get_rotary()

        # Initialize layers
        self.pre_layer = LlamaPreLayer(
            self.model_config, self.weight, self.engine_config.weight_device
        )
        self.pre_layer.weight_to_gpu()
        self.transformer_layers: list[LlamaTransformerLayer] = []
        for layer_id in range(self.model_config.num_layers):
            layer = LlamaTransformerLayer(
                self.model_config,
                self.engine_config,
                self.weight.layers[layer_id],
                self.engine_config.weight_device,
                layer_id,
            )
            self.transformer_layers.append(layer)
        self.post_layer = LlamaPostLayer(
            self.model_config, self.weight, self.engine_config.weight_device
        )
        self.post_layer.weight_to_gpu()

    @torch.inference_mode()
    def profile_num_blocks(self) -> int:
        """
        Profiler the number of GPU blocks

        We run a forged prefill batch with the maximum number of tokens and
        sequences, record the peak memory usage, and infer the number of blocks
        that can be allocated.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Synthesis a prefill batch
        num_tokens = self.engine_config.max_tokens_in_batch
        batch_size = self.engine_config.max_batch_size
        profile_scheduling_strategy = (
            self.engine_config.profile_scheduling_strategy
        )
        input_lens = [num_tokens // batch_size] * batch_size
        input_lens[-1] += num_tokens % batch_size
        input_ids = [[0 for _ in range(input_len)] for input_len in input_lens]
        seq_ids = list(range(batch_size))
        self.k_cache = self.v_cache = (
            None  # pylint: disable=attribute-defined-outside-init
        )
        _ = self.prefill(
            input_ids,
            seq_ids,
            ignore_kvcache=True,
            scheduling_strategy=profile_scheduling_strategy,
        )
        torch.cuda.synchronize()

        # peak_memory = torch.cuda.max_memory_allocated()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        peak_memory = total_memory - free_memory
        useable_memory = total_memory * self.engine_config.gpu_mem_utilization
        print(
            f"[Model.profile] GPU total memory: {total_memory/GB:.2f} GB, runtime peak memory: {peak_memory/GB:.2f} GB"
        )
        if useable_memory < peak_memory:
            raise RuntimeError(
                f"Peak memory {peak_memory/GB:.2f} GB exceeds usable memory {useable_memory/GB:.2f} GB ({total_memory/GB:.2f} GB * {self.engine_config.gpu_mem_utilization})"
            )
        block_size_bytes = (
            self.engine_config.block_size * self.model_config.get_kvslot_size()
        )
        num_gpu_blocks = math.floor(
            (useable_memory - peak_memory) / block_size_bytes
        )

        torch.cuda.empty_cache()
        return num_gpu_blocks

    @torch.inference_mode()
    def init_kvcache_and_swap(self, num_blocks: int):
        self.num_blocks = num_blocks

        # Initialize KV cache
        kvcache_shape = (
            self.num_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim,
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(
            kvcache_shape, dtype=torch.float16, device="cuda"
        )
        self.v_cache = torch.zeros(
            kvcache_shape, dtype=torch.float16, device="cuda"
        )

        # Initialize KV swap space
        kvswap_shape = (
            self.engine_config.num_cpu_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim,
        )
        self.k_swap = torch.zeros(
            kvswap_shape, dtype=torch.float16, device="cpu"
        )
        self.v_swap = torch.zeros(
            kvswap_shape, dtype=torch.float16, device="cpu"
        )

        # Initialize block manager
        self.gpu_block_manager = BlockManager(
            "GPU",
            self.num_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size,
        )
        self.cpu_block_manager = BlockManager(
            "CPU",
            self.engine_config.num_cpu_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size,
        )

    def _init_to_get_rotary(self):
        rope_scaling_factor = self.model_config.rope_scaling
        base = self.model_config.rope_theta
        max_position_embeddings = self.model_config.max_position_embeddings
        max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    self.model_config.head_dim,
                    2,
                    device="cuda",
                    dtype=torch.float32,
                )
                / self.model_config.head_dim
            )
        )
        t = (
            torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32)
            / rope_scaling_factor
        )
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16)
        self._sin_cached = torch.sin(freqs).to(torch.float16)

    def _allocate_blocks_for_seqs(
        self,
        seq_ids: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """
        Allocate blocks for the given sequences.
        This is a helper function to allocate blocks for the sequences in prefill and decode.

        Arg:
            seq_ids torch.Tensor: A tensor of sequence IDs, shape [num_seqs].
            seq_lens torch.Tensor: A tensor of sequence lengths, shape [num_seqs].
        """
        self.gpu_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lens)

    @torch.inference_mode()
    def _prefill(
        self,
        args: _PrefillArguments,
    ) -> torch.Tensor:
        """
        Run a prefill pass of the LlamaModel.
        """
        assert self.transformer_layers is not None
        input_embds = self.pre_layer.forward(args.input_ids)
        residual_buf = torch.zeros_like(input_embds)
        for layer in self.transformer_layers:
            input_embds = layer.prefill(
                input_embds,
                residual_buf,
                self.k_cache,
                self.v_cache,
                (
                    self.gpu_block_manager.block_table
                    if not args.ignore_kvcache
                    else None
                ),
                args.softmax_scale,
                args.seq_ids,
                args.seq_start_locs,
                args.seq_lens,
                args.position_cos,
                args.position_sin,
                args.ignore_kvcache,
            )
        input_embds += residual_buf
        output_tokens = self.post_layer.prefill(
            input_embds, args.batch_size, args.seq_start_locs, args.seq_lens
        )
        return output_tokens

    @torch.inference_mode()
    def _prefill_offload_weight(
        self,
        args: _PrefillArguments,
    ) -> torch.Tensor:
        """
        Run a prefill pass of the LlamaModel with weight offloading.
        """
        assert self.transformer_layers is not None
        input_embds_list = [None]
        residual_buf_list = [None]

        def is_layer_id_in_range(layer_id: int):
            return layer_id >= 0 and layer_id < self.model_config.num_layers

        def prefill_load_weight(layer_id: int):
            # Load the weight of the next layer
            if not is_layer_id_in_range(layer_id):
                return
            with torch.cuda.stream(self.htod_stream):
                self.transformer_layers[layer_id].weight_to_gpu()

        def prefill_free_weight(layer_id: int):
            # Free the weight of the current layer
            self.transformer_layers[layer_id].weight_gpu_free()

        def prefill_compute(layer_id: int):
            # Compute the prefill pass of the current layer
            block_table = (
                self.gpu_block_manager.block_table
                if not args.ignore_kvcache
                else None
            )
            input_embds_list[0] = self.transformer_layers[layer_id].prefill(
                input_embds_list[0],
                residual_buf_list[0],
                self.k_cache,
                self.v_cache,
                (
                    self.gpu_block_manager.block_table
                    if not args.ignore_kvcache
                    else None
                ),
                args.softmax_scale,
                args.seq_ids,
                args.seq_start_locs,
                args.seq_lens,
                args.position_cos,
                args.position_sin,
                args.ignore_kvcache,
            )

        input_embds_list[0] = self.pre_layer.forward(args.input_ids)
        residual_buf_list[0] = torch.zeros_like(input_embds_list[0])
        # load the weight of the first layer
        prefill_load_weight(0)
        torch.cuda.synchronize()
        for layer_id in range(self.model_config.num_layers):
            # load the weight of the next layer
            prefill_load_weight(layer_id + 1)
            # compute the prefill pass of the current layer
            prefill_compute(layer_id)
            torch.cuda.synchronize()
            # free the weight of the current layer
            prefill_free_weight(layer_id)
        input_embds_list[0] += residual_buf_list[0]
        output_tokens = self.post_layer.prefill(
            input_embds_list[0],
            args.batch_size,
            args.seq_start_locs,
            args.seq_lens,
        )
        return output_tokens

    @torch.inference_mode()
    def _pre_prefill(
        self,
        input_ids_list: list[list[int]],  # [batch_size, *]
        seq_ids_list: list[int],  # [batch_size]
        ignore_kvcache: bool = False,  # Skip actions related to kv cache, useful when profiling the number of kv blocks
        **kwargs,
    ):
        flattened_input_ids = list(itertools.chain(*input_ids_list))
        input_ids = torch.tensor(
            flattened_input_ids, dtype=torch.int32, device="cuda"
        )
        batch_size = len(input_ids_list)
        softmax_scale = self.model_config.head_dim**-0.5
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_len_list = [len(seq) for seq in input_ids_list]
        seq_lens = torch.tensor(seq_len_list, dtype=torch.int32, device="cuda")
        seq_start_locs = (
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32) - seq_lens
        )
        position_indices = torch.concat(
            [
                torch.arange(
                    0,
                    seq_len,
                    device="cuda",
                    dtype=torch.int32,
                )
                for seq_len in seq_len_list
            ]
        )
        position_cos = self._cos_cached[position_indices]
        position_sin = self._sin_cached[position_indices]

        if not ignore_kvcache:
            self._allocate_blocks_for_seqs(seq_ids, seq_lens)

        return _PrefillArguments(
            input_ids=input_ids,
            batch_size=batch_size,
            softmax_scale=softmax_scale,
            seq_ids=seq_ids,
            seq_start_locs=seq_start_locs,
            seq_lens=seq_lens,
            position_cos=position_cos,
            position_sin=position_sin,
            ignore_kvcache=ignore_kvcache,
        )

    @torch.inference_mode()
    def prefill(
        self,
        input_ids_list: list[list[int]],  # [batch_size, *]
        seq_ids_list: list[int],  # [batch_size]
        ignore_kvcache: bool = False,  # Skip actions related to kv cache, useful when profiling the number of kv blocks
        scheduling_strategy: SchedulingStrategy = "gpu",
        **kwargs,
    ):
        """
        Run a prefill pass of the LlamaModel.
        """
        args = self._pre_prefill(
            input_ids_list,
            seq_ids_list,
            ignore_kvcache=ignore_kvcache,
            **kwargs,
        )
        if scheduling_strategy == "gpu":
            output_tokens = self._prefill(args).tolist()
        elif scheduling_strategy == "offload-weight":
            output_tokens = self._prefill_offload_weight(args).tolist()
        else:
            raise ValueError(
                f"Unsupported scheduling strategy: {scheduling_strategy}"
            )
        return output_tokens

    @torch.inference_mode()
    def _decode(
        self,
        args: _DecodeArguments,
    ) -> torch.Tensor:
        """
        Run a decode pass of the LlamaModel.
        """
        assert self.transformer_layers is not None
        input_embds = self.pre_layer.forward(args.input_ids)
        residual_buf = torch.zeros_like(input_embds)
        for layer in self.transformer_layers:
            input_embds = layer.decode(
                input_embds,
                residual_buf,
                self.k_cache,
                self.v_cache,
                (
                    self.gpu_block_manager.block_table
                    if not args.ignore_kvcache
                    else None
                ),
                args.seq_block_size,
                args.num_seq_blocks,
                args.softmax_scale,
                args.seq_ids,
                args.seq_lens,
                args.position_cos,
                args.position_sin,
                args.ignore_kvcache,
            )
        input_embds += residual_buf
        output_tokens = self.post_layer.decode(input_embds, args.batch_size)
        return output_tokens

    @torch.inference_mode()
    def _decode_weight_offload(
        self,
        args: _DecodeArguments,
    ) -> torch.Tensor:
        """
        Run a decode pass of the LlamaModel with weight offloading.
        """
        assert self.transformer_layers is not None
        input_embds_list = [None]
        residual_buf_list = [None]

        def is_layer_id_in_range(layer_id: int):
            return layer_id >= 0 and layer_id < self.model_config.num_layers

        def decode_load_weight(layer_id: int):
            # Load the weight of the next layer
            if not is_layer_id_in_range(layer_id):
                return
            with torch.cuda.stream(self.htod_stream):
                self.transformer_layers[layer_id].weight_to_gpu()

        def decode_free_weight(layer_id: int):
            # Free the weight of the current layer
            self.transformer_layers[layer_id].weight_gpu_free()

        def decode_compute(layer_id: int):
            # Compute the decode pass of the current layer
            block_table = (
                self.gpu_block_manager.block_table
                if not args.ignore_kvcache
                else None
            )
            input_embds_list[0] = self.transformer_layers[layer_id].decode(
                input_embds_list[0],
                residual_buf_list[0],
                self.k_cache,
                self.v_cache,
                block_table,
                args.seq_block_size,
                args.num_seq_blocks,
                args.softmax_scale,
                args.seq_ids,
                args.seq_lens,
                args.position_cos,
                args.position_sin,
                args.ignore_kvcache,
            )

        input_embds_list[0] = self.pre_layer.forward(args.input_ids)
        residual_buf_list[0] = torch.zeros_like(input_embds_list[0])
        # load the weight of the first layer
        decode_load_weight(0)
        torch.cuda.synchronize()
        for layer_id in range(self.model_config.num_layers):
            # load the weight of the next layer
            decode_load_weight(layer_id + 1)
            # compute the decode pass of the current layer
            decode_compute(layer_id)
            torch.cuda.synchronize()
            # free the weight of the current layer
            decode_free_weight(layer_id)
        input_embds_list[0] += residual_buf_list[0]
        output_tokens = self.post_layer.decode(
            input_embds_list[0], args.batch_size
        )
        return output_tokens

    @torch.inference_mode()
    def _pre_decode(
        self,
        input_ids_list: list[list[int]],  # [batch_size, *]
        seq_ids_list: list[int],  # [batch_size]
        seq_len_list: list[int],  # [batch_size]
        ignore_kvcache: bool = False,  # Skip actions related to kv cache, useful when profiling the number of kv blocks
        **kwargs,
    ):
        flattened_input_ids = list(itertools.chain(*input_ids_list))
        input_ids = torch.tensor(
            flattened_input_ids, dtype=torch.int32, device="cuda"
        )
        batch_size = len(input_ids_list)
        softmax_scale = self.model_config.head_dim**-0.5
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor(seq_len_list, dtype=torch.int32, device="cuda")
        max_seq_len = seq_lens.max().item()
        position_indices = seq_lens - 1
        position_cos = self._cos_cached[position_indices]
        position_sin = self._sin_cached[position_indices]

        if not ignore_kvcache:
            self._allocate_blocks_for_seqs(seq_ids, seq_lens)

        # Select the seq_block_size
        #
        # Here we use a simple heuristic:
        #
        # In paged attention phase 1, the grid shape is (num_decoding_seqs, num_kv_heads, cdiv(max_decoding_len, seq_block_size))
        # and among these blocks, num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) blocks are useful.
        # Thus we set seq_block_size to be the largest integer that satisfies
        #      num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) >= 1024
        # to fully utilize the GPU. Here 1024 is a magic number (since most high-end
        # GPUs have ~128 SMs, so ~512 SMSPs. Since the decoding-stage attention
        # is mostly a memory-bound operation, I think 1024 is a reasonable number.)
        #
        # In practice, we use `decoding_seq_lens_sum/seq_block_size` to approximate
        # sum(cdiv(decoding_seq_lens, seq_block_size))

        seq_block_size = 2048
        decoding_seq_lens_sum = sum(seq_len_list)
        while (
            self.model_config.num_kv_heads
            * (decoding_seq_lens_sum / seq_block_size)
            < 1024
            and seq_block_size // 2 >= 64
            and max_seq_len / (seq_block_size // 2) <= 128
        ):
            seq_block_size //= 2
        num_seq_blocks = (max_seq_len + seq_block_size - 1) // seq_block_size

        return _DecodeArguments(
            input_ids=input_ids,
            batch_size=batch_size,
            seq_block_size=seq_block_size,
            num_seq_blocks=num_seq_blocks,
            softmax_scale=softmax_scale,
            seq_ids=seq_ids,
            seq_lens=seq_lens,
            position_cos=position_cos,
            position_sin=position_sin,
            ignore_kvcache=ignore_kvcache,
        )

    def decode(
        self,
        input_ids_list: list[list[int]],  # [batch_size, *]
        seq_ids_list: list[int],  # [batch_size]
        seq_len_list: list[int],  # [batch_size]
        ignore_kvcache: bool = False,  # Skip actions related to kv cache, useful when profiling the number of kv blocks
        scheduling_strategy: SchedulingStrategy = "gpu",
        **kwargs,
    ):
        """
        Run a decode pass of the LlamaModel.
        """
        args = self._pre_decode(
            input_ids_list,
            seq_ids_list,
            seq_len_list,
            ignore_kvcache=ignore_kvcache,
            **kwargs,
        )
        if scheduling_strategy == "gpu":
            output_tokens = self._decode(args).tolist()
        elif scheduling_strategy == "offload-weight":
            output_tokens = self._decode_weight_offload(args).tolist()
        else:
            raise ValueError(
                f"Unsupported scheduling strategy: {scheduling_strategy}"
            )
        return output_tokens

    def _swap(self, seq_ids_list: list[int], is_swap_in: bool):
        src_block_manager = (
            self.cpu_block_manager if is_swap_in else self.gpu_block_manager
        )
        dst_block_manager = (
            self.gpu_block_manager if is_swap_in else self.cpu_block_manager
        )
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = (
            src_block_manager.get_num_allocated_blocks(seq_ids)
            * self.engine_config.block_size
        )
        src_block_ids = src_block_manager.gather_allocated_blocks_and_free(
            seq_ids
        )
        dst_block_ids = dst_block_manager.allocate_blocks_for_seqs(
            seq_ids, seq_lengths
        )
        swiftllm_c.swap_blocks(
            src_block_ids.tolist(),
            dst_block_ids.tolist(),
            is_swap_in,
            self.k_cache,
            self.v_cache,
            self.k_swap,
            self.v_swap,
        )

    @torch.inference_mode()
    def swap_in_seqs(self, seq_ids_list: list[int]):
        """
        Swap in (move blocks from CPU to GPU) the specified sequences.
        """
        self._swap(seq_ids_list, True)

    @torch.inference_mode()
    def swap_out_seqs(self, seq_ids_list: list[int]):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        self._swap(seq_ids_list, False)

    @torch.inference_mode()
    def free_seqs_resources(self, seq_ids_list: list[int]):
        """
        Free the resources of the specified sequences.
        """
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        self.gpu_block_manager.free_blocks_for_seqs(seq_ids)
        self.cpu_block_manager.free_blocks_for_seqs(seq_ids)

    @torch.inference_mode()
    def get_num_used_gpu_blocks(self) -> int:
        """
        Get the number of used GPU blocks.
        """
        return (
            self.gpu_block_manager.num_blocks
            - self.gpu_block_manager.num_free_blocks
        )

    @torch.inference_mode()
    def get_num_used_cpu_blocks(self) -> int:
        """
        Get the number of used CPU blocks.
        """
        return (
            self.cpu_block_manager.num_blocks
            - self.cpu_block_manager.num_free_blocks
        )
