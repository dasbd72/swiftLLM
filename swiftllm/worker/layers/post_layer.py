import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace
from swiftllm.worker.weight import LlamaWeight


class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights

    def prefill(
        self,
        input_embds: torch.Tensor,  # [num_prefill_tokens, hidden_size]
        batch_size: int,
        seq_start_locs: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prefill phase of the post layer.

        Args:
            input_embds: The input embeddings tensor of shape [num_prefill_tokens, hidden_size].
            batch_size: The number of sequences in the batch.
            seq_start_locs: Tensor of shape [num_prefill_seqs] indicating the start locations of each sequence.
            seq_lens: Tensor of shape [num_prefill_seqs] indicating the lengths of each sequence.
        """
        last_token_indices = seq_start_locs + seq_lens - 1
        last_input = torch.empty(
            (batch_size, self.model_config.hidden_size),
            device=input_embds.device,
            dtype=input_embds.dtype,
        )
        last_input[:, :] = input_embds[last_token_indices, :]
        # Apply RMS-norm
        rmsnorm_inplace(
            last_input, self.weights.final_norm, self.model_config.rms_norm_eps
        )
        logits = linear(
            last_input, self.weights.lm_head
        )  # [batch_size, vocab_size]
        output_tokens = torch.argmax(logits, dim=1)
        return output_tokens

    def decode(
        self,
        input_embds: torch.Tensor,  # [num_decode_tokens, hidden_size]
        batch_size: int,
    ) -> torch.Tensor:
        """
        Decode phase of the post layer.

        Args:
            input_embds: The input embeddings tensor of shape [num_prefill_tokens, hidden_size].
            batch_size: The number of sequences in the batch.
        """
        last_token_indices = torch.arange(
            0, batch_size, device=input_embds.device, dtype=torch.int32
        )
        last_input = torch.empty(
            (batch_size, self.model_config.hidden_size),
            device=input_embds.device,
            dtype=input_embds.dtype,
        )
        last_input[:, :] = input_embds[last_token_indices, :]
        # Apply RMS-norm
        rmsnorm_inplace(
            last_input, self.weights.final_norm, self.model_config.rms_norm_eps
        )
        logits = linear(
            last_input, self.weights.lm_head
        )  # [batch_size, vocab_size]
        output_tokens = torch.argmax(logits, dim=1)
        return output_tokens
