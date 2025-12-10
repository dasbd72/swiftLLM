import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight


class LlamaPreLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
        weight_device: str,
        pin_weight_cpu: bool,
    ):
        self.model_config = model_config
        self.weights = weights
        self.weight_device = weight_device
        self.weight_names = ["wte"]
        self.weights_cpu = {name: None for name in self.weight_names}
        if weight_device == "cpu":
            for name in self.weight_names:
                self.weights_cpu[name] = (
                    getattr(weights, name).pin_memory()
                    if pin_weight_cpu
                    else getattr(weights, name).to("cpu")
                )
                setattr(weights, name, None)

    def weight_to_gpu(self):
        """
        Load weights to GPU if they are on CPU
        """
        if self.weight_device == "cpu":
            for name in self.weight_names:
                setattr(
                    self.weights,
                    name,
                    self.weights_cpu[name].to("cuda", non_blocking=True),
                )

    def weight_gpu_free(self):
        """
        Free weights if we have a copy on CPU
        """
        if self.weight_device == "cpu":
            for name in self.weight_names:
                setattr(self.weights, name, None)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_embdings = torch.embedding(
            self.weights.wte, input_ids, padding_idx=-1
        )
        return input_embdings
