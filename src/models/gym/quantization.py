from torch import Tensor, round as tensor_round, rand, device, float32
from torch.nn import Module


class Quantization(Module):
    def __init__(self) -> None:
        super().__init__()

    def __quantize_training(self, vector: Tensor) -> Tensor:
        detached = vector.detach()
        rounded = tensor_round(detached)
        return (rounded - detached) + vector

    def forward(self, vector: Tensor, is_training: bool) -> Tensor:
        if is_training:
            return self.__quantize_training(vector)  # soft quantization
        return tensor_round(vector)  # hard quantization


class FuzzyQuantization(Module):
    def __init__(self, device: device) -> None:
        super().__init__()
        self.__device = device

    def __fuzzy_quantization(self, vector: Tensor) -> Tensor:
        return vector + (rand(vector.shape, dtype=float32) - 0.5).to(self.__device)

    def forward(self, vector: Tensor, is_training: bool) -> Tensor:
        if is_training:
            return self.__fuzzy_quantization(vector)
        return tensor_round(vector)
