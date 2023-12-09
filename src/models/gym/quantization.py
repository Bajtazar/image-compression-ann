from torch import Tensor, round as tensor_round
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
