from __future__ import annotations
from typing import TypeVar

from torch.nn import (
    Module,
    Conv2d,
    BatchNorm2d,
    LeakyReLU,
    Sequential,
    ConvTranspose2d,
    Sigmoid,
    Flatten,
    Unflatten,
    Linear,
    BatchNorm1d,
)
from torch.nn import Conv3d, ConvTranspose3d, Softplus
from torch import Tensor, unbind, flatten, cat, chunk, stack, split, clamp

import pytorch_wavelets as wvlt

from pytorch_gdn import GDN

from gym.quantization import Quantization
from gym.modules import DWT, IDWT, Squeeze, Unsqueeze, MaskedConv2d, Wavelon
from gym.wavelets import standard_mexican_hat_wavelet


N: int = 64
WAVELET: str = "haar"


class ConvexWaveletLens(Module):
    def __init__(self, device: str, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv3d_kernel = (4, 1, 1)
        conv3d_padding = (0, 0, 0)
        self.__model = Sequential(
            Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            DWT(WAVELET),
            Conv3d(
                in_channels,
                out_channels,
                kernel_size=conv3d_kernel,
                padding=conv3d_padding,
            ),
            Squeeze(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.__model(x)


class ConcaveWaveletLens(Module):
    def __init__(self, device: str, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv3d_kernel = (4, 1, 1)
        conv3d_padding = (0, 0, 0)
        self.__model = Sequential(
            Unsqueeze(2),
            ConvTranspose3d(
                in_channels,
                in_channels,
                kernel_size=conv3d_kernel,
                padding=conv3d_padding,
            ),
            IDWT(WAVELET),
            ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.__model(x)


class Encoder(Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.__model = Sequential(
            ConvexWaveletLens(device, 3, N),
            GDN(N, device),
            BatchNorm2d(N),
            ConvexWaveletLens(device, N, N),
            GDN(N, device),
            BatchNorm2d(N),
            ConvexWaveletLens(device, N, N),
            GDN(N, device),
            BatchNorm2d(N),
            ConvexWaveletLens(device, N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.__model(x)


class Decoder(Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.__model = Sequential(
            ConcaveWaveletLens(device, N, N),
            GDN(N, device, inverse=True),
            BatchNorm2d(N),
            ConcaveWaveletLens(device, N, N),
            GDN(N, device, inverse=True),
            BatchNorm2d(N),
            ConcaveWaveletLens(device, N, N),
            GDN(N, device, inverse=True),
            BatchNorm2d(N),
            ConcaveWaveletLens(device, N, 3),
            Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.__model(x)


class HyperEncoder(Module):
    def __init__(self) -> None:
        super().__init__()
        self.__model = Sequential(
            Conv2d(N, N, kernel_size=5, stride=1),
            Wavelon(standard_mexican_hat_wavelet),
            BatchNorm2d(N),
            Conv2d(N, N, kernel_size=5, stride=1, padding=1),
            Wavelon(standard_mexican_hat_wavelet),
            BatchNorm2d(N),
            Conv2d(N, N, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.__model(x)


class HyperDecoder(Module):
    def __init__(self) -> None:
        super().__init__()
        middle_channels = int(N * 1.5)
        out_channels = int(N * 2)
        self.__model = Sequential(
            ConvTranspose2d(N, N, kernel_size=3, stride=2, padding=1),
            Wavelon(standard_mexican_hat_wavelet),
            BatchNorm2d(N),
            ConvTranspose2d(N, middle_channels, kernel_size=5, stride=1, padding=1),
            Wavelon(standard_mexican_hat_wavelet),
            BatchNorm2d(middle_channels),
            ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.__model(x)


class ContextPredictor(Module):
    def __init__(self) -> None:
        super().__init__()
        self.__model = MaskedConv2d(
            in_channels=N, out_channels=int(2 * N), kernel_size=5, padding=2
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.__model(x)


class EntropyParameters(Module):
    def __init__(self) -> None:
        super().__init__()
        self.__min_stddev = float(get_config()["network"]["min_stddev"])
        in_channels = int(N * 4)
        l1_channels = int(10 * N / 3)
        l2_channels = int(7 * N / 3)
        out_channels = int(2 * N)
        self.__model = Sequential(
            Conv2d(in_channels, l1_channels, kernel_size=1),
            Wavelon(standard_mexican_hat_wavelet),
            BatchNorm2d(l1_channels),
            Conv2d(l1_channels, l2_channels, kernel_size=1),
            Wavelon(standard_mexican_hat_wavelet),
            BatchNorm2d(l2_channels),
            Conv2d(l2_channels, out_channels, kernel_size=1),
        )
        self.__softplus = Softplus()

    def forward(self, x: Tensor, shape: int) -> tuple[Tensor, Tensor]:
        stddev, mean = split(self.__model(x), shape, dim=1)
        return self.__softplus(stddev) + self.__min_stddev, mean


LossFunction = TypeVar("LossFunction")


class Network(Module):
    def __init__(
        self,
        loss_function: LossFunction,
        device: str,
        quantization: Quantization | None = None,
    ) -> None:
        super().__init__()
        self.__encoder = Encoder(device)
        self.__decoder = Decoder(device)
        self.__hyperencoder = HyperEncoder()
        self.__hyperdecoder = HyperDecoder()
        self.__context = ContextPredictor()
        self.__entropy = EntropyParameters()
        self.__loss_function = loss_function
        self.__quant = quantization

    def __quantize(self, x: Tensor) -> Tensor:
        if self.__quant is not None:
            x = self.__quant(x, self.training)
        return x

    def forward(self, x: Tensor, quantization_step: float = 1) -> tuple[Tensor, Tensor]:
        latent = self.__encoder(x)
        quantized_latent = self.__quantize(
            latent * quantization_step if quantization_step != 1 else latent
        )
        decoded = self.__decoder(
            quantized_latent / quantization_step
            if quantization_step != 1
            else quantized_latent
        )
        hyperlatent = self.__quantize(self.__hyperencoder(latent))
        hyperpriors = self.__hyperdecoder(hyperlatent)
        context = self.__context(
            quantized_latent / quantization_step
            if quantization_step != 1
            else quantized_latent
        )
        stddev, mean = self.__entropy(
            cat([context, hyperpriors], dim=1), latent.shape[1]
        )
        return quantized_latent, hyperlatent, stddev, mean, decoded

    def loss(
        self,
        latent: Tensor,
        hyperlatent: Tensor,
        stddev: Tensor,
        mean: Tensor,
        x: Tensor,
        x_prime: Tensor,
    ) -> Tensor:
        return self.__loss_function(x, x_prime, mean, stddev, latent, hyperlatent)
