from __future__ import annotations

import pytorch_wavelets as wvlt

from torch.nn.init import xavier_uniform_
from torch.nn import Module, Conv2d, Parameter
from torch.nn.functional import conv2d, conv_transpose2d, pad as torch_pad
from torch import Tensor, cat, chunk, flatten, stack, empty, ones, unbind
from torch.autograd import Function

import pywt


class FlatDWT(Module):
    def __init__(self, family: str) -> None:
        super().__init__()
        self.__dwt = wvlt.DWT(J=1, wave=family)

    def forward(self, x: Tensor) -> Tensor:
        ll, lh = self.__dwt(x)
        (lh,) = lh
        return cat((ll, flatten(lh, start_dim=2, end_dim=3)), dim=2)


class FlatIDWT(Module):
    def __init__(self, family: str) -> None:
        super().__init__()
        self.__idwt = wvlt.IDWT(wave=family)

    def forward(self, x: Tensor) -> Tensor:
        ll, l1, l2, l3 = chunk(x, 4, dim=2)
        stacked = stack((l1, l2, l3), dim=2)
        return self.__idwt((ll, [stacked]))


class DWT(Module):
    def __init__(self, family: str) -> None:
        super().__init__()
        self.__dwt = wvlt.DWT(J=1, wave=family)

    def forward(self, x: Tensor) -> Tensor:
        ll, lh = self.__dwt(x)
        (lh,) = lh
        return cat((ll.unsqueeze(2), lh), dim=2)


class IDWT(Module):
    def __init__(self, family: str) -> None:
        super().__init__()
        self.__idwt = wvlt.IDWT(wave=family)

    def forward(self, x: Tensor) -> Tensor:
        ll, l1, l2, l3 = chunk(x, 4, dim=2)
        return self.__idwt((ll.squeeze(2), [cat((l1, l2, l3), dim=2)]))


class Squeeze(Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.__dimension = dimension

    def forward(self, x: Tensor) -> Tensor:
        return x.squeeze(self.__dimension)


class Unsqueeze(Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.__dimension = dimension

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(self.__dimension)


class MaskedConv2d(Conv2d):
    """
    Code from https://www.codeproject.com/Articles/5061271/PixelCNN-in-Autoregressive-Models

        Implementation of the Masked convolution from the paper
        Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders."
    Advances in neural information processing systems. 2016.
        https://arxiv.org/pdf/1606.05328.pdf
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", self.weight.data.clone())
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, height // 2, width // 2 :] = 0
        self.mask[:, :, height // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class Wavelon(Module):
    def __init__(self, wavelet: callable) -> None:
        super().__init__()
        self.__wavelet = wavelet

    def forward(self, x: Tensor) -> Tensor:
        return self.__wavelet(x)


class AdaptiveDWT(Module):
    """This class is a rework of the pytorch dwt's DWT with learnable filters"""

    def __init__(self) -> None:
        super().__init__()
        self.hi_cols = Parameter(xavier_uniform_(ones((1, 1, 2, 1))))
        self.lo_cols = Parameter(xavier_uniform_(ones((1, 1, 2, 1))))
        self.hi_rows = Parameter(xavier_uniform_(ones((1, 1, 1, 2))))
        self.lo_rows = Parameter(xavier_uniform_(ones((1, 1, 1, 2))))

    @staticmethod
    def __afb1d(x, h0, h1, dim=-1):
        C = x.shape[1]
        # Convert the dim to positive
        d = dim % 4
        s = (2, 1) if d == 2 else (1, 2)
        N = x.shape[d]

        # If h0, h1 are not tensors, make them. If they are, then assume that they
        # are in the right order
        L = h0.numel()
        L2 = L // 2
        shape = [1, 1, 1, 1]
        shape[d] = L
        # If h aren't in the right shape, make them so
        if h0.shape != tuple(shape):
            h0 = h0.reshape(*shape)
        if h1.shape != tuple(shape):
            h1 = h1.reshape(*shape)
        h = cat([h0, h1] * C, dim=0)
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode="zero")
        p = 2 * (outsize - 1) - N + L
        # Sadly, pytorch only allows for same padding before and after, if
        # we need to do more padding after for odd length signals, have to
        # prepad
        if p % 2 == 1:
            pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
            x = torch_pad(x, pad)
        pad = (p // 2, 0) if d == 2 else (0, p // 2)
        # Calculate the high and lowpass
        return conv2d(x, h, padding=pad, stride=s, groups=C)

    def forward(self, x: Tensor) -> Tensor:
        lohi = self.__afb1d(x, self.hi_rows, self.lo_rows, dim=3)
        y = self.__afb1d(lohi, self.hi_cols, self.lo_cols, dim=2)
        shape = y.shape
        return y.reshape(shape[0], -1, 4, shape[-2], shape[-1])


class AdaptiveIDWT(Module):
    """This class is a rework of the pytorch dwt's IDWT with learnable filters"""

    def __init__(self) -> None:
        super().__init__()
        self.hi_cols = Parameter(xavier_uniform_(ones((1, 1, 2, 1))))
        self.lo_cols = Parameter(xavier_uniform_(ones((1, 1, 2, 1))))
        self.hi_rows = Parameter(xavier_uniform_(ones((1, 1, 1, 2))))
        self.lo_rows = Parameter(xavier_uniform_(ones((1, 1, 1, 2))))

    @staticmethod
    def __sfb1d(lo, hi, g0, g1, dim=-1):
        C = lo.shape[1]
        d = dim % 4
        L = g0.numel()
        shape = [1, 1, 1, 1]
        shape[d] = L
        N = 2 * lo.shape[d]
        # If g aren't in the right shape, make them so
        if g0.shape != tuple(shape):
            g0 = g0.reshape(*shape)
        if g1.shape != tuple(shape):
            g1 = g1.reshape(*shape)

        s = (2, 1) if d == 2 else (1, 2)
        g0 = cat([g0] * C, dim=0)
        g1 = cat([g1] * C, dim=0)
        pad = (L - 2, 0) if d == 2 else (0, L - 2)
        return conv_transpose2d(
            lo, g0, stride=s, padding=pad, groups=C
        ) + conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)

    def forward(self, x: Tensor) -> Tensor:
        ll, lh, hl, hh = unbind(x, dim=2)
        lo = self.__sfb1d(ll, lh, self.hi_cols, self.lo_cols, dim=2)
        hi = self.__sfb1d(hl, hh, self.hi_cols, self.lo_cols, dim=2)
        return self.__sfb1d(lo, hi, self.hi_rows, self.lo_rows, dim=3)


class FullDWT(Module):
    def __init__(self, decomposition_levels: int = 1, wavelet: str = "haar") -> None:
        super().__init__()
        self.__decomposition_levels = decomposition_levels
        self.__dwt = wvlt.DWT(J=1, wave=wavelet)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.__decomposition_levels):
            ll, lh = self.__dwt(x)
            (lh,) = lh
            x = cat((ll.unsqueeze(2), lh), dim=2).flatten(1, 2)
        return x


class FullIDWT(Module):
    def __init__(self, decomposition_levels: int = 1, wavelet: str = "haar") -> None:
        super().__init__()
        self.__decomposition_levels = decomposition_levels
        self.__idwt = wvlt.IDWT(wave=wavelet)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.__decomposition_levels):
            x = x.unflatten(1, (int(x.shape[1] / 4), 4))
            ll, l1, l2, l3 = chunk(x, 4, dim=2)
            x = self.__idwt((ll.squeeze(2), [cat((l1, l2, l3), dim=2)]))
        return x
