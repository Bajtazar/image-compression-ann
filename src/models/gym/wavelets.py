from torch import Tensor, exp, sqrt
from math import pi


def mexican_hat_wavelet(points: Tensor, sigma: Tensor) -> Tensor:
    coeff = 2.0 / (sqrt(3 * sigma) * pi**0.25)
    return (
        coeff * (1.0 - points**2 / sigma) * exp(-(points**2) / (2.0 * sigma**2))
    )


def standard_mexican_hat_wavelet(points: Tensor) -> Tensor:
    coeff = 2.0 / (3.0**0.5 * pi**0.25)
    return coeff * (1.0 - points**2) * exp(-(points**2) / 2.0)
