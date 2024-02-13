from torch import (
    mean,
    Tensor,
    sum as tensor_sum,
    log2,
    erf,
    clamp,
)
from torch.nn.functional import mse_loss
from torch.nn import Module

from gym.config import get_config


class NormalDistributionRateDistortionLoss(Module):
    def __init__(self) -> None:
        super().__init__()
        self.__lambda = float(get_config()["network"]["lambda"])
        self.__epsilon = float(get_config()["network"]["epsilon"])

    @staticmethod
    def __normal_distribution_cdf(
        mean_: Tensor, standard_deviation: Tensor, points: Tensor
    ) -> Tensor:
        return (1.0 + erf((points - mean_) / ((2.0**0.5) * standard_deviation))) / 2.0

    @staticmethod
    def __standard_distribution_cdf(points: Tensor) -> Tensor:
        return (1.0 + erf(points / (2.0**0.5))) / 2.0

    def __entropy(self, upper_bound: Tensor, lower_bound: Tensor) -> Tensor:
        difference = clamp(upper_bound - lower_bound, min=self.__epsilon)
        return -tensor_sum(log2(difference), dim=(1, 2, 3))

    def __latent_rate(self, mean_: Tensor, stddev: Tensor, latent: Tensor) -> Tensor:
        upper_bound = self.__normal_distribution_cdf(mean_, stddev, latent + 0.5)
        lower_bound = self.__normal_distribution_cdf(mean_, stddev, latent - 0.5)
        return self.__entropy(upper_bound, lower_bound)

    def __hyperlatent_rate(self, hyperlatent: Tensor) -> Tensor:
        upper_bound = self.__standard_distribution_cdf(hyperlatent + 0.5)
        lower_bound = self.__standard_distribution_cdf(hyperlatent - 0.5)
        return self.__entropy(upper_bound, lower_bound)

    def forward(
        self,
        original: Tensor,
        reconstrution: Tensor,
        latent_mean: Tensor,
        latent_standard_deviation: Tensor,
        latent: Tensor,
        hyperlatent: Tensor,
    ) -> Tensor:
        distortion = mse_loss(reconstrution, original)
        latent_rate = mean(
            self.__latent_rate(latent_mean, latent_standard_deviation, latent)
        )
        hyperlatent_rate = mean(self.__hyperlatent_rate(hyperlatent))
        return self.__lambda * distortion + latent_rate + hyperlatent_rate
