from __future__ import annotations
from statistics import median
from itertools import chain
from json import dump
from collections import Counter

import matplotlib.pyplot as plt

from torch import Tensor, tensor, float32, unique


class CdfDistributionCalculator:
    def __init__(self) -> None:
        self.__means = []
        self.__stddevs = []

    def update(self, mean: Tensor, stddev: Tensor) -> CdfDistributionCalculator:
        self.__means.extend(mean.data.flatten(start_dim=0).tolist())
        self.__stddevs.extend(stddev.data.flatten(start_dim=0).tolist())
        return self

    def save(self, file_path: str) -> None:
        with open(file_path, "w") as handle:
            dump({"means": self.__means, "stddevs": self.__stddevs}, handle)


class DistributionCalculator:
    def __init__(self) -> None:
        self.__distribution = Counter()

    def update(self, vector: Tensor) -> DistributionCalculator:
        number, counts = unique(
            vector.data.flatten(start_dim=0), dim=0, return_counts=True
        )
        for symbol, count in zip(number.cpu(), counts.cpu()):
            self.__distribution[symbol] += count.item()
        return self

    @property
    def distribution(self) -> dict[float, int]:
        return dict(self.__distribution)

    @property
    def min_symbol(self) -> float:
        return min(self.distribution.keys())

    @property
    def max_symbol(self) -> float:
        return max(self.distribution.keys())

    def cumulative_distribution_function(self, start: int, end: int) -> Tensor:
        if (min_symbol := self.min_symbol) < start:
            raise RuntimeError(
                f"Given start of cdf ({start}) is bigger than the smallest distribution symbol ({min_symbol})"
            )
        if (max_symbol := self.max_symbol) >= end:
            raise RuntimeError(
                f"Given end of cdf ({end}) is smaller or equal than the largest distribution symbol ({max_symbol})"
            )
        cdf = [0]
        accumulator = 0
        for key in range(start, end):
            if key in self.__distribution:
                accumulator += self.__distribution[key]
            cdf.append(accumulator)
        return tensor(cdf, dtype=float32) / sum(self.__distribution.values())

    @property
    def laplace_distribution_params(self) -> tuple[float, float, int]:
        symbols = chain(
            *[[x for _ in range(count)] for x, count in self.__distribution.items()]
        )
        median_estimator = median(symbols)
        scale_factor_estimator = 0
        samples = 0
        for key, value in self.__distribution.items():
            scale_factor_estimator = abs(key - median_estimator) * value
            samples += value
        return median_estimator, scale_factor_estimator / samples, samples

    def plot_distribution(self, file_path: str, title: str) -> None:
        plt.clf()
        plt.bar(self.__distribution.keys(), self.__distribution.values(), color="g")
        plt.title(title)
        plt.savefig(file_path)
