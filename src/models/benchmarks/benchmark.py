from __future__ import annotations
from typing import TypeVar, Iterable

from gym.data_manager import DataManager
from gym.progress_bar import ProgressBar

from benchmarks.compressor import Compressor
from benchmarks.distribution import (
    DistributionCalculator,
    TensorDistributionCalculator,
)

from torch import no_grad, Size, Tensor


Model = TypeVar("Model")


class Benchmark:
    def __init__(self, model: Model, data_manager: DataManager) -> None:
        self.__data_manager = data_manager
        self.__model = model.to(data_manager.platform)
        self.__bar = ProgressBar()

    def __set_distribution(
        self, quantization_step: float, dataset_name: str, dataset: Iterable[Tensor]
    ) -> tuple[
        Size,
        Size,
        DistributionCalculator,
        DistributionCalculator,
        TensorDistributionCalculator,
    ]:
        self.__model.eval()
        latent_shape, hyperlatent_shape = None, None
        latent_distrib, hyperlatent_distrib, hyperlatent_tensor = (
            DistributionCalculator(),
            DistributionCalculator(),
            TensorDistributionCalculator(),
        )
        with self.__bar.task(
            f"{dataset_name} distribution", total=self.__data_manager.training_set_len
        ) as task:
            with no_grad():
                for batch in dataset:
                    (latent, hyperlatent, *_) = self.__model(batch, quantization_step)
                    latent_shape = list(latent.shape)
                    hyperlatent_shape = list(hyperlatent.shape)
                    latent_distrib.update(latent)
                    hyperlatent_distrib.update(hyperlatent)
                    hyperlatent_tensor.update(hyperlatent)
                    task.update(1)
        return (
            latent_shape,
            hyperlatent_shape,
            latent_distrib,
            hyperlatent_distrib,
            hyperlatent_tensor,
        )

    def train_set_distribution(
        self, quantization_step: float
    ) -> tuple[
        Size,
        Size,
        DistributionCalculator,
        DistributionCalculator,
        TensorDistributionCalculator,
    ]:
        return self.__set_distribution(
            quantization_step, "Train dataset", self.__data_manager.training_set(None)
        )

    def test_set_distribution(
        self, quantization_step: float
    ) -> tuple[
        Size,
        Size,
        DistributionCalculator,
        DistributionCalculator,
        TensorDistributionCalculator,
    ]:
        return self.__set_distribution(
            quantization_step, "Test dataset", self.__data_manager.test_set(None)
        )

    def benchmark_test_set(
        self, compressor: Compressor, quantization_step: float
    ) -> Compressor:
        self.__model.eval()
        with self.__bar.task(
            f"Benchmarking test set for quantization_step={quantization_step}",
            total=self.__data_manager.test_set_len,
        ) as task:
            with no_grad():
                for batch, origin in self.__data_manager.test_set(None):
                    (
                        latent,
                        hyperlatent,
                        latent_stddev,
                        latent_mean,
                        recon,
                    ) = self.__model(batch, quantization_step)
                    compressor.append(
                        latent, hyperlatent, latent_mean, latent_stddev, recon, origin
                    )
                    task.update(1)
        return compressor

    @property
    def progress_bar(self) -> ProgressBar:
        return self.__bar

    def __enter__(self):
        self.__bar = self.__bar.__enter__()
        return self

    def __exit__(self, *args):
        self.__bar.__exit__(*args)
