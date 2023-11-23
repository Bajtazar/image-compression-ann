#!/usr/bin/env python3
from benchmarks.benchmark import Benchmark
from benchmarks.compressor import Compressor
from benchmarks.stats import Stats

from gym.config import get_config, current_network_path
from gym.network_loader import load_network
from gym.quantization import Quantization
from gym.data_manager import DataManager
from gym.loss_function import NormalDistributionRateDistortionLoss

import matplotlib.pyplot as plt

from numpy import mean

from collections import defaultdict
from dataclasses import dataclass
from json import dump, load
from os import makedirs

from torch import Tensor, uint8

from torchvision.io import write_png


QUANTIZATION_STEPS: list[float] = [1, 2, 5, 10]


@dataclass
class NetworkStats:
    network_run_path: str
    min_symbol: int = -1023
    max_symbol: int = 1024
    shift: int = 1023

    def __post_init__(self) -> None:
        config = get_config()
        self.__manager = DataManager(
            test_set_path=config["datasets"]["test"],
            raw_test_set_path=config["datasets"]["raw_test"],
            train_set_path=config["datasets"]["train"],
            batch_size=int(config["environment"]["batch_size"]),
            block_overlap_size=int(config["session"]["block_overlap_size"]),
            train_dataset_split_coeff=float(
                config["environment"]["training_validation_set_split_coeff"]
            ),
            workers=int(config["environment"]["workers"]),
            distributed=False,
        )
        self.__autoencoder, self.__first_epoch = load_network(
            NormalDistributionRateDistortionLoss,
            Quantization(),
            self.__manager.platform,
            self.network_run_path,
        )
        self.__stats = defaultdict(lambda: defaultdict(dict))
        self.__target_path = f"{self.network_run_path}/benchmarks"
        makedirs(self.__target_path, exist_ok=True)

    def __average_stats(self, quantization_step: float) -> None:
        self.__stats["Average stats"][quantization_step] = {
            key: mean(list(self.__stats["Run stats"][quantization_step][key].values()))
            for key in self.__stats["Run stats"][quantization_step].keys()
        }

    def __quant_path(self, quantization_step: float) -> str:
        path = f"{self.__target_path}/quantization_step_{quantization_step}"
        makedirs(path, exist_ok=True)
        return path

    def __distributions(self, benchmark: Benchmark, quantization_step: float) -> Tensor:
        (
            self.__stats["Latent shape"],
            self.__stats["Hyperlatent shape"],
            latent_distrib,
            hyper_distrib,
        ) = benchmark.train_set_distribution(quantization_step)
        self.__stats["Min latent symbol"] = latent_distrib.min_symbol
        self.__stats["Max latent symbol"] = latent_distrib.max_symbol
        self.__stats["Min hyperlatent symbol"] = hyper_distrib.min_symbol
        self.__stats["Max hyperlatent symbol"] = hyper_distrib.max_symbol
        latent_distrib.plot_distribution(
            f"{self.__quant_path(quantization_step)}/latent.png",
            "Latent symbol distribution",
        )
        hyper_distrib.plot_distribution(
            f"{self.__quant_path(quantization_step)}/hyperlatent.png",
            "Hyperlatent symbol distribution",
        )
        return hyper_distrib.cumulative_distribution_function(
            self.min_symbol, self.max_symbol
        )

    def __generate_reconstruction_cb(self, quantization_step: float) -> None:
        path = f"{self.__quant_path(quantization_step)}/recons"
        makedirs(path, exist_ok=True)

        def callback(origin: str, reconstruction: Tensor) -> None:
            write_png(
                (reconstruction * 255.0).to(uint8).to("cpu"), f"{path}/{origin}.png"
            )

        return callback

    def benchmark(self, quantization_step: float) -> None:
        with Benchmark(self.__autoencoder, self.__manager) as benchmark:
            hyper_cdf = self.__distributions(benchmark, quantization_step)
            compressor = Compressor(
                hyper_cdf,
                shift=self.shift,
                min_symbol=self.min_symbol,
                max_symbol=self.max_symbol,
            )
            compressor = benchmark.benchmark_test_set(compressor, quantization_step)
            test_path = get_config()["datasets"]["raw_test"]
            compressor.compress(benchmark.progress_bar, test_path, quantization_step)
            self.__stats["Run stats"][quantization_step] = compressor.get_stats(
                benchmark.progress_bar,
                Stats(test_path),
                self.__generate_reconstruction_cb(quantization_step),
            )
            self.__average_stats(quantization_step)

    def log_stats(self) -> None:
        with open(f"{self.__target_path}/stats.json", "w") as handle:
            dump(self.__stats, handle, indent=4)

    def __plot_stat(self, stat_name: str, format_stats: dict) -> None:
        plt.clf()
        keys = self.__stats["Average stats"].keys()
        bpp = [self.__stats["Average stats"][q]["bpp"] for q in keys]
        stat = [self.__stats["Average stats"][q][stat_name] for q in keys]
        plt.plot(bpp, stat, "-o", label=self.network_run_path)
        for format_type, values in format_stats.items():
            plt.plot(values["bpp"], values[stat_name], label=format_type)
        plt.legend()
        plt.title(f"{stat_name} -> {self.network_run_path}")
        plt.xlabel("bpp")
        plt.ylabel(stat_name)
        plt.savefig(f"{self.__target_path}/{stat_name}.png")

    def plot_average_stats(self) -> None:
        with open("format_stats.json", "r") as handle:
            format_stats = load(handle)
        self.__plot_stat("psnr", format_stats)
        self.__plot_stat("ssim", format_stats)
        self.__plot_stat("lpips", format_stats)


def get_stats_of_network(network_run_path: str) -> None:
    net_stats = NetworkStats(network_run_path)
    for quantization_step in QUANTIZATION_STEPS:
        net_stats.benchmark(quantization_step)
    net_stats.log_stats()
    net_stats.plot_average_stats()


def main():
    path = current_network_path()
    print(f"Current model: {path}")
    get_stats_of_network(path)


if __name__ == "__main__":
    main()
