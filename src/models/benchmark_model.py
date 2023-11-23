#!/usr/bin/env python3
from benchmarks.benchmark import Benchmark
from benchmarks.compressor import Compressor
from benchmarks.stats import Stats

from gym.config import get_config, current_network_path
from gym.network_loader import load_network
from gym.loss_functions import entropy_reconstruction_loss
from gym.quantization import Quantization
from gym.data_manager import DataManager

import matplotlib.pyplot as plt

from numpy import mean

from dataclasses import dataclass
from json import dump, load

from torch import Tensor


QUANTIZATION_STEPS: list[float] = [0.1, 0.2, 0.5, 1]


@dataclass
class NetworkStats:
    network_run_path: str
    min_symbol: int = -1023
    max_symbol: int = 1024
    shift: int = 1023

    def __post_init__(self) -> None:
        config = get_config()
        self.__manager = DataManager(
            config['environment']['test_set_path'],
            config['environment']['train_set_path'],
            int(config['environment']['batch_size']),
            block_overlap_size=int(config['environment']['block_overlap_size']),
            block_size=int(config['environment']['block_size']),
            workers=int(config['environment']['workers']),
            distributed=False)
        self.__autoencoder, self.__first_epoch = load_network(entropy_reconstruction_loss, Quantization(), self.__manager.platform, self.network_run_path)
        self.__stats = {}

    def __average_stats(self, quant: float) -> None:
        if 'Average stats' not in self.__stats:
            self.__stats['Average stats'] = {quant: {
                key: mean(list(self.__stats['Run stats'][quant][key].values()))
                for key in self.__stats['Run stats'][quant].keys()
            }}
        else:
            self.__stats['Average stats'][quant] = {
                key: mean(list(self.__stats['Run stats'][quant][key].values()))
                for key in self.__stats['Run stats'][quant].keys()
            }

    def __distributions(self, benchmark: Benchmark, quant: float) -> Tensor:
        cdf_distrib = benchmark.test_set_cdf_distribution(quant)
        cdf_distrib.save(f'{self.network_run_path}/cdf.json')
        self.__stats['Latent shape'], self.__stats['Hyperlatent shape'], latent_distrib, hyper_distrib = benchmark.train_set_distribution()
        self.__stats['Min latent symbol'] = latent_distrib.min_symbol
        self.__stats['Max latent symbol'] = latent_distrib.max_symbol
        self.__stats['Min hyperlatent symbol'] = hyper_distrib.min_symbol
        self.__stats['Max hyperlatent symbol'] = hyper_distrib.max_symbol
        latent_distrib.plot_distribution(f'{self.network_run_path}/latent.png', 'Latent symbol distribution')
        hyper_distrib.plot_distribution(f'{self.network_run_path}/hyperlatent.png', 'Hyperlatent symbol distribution')
        return hyper_distrib.cumulative_distribution_function(self.min_symbol, self.max_symbol)

    def benchmark(self, coeff: float) -> None:
        with Benchmark(self.__autoencoder, self.__manager) as benchmark:
            hyper_cdf = self.__distributions(benchmark, coeff)
            compressor = Compressor(hyper_cdf, shift=self.shift, min_symbol=self.min_symbol, max_symbol=self.max_symbol)
            compressor = benchmark.benchmark_test_set(compressor, coeff)
            test_path = get_config()['environment']['test_set_path']
            compressor.compress(benchmark.progress_bar, f"{test_path}/dataset", coeff)
            if 'Run stats' not in self.__stats:
                self.__stats['Run stats'] = {
                    coeff: compressor.get_stats(benchmark.progress_bar, Stats(test_path))
                }
            else:
                self.__stats['Run stats'][coeff] = compressor.get_stats(benchmark.progress_bar, Stats(test_path))
            self.__average_stats(coeff)

    def log_stats(self) -> None:
        with open(f'{self.network_run_path}/stats.json', 'w') as handle:
            dump(self.__stats, handle, indent=4)

    def __plot_stat(self, stat_name: str, format_stats: dict) -> None:
        plt.clf()
        keys = self.__stats['Average stats'].keys()
        bpp = [self.__stats['Average stats'][q]['bpp'] for q in keys]
        stat = [self.__stats['Average stats'][q][stat_name] for q in keys]
        plt.plot(bpp, stat, '-o', label=self.network_run_path)
        for format_type, values in format_stats.items():
            plt.plot(values['bpp'], values[stat_name], label=format_type)
        plt.legend()
        plt.title(f'{stat_name} -> {self.network_run_path}')
        plt.xlabel('bpp')
        plt.ylabel(stat_name)
        plt.savefig(f'{self.network_run_path}/{stat_name}.png')

    def plot_average_stats(self) -> None:
        with open('format_stats.json', 'r') as handle:
            format_stats = load(handle)
        self.__plot_stat('psnr', format_stats)
        self.__plot_stat('ssim', format_stats)
        self.__plot_stat('lpips', format_stats)


def get_stats_of_network(network_run_path: str) -> None:
    net_stats = NetworkStats(network_run_path)
    for q in QUANTIZATION_STEPS:
        net_stats.benchmark(q)
    net_stats.log_stats()
    net_stats.plot_average_stats()


def main():
    path = current_network_path()
    print(f'Current model: {path}')
    get_stats_of_network(path)


if __name__ == '__main__':
    main()
