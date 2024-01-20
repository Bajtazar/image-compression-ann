from torch import (
    Tensor,
    stack,
    int16,
    add,
    linspace,
    uint8,
    float32,
    erf,
    ones,
    all as tensor_all,
)

from torchac import encode_float_cdf, decode_float_cdf

from torchvision.io import read_image, ImageReadMode

from gym.tiles import concatenate_image
from gym.config import get_config
from gym.progress_bar import ProgressBar

from benchmarks.stats import Stats

from gc import collect
from collections import defaultdict
from typing import Callable


class Compressor:
    def __init__(
        self, hyper_cdf: Tensor, min_symbol: int, max_symbol: int
    ) -> None:
        self.__network_recon = defaultdict(dict)
        self.__hyper_cdf = hyper_cdf
        self.__min_symbol = min_symbol
        self.__max_symbol = max_symbol
        assert tensor_all(self.__hyper_cdf[..., -1] == 1.0)
        assert tensor_all(self.__hyper_cdf[..., 0] == 0.0)

    def append(
        self,
        latent: Tensor,
        hyperlatent: Tensor,
        latent_mean: Tensor,
        latent_sttdev: Tensor,
        reconstruction: Tensor,
        origin: str,
    ) -> None:
        (*origin, index) = origin.split("_")
        origin = '_'.join(origin)
        index = int(index.split(".")[0])
        self.__network_recon[origin][index] = {
            "hyperlatent": hyperlatent,
            "latent": latent,
            "latent_mean": latent_mean,
            "latent_stddev": latent_sttdev,
            "reconstruction": (reconstruction.squeeze(dim=0) * 255.0).to(uint8).cpu(),
        }

    def __factorized_step(self, hyperlatents: list[Tensor]) -> bytes:
        hyperlatent_plane = stack(hyperlatents)
        hyperlatent_plane = (
            add(hyperlatent_plane, -self.__min_symbol).to(int16).to("cpu").detach()
        )
        assert hyperlatent_plane.min().item() >= 0
        assert hyperlatent_plane.max().item() <= self.__max_symbol - self.__min_symbol
        ac_cdf = self.__hyper_cdf.expand(
            [*hyperlatent_plane.shape[:2], *self.__hyper_cdf.shape]
        )
        compressed = encode_float_cdf(
            ac_cdf, hyperlatent_plane, needs_normalization=True
        )
        assert tensor_all(
            decode_float_cdf(ac_cdf, compressed, needs_normalization=True)
            == hyperlatent_plane
        )
        return len(compressed)

    def __normal_distrib_cdf(
        self, mean_: Tensor, stddev: Tensor, quantization_step: float
    ) -> Tensor:
        cdf_size = self.__max_symbol - self.__min_symbol + 1
        points = (
            linspace(self.__min_symbol, self.__max_symbol, cdf_size).to("cpu").detach()
            * quantization_step
        )
        points -= 0.5 * quantization_step
        points.expand(*mean_.shape, -1).to("cpu").detach()
        dims = list(map(lambda x: int(x.item()), ones((mean_.dim()))))
        mean_ = mean_.unsqueeze(mean_.dim()).repeat(*dims, cdf_size).to("cpu").detach()
        stddev = (
            stddev.unsqueeze(stddev.dim()).repeat(*dims, cdf_size).to("cpu").detach()
        )
        return (1.0 + erf((points - mean_) / ((2.0**0.5) * stddev))) / 2.0

    def __hyperprior_step(
        self,
        latent: list[Tensor],
        latent_mean: list[Tensor],
        latent_stddev: list[Tensor],
        quantization_step: float,
    ) -> bytes:
        latent_plate = stack(latent)
        mean_plane = stack(latent_mean)
        stddev_plane = stack(latent_stddev)
        latent_plate = add(latent_plate, -self.__min_symbol).to(int16).to("cpu").detach()
        assert latent_plate.min().item() >= 0
        assert latent_plate.max().item() <= self.__max_symbol - self.__min_symbol
        cdf = self.__normal_distrib_cdf(mean_plane, stddev_plane, quantization_step)
        cdf[..., -1] = 1.0
        encoded = encode_float_cdf(cdf, latent_plate, needs_normalization=True)
        assert (
            decode_float_cdf(cdf, encoded, needs_normalization=True) == latent_plate
        ).min()
        return len(encoded)

    def __compression_step(
        self,
        test_set_path: str,
        origin: str,
        quantization_step: float,
        hyperlatent: list[Tensor],
        latent: list[Tensor],
        latent_mean: list[Tensor],
        latent_stddev: list[Tensor],
        reconstruction: list[Tensor],
    ) -> None:
        streamsize = self.__factorized_step(hyperlatent)
        collect()
        streamsize += self.__hyperprior_step(
            latent, latent_mean, latent_stddev, quantization_step
        )
        collect()
        size = read_image(f"{test_set_path}/{origin}.png", ImageReadMode.RGB).size()
        self.__compressed[origin] = (
            streamsize,
            concatenate_image(
                reconstruction,
                size,
                int(get_config()["session"]["block_overlap_size"]),
            ).to(float32)
            / 255.0,
        )

    def compress(
        self, progress_bar: ProgressBar, test_set_path: str, quantization_step: float
    ) -> None:
        self.__compressed = {}
        with progress_bar.task(
            "Compressing data", len(self.__network_recon.items())
        ) as task:
            for origin, product in self.__network_recon.items():
                package = defaultdict(list)
                for tpl in [product[key] for key in sorted(product.keys())]:
                    for key, value in tpl.items():
                        package[key].append(value)
                self.__compression_step(
                    test_set_path, origin, quantization_step, **package
                )
                task.update(1)

    def get_stats(
        self,
        progress_bar: ProgressBar,
        stats: Stats,
        reconstruction_callaback: Callable[[str, Tensor], None],
    ) -> dict[str, dict[str, float]]:
        metrics = {"bpp": {}, "psnr": {}, "ssim": {}, "lpips": {}}
        with progress_bar.task("Calculating metrics", len(self.__compressed)) as task:
            for origin, (stream, recon) in self.__compressed.items():
                reconstruction_callaback(origin, recon)
                metrics["bpp"][origin] = stats.bits_per_pixel_len(origin, stream)
                metrics["psnr"][origin] = stats.psnr(origin, recon)
                metrics["ssim"][origin] = stats.ssim(origin, recon)
                metrics["lpips"][origin] = stats.lpips(origin, recon)
                task.update(1)
        return metrics
