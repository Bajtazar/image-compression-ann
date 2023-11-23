from dataclasses import dataclass

from PIL import Image

from torch import Tensor, float32

from torchvision.io import read_image, ImageReadMode

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)


@dataclass
class Stats:
    test_set_path: str

    def __post_init__(self) -> None:
        self.__psnr = PeakSignalNoiseRatio()
        self.__ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.__lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def __img_path(self, origin: str) -> str:
        return f"{self.test_set_path}/dataset/{origin}.png"

    def bits_per_pixel_len(self, origin_path: str, stream_len: int) -> float:
        width, height = Image.open(self.__img_path(origin_path)).size
        return stream_len * 8 / (width * height)

    def bits_per_pixel(self, origin: str, bytestream: bytes) -> float:
        return self.bits_per_pixel_len(origin, len(bytestream))

    def __read_image(self, origin: str) -> Tensor:
        return (
            read_image(self.__img_path(origin), ImageReadMode.RGB).to(float32) / 255.0
        )

    def psnr(self, origin: str, reconstructed: Tensor) -> float:
        return self.__psnr(self.__read_image(origin), reconstructed).item()

    def ssim(self, origin: str, reconstructed: Tensor) -> float:
        return self.__ssim(
            self.__read_image(origin).unsqueeze(dim=0), reconstructed.unsqueeze(dim=0)
        ).item()

    def lpips(self, origin: str, reconstructed: Tensor) -> float:
        return self.__lpips(
            self.__read_image(origin).unsqueeze(dim=0), reconstructed.unsqueeze(dim=0)
        ).item()
