from torch.utils.data import Dataset
from torch import Tensor, uint8, frombuffer, float32

from torchvision.io import decode_image, ImageReadMode

from gym.progress_bar import ProgressBar

from typing import Optional, Callable
import os

import numpy as np


class JitImageSet(Dataset):
    def __init__(self, dir_path: str) -> None:
        super().__init__()
        self.cache = []
        self.__load_paths(dir_path)
        with ProgressBar() as bar:
            with bar.task(f'"{dir_path}" set loading', len(self.__paths)) as task:
                self.__load_images_to_cache(task)

    def __load_paths(self, dir_path: str) -> None:
        self.__paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                self.__paths.append(f"{root}/{file}")

    def __load_images_to_cache(self, task: ProgressBar.RealTask) -> None:
        for path in self.__paths:
            with open(path, "rb") as handle:
                self.cache.append(
                    (path, frombuffer(np.array(handle.read()), dtype=uint8))
                )
            task.update(1)

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, index: int) -> Tensor:
        _, buffer = self.cache[index]
        return decode_image(buffer, ImageReadMode.RGB).to(float32) / 255
