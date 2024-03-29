from torch import Tensor, device, cuda, uint8

from torchvision.io import read_image, ImageReadMode, write_png

from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataclasses import dataclass
from typing import TypeVar, Iterable, Callable
import os

from gym.jit_image_set import JitImageSet
from gym.tiles import concatenate_image


Transform = TypeVar("Transform")


@dataclass
class DataManager:
    test_set_path: str
    raw_test_set_path: str
    train_set_path: str
    batch_size: int
    block_overlap_size: int
    train_dataset_split_coeff: float
    platform: device = device("cuda" if cuda.is_available() else "cpu")
    workers: int = 0
    distributed: bool = True

    def __post_init__(self) -> None:
        self.__load_train_dataset()
        self.__load_test_dataset()

    def __load_distributed_train_dataset(
        self, trainset: JitImageSet, valset: JitImageSet
    ) -> None:
        self.__train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        self.__train_loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.workers,
            sampler=self.__train_sampler,
        )
        self.__validation_sampler = DistributedSampler(dataset=valset, shuffle=False)
        self.__validation_loader = DataLoader(
            valset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.workers,
            sampler=self.__validation_sampler,
        )

    def __load_sequential_train_dataset(
        self, trainset: JitImageSet, valset: JitImageSet
    ) -> None:
        self.__train_loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.workers,
            shuffle=True,
        )
        self.__validation_loader = DataLoader(
            valset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.workers,
            shuffle=False,
        )

    def __load_train_dataset(self) -> None:
        dataset = JitImageSet(self.train_set_path)
        train_size = int(len(dataset) * self.train_dataset_split_coeff)
        trainset, valset = random_split(
            dataset, (train_size, len(dataset) - train_size)
        )
        if self.distributed:
            self.__load_distributed_train_dataset(trainset, valset)
        else:
            self.__load_sequential_train_dataset(trainset, valset)

    def __load_test_dataset(self) -> None:
        self.__test_dataset = JitImageSet(self.test_set_path)
        self.__test_loader = DataLoader(
            self.__test_dataset,
            batch_size=1,
            drop_last=False,
            num_workers=self.workers,
            shuffle=False,
        )

    def training_set(self, epoch: int) -> Iterable[Tensor]:
        if self.distributed:
            self.__train_sampler.set_epoch(epoch)
        for batch in self.__train_loader:
            yield batch.cuda()

    def validation_set(self, epoch: int) -> Iterable[Tensor]:
        if self.distributed:
            self.__validation_sampler.set_epoch(epoch)
        for batch in self.__validation_loader:
            yield batch.cuda()

    def test_set(self, epoch: int) -> Iterable[tuple[Tensor, str]]:
        for i, batch in enumerate(self.__test_loader):
            path, _ = self.__test_dataset.cache[i]
            yield batch.cuda(), path.split("/" if "/" in path else "\\")[-1]

    @property
    def training_set_len(self) -> int:
        return len(self.__train_loader)

    @property
    def validation_set_len(self) -> int:
        return len(self.__validation_loader)

    @property
    def test_set_len(self) -> int:
        return len(self.__test_loader)

    @property
    def real_test_set_len(self) -> int:
        return sum(len(files) for (*_, files) in os.walk(self.raw_test_set_path))

    @staticmethod
    def save_image(image: Tensor, path: str) -> None:
        (image,) = image
        write_png((image * 255.0).to(uint8).to(device("cpu")), path)

    @staticmethod
    def __piece_key(piece_name: str) -> int:
        piece = piece_name.split(".")[-2]
        return int(piece.split("_")[-1])

    def __get_ordered_pieces(self, catalog: str, file_name: str) -> list[Tensor]:
        prefix = (".".join(file_name.split(".")[:-1])).split("/")[-1]
        pieces = [piece for piece in os.listdir(catalog) if piece.startswith(prefix)]
        pieces = list(sorted(pieces, key=self.__piece_key))
        return [read_image(catalog + piece, ImageReadMode.RGB) for piece in pieces]

    def reconstruct_images(
        self, in_path: str, out_path: str, callback: Callable[[], None]
    ) -> None:
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for root, _, files in os.walk(self.raw_test_set_path):
            for file in files:
                fname = f"{root}/{file}"
                img = read_image(fname, ImageReadMode.RGB)
                pieces = self.__get_ordered_pieces(f"{in_path}/", fname)
                recon = concatenate_image(pieces, img.size(), self.block_overlap_size)
                write_png(recon.to(uint8), f"{out_path}/{file}")
                callback()
