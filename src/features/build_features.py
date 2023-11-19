import click

from utils import get_current_logger

from os import walk, makedirs
from os.path import exists
from itertools import chain
from random import sample
from shutil import copyfile, rmtree

from tiles import split_image

from torchvision.io import read_image, ImageReadMode, write_png

from torch import uint8, Tensor


FLICKR_DATASET_DIR: str = "flickr-image-dataset"
FLICKR_RAW_DATASET_DIR: str = f"{FLICKR_DATASET_DIR}/flickr30k_images/flickr30k_images"
LIU4K_DATASET_DIR: str = "LIU4K_v2_train"

INTERMEDIATE_FLICKR_SIZE: int = 10_000


def create_intermediate_flickr30k_set(
    raw_dataset_path: str, intermediate_dataset_path: str
) -> None:
    raw_path = f"{raw_dataset_path}/{FLICKR_RAW_DATASET_DIR}"
    files = [
        file_path
        for file_path in chain(*[file_ for *_, file_ in walk(raw_path)])
        if ".jpg" in file_path
    ]
    for file_path in sample(files, INTERMEDIATE_FLICKR_SIZE):
        copyfile(
            f"{raw_path}/{file_path}",
            f"{intermediate_dataset_path}/{FLICKR_DATASET_DIR}/{file_path}",
        )
        get_current_logger().info(f"Copying the {file_path}")


def prepate_dir(directory: str) -> None:
    if exists(directory):
        rmtree(directory)
    makedirs(directory, exist_ok=True)


def prepare_flickr30k(intermediate_dataset_path: str) -> None:
    get_current_logger().info("Destroying old flickr30k intermediate dataset")
    prepate_dir(f"{intermediate_dataset_path}/{FLICKR_DATASET_DIR}")


def save_image(image: Tensor, file_path: str) -> None:
    get_current_logger().info(f"Saving {file_path}")
    write_png(image, file_path)


def split_dataset(
    origin_path: str, target_path: str, block_size: int, block_overlap_size: int
) -> None:
    get_current_logger().info(f"Splitting {origin_path} into tiles")
    prepate_dir(target_path)
    for root, _, files in walk(origin_path):
        for file in files:
            image = read_image(f"{root}/{file}", ImageReadMode.RGB)
            splitted_image = split_image(image, block_size, block_overlap_size)
            for i, image in enumerate(splitted_image):
                file_name_pure = ".".join(file.split(".")[:-1])
                save_image(image.to(uint8), f"{target_path}/{file_name_pure}_{i}.png")
    get_current_logger().info(f"Splitting {origin_path} done")


@click.command()
@click.argument("raw_dataset_path", type=click.Path())
@click.argument("intermediate_dataset_path", type=click.Path())
@click.argument("processed_dataset_path", type=click.Path())
@click.option("--block_size", type=int, default=128)
@click.option("--block_overlap_size", type=int, default=0)
def main(
    raw_dataset_path: str,
    intermediate_dataset_path: str,
    processed_dataset_path: str,
    block_size: int,
    block_overlap_size: int,
) -> None:
    prepare_flickr30k(intermediate_dataset_path)
    create_intermediate_flickr30k_set(raw_dataset_path, intermediate_dataset_path)
    split_dataset(
        f"{intermediate_dataset_path}/{FLICKR_DATASET_DIR}",
        f"{processed_dataset_path}/{FLICKR_DATASET_DIR}",
        block_size,
        block_overlap_size,
    )


if __name__ == "__main__":
    main()
