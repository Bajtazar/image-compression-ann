import click

from utils import get_current_logger

from os import walk, makedirs
from os.path import exists
from itertools import chain
from random import sample
from shutil import copyfile, rmtree


FLICKR_DATASET_DIR: str = "flickr-image-dataset"
FLICKR_RAW_DATASET_DIR: str = f"{FLICKR_DATASET_DIR}/flickr30k_images/flickr30k_images"
LIU4K_DATASET_DIR: str = "LIU4K_v2_train"

INTERMEDIATE_FLICKR_SIZE: int = 10_000


def create_intermediate_flickr30k_set(raw_dataset_path: str, intermediate_dataset_path: str) -> None:
    raw_path = f'{raw_dataset_path}/{FLICKR_RAW_DATASET_DIR}'
    files = [file_path for file_path in chain(*[file_ for *_, file_ in walk(raw_path)]) if ".jpg" in file_path]
    for file_path in sample(files, INTERMEDIATE_FLICKR_SIZE):
        copyfile(f'{raw_path}/{file_path}', f'{intermediate_dataset_path}/{FLICKR_DATASET_DIR}/{file_path}')
        get_current_logger().info(f"Copying the {file_path}")


def prepate_dir(directory: str) -> None:
    if exists(directory):
        rmtree(directory)
    makedirs(directory, exist_ok=True)


def prepare_flickr30k(intermediate_dataset_path: str) -> None:
    get_current_logger().info("Destroying old flickr30k intermediate dataset")
    prepate_dir(f'{intermediate_dataset_path}/{FLICKR_DATASET_DIR}')


@click.command()
@click.argument("raw_dataset_path", type=click.Path())
@click.argument("intermediate_dataset_path", type=click.Path())
@click.argument("processed_dataset_path", type=click.Path())
def main(raw_dataset_path: str, intermediate_dataset_path: str, processed_dataset_path: str) -> None:
    prepare_flickr30k(intermediate_dataset_path)
    create_intermediate_flickr30k_set(raw_dataset_path, intermediate_dataset_path)


if __name__ == '__main__':
    main()
