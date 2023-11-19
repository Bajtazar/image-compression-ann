# -*- coding: utf-8 -*-
import click

from pathlib import Path
from os.path import exists
from typing import Callable
from subprocess import Popen

from opendatasets import download as kaggle_download

from gdown import download_folder as gdrive_download

from utils import get_current_logger


FLICKR_DATASET_URL: str = (
    "https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset"
)
LIU4K_DATASET_URL: str = (
    "https://drive.google.com/drive/folders/1FtVQtY2t_ecuy_gzJqZ-CatqrJBAdq_d"
)
FLICKR_DATASET_DIR: str = "flickr-image-dataset"
LIU4K_DATASET_DIR: str = "LIU4K_v2_train"


def download_flickr_dataset(dataset_path: str) -> None:
    get_current_logger().info("Downloading Flickr dataset")
    kaggle_download(FLICKR_DATASET_URL, dataset_path)
    get_current_logger().info("Flickr dataset has been successfully downloaded")


def execute_bash_command(command: str) -> None:
    Popen(["/bin/bash", "-c", command]).wait()


def decompress_liu4k_dataset(dataset_path: str) -> None:
    for item in ["Animal", "Building", "Mountain", "Street"]:
        get_current_logger().info(f"Started concatenating LIU4K's {item} zip pieces")
        execute_bash_command(
            f"zip -s 0 {dataset_path}/{item}.zip -O {dataset_path}/{item}_temp.zip"
        )
        get_current_logger().info(f"Finished concatenating LIU4K's {item} zip pieces")
        execute_bash_command(f"rm {dataset_path}/{item}.*")
        get_current_logger().info(f"Removed LIU4K's {item} zip pieces")
        get_current_logger().info(f"Started uzipping LIU4K's {item}")
        execute_bash_command(f"unzip {dataset_path}/{item}_temp.zip -d {dataset_path}")
        get_current_logger().info(f"Finished uzipping LIU4K's {item}")
        execute_bash_command(f"rm {dataset_path}/{item}_temp.zip")
        get_current_logger().info(f"Removed LIU4K's {item} temp zip")


def download_liu4k_dataset(dataset_path: str) -> None:
    get_current_logger().info("Downloading LIU4K dataset")
    gdrive_download(
        LIU4K_DATASET_URL, output=dataset_path, quiet=False, use_cookies=False
    )
    get_current_logger().info("LIU4K dataset has been successfully downloaded")
    decompress_liu4k_dataset(f"{dataset_path}/{LIU4K_DATASET_DIR}")
    get_current_logger().info("LIU4K dataset has been successfully decompressed")


def resolve_raw_dataset_download(
    dataset_path: str,
    dataset_dir: str,
    dataset_name: str,
    always_download: bool,
    download_cb: Callable[[str], None],
) -> None:
    if not exists(f"{dataset_path}/{dataset_dir}"):
        if always_download:
            return download_cb(dataset_path)
        get_current_logger().info(
            f"{dataset_name} dataset has not been detected - would you like to download it automatically or rather do it manually?"
        )
        command = input(
            "[Press D to download automatically otherwise press any other key]:"
        )
        if command == "D":
            download_cb(dataset_path)
    else:
        get_current_logger().info(f"{dataset_name} dataset has been found")


@click.command()
@click.argument("raw_datasets_path", type=click.Path())
@click.option("--always_download", type=bool, default=False)
def main(raw_datasets_path: str, always_download: bool) -> None:
    resolve_raw_dataset_download(
        raw_datasets_path,
        FLICKR_DATASET_DIR,
        "Flickr30k",
        always_download,
        download_flickr_dataset,
    )
    resolve_raw_dataset_download(
        raw_datasets_path,
        LIU4K_DATASET_DIR,
        "LIU4K",
        always_download,
        download_liu4k_dataset,
    )
    get_current_logger().info("Done")


if __name__ == "__main__":
    main()
