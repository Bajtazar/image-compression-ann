# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from sys import argv
from os.path import exists

from opendatasets import download

from utils import get_current_logger


FLICKR_DATASET: str = "https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset"
LIU4K_DATASET: str = (
    "https://drive.google.com/drive/folders/1FtVQtY2t_ecuy_gzJqZ-CatqrJBAdq_d"
)


def download_flickr_dataset(dataset_path: str) -> None:
    get_current_logger().info(f"Downloading {dataset_path} dataset")
    download(FLICKR_DATASET, dataset_path)
    get_current_logger().info(
        f"{dataset_path} dataset has been successfully downloaded"
    )


def resolve_raw_dataset(dataset_path: str) -> None:
    if not exists(dataset_path):
        get_current_logger().info(
            f"{dataset_path} dataset has not been detected - would you like to download it automatically or rather do it manually?"
        )
        command = input(
            "[Press D to download automatically otherwise press any other key]:"
        )
        if command == "D":
            download_flickr_dataset()
    else:
        get_current_logger().info(f"{dataset_path} dataset has been found")


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    resolve_raw_dataset(input_filepath)
    get_current_logger().info("making final data set from raw data")


if __name__ == "__main__":
    main()
