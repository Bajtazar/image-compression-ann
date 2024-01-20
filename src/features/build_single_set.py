import click

from build_features import split_dataset

from os import makedirs


@click.command()
@click.argument("raw_dataset_path", type=click.Path())
@click.argument("processed_dataset_path", type=click.Path())
@click.option("--block_size", type=int, default=128)
@click.option("--block_overlap_size", type=int, default=0)
def main(
    raw_dataset_path: str,
    processed_dataset_path: str,
    block_size: int,
    block_overlap_size: int,
) -> None:
    makedirs(processed_dataset_path, exist_ok=True)
    split_dataset(
        raw_dataset_path,
        processed_dataset_path,
        block_size,
        block_overlap_size,
    )


if __name__ == "__main__":
    main()
