from torch import Tensor, Size, zeros, float32, div, uint8
from torch.nn.functional import pad

from math import ceil
from typing import Generator, Tuple, List


def calculate_offset_and_end(
    chunk_size: int, overlay_size: int, position: int
) -> Tuple[int, int]:
    offset = position * (chunk_size - overlay_size)
    block_end = offset + chunk_size
    return offset, block_end


def generate_tiles(
    rows: int, columns: int, chunk_size: int, overlay_size: int
) -> Generator[Tuple[int, int, int, int], None, None]:
    for y in range(rows):
        for x in range(columns):
            y_offset, y_block_end = calculate_offset_and_end(
                chunk_size, overlay_size, y
            )
            x_offset, x_block_end = calculate_offset_and_end(
                chunk_size, overlay_size, x
            )
            yield x_offset, y_offset, x_block_end, y_block_end


def calculate_padding(
    rows: int, columns: int, width: int, height: int, chunk_size: int, overlay_size: int
) -> Tuple[int, int, int, int]:
    total_width = overlay_size + (chunk_size - overlay_size) * columns
    total_height = overlay_size + (chunk_size - overlay_size) * rows
    left_padding = (total_width - width) // 2
    upper_padding = (total_height - height) // 2
    right_padding = total_width - width - left_padding
    lower_padding = total_height - height - upper_padding
    return left_padding, right_padding, upper_padding, lower_padding


def calculate_number_of_rows_and_columns(
    width: int, height: int, chunk_size: int, overlay_size: int
) -> Tuple[int, int]:
    columns = ceil((width - overlay_size) / (chunk_size - overlay_size))
    rows = ceil((height - overlay_size) / (chunk_size - overlay_size))
    return rows, columns


def split_image(image: Tensor, chunk_size: int, overlay_size: int = 0) -> List[Tensor]:
    height, width = image.size()[-2:]
    rows, columns = calculate_number_of_rows_and_columns(
        width, height, chunk_size, overlay_size
    )
    padding = calculate_padding(rows, columns, width, height, chunk_size, overlay_size)
    image = pad(image, padding, "constant", 0)
    return [
        image[..., y_offset:y_block_end, x_offset:x_block_end]
        for x_offset, y_offset, x_block_end, y_block_end in generate_tiles(
            rows, columns, chunk_size, overlay_size
        )
    ]
