from torch import Tensor, Size, zeros, float32, div, uint8

from math import ceil
from typing import Generator, Tuple


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


def tile_sequencer(
    chunks: list[Tensor],
    chunk_size: int,
    columns: int,
    padding: Tuple[int, int, int, int],
    overlay_size: int,
) -> Generator[Tuple[Tensor, int, int], None, None]:
    row_counter = 0
    col_couter = 0
    left_padding, _, upper_padding, _ = padding
    x_start = -left_padding
    y_start = -upper_padding
    for chunk in chunks:
        yield chunk.to(float32), x_start, y_start
        col_couter += 1
        x_start += chunk_size - overlay_size
        if col_couter == columns:
            row_counter += 1
            col_couter = 0
            x_start = -left_padding
            y_start += chunk_size - overlay_size


def caclulate_weight(chunk_size: int, overlay_size: int, position: Tuple[int]) -> int:
    x, y = position
    pivot = chunk_size - overlay_size
    weight = 0
    if x < overlay_size:
        weight += 2**x
    elif x >= pivot:
        weight += 2 ** (x - pivot)
    if y < overlay_size:
        weight += 2**y
    elif y >= pivot:
        weight += 2 ** (y - pivot)
    return weight if weight else 1


def apply_tiles(
    image: Tensor,
    weights: Tensor,
    size: Tuple[int, int],
    chunks: list[Tensor],
    chunk_size: int,
    columns: int,
    padding: Tuple[int, int, int, int],
    overlay_size: int,
) -> None:
    height, width = size
    for tile, x_start, y_start in tile_sequencer(
        chunks, chunk_size, columns, padding, overlay_size
    ):
        for x in range(x_start, x_start + chunk_size):
            for y in range(y_start, y_start + chunk_size):
                if 0 <= x < width and 0 <= y < height:
                    r_x = x - x_start
                    r_y = y - y_start
                    weight = caclulate_weight(chunk_size, overlay_size, (r_x, r_y))
                    image[..., y, x] += tile[..., r_y, r_x] * weight
                    weights[y, x] += weight


def concatenate_image(
    chunks: list[Tensor], original_size: Size, overlay_size: int = 0
) -> Tensor:
    chunk_size = chunks[0].size()[-1]
    height, width = original_size[-2:]
    rows, columns = calculate_number_of_rows_and_columns(
        width, height, chunk_size, overlay_size
    )
    padding = calculate_padding(rows, columns, width, height, chunk_size, overlay_size)
    image = zeros(original_size, dtype=float32)
    weight = zeros((height, width), dtype=float32)
    apply_tiles(
        image,
        weight,
        (height, width),
        chunks,
        chunk_size,
        columns,
        padding,
        overlay_size,
    )
    return div(image, weight).to(uint8)
