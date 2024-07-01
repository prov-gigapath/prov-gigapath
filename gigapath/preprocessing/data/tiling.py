#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# These tiling implementations are adapted from PANDA Kaggle solutions, for example:
# https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution/blob/master/src/data_process/a00_save_tiles.py
#
# Original: https://github.com/microsoft/hi-ml/blob/main/hi-ml-cpath/src/health_cpath/preprocessing/tiling.py
from typing import Any, Optional, Tuple

import numpy as np


def get_1d_padding(length: int, tile_size: int) -> Tuple[int, int]:
    """Computes symmetric padding for `length` to be divisible by `tile_size`."""
    pad = (tile_size - length % tile_size) % tile_size
    return (pad // 2, pad - pad // 2)


def pad_for_tiling_2d(array: np.ndarray, tile_size: int, channels_first: Optional[bool] = True,
                      **pad_kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetrically pads a 2D `array` such that both dimensions are divisible by `tile_size`.

    :param array: 2D image array.
    :param tile_size: Width/height of each tile in pixels.
    :param channels_first: Whether `array` is in CHW (`True`, default) or HWC (`False`) layout.
    :param pad_kwargs: Keyword arguments to be passed to `np.pad()` (e.g. `constant_values=0`).
    :return: A tuple containing:
        - `padded_array`: Resulting array, in the same CHW/HWC layout as the input.
        - `offset`: XY offset introduced by the padding. Add this to coordinates relative to the
        original array to obtain indices for the padded array.
    """
    height, width = array.shape[1:] if channels_first else array.shape[:-1]
    padding_h = get_1d_padding(height, tile_size)
    padding_w = get_1d_padding(width, tile_size)
    padding = [padding_h, padding_w]
    channels_axis = 0 if channels_first else 2
    padding.insert(channels_axis, (0, 0))  # zero padding on channels axis
    padded_array = np.pad(array, padding, **pad_kwargs)
    offset = (padding_w[0], padding_h[0])
    return padded_array, np.array(offset)


def tile_array_2d(array: np.ndarray, tile_size: int, channels_first: Optional[bool] = True,
                  **pad_kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Split an image array into square non-overlapping tiles.

    The array will be padded symmetrically if its dimensions are not exact multiples of `tile_size`.

    :param array: Image array.
    :param tile_size: Width/height of each tile in pixels.
    :param pad_kwargs: Keyword arguments to be passed to `np.pad()` (e.g. `constant_values=0`).
    :param channels_first: Whether `array` is in CHW (`True`, default) or HWC (`False`) layout.
    :return: A tuple containing:
        - `tiles`: A batch of tiles in NCHW layout.
        - `coords`: XY coordinates of each tile, in the same order.
    """
    padded_array, (offset_w, offset_h) = pad_for_tiling_2d(array, tile_size, channels_first, **pad_kwargs)
    if channels_first:
        channels, height, width = padded_array.shape
    else:
        height, width, channels = padded_array.shape
    n_tiles_h = height // tile_size
    n_tiles_w = width // tile_size

    if channels_first:
        intermediate_shape = (channels, n_tiles_h, tile_size, n_tiles_w, tile_size)
        axis_order = (1, 3, 0, 2, 4)  # (n_tiles_h, n_tiles_w, channels, tile_size, tile_size)
        output_shape = (n_tiles_h * n_tiles_w, channels, tile_size, tile_size)
    else:
        intermediate_shape = (n_tiles_h, tile_size, n_tiles_w, tile_size, channels)
        axis_order = (0, 2, 1, 3, 4)  # (n_tiles_h, n_tiles_w, tile_size, tile_size, channels)
        output_shape = (n_tiles_h * n_tiles_w, tile_size, tile_size, channels)

    tiles = padded_array.reshape(intermediate_shape)  # Split width and height axes
    tiles = tiles.transpose(axis_order)
    tiles = tiles.reshape(output_shape)  # Flatten tile batch dimension

    # Compute top-left coordinates of every tile, relative to the original array's origin
    coords_h = tile_size * np.arange(n_tiles_h) - offset_h
    coords_w = tile_size * np.arange(n_tiles_w) - offset_w
    # Shape: (n_tiles_h * n_tiles_w, 2)
    coords = np.stack(np.meshgrid(coords_w, coords_h), axis=-1).reshape(-1, 2)

    return tiles, coords


def assemble_tiles_2d(tiles: np.ndarray, coords: np.ndarray, fill_value: Optional[float] = np.nan,
                      channels_first: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
    """Assembles a 2D array from sequences of tiles and coordinates.

    :param tiles: Stack of tiles with batch dimension first.
    :param coords: XY tile coordinates, assumed to be spaced by multiples of `tile_size` (shape: [N, 2]).
    :param tile_size: Size of each tile; must be >0.
    :param fill_value: Value to assign to empty elements (default: `NaN`).
    :param channels_first: Whether each tile is in CHW (`True`, default) or HWC (`False`) layout.
    :return: A tuple containing:
        - `array`: The reassembled 2D array with the smallest dimensions to contain all given tiles.
        - `offset`: The lowest XY coordinates.
        - `offset`: XY offset introduced by the assembly. Add this to tile coordinates to obtain
        indices for the assembled array.
    """
    if coords.shape[0] != tiles.shape[0]:
        raise ValueError(f"Tile coordinates and values must have the same length, "
                         f"got {coords.shape[0]} and {tiles.shape[0]}")

    if channels_first:
        n_tiles, channels, tile_size, _ = tiles.shape
    else:
        n_tiles, tile_size, _, channels = tiles.shape
    tile_xs, tile_ys = coords.T

    x_min, x_max = min(tile_xs), max(tile_xs + tile_size)
    y_min, y_max = min(tile_ys), max(tile_ys + tile_size)
    width = x_max - x_min
    height = y_max - y_min
    output_shape = (channels, height, width) if channels_first else (height, width, channels)
    array = np.full(output_shape, fill_value)

    offset = np.array([-x_min, -y_min])
    for idx in range(n_tiles):
        row = coords[idx, 1] + offset[1]
        col = coords[idx, 0] + offset[0]
        if channels_first:
            array[:, row:row + tile_size, col:col + tile_size] = tiles[idx]
        else:
            array[row:row + tile_size, col:col + tile_size, :] = tiles[idx]

    return array, offset
