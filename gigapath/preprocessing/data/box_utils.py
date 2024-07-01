#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#
# Original: https://github.com/microsoft/hi-ml/blob/main/hi-ml/src/health_ml/utils/box_utils.py
#  ------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Box:
    """Utility class representing rectangular regions in 2D images.

    :param x: Horizontal coordinate of the top-left corner.
    :param y: Vertical coordinate of the top-left corner.
    :param w: Box width.
    :param h: Box height.
    :raises ValueError: If either `w` or `h` are <= 0.
    """
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0:
            raise ValueError(f"Width must be strictly positive, received {self.w}")
        if self.h <= 0:
            raise ValueError(f"Height must be strictly positive, received {self.w}")

    def __add__(self, shift: Sequence[int]) -> 'Box':
        """Translates the box's location by a given shift.

        :param shift: A length-2 sequence containing horizontal and vertical shifts.
        :return: A new box with updated `x = x + shift[0]` and `y = y + shift[1]`.
        :raises ValueError: If `shift` does not have two elements.
        """
        if len(shift) != 2:
            raise ValueError("Shift must be two-dimensional")
        return Box(x=self.x + shift[0],
                   y=self.y + shift[1],
                   w=self.w,
                   h=self.h)

    def __mul__(self, factor: float) -> 'Box':
        """Scales the box by a given factor, e.g. when changing resolution.

        :param factor: The factor by which to multiply the box's location and dimensions.
        :return: The updated box, with location and dimensions rounded to `int`.
        """
        return Box(x=int(self.x * factor),
                   y=int(self.y * factor),
                   w=int(self.w * factor),
                   h=int(self.h * factor))

    def __rmul__(self, factor: float) -> 'Box':
        """Scales the box by a given factor, e.g. when changing resolution.

        :param factor: The factor by which to multiply the box's location and dimensions.
        :return: The updated box, with location and dimensions rounded to `int`.
        """
        return self * factor

    def __truediv__(self, factor: float) -> 'Box':
        """Scales the box by a given factor, e.g. when changing resolution.

        :param factor: The factor by which to divide the box's location and dimensions.
        :return: The updated box, with location and dimensions rounded to `int`.
        """
        return self * (1. / factor)

    def add_margin(self, margin: int) -> 'Box':
        """Adds a symmetric margin on all sides of the box.

        :param margin: The amount by which to enlarge the box.
        :return: A new box enlarged by `margin` on all sides.
        """
        return Box(x=self.x - margin,
                   y=self.y - margin,
                   w=self.w + 2 * margin,
                   h=self.h + 2 * margin)

    def clip(self, other: 'Box') -> Optional['Box']:
        """Clips a box to the interior of another.

        This is useful to constrain a region to the interior of an image.

        :param other: Box representing the new constraints.
        :return: A new constrained box, or `None` if the boxes do not overlap.
        """
        x0 = max(self.x, other.x)
        y0 = max(self.y, other.y)
        x1 = min(self.x + self.w, other.x + other.w)
        y1 = min(self.y + self.h, other.y + other.h)
        try:
            return Box(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
        except ValueError:  # Empty result, boxes don't overlap
            return None

    def to_slices(self) -> Tuple[slice, slice]:
        """Converts the box to slices for indexing arrays.

        For example: `my_2d_array[my_box.to_slices()]`.

        :return: A 2-tuple with vertical and horizontal slices.
        """
        return (slice(self.y, self.y + self.h),
                slice(self.x, self.x + self.w))

    @staticmethod
    def from_slices(slices: Sequence[slice]) -> 'Box':
        """Converts a pair of vertical and horizontal slices into a box.

        :param slices: A length-2 sequence containing vertical and horizontal `slice` objects.
        :return: A box with corresponding location and dimensions.
        """
        vert_slice, horz_slice = slices
        return Box(x=horz_slice.start,
                   y=vert_slice.start,
                   w=horz_slice.stop - horz_slice.start,
                   h=vert_slice.stop - vert_slice.start)


def get_bounding_box(mask: np.ndarray) -> Box:
    """Extracts a bounding box from a binary 2D array.

    :param mask: A 2D array with 0 (or `False`) as background and >0 (or `True`) as foreground.
    :return: The smallest box covering all non-zero elements of `mask`.
    :raises TypeError: When the input mask has more than two dimensions.
    :raises RuntimeError: When all elements in the mask are zero.
    """
    if mask.ndim != 2:
        raise TypeError(f"Expected a 2D array but got an array with shape {mask.shape}")

    slices = ndimage.find_objects(mask > 0)
    if not slices:
        raise RuntimeError("The input mask is empty")
    assert len(slices) == 1

    return Box.from_slices(slices[0])
