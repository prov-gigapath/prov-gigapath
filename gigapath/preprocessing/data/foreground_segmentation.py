#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#
# Original: https://github.com/microsoft/hi-ml/blob/main/hi-ml-cpath/src/health_cpath/preprocessing/loading.py
#  ------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import PIL
import skimage.filters
from monai.config.type_definitions import KeysCollection
from monai.data.wsi_reader import WSIReader
from monai.transforms.transform import MapTransform
from openslide import OpenSlide

from gigapath.preprocessing.data import box_utils


def get_luminance(slide: np.ndarray) -> np.ndarray:
    """Compute a grayscale version of the input slide.

    :param slide: The RGB image array in (*, C, H, W) format.
    :return: The single-channel luminance array as (*, H, W).
    """
    # TODO: Consider more sophisticated luminance calculation if necessary
    return slide.mean(axis=-3, dtype=np.float16)  # type: ignore


def segment_foreground(slide: np.ndarray, threshold: Optional[float] = None) \
        -> Tuple[np.ndarray, float]:
    """Segment the given slide by thresholding its luminance.

    :param slide: The RGB image array in (*, C, H, W) format.
    :param threshold: Pixels with luminance below this value will be considered foreground.
    If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.
    :return: A tuple containing the boolean output array in (*, H, W) format and the threshold used.
    """
    luminance = get_luminance(slide)
    if threshold is None:
        threshold = skimage.filters.threshold_otsu(luminance)
    logging.info(f"Otsu threshold from luminance: {threshold}")
    return luminance < threshold, threshold


# MONAI's convention is that dictionary transforms have a 'd' suffix in the class name
class ReadImaged(MapTransform):
    """Basic transform to read image files."""

    def __init__(self, reader: WSIReader, keys: KeysCollection,
                 allow_missing_keys: bool = False, **kwargs: Any) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.reader = reader
        self.kwargs = kwargs

    def __call__(self, data: Dict) -> Dict:
        for key in self.keys:
            if key in data or not self.allow_missing_keys:
                data[key] = self.reader.read(data[key], **self.kwargs)
        return data


# Temporary workaround for MONAI bug (https://github.com/Project-MONAI/MONAI/pull/3417/files)
def _get_image_size(img, size=None, level=None, location=(0, 0), backend="openslide"):
    max_size = []
    downsampling_factor = []
    if backend == "openslide":
        downsampling_factor = img.level_downsamples[level]
        max_size = img.level_dimensions[level][::-1]
    elif backend == "cucim":
        downsampling_factor = img.resolutions["level_downsamples"][level]
        max_size = img.resolutions["level_dimensions"][level][::-1]
    elif backend == "tifffile":
        level0_size = img.pages[0].shape[:2]
        max_size = img.pages[level].shape[:2]
        downsampling_factor = np.mean([level0_size[i] / max_size[i] for i in range(len(max_size))])

    # subtract the top left corner of the patch from maximum size
    level_location = [round(location[i] / downsampling_factor) for i in range(len(location))]
    size = [max_size[i] - level_location[i] for i in range(len(max_size))]

    return size


def load_slide_at_level(reader: WSIReader, slide_obj: OpenSlide, level: int) -> np.ndarray:
    """Load full slide array at the given magnification level.

    This is a manual workaround for a MONAI bug (https://github.com/Project-MONAI/MONAI/issues/3415)
    fixed in a currently unreleased PR (https://github.com/Project-MONAI/MONAI/pull/3417).

    :param reader: A MONAI `WSIReader` using OpenSlide backend.
    :param slide_obj: The OpenSlide image object returned by `reader.read(<image_file>)`.
    :param level: Index of the desired magnification level as defined in the `slide_obj` headers.
    :return: The loaded image array in (C, H, W) format.
    """
    size = _get_image_size(slide_obj, level=level)
    img_data, meta_data = reader.get_data(slide_obj, size=size, level=level)
    logging.info(f"img: {img_data.dtype} {img_data.shape}, metadata: {meta_data}")

    return img_data

def save_image(array_chw: np.ndarray, path: Path) -> PIL.Image:
    """Save an image array in (C, H, W) format to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image

class LoadROId(MapTransform):
    """Transform that loads a pathology slide, cropped to the foreground bounding box (ROI).

    Operates on dictionaries, replacing the file paths in `image_key` with the
    respective loaded arrays, in (C, H, W) format. Also adds the following meta-data entries:
    - `'location'` (tuple): top-right coordinates of the bounding box
    - `'size'` (tuple): width and height of the bounding box
    - `'level'` (int): chosen magnification level
    - `'scale'` (float): corresponding scale, loaded from the file
    """

    def __init__(self, image_reader: WSIReader, image_key: str = "image", level: int = 0,
                 margin: int = 0, foreground_threshold: Optional[float] = None) -> None:
        """
        :param reader: An instance of MONAI's `WSIReader`.
        :param image_key: Image key in the input and output dictionaries.
        :param level: Magnification level to load from the raw multi-scale files.
        :param margin: Amount in pixels by which to enlarge the estimated bounding box for cropping.
        """
        super().__init__([image_key], allow_missing_keys=False)
        self.image_reader = image_reader
        self.image_key = image_key
        self.level = level
        self.margin = margin
        self.foreground_threshold = foreground_threshold

    def _get_bounding_box(self, slide_obj: OpenSlide) -> box_utils.Box:
        # Estimate bounding box at the lowest resolution (i.e. highest level)
        highest_level = slide_obj.level_count - 1
        slide = load_slide_at_level(self.image_reader, slide_obj, level=highest_level)

        if slide_obj.level_count == 1:
            logging.warning(f"Only one image level found. segment_foregound will use a lot of memory.")

        foreground_mask, threshold = segment_foreground(slide, self.foreground_threshold)

        scale = slide_obj.level_downsamples[highest_level]
        bbox = scale * box_utils.get_bounding_box(foreground_mask).add_margin(self.margin)
        return bbox, threshold

    def __call__(self, data: Dict) -> Dict:
        logging.info(f"LoadROId: read {data[self.image_key]}")
        image_obj: OpenSlide = self.image_reader.read(data[self.image_key])

        logging.info("LoadROId: get bbox")
        level0_bbox, threshold = self._get_bounding_box(image_obj)
        logging.info(f"LoadROId: level0_bbox: {level0_bbox}")

        # OpenSlide takes absolute location coordinates in the level 0 reference frame,
        # but relative region size in pixels at the chosen level
        scale = image_obj.level_downsamples[self.level]
        scaled_bbox = level0_bbox / scale
        # Monai image_reader.get_data old bug: order of location/size arguments is reversed
        origin = (level0_bbox.y, level0_bbox.x)
        get_data_kwargs = dict(location=origin,
                               size=(scaled_bbox.h, scaled_bbox.w),
                               level=self.level)

        img_data, _ = self.image_reader.get_data(image_obj, **get_data_kwargs)  # type: ignore
        logging.info(f"img_data: {img_data.dtype} {img_data.shape}")
        data[self.image_key] = img_data
        data.update(get_data_kwargs)
        data["origin"] = origin
        data["scale"] = scale
        data["foreground_threshold"] = threshold

        image_obj.close()
        return data
