#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#
# Original: https://github.com/microsoft/hi-ml/blob/main/hi-ml-cpath/src/health_cpath/preprocessing/create_tiles_dataset.py
#  ------------------------------------------------------------------------------------------

import functools
import logging
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import PIL
from matplotlib import collections, patches, pyplot as plt
from monai.data import Dataset
from monai.data.wsi_reader import WSIReader
from openslide import OpenSlide
from tqdm import tqdm

from gigapath.preprocessing.data import tiling
from gigapath.preprocessing.data.foreground_segmentation import LoadROId, segment_foreground


def select_tiles(foreground_mask: np.ndarray, occupancy_threshold: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Exclude tiles that are mostly background based on estimated occupancy.

    :param foreground_mask: Boolean array of shape (*, H, W).
    :param occupancy_threshold: Tiles with lower occupancy (between 0 and 1) will be discarded.
    :return: A tuple containing which tiles were selected and the estimated occupancies. These will
    be boolean and float arrays of shape (*,), or scalars if `foreground_mask` is a single tile.
    """
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")
    occupancy = foreground_mask.mean(axis=(-2, -1), dtype=np.float16)
    return (occupancy > occupancy_threshold).squeeze(), occupancy.squeeze()  # type: ignore


def get_tile_descriptor(tile_location: Sequence[int]) -> str:
    """Format the XY tile coordinates into a tile descriptor."""
    return f"{tile_location[0]:05d}x_{tile_location[1]:05d}y"


def get_tile_id(slide_id: str, tile_location: Sequence[int]) -> str:
    """Format the slide ID and XY tile coordinates into a unique tile ID."""
    return f"{slide_id}.{get_tile_descriptor(tile_location)}"


def save_image(array_chw: np.ndarray, path: Path) -> PIL.Image:
    """Save an image array in (C, H, W) format to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image


def check_empty_tiles(tiles: np.ndarray, std_th: int = 5, extreme_value_portion_th: float = 0.5) -> np.ndarray:
    """Determine if a tile is empty. Hacky.

    :param tiles: The tile array in (N, C, H, W) format.
    :return: Boolean array of shape (N,).
    """
    # calculate standard deviation of rgb image
    b, c, h, w = tiles.shape
    flattned_tiles = tiles.reshape(b, c, h * w)

    std_rgb = flattned_tiles[:, :, :].std(axis=2)
    std_rgb_mean = std_rgb.mean(axis=1)

    low_std_mask = std_rgb_mean < std_th

    # count 0 pixel values
    extreme_value_count = ((flattned_tiles == 0)).sum(axis=2)
    extreme_value_proportion = extreme_value_count / (h * w)
    extreme_value_mask = extreme_value_proportion.max(axis=1) > extreme_value_portion_th

    return low_std_mask | extreme_value_mask


def generate_tiles(slide_image: np.ndarray, tile_size: int, foreground_threshold: float,
                   occupancy_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Split the foreground of an input slide image into tiles.

    :param slide_image: The RGB image array in (C, H, W) format.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :return: A tuple containing the image tiles (N, C, H, W), tile coordinates (N, 2), occupancies
    (N,), and total number of discarded empty tiles.
    """
    image_tiles, tile_locations = tiling.tile_array_2d(slide_image, tile_size=tile_size,
                                                       constant_values=255)
    logging.info(f"image_tiles.shape: {image_tiles.shape}, dtype: {image_tiles.dtype}")
    logging.info(f"Tiled {slide_image.shape} to {image_tiles.shape}")
    foreground_mask, _ = segment_foreground(image_tiles, foreground_threshold)
    selected, occupancies = select_tiles(foreground_mask, occupancy_threshold)
    n_discarded = (~selected).sum()
    logging.info(f"Percentage tiles discarded: {n_discarded / len(selected) * 100:.2f}")

    # FIXME: this uses too much memory
    # empty_tile_bool_mask = check_empty_tiles(image_tiles)
    # selected = selected & (~empty_tile_bool_mask)
    # n_discarded = (~selected).sum()
    # logging.info(f"Percentage tiles discarded after filtering empty tiles: {n_discarded / len(selected) * 100:.2f}")

    # logging.info(f"Before filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")

    image_tiles = image_tiles[selected]
    tile_locations = tile_locations[selected]
    occupancies = occupancies[selected]

    if len(tile_locations) == 0:
        logging.warn("No tiles selected")
    else:
        logging.info(f"After filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")

    return image_tiles, tile_locations, occupancies, n_discarded


def get_tile_info(sample: Dict["SlideKey", Any], occupancy: float, tile_location: Sequence[int],
                  rel_slide_dir: Path) -> Dict["TileKey", Any]:
    """Map slide information and tiling outputs into tile-specific information dictionary.

    :param sample: Slide dictionary.
    :param occupancy: Estimated tile foreground occuppancy.
    :param tile_location: Tile XY coordinates.
    :param rel_slide_dir: Directory where tiles are saved, relative to dataset root.
    :return: Tile information dictionary.
    """
    slide_id = sample["slide_id"]
    descriptor = get_tile_descriptor(tile_location)
    rel_image_path = f"{rel_slide_dir}/{descriptor}.png"

    tile_info = {
        "slide_id": slide_id,
        "tile_id": get_tile_id(slide_id, tile_location),
        "image": rel_image_path,
        "label": sample.get("label", None),
        "tile_x": tile_location[0],
        "tile_y": tile_location[1],
        "occupancy": occupancy,
        "metadata": {"slide_" + key: value for key, value in sample["metadata"].items()}
    }

    return tile_info


def format_csv_row(tile_info: Dict["TileKey", Any], keys_to_save: Iterable["TileKey"],
                   metadata_keys: Iterable[str]) -> str:
    """Format tile information dictionary as a row to write to a dataset CSV tile.

    :param tile_info: Tile information dictionary.
    :param keys_to_save: Which main keys to include in the row, and in which order.
    :param metadata_keys: Likewise for metadata keys.
    :return: The formatted CSV row.
    """
    tile_slide_metadata = tile_info.pop("metadata")
    fields = [str(tile_info[key]) for key in keys_to_save]
    fields.extend(str(tile_slide_metadata[key]) for key in metadata_keys)
    dataset_row = ','.join(fields)
    return dataset_row


def load_image_dict(sample: dict, level: int, margin: int, foreground_threshold: Optional[float] = None) -> Dict["SlideKey", Any]:
    """
    Load image from metadata dictionary
    :param sample: dict describing image metadata. Example:
        {'image_id': ['1ca999adbbc948e69783686e5b5414e4'],
        'image': ['/tmp/datasets/PANDA/train_images/1ca999adbbc948e69783686e5b5414e4.tiff'],
         'mask': ['/tmp/datasets/PANDA/train_label_masks/1ca999adbbc948e69783686e5b5414e4_mask.tiff'],
         'data_provider': ['karolinska'],
         'isup_grade': tensor([0]),
         'gleason_score': ['0+0']}
    :param level: level of resolution to be loaded
    :param margin: margin to be included
    :return: a dict containing the image data and metadata
    """
    loader = LoadROId(WSIReader(backend="OpenSlide"), level=level, margin=margin,
                      foreground_threshold=foreground_threshold)
    img = loader(sample)

    return img


def save_thumbnail(slide_path, output_path, size_target=1024):
    with OpenSlide(str(slide_path)) as openslide_obj:
        scale = size_target / max(openslide_obj.dimensions)
        thumbnail = openslide_obj.get_thumbnail([int(m * scale) for m in openslide_obj.dimensions])
        thumbnail.save(output_path)
        logging.info(f"Saving thumbnail {output_path}, shape {thumbnail.size}")


def visualize_tile_locations(slide_sample, output_path, tile_info_list, tile_size, origin_offset):
    # check slide_image size. should be thumbnail size?
    slide_image = slide_sample["image"]
    downscale_factor = slide_sample["scale"]

    fig, ax = plt.subplots()
    ax.imshow(slide_image.transpose(1, 2, 0))
    rects = []
    for tile_info in tile_info_list:
        # change coordinate to the current level from level-0
        # tile location is in the original image cooridnate, while the slide image is after selecting ROI
        xy = ((tile_info["tile_x"] - origin_offset[0]) / downscale_factor,
              (tile_info["tile_y"] - origin_offset[1]) / downscale_factor)
        rects.append(patches.Rectangle(xy, tile_size, tile_size))
    pc = collections.PatchCollection(rects, match_original=True, alpha=0.5, edgecolor="black")
    pc.set_array(np.array([100] * len(tile_info_list)))
    ax.add_collection(pc)
    fig.savefig(output_path)
    plt.close()


def is_already_processed(output_tiles_dir):
    if not output_tiles_dir.exists():
        return False

    if len(list(output_tiles_dir.glob("*.png"))) == 0:
        return False

    dataset_csv_path = output_tiles_dir / "dataset.csv"
    try:
        df = pd.read_csv(dataset_csv_path)
    except:
        return False

    return len(df) > 0


def process_slide(sample: Dict["SlideKey", Any], level: int, margin: int, tile_size: int,
                  foreground_threshold: Optional[float], occupancy_threshold: float, output_dir: Path,
                  thumbnail_dir: Path,
                  tile_progress: bool = False) -> str:
    """Load and process a slide, saving tile images and information to a CSV file.

    :param sample: Slide information dictionary, returned by the input slide dataset.
    :param level: Magnification level at which to process the slide.
    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    If `None` (default), an optimal threshold will be estimated automatically.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :param output_dir: Root directory for the output dataset; outputs for a single slide will be
    saved inside `output_dir/slide_id/`.
    :param tile_progress: Whether to display a progress bar in the terminal.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    slide_metadata: Dict[str, Any] = sample["metadata"]
    keys_to_save = ("slide_id", "tile_id", "image", "label",
                    "tile_x", "tile_y", "occupancy")
    metadata_keys = tuple("slide_" + key for key in slide_metadata)
    csv_columns: Tuple[str, ...] = (*keys_to_save, *metadata_keys)
    print(csv_columns)
    slide_id: str = sample["slide_id"]
    rel_slide_dir = Path(slide_id)
    output_tiles_dir = output_dir / rel_slide_dir
    logging.info(f">>> Slide dir {output_tiles_dir}")
    if is_already_processed(output_tiles_dir):
        logging.info(f">>> Skipping {output_tiles_dir} - already processed")
        return output_tiles_dir

    else:
        output_tiles_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv_path = output_tiles_dir / "dataset.csv"
        dataset_csv_file = dataset_csv_path.open('w')
        dataset_csv_file.write(','.join(csv_columns) + '\n')  # write CSV header

        n_failed_tiles = 0
        failed_tiles_csv_path = output_tiles_dir / "failed_tiles.csv"
        failed_tiles_file = failed_tiles_csv_path.open('w')
        failed_tiles_file.write('tile_id' + '\n')

        slide_image_path = Path(sample["image"])
        logging.info(f"Loading slide {slide_id} ...\nFile: {slide_image_path}")

        # Somehow it's very slow on Datarbicks
        # hack: copy the slide file to a temporary directory
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_slide_image_path = Path(tmp_dir.name) / slide_image_path.name
        logging.info(f">>> Copying {slide_image_path} to {tmp_slide_image_path}")
        shutil.copy(slide_image_path, tmp_slide_image_path)
        sample["image"] = tmp_slide_image_path
        logging.info(f">>> Finished copying {slide_image_path} to {tmp_slide_image_path}")

        # Save original slide thumbnail
        save_thumbnail(slide_image_path, thumbnail_dir / (slide_image_path.name + "_original.png"))

        loader = LoadROId(WSIReader(backend="OpenSlide"), level=level, margin=margin,
                          foreground_threshold=foreground_threshold)
        sample = loader(sample)  # load 'image' from disk

        # Save ROI thumbnail
        slide_image = sample["image"]
        plt.figure()
        plt.imshow(slide_image.transpose(1, 2, 0))
        plt.savefig(thumbnail_dir / (slide_image_path.name + "_roi.png"))
        plt.close()
        logging.info(f"Saving thumbnail {thumbnail_dir / (slide_image_path.name + '_roi.png')}, shape {slide_image.shape}")

        logging.info(f"Tiling slide {slide_id} ...")
        image_tiles, rel_tile_locations, occupancies, _ = \
            generate_tiles(sample["image"], tile_size,
                            sample["foreground_threshold"],
                            occupancy_threshold)

        # origin in level-0 coordinate
        # location in the current level coordiante
        # tile_locations in level-0 coordinate
        tile_locations = (sample["scale"] * rel_tile_locations
                            + sample["origin"]).astype(int)  # noqa: W503

        n_tiles = image_tiles.shape[0]
        logging.info(f"{n_tiles} tiles found")

        tile_info_list = []

        logging.info(f"Saving tiles for slide {slide_id} ...")
        for i in tqdm(range(n_tiles), f"Tiles ({slide_id[:6]}…)", unit="img", disable=not tile_progress):
            try:
                tile_info = get_tile_info(sample, occupancies[i], tile_locations[i], rel_slide_dir)
                tile_info_list.append(tile_info)

                save_image(image_tiles[i], output_dir / tile_info["image"])
                dataset_row = format_csv_row(tile_info, keys_to_save, metadata_keys)
                dataset_csv_file.write(dataset_row + '\n')
            except Exception as e:
                n_failed_tiles += 1
                descriptor = get_tile_descriptor(tile_locations[i])
                failed_tiles_file.write(descriptor + '\n')
                traceback.print_exc()
                warnings.warn(f"An error occurred while saving tile "
                                f"{get_tile_id(slide_id, tile_locations[i])}: {e}")

        dataset_csv_file.close()
        failed_tiles_file.close()

        # tile location overlay
        visualize_tile_locations(sample, thumbnail_dir / (slide_image_path.name + "_roi_tiles.png"), tile_info_list, tile_size, origin_offset=sample["origin"])

        if n_failed_tiles > 0:
            # TODO what we want to do with slides that have some failed tiles?
            logging.warning(f"{slide_id} is incomplete. {n_failed_tiles} tiles failed.")

        logging.info(f"Finished processing slide {slide_id}")

        return output_tiles_dir


def merge_dataset_csv_files(dataset_dir: Path) -> Path:
    """Combines all "*/dataset.csv" files into a single "dataset.csv" file in the given directory."""
    full_csv = dataset_dir / "dataset.csv"
    # TODO change how we retrieve these filenames, probably because mounted, the operation is slow
    #  and it seems to find many more files
    # print("List of files")
    # print([str(file) + '\n' for file in dataset_dir.glob("*/dataset.csv")])
    with full_csv.open('w') as full_csv_file:
        # full_csv_file.write(','.join(CSV_COLUMNS) + '\n')  # write CSV header
        first_file = True
        for slide_csv in tqdm(dataset_dir.glob("*/dataset.csv"), desc="Merging dataset.csv", unit='file'):
            logging.info(f"Merging slide {slide_csv}")
            content = slide_csv.read_text()
            if not first_file:
                content = content[content.index('\n') + 1:]  # discard header row for all but the first file
            full_csv_file.write(content)
            first_file = False
    return full_csv


def main(slides_dataset: "SlidesDataset", root_output_dir: Union[str, Path],
         level: int, tile_size: int, margin: int, foreground_threshold: Optional[float],
         occupancy_threshold: float, parallel: bool = False, overwrite: bool = False,
         n_slides: Optional[int] = None) -> None:
    """Process a slides dataset to produce a tiles dataset.

    :param slides_dataset: Input tiles dataset object.
    :param root_output_dir: The root directory of the output tiles dataset.
    :param level: Magnification level at which to process the slide.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    If `None` (default), an optimal threshold will be estimated automatically.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :param parallel: Whether slides should be processed in parallel with multiprocessing.
    :param overwrite: Whether to overwrite an existing output tiles dataset. If `True`, will delete
    and recreate `root_output_dir`, otherwise will resume by skipping already processed slides.
    :param n_slides: If given, limit the total number of slides for debugging.
    """

    # Ignoring some types here because mypy is getting confused with the MONAI Dataset class
    # to select a subsample use keyword n_slides
    dataset = Dataset(slides_dataset)[:n_slides]  # type: ignore

    # make sure all slide files exist in the image dir
    for sample in dataset:
        image_path = Path(sample["image_path"])
        assert image_path.exists(), f"{image_path} doesn't exist"

    output_dir = Path(root_output_dir)
    logging.info(f"Creating dataset of level-{level} {tile_size}x{tile_size} "
                 f"{slides_dataset.__class__.__name__} tiles at: {output_dir}")

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=not overwrite)
    thumbnail_dir = output_dir / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True)
    logging.info(f"Thumbnail directory: {thumbnail_dir}")

    func = functools.partial(process_slide, level=level, margin=margin, tile_size=tile_size,
                             foreground_threshold=foreground_threshold,
                             occupancy_threshold=occupancy_threshold, output_dir=output_dir,
                             thumbnail_dir=thumbnail_dir,
                             tile_progress=not parallel)

    if parallel:
        import multiprocessing

        pool = multiprocessing.Pool()
        map_func = pool.imap_unordered  # type: ignore
    else:
        map_func = map  # type: ignore

    list(tqdm(map_func(func, dataset), desc="Slides", unit="img", total=len(dataset)))  # type: ignore

    if parallel:
        pool.close()

    logging.info("Merging slide files in a single file")
    merge_dataset_csv_files(output_dir)
