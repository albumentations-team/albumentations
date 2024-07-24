from __future__ import annotations

from typing import Sequence, cast

import cv2
from albucore.utils import maybe_process_in_chunks, preserve_channel_dim

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox
from albumentations.core.types import BoxInternalType, ColorType, KeypointInternalType


import numpy as np

__all__ = [
    "get_crop_coords",
    "crop_bbox_by_coords",
    "crop_keypoint_by_coords",
    "crop",
    "crop_and_pad",
    "crop_and_pad_bbox",
    "crop_and_pad_keypoint",
]


def get_crop_coords(
    height: int,
    width: int,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
) -> tuple[int, int, int, int]:
    # h_start is [0, 1) and should map to [0, (height - crop_height)]  (note inclusive)
    # This is conceptually equivalent to mapping onto `range(0, (height - crop_height + 1))`
    # See: https://github.com/albumentations-team/albumentations/pull/1080
    # We want range for coordinated to be [0, image_size], right side is included
    y_min = int((height - crop_height + 1) * h_start)
    y_max = y_min + crop_height
    x_min = int((width - crop_width + 1) * w_start)
    x_max = x_min + crop_width
    return x_min, y_min, x_max, y_max


def crop_bbox_by_coords(
    bbox: BoxInternalType,
    crop_coords: tuple[int, int, int, int],
    rows: int,
    cols: int,
) -> BoxInternalType:
    denormalized_bbox = denormalize_bbox(bbox, rows, cols)

    x_min, y_min, x_max, y_max = denormalized_bbox[:4]
    x1, y1 = crop_coords[:2]
    cropped_bbox = x_min - x1, y_min - y1, x_max - x1, y_max - y1
    crop_height = crop_coords[3] - crop_coords[1]
    crop_width = crop_coords[2] - crop_coords[0]

    return cast(BoxInternalType, normalize_bbox(cropped_bbox, crop_height, crop_width))


def crop_keypoint_by_coords(
    keypoint: KeypointInternalType,
    crop_coords: tuple[int, int, int, int],
) -> KeypointInternalType:
    """Crop a keypoint using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        crop_coords (tuple): Crop box coords `(x1, x2, y1, y2)`.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    x1, y1 = crop_coords[:2]
    return x - x1, y - y1, angle, scale


def get_center_crop_coords(height: int, width: int, crop_height: int, crop_width: int) -> tuple[int, int, int, int]:
    y_min = (height - crop_height) // 2
    y_max = y_min + crop_height
    x_min = (width - crop_width) // 2
    x_max = x_min + crop_width
    return x_min, y_min, x_max, y_max


def crop(img: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int) -> np.ndarray:
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            f" (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})",
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            f"(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            f"height = {height}, width = {width})",
        )

    return img[y_min:y_max, x_min:x_max]


@preserve_channel_dim
def crop_and_pad(
    img: np.ndarray,
    crop_params: Sequence[int] | None,
    pad_params: Sequence[int] | None,
    pad_value: ColorType | None,
    rows: int,
    cols: int,
    interpolation: int,
    pad_mode: int,
    keep_size: bool,
) -> np.ndarray:
    if crop_params is not None and any(i != 0 for i in crop_params):
        img = crop(img, *crop_params)
    if pad_params is not None and any(i != 0 for i in pad_params):
        img = fgeometric.pad_with_params(
            img,
            pad_params[0],
            pad_params[1],
            pad_params[2],
            pad_params[3],
            border_mode=pad_mode,
            value=pad_value,
        )

    if keep_size:
        resize_fn = maybe_process_in_chunks(cv2.resize, dsize=(cols, rows), interpolation=interpolation)
        return resize_fn(img)

    return img


def crop_and_pad_bbox(
    bbox: BoxInternalType,
    crop_params: Sequence[int] | None,
    pad_params: Sequence[int] | None,
    rows: int,
    cols: int,
    result_rows: int,
    result_cols: int,
) -> BoxInternalType:
    x1, y1, x2, y2 = denormalize_bbox(bbox, rows, cols)[:4]

    if crop_params is not None:
        crop_x, crop_y = crop_params[:2]

        x1 -= crop_x
        y1 -= crop_y
        x2 -= crop_x
        y2 -= crop_y

    if pad_params is not None:
        top = pad_params[0]
        left = pad_params[2]

        x1 += left
        y1 += top
        x2 += left
        y2 += top

    return cast(BoxInternalType, normalize_bbox((x1, y1, x2, y2), result_rows, result_cols))


def crop_and_pad_keypoint(
    keypoint: KeypointInternalType,
    crop_params: Sequence[int] | None,
    pad_params: Sequence[int] | None,
    rows: int,
    cols: int,
    result_rows: int,
    result_cols: int,
    keep_size: bool,
) -> KeypointInternalType:
    x, y, angle, scale = keypoint[:4]

    if crop_params is not None:
        crop_x1, crop_y1 = crop_params[:2]
        x, y = x - crop_x1, y - crop_y1
    if pad_params is not None:
        top = pad_params[0]
        left = pad_params[2]

        x += left
        y += top

    if keep_size and (result_cols != cols or result_rows != rows):
        scale_x = cols / result_cols
        scale_y = rows / result_rows
        return fgeometric.keypoint_scale((x, y, angle, scale), scale_x, scale_y)

    return x, y, angle, scale
