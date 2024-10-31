from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from albucore import maybe_process_in_chunks, preserve_channel_dim

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from albumentations.core.types import ColorType

__all__ = [
    "get_crop_coords",
    "crop_bboxes_by_coords",
    "crop_keypoints_by_coords",
    "get_center_crop_coords",
    "crop",
    "crop_and_pad",
    "crop_and_pad_bboxes",
    "crop_and_pad_keypoints",
]


def get_crop_coords(
    image_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    h_start: float,
    w_start: float,
) -> tuple[int, int, int, int]:
    # h_start is [0, 1) and should map to [0, (height - crop_height)]  (note inclusive)
    # This is conceptually equivalent to mapping onto `range(0, (height - crop_height + 1))`
    # See: https://github.com/albumentations-team/albumentations/pull/1080
    # We want range for coordinated to be [0, image_size], right side is included

    height, width = image_shape[:2]

    # Clip crop dimensions to image dimensions
    crop_height = min(crop_shape[0], height)
    crop_width = min(crop_shape[1], width)

    y_min = int((height - crop_height + 1) * h_start)
    y_max = y_min + crop_height
    x_min = int((width - crop_width + 1) * w_start)
    x_max = x_min + crop_width
    return x_min, y_min, x_max, y_max


def crop_bboxes_by_coords(
    bboxes: np.ndarray,
    crop_coords: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    normalized_input: bool = True,
) -> np.ndarray:
    """Crop bounding boxes based on given crop coordinates.

    This function adjusts bounding boxes to fit within a cropped image.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (N, 4+) where each row is
                             [x_min, y_min, x_max, y_max, ...]. The bounding box coordinates
                             can be either normalized (in [0, 1]) if normalized_input=True or
                             absolute pixel values if normalized_input=False.
        crop_coords (tuple[int, int, int, int]): Crop coordinates (x_min, y_min, x_max, y_max)
                                                 in absolute pixel values.
        image_shape (tuple[int, int]): Original image shape (height, width).
        normalized_input (bool): Whether input boxes are in normalized coordinates.
                               If True, assumes input is normalized [0,1] and returns normalized coordinates.
                               If False, assumes input is in absolute pixels and returns absolute coordinates.
                               Default: True for backward compatibility.

    Returns:
        np.ndarray: Array of cropped bounding boxes. Coordinates will be in the same format as input
                   (normalized if normalized_input=True, absolute pixels if normalized_input=False).

    Note:
        Bounding boxes that fall completely outside the crop area will be removed.
        Bounding boxes that partially overlap with the crop area will be adjusted to fit within it.
    """
    if not bboxes.size:
        return bboxes

    # Convert to absolute coordinates if needed
    if normalized_input:
        cropped_bboxes = denormalize_bboxes(bboxes.copy().astype(np.float32), image_shape)
    else:
        cropped_bboxes = bboxes.copy().astype(np.float32)

    x_min, y_min = crop_coords[:2]

    # Subtract crop coordinates
    cropped_bboxes[:, [0, 2]] -= x_min
    cropped_bboxes[:, [1, 3]] -= y_min

    # Calculate crop shape
    crop_height = crop_coords[3] - crop_coords[1]
    crop_width = crop_coords[2] - crop_coords[0]
    crop_shape = (crop_height, crop_width)

    # Return in same format as input
    return normalize_bboxes(cropped_bboxes, crop_shape) if normalized_input else cropped_bboxes


@handle_empty_array
def crop_keypoints_by_coords(
    keypoints: np.ndarray,
    crop_coords: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop keypoints using the provided coordinates of bottom-left and top-right corners in pixels.

    Args:
        keypoints (np.ndarray): An array of keypoints with shape (N, 4+) where each row is (x, y, angle, scale, ...).
        crop_coords (tuple): Crop box coords (x1, y1, x2, y2).

    Returns:
        np.ndarray: An array of cropped keypoints with the same shape as the input.
    """
    x1, y1 = crop_coords[:2]

    cropped_keypoints = keypoints.copy()
    cropped_keypoints[:, 0] -= x1  # Adjust x coordinates
    cropped_keypoints[:, 1] -= y1  # Adjust y coordinates

    return cropped_keypoints


def get_center_crop_coords(image_shape: tuple[int, int], crop_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    crop_height, crop_width = crop_shape[:2]

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
    image_shape: tuple[int, int],
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
        rows, cols = image_shape[:2]
        resize_fn = maybe_process_in_chunks(cv2.resize, dsize=(cols, rows), interpolation=interpolation)
        return resize_fn(img)

    return img


def crop_and_pad_bboxes(
    bboxes: np.ndarray,
    crop_params: tuple[int, int, int, int] | None,
    pad_params: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
    result_shape: tuple[int, int],
) -> np.ndarray:
    if len(bboxes) == 0:
        return bboxes

    # Denormalize bboxes
    denormalized_bboxes = denormalize_bboxes(bboxes, image_shape)

    if crop_params is not None:
        crop_x, crop_y = crop_params[:2]
        # Subtract crop values from x and y coordinates
        denormalized_bboxes[:, [0, 2]] -= crop_x
        denormalized_bboxes[:, [1, 3]] -= crop_y

    if pad_params is not None:
        top, _, left, _ = pad_params
        # Add pad values to x and y coordinates
        denormalized_bboxes[:, [0, 2]] += left
        denormalized_bboxes[:, [1, 3]] += top

    # Normalize bboxes to the result shape
    return normalize_bboxes(denormalized_bboxes, result_shape)


@handle_empty_array
def crop_and_pad_keypoints(
    keypoints: np.ndarray,
    crop_params: tuple[int, int, int, int] | None = None,
    pad_params: tuple[int, int, int, int] | None = None,
    image_shape: tuple[int, int] = (0, 0),
    result_shape: tuple[int, int] = (0, 0),
    keep_size: bool = False,
) -> np.ndarray:
    """Crop and pad multiple keypoints simultaneously.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 4+) where each row is (x, y, angle, scale, ...).
        crop_params (Sequence[int], optional): Crop parameters [crop_x1, crop_y1, ...].
        pad_params (Sequence[int], optional): Pad parameters [top, bottom, left, right].
        image_shape (Tuple[int, int]): Original image shape (rows, cols).
        result_shape (Tuple[int, int]): Result image shape (rows, cols).
        keep_size (bool): Whether to keep the original size.

    Returns:
        np.ndarray: Array of transformed keypoints with the same shape as input.
    """
    transformed_keypoints = keypoints.copy()

    if crop_params is not None:
        crop_x1, crop_y1 = crop_params[:2]
        transformed_keypoints[:, 0] -= crop_x1
        transformed_keypoints[:, 1] -= crop_y1

    if pad_params is not None:
        top, _, left, _ = pad_params
        transformed_keypoints[:, 0] += left
        transformed_keypoints[:, 1] += top

    rows, cols = image_shape[:2]
    result_rows, result_cols = result_shape[:2]

    if keep_size and (result_cols != cols or result_rows != rows):
        scale_x = cols / result_cols
        scale_y = rows / result_rows
        return fgeometric.keypoints_scale(transformed_keypoints, scale_x, scale_y)

    return transformed_keypoints
