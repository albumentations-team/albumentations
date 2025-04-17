"""Functional implementations of image cropping operations.

This module provides utility functions for performing various cropping operations on images,
bounding boxes, and keypoints. It includes functions to calculate crop coordinates, crop images,
and handle the corresponding transformations for bounding boxes and keypoints to maintain
consistency between different data types during cropping operations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from albucore import maybe_process_in_chunks, preserve_channel_dim

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes

__all__ = [
    "crop",
    "crop_and_pad",
    "crop_and_pad_bboxes",
    "crop_and_pad_keypoints",
    "crop_bboxes_by_coords",
    "crop_keypoints_by_coords",
    "get_center_crop_coords",
    "get_crop_coords",
    "pad_along_axes",
    "volume_crop_yx",
    "volumes_crop_yx",
]


def get_crop_coords(
    image_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    h_start: float,
    w_start: float,
) -> tuple[int, int, int, int]:
    """Get crop coordinates.

    This function gets the crop coordinates.

    Args:
        image_shape (tuple[int, int]): Original image shape.
        crop_shape (tuple[int, int]): Crop shape.
        h_start (float): Start height.
        w_start (float): Start width.

    Returns:
        tuple[int, int, int, int]: Crop coordinates.

    """
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


@handle_empty_array("keypoints")
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
    """Get center crop coordinates.

    This function gets the center crop coordinates.

    Args:
        image_shape (tuple[int, int]): Original image shape.
        crop_shape (tuple[int, int]): Crop shape.

    Returns:
        tuple[int, int, int, int]: Center crop coordinates.

    """
    height, width = image_shape[:2]
    crop_height, crop_width = crop_shape[:2]

    y_min = (height - crop_height) // 2
    y_max = y_min + crop_height
    x_min = (width - crop_width) // 2
    x_max = x_min + crop_width
    return x_min, y_min, x_max, y_max


def crop(img: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int) -> np.ndarray:
    """Crop an image.

    This function crops an image.

    Args:
        img (np.ndarray): Input image.
        x_min (int): Minimum x coordinate.
        y_min (int): Minimum y coordinate.
        x_max (int): Maximum x coordinate.
        y_max (int): Maximum y coordinate.

    Returns:
        np.ndarray: Cropped image.

    """
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
    crop_params: tuple[int, int, int, int] | None,
    pad_params: tuple[int, int, int, int] | None,
    pad_value: tuple[float, ...] | float | None,
    image_shape: tuple[int, int],
    interpolation: int,
    pad_mode: int,
    keep_size: bool,
) -> np.ndarray:
    """Crop and pad an image.

    This function crops and pads an image.

    Args:
        img (np.ndarray): Input image.
        crop_params (tuple[int, int, int, int] | None): Crop parameters.
        pad_params (tuple[int, int, int, int] | None): Pad parameters.
        pad_value (tuple[float, ...] | float | None): Pad value.
        image_shape (tuple[int, int]): Original image shape.
        interpolation (int): Interpolation method.
        pad_mode (int): Pad mode.
        keep_size (bool): Whether to keep the original size.

    Returns:
        np.ndarray: Cropped and padded image.

    """
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
    """Crop and pad bounding boxes.

    This function crops and pads bounding boxes.

    Args:
        bboxes (np.ndarray): Array of bounding boxes.
        crop_params (tuple[int, int, int, int] | None): Crop parameters.
        pad_params (tuple[int, int, int, int] | None): Pad parameters.
        image_shape (tuple[int, int]): Original image shape.
        result_shape (tuple[int, int]): Result image shape.

    Returns:
        np.ndarray: Array of cropped and padded bounding boxes.

    """
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


@handle_empty_array("keypoints")
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


def volume_crop_yx(
    volume: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> np.ndarray:
    """Crop a single volume along Y (height) and X (width) axes only.

    Args:
        volume (np.ndarray): Input volume with shape (D, H, W) or (D, H, W, C).
        x_min (int): Minimum width coordinate.
        y_min (int): Minimum height coordinate.
        x_max (int): Maximum width coordinate.
        y_max (int): Maximum height coordinate.

    Returns:
        np.ndarray: Cropped volume (D, H_new, W_new, [C]).

    Raises:
        ValueError: If crop coordinates are invalid.

    """
    _, height, width = volume.shape[:3]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "Crop coordinates must satisfy min < max. Got: "
            f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})",
        )

    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValueError(
            "Crop coordinates must be within image dimensions (H, W). Got: "
            f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}) "
            f"for volume shape {volume.shape[:3]}",
        )

    # Crop along H (axis 1) and W (axis 2)
    return volume[:, y_min:y_max, x_min:x_max]


def volumes_crop_yx(
    volumes: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> np.ndarray:
    """Crop a batch of volumes along Y (height) and X (width) axes only.

    Args:
        volumes (np.ndarray): Input batch of volumes with shape (B, D, H, W) or (B, D, H, W, C).
        x_min (int): Minimum width coordinate.
        y_min (int): Minimum height coordinate.
        x_max (int): Maximum width coordinate.
        y_max (int): Maximum height coordinate.

    Returns:
        np.ndarray: Cropped batch of volumes (B, D, H_new, W_new, [C]).

    Raises:
        ValueError: If crop coordinates are invalid or volumes shape is incorrect.

    """
    if not 4 <= volumes.ndim <= 5:
        raise ValueError(f"Input volumes should have 4 or 5 dimensions, got {volumes.ndim}")

    depth, height, width = volumes.shape[1:4]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "Crop coordinates must satisfy min < max. Got: "
            f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max})",
        )

    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValueError(
            "Crop coordinates must be within image dimensions (H, W). Got: "
            f"(x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}) "
            f"for volume shape {(depth, height, width)}",
        )

    # Crop along H (axis 2) and W (axis 3)
    return volumes[:, :, y_min:y_max, x_min:x_max]


def pad_along_axes(
    arr: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    h_axis: int,
    w_axis: int,
    border_mode: int,
    pad_value: float | Sequence[float] = 0,
) -> np.ndarray:
    """Pad an array along specified height (H) and width (W) axes using np.pad.

    Args:
        arr (np.ndarray): Input array.
        pad_top (int): Padding added to the top (start of H axis).
        pad_bottom (int): Padding added to the bottom (end of H axis).
        pad_left (int): Padding added to the left (start of W axis).
        pad_right (int): Padding added to the right (end of W axis).
        h_axis (int): Index of the height axis (Y).
        w_axis (int): Index of the width axis (X).
        border_mode (int): OpenCV border mode.
        pad_value (float | Sequence[float]): Value for constant padding.

    Returns:
        np.ndarray: Padded array.

    Raises:
        ValueError: If border_mode is unsupported or axis indices are out of bounds.

    """
    ndim = arr.ndim
    if not (0 <= h_axis < ndim and 0 <= w_axis < ndim):
        raise ValueError(f"Axis indices {h_axis}, {w_axis} are out of bounds for array with {ndim} dimensions.")
    if h_axis == w_axis:
        raise ValueError(f"Height axis {h_axis} and width axis {w_axis} cannot be the same.")

    mode_map = {
        cv2.BORDER_CONSTANT: "constant",
        cv2.BORDER_REPLICATE: "edge",
        cv2.BORDER_REFLECT: "reflect",
        cv2.BORDER_REFLECT_101: "symmetric",
        cv2.BORDER_WRAP: "wrap",
    }
    if border_mode not in mode_map:
        raise ValueError(f"Unsupported border_mode: {border_mode}")
    np_mode = mode_map[border_mode]

    pad_width = [(0, 0)] * ndim  # Initialize padding for all dimensions
    pad_width[h_axis] = (pad_top, pad_bottom)
    pad_width[w_axis] = (pad_left, pad_right)

    # Initialize kwargs with mode
    kwargs: dict[str, Any] = {"mode": np_mode}
    # Add constant_values only if mode is constant
    if np_mode == "constant":
        kwargs["constant_values"] = pad_value

    return np.pad(arr, pad_width, **kwargs)
