from __future__ import annotations

import numpy as np
from albucore.utils import MAX_VALUES_BY_DTYPE, is_grayscale_image, preserve_channel_dim
from typing_extensions import Literal

from albumentations import random_utils
from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.types import MONO_CHANNEL_DIMENSIONS, ColorType

__all__ = ["cutout", "channel_dropout", "filter_keypoints_in_holes", "generate_random_fill"]


@preserve_channel_dim
def channel_dropout(
    img: np.ndarray,
    channels_to_drop: int | tuple[int, ...] | np.ndarray,
    fill_value: ColorType = 0,
) -> np.ndarray:
    if is_grayscale_image(img):
        msg = "Only one channel. ChannelDropout is not defined."
        raise NotImplementedError(msg)

    img = img.copy()
    img[..., channels_to_drop] = fill_value
    return img


def generate_random_fill(dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Generate a random fill based on dtype and target shape."""
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    if np.issubdtype(dtype, np.integer):
        return random_utils.randint(0, max_value + 1, size=shape, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        return random_utils.uniform(0, max_value, size=shape).astype(dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


def cutout(
    img: np.ndarray,
    holes: np.ndarray,
    fill_value: ColorType | Literal["random"] = 0,
) -> np.ndarray:
    """Apply cutout augmentation to the image by cutting out holes and filling them
    with either a given value or random noise.

    Args:
        img (np.ndarray): The image to augment.
        holes (np.ndarray): An array of holes with shape (num_holes, 4).
            Each hole is represented as [x1, y1, x2, y2].
        fill_value (Union[ColorType, Literal["random"]]): The fill value to use for the hole. Can be
            a single integer, a tuple or list of numbers for multichannel,
            or the string "random" to fill with random noise.

    Returns:
        np.ndarray: The augmented image.
    """
    img = img.copy()

    if isinstance(fill_value, (int, float, tuple, list)):
        fill_value = np.array(fill_value, dtype=img.dtype)

    for x_min, y_min, x_max, y_max in holes:
        if isinstance(fill_value, str) and fill_value == "random":
            shape = (
                (y_max - y_min, x_max - x_min)
                if img.ndim == MONO_CHANNEL_DIMENSIONS
                else (y_max - y_min, x_max - x_min, img.shape[2])
            )
            random_fill = generate_random_fill(img.dtype, shape)
            img[y_min:y_max, x_min:x_max] = random_fill
        else:
            img[y_min:y_max, x_min:x_max] = fill_value

    return img


@handle_empty_array
def filter_keypoints_in_holes(keypoints: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Filter out keypoints that are inside any of the holes.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+).
                                The first two columns are x and y coordinates.
        holes (np.ndarray): Array of holes with shape (num_holes, 4).
                            Each hole is represented as [x1, y1, x2, y2].

    Returns:
        np.ndarray: Array of keypoints that are not inside any hole.
    """
    # Broadcast keypoints and holes for vectorized comparison
    kp_x = keypoints[:, 0][:, np.newaxis]  # Shape: (num_keypoints, 1)
    kp_y = keypoints[:, 1][:, np.newaxis]  # Shape: (num_keypoints, 1)

    hole_x1 = holes[:, 0]  # Shape: (num_holes,)
    hole_y1 = holes[:, 1]  # Shape: (num_holes,)
    hole_x2 = holes[:, 2]  # Shape: (num_holes,)
    hole_y2 = holes[:, 3]  # Shape: (num_holes,)

    # Check if each keypoint is inside each hole
    inside_hole = (kp_x >= hole_x1) & (kp_x < hole_x2) & (kp_y >= hole_y1) & (kp_y < hole_y2)

    # A keypoint is valid if it's not inside any hole
    valid_keypoints = ~np.any(inside_hole, axis=1)

    return keypoints[valid_keypoints]
