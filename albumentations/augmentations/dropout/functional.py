from typing import Iterable, Tuple, Union

import numpy as np
from albucore.utils import MAX_VALUES_BY_DTYPE, is_grayscale_image, preserve_channel_dim
from typing_extensions import Literal

from albumentations import random_utils
from albumentations.core.types import MONO_CHANNEL_DIMENSIONS, ColorType, KeypointType

__all__ = ["cutout", "channel_dropout", "keypoint_in_hole"]


@preserve_channel_dim
def channel_dropout(
    img: np.ndarray,
    channels_to_drop: Union[int, Tuple[int, ...], np.ndarray],
    fill_value: ColorType = 0,
) -> np.ndarray:
    if is_grayscale_image(img):
        msg = "Only one channel. ChannelDropout is not defined."
        raise NotImplementedError(msg)

    img = img.copy()
    img[..., channels_to_drop] = fill_value
    return img


def generate_random_fill(dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
    """Generate a random fill based on dtype and target shape."""
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    if np.issubdtype(dtype, np.integer):
        return random_utils.randint(0, max_value + 1, size=shape, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        return random_utils.uniform(0, max_value, size=shape).astype(dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


def cutout(
    img: np.ndarray,
    holes: Iterable[Tuple[int, int, int, int]],
    fill_value: Union[ColorType, Literal["random"]] = 0,
) -> np.ndarray:
    """Apply cutout augmentation to the image by cutting out holes and filling them
    with either a given value or random noise.

    Args:
        img (np.ndarray): The image to augment.
        holes (Iterable[Tuple[int, int, int, int]]): An iterable of tuples where each
            tuple contains the coordinates of the top-left and bottom-right corners of
            the rectangular hole (x1, y1, x2, y2).
        fill_value (Union[ColorType, Literal["random"]]): The fill value to use for the hole. Can be
            a single integer, a tuple or list of numbers for multichannel,
            or the string "random" to fill with random noise.

    Returns:
        np.ndarray: The augmented image.
    """
    img = img.copy()

    if isinstance(fill_value, (int, float, tuple, list)):
        fill_value = np.array(fill_value, dtype=img.dtype)

    for x1, y1, x2, y2 in holes:
        if isinstance(fill_value, str) and fill_value == "random":
            shape = (y2 - y1, x2 - x1) if img.ndim == MONO_CHANNEL_DIMENSIONS else (y2 - y1, x2 - x1, img.shape[2])
            random_fill = generate_random_fill(img.dtype, shape)
            img[y1:y2, x1:x2] = random_fill
        else:
            img[y1:y2, x1:x2] = fill_value

    return img


def keypoint_in_hole(keypoint: KeypointType, hole: Tuple[int, int, int, int]) -> bool:
    x, y = keypoint[:2]
    x1, y1, x2, y2 = hole
    return x1 <= x < x2 and y1 <= y < y2
