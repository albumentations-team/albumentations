from typing import Iterable, Tuple, Union

import numpy as np

from albumentations.augmentations.utils import preserve_shape
from albumentations.core.types import ColorType, KeypointType

TWO = 2

__all__ = ["cutout", "channel_dropout", "keypoint_in_hole"]


@preserve_shape
def channel_dropout(
    img: np.ndarray, channels_to_drop: Union[int, Tuple[int, ...], np.ndarray], fill_value: ColorType = 0
) -> np.ndarray:
    if len(img.shape) == TWO or img.shape[2] == 1:
        msg = "Only one channel. ChannelDropout is not defined."
        raise NotImplementedError(msg)

    img = img.copy()
    img[..., channels_to_drop] = fill_value
    return img


def cutout(img: np.ndarray, holes: Iterable[Tuple[int, int, int, int]], fill_value: ColorType = 0) -> np.ndarray:
    img = img.copy()
    # Convert fill_value to a NumPy array for consistent broadcasting
    if isinstance(fill_value, (tuple, list)):
        fill_value = np.array(fill_value)

    for x1, y1, x2, y2 in holes:
        img[y1:y2, x1:x2] = fill_value
    return img


def keypoint_in_hole(keypoint: KeypointType, hole: Tuple[int, int, int, int]) -> bool:
    x, y = keypoint[:2]
    x1, y1, x2, y2 = hole
    return x1 <= x < x2 and y1 <= y < y2
