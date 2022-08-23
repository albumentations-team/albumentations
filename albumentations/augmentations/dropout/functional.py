from typing import Iterable, List, Tuple, Union

import numpy as np

from albumentations.augmentations.utils import preserve_shape

__all__ = ["cutout", "channel_dropout"]


@preserve_shape
def channel_dropout(
    img: np.ndarray, channels_to_drop: Union[int, Tuple[int, ...], np.ndarray], fill_value: Union[int, float] = 0
) -> np.ndarray:
    if len(img.shape) == 2 or img.shape[2] == 1:
        raise NotImplementedError("Only one channel. ChannelDropout is not defined.")

    img = img.copy()
    img[..., channels_to_drop] = fill_value
    return img


def cutout(
    img: np.ndarray, holes: Iterable[Tuple[int, int, int, int]], fill_value: Union[int, float] = 0
) -> np.ndarray:
    # Make a copy of the input image since we don't want to modify it directly
    img = img.copy()
    for x1, y1, x2, y2 in holes:
        img[y1:y2, x1:x2] = fill_value
    return img
