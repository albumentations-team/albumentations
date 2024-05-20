from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np
from albucore.utils import (
    is_grayscale_image,
    is_multispectral_image,
    is_rgb_image,
)
from typing_extensions import Concatenate, ParamSpec

from albumentations.core.keypoints_utils import angle_to_2pi_range
from albumentations.core.types import (
    KeypointInternalType,
)

__all__ = [
    "read_bgr_image",
    "read_rgb_image",
    "read_grayscale",
    "angle_2pi_range",
    "non_rgb_warning",
]

P = ParamSpec("P")


def read_bgr_image(path: Union[str, Path]) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def read_rgb_image(path: Union[str, Path]) -> np.ndarray:
    image = read_bgr_image(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_grayscale(path: Union[str, Path]) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def angle_2pi_range(
    func: Callable[Concatenate[KeypointInternalType, P], KeypointInternalType],
) -> Callable[Concatenate[KeypointInternalType, P], KeypointInternalType]:
    @wraps(func)
    def wrapped_function(keypoint: KeypointInternalType, *args: P.args, **kwargs: P.kwargs) -> KeypointInternalType:
        (x, y, a, s) = func(keypoint, *args, **kwargs)[:4]
        return (x, y, angle_to_2pi_range(a), s)

    return wrapped_function


def non_rgb_warning(image: np.ndarray) -> None:
    if not is_rgb_image(image):
        message = "This transformation expects 3-channel images"
        if is_grayscale_image(image):
            message += "\nYou can convert your grayscale image to RGB using cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))"
        if is_multispectral_image(image):  # Any image with a number of channels other than 1 and 3
            message += "\nThis transformation cannot be applied to multi-spectral images"

        raise ValueError(message)


def check_range(value: Tuple[float, float], lower_bound: float, upper_bound: float, name: Optional[str]) -> None:
    """Checks if the given value is within the specified bounds

    Args:
        value: The value to check and convert. Can be a single float or a tuple of floats.
        lower_bound: The lower bound for the range check.
        upper_bound: The upper bound for the range check.
        name: The name of the parameter being checked. Used for error messages.

    Raises:
        ValueError: If the value is outside the bounds or if the tuple values are not ordered correctly.
    """
    if not all(lower_bound <= x <= upper_bound for x in value):
        raise ValueError(f"All values in {name} must be within [{lower_bound}, {upper_bound}] for tuple inputs.")
    if not value[0] <= value[1]:
        raise ValueError(f"{name!s} tuple values must be ordered as (min, max). Got: {value}")
