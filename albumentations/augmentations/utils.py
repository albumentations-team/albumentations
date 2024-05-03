from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import cv2
import numpy as np
from typing_extensions import Concatenate, ParamSpec

from albumentations.core.keypoints_utils import angle_to_2pi_range
from albumentations.core.types import (
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
    KeypointInternalType,
)

__all__ = [
    "read_bgr_image",
    "read_rgb_image",
    "read_grayscale",
    "MAX_VALUES_BY_DTYPE",
    "NPDTYPE_TO_OPENCV_DTYPE",
    "clipped",
    "get_opencv_dtype_from_numpy",
    "angle_2pi_range",
    "clip",
    "preserve_channel_dim",
    "ensure_contiguous",
    "is_rgb_image",
    "is_grayscale_image",
    "is_multispectral_image",
    "get_num_channels",
    "non_rgb_warning",
    "_maybe_process_in_chunks",
]

P = ParamSpec("P")

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
    np.uint8: 255,
    np.uint16: 65535,
    np.uint32: 4294967295,
    np.float32: 1.0,
}

NPDTYPE_TO_OPENCV_DTYPE = {
    np.uint8: cv2.CV_8U,
    np.uint16: cv2.CV_16U,
    np.int32: cv2.CV_32S,
    np.float32: cv2.CV_32F,
    np.float64: cv2.CV_64F,
    np.dtype("uint8"): cv2.CV_8U,
    np.dtype("uint16"): cv2.CV_16U,
    np.dtype("int32"): cv2.CV_32S,
    np.dtype("float32"): cv2.CV_32F,
    np.dtype("float64"): cv2.CV_64F,
}

TWO = 2
FOUR = 4
BIG_INTEGER = MAX_VALUES_BY_DTYPE[np.uint32]


def read_bgr_image(path: Union[str, Path]) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def read_rgb_image(path: Union[str, Path]) -> np.ndarray:
    image = read_bgr_image(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_grayscale(path: Union[str, Path]) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def clipped(func: Callable[Concatenate[np.ndarray, P], np.ndarray]) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


def clip(img: np.ndarray, dtype: np.dtype, maxval: float) -> np.ndarray:
    return np.clip(img, 0, maxval).astype(dtype)


def get_opencv_dtype_from_numpy(value: Union[np.ndarray, int, np.dtype, object]) -> int:
    """Return a corresponding OpenCV dtype for a numpy's dtype
    :param value: Input dtype of numpy array
    :return: Corresponding dtype for OpenCV
    """
    if isinstance(value, np.ndarray):
        value = value.dtype
    return NPDTYPE_TO_OPENCV_DTYPE[value]


def angle_2pi_range(
    func: Callable[Concatenate[KeypointInternalType, P], KeypointInternalType],
) -> Callable[Concatenate[KeypointInternalType, P], KeypointInternalType]:
    @wraps(func)
    def wrapped_function(keypoint: KeypointInternalType, *args: P.args, **kwargs: P.kwargs) -> KeypointInternalType:
        (x, y, a, s) = func(keypoint, *args, **kwargs)[:4]
        return (x, y, angle_to_2pi_range(a), s)

    return wrapped_function


def preserve_channel_dim(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray],
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Preserve dummy channel dim."""

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        shape = img.shape
        result = func(img, *args, **kwargs)
        if (
            len(shape) == NUM_MULTI_CHANNEL_DIMENSIONS
            and shape[-1] == 1
            and len(result.shape) == MONO_CHANNEL_DIMENSIONS
        ):
            return np.expand_dims(result, axis=-1)

        if len(shape) == MONO_CHANNEL_DIMENSIONS and len(result.shape) == NUM_MULTI_CHANNEL_DIMENSIONS:
            return result[:, :, 0]
        return result

    return wrapped_function


def ensure_contiguous(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray],
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Ensure that input img is contiguous."""

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        img = np.require(img, requirements=["C_CONTIGUOUS"])
        return func(img, *args, **kwargs)

    return wrapped_function


def get_num_channels(image: np.ndarray) -> int:
    return image.shape[2] if len(image.shape) == NUM_MULTI_CHANNEL_DIMENSIONS else 1


def is_rgb_image(image: np.ndarray) -> bool:
    return get_num_channels(image) == NUM_RGB_CHANNELS


def is_grayscale_image(image: np.ndarray) -> bool:
    return get_num_channels(image) == 1


def is_multispectral_image(image: np.ndarray) -> bool:
    num_channels = get_num_channels(image)
    return num_channels not in {1, 3}


def non_rgb_warning(image: np.ndarray) -> None:
    if not is_rgb_image(image):
        message = "This transformation expects 3-channel images"
        if is_grayscale_image(image):
            message += "\nYou can convert your grayscale image to RGB using cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))"
        if is_multispectral_image(image):  # Any image with a number of channels other than 1 and 3
            message += "\nThis transformation cannot be applied to multi-spectral images"

        raise ValueError(message)


def _maybe_process_in_chunks(
    process_fn: Callable[Concatenate[np.ndarray, P], np.ndarray],
    **kwargs: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap OpenCV function to enable processing images with more than 4 channels.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.

    Returns:
        numpy.ndarray: Transformed image.

    """

    @wraps(process_fn)
    def __process_fn(img: np.ndarray) -> np.ndarray:
        num_channels = get_num_channels(img)
        if num_channels > FOUR:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == TWO:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            return np.dstack(chunks)

        return process_fn(img, **kwargs)

    return __process_fn


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
