"""Module containing utility functions for augmentation operations.

This module provides a collection of helper functions and utilities used throughout
the augmentation pipeline. It includes functions for image loading, type checking,
error handling, mathematical operations, and decorators that add functionality to
other functions in the codebase. These utilities help ensure consistent behavior
and simplify common operations across different augmentation transforms.
"""

from __future__ import annotations

import functools
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import cv2
import numpy as np
from albucore.utils import (
    is_grayscale_image,
    is_multispectral_image,
    is_rgb_image,
)
from typing_extensions import Concatenate, ParamSpec

from albumentations.core.keypoints_utils import angle_to_2pi_range

if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "angle_2pi_range",
    "non_rgb_error",
    "read_bgr_image",
    "read_grayscale",
    "read_rgb_image",
]

P = ParamSpec("P")
T = TypeVar("T", bound=np.ndarray)
F = TypeVar("F", bound=Callable[..., Any])


def read_bgr_image(path: str | Path) -> np.ndarray:
    """Read an image in BGR format from the specified path.

    Args:
        path (str | Path): Path to the image file.

    Returns:
        np.ndarray: Image in BGR format as a numpy array.

    """
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def read_rgb_image(path: str | Path) -> np.ndarray:
    """Read an image in RGB format from the specified path.

    This function reads an image in BGR format using OpenCV and then
    converts it to RGB format.

    Args:
        path (str | Path): Path to the image file.

    Returns:
        np.ndarray: Image in RGB format as a numpy array.

    """
    image = read_bgr_image(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_grayscale(path: str | Path) -> np.ndarray:
    """Read a grayscale image from the specified path.

    Args:
        path (str | Path): Path to the image file.

    Returns:
        np.ndarray: Grayscale image as a numpy array.

    """
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def angle_2pi_range(
    func: Callable[Concatenate[np.ndarray, P], np.ndarray],
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Decorator to normalize angle values to the range [0, 2π).

    This decorator wraps a function that processes keypoints, ensuring that
    angle values (stored in the 4th column, index 3) are normalized to the
    range [0, 2π) after the wrapped function executes.

    Args:
        func (Callable): Function that processes keypoints and returns a numpy array.
            The function should take a keypoints array as its first parameter.

    Returns:
        Callable: Wrapped function that normalizes angles after processing keypoints.

    """

    @wraps(func)
    def wrapped_function(keypoints: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        result = func(keypoints, *args, **kwargs)
        if len(result) > 0 and result.shape[1] > 3:
            result[:, 3] = angle_to_2pi_range(result[:, 3])
        return result

    return wrapped_function


def non_rgb_error(image: np.ndarray) -> None:
    """Check if the input image is RGB and raise a ValueError if it's not.

    This function is used to ensure that certain transformations are only applied to
    RGB images. It provides helpful error messages for grayscale and multi-spectral images.

    Args:
        image (np.ndarray): The input image to check. Expected to be a numpy array
                            representing an image.

    Raises:
        ValueError: If the input image is not an RGB image (i.e., does not have exactly 3 channels).
                    The error message includes specific instructions for grayscale images
                    and a note about incompatibility with multi-spectral images.

    Note:
        - RGB images are expected to have exactly 3 channels.
        - Grayscale images (1 channel) will trigger an error with conversion instructions.
        - Multi-spectral images (more than 3 channels) will trigger an error stating incompatibility.

    Example:
        >>> import numpy as np
        >>> rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> non_rgb_error(rgb_image)  # No error raised
        >>>
        >>> grayscale_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> non_rgb_error(grayscale_image)  # Raises ValueError with conversion instructions
        >>>
        >>> multispectral_image = np.random.randint(0, 256, (100, 100, 5), dtype=np.uint8)
        >>> non_rgb_error(multispectral_image)  # Raises ValueError stating incompatibility

    """
    if not is_rgb_image(image):
        message = "This transformation expects 3-channel images"
        if is_grayscale_image(image):
            message += "\nYou can convert your grayscale image to RGB using cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))"
        if is_multispectral_image(image):  # Any image with a number of channels other than 1 and 3
            message += "\nThis transformation cannot be applied to multi-spectral images"

        raise ValueError(message)


def check_range(value: tuple[float, float], lower_bound: float, upper_bound: float, name: str | None) -> None:
    """Checks if the given value is within the specified bounds

    Args:
        value (tuple[float, float]): The value to check and convert. Can be a single float or a tuple of floats.
        lower_bound (float): The lower bound for the range check.
        upper_bound (float): The upper bound for the range check.
        name (str | None): The name of the parameter being checked. Used for error messages.

    Raises:
        ValueError: If the value is outside the bounds or if the tuple values are not ordered correctly.

    """
    if not all(lower_bound <= x <= upper_bound for x in value):
        raise ValueError(f"All values in {name} must be within [{lower_bound}, {upper_bound}] for tuple inputs.")
    if not value[0] <= value[1]:
        raise ValueError(f"{name!s} tuple values must be ordered as (min, max). Got: {value}")


class PCA:
    def __init__(self, n_components: int | None = None) -> None:
        if n_components is not None and n_components <= 0:
            raise ValueError("Number of components must be greater than zero.")
        self.n_components = n_components
        self.mean: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        x = x.astype(np.float64, copy=False)  # avoid unnecessary copy if already float64
        n_samples, n_features = x.shape

        # Determine the number of components if not set
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        self.mean, eigenvectors, eigenvalues = cv2.PCACompute2(x, mean=None, maxComponents=self.n_components)
        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues.flatten()

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise ValueError(
                "This PCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )
        x = x.astype(np.float64, copy=False)  # avoid unnecessary copy if already float64
        return cv2.PCAProject(x, self.mean, self.components_)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise ValueError(
                "This PCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )
        return cv2.PCABackProject(x, self.mean, self.components_)

    def explained_variance_ratio(self) -> np.ndarray:
        if self.explained_variance_ is None:
            raise ValueError(
                "This PCA instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.",
            )
        total_variance = np.sum(self.explained_variance_)
        return self.explained_variance_ / total_variance

    def cumulative_explained_variance_ratio(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio())


def handle_empty_array(param_name: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if the parameter is passed as positional argument
            if len(args) > 0:
                array = args[0]
            # Check if the parameter is passed as keyword argument
            elif param_name in kwargs:
                array = kwargs[param_name]
            else:
                raise ValueError(f"Missing required argument: {param_name}")

            if len(array) == 0:
                return array
            return func(*args, **kwargs)

        return cast("F", wrapper)

    return decorator
