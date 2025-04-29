"""Functional implementations of various blur operations for image processing.

This module provides a collection of low-level functions for applying different blur effects
to images, including standard blur, median blur, glass blur, defocus, and zoom effects.
These functions form the foundation for the corresponding transform classes.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import product
from math import ceil
from typing import Literal
from warnings import warn

import cv2
import numpy as np
from albucore import clipped, float32_io, maybe_process_in_chunks, preserve_channel_dim, uint8_io
from pydantic import ValidationInfo

from albumentations.augmentations.functional import convolve
from albumentations.augmentations.geometric.functional import scale
from albumentations.core.type_definitions import EIGHT

__all__ = ["box_blur", "central_zoom", "defocus", "glass_blur", "median_blur", "zoom_blur"]


@preserve_channel_dim
def box_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Blur an image.

    This function applies a blur to an image.

    Args:
        img (np.ndarray): Input image.
        ksize (int): Kernel size.

    Returns:
        np.ndarray: Blurred image.

    """
    blur_fn = maybe_process_in_chunks(cv2.blur, ksize=(ksize, ksize))
    return blur_fn(img)


@preserve_channel_dim
@uint8_io
def median_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Median blur an image.

    This function applies a median blur to an image.

    Args:
        img (np.ndarray): Input image.
        ksize (int): Kernel size.

    Returns:
        np.ndarray: Median blurred image.

    """
    blur_fn = maybe_process_in_chunks(cv2.medianBlur, ksize=ksize)
    return blur_fn(img)


@preserve_channel_dim
def glass_blur(
    img: np.ndarray,
    sigma: float,
    max_delta: int,
    iterations: int,
    dxy: np.ndarray,
    mode: Literal["fast", "exact"],
) -> np.ndarray:
    """Glass blur an image.

    This function applies a glass blur to an image.

    Args:
        img (np.ndarray): Input image.
        sigma (float): Sigma.
        max_delta (int): Maximum delta.
        iterations (int): Number of iterations.
        dxy (np.ndarray): Dxy.
        mode (Literal["fast", "exact"]): Mode.

    Returns:
        np.ndarray: Glass blurred image.

    """
    x = cv2.GaussianBlur(np.array(img), sigmaX=sigma, ksize=(0, 0))

    if mode == "fast":
        hs = np.arange(img.shape[0] - max_delta, max_delta, -1)
        ws = np.arange(img.shape[1] - max_delta, max_delta, -1)
        h: int | np.ndarray = np.tile(hs, ws.shape[0])
        w: int | np.ndarray = np.repeat(ws, hs.shape[0])

        for i in range(iterations):
            dy = dxy[:, i, 0]
            dx = dxy[:, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]

    elif mode == "exact":
        for ind, (i, h, w) in enumerate(
            product(
                range(iterations),
                range(img.shape[0] - max_delta, max_delta, -1),
                range(img.shape[1] - max_delta, max_delta, -1),
            ),
        ):
            idx = ind if ind < len(dxy) else ind % len(dxy)
            dy = dxy[idx, i, 0]
            dx = dxy[idx, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]
    else:
        raise ValueError(f"Unsupported mode `{mode}`. Supports only `fast` and `exact`.")

    return cv2.GaussianBlur(x, sigmaX=sigma, ksize=(0, 0))


def defocus(img: np.ndarray, radius: int, alias_blur: float) -> np.ndarray:
    """Defocus an image.

    This function defocuses an image.

    Args:
        img (np.ndarray): Input image.
        radius (int): Radius.
        alias_blur (float): Alias blur.

    Returns:
        np.ndarray: Defocused image.

    """
    length = np.arange(-max(8, radius), max(8, radius) + 1)
    ksize = 3 if radius <= EIGHT else 5

    x, y = np.meshgrid(length, length)
    aliased_disk = np.array((x**2 + y**2) <= radius**2, dtype=np.float32)
    aliased_disk /= np.sum(aliased_disk)

    kernel = cv2.GaussianBlur(aliased_disk, (ksize, ksize), sigmaX=alias_blur)

    return convolve(img, kernel=kernel)


def central_zoom(img: np.ndarray, zoom_factor: int) -> np.ndarray:
    """Central zoom an image.

    This function zooms an image.

    Args:
        img (np.ndarray): Input image.
        zoom_factor (int): Zoom factor.

    Returns:
        np.ndarray: Zoomed image.

    """
    height, width = img.shape[:2]
    h_ch, w_ch = ceil(height / zoom_factor), ceil(width / zoom_factor)
    h_top, w_top = (height - h_ch) // 2, (width - w_ch) // 2

    img = scale(img[h_top : h_top + h_ch, w_top : w_top + w_ch], zoom_factor, cv2.INTER_LINEAR)
    h_trim_top, w_trim_top = (img.shape[0] - height) // 2, (img.shape[1] - width) // 2
    return img[h_trim_top : h_trim_top + height, w_trim_top : w_trim_top + width]


@float32_io
@clipped
def zoom_blur(img: np.ndarray, zoom_factors: np.ndarray | Sequence[int]) -> np.ndarray:
    """Zoom blur an image.

    This function zooms and blurs an image.

    Args:
        img (np.ndarray): Input image.
        zoom_factors (np.ndarray | Sequence[int]): Zoom factors.

    Returns:
        np.ndarray: Zoomed and blurred image.

    """
    out = np.zeros_like(img, dtype=np.float32)

    for zoom_factor in zoom_factors:
        out += central_zoom(img, zoom_factor)

    return (img + out) / (len(zoom_factors) + 1)


def _ensure_min_value(result: tuple[int, int], min_value: int, field_name: str | None) -> tuple[int, int]:
    if result[0] < min_value or result[1] < min_value:
        new_result = (max(min_value, result[0]), max(min_value, result[1]))
        warn(
            f"{field_name}: Invalid kernel size range {result}. "
            f"Values less than {min_value} are not allowed. "
            f"Range automatically adjusted to {new_result}.",
            UserWarning,
            stacklevel=2,
        )
        return new_result
    return result


def _ensure_odd_values(result: tuple[int, int], field_name: str | None = None) -> tuple[int, int]:
    new_result = (
        result[0] if result[0] == 0 or result[0] % 2 == 1 else result[0] + 1,
        result[1] if result[1] == 0 or result[1] % 2 == 1 else result[1] + 1,
    )
    if new_result != result:
        warn(
            f"{field_name}: Non-zero kernel sizes must be odd. Range {result} automatically adjusted to {new_result}.",
            UserWarning,
            stacklevel=2,
        )
    return new_result


def process_blur_limit(value: int | tuple[int, int], info: ValidationInfo, min_value: int = 0) -> tuple[int, int]:
    """Process blur limit to ensure valid kernel sizes."""
    # Convert value to tuple[int, int]
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError("Sequence must contain exactly 2 elements")
        result = (int(value[0]), int(value[1]))
    else:
        result = (min_value, int(value))

    result = _ensure_min_value(result, min_value, info.field_name)
    result = _ensure_odd_values(result, info.field_name)

    if result[0] > result[1]:
        final_result = (result[1], result[1])
        warn(
            f"{info.field_name}: Invalid range {result} (min > max). Range automatically adjusted to {final_result}.",
            UserWarning,
            stacklevel=2,
        )
        return final_result

    return result


def create_motion_kernel(
    kernel_size: int,
    angle: float,
    direction: float,
    allow_shifted: bool,
    random_state: random.Random,
) -> np.ndarray:
    """Create a motion blur kernel.

    Args:
        kernel_size (int): Size of the kernel (must be odd)
        angle (float): Angle in degrees (counter-clockwise)
        direction (float): Blur direction (-1.0 to 1.0)
        allow_shifted (bool): Allow kernel to be randomly shifted from center
        random_state (random.Random): Python's random.Random instance

    Returns:
        np.ndarray: Motion blur kernel

    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Calculate direction vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # Create line points with direction bias
    line_length = kernel_size // 2
    t = np.linspace(-line_length, line_length, kernel_size * 2)

    # Apply direction bias
    if direction != 0:
        t = t * (1 + abs(direction))
        if direction < 0:
            t = t * -1

    # Generate line coordinates
    x = center + dx * t
    y = center + dy * t

    # Apply random shift if allowed
    if allow_shifted and random_state is not None:
        shift_x = random_state.uniform(-1, 1) * line_length / 2
        shift_y = random_state.uniform(-1, 1) * line_length / 2
        x += shift_x
        y += shift_y

    # Round coordinates and clip to kernel bounds
    x = np.clip(np.round(x), 0, kernel_size - 1).astype(int)
    y = np.clip(np.round(y), 0, kernel_size - 1).astype(int)

    # Keep only unique points to avoid multiple assignments
    points = np.unique(np.column_stack([y, x]), axis=0)
    kernel[points[:, 0], points[:, 1]] = 1

    # Ensure at least one point is set
    if not kernel.any():
        kernel[center, center] = 1

    return kernel


def sample_odd_from_range(random_state: random.Random, low: int, high: int) -> int:
    """Sample an odd number from the range [low, high] (inclusive).

    Args:
        random_state (random.Random): instance of random.Random
        low (int): lower bound (will be converted to nearest valid odd number)
        high (int): upper bound (will be converted to nearest valid odd number)

    Returns:
        int: Randomly sampled odd number from the range

    Note:
        - Input values will be converted to nearest valid odd numbers:
          * Values less than 3 will become 3
          * Even values will be rounded up to next odd number
        - After normalization, high must be >= low

    """
    # Normalize low value
    low = max(3, low + (low % 2 == 0))
    # Normalize high value
    high = max(3, high + (high % 2 == 0))

    # Ensure high >= low after normalization
    high = max(high, low)

    if low == high:
        return low

    # Calculate number of possible odd values
    num_odd_values = (high - low) // 2 + 1
    # Generate random index and convert to corresponding odd number
    rand_idx = random_state.randint(0, num_odd_values - 1)
    return low + (2 * rand_idx)


def create_gaussian_kernel(sigma: float, ksize: int = 0) -> np.ndarray:
    """Create a Gaussian kernel following PIL's approach.

    Args:
        sigma (float): Standard deviation for Gaussian kernel.
        ksize (int): Kernel size. If 0, size is computed as int(sigma * 3.5) * 2 + 1
               to match PIL's implementation. Otherwise, must be positive and odd.

    Returns:
        np.ndarray: 2D normalized Gaussian kernel.

    """
    # PIL's kernel creation approach
    size = int(sigma * 3.5) * 2 + 1 if ksize == 0 else ksize

    # Ensure odd size
    size = size + 1 if size % 2 == 0 else size

    # Create x coordinates
    x = np.linspace(-(size // 2), size // 2, size)

    # Compute 1D kernel using vectorized operations
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D kernel
    return kernel_1d[:, np.newaxis] @ kernel_1d[np.newaxis, :]


def create_gaussian_kernel_1d(sigma: float, ksize: int = 0) -> np.ndarray:
    """Create a 1D Gaussian kernel following PIL's approach.

    Args:
        sigma (float): Standard deviation for Gaussian kernel.
        ksize (int): Kernel size. If 0, size is computed as int(sigma * 3.5) * 2 + 1
               to match PIL's implementation. Otherwise, must be positive and odd.

    Returns:
        np.ndarray: 1D normalized Gaussian kernel.

    """
    # PIL's kernel creation approach
    size = int(sigma * 3.5) * 2 + 1 if ksize == 0 else ksize

    # Ensure odd size
    size = size + 1 if size % 2 == 0 else size

    # Create x coordinates
    x = create_gaussian_kernel_input_array(size=size)

    # Compute 1D kernel using vectorized operations
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel_1d / kernel_1d.sum()


def create_gaussian_kernel_input_array(size: int) -> np.ndarray:
    """Creates a 1-D array which will create an array of x-coordinates which will be input for the
    gaussian function (values from -size/2 to size/2 with step size of 1)

    Piecewise function is needed as equivalent python list comprehension is faster than np.linspace
    for values of size < 100

    Args:
        size (int): kernel size

    Returns:
        np.ndarray: x-coordinate array which will be input for gaussian function that will be used for
        separable gaussian blur

    """
    if size < 100:
        return np.array(list(range(-(size // 2), (size // 2) + 1, 1)))

    return np.linspace(-(size // 2), size // 2, size)
