from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import ceil
from random import Random
from typing import Literal
from warnings import warn

import cv2
import numpy as np
from albucore import clipped, float32_io, maybe_process_in_chunks, preserve_channel_dim, uint8_io
from pydantic import ValidationInfo

from albumentations.augmentations.functional import convolve
from albumentations.augmentations.geometric.functional import scale
from albumentations.core.types import EIGHT, ScaleIntType

__all__ = ["blur", "central_zoom", "defocus", "gaussian_blur", "glass_blur", "median_blur", "zoom_blur"]


@preserve_channel_dim
def blur(img: np.ndarray, ksize: int) -> np.ndarray:
    blur_fn = maybe_process_in_chunks(cv2.blur, ksize=(ksize, ksize))
    return blur_fn(img)


@preserve_channel_dim
@uint8_io
def median_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    blur_fn = maybe_process_in_chunks(cv2.medianBlur, ksize=ksize)
    return blur_fn(img)


@preserve_channel_dim
def gaussian_blur(img: np.ndarray, ksize: int, sigma: float = 0) -> np.ndarray:
    # When sigma=0, it is computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`
    blur_fn = maybe_process_in_chunks(cv2.GaussianBlur, ksize=(ksize, ksize), sigmaX=sigma)
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
    length = np.arange(-max(8, radius), max(8, radius) + 1)
    ksize = 3 if radius <= EIGHT else 5

    x, y = np.meshgrid(length, length)
    aliased_disk = np.array((x**2 + y**2) <= radius**2, dtype=np.float32)
    aliased_disk /= np.sum(aliased_disk)

    kernel = gaussian_blur(aliased_disk, ksize, sigma=alias_blur)
    return convolve(img, kernel=kernel)


def central_zoom(img: np.ndarray, zoom_factor: int) -> np.ndarray:
    height, width = img.shape[:2]
    h_ch, w_ch = ceil(height / zoom_factor), ceil(width / zoom_factor)
    h_top, w_top = (height - h_ch) // 2, (width - w_ch) // 2

    img = scale(img[h_top : h_top + h_ch, w_top : w_top + w_ch], zoom_factor, cv2.INTER_LINEAR)
    h_trim_top, w_trim_top = (img.shape[0] - height) // 2, (img.shape[1] - width) // 2
    return img[h_trim_top : h_trim_top + height, w_trim_top : w_trim_top + width]


@float32_io
@clipped
def zoom_blur(img: np.ndarray, zoom_factors: np.ndarray | Sequence[int]) -> np.ndarray:
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
            f"{field_name}: Non-zero kernel sizes must be odd. "
            f"Range {result} automatically adjusted to {new_result}.",
            UserWarning,
            stacklevel=2,
        )
    return new_result


def process_blur_limit(value: ScaleIntType, info: ValidationInfo, min_value: int = 0) -> tuple[int, int]:
    """Process blur limit to ensure valid kernel sizes."""
    result = value if isinstance(value, Sequence) else (min_value, value)

    result = _ensure_min_value(result, min_value, info.field_name)
    result = _ensure_odd_values(result, info.field_name)

    if result[0] > result[1]:
        final_result = (result[1], result[1])
        warn(
            f"{info.field_name}: Invalid range {result} (min > max). "
            f"Range automatically adjusted to {final_result}.",
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
    random_state: Random,
) -> np.ndarray:
    """Create a motion blur kernel.

    Args:
        kernel_size: Size of the kernel (must be odd)
        angle: Angle in degrees (counter-clockwise)
        direction: Blur direction (-1.0 to 1.0)
        allow_shifted: Allow kernel to be randomly shifted from center
        random_state: Python's random.Random instance

    Returns:
        Motion blur kernel
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


def sample_odd_from_range(random_state: Random, low: int, high: int) -> int:
    """Sample an odd number from the range [low, high] (inclusive).

    Args:
        random_state: instance of random.Random
        low: lower bound (will be converted to nearest valid odd number)
        high: upper bound (will be converted to nearest valid odd number)

    Returns:
        Randomly sampled odd number from the range

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
