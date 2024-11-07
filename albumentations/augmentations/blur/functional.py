from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import ceil
from typing import Literal
from warnings import warn

import cv2
import numpy as np
from albucore import clipped, float32_io, maybe_process_in_chunks, preserve_channel_dim
from pydantic import ValidationInfo

from albumentations.augmentations.functional import convolve
from albumentations.augmentations.geometric.functional import scale
from albumentations.core.types import EIGHT, ScaleIntType

__all__ = ["blur", "median_blur", "gaussian_blur", "glass_blur", "defocus", "central_zoom", "zoom_blur"]


@preserve_channel_dim
def blur(img: np.ndarray, ksize: int) -> np.ndarray:
    blur_fn = maybe_process_in_chunks(cv2.blur, ksize=(ksize, ksize))
    return blur_fn(img)


@preserve_channel_dim
def median_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    if img.dtype == np.float32 and ksize not in {3, 5}:
        raise ValueError(f"Invalid ksize value {ksize}. For a float32 image the only valid ksize values are 3 and 5")

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
    result = (min_value, value) if not isinstance(value, Sequence) else value

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
