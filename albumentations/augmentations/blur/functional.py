from itertools import product
from math import ceil
from typing import Sequence, Union

import cv2
import numpy as np

from albumentations.augmentations.functional import convolve
from albumentations.augmentations.geometric.functional import scale
from albumentations.augmentations.utils import (
    _maybe_process_in_chunks,
    clipped,
    preserve_shape,
)

__all__ = ["blur", "median_blur", "gaussian_blur", "glass_blur"]


@preserve_shape
def blur(img: np.ndarray, ksize: int) -> np.ndarray:
    blur_fn = _maybe_process_in_chunks(cv2.blur, ksize=(ksize, ksize))
    return blur_fn(img)


@preserve_shape
def median_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    if img.dtype == np.float32 and ksize not in {3, 5}:
        raise ValueError(f"Invalid ksize value {ksize}. For a float32 image the only valid ksize values are 3 and 5")

    blur_fn = _maybe_process_in_chunks(cv2.medianBlur, ksize=ksize)
    return blur_fn(img)


@preserve_shape
def gaussian_blur(img: np.ndarray, ksize: int, sigma: float = 0) -> np.ndarray:
    # When sigma=0, it is computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`
    blur_fn = _maybe_process_in_chunks(cv2.GaussianBlur, ksize=(ksize, ksize), sigmaX=sigma)
    return blur_fn(img)


@preserve_shape
def glass_blur(
    img: np.ndarray, sigma: float, max_delta: int, iterations: int, dxy: np.ndarray, mode: str
) -> np.ndarray:
    x = cv2.GaussianBlur(np.array(img), sigmaX=sigma, ksize=(0, 0))

    if mode == "fast":
        hs = np.arange(img.shape[0] - max_delta, max_delta, -1)
        ws = np.arange(img.shape[1] - max_delta, max_delta, -1)
        h: Union[int, np.ndarray] = np.tile(hs, ws.shape[0])
        w: Union[int, np.ndarray] = np.repeat(ws, hs.shape[0])

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
            )
        ):
            ind = ind if ind < len(dxy) else ind % len(dxy)
            dy = dxy[ind, i, 0]
            dx = dxy[ind, i, 1]
            x[h, w], x[h + dy, w + dx] = x[h + dy, w + dx], x[h, w]
    else:
        ValueError(f"Unsupported mode `{mode}`. Supports only `fast` and `exact`.")

    return cv2.GaussianBlur(x, sigmaX=sigma, ksize=(0, 0))


def defocus(img: np.ndarray, radius: int, alias_blur: float) -> np.ndarray:
    length = np.arange(-max(8, radius), max(8, radius) + 1)
    ksize = 3 if radius <= 8 else 5

    x, y = np.meshgrid(length, length)
    aliased_disk = np.array((x**2 + y**2) <= radius**2, dtype=np.float32)
    aliased_disk /= np.sum(aliased_disk)

    kernel = gaussian_blur(aliased_disk, ksize, sigma=alias_blur)
    return convolve(img, kernel=kernel)


def central_zoom(img: np.ndarray, zoom_factor: int) -> np.ndarray:
    h, w = img.shape[:2]
    h_ch, w_ch = ceil(h / zoom_factor), ceil(w / zoom_factor)
    h_top, w_top = (h - h_ch) // 2, (w - w_ch) // 2

    img = scale(img[h_top : h_top + h_ch, w_top : w_top + w_ch], zoom_factor, cv2.INTER_LINEAR)
    h_trim_top, w_trim_top = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    return img[h_trim_top : h_trim_top + h, w_trim_top : w_trim_top + w]


@clipped
def zoom_blur(img: np.ndarray, zoom_factors: Union[np.ndarray, Sequence[int]]) -> np.ndarray:
    out = np.zeros_like(img, dtype=np.float32)
    for zoom_factor in zoom_factors:
        out += central_zoom(img, zoom_factor)

    img = ((img + out) / (len(zoom_factors) + 1)).astype(img.dtype)

    return img
