from itertools import product

import cv2
import numpy as np

from albumentations.augmentations.functional import (
    _maybe_process_in_chunks,
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
        h = np.tile(hs, ws.shape[0])
        w = np.repeat(ws, hs.shape[0])

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

    return cv2.GaussianBlur(x, sigmaX=sigma, ksize=(0, 0))
