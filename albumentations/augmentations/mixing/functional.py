"""Functional implementations for image mixing operations.

This module provides utility functions for blending and combining images,
such as copy-and-paste operations with masking.
"""

from __future__ import annotations

import numpy as np
from albucore import get_num_channels

import albumentations.augmentations.geometric.functional as fgeometric

__all__ = ["copy_and_paste_blend", "create_2x2_mosaic_image"]


def copy_and_paste_blend(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    overlay_mask: np.ndarray,
    offset: tuple[int, int],
) -> np.ndarray:
    """Blend images by copying pixels from an overlay image to a base image using a mask.

    This function copies pixels from the overlay image to the base image only where
    the mask has non-zero values. The overlay is placed at the specified offset
    from the top-left corner of the base image.

    Args:
        base_image (np.ndarray): The destination image that will be modified.
        overlay_image (np.ndarray): The source image containing pixels to copy.
        overlay_mask (np.ndarray): Binary mask indicating which pixels to copy from the overlay.
            Pixels are copied where mask > 0.
        offset (tuple[int, int]): The (y, x) offset specifying where to place the
            top-left corner of the overlay relative to the base image.

    Returns:
        np.ndarray: The blended image with the overlay applied to the base image.

    """
    y_offset, x_offset = offset

    blended_image = base_image.copy()
    mask_indices = np.where(overlay_mask > 0)
    blended_image[mask_indices[0] + y_offset, mask_indices[1] + x_offset] = overlay_image[
        mask_indices[0],
        mask_indices[1],
    ]
    return blended_image


def create_2x2_mosaic_image(
    four_images: list[np.ndarray],
    center_pt: tuple[int, int],
    mosaic_size: tuple[int, int],
    keep_aspect_ratio: bool,
    interpolation: int,
    fill: tuple[float, ...] | float,
) -> np.ndarray:
    target_h, target_w = mosaic_size
    num_channels = get_num_channels(four_images[0])
    quadrant_shape = (target_h, target_w, num_channels) if num_channels > 1 else (target_h, target_w)
    mosaic_shape = (2 * target_h, 2 * target_w, num_channels) if num_channels > 1 else (2 * target_h, 2 * target_w)
    fill_value = fgeometric.extend_value(fill, num_channels)

    # Resize images to match mosaic quadrant size
    if not keep_aspect_ratio:
        # Direct resize (distorts aspect ratio)
        resized_images = [fgeometric.resize(img, (target_h, target_w), interpolation) for img in four_images]
    else:
        # Preserve aspect ratio: scale to fit within (target_h, target_w), then pad
        resized_images = []
        for idx, img in enumerate(four_images):
            h, w = img.shape[:2]
            scale = min(target_h / h, target_w / w)
            resized_img = fgeometric.scale(img, scale, interpolation)
            resized_h, resized_w = resized_img.shape[:2]

            # Create a blank canvas and place the resized image in the good position
            canvas = np.full(
                quadrant_shape,
                fill_value,
                dtype=img.dtype,
            )  # We checked that all images have same type before
            if idx == 0:  # Top-left quadrant -> resized image should be aligned on bottom-right corner
                top = max(0, target_h - resized_h)
                left = max(0, target_w - resized_w)
            elif idx == 1:  # Top-right quadrant -> resized image should be aligned on bottom-left corner
                top = max(0, target_h - resized_h)
                left = 0
            elif idx == 2:  # Bottom-left quadrant -> resized image should be aligned on top-right corner
                top = 0
                left = max(0, target_w - resized_w)
            else:  # Bottom-right quadrant -> resized image should be aligned on top-left corner
                top = 0
                left = 0
            canvas[top : top + resized_h, left : left + resized_w] = resized_img

            resized_images.append(canvas)

    # Create 2x2 mosaic
    mosaic = np.zeros(mosaic_shape, dtype=resized_images[0].dtype)
    # Place images in quadrants
    mosaic[:target_h, :target_w] = resized_images[0]  # Top-left quadrant
    mosaic[:target_h, target_w:] = resized_images[1]  # Top-right quadrant
    mosaic[target_h:, :target_w] = resized_images[2]  # Bottom-left quadrant
    mosaic[target_h:, target_w:] = resized_images[3]  # Bottom-right quadrant

    # Extract the final (target_h, target_w) crop centered at center_pt
    x1, y1 = max(0, center_pt[0] - target_w // 2), max(0, center_pt[1] - target_h // 2)
    x2, y2 = x1 + target_w, y1 + target_h
    # Ensure the crop is within bounds
    x1, x2 = min(x1, target_w), min(x2, 2 * target_w)
    y1, y2 = min(y1, target_h), min(y2, 2 * target_h)

    return mosaic[y1:y2, x1:x2]
