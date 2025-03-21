"""Functional implementations for image mixing operations.

This module provides utility functions for blending and combining images,
such as copy-and-paste operations with masking.
"""

from __future__ import annotations

import numpy as np

__all__ = ["copy_and_paste_blend"]


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
