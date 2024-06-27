import random
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "copy_and_paste_blend",
    "mask2bbox",
    "calculate_offsets",
]


def copy_and_paste_blend(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    offset: Tuple[int, int],
    overlay_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if overlay_image.size == 0:
        return base_image

    x_offset, y_offset = offset

    blended_image = base_image.copy()
    overlay_height, overlay_width = overlay_image.shape[:2]

    if overlay_mask is None:
        x_min, y_min = x_offset, y_offset
        x_max, y_max = x_min + overlay_width, y_min + overlay_height
        blended_image[y_min:y_max, x_min:x_max] = overlay_image
    else:
        mask_indices = np.where(overlay_mask > 0)
        blended_image[mask_indices[0] + y_offset, mask_indices[1] + x_offset] = overlay_image[
            mask_indices[0],
            mask_indices[1],
        ]
    return blended_image


def mask2bbox(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    # Find the coordinates of the non-zero elements
    y_indices, x_indices = np.where(binary_mask)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return 0, 0, 0, 0

    # Get the minimum and maximum coordinates
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    return x_min, y_min, x_max, y_max


def calculate_offsets(
    image_shape: Sequence[int],
    overlay_shape: Sequence[int],
) -> Tuple[int, int]:
    image_height, image_width = image_shape[:2]
    overlay_height, overlay_width = overlay_shape[:2]
    max_x_offset = image_width - overlay_width
    max_y_offset = image_height - overlay_height
    offset_x = random.randint(0, max_x_offset)
    offset_y = random.randint(0, max_y_offset)
    return offset_x, offset_y
