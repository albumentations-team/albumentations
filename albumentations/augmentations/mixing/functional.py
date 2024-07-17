from __future__ import annotations

import numpy as np


def copy_and_paste_blend(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    mask: np.ndarray | None = None,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    y_offset, x_offset = offset

    if mask is None:
        # Create a default mask with the same shape as overlay_image
        mask = np.ones_like(overlay_image)

    blended_image = base_image.copy()
    mask_indices = np.where(mask > 0)
    blended_image[mask_indices[0] + y_offset, mask_indices[1] + x_offset] = overlay_image[
        mask_indices[0],
        mask_indices[1],
    ]
    return blended_image
