import numpy as np
import pytest
from albumentations.augmentations.mixing import functional as fmixing

def find_mix_coef(r: np.ndarray, array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Finds the mixing coefficient used to combine array1 and array2 into r,
    based on the cv2.addWeighted operation.

    Args:
        r (np.ndarray): The resulting array.
        array1 (np.ndarray): The first array.
        array2 (np.ndarray): The second array.

    Returns:
        float: The estimated mixing coefficient
    """
    r = r.astype(np.float64)
    array1 = array1.astype(np.float64)
    array2 = array2.reshape(array1.shape).astype(np.float64)

    denominator = array1 - array2
    valid = denominator != 0  # Mask of valid positions where denominator is not zero

    # Initialize mix_coef array full of NaNs for invalid divisions or full of zeros
    mix_coef = np.full_like(r, np.nan)  # Or np.full_like(r, np.nan) if you prefer NaN for invalid/indeterminate values

    # Compute mix_coef only where valid
    mix_coef[valid] = (r[valid] - array2[valid]) / denominator[valid]

    return mix_coef[valid].mean()


@pytest.mark.parametrize("base_image, overlay_image, mask, offset, expected_shape, expected_comparison", [
    (
        np.ones((200, 200, 3), dtype=np.uint8) * 255,
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.ones((100, 100), dtype=np.uint8) * 255,
        (50, 50),
        (200, 200, 3),
        lambda result, base_image, overlay_image, mask: np.array_equal(result[50:150, 50:150][mask > 0], overlay_image[mask > 0])
    ),
    (
        np.ones((200, 200, 3), dtype=np.uint8) * 255,
        np.zeros((100, 100, 3), dtype=np.uint8),
        None,
        (50, 50),
        (200, 200, 3),
        lambda result, base_image, overlay_image, _: np.all(result[50:150, 50:150] != base_image[50:150, 50:150])
    ),
])
def test_copy_and_paste_blend(base_image, overlay_image, mask, offset, expected_shape, expected_comparison):
    if mask is None:
        mask = np.ones_like(overlay_image[:, :, 0])
    result = fmixing.copy_and_paste_blend(base_image, overlay_image, mask, offset)
    assert result.shape == expected_shape
    assert expected_comparison(result, base_image, overlay_image, mask)
