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

@pytest.mark.parametrize(
    "base_image, overlay_image, offset, overlay_mask, expected",
    [
        # Basic blending without a mask
        (
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8) * 255,
            (1, 1),
            None,
            np.array([
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [255, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ], dtype=np.uint8)
        ),
        # Blending with a mask
        (
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8) * 255,
            (1, 1),
            np.array([
                [1, 0],
                [0, 1],
            ]),
            np.array([
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ], dtype=np.uint8)
        ),
        # Edge case: Empty overlay
        (
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.zeros((0, 0, 3), dtype=np.uint8),
            (1, 1),
            None,
            np.zeros((5, 5, 3), dtype=np.uint8)
        ),
    ]
)

def test_copy_and_paste_blend(base_image, overlay_image, offset, overlay_mask, expected):
    result = fmixing.copy_and_paste_blend(base_image, overlay_image, offset, overlay_mask)
    assert np.array_equal(result, expected), f"Expected:\n{expected}\nGot:\n{result}"
