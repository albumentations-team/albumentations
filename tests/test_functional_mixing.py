import numpy as np
import pytest
from albumentations.augmentations.functional import add_weighted


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
