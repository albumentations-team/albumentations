import numpy as np
import pytest
from albumentations.augmentations.mixing.functional import mix_arrays

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
    "array1, array2, mix_coef, expected_shape",
    [
        # Test case 1: Both arrays are of shape (100, 100, 3)
        (np.zeros((100, 100, 3)), np.ones((100, 100, 3)), 0.5, (100, 100, 3)),

        # Test case 2: First array is of shape (100, 100, 1), second array is of shape (100, 100)
        (np.zeros((100, 100, 1)), np.ones((100, 100)), 0.5, (100, 100, 1)),

        # Test case 3: First array is of shape (100, 100), second array is of shape (100, 100, 1)
        (np.zeros((100, 100)), np.ones((100, 100, 1)), 0.5, (100, 100)),

        # Additional test cases can be added here.
    ]
)
def test_mix_arrays_shapes(array1, array2, mix_coef, expected_shape):
    result = mix_arrays(array1, array2, mix_coef)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Additionally, you can check if the result type matches the first array type
    assert result.dtype == array1.dtype, f"Expected dtype {array1.dtype}, got {result.dtype}"

    deduced_mix_coef = find_mix_coef(result, array1, array2)

    assert mix_coef == deduced_mix_coef
