import numpy as np
import pytest
from albumentations.augmentations.mixing.functional import mix_arrays

@pytest.mark.parametrize(
    "array1, array2, lam, expected_shape",
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
def test_mix_arrays_shapes(array1, array2, lam, expected_shape):
    result = mix_arrays(array1, array2, lam)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"

    # Additionally, you can check if the result type matches the first array type
    assert result.dtype == array1.dtype, f"Expected dtype {array1.dtype}, got {result.dtype}"
