import numpy as np
import pytest

from albumentations.augmentations.dropout.functional import cutout


@pytest.mark.parametrize(
    "img, fill_value",
    [
        # Single-channel image, fill_value is a number
        (np.zeros((10, 10), dtype=np.uint8), 255),
        # Multi-channel image with different channel counts, fill_value is a number (applied to all channels)
        (np.zeros((10, 10, 3), dtype=np.uint8), 255),
        # Multi-channel image, fill_value is a tuple with different values for different channels
        (np.zeros((10, 10, 3), dtype=np.uint8), (128, 128, 128)),
        # Multi-channel image, fill_value as list with different values
        (np.zeros((10, 10, 2), dtype=np.uint8), [64, 192]),
        # Multi-channel image, fill_value as np.ndarray with different values
        (np.zeros((10, 10, 3), dtype=np.uint8), np.array([32, 64, 96], dtype=np.uint8)),
    ],
)
def test_cutout_with_various_fill_values(img, fill_value):
    holes = [(2, 2, 5, 5)]
    result = cutout(img, holes, fill_value=fill_value)

    # Compute expected result
    expected_result = img.copy()
    for x1, y1, x2, y2 in holes:
        if isinstance(fill_value, (int, float)):
            fill_array = np.array(fill_value, dtype=img.dtype)
        else:
            fill_array = np.array(fill_value, dtype=img.dtype).reshape(-1)
        if img.ndim == 2:  # Single-channel image
            expected_result[y1:y2, x1:x2] = fill_array
        else:  # Multi-channel image
            fill_shape = (y2 - y1, x2 - x1, img.shape[2]) if img.ndim == 3 else (y2 - y1, x2 - x1)
            expected_fill = np.full(fill_shape, fill_array, dtype=img.dtype)
            expected_result[y1:y2, x1:x2] = expected_fill[: y2 - y1, : x2 - x1]

    # Check the filled values
    assert np.all(result == expected_result), "The result does not match the expected output."
