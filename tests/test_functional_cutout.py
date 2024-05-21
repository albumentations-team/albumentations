import numpy as np
import pytest

from albumentations.augmentations.dropout.functional import cutout
from albucore.utils import MAX_VALUES_BY_DTYPE
from tests.utils import set_seed


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


@pytest.mark.parametrize("dtype, max_value", [
    (np.uint8, MAX_VALUES_BY_DTYPE[np.uint8]),
    (np.uint16, MAX_VALUES_BY_DTYPE[np.uint16]),
    (np.uint32, MAX_VALUES_BY_DTYPE[np.uint32]),
    (np.float32, MAX_VALUES_BY_DTYPE[np.float32]),
])
@pytest.mark.parametrize("shape", [
    (100, 100),
    (100, 100, 1),
    (100, 100, 3),
    (100, 100, 7),
])
@pytest.mark.parametrize("fill_type", [
    "random",
    "single_value",
    "channel_specific",
])
def test_cutout_various_types_and_fills(dtype, max_value, shape, fill_type):
    set_seed(0)
    img = np.zeros(shape, dtype=dtype)
    holes = [(10, 10, 50, 50)]

    if fill_type == "random":
        fill_value = "random"
    elif fill_type == "single_value":
        fill_value = max_value if dtype != np.float32 else 0.5  # Use middle value for float32
    elif fill_type == "channel_specific":
        if len(shape) == 2:  # Grayscale image, no channel dimension
            fill_value = [max_value] if dtype != np.float32 else [0.5]
        else:
            fill_value = [i % max_value for i in range(shape[2])] if dtype != np.float32 else [(i / shape[2]) for i in range(shape[2])]

    result_img = cutout(img, holes, fill_value)

    if fill_type == "random":
        assert result_img.dtype == dtype
        # Check if the hole is not all zeros
        assert not np.all(result_img[10:50, 10:50] == 0)
        assert result_img[10:50, 10:50].mean() == pytest.approx(max_value / 2, abs=0.05 * max_value)
        assert result_img[10:50, 10:50].max() == pytest.approx(max_value, abs=0.05 * max_value)
        assert result_img[10:50, 10:50].min() == pytest.approx(0, abs=0.05 * max_value)
    else:
        if isinstance(fill_value, (list, tuple)):
            expected_fill_value = np.array(fill_value, dtype=dtype)
        else:
            expected_fill_value = np.array([fill_value] * img.shape[-1], dtype=dtype)

        # Ensure the hole has the correct fill value
        if len(shape) == 2:  # Handle no channel dimension in grayscale
            assert np.all(result_img[10:50, 10:50] == expected_fill_value[0])
        else:
            for channel_index in range(result_img.shape[-1]):
                assert np.all(result_img[10:50, 10:50, channel_index] == expected_fill_value[channel_index])
