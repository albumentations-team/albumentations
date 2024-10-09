import numpy as np
import pytest
from albucore import MAX_VALUES_BY_DTYPE

from albumentations.augmentations.dropout.functional import cutout, filter_bboxes_by_holes
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


@pytest.mark.parametrize(
    "dtype, max_value",
    [
        (np.uint8, MAX_VALUES_BY_DTYPE[np.uint8]),
        (np.float32, MAX_VALUES_BY_DTYPE[np.float32]),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (100, 100),
        (100, 100, 1),
        (100, 100, 3),
        (100, 100, 7),
    ],
)
@pytest.mark.parametrize(
    "fill_type",
    [
        "random",
        "single_value",
        "channel_specific",
    ],
)
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
            fill_value = (
                [i % max_value for i in range(shape[2])]
                if dtype != np.float32
                else [(i / shape[2]) for i in range(shape[2])]
            )

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


@pytest.mark.parametrize(
    "bboxes, holes, image_shape, min_area, min_visibility, expected_bboxes",
    [
        # Test case 1: No intersection
        (
            np.array([[10, 10, 20, 20]]),
            np.array([[30, 30, 40, 40]]),
            (50, 50),
            100,
            0.5,
            np.array([[10, 10, 20, 20]]),
        ),
        # Test case 2: Small intersection
        (
            np.array([[10, 10, 30, 30]]),
            np.array([[25, 25, 35, 35]]),
            (50, 50),
            100,
            0.5,
            np.array([[10, 10, 30, 30]]),
        ),
        # Test case 3: Large intersection
        (
            np.array([[10, 10, 40, 40]]),
            np.array([[20, 20, 30, 30]]),
            (50, 50),
            100,
            0.5,
            np.array([[10, 10, 40, 40]]),
        ),
        # Test case 4: Multiple bboxes, some intersecting
        (
            np.array([[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]]),
            np.array([[15, 15, 25, 25], [45, 45, 55, 55]]),
            (100, 100),
            100,
            0.5,
            np.array([[30, 30, 40, 40]]),
        ),
        # Test case 5: Multiple holes
        (
            np.array([[10, 10, 30, 30], [40, 40, 60, 60]]),
            np.array([[15, 15, 25, 25], [45, 45, 55, 55]]),
            (100, 100),
            100,
            0.5,
            np.array([[10, 10, 30, 30], [40, 40, 60, 60]]),
        ),
        # Test case 6: Empty bboxes
        (
            np.array([]),
            np.array([[15, 15, 25, 25]]),
            (50, 50),
            100,
            0.5,
            np.array([]),
        ),
        # Test case 7: Empty holes
        (
            np.array([[10, 10, 20, 20]]),
            np.array([]),
            (50, 50),
            100,
            0.5,
            np.array([[10, 10, 20, 20]]),
        ),
        # Test case 8: Bbox exactly equal to min_area
        (
            np.array([[10, 10, 20, 20]]),
            np.array([[10, 10, 20, 20]]),
            (50, 50),
            100,
            0.5,
            np.array([]).reshape(0, 4),
        ),
        # Test case 9: High min_visibility
        (
            np.array([[10, 10, 30, 30]]),
            np.array([[15, 15, 25, 25]]),
            (50, 50),
            100,
            0.9,
            np.array([]).reshape(0, 4),
        ),
        # Test case 10: Low min_visibility
        (
            np.array([[10, 10, 30, 30]]),
            np.array([[15, 15, 25, 25]]),
            (50, 50),
            100,
            0.1,
            np.array([[10, 10, 30, 30]]),
        ),
    ],
)
def test_filter_bboxes_by_holes(bboxes, holes, image_shape, min_area, min_visibility, expected_bboxes):
    filtered_bboxes = filter_bboxes_by_holes(bboxes, holes, image_shape, min_area, min_visibility)
    np.testing.assert_array_equal(filtered_bboxes, expected_bboxes)


@pytest.mark.parametrize(
    "min_area, min_visibility, expected",
    [
        (50, 0.5, np.array([[10, 10, 30, 30]])),
        (150, 0.5, np.array([[10, 10, 30, 30]])),
        (310, 0.5, np.array([]).reshape(0, 4)),
        (50, 0.9, np.array([]).reshape(0, 4)),
        (50, 0.1, np.array([[10, 10, 30, 30]])),
    ],
)
def test_filter_bboxes_by_holes_different_params(min_area, min_visibility, expected):
    bboxes = np.array([[10, 10, 30, 30]])
    holes = np.array([[15, 15, 25, 25]])
    image_shape = (50, 50)
    filtered_bboxes = filter_bboxes_by_holes(bboxes, holes, image_shape, min_area, min_visibility)
    np.testing.assert_array_equal(filtered_bboxes, expected)


def test_filter_bboxes_by_holes_edge_cases():
    # Test with min_visibility = 0 (should keep all bboxes)
    bboxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
    holes = np.array([[15, 15, 25, 25]])
    image_shape = (50, 50)
    filtered_bboxes = filter_bboxes_by_holes(bboxes, holes, image_shape, min_area=1, min_visibility=0)
    np.testing.assert_array_equal(filtered_bboxes, bboxes)

    # Test with min_visibility = 1 (should remove all intersecting bboxes)
    filtered_bboxes = filter_bboxes_by_holes(bboxes, holes, image_shape, min_area=1, min_visibility=1)
    np.testing.assert_array_equal(filtered_bboxes, np.array([[30, 30, 40, 40]]))

    # Test with very large hole (should remove all bboxes)
    large_hole = np.array([[0, 0, 50, 50]])
    filtered_bboxes = filter_bboxes_by_holes(bboxes, large_hole, image_shape, min_area=1, min_visibility=0.1)
    np.testing.assert_array_equal(filtered_bboxes, np.array([]).reshape(0, 4))
