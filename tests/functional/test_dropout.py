import pytest
import numpy as np
from skimage.measure import label as ski_label
from albumentations.augmentations.dropout.functional import label as cv_label

from albucore import MAX_VALUES_BY_DTYPE

import albumentations.augmentations.dropout.functional as fdropout

from tests.utils import set_seed

@pytest.mark.parametrize("shape, dtype, connectivity", [
    ((8, 8), np.uint8, 1),
    ((10, 10), np.uint8, 2),
    ((12, 12), np.int32, 1),
    ((12, 12), np.int32, 2),
    ((14, 14), np.uint8, 1),
    ((35, 35), np.uint8, 2),
])
def test_label_function(shape, dtype, connectivity):
    set_seed(42)
    # Generate a random binary mask
    mask = np.random.randint(0, 2, shape).astype(dtype)

    # Compare results with scikit-image
    ski_result = ski_label(mask, connectivity=connectivity)
    cv_result = cv_label(mask, connectivity=connectivity)

    np.testing.assert_array_equal(cv_result, ski_result), "Label results do not match"

@pytest.mark.parametrize("shape, dtype, connectivity", [
    ((10, 10), np.uint8, 1),
    ((20, 20), np.int32, 2),
    ((30, 30), np.uint8, 1),
])
def test_label_function_return_num(shape, dtype, connectivity):
    mask = np.random.randint(0, 2, shape).astype(dtype)

    ski_result, ski_num = ski_label(mask, connectivity=connectivity, return_num=True)
    cv_result, cv_num = cv_label(mask, connectivity=connectivity, return_num=True)

    np.testing.assert_array_equal(cv_result, ski_result), "Label results do not match"
    assert ski_num == cv_num, "Number of labels do not match"

@pytest.mark.parametrize("shape, num_objects", [
    ((10, 10), 3),
    ((20, 20), 5),
    ((30, 30), 10),
])
def test_label_function_with_multiple_objects(shape, num_objects):
    set_seed(43)
    mask = np.zeros(shape, dtype=np.uint8)
    for i in range(1, num_objects + 1):
        x, y = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
        mask[x:x+3, y:y+3] = i

    ski_result, ski_num = ski_label(mask, return_num=True)
    cv_result, cv_num = cv_label(mask, return_num=True)

    # Check for one-to-one mapping
    combined = np.stack((ski_result, cv_result))
    unique_combinations = np.unique(combined.reshape(2, -1).T, axis=0)

    assert len(unique_combinations) == len(np.unique(ski_result)) == len(np.unique(cv_result)), \
        "Labels are not equal up to enumeration"

    assert ski_num == cv_num, "Number of labels do not match"
    assert cv_num == num_objects, f"Expected {num_objects} labels, got {cv_num}"

def test_label_function_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)

    ski_result, ski_num = ski_label(mask, return_num=True)
    cv_result, cv_num = cv_label(mask, return_num=True)

    np.testing.assert_array_equal(cv_result, ski_result), "Label results do not match for empty mask"
    assert ski_num == cv_num == 0, "Number of labels should be 0 for empty mask"

def test_label_function_full_mask():
    mask = np.ones((10, 10), dtype=np.uint8)

    ski_result, ski_num = ski_label(mask, return_num=True)
    cv_result, cv_num = cv_label(mask, return_num=True)

    np.testing.assert_array_equal(cv_result, ski_result), "Label results do not match for full mask"
    assert ski_num == cv_num, "Number of labels should be 2 for full mask (background + one object)"



@pytest.mark.parametrize(
    "img, fill",
    [
        # Single-channel image, fill is a number
        (np.zeros((10, 10), dtype=np.uint8), 255),
        # Multi-channel image with different channel counts, fill is a number (applied to all channels)
        (np.zeros((10, 10, 3), dtype=np.uint8), 255),
        # Multi-channel image, fill is a tuple with different values for different channels
        (np.zeros((10, 10, 3), dtype=np.uint8), (128, 128, 128)),
        # Multi-channel image, fill is a list with different values
        (np.zeros((10, 10, 2), dtype=np.uint8), [64, 192]),
        # Multi-channel image, fill is a np.ndarray with different values
        (np.zeros((10, 10, 3), dtype=np.uint8), np.array([32, 64, 96], dtype=np.uint8)),
        (np.zeros((10, 10, 3), dtype=np.uint8), np.array([[32], [64], [96]], dtype=np.uint8)),
        (np.zeros((10, 10, 3), dtype=np.uint8), np.array([32, 64, 96]).T),
    ],
)
def test_cutout_with_various_fill_values(img, fill):
    holes = [(2, 2, 5, 5)]
    generator = np.random.default_rng(42)
    result = fdropout.cutout(img, holes, fill, generator)

    # Compute expected result
    expected_result = img.copy()
    for x1, y1, x2, y2 in holes:
        if isinstance(fill, (int, float)):
            fill_array = np.array(fill, dtype=img.dtype)
        else:
            fill_array = np.array(fill, dtype=img.dtype).reshape(-1)
        if img.ndim == 2:  # Single-channel image
            expected_result[y1:y2, x1:x2] = fill_array
        else:  # Multi-channel image
            fill_shape = (y2 - y1, x2 - x1, img.shape[2]) if img.ndim == 3 else (y2 - y1, x2 - x1)
            expected_fill = np.full(fill_shape, fill_array, dtype=img.dtype)
            expected_result[y1:y2, x1:x2] = expected_fill[: y2 - y1, : x2 - x1]

    # Check the filled values
    assert np.all(result == expected_result), "The result does not match the expected output."


@pytest.mark.parametrize(
    "img_shape, fill",
    [
        ((10, 10), "random"),
        ((10, 10, 3), "random"),
        ((10, 10), "random_uniform"),
        ((10, 10, 3), "random_uniform"),
    ],
)
def test_cutout_with_random_fills(img_shape, fill):
    img = np.zeros(img_shape, dtype=np.uint8)
    holes = np.array([[2, 2, 5, 5]])
    generator = np.random.default_rng(42)

    result = fdropout.cutout(img, holes, fill, generator)


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
    generator = np.random.default_rng(42)

    img = np.zeros(shape, dtype=dtype)
    holes = [(10, 10, 50, 50)]

    if fill_type == "random":
        fill = "random"
    elif fill_type == "single_value":
        fill = max_value if dtype != np.float32 else 0.5  # Use middle value for float32
    elif fill_type == "channel_specific":
        if len(shape) == 2:  # Grayscale image, no channel dimension
            fill = [max_value] if dtype != np.float32 else [0.5]
        else:
            fill = (
                [i % max_value for i in range(shape[2])]
                if dtype != np.float32
                else [(i / shape[2]) for i in range(shape[2])]
            )

    result_img = fdropout.cutout(img, holes, fill, generator)

    if fill_type == "random":
        assert result_img.dtype == dtype
        # Check if the hole is not all zeros
        assert not np.all(result_img[10:50, 10:50] == 0)
        assert result_img[10:50, 10:50].mean() == pytest.approx(max_value / 2, abs=0.05 * max_value)
        assert result_img[10:50, 10:50].max() == pytest.approx(max_value, abs=0.05 * max_value)
        assert result_img[10:50, 10:50].min() == pytest.approx(0, abs=0.05 * max_value)
    else:
        if isinstance(fill, (list, tuple)):
            expected_fill = np.array(fill, dtype=dtype)
        else:
            expected_fill = np.array([fill] * img.shape[-1], dtype=dtype)

        # Ensure the hole has the correct fill value
        if len(shape) == 2:  # Handle no channel dimension in grayscale
            assert np.all(result_img[10:50, 10:50] == expected_fill[0])
        else:
            for channel_index in range(result_img.shape[-1]):
                assert np.all(result_img[10:50, 10:50, channel_index] == expected_fill[channel_index])


@pytest.mark.parametrize(
    ["regions", "expected"],
    [
        # Test case 1: Single scattered holes - return full region
        (
            np.array([[
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]]),
            np.array([[0, 0, 3, 3]])
        ),
        # Test case 2: Perfect visible rectangle in middle
        (
            np.array([[
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ]]),
            np.array([[1, 1, 3, 3]])
        ),
        # Test case 3: Multiple regions of different types
        (
            np.array([
                # Region 1: L-shaped visible region
                [
                    [0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 1],
                ],
                # Region 2: Single visible pixel
                [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ],
                # Region 3: Vertical slice
                [
                    [1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1],
                ],
            ]),
            np.array([
                [0, 0, 3, 3],  # Full region for L-shape
                [1, 1, 2, 2],  # Single pixel
                [1, 0, 2, 3],  # Vertical slice
            ])
        ),
        # Test case 4: Mix of fully covered and visible regions
        (
            np.array([
                # Region 1: Fully covered (3x3)
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                # Region 2: All visible (padded to 3x3)
                [
                    [0, 0, 1],
                    [0, 0, 1],
                    [1, 1, 1],
                ],
                # Region 3: Horizontal slice
                [
                    [1, 1, 1],
                    [0, 0, 0],
                    [1, 1, 1],
                ],
            ], dtype=np.uint8),
            np.array([
                [0, 0, 0, 0],  # Fully covered
                [0, 0, 2, 2],  # Visible region
                [0, 1, 3, 2],  # Horizontal slice
            ])
        ),
    ]
)
def test_find_region_coordinates_vectorized(regions, expected):
    """Test vectorized version of find_region_coordinates."""
    result = fdropout.find_region_coordinates(regions)
    np.testing.assert_array_equal(
        result, expected,
        err_msg=f"Expected {expected}, but got {result}"
    )


@pytest.mark.parametrize(
    ["boxes", "hole_mask", "expected"],
    [
        # Test case 1: Single box with perfect visible rectangle in middle
        (
            np.array([[10, 20, 14, 24]]),  # box: x1=10, y1=20, x2=14, y2=24
            np.zeros((100, 100), dtype=np.uint8),  # Full image visible
            np.array([[10, 20, 14, 24]])  # Expected: unchanged as no holes
        ),

        # Test case 2: Single box fully covered by hole
        (
            np.array([[5, 5, 8, 8]]),
            np.ones((100, 100), dtype=np.uint8),  # Full image covered
            np.array([[5, 5, 5, 5]])  # Box collapses to point when fully covered
        ),

        # Test case 3: Box partially covered by hole
        (
            np.array([[10, 10, 20, 20]]),
            np.zeros((100, 100), dtype=np.uint8).copy(),  # Start with all visible
            np.array([[10, 10, 15, 20]])  # Only left part visible after hole
        ),

        # Test case 4: Multiple boxes with different coverage
        (
            np.array([
                [0, 0, 10, 10],    # Box 1: fully visible
                [20, 20, 30, 30],  # Box 2: fully covered
                [40, 40, 50, 50],  # Box 3: partially covered
            ]),
            np.zeros((100, 100), dtype=np.uint8).copy(),  # Start with all visible
            np.array([
                [0, 0, 10, 10],     # Box 1: unchanged (no hole)
                [20, 20, 20, 20],   # Box 2: collapsed (fully covered)
                [40, 40, 45, 50],   # Box 3: width reduced (partially covered)
            ])
        ),
    ]
)
def test_resize_boxes_to_visible_area(boxes, hole_mask, expected):
    # Create holes in the mask for specific test cases
    if len(boxes) == 1 and np.array_equal(boxes[0], [10, 10, 20, 20]):
        # For test case 3: create hole in right half of the box
        hole_mask[10:20, 15:20] = 1

    elif len(boxes) == 3:
        # For test case 4: create specific holes
        hole_mask[20:30, 20:30] = 1  # Fully cover second box
        hole_mask[40:50, 45:50] = 1  # Partially cover third box (right part)

    result = fdropout.resize_boxes_to_visible_area(boxes, hole_mask)
    np.testing.assert_array_equal(
        result, expected,
        err_msg=f"Expected {expected}, but got {result}"
    )


@pytest.mark.parametrize(
    ["bboxes", "holes", "image_shape", "min_area", "min_visibility", "expected"],
    [
        # Test case 1: No boxes or holes
        (
            np.array([]),  # empty bboxes
            np.array([]),  # empty holes
            (100, 100),
            1.0,
            0.1,
            np.array([])
        ),

        # Test case 2: Single box, single hole - box remains visible
        (
            np.array([[10, 10, 30, 30]]),  # 20x20 box
            np.array([[15, 15, 20, 20]]),  # 5x5 hole in middle
            (100, 100),
            100,  # min area
            0.5,  # min visibility
            np.array([[10, 10, 30, 30]])  # box remains as >50% visible
        ),

        # Test case 3: Single box, single hole - box filtered out
        (
            np.array([[10, 10, 20, 20]]),  # 10x10 box
            np.array([[10, 10, 19, 19]]),  # 9x9 hole covering most of box
            (100, 100),
            50,    # min area
            0.5,   # min visibility
            np.array([], dtype=np.float32).reshape(0, 4)  # box removed as <50% visible
        ),

        # Test case 4: Multiple boxes, multiple holes
        (
            np.array([
                [10, 10, 20, 20],  # box 1: will be filtered out
                [30, 30, 40, 40],  # box 2: will remain
                [50, 50, 60, 60],  # box 3: will be resized
            ]),
            np.array([
                [10, 10, 19, 19],  # hole covering box 1
                [50, 50, 55, 60],  # hole covering half of box 3
            ]),
            (100, 100),
            25,    # min area
            0.3,   # min visibility
            np.array([
                [30, 30, 40, 40],  # box 2: unchanged
                [55, 50, 60, 60],  # box 3: resized to visible part
            ])
        ),

        # Test case 5: Edge cases with box sizes
         (
            np.array([
                [0, 0, 10, 10],     # box at edge
                [90, 90, 100, 100], # box at other edge
                [45, 45, 55, 55],   # box in middle
            ]),
            np.array([
                [0, 0, 5, 10],      # partial hole for first box
                [95, 95, 100, 100], # small hole for second box
                [45, 45, 55, 50],   # partial hole for middle box
            ]),
            (100, 100),
            20,    # min area
            0.4,   # min visibility
            np.array([
                [5, 0, 10, 10],      # first box resized from left
                [90, 90, 100, 100],  # second box unchanged (hole too small)
                [45, 50, 55, 55],    # middle box resized from top
            ])
        ),
    ]
)
def test_filter_bboxes_by_holes(bboxes, holes, image_shape, min_area, min_visibility, expected):
    if len(bboxes) > 0:
        bboxes = bboxes.astype(np.float32)
    if len(expected) > 0:
        expected = expected.astype(np.float32)

    result = fdropout.filter_bboxes_by_holes(bboxes, holes, image_shape, min_area, min_visibility)
    np.testing.assert_array_almost_equal(
        result, expected,
        err_msg=f"Expected {expected}, but got {result}"
    )
