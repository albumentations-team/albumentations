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
    set_seed(137)
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
    generator = np.random.default_rng(137)
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
    generator = np.random.default_rng(137)

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
    generator = np.random.default_rng(137)

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
            np.zeros((0, 4), dtype=np.int32)  # Empty array with correct shape
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
                [40, 40, 45, 50],   # Box 3: width reduced (partially covered)
            ])
        ),
        # Test case 5: Boxes of different sizes
        (
            np.array([
                [10, 10, 15, 20],   # Tall box: 5x10
                [30, 30, 50, 40],   # Wide box: 20x10
                [60, 60, 80, 80],   # Square box: 20x20
                [90, 90, 92, 98],   # Very thin tall box: 2x8
            ]),
            np.zeros((100, 100), dtype=np.uint8).copy(),
            np.array([
                [10, 10, 15, 20],   # Tall box: unchanged
                [30, 30, 40, 40],   # Wide box: right part removed
                [60, 60, 70, 80],   # Square box: right part removed (not bottom)
                [90, 90, 92, 98],   # Thin tall box: unchanged
            ])
        ),
        # Test case 6: Overlapping holes on different sized boxes
        (
            np.array([
                [10, 10, 30, 20],   # Wide box: 20x10
                [40, 40, 45, 60],   # Tall box: 5x20
                [70, 70, 85, 85],   # Square box: 15x15
            ]),
            np.zeros((100, 100), dtype=np.uint8).copy(),  # Start with all visible
            np.array([
                [10, 10, 20, 20],   # Wide box: right part removed
                [40, 50, 45, 60],   # Tall box: top part removed
                [75, 70, 85, 80],   # Square box: left and bottom parts removed
            ])
        ),

        # Test case 7: Complex hole patterns on different sized boxes
        (
            np.array([
                [10, 10, 40, 20],   # Wide box: 30x10
                [50, 50, 55, 80],   # Tall box: 5x30
                [70, 70, 90, 90],   # Square box: 20x20
            ]),
            np.zeros((100, 100), dtype=np.uint8).copy(),  # Start with all visible
            np.array([
                [20, 10, 30, 20],   # Wide box: ends removed
                [50, 60, 55, 70],   # Tall box: top and bottom removed
                [75, 75, 85, 85],   # Square box: corners removed
            ])
        ),
    ]
)
def test_resize_boxes_to_visible_area(boxes, hole_mask, expected):
    # Create holes in the mask for specific test cases
    if len(boxes) == 1 and np.array_equal(boxes[0], [10, 10, 20, 20]):
        # For test case 3: create hole in right half of the box
        hole_mask[10:20, 15:20] = 1

    elif len(boxes) == 3 and np.array_equal(boxes[0], [0, 0, 10, 10]):
        # For test case 4: create specific holes
        hole_mask[20:30, 20:30] = 1  # Fully cover second box
        hole_mask[40:50, 45:50] = 1  # Partially cover third box (right part)

    elif len(boxes) == 4:  # Test case 5
        hole_mask[30:40, 40:50] = 1  # Remove right part of wide box
        hole_mask[60:80, 70:80] = 1  # Remove bottom part of square box

    elif len(boxes) == 3 and np.array_equal(boxes[0], [10, 10, 30, 20]):  # Test case 6
        hole_mask[10:20, 20:30] = 1  # Right part of wide box
        hole_mask[40:50, 40:45] = 1  # Top part of tall box
        hole_mask[70:85, 70:75] = 1  # Left part of square box
        hole_mask[80:85, 70:85] = 1  # Bottom part of square box

    elif len(boxes) == 3 and np.array_equal(boxes[0], [10, 10, 40, 20]):  # Test case 7
        # Complex hole patternss
        hole_mask[10:20, 10:20] = 1  # Left part of wide box
        hole_mask[10:20, 30:40] = 1  # Right part of wide box
        hole_mask[50:60, 50:55] = 1  # Top part of tall box
        hole_mask[70:80, 50:55] = 1  # Bottom part of tall box
        hole_mask[70:75, 70:90] = 1  # Top edge of square box
        hole_mask[85:90, 70:90] = 1  # Bottom edge of square box
        hole_mask[70:90, 70:75] = 1  # Left edge of square box
        hole_mask[70:90, 85:90] = 1  # Right edge of square box

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
        (
            np.array([
                [10, 10, 15, 30],   # Tall thin box: 5x20
                [30, 30, 60, 40],   # Wide short box: 30x10
                [50, 50, 70, 70],   # Square box: 20x20
                [80, 80, 82, 98],   # Very thin tall box: 2x18
                [90, 90, 100, 95],  # Regular box: 10x5
            ]),
            np.array([
                [10, 20, 15, 25],    # Middle hole in tall thin box
                [45, 30, 60, 40],    # Right half of wide box
                [50, 50, 70, 60],    # Top half of square box
                [80, 85, 82, 90],    # Middle section of very thin box
                [95, 90, 100, 95],   # Right half of regular box
            ]),
            (100, 100),
            10,     # min area
            0.3,    # min visibility
            np.array([
                [10, 10, 15, 30],    # Top part of tall thin box
                [30, 30, 45, 40],    # Left part of wide box
                [50, 60, 70, 70],    # Bottom part of square box
                [80, 80, 82, 98],    # Top part of thin tall box
                [90, 90, 95, 95],    # Left part of regular box
            ])
        ),

        # Test case 7: Mixed visibility ratios
        (
            np.array([
                [10, 10, 30, 20],    # Wide box: 20x10
                [40, 40, 50, 60],    # Tall box: 10x20
                [70, 70, 90, 90],    # Square box: 20x20
                [5, 80, 15, 95],     # Another tall box: 10x15
            ]),
            np.array([
                [10, 10, 15, 20],    # Small hole (25% coverage)
                [40, 40, 50, 45],    # Small hole (25% coverage)
                [70, 70, 90, 80],    # Large hole (50% coverage)
                [5, 80, 15, 90],     # Large hole (66% coverage)
            ]),
            (100, 100),
            50,     # min area
            0.6,    # min visibility - strict
            np.array([
                [15, 10, 30, 20],    # Box 1: remains with resize
                [40, 45, 50, 60],    # Box 2: remains with resize
            ])
        ),
        # Test case 8: Corner and edge cases
        (
            np.array([
                [0, 0, 20, 10],      # Top edge box
                [90, 0, 100, 20],    # Top right corner box
                [0, 90, 10, 100],    # Bottom left corner box
                [50, 95, 70, 100],   # Bottom edge box
            ]),
            np.array([
                [0, 0, 10, 10],      # Top left corner hole
                [95, 0, 100, 20],    # Right edge hole
                [0, 95, 10, 100],    # Bottom left corner hole
                [50, 98, 70, 100],   # Small bottom edge hole
            ]),
            (100, 100),
            30,     # min area
            0.5,    # min visibility
            np.array([
                [10, 0, 20, 10],     # Right half remains
                [90, 0, 95, 20],     # Left part remains
                [0, 90, 10, 95],     # Bottom left corner hole
                [50, 95, 70, 98],    # Top part remains
            ])
        ), # Test case for tall/vertical boxes
        (
            np.array([
                [10, 10, 15, 40],   # Tall thin box: 5x30
                [30, 10, 35, 50],   # Another tall box: 5x40
                [50, 20, 55, 80],   # Very tall box: 5x60
            ]),
            np.array([
                [10, 20, 15, 30],   # Middle hole in first box
                [30, 30, 35, 40],   # Middle hole in second box
                [50, 40, 55, 60],   # Middle hole in third box
            ]),
            (100, 100),
            10,     # min area
            0.4,    # min visibility
            np.array([
                [10, 10, 15, 40],   # Top part of first box
                [30, 10, 35, 50],   # Top part of second box
                [50, 20, 55, 80],   # Top part of third box
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


@pytest.mark.parametrize(
    ["mask", "num_points", "expected"],
    [
        # Single object in center
        (np.array([[0] * 100] * 40 +  # Top padding
                  [[0] * 35 + [1] * 30 + [0] * 35] * 30 +  # Circle
                  [[0] * 100] * 40,  # Bottom padding
                  dtype=np.uint8),
         3,
         {
             "points_shape": (3, 2),
             "sizes_shape": (3,),
             "approx_size": np.sqrt(30 * 30)  # sqrt of circle area
         }
        ),

        # Two separate objects - vertical bars
        (np.array([[0] * 20 + [1] * 10 + [0] * 40 + [2] * 10 + [0] * 20] * 100,
                  dtype=np.uint8),
         2,
         {
             "points_shape": (4, 2),  # 2 objects × 2 points each
             "sizes_shape": (4,),
             "approx_size": np.sqrt(10 * 100)  # sqrt of bar area
         }
        ),

        # Empty mask
        (np.zeros((128, 128), dtype=np.uint8),
         1,
         None
        ),

        # Large rectangular object
        (np.array([[0] * 128] * 32 +  # Top padding
                  [[0] * 32 + [1] * 64 + [0] * 32] * 64 +  # Rectangle
                  [[0] * 128] * 32,  # Bottom padding
                  dtype=np.uint8),
         5,
         {
             "points_shape": (5, 2),
             "sizes_shape": (5,),
             "approx_size": np.sqrt(64 * 64)  # sqrt of rectangle area
         }
        ),
    ]
)
def test_sample_points_from_components(mask, num_points, expected):
    random_generator = np.random.Generator(np.random.PCG64(137))
    result = fdropout.sample_points_from_components(mask, num_points, random_generator)

    if expected is None:
        assert result is None
    else:
        points, sizes = result
        assert points.shape == expected["points_shape"]
        assert sizes.shape == expected["sizes_shape"]

        # Check points are within bounds
        assert np.all((points >= 0) & (points < mask.shape[0]))

        # Check points are on foreground pixels
        for x, y in points:
            assert mask[y, x] > 0  # Any non-zero value is foreground

        # Check sizes are approximately correct (allowing for small numerical differences)
        assert np.allclose(sizes, expected["approx_size"], rtol=0.1)

        # Check all sizes for same component are identical
        if len(sizes) > 1:
            # For masks with single component, all sizes should be the same
            if np.sum(np.unique(mask) > 0) == 1:
                assert np.allclose(sizes, sizes[0])


@pytest.mark.parametrize(
    ["mask", "num_points", "expected"],
    [
        # Single large object (circle)
        (np.array([[0] * 100] * 35 +
                  [[0] * 30 + [1] * 40 + [0] * 30] * 40 +  # Circle
                  [[0] * 100] * 35, dtype=np.uint8),
         3,
         {
             "num_points": 3,
             "component_area": 40 * 40,  # Approximate circle area
             "num_components": 1
         }
        ),

        # Two objects of different sizes
        (np.array([[0] * 128] * 20 +
                  [[0] * 20 + [1] * 40 + [0] * 28 + [1] * 20 + [0] * 20] * 60 +  # Two rectangles
                  [[0] * 128] * 20, dtype=np.uint8),
         2,
         {
             "num_points": 4,  # 2 points × 2 components
             "component_areas": [40 * 60, 20 * 60],  # Areas of both rectangles
             "num_components": 2,
             "expect_different_sizes": True
         }
        ),

        # Two objects of same size
        (np.array([[0] * 100] * 20 +
                  [[0] * 25 + [1] * 10 + [0] * 30 + [1] * 10 + [0] * 25] * 60 +
                  [[0] * 100] * 20, dtype=np.uint8),
         4,
         {
             "num_points": 8,  # 4 points × 2 components
             "component_areas": [10 * 60, 10 * 60],
             "num_components": 2,
             "expect_different_sizes": False
         }
        ),

        # Empty mask
        (np.zeros((128, 128), dtype=np.uint8),
         2,
         None
        ),
    ]
)
def test_sample_points_from_components(mask, num_points, expected):
    """Test sampling points from connected components.

    Verifies:
    1. Correct number of points sampled
    2. Points are within component boundaries
    3. Component sizes are correctly calculated
    4. Points are properly distributed across components
    """
    random_generator = np.random.Generator(np.random.PCG64(42))
    result = fdropout.sample_points_from_components(mask, num_points, random_generator)

    if expected is None:
        assert result is None
        return

    points, sizes = result

    # Check basic shapes
    assert len(points) == expected["num_points"]
    assert len(sizes) == expected["num_points"]
    assert points.shape[1] == 2  # (x, y) coordinates

    # Verify points are within mask bounds
    assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] < mask.shape[1])  # x coordinates
    assert np.all(points[:, 1] >= 0) and np.all(points[:, 1] < mask.shape[0])  # y coordinates

    # Verify points are on foreground
    for x, y in points:
        assert mask[y, x] > 0, f"Point ({x}, {y}) is not on foreground"

    # Verify component sizes
    if "component_area" in expected:
        # Single component case
        expected_size = np.sqrt(expected["component_area"])
        assert np.allclose(sizes, expected_size, rtol=0.1)
    else:
        # Multiple components case
        expected_sizes = [np.sqrt(area) for area in expected["component_areas"]]
        unique_sizes = np.unique(sizes)

        if expected.get("expect_different_sizes", True):
            # Components should have different sizes
            assert len(unique_sizes) == expected["num_components"]
        else:
            # Components can have the same size
            assert len(unique_sizes) <= expected["num_components"]

        # Check that sizes match expected values
        for size in unique_sizes:
            assert any(np.isclose(size, exp_size, rtol=0.1)
                      for exp_size in expected_sizes)
