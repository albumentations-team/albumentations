import numpy as np
import pytest

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.geometric.functional import from_distance_maps, to_distance_maps
from tests.utils import set_seed

import albumentations as A


@pytest.mark.parametrize(
    "image_shape, keypoints, inverted",
    [
        ((100, 100), [(50, 50), (25, 75)], False),
        ((100, 100), [(50, 50), (25, 75)], True),
        ((200, 300), [(100, 150), (50, 199), (150, 50)], False),
        ((200, 300), [(100, 150), (50, 199), (150, 50)], True),
    ],
)
def test_to_distance_maps(image_shape, keypoints, inverted):
    distance_maps = to_distance_maps(keypoints, image_shape, inverted)

    assert distance_maps.shape == (*image_shape, len(keypoints))
    assert distance_maps.dtype == np.float32

    for i, (x, y) in enumerate(keypoints):
        if inverted:
            assert np.isclose(distance_maps[int(y), int(x), i], 1.0)
        else:
            assert np.isclose(distance_maps[int(y), int(x), i], 0.0)

    if inverted:
        assert np.all(distance_maps > 0) and np.all(distance_maps <= 1)
    else:
        assert np.all(distance_maps >= 0)


@pytest.mark.parametrize(
    "image_shape, keypoints, inverted, threshold, if_not_found_coords",
    [
        ((100, 100), [(50, 50), (25, 75)], False, None, None),
        ((100, 100), [(50, 50), (25, 75)], True, None, None),
        ((200, 300), [(100, 150), (50, 199), (150, 50)], False, 10, None),
        ((200, 300), [(100, 150), (50, 199), (150, 50)], True, 0.5, [0, 0]),
        ((150, 150), [(75, 75), (25, 125), (125, 25)], False, None, {"x": -1, "y": -1}),
    ],
)
def test_from_distance_maps(image_shape, keypoints, inverted, threshold, if_not_found_coords):
    distance_maps = to_distance_maps(keypoints, image_shape, inverted)
    recovered_keypoints = from_distance_maps(distance_maps, inverted, if_not_found_coords, threshold)

    assert len(recovered_keypoints) == len(keypoints)

    for original, recovered in zip(keypoints, recovered_keypoints):
        if threshold is None:
            np.testing.assert_allclose(original, recovered, atol=1)
        else:
            x, y = original
            i = keypoints.index(original)
            if (inverted and distance_maps[int(y), int(x), i] >= threshold) or (
                not inverted and distance_maps[int(y), int(x), i] <= threshold
            ):
                np.testing.assert_allclose(original, recovered, atol=1)
            elif if_not_found_coords is not None:
                if isinstance(if_not_found_coords, dict):
                    assert np.allclose(recovered, [if_not_found_coords["x"], if_not_found_coords["y"]])
                else:
                    assert np.allclose(recovered, if_not_found_coords)
            else:
                np.testing.assert_allclose(original, recovered, atol=1)


@pytest.mark.parametrize(
    "image_shape, keypoints, inverted",
    [
        ((100, 100), [(50, 50), (25, 75)], False),
        ((200, 300), [(100, 150), (50, 199), (150, 50)], True),
    ],
)
def test_to_distance_maps_extra_columns(image_shape, keypoints, inverted):
    keypoints_with_extra = [(x, y, 0, 1) for x, y in keypoints]
    distance_maps = to_distance_maps(keypoints, image_shape, inverted)

    assert distance_maps.shape == (*image_shape, len(keypoints))
    assert distance_maps.dtype == np.float32

    for i, (x, y, _, _) in enumerate(keypoints_with_extra):
        if inverted:
            assert np.isclose(distance_maps[int(y), int(x), i], 1.0)
        else:
            assert np.isclose(distance_maps[int(y), int(x), i], 0.0)


@pytest.mark.parametrize(
    "image_shape, grid, expected",
    [
        # Normal case: standard grids
        ((100, 200), (2, 2), np.array([[0, 0, 50, 100], [0, 100, 50, 200], [50, 0, 100, 100], [50, 100, 100, 200]])),
        # Single row grid
        ((100, 200), (1, 4), np.array([[0, 0, 100, 50], [0, 50, 100, 100], [0, 100, 100, 150], [0, 150, 100, 200]])),
        # Single column grid
        ((100, 200), (4, 1), np.array([[0, 0, 25, 200], [25, 0, 50, 200], [50, 0, 75, 200], [75, 0, 100, 200]])),
        # Edge case: Grid size equals image size
        ((100, 200), (100, 200), np.array([[i, j, i + 1, j + 1] for i in range(100) for j in range(200)])),
        # Edge case: Image where width is much larger than height
        ((10, 1000), (1, 10), np.array([[0, i * 100, 10, (i + 1) * 100] for i in range(10)])),
        # Edge case: Image where height is much larger than width
        ((1000, 10), (10, 1), np.array([[i * 100, 0, (i + 1) * 100, 10] for i in range(10)])),
        # Corner case: height and width are not divisible by the number of splits
        (
            (105, 205),
            (3, 4),
            np.array(
                    [[0, 0, 35, 51], [0, 51, 35, 103], [0, 103, 35, 154], [0, 154, 35, 205], [35, 0, 70, 51], [35, 51, 70, 103], [35, 103, 70, 154], [35, 154, 70, 205], [70, 0, 105, 51], [70, 51, 105, 103], [70, 103, 105, 154], [70, 154, 105, 205]]
            ),
        ),
    ],
)
def test_split_uniform_grid(image_shape, grid, expected):
    random_seed = 42
    result = fgeometric.split_uniform_grid(image_shape, grid, random_generator=np.random.default_rng(random_seed))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "size, divisions, random_seed, expected",
    [
        (10, 2, None, [0, 5, 10]),
        (10, 2, 42, [0, 5, 10]),  # Consistent shuffling with seed
        (9, 3, None, [0, 3, 6, 9]),
        (9, 3, 42, [0, 3, 6, 9]),  # Expected shuffle result with a specific seed
        (20, 5, 42, [0, 4, 8, 12, 16, 20]),  # Regular intervals
        (7, 3, 42, [0, 2, 4, 7]),  # Irregular intervals, specific seed
        (7, 3, 41, [0, 3, 5, 7]),  # Irregular intervals, specific seed
    ],
)
def test_generate_shuffled_splits(size, divisions, random_seed, expected):
    result = fgeometric.generate_shuffled_splits(
        size,
        divisions,
        random_generator=np.random.default_rng(random_seed),
    )
    assert len(result) == divisions + 1
    np.testing.assert_array_equal(
        result,
        expected,
    ), f"Failed for size={size}, divisions={divisions}, random_seed={random_seed}"


@pytest.mark.parametrize(
    "size, divisions, random_seed",
    [
        (10, 2, 42),
        (9, 3, 99),
        (20, 5, 101),
        (7, 3, 42),
    ],
)
def test_consistent_shuffling(size, divisions, random_seed):
    set_seed(random_seed)
    result1 = fgeometric.generate_shuffled_splits(size, divisions, random_generator=np.random.default_rng(random_seed))
    assert len(result1) == divisions + 1
    set_seed(random_seed)
    result2 = fgeometric.generate_shuffled_splits(size, divisions, random_generator=np.random.default_rng(random_seed))
    assert len(result2) == divisions + 1
    np.testing.assert_array_equal(result1, result2), "Shuffling is not consistent with the given random state"


@pytest.mark.parametrize(
    ["image_shape", "grid", "scale", "absolute_scale", "expected_shape"],
    [
        # Basic test cases
        ((100, 100), (4, 4), 0.05, False, ((100, 100), (100, 100))),
        ((200, 100), (3, 5), 0.03, False, ((200, 100), (200, 100))),

        # Test different image shapes
        ((50, 75), (2, 2), 0.05, False, ((50, 75), (50, 75))),
        ((300, 200), (5, 5), 0.05, False, ((300, 200), (300, 200))),

        # Test different grid sizes
        ((100, 100), (2, 3), 0.05, False, ((100, 100), (100, 100))),
        ((100, 100), (6, 6), 0.05, False, ((100, 100), (100, 100))),

        # Test with absolute scale
        ((100, 100), (4, 4), 5.0, True, ((100, 100), (100, 100))),
        ((200, 200), (3, 3), 10.0, True, ((200, 200), (200, 200))),
    ]
)
def test_create_piecewise_affine_maps_shapes(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    scale: float,
    absolute_scale: bool,
    expected_shape: tuple[tuple[int, int], tuple[int, int]]
):
    """Test that output maps have correct shapes and types."""
    generator = np.random.default_rng(42)
    map_x, map_y = fgeometric.create_piecewise_affine_maps(image_shape, grid, scale, absolute_scale, generator)

    assert map_x is not None and map_y is not None
    assert map_x.shape == expected_shape[0]
    assert map_y.shape == expected_shape[1]
    assert map_x.dtype == np.float32
    assert map_y.dtype == np.float32

@pytest.mark.parametrize(
    ["image_shape", "grid", "scale"],
    [
        ((100, 100), (4, 4), 0.05),
        ((200, 100), (3, 5), 0.03),
    ]
)
def test_create_piecewise_affine_maps_bounds(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    scale: float
):
    """Test that output maps stay within image bounds."""
    generator = np.random.default_rng(42)
    map_x, map_y = fgeometric.create_piecewise_affine_maps(image_shape, grid, scale, False, generator)

    assert map_x is not None and map_y is not None
    height, width = image_shape

    # Check bounds
    assert np.all(map_x >= 0)
    assert np.all(map_x <= width - 1)
    assert np.all(map_y >= 0)
    assert np.all(map_y <= height - 1)

@pytest.mark.parametrize(
    ["scale", "expected_result"],
    [
        (0.0, (None, None)),  # Zero scale should return None
        (-1.0, (None, None)), # Negative scale should return None
    ]
)
def test_create_piecewise_affine_maps_edge_cases(
    scale: float,
    expected_result: tuple[None, None]
):
    """Test edge cases with zero or negative scale."""
    generator = np.random.default_rng(42)
    result = fgeometric.create_piecewise_affine_maps((100, 100), (4, 4), scale, False, generator)
    assert result == expected_result

def test_create_piecewise_affine_maps_reproducibility():
    """Test that the function produces the same output with the same random seed."""
    result1 = fgeometric.create_piecewise_affine_maps((100, 100), (4, 4), 0.05, False, random_generator=np.random.default_rng(42))
    result2 = fgeometric.create_piecewise_affine_maps((100, 100), (4, 4), 0.05, False, random_generator=np.random.default_rng(42))

    assert result1[0] is not None and result1[1] is not None
    assert result2[0] is not None and result2[1] is not None
    np.testing.assert_array_almost_equal(result1[0], result2[0])
    np.testing.assert_array_almost_equal(result1[1], result2[1])

@pytest.mark.parametrize(
    ["image_shape", "grid"],
    [
        ((0, 100), (4, 4)),    # Zero height
        ((100, 0), (4, 4)),    # Zero width
        ((100, 100), (0, 4)),  # Zero grid rows
        ((100, 100), (4, 0)),  # Zero grid columns
    ]
)
def test_create_piecewise_affine_maps_zero_dimensions(
    image_shape: tuple[int, int],
    grid: tuple[int, int]
):
    """Test handling of zero dimensions."""
    generator = np.random.default_rng(42)
    with pytest.raises(ValueError):
        fgeometric.create_piecewise_affine_maps(image_shape, grid, 0.05, False, generator)

@pytest.mark.parametrize(
    ["image_shape", "grid", "scale", "absolute_scale"],
    [
        ((100, 100), (4, 4), 0.05, False),
        ((200, 100), (3, 5), 0.03, True),
    ]
)
def test_create_piecewise_affine_maps_grid_points(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    scale: float,
    absolute_scale: bool
):
    """Test that grid points are properly distributed."""
    generator = np.random.default_rng(42)
    map_x, map_y = fgeometric.create_piecewise_affine_maps(image_shape, grid, scale, absolute_scale, generator)

    assert map_x is not None and map_y is not None

    height, width = image_shape
    nb_rows, nb_cols = grid

    # Sample points should roughly correspond to grid intersections
    y_steps = np.linspace(0, height - 1, nb_rows)
    x_steps = np.linspace(0, width - 1, nb_cols)

    # Check that grid points are present in the maps
    for y in y_steps:
        for x in x_steps:
            y_idx = int(y)
            x_idx = int(x)

            # Calculate neighborhood size based on scale
            if absolute_scale:
                radius = int(scale * 3)  # 3 sigma radius
            else:
                radius = int(min(width, height) * scale * 3)
            radius = max(1, min(radius, 5))  # Keep radius reasonable

            # Get valid slice bounds
            y_start = max(0, y_idx - radius)
            y_end = min(height, y_idx + radius + 1)
            x_start = max(0, x_idx - radius)
            x_end = min(width, x_idx + radius + 1)

            # Extract neighborhood
            neighborhood = map_x[y_start:y_end, x_start:x_end]

            # Calculate maximum allowed deviation
            if absolute_scale:
                max_deviation = scale * 3
            else:
                max_deviation = min(width, height) * scale * 3

            # Check if any point in neighborhood is close to expected x coordinate
            assert np.any(np.abs(neighborhood - x) < max_deviation), \
                f"No points near grid intersection ({x}, {y}) within allowed deviation"
