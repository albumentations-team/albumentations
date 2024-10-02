import numpy as np
import pytest
import skimage
from albumentations.augmentations.geometric import functional as fgeometric

import numpy as np
import pytest
from albumentations.augmentations.geometric.functional import from_distance_maps, to_distance_maps
from tests.utils import set_seed


@pytest.mark.parametrize("image_shape, keypoints, inverted", [
    ((100, 100), [(50, 50), (25, 75)], False),
    ((100, 100), [(50, 50), (25, 75)], True),
    ((200, 300), [(100, 150), (50, 199), (150, 50)], False),
    ((200, 300), [(100, 150), (50, 199), (150, 50)], True),
])
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

@pytest.mark.parametrize("image_shape, keypoints, inverted, threshold, if_not_found_coords", [
    ((100, 100), [(50, 50), (25, 75)], False, None, None),
    ((100, 100), [(50, 50), (25, 75)], True, None, None),
    ((200, 300), [(100, 150), (50, 199), (150, 50)], False, 10, None),
    ((200, 300), [(100, 150), (50, 199), (150, 50)], True, 0.5, [0, 0]),
    ((150, 150), [(75, 75), (25, 125), (125, 25)], False, None, {"x": -1, "y": -1}),
])
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
            if (inverted and distance_maps[int(y), int(x), i] >= threshold) or (not inverted and distance_maps[int(y), int(x), i] <= threshold):
                np.testing.assert_allclose(original, recovered, atol=1)
            elif if_not_found_coords is not None:
                if isinstance(if_not_found_coords, dict):
                    assert np.allclose(recovered, [if_not_found_coords['x'], if_not_found_coords['y']])
                else:
                    assert np.allclose(recovered, if_not_found_coords)
            else:
                np.testing.assert_allclose(original, recovered, atol=1)

@pytest.mark.parametrize("image_shape, keypoints, inverted", [
    ((100, 100), [(50, 50), (25, 75)], False),
    ((200, 300), [(100, 150), (50, 199), (150, 50)], True),
])
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
        ((100, 200), (100, 200), np.array([[i, j, i+1, j+1] for i in range(100) for j in range(200)])),

        # Edge case: Image where width is much larger than height
        ((10, 1000), (1, 10), np.array([[0, i * 100, 10, (i + 1) * 100] for i in range(10)])),

        # Edge case: Image where height is much larger than width
        ((1000, 10), (10, 1), np.array([[i * 100, 0, (i + 1) * 100, 10] for i in range(10)])),

        # Corner case: height and width are not divisible by the number of splits
        ((105, 205), (3, 4), np.array([
            [0, 0, 35, 51], [0, 51, 35, 102], [0, 102, 35, 153], [0, 153, 35, 205],  # First row splits
            [35, 0, 70, 51], [35, 51, 70, 102], [35, 102, 70, 153], [35, 153, 70, 205],  # Second row splits
            [70, 0, 105, 51], [70, 51, 105, 102], [70, 102, 105, 153], [70, 153, 105, 205]  # Third row splits
        ])),
    ]
)
def test_split_uniform_grid(image_shape, grid, expected):
    random_seed = 42
    result = fgeometric.split_uniform_grid(image_shape, grid, random_state=np.random.RandomState(random_seed))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("size, divisions, random_state, expected", [
    (10, 2, None, [0, 5, 10]),
    (10, 2, 42, [0, 5, 10]),  # Consistent shuffling with seed
    (9, 3, None, [0, 3, 6, 9]),
    (9, 3, 42, [0, 3, 6, 9]),  # Expected shuffle result with a specific seed
    (20, 5, 42, [0, 4, 8, 12, 16, 20]),  # Regular intervals
    (7, 3, 42, [0, 3, 5, 7]),  # Irregular intervals, specific seed
    (7, 3, 41, [0, 2, 4, 7]),  # Irregular intervals, specific seed
])
def test_generate_shuffled_splits(size, divisions, random_state, expected):
    result = fgeometric.generate_shuffled_splits(
        size, divisions, random_state=np.random.RandomState(random_state) if random_state else None
    )
    assert len(result) == divisions + 1
    assert np.array_equal(result, expected), \
        f"Failed for size={size}, divisions={divisions}, random_state={random_state}"


@pytest.mark.parametrize("size, divisions, random_state", [
    (10, 2, 42),
    (9, 3, 99),
    (20, 5, 101),
    (7, 3, 42),
])
def test_consistent_shuffling(size, divisions, random_state):
    set_seed(random_state)
    result1 = fgeometric.generate_shuffled_splits(size, divisions, random_state=np.random.RandomState(random_state))
    assert len(result1) == divisions + 1
    set_seed(random_state)
    result2 = fgeometric.generate_shuffled_splits(size, divisions, random_state=np.random.RandomState(random_state))
    assert len(result2) == divisions + 1
    assert np.array_equal(result1, result2), "Shuffling is not consistent with the given random state"
