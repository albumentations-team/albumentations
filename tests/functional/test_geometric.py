import numpy as np
import pytest
from albumentations.augmentations.geometric import functional as fgeometric

import numpy as np
import pytest
from albumentations.augmentations.geometric.functional import calculate_grid_dimensions

@pytest.mark.parametrize("image_shape, num_grid_xy, expected_shape, expected_first, expected_last", [
    ((100, 150), (3, 2), (2, 3, 4), [0, 0, 50, 50], [100, 50, 150, 100]),
    ((101, 151), (3, 2), (2, 3, 4), [0, 0, 50, 50], [100, 50, 151, 101]),
    ((100, 150), (1, 1), (1, 1, 4), [0, 0, 150, 100], [0, 0, 150, 100]),
    ((10000, 10000), (100, 100), (100, 100, 4), [0, 0, 100, 100], [9900, 9900, 10000, 10000]),
])
def test_calculate_grid_dimensions_vectorized(image_shape, num_grid_xy, expected_shape, expected_first, expected_last):
    dimensions = calculate_grid_dimensions(image_shape, num_grid_xy)
    assert dimensions.shape == expected_shape
    assert np.array_equal(dimensions[0, 0], expected_first)
    assert np.array_equal(dimensions[-1, -1], expected_last)

def test_calculate_grid_dimensions_vectorized_dtype():
    image_shape = (100, 150)
    num_grid_xy = (3, 2)
    dimensions = calculate_grid_dimensions(image_shape, num_grid_xy)
    assert dimensions.dtype == np.int16


@pytest.mark.parametrize("image_shape, num_grid_xy", [
    ((100, 150), (3, 2)),
    ((101, 151), (3, 2)),
    ((100, 150), (1, 1)),
    ((10000, 10000), (100, 100)),
])
def test_calculate_grid_dimensions_vectorized_properties(image_shape, num_grid_xy):
    dimensions = calculate_grid_dimensions(image_shape, num_grid_xy)

    # Check that x_min and y_min are always smaller than x_max and y_max respectively
    assert np.all(dimensions[..., 0] < dimensions[..., 2])  # x_min < x_max
    assert np.all(dimensions[..., 1] < dimensions[..., 3])  # y_min < y_max

    # Check that the last row and column reach the image boundaries
    assert np.all(dimensions[-1, :, 3] == image_shape[0])  # y_max of last row
    assert np.all(dimensions[:, -1, 2] == image_shape[1])  # x_max of last column

    # Check that there are no gaps between cells
    assert np.all(dimensions[1:, :, 1] == dimensions[:-1, :, 3])  # y_min of next row == y_max of current row
    assert np.all(dimensions[:, 1:, 0] == dimensions[:, :-1, 2])  # x_min of next column == x_max of current column



@pytest.mark.parametrize("image_shape, num_grid_xy, magnitude, expected_shape", [
    ((100, 100), (2, 2), 10, (4, 8)),
    ((200, 300), (3, 4), 20, (12, 8)),
    ((150, 150), (5, 5), 5, (25, 8)),
])
def test_generate_distorted_grid_polygons_shape(image_shape, num_grid_xy, magnitude, expected_shape):
    dimensions = calculate_grid_dimensions(image_shape, num_grid_xy)
    polygons = fgeometric.generate_distorted_grid_polygons(dimensions, magnitude)
    assert polygons.shape == expected_shape

@pytest.mark.parametrize("image_shape, num_grid_xy, magnitude", [
    ((100, 100), (2, 2), 10),
    ((200, 300), (3, 4), 20),
    ((150, 150), (5, 5), 5),
])
def test_generate_distorted_grid_polygons_boundary_unchanged(image_shape, num_grid_xy, magnitude):
    dimensions = calculate_grid_dimensions(image_shape, num_grid_xy)
    original_dimensions = dimensions.reshape(-1, 4)
    polygons = fgeometric.generate_distorted_grid_polygons(dimensions, magnitude)

    grid_height, grid_width = dimensions.shape[:2]

    # Check top row
    assert np.allclose(polygons[:grid_width, :4], original_dimensions[:grid_width, [0, 1, 2, 1]])

    # Check bottom row
    assert np.allclose(polygons[-grid_width:, 4:], original_dimensions[-grid_width:, [2, 3, 0, 3]])

    # Check left column
    left_column = polygons[::grid_width]
    assert np.allclose(left_column[:, [0, 1, 6, 7]], original_dimensions[::grid_width, [0, 1, 0, 3]])

    # Check right column
    right_column = polygons[grid_width-1::grid_width]
    assert np.allclose(right_column[:, [2, 3, 4, 5]], original_dimensions[grid_width-1::grid_width, [2, 1, 2, 3]])

@pytest.mark.parametrize("image_shape, num_grid_xy, magnitude", [
    ((100, 100), (3, 3), 10),
    ((200, 300), (4, 5), 20),
    ((150, 150), (6, 6), 5),
])
def test_generate_distorted_grid_polygons_internal_points_moved(image_shape, num_grid_xy, magnitude):
    dimensions = calculate_grid_dimensions(image_shape, num_grid_xy)
    original_dimensions = dimensions.reshape(-1, 4)
    polygons = fgeometric.generate_distorted_grid_polygons(dimensions, magnitude)

    grid_height, grid_width = dimensions.shape[:2]

    # Check that internal points have moved
    for i in range(1, grid_height - 1):
        for j in range(1, grid_width - 1):
            cell_idx = i * grid_width + j
            assert not np.allclose(polygons[cell_idx], [
                original_dimensions[cell_idx, 0], original_dimensions[cell_idx, 1],
                original_dimensions[cell_idx, 2], original_dimensions[cell_idx, 1],
                original_dimensions[cell_idx, 2], original_dimensions[cell_idx, 3],
                original_dimensions[cell_idx, 0], original_dimensions[cell_idx, 3]
            ])

@pytest.mark.parametrize("image_shape, num_grid_xy, magnitude", [
    ((100, 100), (3, 3), 10),
    ((200, 300), (4, 5), 20),
    ((150, 150), (6, 6), 5),
])
def test_generate_distorted_grid_polygons_consistent_shared_points(image_shape, num_grid_xy, magnitude):
    dimensions = calculate_grid_dimensions(image_shape, num_grid_xy)
    polygons = fgeometric.generate_distorted_grid_polygons(dimensions, magnitude)

    grid_height, grid_width = dimensions.shape[:2]

    # Check that shared points between adjacent cells are consistent
    for i in range(1, grid_height):
        for j in range(1, grid_width):
            top_right = polygons[(i - 1) * grid_width + (j - 1), 4:6]
            bottom_left = polygons[i * grid_width + j, 0:2]
            assert np.allclose(top_right, bottom_left)
