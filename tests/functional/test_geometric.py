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
