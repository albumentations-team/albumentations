from __future__ import annotations

import pytest
import numpy as np
from albumentations.augmentations.transforms3d import functional as f3d

@pytest.mark.parametrize(
    "input_shape,n_channels", [
        ((3, 3, 3), None),  # 3D case
        ((3, 3, 3, 2), 2),  # 4D case
    ]
)
def test_uniqueness(input_shape: tuple, n_channels: int | None):
    # Create test cube with unique values
    n_elements = np.prod(input_shape)
    test_cube = np.arange(n_elements).reshape(input_shape)

    # Generate all 48 transformations
    transformations = [f3d.transform_cube(test_cube, i) for i in range(48)]

    # Check uniqueness
    unique_transforms = set(str(t) for t in transformations)
    assert len(unique_transforms) == 48, "Not all transformations are unique!"

    # Check shape preservation
    expected_shape = input_shape
    for t in transformations:
        assert t.shape == expected_shape, f"Wrong shape: got {t.shape}, expected {expected_shape}"
