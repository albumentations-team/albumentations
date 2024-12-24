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


@pytest.mark.parametrize(
    ["keypoints", "holes", "expected"],
    [
        # Basic case: single hole, some points inside/outside
        (
            np.array([[1, 1, 1], [5, 5, 5], [8, 8, 8]], dtype=np.float32),  # keypoints (XYZ)
            np.array([[4, 4, 4, 6, 6, 6]], dtype=np.float32),               # holes (Z1,Y1,X1,Z2,Y2,X2)
            np.array([[1, 1, 1], [8, 8, 8]], dtype=np.float32),            # expected (points outside hole)
        ),
        # Multiple holes
        (
            np.array([[1, 1, 1], [5, 5, 5], [8, 8, 8]], dtype=np.float32),
            np.array([
                [0, 0, 0, 2, 2, 2],    # hole covering [1,1,1]
                [4, 4, 4, 6, 6, 6],    # hole covering [5,5,5]
            ], dtype=np.float32),
            np.array([[8, 8, 8]], dtype=np.float32),     # only last point survives
        ),
        # Edge cases: points exactly on boundaries
        (
            np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32),
            np.array([[1, 1, 1, 2, 2, 2]], dtype=np.float32),  # hole with points on boundaries
            np.array([[2, 2, 2], [3, 3, 3]], dtype=np.float32),  # points on/outside boundaries remain
        ),
        # Empty arrays
        (
            np.array([], dtype=np.float32).reshape(0, 3),      # no keypoints
            np.array([[1, 1, 1, 2, 2, 2]], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(0, 3),
        ),
        (
            np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(0, 6),      # no holes
            np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
        ),
        # Extra keypoint dimensions
        (
            np.array([[1, 1, 1, 0.9], [5, 5, 5, 0.8]], dtype=np.float32),
            np.array([[4, 4, 4, 6, 6, 6]], dtype=np.float32),
            np.array([[1, 1, 1, 0.9]], dtype=np.float32),
        ),
        # Overlapping holes
        (
            np.array([[1, 1, 1], [3, 3, 3], [5, 5, 5]], dtype=np.float32),
            np.array([
                [0, 0, 0, 4, 4, 4],
                [2, 2, 2, 6, 6, 6],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(0, 3),  # all points inside holes
        ),
    ],
)
def test_filter_keypoints_in_holes3d(keypoints, holes, expected):
    result = f3d.filter_keypoints_in_holes3d(keypoints, holes)
    np.testing.assert_array_equal(
        result, expected,
        err_msg=f"Failed with keypoints {keypoints} and holes {holes}"
    )

def test_filter_keypoints_in_holes3d_invalid_input():
    """Test error handling for invalid input shapes."""
    # Invalid keypoint dimensions
    with pytest.raises(IndexError):
        f3d.filter_keypoints_in_holes3d(
            np.array([[1, 1]]),  # only 2D points
            np.array([[0, 0, 0, 1, 1, 1]])
        )

    # Invalid hole dimensions
    with pytest.raises(IndexError):
        f3d.filter_keypoints_in_holes3d(
            np.array([[1, 1, 1]]),
            np.array([[0, 0, 0, 1]])  # incomplete hole specification
        )

def test_filter_keypoints_in_holes3d_random():
    """Test with random data to ensure robustness."""
    rng = np.random.default_rng(42)

    # Generate random keypoints and holes
    num_keypoints = 100
    num_holes = 10
    volume_size = 100

    keypoints = rng.integers(0, volume_size, (num_keypoints, 3))
    holes = np.array([
        [
            rng.integers(0, volume_size-10),  # z1
            rng.integers(0, volume_size-10),  # y1
            rng.integers(0, volume_size-10),  # x1
            rng.integers(10, volume_size),    # z2
            rng.integers(10, volume_size),    # y2
            rng.integers(10, volume_size),    # x2
        ]
        for _ in range(num_holes)
    ])

    # Ensure z2>z1, y2>y1, x2>x1 for each hole
    holes[:, 3:] = holes[:, :3] + holes[:, 3:]

    # Test function
    result = f3d.filter_keypoints_in_holes3d(keypoints, holes)

    # Verify each surviving point is actually outside all holes
    for point in result:
        x, y, z = point
        for z1, y1, x1, z2, y2, x2 in holes:
            assert not (
                z1 <= z < z2 and
                y1 <= y < y2 and
                x1 <= x < x2
            ), f"Point {point} should be outside hole {[z1,y1,x1,z2,y2,x2]}"

@pytest.mark.parametrize(
    "factor",  # Remove the brackets - it's just the parameter name
    [
        1,
        2,
        3,
        -1,
        5,
    ]
)
@pytest.mark.parametrize(
    "axes",    # Remove the brackets - it's just the parameter name
    [
        (0, 1),  # rotate in HW plane
        (0, 2),  # rotate in HD plane
        (1, 2),  # rotate in WD plane
    ]
)
def test_keypoints_rot90_matches_numpy(factor, axes):
    """Test that keypoints_rot90 matches np.rot90 behavior."""
    # Create volume with different dimensions to catch edge cases
    volume = np.zeros((5, 6, 7), dtype=np.uint8)  # (H, W, D)

    # Create test points (avoiding edges for clear results)
    keypoints = np.array([
        [1, 1, 1],  # XYZ coordinates
        [1, 3, 1],
        [3, 1, 3],
        [2, 2, 2],
    ], dtype=np.float32)

    # Convert keypoints from XYZ to HWD ordering
    keypoints_hwd = keypoints[:, [2, 1, 0]]  # XYZ -> HWD

    # Mark points in volume
    for h, w, d in keypoints_hwd:
        volume[int(h), int(w), int(d)] = 1

    # Rotate volume
    rotated_volume = np.rot90(volume.copy(), factor, axes=axes)

    # Rotate keypoints
    rotated_keypoints_hwd = f3d.keypoints_rot90(keypoints_hwd, factor, axes, volume_shape=volume.shape)

    # Convert back to XYZ for verification
    rotated_keypoints = rotated_keypoints_hwd[:, [2, 1, 0]]  # HWD -> XYZ

    # Verify each rotated keypoint matches a marked point in rotated volume
    for x, y, z in rotated_keypoints:
        assert rotated_volume[int(z), int(y), int(x)] == 1, (
            f"Keypoint at ({x}, {y}, {z}) should match marked point in volume "
            f"after rotation with factor={factor}, axes={axes}"
        )


@pytest.mark.parametrize("index", range(48))
def test_transform_cube_keypoints_matches_transform_cube(index):
    """Test that transform_cube_keypoints matches transform_cube behavior and preserves extra columns."""
    # Create volume with different dimensions to catch edge cases
    volume = np.zeros((5, 6, 7), dtype=np.uint8)  # (D, H, W)

    # Create test points with additional columns
    keypoints = np.array([
        [1, 1, 1, 0.5, 0.6, 0.7],  # XYZ coordinates + 3 extra values
        [1, 3, 1, 0.2, 0.3, 0.4],
        [3, 1, 3, 0.8, 0.9, 1.0],
        [2, 2, 2, 0.1, 0.2, 0.3],
    ], dtype=np.float32)

    # Store original extra columns for comparison
    original_extra_cols = keypoints[:, 3:].copy()

    # Mark points in volume (converting from XYZ to DHW)
    for x, y, z in keypoints[:, :3]:
        volume[int(z), int(y), int(x)] = 1

    # Transform volume
    transformed_volume = f3d.transform_cube(volume.copy(), index)

    # Transform keypoints
    transformed_keypoints = f3d.transform_cube_keypoints(keypoints.copy(), index, volume_shape=volume.shape)

    # Verify each transformed keypoint matches a marked point in transformed volume
    for x, y, z in transformed_keypoints[:, :3]:
        assert transformed_volume[int(z), int(y), int(x)] == 1, (
            f"Keypoint at ({x}, {y}, {z}) should match marked point in volume "
            f"after transformation with index={index}"
        )

    # Verify extra columns remain unchanged
    np.testing.assert_array_equal(
        transformed_keypoints[:, 3:],
        original_extra_cols,
        err_msg=f"Extra columns should remain unchanged after transformation with index={index}"
    )
