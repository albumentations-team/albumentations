import numpy as np
import pytest

# Import the functions to test
from albumentations.core.utils import (
    reshape_3d, reshape_batch, restore_from_spatial, restore_from_channel, reshape_for_spatial, reshape_for_channel
)

# Test shapes
SPATIAL_SHAPES = [
    # (input_shape, expected_shape, has_batch, has_depth)
    ((5, 32, 32), (32, 32, 5), False, True),  # D,H,W
    ((5, 32, 32, 3), (32, 32, 15), False, True),  # D,H,W,C
    ((10, 32, 32), (32, 32, 10), True, False),  # N,H,W
    ((10, 32, 32, 3), (32, 32, 30), True, False),  # N,H,W,C
    ((10, 5, 32, 32), (32, 32, 50), True, True),  # N,D,H,W
    ((10, 5, 32, 32, 3), (32, 32, 150), True, True),  # N,D,H,W,C
]

CHANNEL_SHAPES = [
    # (input_shape, expected_shape, has_batch, has_depth)
    ((5, 32, 32), (160, 32), False, True),  # D,H,W
    ((5, 32, 32, 3), (160, 32, 3), False, True),  # D,H,W,C
    ((10, 32, 32), (320, 32), True, False),  # N,H,W
    ((10, 32, 32, 3), (320, 32, 3), True, False),  # N,H,W,C
    ((10, 5, 32, 32), (1600, 32), True, True),  # N,D,H,W
    ((10, 5, 32, 32, 3), (1600, 32, 3), True, True),  # N,D,H,W,C
]

@pytest.mark.parametrize("input_shape,expected_shape,has_batch,has_depth", SPATIAL_SHAPES)
def test_spatial_reshape(input_shape: tuple, expected_shape: tuple, has_batch: bool, has_depth: bool):
    """Test spatial reshape for various input shapes."""
    data = np.random.rand(*input_shape)
    reshaped, original_shape = reshape_for_spatial(data, has_batch, has_depth)

    assert reshaped.shape == expected_shape
    assert original_shape == input_shape
    assert reshaped.flags["C_CONTIGUOUS"]

@pytest.mark.parametrize("input_shape,expected_shape,has_batch,has_depth", CHANNEL_SHAPES)
def test_channel_reshape(input_shape: tuple, expected_shape: tuple, has_batch: bool, has_depth: bool):
    """Test channel reshape for various input shapes."""
    data = np.random.rand(*input_shape)
    reshaped, original_shape = reshape_for_channel(data, has_batch, has_depth)

    assert reshaped.shape == expected_shape
    assert original_shape == input_shape
    assert reshaped.flags["C_CONTIGUOUS"]

@pytest.mark.parametrize("input_shape,_,has_batch,has_depth", SPATIAL_SHAPES)
def test_spatial_roundtrip(input_shape: tuple, _, has_batch: bool, has_depth: bool):
    """Test that reshape->restore preserves data for spatial transforms."""
    data = np.random.rand(*input_shape)
    # Use reshape_for_spatial instead of reshape_3d directly
    reshaped, original_shape = reshape_for_spatial(data, has_batch, has_depth)
    restored = restore_from_spatial(reshaped, original_shape, has_batch, has_depth)

    assert restored.shape == input_shape
    assert restored.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(data, restored)

@pytest.mark.parametrize("input_shape,_,has_batch,has_depth", CHANNEL_SHAPES)
def test_channel_roundtrip(input_shape: tuple, _, has_batch: bool, has_depth: bool):
    """Test that reshape->restore preserves data for channel transforms."""
    data = np.random.rand(*input_shape)
    # Use reshape_for_channel instead of reshape_batch
    reshaped, original_shape = reshape_for_channel(data, has_batch, has_depth)
    restored = restore_from_channel(reshaped, original_shape, has_batch, has_depth)

    assert restored.shape == input_shape
    assert restored.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(data, restored)

def test_empty_arrays():
    """Test that empty arrays raise appropriate errors."""
    empty_array = np.array([])

    with pytest.raises(ValueError):
        reshape_for_spatial(empty_array, True, False)

    with pytest.raises(ValueError):
        reshape_for_channel(empty_array, True, False)

@pytest.mark.parametrize("transform_type", ["spatial", "channel"])
def test_non_contiguous_input(transform_type: str):
    """Test that non-contiguous arrays are handled correctly."""
    # Create non-contiguous array by slicing
    data = np.random.rand(10, 32, 32, 3)[::2]
    assert not data.flags["C_CONTIGUOUS"]

    reshape_func = {
        "spatial": reshape_for_spatial,
        "channel": reshape_for_channel,
    }[transform_type]

    reshaped, _ = reshape_func(data, True, False)
    assert reshaped.flags["C_CONTIGUOUS"]

def test_spatial_transform_preserves_spatial_dims():
    """Test that spatial transforms preserve H,W dimensions after restore."""
    input_shape = (10, 5, 32, 32, 3)  # N,D,H,W,C
    data = np.random.rand(*input_shape)

    # Simulate a spatial transform that changes H,W dimensions
    reshaped, original_shape = reshape_3d(data)
    new_spatial_shape = (64, 64, reshaped.shape[2])  # Double spatial dimensions
    transformed = np.random.rand(*new_spatial_shape)  # Simulated transform result

    restored = restore_from_spatial(transformed, original_shape, True, True)
    expected_shape = (10, 5, 64, 64, 3)  # N,D dimensions preserved, H,W changed

    assert restored.shape == expected_shape
    assert restored.flags["C_CONTIGUOUS"]

def test_channel_transform_preserves_width():
    """Test that channel transforms preserve W dimension after restore."""
    input_shape = (10, 5, 32, 32, 3)  # N,D,H,W,C
    data = np.random.rand(*input_shape)

    # Use reshape_for_channel instead of reshape_batch
    reshaped, original_shape = reshape_for_channel(data, True, True)
    # reshaped should be (1600, 32, 3)

    # Double the width dimension
    new_shape = (reshaped.shape[0], 64, reshaped.shape[2])  # (1600, 64, 3)
    transformed = np.random.rand(*new_shape)

    restored = restore_from_channel(transformed, original_shape, True, True)
    # Should restore to (10, 5, 32, 64, 3) - note the new width

    assert restored.shape == (10, 5, 32, 64, 3)


SPATIAL_3D_SHAPES = [
    # (input_shape, expected_shape, has_batch, has_depth, keep_depth)
    ((5, 32, 32), (5, 32, 32), False, True, True),  # D,H,W stays D,H,W
    ((5, 32, 32, 3), (5, 32, 32, 3), False, True, True),  # D,H,W,C stays D,H,W,C
    ((10, 5, 32, 32), (5, 32, 32, 10), True, True, True),  # N,D,H,W => D,H,W,N
    ((10, 5, 32, 32, 3), (5, 32, 32, 30), True, True, True),  # N,D,H,W,C => D,H,W,N*C
]

@pytest.mark.parametrize(
    "input_shape,expected_shape,has_batch,has_depth,keep_depth",
    SPATIAL_3D_SHAPES
)
def test_spatial_3d_reshape(
    input_shape: tuple,
    expected_shape: tuple,
    has_batch: bool,
    has_depth: bool,
    keep_depth: bool
):
    """Test spatial reshape for 3D volumes."""
    data = np.random.rand(*input_shape)
    reshaped, original_shape = reshape_for_spatial(
        data, has_batch, has_depth, keep_depth_dim=keep_depth
    )

    assert reshaped.shape == expected_shape
    assert original_shape == input_shape
    assert reshaped.flags["C_CONTIGUOUS"]

@pytest.mark.parametrize(
    "input_shape,_,has_batch,has_depth,keep_depth",
    SPATIAL_3D_SHAPES
)
def test_spatial_3d_roundtrip(
    input_shape: tuple,
    _: tuple,
    has_batch: bool,
    has_depth: bool,
    keep_depth: bool
):
    """Test that reshape->restore preserves data for 3D spatial transforms."""
    data = np.random.rand(*input_shape)
    reshaped, original_shape = reshape_for_spatial(
        data, has_batch, has_depth, keep_depth_dim=keep_depth
    )
    restored = restore_from_spatial(
        reshaped, original_shape, has_batch, has_depth, keep_depth_dim=keep_depth
    )

    assert restored.shape == input_shape
    assert restored.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(data, restored)

def test_spatial_3d_transform_preserves_dims():
    """Test that 3D spatial transforms preserve N,C dimensions while allowing D,H,W to change."""
    input_shape = (10, 5, 32, 32, 3)  # N,D,H,W,C
    data = np.random.rand(*input_shape)

    # Simulate a spatial transform that changes H,W dimensions
    reshaped, original_shape = reshape_for_spatial(
        data, True, True, keep_depth_dim=True
    )
    # Double H,W dimensions
    transformed = np.random.rand(5, 64, 64, 30)  # (D,H',W',N*C)

    restored = restore_from_spatial(
        transformed, original_shape, True, True, keep_depth_dim=True
    )
    expected_shape = (10, 5, 64, 64, 3)  # (N,D,H',W',C)

    assert restored.shape == expected_shape
    assert restored.flags["C_CONTIGUOUS"]

def test_spatial_3d_transform_preserves_order():
    """Test that 3D spatial transforms preserve data order."""
    input_shape = (2, 3, 4, 4, 1)  # N,D,H,W,C
    data = np.arange(np.prod(input_shape)).reshape(input_shape)

    reshaped, original_shape = reshape_for_spatial(
        data, True, True, keep_depth_dim=True
    )
    restored = restore_from_spatial(
        reshaped, original_shape, True, True, keep_depth_dim=True
    )

    np.testing.assert_array_equal(data, restored)

@pytest.mark.parametrize("keep_depth", [True, False])
def test_spatial_3d_transform_modes(keep_depth: bool):
    """Test both modes of 3D spatial transforms."""
    input_shape = (2, 3, 4, 4, 1)  # N,D,H,W,C
    data = np.random.rand(*input_shape)

    reshaped, original_shape = reshape_for_spatial(
        data, True, True, keep_depth_dim=keep_depth
    )

    if keep_depth:
        assert reshaped.shape == (3, 4, 4, 2)  # D,H,W,N
    else:
        assert reshaped.shape == (4, 4, 6)  # H,W,N*D

    restored = restore_from_spatial(
        reshaped, original_shape, True, True, keep_depth_dim=keep_depth
    )
    np.testing.assert_array_equal(data, restored)


def test_spatial_3d_transform_allows_dhw_changes():
    """Test that 3D spatial transforms allow D,H,W dimensions to change (e.g., for RandomCrop3D)."""
    input_shape = (10, 8, 32, 32, 3)  # N,D,H,W,C
    data = np.random.rand(*input_shape)

    # Simulate a spatial transform that changes D,H,W dimensions
    reshaped, original_shape = reshape_for_spatial(
        data, True, True, keep_depth_dim=True
    )
    # Simulate crop: reduce D,H,W dimensions
    transformed = np.random.rand(4, 16, 16, 30)  # (D',H',W',N*C)

    restored = restore_from_spatial(
        transformed, original_shape, True, True, keep_depth_dim=True
    )
    expected_shape = (10, 4, 16, 16, 3)  # (N,D',H',W',C)

    assert restored.shape == expected_shape
    assert restored.flags["C_CONTIGUOUS"]
