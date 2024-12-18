import numpy as np
import pytest
from albumentations.core.utils import (
    reshape_for_spatial, restore_from_spatial,
    reshape_for_channel, restore_from_channel,
    reshape_for_full, restore_from_full,
    batch_transform
)

# Test data shapes
SHAPES = [
    (2, 31, 32),  # 3D
    (2, 31, 32, 3),  # 4D
    (2, 4, 31, 32, 3),  # 5D
]

@pytest.mark.parametrize("shape", SHAPES)
def test_spatial_reshape(shape):
    """Test spatial reshape maintains relative order of dimensions and moves batch dims to channels."""
    x = np.random.rand(*shape)
    reshaped, original_shape = reshape_for_spatial(x)

    # Check shape preservation
    assert original_shape == shape

    # For 3D input, last dim should be N
    if len(shape) == 3:
        assert reshaped.shape[2] == shape[0]  # N at the end
    # For 4D/5D input, last dim should be N*C
    else:
        *batch_dims, H, W, C = shape
        expected_channels = np.prod(batch_dims) * C
        assert reshaped.shape[2] == expected_channels

    # Simulate a spatial transform that changes H,W dimensions
    new_H, new_W = 16, 16  # smaller size
    transformed = reshaped[:new_H, :new_W]

    # Check content preservation after restore
    restored = restore_from_spatial(transformed, original_shape)
    if len(shape) == 3:
        expected_shape = (shape[0], new_H, new_W)
    else:
        *batch_dims, _, _, C = shape
        expected_shape = (*batch_dims, new_H, new_W, C)

    assert restored.shape == expected_shape
    assert restored.flags["C_CONTIGUOUS"]

@pytest.mark.parametrize("shape", SHAPES[1:])  # Skip 3D as it has no channels
def test_channel_reshape(shape):
    """Test channel reshape maintains channels and combines batch with spatial dims."""
    x = np.random.rand(*shape)
    reshaped, original_shape = reshape_for_channel(x)

    # Check shape preservation
    assert original_shape == shape
    assert reshaped.shape[-1] == shape[-1]  # Channels preserved

    # Check content preservation
    restored = restore_from_channel(reshaped, original_shape)
    np.testing.assert_array_equal(x, restored)
    assert restored.flags["C_CONTIGUOUS"]

@pytest.mark.parametrize("shape", SHAPES)
def test_full_reshape(shape):
    """Test full reshape maintains original shape."""
    x = np.random.rand(*shape)
    reshaped, original_shape = reshape_for_full(x)

    # Check shape preservation
    assert original_shape == shape
    assert reshaped.shape == shape

    # Check content preservation
    restored = restore_from_full(reshaped, original_shape)
    np.testing.assert_array_equal(x, restored)

# Test decorator with mock transforms
class MockTransform:
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img

@pytest.mark.parametrize("transform_type", ["spatial", "channel", "full"])
@pytest.mark.parametrize("shape", SHAPES)
def test_batch_transform_decorator(transform_type, shape):
    """Test batch_transform decorator with different transform types and shapes."""
    class TestTransform(MockTransform):
        @batch_transform(transform_type)
        def apply_to_batch(self, imgs: np.ndarray, **params) -> np.ndarray:
            return self.apply(imgs, **params)

    transform = TestTransform()
    x = np.random.rand(*shape)

    # Skip channel transforms for 3D inputs
    if transform_type == "channel" and len(shape) == 3:
        with pytest.raises(ValueError):
            result = transform.apply_to_batch(x)
        return

    result = transform.apply_to_batch(x)

    # Check shape and content preservation
    assert result.shape == shape
    np.testing.assert_array_equal(x, result)
    assert result.flags["C_CONTIGUOUS"]

@pytest.mark.parametrize("transform_type,shape", [
    ("spatial", (2, 32, 32)),
    ("spatial", (2, 32, 32, 3)),
    ("channel", (2, 32, 32, 3)),
    ("full", (2, 32, 32, 3)),
])
def test_non_contiguous_input(transform_type, shape):
    """Test that non-contiguous input is handled correctly."""
    class TestTransform(MockTransform):
        @batch_transform(transform_type)
        def apply_to_batch(self, imgs: np.ndarray, **params) -> np.ndarray:
            return self.apply(imgs, **params)

    transform = TestTransform()
    x = np.random.rand(*shape)
    # Make array non-contiguous by transposing
    x = np.transpose(x, axes=range(len(shape)-1, -1, -1))
    assert not x.flags["C_CONTIGUOUS"]

    result = transform.apply_to_batch(x)
    assert result.flags["C_CONTIGUOUS"]
