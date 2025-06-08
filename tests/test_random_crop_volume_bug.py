import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.crops.transforms import CropSizeError


def test_random_crop_with_pad_on_volume():
    """Test that RandomCrop with pad_if_needed=True works correctly on volumes after scale reduction."""
    # Create a small volume (8 frames, 100x100 grayscale video)
    volume = np.random.randint(0, 256, (8, 100, 100), dtype=np.uint8)

    # Create pipeline that will make the volume smaller than crop size
    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.RandomScale(scale_limit=(-0.2, 0.3), p=1.0),  # This can make image smaller
        A.RandomCrop(height=256, width=256, pad_if_needed=True, p=1.0),
    ], seed=137)

    # Run multiple times to ensure we hit the case where scale makes it smaller
    for _ in range(20):
        result = transform(volume=volume)
        transformed_volume = result['volume']
        # Check that output has the expected shape
        assert transformed_volume.shape == (8, 256, 256), f"Expected shape (8, 256, 256), got {transformed_volume.shape}"


def test_random_crop_without_pad_raises_error_on_small_volume():
    """Test that RandomCrop without pad_if_needed correctly raises error on small volumes."""
    # Create a volume smaller than crop size
    volume = np.random.randint(0, 256, (8, 100, 100), dtype=np.uint8)

    transform = A.RandomCrop(height=256, width=256, pad_if_needed=False, p=1.0)

    with pytest.raises(CropSizeError):
        transform(volume=volume)


def test_random_crop_edge_case_exact_size_after_scale():
    """Test RandomCrop when scaled size exactly matches crop size."""
    # Create volume that will scale to exactly 256x256
    volume = np.random.randint(0, 256, (8, 256, 256), dtype=np.uint8)

    # Scale will make it exactly 256x256 (scale factor 1.0)
    transform = A.Compose([
        A.RandomScale(scale_limit=(0.0, 0.0), p=1.0),  # No scaling
        A.RandomCrop(height=256, width=256, pad_if_needed=True, p=1.0),
    ], seed=137)

    result = transform(volume=volume)
    assert result['volume'].shape == (8, 256, 256)


def test_random_crop_pad_if_needed_false_with_larger_volume():
    """Test that RandomCrop works correctly when volume is larger than crop size."""
    # Create a large volume
    volume = np.random.randint(0, 256, (8, 300, 300), dtype=np.uint8)

    transform = A.RandomCrop(height=256, width=256, pad_if_needed=False, p=1.0)

    result = transform(volume=volume)
    assert result['volume'].shape == (8, 256, 256)


@pytest.mark.parametrize("scale_factor", [-0.15, -0.1, -0.05, 0.0, 0.1, 0.2])
def test_random_crop_various_scale_factors(scale_factor):
    """Test RandomCrop with different scale factors to ensure padding works correctly."""
    volume = np.random.randint(0, 256, (8, 256, 256), dtype=np.uint8)

    transform = A.Compose([
        A.RandomScale(scale_limit=(scale_factor, scale_factor), p=1.0),
        A.RandomCrop(height=256, width=256, pad_if_needed=True, p=1.0),
    ], seed=137)

    result = transform(volume=volume)
    assert result['volume'].shape == (8, 256, 256), f"Failed with scale_factor={scale_factor}"
