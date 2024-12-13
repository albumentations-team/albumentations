import pytest
import numpy as np
import albumentations as A
from albucore import to_float
import cv2

from tests.conftest import RECTANGULAR_UINT8_IMAGE, SQUARE_FLOAT_IMAGE, SQUARE_UINT8_IMAGE
from tests.utils import get_3d_transforms

@pytest.mark.parametrize(
    ["volume_shape", "min_zyx", "pad_divisor_zyx", "expected_shape"],
    [
        # Test no padding needed
        ((10, 100, 100), (10, 100, 100), None, (10, 100, 100)),

        # Test 2D-like behavior (no z padding)
        ((10, 100, 100), (10, 128, 128), None, (10, 128, 128)),

        # Test padding in all dimensions
        ((10, 100, 100), (16, 128, 128), None, (16, 128, 128)),

        # Test divisibility padding
        ((10, 100, 100), None, (8, 32, 32), (16, 128, 128)),

        # Test mixed min_size and divisibility
        ((10, 100, 100), (16, 128, 128), (8, 32, 32), (16, 128, 128)),
    ]
)
def test_pad_if_needed_3d_shapes(volume_shape, min_zyx, pad_divisor_zyx, expected_shape):
    volume = np.random.randint(0, 256, volume_shape, dtype=np.uint8)
    transform = A.PadIfNeeded3D(
        min_zyx=min_zyx,
        pad_divisor_zyx=pad_divisor_zyx,
        position="center",
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=0
    )
    transformed = transform(images=volume)
    assert transformed["images"].shape == expected_shape

@pytest.mark.parametrize("position", ["center", "random"])
def test_pad_if_needed_3d_positions(position):
    volume = np.ones((5, 50, 50), dtype=np.uint8)
    transform = A.PadIfNeeded3D(
        min_zyx=(10, 100, 100),
        position=position,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=0
    )
    transformed = transform(images=volume)
    # Check that the original volume is preserved somewhere in the padded volume
    assert np.any(transformed["images"] == 1)

def test_pad_if_needed_3d_2d_equivalence():
    """Test that PadIfNeeded3D behaves like PadIfNeeded when no z-padding is needed"""
    # Create a volume with multiple identical slices
    slice_2d = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    volume_3d = np.stack([slice_2d] * 10)

    # Apply 3D padding with no z-axis changes
    transform_3d = A.PadIfNeeded3D(
        min_zyx=(10, 128, 128),
        position="center",
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=0
    )
    transformed_3d = transform_3d(images=volume_3d)

    # Apply 2D padding to a single slice
    transform_2d = A.PadIfNeeded(
        min_height=128,
        min_width=128,
        position="center",
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0
    )
    transformed_2d = transform_2d(image=slice_2d)

    # Compare each slice of 3D result with 2D result
    for slice_idx in range(10):
        np.testing.assert_array_equal(
            transformed_3d["images"][slice_idx],
            transformed_2d["image"]
        )

def test_pad_if_needed_3d_fill_values():
    volume = np.zeros((5, 50, 50), dtype=np.uint8)
    mask = np.ones((5, 50, 50), dtype=np.uint8)

    transform = A.PadIfNeeded3D(
        min_zyx=(10, 100, 100),
        position="center",
        border_mode=cv2.BORDER_CONSTANT,
        fill=255,
        fill_mask=128
    )

    transformed = transform(images=volume, masks=mask)

    # Check fill values in padded regions
    assert np.all(transformed["images"][:, :25, :] == 255)  # top padding
    assert np.all(transformed["masks"][:, :25, :] == 128)  # top padding in mask



@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_3d_transforms(
        custom_arguments={
            A.PadIfNeeded3D: {"min_zyx": (4, 250, 230), "position": "center", "fill": 0, "fill_mask": 0},
        },
        except_augmentations={
        },
    ),
)
def test_augmentations_match_uint8_float32(augmentation_cls, params):
    image_uint8 = RECTANGULAR_UINT8_IMAGE
    image_float32 = image_uint8 / 255.0

    transform = A.Compose([augmentation_cls(p=1, **params)], seed=42)

    images = np.stack([image_uint8, image_uint8])

    data = {"images": images}

    transformed_uint8 = transform(**data)["images"]

    data["images"] = np.stack([image_float32, image_float32])

    transform.set_random_seed(42)
    transformed_float32 = transform(**data)["images"]

    np.testing.assert_array_almost_equal(transformed_uint8 / 255.0, transformed_float32, decimal=2)
