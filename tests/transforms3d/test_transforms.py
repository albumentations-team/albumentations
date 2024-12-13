import pytest
import numpy as np
import albumentations as A
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
        fill=0,
        fill_mask=0
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
            A.Pad3D: {"padding": 10},
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


@pytest.mark.parametrize(
    ["padding", "expected_shape"],
    [
        # Single int - pad all sides equally
        (2, (14, 14, 14, 1)),  # 10+2+2 = 14 for each dimension

        # 3-tuple - symmetric padding per dimension
        ((1, 2, 3), (12, 14, 16, 1)),  # d+2, h+4, w+6

        # 6-tuple - specific padding per side
        ((1, 2, 3, 2, 1, 4), (13, 15, 15, 1)),  # (d+3, h+3, w+7)

        # Zero padding
        (0, (10, 10, 10, 1)),  # Original shape
    ],
)
def test_pad3d_shapes(padding, expected_shape):
    volume = np.ones((10, 10, 10, 1), dtype=np.float32)
    augmentation = A.Pad3D(padding=padding)
    padded = augmentation(images=volume)["images"]
    assert padded.shape == expected_shape


@pytest.mark.parametrize(
    ["fill", "fill_mask"],
    [
        (0, 1),
        (1, 0),
    ],
)
def test_pad3d_fill_values(fill, fill_mask):
    volume = np.ones((3, 3, 3, 1), dtype=np.float32)
    masks = np.ones((3, 3, 3, 1), dtype=np.float32)

    augmentation = A.Pad3D(padding=1, fill=fill, fill_mask=fill_mask)
    transformed = augmentation(images=volume, masks=masks)

    padded_volume = transformed["images"]
    padded_mask = transformed["masks"]

    # Check fill values in padded regions
    assert np.all(padded_volume[0, :, :] == fill)  # front slice
    assert np.all(padded_mask[0, :, :] == fill_mask)  # front slice of mask


@pytest.mark.parametrize(
    "volume_shape",
    [
        (10, 10, 10),      # 3D
        (10, 10, 10, 1),   # 4D single channel
        (10, 10, 10, 3),   # 4D multi-channel
    ],
)
def test_pad3d_different_shapes(volume_shape):
    volume = np.ones(volume_shape, dtype=np.float32)
    augmentation = A.Pad3D(padding=2)
    padded = augmentation(images=volume)["images"]

    expected_shape = tuple(s + 4 for s in volume_shape[:3])  # +4 because padding=2 on each side
    if len(volume_shape) == 4:
        expected_shape += (volume_shape[3],)

    assert padded.shape == expected_shape


def test_pad3d_different_dtypes():
    for dtype in [np.uint8, np.float32]:
        volume = np.ones((5, 5, 5), dtype=dtype)
        augmentation = A.Pad3D(padding=1)
        padded = augmentation(images=volume)["images"]
        assert padded.dtype == dtype


def test_pad3d_preservation():
    """Test that the original data is preserved in the non-padded region"""
    volume = np.random.randint(0, 255, (5, 5, 5), dtype=np.uint8)
    augmentation = A.Pad3D(padding=2)
    padded = augmentation(images=volume)["images"]

    # Check if the original volume is preserved in the center
    np.testing.assert_array_equal(
        padded[2:-2, 2:-2, 2:-2],
        volume
    )

@pytest.mark.parametrize(
    ["pad3d_padding", "pad2d_padding"],
    [
        # 6-tuple for 3D (z_front, z_back, y_top, y_bottom, x_left, x_right) vs
        # 4-tuple for 2D (y_top, y_bottom, x_left, x_right)
        ((0, 0, 10, 10, 20, 20), (20, 10, 20, 10)),
        # 3-tuple for 3D (d,h,w) vs 2-tuple for 2D (w,h) - symmetric padding
        ((0, 10, 20), (20, 10)),
    ],
    ids=[
        "explicit_padding",
        "symmetric_padding",
    ]
)
def test_pad3d_2d_equivalence(pad3d_padding, pad2d_padding):
    """Test that Pad3D behaves like Pad when no z-padding is applied"""
    # Create a volume with multiple identical slices
    num_slices = 4
    slice_2d = np.random.randint(0, 256, (3, 3), dtype=np.uint8)
    volume_3d = np.stack([slice_2d] * num_slices)  # 10 identical slices

    # Apply 3D padding with no z-axis changes
    transform_3d = A.Pad3D(
        padding=pad3d_padding,
        fill=0,
        fill_mask=0
    )
    transformed_3d = transform_3d(images=volume_3d)

    # Apply 2D padding to a single slice
    transform_2d = A.Pad(
        padding=pad2d_padding,
        fill=0,
        fill_mask=0
    )
    transformed_2d = transform_2d(image=slice_2d)

    # Compare each slice of 3D result with 2D result
    for slice_idx in range(num_slices):
        np.testing.assert_array_equal(
            transformed_3d["images"][slice_idx],
            transformed_2d["image"]
        )

    # Also verify that z dimension hasn't changed
    assert transformed_3d["images"].shape[0] == volume_3d.shape[0]


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_3d_transforms(
        custom_arguments={
            A.PadIfNeeded3D: {"min_zyx": (300, 200, 400), "pad_divisor_zyx": (10, 10, 10), "position": "center", "fill": 10, "fill_mask": 20},
            A.Pad3D: {"padding": 10},
        },
        except_augmentations={
        },
    ),
)
def test_change_image(augmentation_cls, params):
    """Checks whether resulting image is different from the original one."""
    aug = A.Compose([augmentation_cls(p=1, **params)], seed=0)

    images = np.array([SQUARE_UINT8_IMAGE] * 4)
    original_images = images.copy()

    data = {
        "images": images,
        "masks": np.array([SQUARE_UINT8_IMAGE] * 3),
    }
    original_masks = data["masks"].copy()
    transformed = aug(**data)

    assert not np.array_equal(transformed["images"], original_images)
    assert not np.array_equal(transformed["masks"], original_masks)
