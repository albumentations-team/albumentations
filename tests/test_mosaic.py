import numpy as np
import pytest

from albumentations.augmentations.mixing.transforms import Mosaic


@pytest.mark.parametrize(
    "img_shape, target_size",
    [
        ((100, 80, 3), (100, 80)),  # Standard RGB
        ((64, 64, 1), (64, 64)),  # Grayscale
        ((128, 50), (128, 50)),   # Grayscale without channel dim
    ],
)
def test_mosaic_identity_single_image(img_shape: tuple[int, ...], target_size: tuple[int, int]) -> None:
    """Check Mosaic returns the original image when metadata is empty and target_size matches."""
    if len(img_shape) == 2:
        img = np.random.randint(0, 256, size=img_shape, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, size=img_shape, dtype=np.uint8)

    transform = Mosaic(target_size=target_size, grid_yx=(1, 1), p=1.0)

    # Input data structure expects a list for metadata
    data = {"image": img, "mosaic_metadata": []}

    result = transform(**data)
    transformed_img = result["image"]

    assert transformed_img.shape == img.shape
    np.testing.assert_array_equal(transformed_img, img)


# Separate parametrize for shapes, sizes, and fill values
@pytest.mark.parametrize(
    "img_shape, target_size, fill_value",
    [
        # Matching sizes
        ((100, 80, 3), (100, 80), 128), # RGB
        ((64, 64, 1), (64, 64), 50),   # Grayscale
        ((128, 50), (128, 50), 0),     # Grayscale 2D
        # Target smaller (cropping)
        ((100, 100, 3), (80, 80), 100),
        # Target larger (padding)
        ((50, 50, 1), (70, 70), 200),
        ((80, 60), (100, 100), 30), # Grayscale 2D padding
    ],
)
# Separate parametrize for grid dimensions
@pytest.mark.parametrize(
    "grid_yx",
    [
        (1, 1),
        (2, 2),
        (1, 2),
        (3, 2),
        (1, 3),
    ],
)
def test_mosaic_identity_monochromatic(
    img_shape: tuple[int, ...],
    target_size: tuple[int, int],
    grid_yx: tuple[int, int],
    fill_value: int,
) -> None:
    """Check Mosaic returns a uniform image if input is uniform (no metadata)."""
    if len(img_shape) == 2:
        img = np.full(img_shape, fill_value=fill_value, dtype=np.uint8)
        expected_output_shape = (*target_size,)
    else:
        img = np.full(img_shape, fill_value=fill_value, dtype=np.uint8)
        expected_output_shape = (*target_size, img_shape[-1])

    # Use fill=0 for padding to check if original value persists where possible
    transform = Mosaic(target_size=target_size, grid_yx=grid_yx, p=1.0, fill=0)

    data = {"image": img, "mosaic_metadata": []}
    result = transform(**data)
    transformed_img = result["image"]

    assert transformed_img.shape == expected_output_shape
    assert transformed_img.dtype == img.dtype

    # Create the expected output: uniform canvas of target size with fill_value
    # Note: If padding occurred (target > img size) with fill=0, the padded area will be 0.
    # This test asserts that the *non-padded* part retains the original fill_value.
    # A simpler assertion is that all pixels are *either* fill_value or 0 (padding value).
    is_padded_h = target_size[0] > img_shape[0]
    is_padded_w = target_size[1] > img_shape[1]

    if not is_padded_h and not is_padded_w:
        # If no padding, the output should be perfectly uniform with fill_value
        expected_output_img = np.full(expected_output_shape, fill_value=fill_value, dtype=np.uint8)
        np.testing.assert_array_equal(transformed_img, expected_output_img)
    else:
        # If padding occurred, check that pixels are either original fill or padding fill (0)
        assert np.all((transformed_img == fill_value) | (transformed_img == 0))

        # Additionally, check if the top-left corner (original image area) has the original value
        orig_h = img_shape[0]
        orig_w = img_shape[1]
        # Assuming PadIfNeeded with default top-left placement
        assert np.all(transformed_img[:orig_h, :orig_w] == fill_value)
