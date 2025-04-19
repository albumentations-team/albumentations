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

    transform = Mosaic(target_size=target_size, p=1.0)

    # Input data structure expects a list for metadata
    data = {"image": img, "mosaic_metadata": []}

    result = transform(**data)
    transformed_img = result["image"]

    assert transformed_img.shape == img.shape
    np.testing.assert_array_equal(transformed_img, img)
