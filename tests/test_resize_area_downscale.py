import cv2
import numpy as np
import pytest

import albumentations as A


def get_downscale_image():
    """Create 100x100 test image for downscaling tests."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add recognizable pattern
    image[20:80, 20:80, 0] = 255  # Red square
    image[40:60, 40:60, 1] = 255  # Green square in the middle
    return image


def get_upscale_image():
    """Create 50x50 test image for upscaling tests."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    # Add recognizable pattern
    image[10:40, 10:40, 0] = 255  # Red square
    image[20:30, 20:30, 1] = 255  # Green square in the middle
    return image


def get_mask(size):
    """Create a mask of the specified size."""
    mask = np.zeros((size, size), dtype=np.uint8)
    # Add a simple pattern
    center = size // 2
    mask[center - size//4:center + size//4, center - size//4:center + size//4] = 1
    return mask


@pytest.mark.parametrize(
    ["transform_cls", "downscale_params", "upscale_params"],
    [
        (
            A.RandomScale,
            {"scale_limit": (-0.5, -0.5)},  # Fixed 0.5 downscale (100px→50px)
            {"scale_limit": (1.0, 1.0)}      # Fixed 2.0 upscale (50px→100px)
        ),
        (
            A.Resize,
            {"height": 50, "width": 50},     # Downscale 100px→50px
            {"height": 100, "width": 100}    # Upscale 50px→100px
        ),
        (
            A.LongestMaxSize,
            {"max_size": 50},                # Downscale 100px→50px
            {"max_size": 100}                # Upscale 50px→100px
        ),
        (
            A.SmallestMaxSize,
            {"max_size": 50},                # Downscale 100px→50px
            {"max_size": 100}                # Upscale 50px→100px
        ),
    ],
)
@pytest.mark.parametrize(
    "interpolation",
    [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ],
)
class TestAreaForDownscaleOutput:
    def test_downscale_area_option_matches_area_interp(self, transform_cls, downscale_params, upscale_params, interpolation):
        """Test that specified interpolation with area_for_downscale='image' matches AREA without area_for_downscale for downscaling."""
        image = get_downscale_image()

        # Transform 1: Specified interpolation with area_for_downscale="image"
        transform1 = transform_cls(
            interpolation=interpolation,
            area_for_downscale="image",
            p=1.0,
            **downscale_params
        )
        result1 = transform1(image=image)

        # Transform 2: AREA interpolation without area_for_downscale
        transform2 = transform_cls(
            interpolation=cv2.INTER_AREA,
            area_for_downscale=None,
            p=1.0,
            **downscale_params
        )
        result2 = transform2(image=image)

        # The image outputs should be identical
        np.testing.assert_array_equal(
            result1["image"],
            result2["image"],
            err_msg=f"Downscale outputs differ for {transform_cls.__name__} with {interpolation}+area_for_image vs AREA"
        )

    def test_downscale_area_option_for_mask(self, transform_cls, downscale_params, upscale_params, interpolation):
        """Test that area_for_downscale='image_mask' affects mask interpolation."""
        image = get_downscale_image()
        mask = get_mask(100)

        # Transform 1: With area_for_downscale="image" (should not affect mask)
        transform1 = transform_cls(
            interpolation=interpolation,
            mask_interpolation=cv2.INTER_NEAREST,
            area_for_downscale="image",
            p=1.0,
            **downscale_params
        )
        result1 = transform1(image=image, mask=mask)

        # Transform 2: With area_for_downscale="image_mask" (should affect mask)
        transform2 = transform_cls(
            interpolation=interpolation,
            mask_interpolation=cv2.INTER_NEAREST,
            area_for_downscale="image_mask",
            p=1.0,
            **downscale_params
        )
        result2 = transform2(image=image, mask=mask)

        # Both should produce identical images (both using AREA for downscaling)
        np.testing.assert_array_equal(
            result1["image"],
            result2["image"],
            err_msg=f"Image outputs differ between area_for_downscale='image' and 'image_mask'"
        )

        # Transform 3: With AREA mask interpolation (should match area_for_downscale="image_mask")
        transform3 = transform_cls(
            interpolation=interpolation,
            mask_interpolation=cv2.INTER_AREA,
            area_for_downscale=None,
            p=1.0,
            **downscale_params
        )
        result3 = transform3(image=image, mask=mask)

        # Mask from transform2 (NEAREST+area_for_image_mask) should match transform3 (AREA)
        np.testing.assert_array_equal(
            result2["mask"],
            result3["mask"],
            err_msg=f"Mask with NEAREST+area_for_image_mask should match mask with AREA interpolation"
        )

    def test_upscale_ignores_area_for_downscale(self, transform_cls, downscale_params, upscale_params, interpolation):
        """Test that area_for_downscale has no effect when upscaling."""
        image = get_upscale_image()

        # Transform 1: With area_for_downscale="image"
        transform1 = transform_cls(
            interpolation=interpolation,
            area_for_downscale="image",
            p=1.0,
            **upscale_params
        )
        result1 = transform1(image=image)

        # Transform 2: Without area_for_downscale
        transform2 = transform_cls(
            interpolation=interpolation,
            area_for_downscale=None,
            p=1.0,
            **upscale_params
        )
        result2 = transform2(image=image)

        # Outputs should be identical since area_for_downscale shouldn't affect upscaling
        np.testing.assert_array_equal(
            result1["image"],
            result2["image"],
            err_msg=f"Upscale outputs differ with/without area_for_downscale"
        )
