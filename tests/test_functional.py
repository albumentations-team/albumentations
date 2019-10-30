from __future__ import absolute_import

import cv2
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal_nulp

import albumentations.augmentations.functional as F
from albumentations.augmentations.bbox_utils import filter_bboxes
from tests.utils import convert_2d_to_target_format


@pytest.mark.parametrize("target", ["image", "mask"])
def test_vflip(target):
    img = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    expected = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    flipped_img = F.vflip(img)
    assert np.array_equal(flipped_img, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_vflip_float(target):
    img = np.array([[0.4, 0.4, 0.4], [0.0, 0.4, 0.4], [0.0, 0.0, 0.4]], dtype=np.float32)
    expected = np.array([[0.0, 0.0, 0.4], [0.0, 0.4, 0.4], [0.4, 0.4, 0.4]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    flipped_img = F.vflip(img)
    assert_array_almost_equal_nulp(flipped_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_hflip(target):
    img = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    expected = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    flipped_img = F.hflip(img)
    assert np.array_equal(flipped_img, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_hflip_float(target):
    img = np.array([[0.4, 0.4, 0.4], [0.0, 0.4, 0.4], [0.0, 0.0, 0.4]], dtype=np.float32)
    expected = np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    flipped_img = F.hflip(img)
    assert_array_almost_equal_nulp(flipped_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
@pytest.mark.parametrize(["code", "func"], [[0, F.vflip], [1, F.hflip], [-1, lambda img: F.vflip(F.hflip(img))]])
def test_random_flip(code, func, target):
    img = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    img = convert_2d_to_target_format([img], target=target)
    assert np.array_equal(F.random_flip(img, code), func(img))


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
@pytest.mark.parametrize(["code", "func"], [[0, F.vflip], [1, F.hflip], [-1, lambda img: F.vflip(F.hflip(img))]])
def test_random_flip_float(code, func, target):
    img = np.array([[0.4, 0.4, 0.4], [0.0, 0.4, 0.4], [0.0, 0.0, 0.4]], dtype=np.float32)
    img = convert_2d_to_target_format([img], target=target)
    assert_array_almost_equal_nulp(F.random_flip(img, code), func(img))


@pytest.mark.parametrize(["input_shape", "expected_shape"], [[(128, 64), (64, 128)], [(128, 64, 3), (64, 128, 3)]])
def test_transpose(input_shape, expected_shape):
    img = np.random.randint(low=0, high=256, size=input_shape, dtype=np.uint8)
    transposed = F.transpose(img)
    assert transposed.shape == expected_shape


@pytest.mark.parametrize(["input_shape", "expected_shape"], [[(128, 64), (64, 128)], [(128, 64, 3), (64, 128, 3)]])
def test_transpose_float(input_shape, expected_shape):
    img = np.random.uniform(low=0.0, high=1.0, size=input_shape).astype("float32")
    transposed = F.transpose(img)
    assert transposed.shape == expected_shape


@pytest.mark.parametrize("target", ["image", "mask"])
def test_rot90(target):
    img = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.uint8)
    expected = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    rotated = F.rot90(img, factor=1)
    assert np.array_equal(rotated, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_rot90_float(target):
    img = np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 0.4], [0.0, 0.0, 0.4]], dtype=np.float32)
    expected = np.array([[0.4, 0.4, 0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    rotated = F.rot90(img, factor=1)
    assert_array_almost_equal_nulp(rotated, expected)


def test_normalize():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    normalized = F.normalize(img, mean=50, std=3)
    expected = (np.ones((100, 100, 3), dtype=np.float32) * 127 / 255 - 50) / 3
    assert_array_almost_equal_nulp(normalized, expected)


def test_normalize_float():
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.4
    normalized = F.normalize(img, mean=50, std=3, max_pixel_value=1.0)
    expected = (np.ones((100, 100, 3), dtype=np.float32) * 0.4 - 50) / 3
    assert_array_almost_equal_nulp(normalized, expected)


def test_compare_rotate_and_shift_scale_rotate(image):
    rotated_img_1 = F.rotate(image, angle=60)
    rotated_img_2 = F.shift_scale_rotate(image, angle=60, scale=1, dx=0, dy=0)
    assert np.array_equal(rotated_img_1, rotated_img_2)


def test_compare_rotate_float_and_shift_scale_rotate_float(float_image):
    rotated_img_1 = F.rotate(float_image, angle=60)
    rotated_img_2 = F.shift_scale_rotate(float_image, angle=60, scale=1, dx=0, dy=0)
    assert np.array_equal(rotated_img_1, rotated_img_2)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_center_crop(target):
    img = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.uint8)
    expected = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    cropped_img = F.center_crop(img, 2, 2)
    assert np.array_equal(cropped_img, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_center_crop_float(target):
    img = np.array(
        [[0.4, 0.4, 0.4, 0.4], [0.0, 0.4, 0.4, 0.4], [0.0, 0.0, 0.4, 0.4], [0.0, 0.0, 0.0, 0.4]], dtype=np.float32
    )
    expected = np.array([[0.4, 0.4], [0.0, 0.4]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    cropped_img = F.center_crop(img, 2, 2)
    assert_array_almost_equal_nulp(cropped_img, expected)


def test_center_crop_with_incorrectly_large_crop_size():
    img = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        F.center_crop(img, 8, 8)
    assert str(exc_info.value) == "Requested crop size (8, 8) is larger than the image size (4, 4)"


@pytest.mark.parametrize("target", ["image", "mask"])
def test_random_crop(target):
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8)
    expected = np.array([[5, 6], [9, 10]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    cropped_img = F.random_crop(img, crop_height=2, crop_width=2, h_start=0.5, w_start=0)
    assert np.array_equal(cropped_img, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_random_crop_float(target):
    img = np.array(
        [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.10, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]],
        dtype=np.float32,
    )
    expected = np.array([[0.05, 0.06], [0.09, 0.10]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    cropped_img = F.random_crop(img, crop_height=2, crop_width=2, h_start=0.5, w_start=0)
    assert_array_almost_equal_nulp(cropped_img, expected)


def test_random_crop_with_incorrectly_large_crop_size():
    img = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        F.random_crop(img, crop_height=8, crop_width=8, h_start=0, w_start=0)
    assert str(exc_info.value) == "Requested crop size (8, 8) is larger than the image size (4, 4)"


def test_clip():
    img = np.array([[-300, 0], [100, 400]], dtype=np.float32)
    expected = np.array([[0, 0], [100, 255]], dtype=np.float32)
    clipped = F.clip(img, dtype=np.uint8, maxval=255)
    assert np.array_equal(clipped, expected)


def test_clip_float():
    img = np.array([[-0.02, 0], [0.5, 2.2]], dtype=np.float32)
    expected = np.array([[0, 0], [0.5, 1.0]], dtype=np.float32)
    clipped = F.clip(img, dtype=np.float32, maxval=1.0)
    assert_array_almost_equal_nulp(clipped, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_pad(target):
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.array([[4, 3, 4, 3], [2, 1, 2, 1], [4, 3, 4, 3], [2, 1, 2, 1]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    padded = F.pad(img, min_height=4, min_width=4)
    assert np.array_equal(padded, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_pad_float(target):
    img = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    expected = np.array(
        [[0.4, 0.3, 0.4, 0.3], [0.2, 0.1, 0.2, 0.1], [0.4, 0.3, 0.4, 0.3], [0.2, 0.1, 0.2, 0.1]], dtype=np.float32
    )
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    padded_img = F.pad(img, min_height=4, min_width=4)
    assert_array_almost_equal_nulp(padded_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_rotate_from_shift_scale_rotate(target):
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8)
    expected = np.array([[0, 0, 0, 0], [4, 8, 12, 16], [3, 7, 11, 15], [2, 6, 10, 14]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    rotated_img = F.shift_scale_rotate(
        img, angle=90, scale=1, dx=0, dy=0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT
    )
    assert np.array_equal(rotated_img, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_rotate_float_from_shift_scale_rotate(target):
    img = np.array(
        [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.10, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]],
        dtype=np.float32,
    )
    expected = np.array(
        [[0.00, 0.00, 0.00, 0.00], [0.04, 0.08, 0.12, 0.16], [0.03, 0.07, 0.11, 0.15], [0.02, 0.06, 0.10, 0.14]],
        dtype=np.float32,
    )
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    rotated_img = F.shift_scale_rotate(
        img, angle=90, scale=1, dx=0, dy=0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT
    )
    assert_array_almost_equal_nulp(rotated_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_scale_from_shift_scale_rotate(target):
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8)
    expected = np.array([[6, 7, 7, 8], [10, 11, 11, 12], [10, 11, 11, 12], [14, 15, 15, 16]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    scaled_img = F.shift_scale_rotate(
        img, angle=0, scale=2, dx=0, dy=0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT
    )
    assert np.array_equal(scaled_img, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_scale_float_from_shift_scale_rotate(target):
    img = np.array(
        [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.10, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]],
        dtype=np.float32,
    )
    expected = np.array(
        [[0.06, 0.07, 0.07, 0.08], [0.10, 0.11, 0.11, 0.12], [0.10, 0.11, 0.11, 0.12], [0.14, 0.15, 0.15, 0.16]],
        dtype=np.float32,
    )
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    scaled_img = F.shift_scale_rotate(
        img, angle=0, scale=2, dx=0, dy=0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT
    )
    assert_array_almost_equal_nulp(scaled_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_shift_x_from_shift_scale_rotate(target):
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8)
    expected = np.array([[0, 0, 1, 2], [0, 0, 5, 6], [0, 0, 9, 10], [0, 0, 13, 14]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    shifted_along_x_img = F.shift_scale_rotate(
        img, angle=0, scale=1, dx=0.5, dy=0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT
    )
    assert np.array_equal(shifted_along_x_img, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_shift_x_float_from_shift_scale_rotate(target):
    img = np.array(
        [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.10, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]],
        dtype=np.float32,
    )
    expected = np.array(
        [[0.00, 0.00, 0.01, 0.02], [0.00, 0.00, 0.05, 0.06], [0.00, 0.00, 0.09, 0.10], [0.00, 0.00, 0.13, 0.14]],
        dtype=np.float32,
    )
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    shifted_along_x_img = F.shift_scale_rotate(
        img, angle=0, scale=1, dx=0.5, dy=0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT
    )
    assert_array_almost_equal_nulp(shifted_along_x_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_shift_y_from_shift_scale_rotate(target):
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.uint8)
    expected = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    shifted_along_y_img = F.shift_scale_rotate(
        img, angle=0, scale=1, dx=0, dy=0.5, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT
    )
    assert np.array_equal(shifted_along_y_img, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_shift_y_float_from_shift_scale_rotate(target):
    img = np.array(
        [[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08], [0.09, 0.10, 0.11, 0.12], [0.13, 0.14, 0.15, 0.16]],
        dtype=np.float32,
    )
    expected = np.array(
        [[0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00], [0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 0.07, 0.08]],
        dtype=np.float32,
    )
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    shifted_along_y_img = F.shift_scale_rotate(
        img, angle=0, scale=1, dx=0, dy=0.5, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT
    )
    assert_array_almost_equal_nulp(shifted_along_y_img, expected)


@pytest.mark.parametrize(
    ["shift_params", "expected"], [[(-10, 0, 10), (117, 127, 137)], [(-200, 0, 200), (0, 127, 255)]]
)
def test_shift_rgb(shift_params, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    r_shift, g_shift, b_shift = shift_params
    img = F.shift_rgb(img, r_shift=r_shift, g_shift=g_shift, b_shift=b_shift)
    expected_r, expected_g, expected_b = expected
    assert img.dtype == np.dtype("uint8")
    assert (img[:, :, 0] == expected_r).all()
    assert (img[:, :, 1] == expected_g).all()
    assert (img[:, :, 2] == expected_b).all()


@pytest.mark.parametrize(
    ["shift_params", "expected"], [[(-0.1, 0, 0.1), (0.3, 0.4, 0.5)], [(-0.6, 0, 0.6), (0, 0.4, 1.0)]]
)
def test_shift_rgb_float(shift_params, expected):
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.4
    r_shift, g_shift, b_shift = shift_params
    img = F.shift_rgb(img, r_shift=r_shift, g_shift=g_shift, b_shift=b_shift)
    expected_r, expected_g, expected_b = [
        np.ones((100, 100), dtype=np.float32) * channel_value for channel_value in expected
    ]
    assert img.dtype == np.dtype("float32")
    assert_array_almost_equal_nulp(img[:, :, 0], expected_r)
    assert_array_almost_equal_nulp(img[:, :, 1], expected_g)
    assert_array_almost_equal_nulp(img[:, :, 2], expected_b)


@pytest.mark.parametrize(["alpha", "expected"], [(1.5, 190), (3, 255)])
def test_random_contrast(alpha, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    img = F.brightness_contrast_adjust(img, alpha=alpha)
    assert img.dtype == np.dtype("uint8")
    assert (img == expected).all()


@pytest.mark.parametrize(["alpha", "expected"], [(1.5, 0.6), (3, 1.0)])
def test_random_contrast_float(alpha, expected):
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.4
    expected = np.ones((100, 100, 3), dtype=np.float32) * expected
    img = F.brightness_contrast_adjust(img, alpha=alpha)
    assert img.dtype == np.dtype("float32")
    assert_array_almost_equal_nulp(img, expected)


@pytest.mark.parametrize(["beta", "expected"], [(-0.5, 50), (0.25, 125)])
def test_random_brightness(beta, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8) * 100
    img = F.brightness_contrast_adjust(img, beta=beta)
    assert img.dtype == np.dtype("uint8")
    assert (img == expected).all()


@pytest.mark.parametrize(["beta", "expected"], [(0.2, 0.48), (-0.1, 0.36)])
def test_random_brightness_float(beta, expected):
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.4
    expected = np.ones_like(img) * expected
    img = F.brightness_contrast_adjust(img, beta=beta)
    assert img.dtype == np.dtype("float32")
    assert_array_almost_equal_nulp(img, expected)


@pytest.mark.parametrize(["gamma", "expected"], [(1, 1), (10, 146)])
def test_gamma_transform(gamma, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8)
    img = F.gamma_transform(img, gamma=gamma)
    assert img.dtype == np.dtype("uint8")
    assert (img == expected).all()


@pytest.mark.parametrize(["gamma", "expected"], [(1, 0.4), (10, 0.00010486)])
def test_gamma_transform_float(gamma, expected):
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.4
    expected = np.ones((100, 100, 3), dtype=np.float32) * expected
    img = F.gamma_transform(img, gamma=gamma)
    assert img.dtype == np.dtype("float32")
    assert np.allclose(img, expected)


@pytest.mark.parametrize(["dtype", "divider"], [(np.uint8, 255), (np.uint16, 65535), (np.uint32, 4294967295)])
def test_to_float_without_max_value_specified(dtype, divider):
    img = np.ones((100, 100, 3), dtype=dtype)
    expected = img.astype("float32") / divider
    assert_array_almost_equal_nulp(F.to_float(img), expected)


@pytest.mark.parametrize("max_value", [255.0, 65535.0, 4294967295.0])
def test_to_float_with_max_value_specified(max_value):
    img = np.ones((100, 100, 3), dtype=np.uint16)
    expected = img.astype("float32") / max_value
    assert_array_almost_equal_nulp(F.to_float(img, max_value=max_value), expected)


def test_to_float_unknown_dtype():
    img = np.ones((100, 100, 3), dtype=np.int16)
    with pytest.raises(RuntimeError) as exc_info:
        F.to_float(img)
    assert str(exc_info.value) == (
        "Can't infer the maximum value for dtype int16. You need to specify the maximum value manually by passing "
        "the max_value argument"
    )


@pytest.mark.parametrize("max_value", [255.0, 65535.0, 4294967295.0])
def test_to_float_unknown_dtype_with_max_value(max_value):
    img = np.ones((100, 100, 3), dtype=np.int16)
    expected = img.astype("float32") / max_value
    assert_array_almost_equal_nulp(F.to_float(img, max_value=max_value), expected)


@pytest.mark.parametrize(["dtype", "multiplier"], [(np.uint8, 255), (np.uint16, 65535), (np.uint32, 4294967295)])
def test_from_float_without_max_value_specified(dtype, multiplier):
    img = np.ones((100, 100, 3), dtype=np.float32)
    expected = (img * multiplier).astype(dtype)
    assert_array_almost_equal_nulp(F.from_float(img, np.dtype(dtype)), expected)


@pytest.mark.parametrize("max_value", [255.0, 65535.0, 4294967295.0])
def test_from_float_with_max_value_specified(max_value):
    img = np.ones((100, 100, 3), dtype=np.float32)
    expected = (img * max_value).astype(np.uint32)
    assert_array_almost_equal_nulp(F.from_float(img, dtype=np.uint32, max_value=max_value), expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_scale(target):
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.uint8)
    expected = np.array(
        [
            [1, 1, 2, 2, 3, 3],
            [2, 2, 2, 3, 3, 4],
            [3, 3, 4, 4, 5, 5],
            [5, 5, 5, 6, 6, 7],
            [6, 6, 7, 7, 8, 8],
            [8, 8, 8, 9, 9, 10],
            [9, 9, 10, 10, 11, 11],
            [10, 10, 11, 11, 12, 12],
        ],
        dtype=np.uint8,
    )

    img, expected = convert_2d_to_target_format([img, expected], target=target)
    scaled = F.scale(img, scale=2, interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(scaled, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_longest_max_size(target):
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.uint8)
    expected = np.array([[2, 3], [6, 7], [10, 11]], dtype=np.uint8)

    img, expected = convert_2d_to_target_format([img, expected], target=target)
    scaled = F.longest_max_size(img, max_size=3, interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(scaled, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_smallest_max_size(target):
    img = np.array(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]], dtype=np.uint8
    )
    expected = np.array([[2, 4, 5, 7], [10, 11, 13, 14], [17, 19, 20, 22]], dtype=np.uint8)

    img, expected = convert_2d_to_target_format([img, expected], target=target)
    scaled = F.smallest_max_size(img, max_size=3, interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(scaled, expected)


def test_from_float_unknown_dtype():
    img = np.ones((100, 100, 3), dtype=np.float32)
    with pytest.raises(RuntimeError) as exc_info:
        F.from_float(img, np.dtype(np.int16))
    assert str(exc_info.value) == (
        "Can't infer the maximum value for dtype int16. You need to specify the maximum value manually by passing "
        "the max_value argument"
    )


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_default_interpolation(target):
    img = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.uint8)
    expected = np.array([[2, 2], [4, 4]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = F.resize(img, 2, 2)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert np.array_equal(resized_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_nearest_interpolation(target):
    img = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.uint8)
    expected = np.array([[1, 1], [3, 3]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = F.resize(img, 2, 2, interpolation=cv2.INTER_NEAREST)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert np.array_equal(resized_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_different_height_and_width(target):
    img = np.ones((100, 100), dtype=np.uint8)
    img = convert_2d_to_target_format([img], target=target)
    resized_img = F.resize(img, height=20, width=30)
    height, width = resized_img.shape[:2]
    assert height == 20
    assert width == 30
    if target == "image":
        num_channels = resized_img.shape[2]
        assert num_channels == 3


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_default_interpolation_float(target):
    img = np.array(
        [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4]], dtype=np.float32
    )
    expected = np.array([[0.15, 0.15], [0.35, 0.35]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = F.resize(img, 2, 2)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert_array_almost_equal_nulp(resized_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_nearest_interpolation_float(target):
    img = np.array(
        [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4]], dtype=np.float32
    )
    expected = np.array([[0.1, 0.1], [0.3, 0.3]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = F.resize(img, 2, 2, interpolation=cv2.INTER_NEAREST)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert np.array_equal(resized_img, expected)


def test_bbox_vflip():
    assert F.bbox_vflip((0.1, 0.2, 0.6, 0.5), 100, 200) == (0.1, 0.5, 0.6, 0.8)


def test_bbox_hflip():
    assert F.bbox_hflip((0.1, 0.2, 0.6, 0.5), 100, 200) == (0.4, 0.2, 0.9, 0.5)


@pytest.mark.parametrize(
    ["code", "func"],
    [
        [0, F.bbox_vflip],
        [1, F.bbox_hflip],
        [-1, lambda bbox, rows, cols: F.bbox_vflip(F.bbox_hflip(bbox, rows, cols), rows, cols)],
    ],
)
def test_bbox_flip(code, func):
    rows, cols = 100, 200
    bbox = [0.1, 0.2, 0.6, 0.5]
    assert F.bbox_flip(bbox, code, rows, cols) == func(bbox, rows, cols)


def test_crop_bbox_by_coords():
    cropped_bbox = F.crop_bbox_by_coords((0.5, 0.2, 0.9, 0.7), (18, 18, 82, 82), 64, 64, 100, 100)
    assert cropped_bbox == (0.5, 0.03125, 1.125, 0.8125)


def test_bbox_center_crop():
    cropped_bbox = F.bbox_center_crop((0.5, 0.2, 0.9, 0.7), 64, 64, 100, 100)
    assert cropped_bbox == (0.5, 0.03125, 1.125, 0.8125)


def test_bbox_crop():
    cropped_bbox = F.bbox_crop([0.5, 0.2, 0.9, 0.7], 24, 24, 64, 64, 100, 100)
    assert cropped_bbox == (0.65, -0.1, 1.65, 1.15)


def test_bbox_random_crop():
    cropped_bbox = F.bbox_random_crop([0.5, 0.2, 0.9, 0.7], 80, 80, 0.2, 0.1, 100, 100)
    assert cropped_bbox == (0.6, 0.2, 1.1, 0.825)


def test_bbox_rot90():
    assert F.bbox_rot90((0.1, 0.2, 0.3, 0.4), 0, 100, 200) == (0.1, 0.2, 0.3, 0.4)
    assert F.bbox_rot90((0.1, 0.2, 0.3, 0.4), 1, 100, 200) == (0.2, 0.7, 0.4, 0.9)
    assert F.bbox_rot90((0.1, 0.2, 0.3, 0.4), 2, 100, 200) == (0.7, 0.6, 0.9, 0.8)
    assert F.bbox_rot90((0.1, 0.2, 0.3, 0.4), 3, 100, 200) == (0.6, 0.1, 0.8, 0.3)


def test_bbox_transpose():
    assert np.allclose(F.bbox_transpose((0.7, 0.1, 0.8, 0.4), 0, 100, 200), (0.1, 0.7, 0.4, 0.8))
    assert np.allclose(F.bbox_transpose((0.7, 0.1, 0.8, 0.4), 1, 100, 200), (0.6, 0.2, 0.9, 0.3))


@pytest.mark.parametrize(
    ["bboxes", "min_area", "min_visibility", "target"],
    [
        (
            [(0.1, 0.5, 1.1, 0.9), (-0.1, 0.5, 0.8, 0.9), (0.1, 0.5, 0.8, 0.9)],
            0,
            0,
            [(0.1, 0.5, 1.0, 0.9), (0.0, 0.5, 0.8, 0.9), (0.1, 0.5, 0.8, 0.9)],
        ),
        ([(0.1, 0.5, 0.8, 0.9), (0.4, 0.5, 0.5, 0.6)], 150, 0, [(0.1, 0.5, 0.8, 0.9)]),
        ([(0.1, 0.5, 0.8, 0.9), (0.4, 0.9, 0.5, 1.6)], 0, 0.75, [(0.1, 0.5, 0.8, 0.9)]),
        ([(0.1, 0.5, 0.8, 0.9), (0.4, 0.7, 0.5, 1.1)], 0, 0.7, [(0.1, 0.5, 0.8, 0.9), (0.4, 0.7, 0.5, 1.0)]),
    ],
)
def test_filter_bboxes(bboxes, min_area, min_visibility, target):
    filtered_bboxes = filter_bboxes(bboxes, min_area=min_area, min_visibility=min_visibility, rows=100, cols=100)
    assert filtered_bboxes == target


def test_fun_max_size():
    target_width = 256

    img = np.empty((330, 49), dtype=np.uint8)
    out = F.smallest_max_size(img, target_width, interpolation=cv2.INTER_LINEAR)

    assert out.shape == (1724, target_width)


def test_is_rgb_image():
    image = np.ones((5, 5, 3), dtype=np.uint8)
    assert F.is_rgb_image(image)

    multispectral_image = np.ones((5, 5, 4), dtype=np.uint8)
    assert not F.is_rgb_image(multispectral_image)

    gray_image = np.ones((5, 5), dtype=np.uint8)
    assert not F.is_rgb_image(gray_image)

    gray_image = np.ones((5, 5, 1), dtype=np.uint8)
    assert not F.is_rgb_image(gray_image)


def test_is_grayscale_image():
    image = np.ones((5, 5, 3), dtype=np.uint8)
    assert not F.is_grayscale_image(image)

    multispectral_image = np.ones((5, 5, 4), dtype=np.uint8)
    assert not F.is_grayscale_image(multispectral_image)

    gray_image = np.ones((5, 5), dtype=np.uint8)
    assert F.is_grayscale_image(gray_image)

    gray_image = np.ones((5, 5, 1), dtype=np.uint8)
    assert F.is_grayscale_image(gray_image)


def test_is_multispectral_image():
    image = np.ones((5, 5, 3), dtype=np.uint8)
    assert not F.is_multispectral_image(image)

    multispectral_image = np.ones((5, 5, 4), dtype=np.uint8)
    assert F.is_multispectral_image(multispectral_image)

    gray_image = np.ones((5, 5), dtype=np.uint8)
    assert not F.is_multispectral_image(gray_image)

    gray_image = np.ones((5, 5, 1), dtype=np.uint8)
    assert not F.is_multispectral_image(gray_image)


def test_brightness_contrast():
    dtype = np.uint8
    min_value = np.iinfo(dtype).min
    max_value = np.iinfo(dtype).max

    image_uint8 = np.random.randint(min_value, max_value, size=(5, 5, 3), dtype=dtype)

    assert np.array_equal(F.brightness_contrast_adjust(image_uint8), F._brightness_contrast_adjust_uint(image_uint8))

    assert np.array_equal(
        F._brightness_contrast_adjust_non_uint(image_uint8), F._brightness_contrast_adjust_uint(image_uint8)
    )

    dtype = np.uint16
    min_value = np.iinfo(dtype).min
    max_value = np.iinfo(dtype).max

    image_uint16 = np.random.randint(min_value, max_value, size=(5, 5, 3), dtype=dtype)

    assert np.array_equal(
        F.brightness_contrast_adjust(image_uint16), F._brightness_contrast_adjust_non_uint(image_uint16)
    )

    F.brightness_contrast_adjust(image_uint16)

    dtype = np.uint32
    min_value = np.iinfo(dtype).min
    max_value = np.iinfo(dtype).max

    image_uint32 = np.random.randint(min_value, max_value, size=(5, 5, 3), dtype=dtype)

    assert np.array_equal(
        F.brightness_contrast_adjust(image_uint32), F._brightness_contrast_adjust_non_uint(image_uint32)
    )

    image_float = np.random.random((5, 5, 3))

    assert np.array_equal(
        F.brightness_contrast_adjust(image_float), F._brightness_contrast_adjust_non_uint(image_float)
    )


def test_swap_tiles_on_image_with_empty_tiles():
    img = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.uint8)

    result_img = F.swap_tiles_on_image(img, [])

    assert np.array_equal(img, result_img)


def test_swap_tiles_on_image_with_non_empty_tiles():
    img = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.uint8)

    tiles = np.array([[0, 0, 2, 2, 2, 2], [2, 2, 0, 0, 2, 2]])

    target = np.array([[3, 3, 1, 1], [4, 4, 2, 2], [3, 3, 1, 1], [4, 4, 2, 2]], dtype=np.uint8)

    result_img = F.swap_tiles_on_image(img, tiles)

    assert np.array_equal(result_img, target)


@pytest.mark.parametrize("dtype", list(F.MAX_VALUES_BY_DTYPE.keys()))
def test_solarize(dtype):
    max_value = F.MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.dtype("float32"):
        img = np.arange(2 ** 10, dtype=np.float32) / (2 ** 10)
        img = img.reshape([2 ** 5, 2 ** 5])
    else:
        max_count = 1024
        count = min(max_value + 1, 1024)
        step = max(1, (max_value + 1) // max_count)
        shape = [int(np.sqrt(count))] * 2
        img = np.arange(0, max_value + 1, step, dtype=dtype).reshape(shape)

    for threshold in [0, max_value // 3, max_value // 3 * 2, max_value, max_value + 1]:
        check_img = img.copy()
        cond = check_img >= threshold
        check_img[cond] = max_value - check_img[cond]

        result_img = F.solarize(img, threshold=threshold)

        assert np.all(np.isclose(result_img, check_img))
        assert np.min(result_img) >= 0
        assert np.max(result_img) <= max_value


def test_posterize_checks():
    img = np.random.random([256, 256, 3])
    with pytest.raises(AssertionError) as exc_info:
        F.posterize(img, 4)
    assert str(exc_info.value) == "Image must have uint8 channel type"

    img = np.random.randint(0, 256, [256, 256], dtype=np.uint8)
    with pytest.raises(AssertionError) as exc_info:
        F.posterize(img, [1, 2, 3])
    assert str(exc_info.value) == "If bits is iterable image must be RGB"


def test_equalize_checks():
    img = np.random.randint(0, 255, [256, 256], dtype=np.uint8)

    with pytest.raises(ValueError) as exc_info:
        F.equalize(img, mode="other")
    assert str(exc_info.value) == "Unsupported equalization mode. Supports: ['cv', 'pil']. Got: other"

    mask = np.random.randint(0, 1, [256, 256, 3], dtype=np.bool)
    with pytest.raises(ValueError) as exc_info:
        F.equalize(img, mask=mask)
    assert str(exc_info.value) == "Wrong mask shape. Image shape: {}. Mask shape: {}".format(img.shape, mask.shape)

    img = np.random.randint(0, 255, [256, 256, 3], dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        F.equalize(img, mask=mask, by_channels=False)
    assert str(exc_info.value) == "When by_channels=False only 1-channel mask supports. " "Mask shape: {}".format(
        mask.shape
    )

    img = np.random.random([256, 256, 3])
    with pytest.raises(AssertionError) as exc_info:
        F.equalize(img, mask=mask, by_channels=False)
    assert str(exc_info.value) == "Image must have uint8 channel type"


def test_equalize_grayscale():
    img = np.random.randint(0, 255, [256, 256], dtype=np.uint8)
    assert np.all(cv2.equalizeHist(img) == F.equalize(img, mode="cv"))


def test_equalize_rgb():
    img = np.random.randint(0, 255, [256, 256, 3], dtype=np.uint8)

    _img = img.copy()
    for i in range(3):
        _img[..., i] = cv2.equalizeHist(_img[..., i])
    assert np.all(_img == F.equalize(img, mode="cv"))

    _img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img_cv = _img.copy()
    img_cv[..., 0] = cv2.equalizeHist(_img[..., 0])
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_YCrCb2RGB)
    assert np.all(img_cv == F.equalize(img, mode="cv", by_channels=False))


def test_equalize_grayscale_mask():
    img = np.random.randint(0, 255, [256, 256], dtype=np.uint8)

    mask = np.zeros([256, 256], dtype=np.bool)
    mask[:10, :10] = True

    assert np.all(cv2.equalizeHist(img[:10, :10]) == F.equalize(img, mask=mask, mode="cv")[:10, :10])


def test_equalize_rgb_mask():
    img = np.random.randint(0, 255, [256, 256, 3], dtype=np.uint8)

    mask = np.zeros([256, 256], dtype=np.bool)
    mask[:10, :10] = True

    _img = img.copy()[:10, :10]
    for i in range(3):
        _img[..., i] = cv2.equalizeHist(_img[..., i])
    assert np.all(_img == F.equalize(img, mask, mode="cv")[:10, :10])

    _img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img_cv = _img.copy()[:10, :10]
    img_cv[..., 0] = cv2.equalizeHist(img_cv[..., 0])
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_YCrCb2RGB)
    assert np.all(img_cv == F.equalize(img, mask=mask, mode="cv", by_channels=False)[:10, :10])

    mask = np.zeros([256, 256, 3], dtype=np.bool)
    mask[:10, :10, 0] = True
    mask[10:20, 10:20, 1] = True
    mask[20:30, 20:30, 2] = True
    img_r = img.copy()[:10, :10, 0]
    img_g = img.copy()[10:20, 10:20, 1]
    img_b = img.copy()[20:30, 20:30, 2]

    img_r = cv2.equalizeHist(img_r)
    img_g = cv2.equalizeHist(img_g)
    img_b = cv2.equalizeHist(img_b)

    result_img = F.equalize(img, mask=mask, mode="cv")
    assert np.all(img_r == result_img[:10, :10, 0])
    assert np.all(img_g == result_img[10:20, 10:20, 1])
    assert np.all(img_b == result_img[20:30, 20:30, 2])


@pytest.mark.parametrize("dtype", ["float32", "uint8"])
def test_downscale_ones(dtype):
    img = np.ones((100, 100, 3), dtype=dtype)
    downscaled = F.downscale(img, scale=0.5)
    assert np.all(downscaled == img)


def test_downscale_random():
    img = np.random.rand(100, 100, 3)
    downscaled = F.downscale(img, scale=0.5)
    assert downscaled.shape == img.shape
    downscaled = F.downscale(img, scale=1)
    assert np.all(img == downscaled)


def test_maybe_process_in_chunks():
    image = np.random.randint(0, 256, (100, 100, 6), np.uint8)

    for i in range(1, image.shape[-1] + 1):
        before = image[:, :, :i]
        after = F.rotate(before, angle=1)
        assert before.shape == after.shape


def test_multiply_uint8_optimized():
    image = np.random.randint(0, 256, [256, 320], np.uint8)
    m = 1.5

    result = F._multiply_uint8_optimized(image, [m])
    tmp = F.clip(image * m, image.dtype, F.MAX_VALUES_BY_DTYPE[image.dtype])
    assert np.all(tmp == result)

    image = np.random.randint(0, 256, [256, 320, 3], np.uint8)
    result = F._multiply_uint8_optimized(image, [m])
    tmp = F.clip(image * m, image.dtype, F.MAX_VALUES_BY_DTYPE[image.dtype])
    assert np.all(tmp == result)

    m = np.array([1.5, 0.75, 1.1])
    image = np.random.randint(0, 256, [256, 320, 3], np.uint8)
    result = F._multiply_uint8_optimized(image, m)
    tmp = F.clip(image * m, image.dtype, F.MAX_VALUES_BY_DTYPE[image.dtype])
    assert np.all(tmp == result)
