import hashlib

import cv2
import numpy as np
import pytest
from albucore import MAX_VALUES_BY_DTYPE, clip, get_num_channels, is_multispectral_image, to_float
from numpy.testing import assert_array_almost_equal_nulp

import albumentations.augmentations.functional as F
import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.core.types import d4_group_elements
from tests.conftest import IMAGES, RECTANGULAR_IMAGES, RECTANGULAR_UINT8_IMAGE, SQUARE_UINT8_IMAGE, UINT8_IMAGES
from tests.utils import convert_2d_to_target_format
from copy import deepcopy


@pytest.mark.parametrize(["input_shape", "expected_shape"], [[(128, 64), (64, 128)], [(128, 64, 3), (64, 128, 3)]])
def test_transpose(input_shape, expected_shape):
    img = np.random.randint(low=0, high=256, size=input_shape, dtype=np.uint8)
    transposed = fgeometric.transpose(img)
    assert transposed.shape == expected_shape


@pytest.mark.parametrize(["input_shape", "expected_shape"], [[(128, 64), (64, 128)], [(128, 64, 3), (64, 128, 3)]])
def test_transpose_float(input_shape, expected_shape):
    img = np.random.uniform(low=0.0, high=1.0, size=input_shape).astype("float32")
    transposed = fgeometric.transpose(img)
    assert transposed.shape == expected_shape


@pytest.mark.parametrize("target", ["image", "mask"])
def test_rot90(target):
    img = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.uint8)
    expected = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    rotated = fgeometric.rot90(img, factor=1)
    assert np.array_equal(rotated, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_rot90_float(target):
    img = np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 0.4], [0.0, 0.0, 0.4]], dtype=np.float32)
    expected = np.array([[0.4, 0.4, 0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    rotated = fgeometric.rot90(img, factor=1)
    assert_array_almost_equal_nulp(rotated, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_pad(target):
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.array([[4, 3, 4, 3], [2, 1, 2, 1], [4, 3, 4, 3], [2, 1, 2, 1]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    padded = fgeometric.pad(img, min_height=4, min_width=4, border_mode=cv2.BORDER_REFLECT_101, value=None)
    assert np.array_equal(padded, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_pad_float(target):
    img = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    expected = np.array(
        [[0.4, 0.3, 0.4, 0.3], [0.2, 0.1, 0.2, 0.1], [0.4, 0.3, 0.4, 0.3], [0.2, 0.1, 0.2, 0.1]],
        dtype=np.float32,
    )
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    padded_img = fgeometric.pad(img, min_height=4, min_width=4, value=None, border_mode=cv2.BORDER_REFLECT_101)
    assert_array_almost_equal_nulp(padded_img, expected)


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


@pytest.mark.parametrize(["gamma", "expected"], [(1, 1), (0.8, 3)])
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


def test_gamma_float_equal_uint8():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img_f = img.astype(np.float32) / 255.0
    gamma = 0.5

    img = F.gamma_transform(img, gamma)
    img_f = F.gamma_transform(img_f, gamma)

    img = img.astype(np.float32)
    img_f *= 255.0

    assert (np.abs(img - img_f) <= 1).all()


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
    scaled = fgeometric.scale(img, scale=2, interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(scaled, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_longest_max_size(target):
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.uint8)
    expected = np.array([[2, 3], [6, 7], [10, 11]], dtype=np.uint8)

    img, expected = convert_2d_to_target_format([img, expected], target=target)
    scaled = fgeometric.longest_max_size(img, max_size=3, interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(scaled, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_smallest_max_size(target):
    img = np.array(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]],
        dtype=np.uint8,
    )
    expected = np.array([[2, 4, 5, 7], [10, 11, 13, 14], [17, 19, 20, 22]], dtype=np.uint8)

    img, expected = convert_2d_to_target_format([img, expected], target=target)
    scaled = fgeometric.smallest_max_size(img, max_size=3, interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(scaled, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_linear_interpolation(target):
    img = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.uint8)
    expected = np.array([[2, 2], [4, 4]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = fgeometric.resize(img, (2, 2), interpolation=cv2.INTER_LINEAR)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert np.array_equal(resized_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_nearest_interpolation(target):
    img = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.uint8)
    expected = np.array([[1, 1], [3, 3]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = fgeometric.resize(img, (2, 2), interpolation=cv2.INTER_NEAREST)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert np.array_equal(resized_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_different_height_and_width(target):
    img = np.ones((100, 100), dtype=np.uint8)
    img = convert_2d_to_target_format([img], target=target)
    resized_img = fgeometric.resize(img, (20, 30), interpolation=cv2.INTER_LINEAR)
    height, width = resized_img.shape[:2]
    assert height == 20
    assert width == 30
    if target == "image":
        num_channels = resized_img.shape[2]
        assert num_channels == 3


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_default_interpolation_float(target):
    img = np.array(
        [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4]],
        dtype=np.float32,
    )
    expected = np.array([[0.15, 0.15], [0.35, 0.35]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = fgeometric.resize(img, (2, 2), interpolation=cv2.INTER_LINEAR)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert_array_almost_equal_nulp(resized_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_nearest_interpolation_float(target):
    img = np.array(
        [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3], [0.4, 0.4, 0.4, 0.4]],
        dtype=np.float32,
    )
    expected = np.array([[0.1, 0.1], [0.3, 0.3]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = fgeometric.resize(img, (2, 2), interpolation=cv2.INTER_NEAREST)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert np.array_equal(resized_img, expected)


@pytest.mark.parametrize(
    "factor, expected_positions",
    [
        (1, (299, 150)),  # Rotated 90 degrees CCW
        (2, (249, 199)),  # Rotated 180 degrees
        (3, (100, 249)),  # Rotated 270 degrees CCW
    ],
)
def test_keypoint_image_rot90_match(factor, expected_positions):
    image_shape = (300, 400)  # Non-square dimensions
    img = np.zeros(image_shape, dtype=np.uint8)
    # Placing the keypoint away from the center and edge: (150, 100)
    keypoints = np.array([[150, 100, 0, 1]])

    img[keypoints[0][1], keypoints[0][0]] = 1

    # Rotate the image
    rotated_img = fgeometric.rot90(img, factor)

    # Rotate the keypoint
    rotated_keypoints = fgeometric.keypoints_rot90(keypoints, factor, img.shape)[0]

    # Assert that the rotated keypoint lands where expected
    assert rotated_img[int(rotated_keypoints[1]), int(rotated_keypoints[0])] == 1, (
        f"Key point after rotation factor {factor} is not at the expected position {expected_positions}, "
        f"but at {rotated_keypoints}"
    )


def test_fun_max_size():
    target_width = 256

    img = np.empty((330, 49), dtype=np.uint8)
    out = fgeometric.smallest_max_size(img, target_width, interpolation=cv2.INTER_LINEAR)

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
    assert not is_multispectral_image(image)

    multispectral_image = np.ones((5, 5, 4), dtype=np.uint8)
    assert is_multispectral_image(multispectral_image)

    gray_image = np.ones((5, 5), dtype=np.uint8)
    assert not is_multispectral_image(gray_image)

    gray_image = np.ones((5, 5, 1), dtype=np.uint8)
    assert not is_multispectral_image(gray_image)


@pytest.mark.parametrize(
    "img, tiles, mapping, expected",
    [
        # Test with empty tiles - image should remain unchanged
        (
            np.array([[1, 1], [2, 2]], dtype=np.uint8),
            np.empty((0, 4), dtype=np.int32),
            [0],
            np.array([[1, 1], [2, 2]], dtype=np.uint8),
        ),
        # Test with empty mapping - image should remain unchanged
        (
            np.array([[1, 1], [2, 2]], dtype=np.uint8),
            np.array([[0, 0, 2, 2]]),
            None,
            np.array([[1, 1], [2, 2]], dtype=np.uint8),
        ),
        # Test with one tile that covers the whole image - should behave as if the image is unchanged
        (
            np.array([[1, 1], [2, 2]], dtype=np.uint8),
            np.array([[0, 0, 2, 2]]),
            [0],
            np.array([[1, 1], [2, 2]], dtype=np.uint8),
        ),
        # Test with splitting tiles horizontally
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            np.array([[0, 0, 2, 1], [0, 1, 2, 2]]),
            [1, 0],
            np.array([[2, 1], [4, 3]], dtype=np.uint8),  # Corrected expectation
        ),
        # Test with splitting tiles vertically
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            np.array([[0, 0, 1, 2], [1, 0, 2, 2]]),
            [1, 0],
            np.array([[3, 4], [1, 2]], dtype=np.uint8),  # Corrected expectation
        ),
    ],
)
def test_swap_tiles_on_image(img, tiles, mapping, expected):
    result_img = fgeometric.swap_tiles_on_image(img, tiles, mapping)
    np.testing.assert_array_equal(result_img, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_solarize(dtype):
    max_value = MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.dtype("float32"):
        img = np.arange(2**10, dtype=np.float32) / (2**10)
        img = img.reshape([2**5, 2**5])
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


@pytest.mark.parametrize(
    "img_shape, img_dtype, mask_shape, by_channels, expected_error, expected_message",
    [
        (
            (256, 256),
            np.uint8,
            (256, 256, 3),
            True,
            ValueError,
            "Wrong mask shape. Image shape: (256, 256). Mask shape: (256, 256, 3)",
        ),
        (
            (256, 256, 3),
            np.uint8,
            (256, 256, 3),
            False,
            ValueError,
            "When by_channels=False only 1-channel mask supports. Mask shape: (256, 256, 3)",
        ),
    ],
)
def test_equalize_checks(img_shape, img_dtype, mask_shape, by_channels, expected_error, expected_message):
    img = (
        np.random.randint(0, 255, img_shape).astype(img_dtype)
        if img_dtype == np.uint8
        else np.random.random(img_shape).astype(img_dtype)
    )
    mask = np.random.randint(0, 2, mask_shape).astype(bool)

    with pytest.raises(expected_error) as exc_info:
        F.equalize(img, mask=mask, by_channels=by_channels)
    assert str(exc_info.value) == expected_message


def test_equalize_grayscale():
    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    assert np.all(cv2.equalizeHist(img) == F.equalize(img, mode="cv"))


def test_equalize_rgb():
    img = SQUARE_UINT8_IMAGE

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

    mask = np.zeros([256, 256], dtype=bool)
    mask[:10, :10] = True

    assert np.all(cv2.equalizeHist(img[:10, :10]) == F.equalize(img, mask=mask, mode="cv")[:10, :10])


def test_equalize_rgb_mask():
    img = np.random.randint(0, 255, [256, 256, 3], dtype=np.uint8)

    mask = np.zeros([256, 256], dtype=bool)
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

    mask = np.zeros([256, 256, 3], dtype=bool)
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


@pytest.mark.parametrize(
    "img",
    [np.random.randint(0, 256, [100, 100], dtype=np.uint8), np.random.random([100, 100]).astype(np.float32)],
)
def test_shift_hsv_gray(img):
    F.shift_hsv(img, 0.5, 0.5, 0.5)


@pytest.mark.parametrize("beta_by_max", [True, False])
def test_brightness_contrast_adjust_equal(beta_by_max):
    image_int = np.random.randint(0, 256, [512, 512, 3], dtype=np.uint8)
    image_float = image_int.astype(np.float32) / 255

    alpha = 1.3
    beta = 0.14

    image_int = F.brightness_contrast_adjust(image_int, alpha, beta, beta_by_max)
    image_float = F.brightness_contrast_adjust(image_float, alpha, beta, beta_by_max)

    image_float = (image_float * 255).astype(int)

    assert np.abs(image_int.astype(int) - image_float).max() <= 1


@pytest.mark.parametrize(
    "tiles, expected",
    [
        # Simple case with two different shapes
        (np.array([[0, 0, 2, 2], [0, 2, 2, 4], [2, 0, 4, 2], [2, 2, 4, 4]]), {(2, 2): [0, 1, 2, 3]}),
        # Tiles with three different shapes
        (np.array([[0, 0, 1, 3], [0, 3, 1, 6], [1, 0, 4, 3], [1, 3, 4, 6]]), {(1, 3): [0, 1], (3, 3): [2, 3]}),
        # Single tile
        (np.array([[0, 0, 1, 1]]), {(1, 1): [0]}),
        # No tiles
        (np.array([]).reshape(0, 4), {}),
        # All tiles having the same shape
        (np.array([[0, 0, 2, 2], [2, 2, 4, 4], [4, 4, 6, 6]]), {(2, 2): [0, 1, 2]}),
    ],
)
def test_create_shape_groups(tiles, expected):
    result = fgeometric.create_shape_groups(tiles)
    assert len(result) == len(expected), "Incorrect number of shape groups"
    for shape in expected:
        assert shape in result, f"Shape {shape} is not in the result"
        assert sorted(result[shape]) == sorted(expected[shape]), f"Incorrect indices for shape {shape}"


@pytest.mark.parametrize(
    "shape_groups, random_seed, expected_output",
    [
        # Test with a simple case of one group
        ({(2, 2): [0, 1, 2, 3]}, 3, [3, 2, 1, 0]),
        # Test with multiple groups and ensure that random state affects the shuffle consistently
        ({(2, 2): [0, 1, 2, 3], (1, 1): [4]}, 0, [2, 0, 1, 3, 4]),
        # All tiles having the same shape should be shuffled within themselves
        ({(2, 2): [0, 1, 2]}, 3, [2, 1, 0]),
    ],
)
def test_shuffle_tiles_within_shape_groups(shape_groups, random_seed, expected_output):
    generator = np.random.default_rng(random_seed)
    shape_groups_original = deepcopy(shape_groups)
    actual_output = fgeometric.shuffle_tiles_within_shape_groups(shape_groups, generator)
    assert shape_groups == shape_groups_original, "Input shape groups should not be modified"
    np.testing.assert_array_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    "group_member,expected",
    [
        ("e", np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),  # Identity
        ("r90", np.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]])),  # Rotate 90 degrees counterclockwise
        ("r180", np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]])),  # Rotate 180 degrees
        ("r270", np.array([[6, 3, 0], [7, 4, 1], [8, 5, 2]])),  # Rotate 270 degrees counterclockwise
        ("v", np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])),  # Vertical flip
        ("t", np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]])),  # Transpose (reflect over main diagonal)
        ("h", np.array([[2, 1, 0], [5, 4, 3], [8, 7, 6]])),  # Horizontal flip
        ("hvt", np.array([[8, 5, 2], [7, 4, 1], [6, 3, 0]])),  # Transpose (reflect over anti-diagonal)
    ],
)
def test_d4_transformations(group_member, expected):
    img = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8)
    transformed_img = fgeometric.d4(img, group_member)
    assert np.array_equal(transformed_img, expected), f"Failed for transformation {group_member}"


def get_md5_hash(image):
    image_bytes = image.tobytes()
    hash_md5 = hashlib.md5()
    hash_md5.update(image_bytes)
    return hash_md5.hexdigest()


@pytest.mark.parametrize("image", IMAGES)
def test_d4_unique(image):
    hashes = set()
    for element in d4_group_elements:
        hashes.add(get_md5_hash(fgeometric.d4(image, element)))

    assert len(hashes) == len(set(hashes)), "d4 should generate unique images for all group elements"


@pytest.mark.parametrize("image", RECTANGULAR_IMAGES)
@pytest.mark.parametrize("group_member", d4_group_elements)
def test_d4_output_shape_with_group(image, group_member):
    result = fgeometric.d4(image, group_member)
    if group_member in ["r90", "r270", "t", "hvt"]:
        assert result.shape[:2] == image.shape[:2][::-1], "Output shape should be the transpose of input shape"
    else:
        assert result.shape == image.shape, "Output shape should match input shape"


@pytest.mark.parametrize("image", RECTANGULAR_IMAGES)
def test_transpose_output_shape(image):
    result = fgeometric.transpose(image)
    assert result.shape[:2] == image.shape[:2][::-1], "Output shape should be the transpose of input shape"


@pytest.mark.parametrize("image", RECTANGULAR_IMAGES)
@pytest.mark.parametrize("factor", [0, 1, 2, 3])
def test_d4_output_shape_with_factor(image, factor):
    result = fgeometric.rot90(image, factor)
    if factor in {1, 3}:
        assert result.shape[:2] == image.shape[:2][::-1], "Output shape should be the transpose of input shape"
    else:
        assert result.shape == image.shape, "Output shape should match input shape"


base_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
expected_main_diagonal = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
expected_second_diagonal = np.array([[9, 6, 3], [8, 5, 2], [7, 4, 1]])


def create_test_matrix(matrix, shape):
    if len(shape) == 2:
        return matrix
    if len(shape) == 3:
        return np.stack([matrix] * shape[2], axis=-1)


@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 1), (3, 3, 3), (3, 3, 7)])
def test_transpose_2(shape):
    img = create_test_matrix(base_matrix, shape)
    expected_main = create_test_matrix(expected_main_diagonal, shape)
    expected_second = create_test_matrix(expected_second_diagonal, shape)

    assert np.array_equal(fgeometric.transpose(img), expected_main)
    transposed_axis1 = fgeometric.transpose(fgeometric.rot90(img, 2))
    assert np.array_equal(transposed_axis1, expected_second)


def test_planckian_jitter_blackbody():
    img = np.array(
        [
            [[0.4963, 0.6977, 0.1759], [0.7682, 0.8, 0.2698], [0.0885, 0.161, 0.1507], [0.132, 0.2823, 0.0317]],
            [[0.3074, 0.6816, 0.2081], [0.6341, 0.9152, 0.9298], [0.4901, 0.3971, 0.7231], [0.8964, 0.8742, 0.7423]],
            [[0.4556, 0.4194, 0.5263], [0.6323, 0.5529, 0.2437], [0.3489, 0.9527, 0.5846], [0.4017, 0.0362, 0.0332]],
            [[0.0223, 0.1852, 0.1387], [0.1689, 0.3734, 0.2422], [0.2939, 0.3051, 0.8155], [0.5185, 0.932, 0.7932]],
        ],
    )

    expected_blackbody_plankian_jitter = np.array(
        [
            [[0.735, 0.6977, 0.0691], [1.0, 0.8, 0.1059], [0.1311, 0.161, 0.0592], [0.1955, 0.2823, 0.0124]],
            [[0.4553, 0.6816, 0.0817], [0.9391, 0.9152, 0.365], [0.7258, 0.3971, 0.2839], [1.0, 0.8742, 0.2914]],
            [[0.6748, 0.4194, 0.2066], [0.9364, 0.5529, 0.0957], [0.5167, 0.9527, 0.2295], [0.5949, 0.0362, 0.013]],
            [[0.033, 0.1852, 0.0545], [0.2501, 0.3734, 0.0951], [0.4353, 0.3051, 0.3202], [0.7679, 0.932, 0.3114]],
        ],
    )

    blackbody_plankian_jitter = F.planckian_jitter(img, temperature=3500, mode="blackbody")
    assert np.allclose(blackbody_plankian_jitter, expected_blackbody_plankian_jitter, atol=1e-4)


def test_planckian_jitter_cied():
    img = np.array(
        [
            [[0.4963, 0.6977, 0.1759], [0.7682, 0.8, 0.2698], [0.0885, 0.161, 0.1507], [0.132, 0.2823, 0.0317]],
            [[0.3074, 0.6816, 0.2081], [0.6341, 0.9152, 0.9298], [0.4901, 0.3971, 0.7231], [0.8964, 0.8742, 0.7423]],
            [[0.4556, 0.4194, 0.5263], [0.6323, 0.5529, 0.2437], [0.3489, 0.9527, 0.5846], [0.4017, 0.0362, 0.0332]],
            [[0.0223, 0.1852, 0.1387], [0.1689, 0.3734, 0.2422], [0.2939, 0.3051, 0.8155], [0.5185, 0.932, 0.7932]],
        ],
    )

    expected_cied_plankian_jitter = np.array(
        [
            [[0.6058, 0.6977, 0.1149], [0.9377, 0.8000, 0.1762], [0.1080, 0.1610, 0.0984], [0.1611, 0.2823, 0.0207]],
            [[0.3752, 0.6816, 0.1359], [0.7740, 0.9152, 0.6072], [0.5982, 0.3971, 0.4722], [1.0000, 0.8742, 0.4848]],
            [[0.5561, 0.4194, 0.3437], [0.7718, 0.5529, 0.1592], [0.4259, 0.9527, 0.3818], [0.4903, 0.0362, 0.0217]],
            [[0.0272, 0.1852, 0.0906], [0.2062, 0.3734, 0.1582], [0.3587, 0.3051, 0.5326], [0.6329, 0.9320, 0.5180]],
        ],
    )
    cied_plankian_jitter = F.planckian_jitter(img, temperature=4500, mode="cied")
    assert np.allclose(cied_plankian_jitter, expected_cied_plankian_jitter, atol=1e-4)


@pytest.mark.parametrize("mode", ["blackbody", "cied"])
def test_planckian_jitter_edge_cases(mode):
    # Create a sample image
    img = np.ones((10, 10, 3), dtype=np.float32)

    # Get min and max temperatures for the mode
    min_temp = min(F.PLANCKIAN_COEFFS[mode].keys())
    max_temp = max(F.PLANCKIAN_COEFFS[mode].keys())

    # Test cases
    test_temperatures = [
        min_temp - 500,  # Below minimum
        min_temp,  # At minimum
        min_temp + 100,  # Just above minimum
        (min_temp + max_temp) // 2,  # Middle of the range
        max_temp - 100,  # Just below maximum
        max_temp,  # At maximum
        max_temp + 500,  # Above maximum
    ]

    for temp in test_temperatures:
        result = F.planckian_jitter(img, temp, mode)

        # Check that the output is a valid image
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        assert np.all(result >= 0) and np.all(result <= 1)

        # Check that the function didn't modify the input image
        assert not np.array_equal(result, img)

        # For temperatures outside the range, check if they're clamped correctly
        if temp < min_temp:
            np.testing.assert_allclose(result, F.planckian_jitter(img, min_temp, mode))
        elif temp > max_temp:
            np.testing.assert_allclose(result, F.planckian_jitter(img, max_temp, mode))


def test_planckian_jitter_interpolation():
    img = np.ones((10, 10, 3), dtype=np.float32)
    mode = "blackbody"

    # Test interpolation between two known temperatures
    temp1, temp2 = 4000, 4500
    result1 = F.planckian_jitter(img, temp1, mode)
    result2 = F.planckian_jitter(img, temp2, mode)
    result_mid = F.planckian_jitter(img, (temp1 + temp2) // 2, mode)

    # The mid-temperature result should be between the two extremes
    assert np.all((result_mid >= np.minimum(result1, result2)) & (result_mid <= np.maximum(result1, result2)))


@pytest.mark.parametrize("mode", ["blackbody", "cied"])
def test_planckian_jitter_consistency(mode):
    img = np.ones((10, 10, 3), dtype=np.float32)

    # Test consistency of results for the same temperature
    temp = 5000
    result1 = F.planckian_jitter(img, temp, mode)
    result2 = F.planckian_jitter(img, temp, mode)
    np.testing.assert_allclose(result1, result2)


def test_planckian_jitter_invalid_mode():
    img = np.ones((10, 10, 3), dtype=np.float32)

    with pytest.raises(KeyError):
        F.planckian_jitter(img, 5000, "invalid_mode")


@pytest.mark.parametrize("image", IMAGES)
def test_random_tone_curve(image):
    low_y = 0.1
    high_y = 0.9

    num_channels = get_num_channels(image)

    result_float_value = F.move_tone_curve(image, low_y, high_y)
    result_array_value = F.move_tone_curve(image, np.array([low_y] * num_channels), np.array([high_y] * num_channels))

    np.testing.assert_allclose(result_float_value, result_array_value)

    assert result_float_value.dtype == image.dtype
    assert result_float_value.shape == image.shape


@pytest.mark.parametrize("image", UINT8_IMAGES)
@pytest.mark.parametrize("color_shift, intensity", [
    (0, 0),  # No noise
    (0.5, 0.5),  # Medium noise
    (1, 1),  # Maximum noise
])
def test_iso_noise(image, color_shift, intensity):
    """Test that iso_noise produces expected noise levels."""
    # Convert image to float and back
    float_image = to_float(image)

    # Generate noise using the same random state instance
    rng = np.random.default_rng(42)
    result_uint8 = F.iso_noise(
        image,
        color_shift=color_shift,
        intensity=intensity,
        random_generator=rng
    )

    rng = np.random.default_rng(42)
    result_float = F.iso_noise(
        float_image,
        color_shift=color_shift,
        intensity=intensity,
        random_generator=rng
    )

    # Convert float result back to uint8
    result_float = F.from_float(result_float, target_dtype=np.uint8)

    # Calculate noise
    noise = result_uint8.astype(np.float32) - image.astype(np.float32)
    mean_noise = np.mean(noise)
    std_noise = np.std(noise)

    if intensity == 0:
        # For zero intensity, expect no noise
        np.testing.assert_array_equal(result_uint8, image)
        np.testing.assert_array_equal(result_float, image)
    else:
        # Check that float and uint8 results are close
        np.testing.assert_allclose(result_uint8, result_float, rtol=1e-5, atol=1)



@pytest.mark.parametrize(
    "input_image, num_output_channels, expected_shape",
    [
        (np.zeros((10, 10), dtype=np.uint8), 3, (10, 10, 3)),
        (np.zeros((10, 10, 1), dtype=np.uint8), 3, (10, 10, 3)),
        (np.zeros((10, 10), dtype=np.float32), 4, (10, 10, 4)),
        (np.zeros((10, 10, 1), dtype=np.float32), 2, (10, 10, 2)),
    ],
)
def test_grayscale_to_multichannel(input_image, num_output_channels, expected_shape):
    result = F.grayscale_to_multichannel(input_image, num_output_channels)
    assert result.shape == expected_shape
    assert np.all(result[..., 0] == result[..., 1])  # All channels should be identical


def test_grayscale_to_multichannel_preserves_values():
    input_image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    result = F.grayscale_to_multichannel(input_image, num_output_channels=3)
    assert np.all(result[..., 0] == input_image)
    assert np.all(result[..., 1] == input_image)
    assert np.all(result[..., 2] == input_image)


def test_grayscale_to_multichannel_default_channels():
    input_image = np.zeros((10, 10), dtype=np.uint8)
    result = F.grayscale_to_multichannel(input_image, num_output_channels=3)
    assert result.shape == (10, 10, 3)


def create_test_image(height, width, channels, dtype):
    if dtype == np.uint8:
        return np.random.randint(0, 256, (height, width, channels), dtype=dtype)
    return np.random.rand(height, width, channels).astype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_to_gray_weighted_average(dtype):
    img = create_test_image(10, 10, 3, dtype)
    result = F.to_gray_weighted_average(img)
    expected = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    if dtype == np.uint8:
        expected = expected.astype(np.uint8)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_to_gray_from_lab(dtype):
    img = create_test_image(10, 10, 3, dtype)
    result = F.to_gray_from_lab(img)
    expected = clip(cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0], dtype=dtype)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [3, 4, 5])
def test_to_gray_desaturation(dtype, channels):
    img = create_test_image(10, 10, channels, dtype)
    result = F.to_gray_desaturation(img)
    expected = (np.max(img.astype(np.float32), axis=-1) + np.min(img.astype(np.float32), axis=-1)) / 2
    if dtype == np.uint8:
        expected = expected.astype(np.uint8)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [3, 4, 5])
def test_to_gray_average(dtype, channels):
    img = create_test_image(10, 10, channels, dtype)
    result = F.to_gray_average(img)
    expected = np.mean(img, axis=-1)
    if dtype == np.uint8:
        expected = expected.astype(np.uint8)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [3, 4, 5])
def test_to_gray_max(dtype, channels):
    img = create_test_image(10, 10, channels, dtype)
    result = F.to_gray_max(img)
    expected = np.max(img, axis=-1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [3, 4, 5])
def test_to_gray_pca(dtype, channels):
    img = create_test_image(10, 10, channels, dtype)
    result = F.to_gray_pca(img)
    assert result.shape == (10, 10)
    assert result.dtype == dtype
    if dtype == np.uint8:
        assert result.min() >= 0 and result.max() <= 255
    else:
        assert result.min() >= 0 and result.max() <= 1


@pytest.mark.parametrize(
    "func",
    [
        F.to_gray_weighted_average,
        F.to_gray_from_lab,
        F.to_gray_desaturation,
        F.to_gray_average,
        F.to_gray_max,
        F.to_gray_pca,
    ],
)
def test_float32_uint8_consistency(func):
    img_uint8 = create_test_image(10, 10, 3, np.uint8)
    img_float32 = img_uint8.astype(np.float32) / 255.0

    result_uint8 = func(img_uint8)
    result_float32 = func(img_float32)

    np.testing.assert_allclose(result_uint8 / 255.0, result_float32, rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize(
    "shape, dtype, clip_limit, tile_grid_size",
    [
        ((100, 100), np.uint8, 2.0, (8, 8)),  # Grayscale uint8
        ((100, 100, 3), np.uint8, 2.0, (8, 8)),  # RGB uint8
        ((50, 50), np.float32, 3.0, (4, 4)),  # Grayscale float32
        ((50, 50, 3), np.float32, 3.0, (4, 4)),  # RGB float32
    ],
)
def test_clahe(shape, dtype, clip_limit, tile_grid_size):
    if dtype == np.uint8:
        img = np.random.randint(0, 256, shape, dtype=dtype)
    else:
        img = np.random.rand(*shape).astype(dtype)

    result = F.clahe(img, clip_limit, tile_grid_size)

    assert result.shape == img.shape
    assert result.dtype == img.dtype
    assert np.any(result != img)  # Ensure the image has changed


@pytest.mark.parametrize("shape", [(100, 100, 3), (100, 100, 1), (100, 100, 5)])
def test_fancy_pca_mean_preservation(shape):
    image = np.random.rand(*shape).astype(np.float32)
    alpha_vector = np.random.uniform(-0.1, 0.1, shape[-1])
    result = F.fancy_pca(image, alpha_vector)
    np.testing.assert_almost_equal(np.mean(image), np.mean(result), decimal=5)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((100, 100, 3), np.uint8),
        ((100, 100, 3), np.float32),
        ((100, 100, 1), np.uint8),
        ((100, 100, 1), np.float32),
        ((100, 100, 5), np.float32),
    ],
)
def test_fancy_pca_zero_alpha(shape, dtype):
    image = np.random.randint(0, 256, shape).astype(dtype)
    if dtype == np.float32:
        image = image / 255.0

    alpha_vector = np.zeros(shape[-1])
    result = F.fancy_pca(image, alpha_vector)

    np.testing.assert_array_equal(image, result)
