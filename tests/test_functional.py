import hashlib
import cv2
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal_nulp, assert_almost_equal
import skimage

import albumentations.augmentations.functional as F
import albumentations.augmentations.geometric.functional as FGeometric
from albucore.utils import is_multispectral_image, MAX_VALUES_BY_DTYPE, get_num_channels

from albumentations.core.types import d4_group_elements
from tests.conftest import IMAGES, RECTANGULAR_IMAGES, RECTANGULAR_UINT8_IMAGE, UINT8_IMAGES
from tests.utils import convert_2d_to_target_format, set_seed


@pytest.mark.parametrize("target", ["image", "mask"])
def test_vflip(target):
    img = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    expected = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    flipped_img = FGeometric.vflip(img)
    assert np.array_equal(flipped_img, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_vflip_float(target):
    img = np.array([[0.4, 0.4, 0.4], [0.0, 0.4, 0.4], [0.0, 0.0, 0.4]], dtype=np.float32)
    expected = np.array([[0.0, 0.0, 0.4], [0.0, 0.4, 0.4], [0.4, 0.4, 0.4]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    flipped_img = FGeometric.vflip(img)
    assert_array_almost_equal_nulp(flipped_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_hflip(target):
    img = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    expected = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    flipped_img = FGeometric.hflip(img)
    assert np.array_equal(flipped_img, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_hflip_float(target):
    img = np.array([[0.4, 0.4, 0.4], [0.0, 0.4, 0.4], [0.0, 0.0, 0.4]], dtype=np.float32)
    expected = np.array([[0.4, 0.4, 0.4], [0.4, 0.4, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    flipped_img = FGeometric.hflip(img)
    assert_array_almost_equal_nulp(flipped_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
@pytest.mark.parametrize(
    ["code", "func"],
    [[0, FGeometric.vflip], [1, FGeometric.hflip], [-1, lambda img: FGeometric.vflip(FGeometric.hflip(img))]],
)
def test_random_flip(code, func, target):
    img = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    img = convert_2d_to_target_format([img], target=target)
    assert np.array_equal(FGeometric.random_flip(img, code), func(img))


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
@pytest.mark.parametrize(
    ["code", "func"],
    [[0, FGeometric.vflip], [1, FGeometric.hflip], [-1, lambda img: FGeometric.vflip(FGeometric.hflip(img))]],
)
def test_random_flip_float(code, func, target):
    img = np.array([[0.4, 0.4, 0.4], [0.0, 0.4, 0.4], [0.0, 0.0, 0.4]], dtype=np.float32)
    img = convert_2d_to_target_format([img], target=target)
    assert_array_almost_equal_nulp(FGeometric.random_flip(img, code), func(img))


@pytest.mark.parametrize(["input_shape", "expected_shape"], [[(128, 64), (64, 128)], [(128, 64, 3), (64, 128, 3)]])
def test_transpose(input_shape, expected_shape):
    img = np.random.randint(low=0, high=256, size=input_shape, dtype=np.uint8)
    transposed = FGeometric.transpose(img)
    assert transposed.shape == expected_shape


@pytest.mark.parametrize(["input_shape", "expected_shape"], [[(128, 64), (64, 128)], [(128, 64, 3), (64, 128, 3)]])
def test_transpose_float(input_shape, expected_shape):
    img = np.random.uniform(low=0.0, high=1.0, size=input_shape).astype("float32")
    transposed = FGeometric.transpose(img)
    assert transposed.shape == expected_shape


@pytest.mark.parametrize("target", ["image", "mask"])
def test_rot90(target):
    img = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.uint8)
    expected = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    rotated = FGeometric.rot90(img, factor=1)
    assert np.array_equal(rotated, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_rot90_float(target):
    img = np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 0.4], [0.0, 0.0, 0.4]], dtype=np.float32)
    expected = np.array([[0.4, 0.4, 0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    rotated = FGeometric.rot90(img, factor=1)
    assert_array_almost_equal_nulp(rotated, expected)


def generate_rotation_matrix(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Generates a rotation matrix for the given angle with rotation around the center of the image.
    """
    height, width = image.shape[:2]
    center = (width / 2 - 0.5, height / 2 - 0.5)
    return cv2.getRotationMatrix2D(center, angle, 1.0)


@pytest.mark.parametrize("image", IMAGES)
def test_compare_rotate_and_affine(image):
    # Generate the rotation matrix for a 60-degree rotation around the image center
    rotation_matrix = generate_rotation_matrix(image, 60)

    # Apply rotation using FGeometric.rotate
    rotated_img_1 = FGeometric.rotate(
        image, angle=60, border_mode=cv2.BORDER_CONSTANT, value=0, interpolation=cv2.INTER_LINEAR
    )

    # Convert 2x3 cv2 matrix to 3x3 for skimage's ProjectiveTransform
    full_matrix = np.vstack([rotation_matrix, [0, 0, 1]])
    projective_transform = skimage.transform.ProjectiveTransform(matrix=full_matrix)

    # Apply rotation using warp_affine
    rotated_img_2 = FGeometric.warp_affine(
        img=image,
        matrix=projective_transform,
        interpolation=cv2.INTER_LINEAR,
        cval=0,
        mode=cv2.BORDER_CONSTANT,
        output_shape=image.shape[:2]
    )

    # Assert that the two rotated images are equal
    assert np.array_equal(rotated_img_1, rotated_img_2), "Rotated images should be identical."


@pytest.mark.parametrize("target", ["image", "mask"])
def test_pad(target):
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.array([[4, 3, 4, 3], [2, 1, 2, 1], [4, 3, 4, 3], [2, 1, 2, 1]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    padded = FGeometric.pad(img, min_height=4, min_width=4, border_mode=cv2.BORDER_REFLECT_101, value=None)
    assert np.array_equal(padded, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_pad_float(target):
    img = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    expected = np.array(
        [[0.4, 0.3, 0.4, 0.3], [0.2, 0.1, 0.2, 0.1], [0.4, 0.3, 0.4, 0.3], [0.2, 0.1, 0.2, 0.1]], dtype=np.float32
    )
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    padded_img = FGeometric.pad(img, min_height=4, min_width=4, value=None, border_mode=cv2.BORDER_REFLECT_101)
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


@pytest.mark.parametrize(
    ["dtype", "expected_divider", "max_value"],
    [
        (np.uint8, 255, None),
        (np.uint16, 65535, None),
        (np.uint32, 4294967295, None),
        (np.float32, 1.0, None),
        (np.int16, None, 32767),  # Unsupported dtype with max_value provided
    ],
)
def test_to_float(dtype, expected_divider, max_value):
    img = np.ones((100, 100, 3), dtype=dtype)
    if expected_divider is not None:
        expected = (img.astype(np.float32) / expected_divider).astype(np.float32)
    else:
        # For unsupported dtype with max_value, use max_value for conversion
        expected = (img.astype(np.float32) / max_value).astype(np.float32)

    actual = F.to_float(img, max_value=max_value)
    assert_almost_equal(actual, expected, decimal=6)
    assert actual.dtype == np.float32, "Resulting dtype is not float32."


@pytest.mark.parametrize("dtype", [np.int64])
def test_to_float_raises_for_unsupported_dtype_without_max_value(dtype):
    img = np.ones((100, 100, 3), dtype=dtype)
    with pytest.raises(RuntimeError) as exc_info:
        F.to_float(img)
    assert "Unsupported dtype" in str(exc_info.value)


@pytest.mark.parametrize("dtype", [np.int64])
def test_to_float_with_max_value_for_unsupported_dtypes(dtype):
    img = np.ones((100, 100, 3), dtype=dtype)
    max_value = np.iinfo(dtype).max
    expected = (img.astype(np.float32) / max_value).astype(np.float32)
    actual = F.to_float(img, max_value=max_value)
    assert_almost_equal(actual, expected, decimal=6)
    assert actual.dtype == np.float32, "Resulting dtype is not float32."


@pytest.mark.parametrize(
    "dtype, multiplier, max_value",
    [
        (np.uint8, 255, None),
        (np.uint16, 65535, None),
        (np.uint32, 4294967295, None),
        (np.uint32, 4294967295, 4294967295.0),  # Custom max_value equal to the default to test the parameter is used
    ],
)
def test_from_float(dtype, multiplier, max_value):
    img = RECTANGULAR_UINT8_IMAGE.astype(np.float32)  # Use random data for more robust testing
    expected_multiplier = multiplier if max_value is None else max_value
    expected = (img * expected_multiplier).astype(dtype)
    actual = F.from_float(img, dtype=np.dtype(dtype), max_value=max_value)
    assert_array_almost_equal_nulp(actual, expected)


@pytest.mark.parametrize("dtype", [np.int64])
def test_from_float_unsupported_dtype_without_max_value(dtype):
    img = RECTANGULAR_UINT8_IMAGE.astype(np.float32)
    with pytest.raises(RuntimeError) as exc_info:
        F.from_float(img, dtype=dtype)
    expected_part_of_message = "Can't infer the maximum value for dtype"
    assert expected_part_of_message in str(exc_info.value), "Expected error message not found."


@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [
        (np.uint8, np.uint8),
        (np.uint16, np.uint16),
        (np.uint32, np.uint32),
    ],
)
def test_from_float_dtype_consistency(dtype, expected_dtype):
    # The code snippet is generating a random 100x100x3 array of values between 0 and the maximum
    # value allowed for the specified data type `dtype`. The `MAX_VALUES_BY_DTYPE` dictionary is used
    # to determine the maximum value for the given data type.
    img = np.random.rand(100, 100, 3) * MAX_VALUES_BY_DTYPE[dtype]
    actual = F.from_float(img.astype(np.float32), dtype=dtype)
    assert actual.dtype == expected_dtype, f"Expected dtype {expected_dtype} but got {actual.dtype}"


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
    scaled = FGeometric.scale(img, scale=2, interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(scaled, expected)


def test_to_from_float():
    image = RECTANGULAR_UINT8_IMAGE
    float_image = F.to_float(image)
    uint8_image = F.from_float(float_image, dtype=np.uint8)
    assert np.array_equal(image, uint8_image)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_longest_max_size(target):
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.uint8)
    expected = np.array([[2, 3], [6, 7], [10, 11]], dtype=np.uint8)

    img, expected = convert_2d_to_target_format([img, expected], target=target)
    scaled = FGeometric.longest_max_size(img, max_size=3, interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(scaled, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_smallest_max_size(target):
    img = np.array(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]], dtype=np.uint8
    )
    expected = np.array([[2, 4, 5, 7], [10, 11, 13, 14], [17, 19, 20, 22]], dtype=np.uint8)

    img, expected = convert_2d_to_target_format([img, expected], target=target)
    scaled = FGeometric.smallest_max_size(img, max_size=3, interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(scaled, expected)


def test_from_float_unknown_dtype():
    img = np.ones((100, 100, 3), dtype=np.float32)
    with pytest.raises(RuntimeError) as exc_info:
        F.from_float(img, np.dtype(np.int16))
    expected_message = (
        "Can't infer the maximum value for dtype int16. You need to specify the maximum value manually by passing "
        "the max_value argument"
    )
    actual_message = str(exc_info.value)
    assert (
        expected_message in actual_message or actual_message in expected_message
    ), f"Expected part of the error message to be: '{expected_message}', got: '{actual_message}'"


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_linear_interpolation(target):
    img = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.uint8)
    expected = np.array([[2, 2], [4, 4]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = FGeometric.resize(img, 2, 2, interpolation=cv2.INTER_LINEAR)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert np.array_equal(resized_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_nearest_interpolation(target):
    img = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.uint8)
    expected = np.array([[1, 1], [3, 3]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = FGeometric.resize(img, 2, 2, interpolation=cv2.INTER_NEAREST)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert np.array_equal(resized_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_different_height_and_width(target):
    img = np.ones((100, 100), dtype=np.uint8)
    img = convert_2d_to_target_format([img], target=target)
    resized_img = FGeometric.resize(img, height=20, width=30, interpolation=cv2.INTER_LINEAR)
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
    resized_img = FGeometric.resize(img, 2, 2, interpolation=cv2.INTER_LINEAR)
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
    resized_img = FGeometric.resize(img, 2, 2, interpolation=cv2.INTER_NEAREST)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    assert np.array_equal(resized_img, expected)


def test_bbox_vflip():
    assert FGeometric.bbox_vflip((0.1, 0.2, 0.6, 0.5), 100, 200) == (0.1, 0.5, 0.6, 0.8)


def test_bbox_hflip():
    assert FGeometric.bbox_hflip((0.1, 0.2, 0.6, 0.5), 100, 200) == (0.4, 0.2, 0.9, 0.5)


@pytest.mark.parametrize(
    ["code", "func"],
    [
        [0, FGeometric.bbox_vflip],
        [1, FGeometric.bbox_hflip],
        [-1, lambda bbox, rows, cols: FGeometric.bbox_vflip(FGeometric.bbox_hflip(bbox, rows, cols), rows, cols)],
    ],
)
def test_bbox_flip(code, func):
    rows, cols = 100, 200
    bbox = [0.1, 0.2, 0.6, 0.5]
    assert FGeometric.bbox_flip(bbox, code, rows, cols) == func(bbox, rows, cols)


@pytest.mark.parametrize("factor, expected_positions", [
    (1, (199, 150)),  # Rotated 90 degrees CCW
    (2, (249, 199)),  # Rotated 180 degrees
    (3, (100, 249)),  # Rotated 270 degrees CCW
])
def test_keypoint_image_rot90_match(factor, expected_positions):
    rows, cols = 300, 400  # Non-square dimensions
    img = np.zeros((rows, cols), dtype=int)
    # Placing the keypoint away from the center and edge: (150, 100)
    keypoint = (150, 100, 0, 1)
    img[keypoint[1], keypoint[0]] = 1

    # Rotate the image
    rotated_img = FGeometric.rot90(img, factor)

    # Rotate the keypoint
    rotated_keypoint = FGeometric.keypoint_rot90(keypoint, factor, img.shape[0], img.shape[1])

    # Assert that the rotated keypoint lands where expected
    assert rotated_img[rotated_keypoint[1], rotated_keypoint[0]] == 1, \
        f"Key point after rotation factor {factor} is not at the expected position {expected_positions}, "\
        f"but at {rotated_keypoint}"


def test_bbox_rot90():
    assert FGeometric.bbox_rot90((0.1, 0.2, 0.3, 0.4), 0, 100, 200) == (0.1, 0.2, 0.3, 0.4)
    assert FGeometric.bbox_rot90((0.1, 0.2, 0.3, 0.4), 1, 100, 200) == (0.2, 0.7, 0.4, 0.9)
    assert FGeometric.bbox_rot90((0.1, 0.2, 0.3, 0.4), 2, 100, 200) == (0.7, 0.6, 0.9, 0.8)
    assert FGeometric.bbox_rot90((0.1, 0.2, 0.3, 0.4), 3, 100, 200) == (0.6, 0.1, 0.8, 0.3)


def test_bbox_transpose():
    assert np.allclose(FGeometric.bbox_transpose((0.7, 0.1, 0.8, 0.4), 100, 200), (0.1, 0.7, 0.4, 0.8))
    rot90 = FGeometric.bbox_rot90((0.7, 0.1, 0.8, 0.4), 2, 100, 200)
    reflected_anti_diagonal = FGeometric.bbox_transpose(rot90, 100, 200)
    assert np.allclose(reflected_anti_diagonal, (0.6, 0.2, 0.9, 0.3))


def test_fun_max_size():
    target_width = 256

    img = np.empty((330, 49), dtype=np.uint8)
    out = FGeometric.smallest_max_size(img, target_width, interpolation=cv2.INTER_LINEAR)

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
            np.array([[1, 1], [2, 2]], dtype=np.uint8)
        ),

        # Test with empty mapping - image should remain unchanged
        (
            np.array([[1, 1], [2, 2]], dtype=np.uint8),
            np.array([[0, 0, 2, 2]]),
            None,
            np.array([[1, 1], [2, 2]], dtype=np.uint8)
        ),

        # Test with one tile that covers the whole image - should behave as if the image is unchanged
        (
            np.array([[1, 1], [2, 2]], dtype=np.uint8),
            np.array([[0, 0, 2, 2]]),
            [0],
            np.array([[1, 1], [2, 2]], dtype=np.uint8)
        ),

        # Test with splitting tiles horizontally
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            np.array([[0, 0, 2, 1], [0, 1, 2, 2]]),
            [1, 0],
            np.array([[2, 1], [4, 3]], dtype=np.uint8)  # Corrected expectation
        ),

        # Test with splitting tiles vertically
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            np.array([[0, 0, 1, 2], [1, 0, 2, 2]]),
            [1, 0],
            np.array([[3, 4], [1, 2]], dtype=np.uint8)  # Corrected expectation
        ),
    ]
)
def test_swap_tiles_on_image(img, tiles, mapping, expected):
    result_img = F.swap_tiles_on_image(img, tiles, mapping)
    assert np.array_equal(result_img, expected)


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


def test_posterize_checks():
    img = np.random.random([256, 256, 3])
    with pytest.raises(TypeError) as exc_info:
        F.posterize(img, 4)
    assert str(exc_info.value) == "Image must have uint8 channel type"

    img = np.random.randint(0, 256, [256, 256], dtype=np.uint8)
    with pytest.raises(TypeError) as exc_info:
        F.posterize(img, [1, 2, 3])
    assert str(exc_info.value) == "If bits is iterable image must be RGB"


def test_equalize_checks():
    img = np.random.randint(0, 255, [256, 256], dtype=np.uint8)

    mask = np.random.randint(0, 1, [256, 256, 3], dtype=bool)
    with pytest.raises(ValueError) as exc_info:
        F.equalize(img, mask=mask)
    assert str(exc_info.value) == f"Wrong mask shape. Image shape: {img.shape}. Mask shape: {mask.shape}"

    img = np.random.randint(0, 255, [256, 256, 3], dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        F.equalize(img, mask=mask, by_channels=False)
    assert str(exc_info.value) == f"When by_channels=False only 1-channel mask supports. Mask shape: {mask.shape}"

    img = np.random.random([256, 256, 3])
    with pytest.raises(TypeError) as exc_info:
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


def test_maybe_process_in_chunks():
    image = np.random.randint(0, 256, (100, 100, 6), np.uint8)

    for i in range(1, image.shape[-1] + 1):
        before = image[:, :, :i]
        after = FGeometric.rotate(before, angle=1, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101)
        assert before.shape == after.shape


@pytest.mark.parametrize(
    "img", [np.random.randint(0, 256, [100, 100], dtype=np.uint8), np.random.random([100, 100]).astype(np.float32)]
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
    "image_shape, grid, expected",
    [
        # Normal case: standard grids
        ((100, 200), (2, 2), np.array([[0, 0, 50, 100], [0, 100, 50, 200], [50, 0, 100, 100], [50, 100, 100, 200]])),

        # Single row grid
        ((100, 200), (1, 4), np.array([[0, 0, 100, 50], [0, 50, 100, 100], [0, 100, 100, 150], [0, 150, 100, 200]])),

        # Single column grid
        ((100, 200), (4, 1), np.array([[0, 0, 25, 200], [25, 0, 50, 200], [50, 0, 75, 200], [75, 0, 100, 200]])),

        # Edge case: Grid size equals image size
        ((100, 200), (100, 200), np.array([[i, j, i+1, j+1] for i in range(100) for j in range(200)])),

        # Edge case: Image where width is much larger than height
        ((10, 1000), (1, 10), np.array([[0, i * 100, 10, (i + 1) * 100] for i in range(10)])),

        # Edge case: Image where height is much larger than width
        ((1000, 10), (10, 1), np.array([[i * 100, 0, (i + 1) * 100, 10] for i in range(10)])),

        # Corner case: height and width are not divisible by the number of splits
        ((105, 205), (3, 4), np.array([
            [0, 0, 35, 51], [0, 51, 35, 102], [0, 102, 35, 153], [0, 153, 35, 205],  # First row splits
            [35, 0, 70, 51], [35, 51, 70, 102], [35, 102, 70, 153], [35, 153, 70, 205],  # Second row splits
            [70, 0, 105, 51], [70, 51, 105, 102], [70, 102, 105, 153], [70, 153, 105, 205]  # Third row splits
        ])),
    ]
)
def test_split_uniform_grid(image_shape, grid, expected):
    random_seed = 42
    result = F.split_uniform_grid(image_shape, grid, random_state=np.random.RandomState(random_seed))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("size, divisions, random_state, expected", [
    (10, 2, None, [0, 5, 10]),
    (10, 2, 42, [0, 5, 10]),  # Consistent shuffling with seed
    (9, 3, None, [0, 3, 6, 9]),
    (9, 3, 42, [0, 3, 6, 9]),  # Expected shuffle result with a specific seed
    (20, 5, 42, [0, 4, 8, 12, 16, 20]),  # Regular intervals
    (7, 3, 42, [0, 3, 5, 7]),  # Irregular intervals, specific seed
    (7, 3, 41, [0, 2, 4, 7]),  # Irregular intervals, specific seed
])
def test_generate_shuffled_splits(size, divisions, random_state, expected):
    result = F.generate_shuffled_splits(
        size, divisions, random_state=np.random.RandomState(random_state) if random_state else None
    )
    assert len(result) == divisions + 1
    assert np.array_equal(result, expected), \
        f"Failed for size={size}, divisions={divisions}, random_state={random_state}"


@pytest.mark.parametrize("size, divisions, random_state", [
    (10, 2, 42),
    (9, 3, 99),
    (20, 5, 101),
    (7, 3, 42),
])
def test_consistent_shuffling(size, divisions, random_state):
    set_seed(random_state)
    result1 = F.generate_shuffled_splits(size, divisions, random_state=np.random.RandomState(random_state))
    assert len(result1) == divisions + 1
    set_seed(random_state)
    result2 = F.generate_shuffled_splits(size, divisions, random_state=np.random.RandomState(random_state))
    assert len(result2) == divisions + 1
    assert np.array_equal(result1, result2), "Shuffling is not consistent with the given random state"


@pytest.mark.parametrize("tiles, expected", [
    # Simple case with two different shapes
    (np.array([[0, 0, 2, 2], [0, 2, 2, 4], [2, 0, 4, 2], [2, 2, 4, 4]]),
     {(2, 2): [0, 1, 2, 3]}),
    # Tiles with three different shapes
    (np.array([[0, 0, 1, 3], [0, 3, 1, 6], [1, 0, 4, 3], [1, 3, 4, 6]]),
     {(1, 3): [0, 1], (3, 3): [2, 3]}),
    # Single tile
    (np.array([[0, 0, 1, 1]]),
     {(1, 1): [0]}),
    # No tiles
    (np.array([]).reshape(0, 4),
     {}),
    # All tiles having the same shape
    (np.array([[0, 0, 2, 2], [2, 2, 4, 4], [4, 4, 6, 6]]),
     {(2, 2): [0, 1, 2]}),
])
def test_create_shape_groups(tiles, expected):
    result = F.create_shape_groups(tiles)
    assert len(result) == len(expected), "Incorrect number of shape groups"
    for shape in expected:
        assert shape in result, f"Shape {shape} is not in the result"
        assert sorted(result[shape]) == sorted(expected[shape]), f"Incorrect indices for shape {shape}"


@pytest.mark.parametrize("shape_groups, random_state, expected_output", [
    # Test with a simple case of one group
    ({(2, 2): [0, 1, 2, 3]}, 42, [1, 3, 0, 2]),
    # Test with multiple groups and ensure that random state affects the shuffle consistently
    ({(2, 2): [0, 1, 2, 3], (1, 1): [4]}, 42, [1, 3, 0, 2, 4]),
    # All tiles having the same shape should be shuffled within themselves
    ({(2, 2): [0, 1, 2]}, 2, [2, 1, 0])
])
def test_shuffle_tiles_within_shape_groups(shape_groups, random_state, expected_output):
    random_state = np.random.RandomState(random_state)
    actual_output = F.shuffle_tiles_within_shape_groups(shape_groups, random_state)
    assert actual_output == expected_output, "Output did not match expected mapping"


@pytest.mark.parametrize("group_member,expected", [
    ("e", np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),  # Identity
    ("r90", np.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]])),  # Rotate 90 degrees counterclockwise
    ("r180", np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]])),  # Rotate 180 degrees
    ("r270", np.array([[6, 3, 0], [7, 4, 1], [8, 5, 2]])),  # Rotate 270 degrees counterclockwise
    ("v", np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])),  # Vertical flip
    ("t", np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]])),  # Transpose (reflect over main diagonal)
    ("h", np.array([[2, 1, 0], [5, 4, 3], [8, 7, 6]])),  # Horizontal flip
    ("hvt", np.array([[8, 5, 2], [7, 4, 1], [6, 3, 0]]))  # Transpose (reflect over anti-diagonal)
])
def test_d4_transformations(group_member, expected):
    img = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8)
    transformed_img = FGeometric.d4(img, group_member)
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
        hashes.add(get_md5_hash(FGeometric.d4(image, element)))

    assert len(hashes) == len(set(hashes)), "d4 should generate unique images for all group elements"


@pytest.mark.parametrize("image", RECTANGULAR_IMAGES)
@pytest.mark.parametrize("group_member", d4_group_elements)
def test_d4_output_shape_with_group(image, group_member):
    result = FGeometric.d4(image, group_member)
    if group_member in ['r90', 'r270', 't', 'hvt']:
        assert result.shape[:2] == image.shape[:2][::-1], "Output shape should be the transpose of input shape"
    else:
        assert result.shape == image.shape, "Output shape should match input shape"


@pytest.mark.parametrize("image", RECTANGULAR_IMAGES)
def test_transpose_output_shape(image):
    result = FGeometric.transpose(image)
    assert result.shape[:2] == image.shape[:2][::-1], "Output shape should be the transpose of input shape"


@pytest.mark.parametrize("image", RECTANGULAR_IMAGES)
@pytest.mark.parametrize("factor", [0, 1, 2, 3])
def test_d4_output_shape_with_factor(image, factor):
    result = FGeometric.rot90(image, factor)
    if factor in {1, 3}:
        assert result.shape[:2] == image.shape[:2][::-1], "Output shape should be the transpose of input shape"
    else:
        assert result.shape == image.shape, "Output shape should match input shape"


@pytest.mark.parametrize("bbox, group_member, rows, cols, expected", [
    ((0.05, 0.1, 0.55, 0.6), 'e', 200, 200, (0.05, 0.1, 0.55, 0.6)),  # Identity
    ((0.05, 0.1, 0.55, 0.6), 'r90', 200, 200, (0.1, 0.45, 0.6, 0.95)),  # Rotate 90 degrees CCW
    ((0.05, 0.1, 0.55, 0.6), 'r180', 200, 200, (0.45, 0.4, 0.95, 0.9)),  # Rotate 180 degrees
    ((0.05, 0.1, 0.55, 0.6), 'r270', 200, 200, (0.4, 0.05, 0.9, 0.55)),  # Rotate 270 degrees CCW
    ((0.05, 0.1, 0.55, 0.6), 'v', 200, 200, (0.05, 0.4, 0.55, 0.9)),  # Vertical flip
    ((0.05, 0.1, 0.55, 0.6), 't', 200, 200, (0.1, 0.05, 0.6, 0.55)),  # Transpose around main diagonal
    ((0.05, 0.1, 0.55, 0.6), 'h', 200, 200, (0.45, 0.1, 0.95, 0.6)),  # Horizontal flip
    ((0.05, 0.1, 0.55, 0.6), 'hvt', 200, 200, (1 - 0.6, 1 - 0.55, 1 - 0.1, 1 - 0.05)),  # Transpose around second diagonal
])
def test_bbox_d4(bbox, group_member, rows, cols, expected):
    result = FGeometric.bbox_d4(bbox, group_member, rows, cols)
    assert result == pytest.approx(expected, rel=1e-5), f"Failed for transformation {group_member} with bbox {bbox}"


@pytest.mark.parametrize("keypoint, rows, cols", [
    ((100, 150, 0, 1), 300, 400),  # Example keypoint with arbitrary angle and scale
    ((200, 100, np.pi/4, 0.5), 300, 400),
    ((50, 250, np.pi/2, 2), 300, 400),
])
def test_keypoint_vh_flip_equivalence(keypoint, rows, cols):

    # Perform vertical and then horizontal flip
    hflipped_keypoint = FGeometric.keypoint_hflip(keypoint, rows, cols)
    vhflipped_keypoint = FGeometric.keypoint_vflip(hflipped_keypoint, rows, cols)

    vflipped_keypoint = FGeometric.keypoint_vflip(keypoint, rows, cols)
    hvflipped_keypoint = FGeometric.keypoint_hflip(vflipped_keypoint, rows, cols)

    assert vhflipped_keypoint == pytest.approx(hvflipped_keypoint), \
        "Sequential vflip + hflip not equivalent to hflip + vflip"
    assert vhflipped_keypoint == pytest.approx(FGeometric.keypoint_rot90(keypoint, 2, rows, cols)), \
        "rot180 not equivalent to vflip + hflip"


base_matrix = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
expected_main_diagonal = np.array([[1, 4, 7],
                                   [2, 5, 8],
                                   [3, 6, 9]])
expected_second_diagonal = np.array([[9, 6, 3],
                                     [8, 5, 2],
                                     [7, 4, 1]])


def create_test_matrix(matrix, shape):
    if len(shape) == 2:
        return matrix
    elif len(shape) == 3:
        return np.stack([matrix] * shape[2], axis=-1)


@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 1), (3, 3, 3), (3, 3, 7)])
def test_transpose_2(shape):
    img = create_test_matrix(base_matrix, shape)
    expected_main = create_test_matrix(expected_main_diagonal, shape)
    expected_second = create_test_matrix(expected_second_diagonal, shape)

    assert np.array_equal(FGeometric.transpose(img), expected_main)
    transposed_axis1 = FGeometric.transpose(FGeometric.rot90(img, 2))
    assert np.array_equal(transposed_axis1, expected_second)


def test_planckian_jitter_blackbody():
    img = np.array([[
        [0.4963, 0.6977, 0.1759], [0.7682, 0.8   , 0.2698], [0.0885, 0.161 , 0.1507], [0.132 , 0.2823, 0.0317]],
        [[0.3074, 0.6816, 0.2081], [0.6341, 0.9152, 0.9298], [0.4901, 0.3971, 0.7231],[0.8964, 0.8742, 0.7423]],
        [[0.4556, 0.4194, 0.5263], [0.6323, 0.5529, 0.2437], [0.3489, 0.9527, 0.5846], [0.4017, 0.0362, 0.0332]],
        [[0.0223, 0.1852, 0.1387], [0.1689, 0.3734, 0.2422], [0.2939, 0.3051, 0.8155], [0.5185, 0.932 , 0.7932]]]
    )

    expected_blackbody_plankian_jitter = np.array([
        [[0.735 , 0.6977, 0.0691], [1.    , 0.8   , 0.1059], [0.1311, 0.161 , 0.0592], [0.1955, 0.2823, 0.0124]],
        [[0.4553, 0.6816, 0.0817], [0.9391, 0.9152, 0.365 ], [0.7258, 0.3971, 0.2839], [1.    , 0.8742, 0.2914]],
        [[0.6748, 0.4194, 0.2066], [0.9364, 0.5529, 0.0957], [0.5167, 0.9527, 0.2295], [0.5949, 0.0362, 0.013 ]],
        [[0.033 , 0.1852, 0.0545], [0.2501, 0.3734, 0.0951], [0.4353, 0.3051, 0.3202], [0.7679, 0.932 , 0.3114]]]
    )

    blackbody_plankian_jitter = F.planckian_jitter(img, temperature=3500, mode="blackbody")
    assert np.allclose(blackbody_plankian_jitter, expected_blackbody_plankian_jitter, atol=1e-4)


def test_planckian_jitter_cied():
    img = np.array([
        [[0.4963, 0.6977, 0.1759], [0.7682, 0.8   , 0.2698], [0.0885, 0.161 , 0.1507], [0.132 , 0.2823, 0.0317]],
        [[0.3074, 0.6816, 0.2081], [0.6341, 0.9152, 0.9298], [0.4901, 0.3971, 0.7231], [0.8964, 0.8742, 0.7423]],
        [[0.4556, 0.4194, 0.5263], [0.6323, 0.5529, 0.2437], [0.3489, 0.9527, 0.5846], [0.4017, 0.0362, 0.0332]],
        [[0.0223, 0.1852, 0.1387], [0.1689, 0.3734, 0.2422], [0.2939, 0.3051, 0.8155], [0.5185, 0.932 , 0.7932]]]
    )

    expected_cied_plankian_jitter = np.array([
        [[0.6058, 0.6977, 0.1149], [0.9377, 0.8000, 0.1762], [0.1080, 0.1610, 0.0984], [0.1611, 0.2823, 0.0207]],
        [[0.3752, 0.6816, 0.1359], [0.7740, 0.9152, 0.6072], [0.5982, 0.3971, 0.4722], [1.0000, 0.8742, 0.4848]],
        [[0.5561, 0.4194, 0.3437], [0.7718, 0.5529, 0.1592], [0.4259, 0.9527, 0.3818], [0.4903, 0.0362, 0.0217]],
        [[0.0272, 0.1852, 0.0906], [0.2062, 0.3734, 0.1582], [0.3587, 0.3051, 0.5326], [0.6329, 0.9320, 0.5180]]]
    )

    cied_plankian_jitter = F.planckian_jitter(img, temperature=4500, mode="cied")
    assert np.allclose(cied_plankian_jitter, expected_cied_plankian_jitter, atol=1e-4)


@pytest.mark.parametrize("image", IMAGES)
def test_random_tone_curve(image):
    low_y = 0.1
    high_y = 0.9

    num_channels = get_num_channels(image)

    result_float_value = F.move_tone_curve(image, low_y, high_y)
    result_array_value = F.move_tone_curve(image, np.array([low_y] * num_channels), np.array([high_y] * num_channels))

    assert np.array_equal(result_float_value, result_array_value)

    assert result_float_value.dtype == image.dtype
    assert result_float_value.shape == image.shape


@pytest.mark.parametrize("image", UINT8_IMAGES)
@pytest.mark.parametrize("color_shift, intensity", [(0, 0), (0.5, 0.5), (1, 1)])
def test_iso_noise(image, color_shift, intensity):
    image = RECTANGULAR_UINT8_IMAGE

    # Convert image to float and back
    float_image = F.to_float(image)

    # Generate noise using the same random state instance
    set_seed(42)
    result_uint8 = F.iso_noise(image, color_shift=color_shift, intensity=intensity)

    set_seed(42)
    result_float = F.iso_noise(float_image, color_shift=color_shift, intensity=intensity)

    result_float = F.from_float(result_float, dtype=np.uint8)  # Convert the float result back to uint8

    assert np.array_equal(result_uint8, result_float)
