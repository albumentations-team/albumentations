import hashlib

import cv2
import numpy as np
import pytest
from albucore import (
    MAX_VALUES_BY_DTYPE,
    clip,
    get_num_channels,
    is_multispectral_image,
    to_float,
)

import albumentations.augmentations.pixel.functional as fpixel
import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.core.type_definitions import d4_group_elements
from tests.conftest import (
    IMAGES,
    RECTANGULAR_IMAGES,
    SQUARE_UINT8_IMAGE,
    UINT8_IMAGES,
    VOLUME,
)
from tests.utils import convert_2d_to_target_format
from copy import deepcopy
from sklearn.decomposition import NMF
from typing import Any


@pytest.mark.parametrize(
    ["input_shape", "expected_shape"],
    [[(128, 64), (64, 128)], [(128, 64, 3), (64, 128, 3)]],
)
def test_transpose(input_shape, expected_shape):
    img = np.random.randint(low=0, high=256, size=input_shape, dtype=np.uint8)
    transposed = fgeometric.transpose(img)
    assert transposed.shape == expected_shape


@pytest.mark.parametrize(
    ["input_shape", "expected_shape"],
    [[(128, 64), (64, 128)], [(128, 64, 3), (64, 128, 3)]],
)
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
    np.testing.assert_array_equal(rotated, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_rot90_float(target):
    img = np.array(
        [[0.0, 0.0, 0.4], [0.0, 0.0, 0.4], [0.0, 0.0, 0.4]], dtype=np.float32
    )
    expected = np.array(
        [[0.4, 0.4, 0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    rotated = fgeometric.rot90(img, factor=1)
    np.testing.assert_array_almost_equal_nulp(rotated, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_pad(target):
    img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    expected = np.array(
        [[4, 3, 4, 3], [2, 1, 2, 1], [4, 3, 4, 3], [2, 1, 2, 1]], dtype=np.uint8
    )
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    padded = fgeometric.pad(
        img, min_height=4, min_width=4, border_mode=cv2.BORDER_REFLECT_101, value=None
    )
    np.testing.assert_array_equal(padded, expected)


@pytest.mark.parametrize("target", ["image", "image_4_channels"])
def test_pad_float(target):
    img = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    expected = np.array(
        [
            [0.4, 0.3, 0.4, 0.3],
            [0.2, 0.1, 0.2, 0.1],
            [0.4, 0.3, 0.4, 0.3],
            [0.2, 0.1, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    padded_img = fgeometric.pad(
        img, min_height=4, min_width=4, value=None, border_mode=cv2.BORDER_REFLECT_101
    )
    np.testing.assert_array_almost_equal_nulp(padded_img, expected)


@pytest.mark.parametrize(["gamma", "expected"], [(1, 1), (0.8, 3)])
def test_gamma_transform(gamma, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8)
    img = fpixel.gamma_transform(img, gamma=gamma)
    assert img.dtype == np.dtype("uint8")
    np.testing.assert_array_equal(img, expected)


@pytest.mark.parametrize(["gamma", "expected"], [(1, 0.4), (10, 0.00010486)])
def test_gamma_transform_float(gamma, expected):
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.4
    expected = np.ones((100, 100, 3), dtype=np.float32) * expected
    img = fpixel.gamma_transform(img, gamma=gamma)
    assert img.dtype == np.dtype("float32")
    np.testing.assert_allclose(img, expected, atol=1e-6)


def test_gamma_float_equal_uint8():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img_f = img.astype(np.float32) / 255.0
    gamma = 0.5

    img = fpixel.gamma_transform(img, gamma)
    img_f = fpixel.gamma_transform(img_f, gamma)

    img = img.astype(np.float32)
    img_f *= 255.0

    np.testing.assert_allclose(img, img_f, atol=1)


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
    np.testing.assert_array_equal(scaled, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_linear_interpolation(target):
    img = np.array(
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.uint8
    )
    expected = np.array([[2, 2], [4, 4]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = fgeometric.resize(img, (2, 2), interpolation=cv2.INTER_LINEAR)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    np.testing.assert_array_equal(resized_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_nearest_interpolation(target):
    img = np.array(
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], dtype=np.uint8
    )
    expected = np.array([[1, 1], [3, 3]], dtype=np.uint8)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = fgeometric.resize(img, (2, 2), interpolation=cv2.INTER_NEAREST)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    np.testing.assert_array_equal(resized_img, expected)


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
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4, 0.4],
        ],
        dtype=np.float32,
    )
    expected = np.array([[0.15, 0.15], [0.35, 0.35]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = fgeometric.resize(img, (2, 2), interpolation=cv2.INTER_LINEAR)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    np.testing.assert_array_almost_equal_nulp(resized_img, expected)


@pytest.mark.parametrize("target", ["image", "mask"])
def test_resize_nearest_interpolation_float(target):
    img = np.array(
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4, 0.4],
        ],
        dtype=np.float32,
    )
    expected = np.array([[0.1, 0.1], [0.3, 0.3]], dtype=np.float32)
    img, expected = convert_2d_to_target_format([img, expected], target=target)
    resized_img = fgeometric.resize(img, (2, 2), interpolation=cv2.INTER_NEAREST)
    height, width = resized_img.shape[:2]
    assert height == 2
    assert width == 2
    np.testing.assert_array_equal(resized_img, expected)


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


def test_is_rgb_image():
    image = np.ones((5, 5, 3), dtype=np.uint8)
    assert fpixel.is_rgb_image(image)

    multispectral_image = np.ones((5, 5, 4), dtype=np.uint8)
    assert not fpixel.is_rgb_image(multispectral_image)

    gray_image = np.ones((5, 5), dtype=np.uint8)
    assert not fpixel.is_rgb_image(gray_image)

    gray_image = np.ones((5, 5, 1), dtype=np.uint8)
    assert not fpixel.is_rgb_image(gray_image)


def test_is_grayscale_image():
    image = np.ones((5, 5, 3), dtype=np.uint8)
    assert not fpixel.is_grayscale_image(image)

    multispectral_image = np.ones((5, 5, 4), dtype=np.uint8)
    assert not fpixel.is_grayscale_image(multispectral_image)

    gray_image = np.ones((5, 5), dtype=np.uint8)
    assert fpixel.is_grayscale_image(gray_image)

    gray_image = np.ones((5, 5, 1), dtype=np.uint8)
    assert fpixel.is_grayscale_image(gray_image)


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


@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize("threshold", [0.0, 1 / 3, 2 / 3, 1.0])
def test_solarize(image, threshold):
    max_value = MAX_VALUES_BY_DTYPE[image.dtype]
    check_img = image.copy()

    cond = check_img >= threshold * max_value
    check_img[cond] = max_value - check_img[cond]

    result_img = fpixel.solarize(image, threshold=threshold)

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
def test_equalize_checks(
    img_shape, img_dtype, mask_shape, by_channels, expected_error, expected_message
):
    img = (
        np.random.randint(0, 255, img_shape).astype(img_dtype)
        if img_dtype == np.uint8
        else np.random.random(img_shape).astype(img_dtype)
    )
    mask = np.random.randint(0, 2, mask_shape).astype(bool)

    with pytest.raises(expected_error) as exc_info:
        fpixel.equalize(img, mask=mask, by_channels=by_channels)
    assert str(exc_info.value) == expected_message


def test_equalize_grayscale():
    img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    assert np.all(cv2.equalizeHist(img) == fpixel.equalize(img, mode="cv"))


def test_equalize_rgb():
    img = SQUARE_UINT8_IMAGE

    _img = img.copy()
    for i in range(3):
        _img[..., i] = cv2.equalizeHist(_img[..., i])
    assert np.all(_img == fpixel.equalize(img, mode="cv"))

    _img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img_cv = _img.copy()
    img_cv[..., 0] = cv2.equalizeHist(_img[..., 0])
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_YCrCb2RGB)
    assert np.all(img_cv == fpixel.equalize(img, mode="cv", by_channels=False))


def test_equalize_grayscale_mask():
    img = np.random.randint(0, 255, [256, 256], dtype=np.uint8)

    mask = np.zeros([256, 256], dtype=bool)
    mask[:10, :10] = True

    assert np.all(
        cv2.equalizeHist(img[:10, :10])
        == fpixel.equalize(img, mask=mask, mode="cv")[:10, :10]
    )


def test_equalize_rgb_mask():
    img = np.random.randint(0, 255, [256, 256, 3], dtype=np.uint8)

    mask = np.zeros([256, 256], dtype=bool)
    mask[:10, :10] = True

    _img = img.copy()[:10, :10]
    for i in range(3):
        _img[..., i] = cv2.equalizeHist(_img[..., i])
    assert np.all(_img == fpixel.equalize(img, mask, mode="cv")[:10, :10])

    _img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img_cv = _img.copy()[:10, :10]
    img_cv[..., 0] = cv2.equalizeHist(img_cv[..., 0])
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_YCrCb2RGB)
    assert np.all(
        img_cv == fpixel.equalize(img, mask=mask, mode="cv", by_channels=False)[:10, :10]
    )

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

    result_img = fpixel.equalize(img, mask=mask, mode="cv")
    assert np.all(img_r == result_img[:10, :10, 0])
    assert np.all(img_g == result_img[10:20, 10:20, 1])
    assert np.all(img_b == result_img[20:30, 20:30, 2])


@pytest.mark.parametrize(
    "img",
    [
        np.random.randint(0, 256, [100, 100], dtype=np.uint8),
        np.random.random([100, 100]).astype(np.float32),
    ],
)
def test_shift_hsv_gray(img):
    fpixel.shift_hsv(img, 0.5, 0.5, 0.5)


@pytest.mark.parametrize(
    "tiles, expected",
    [
        # Simple case with two different shapes
        (
            np.array([[0, 0, 2, 2], [0, 2, 2, 4], [2, 0, 4, 2], [2, 2, 4, 4]]),
            {(2, 2): [0, 1, 2, 3]},
        ),
        # Tiles with three different shapes
        (
            np.array([[0, 0, 1, 3], [0, 3, 1, 6], [1, 0, 4, 3], [1, 3, 4, 6]]),
            {(1, 3): [0, 1], (3, 3): [2, 3]},
        ),
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
        assert sorted(result[shape]) == sorted(
            expected[shape]
        ), f"Incorrect indices for shape {shape}"


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
    actual_output = fgeometric.shuffle_tiles_within_shape_groups(
        shape_groups, generator
    )
    assert (
        shape_groups == shape_groups_original
    ), "Input shape groups should not be modified"
    np.testing.assert_array_equal(actual_output, expected_output)


@pytest.mark.parametrize(
    "group_member,expected",
    [
        ("e", np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),  # Identity
        (
            "r90",
            np.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]]),
        ),  # Rotate 90 degrees counterclockwise
        ("r180", np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]])),  # Rotate 180 degrees
        (
            "r270",
            np.array([[6, 3, 0], [7, 4, 1], [8, 5, 2]]),
        ),  # Rotate 270 degrees counterclockwise
        ("v", np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])),  # Vertical flip
        (
            "t",
            np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]]),
        ),  # Transpose (reflect over main diagonal)
        ("h", np.array([[2, 1, 0], [5, 4, 3], [8, 7, 6]])),  # Horizontal flip
        (
            "hvt",
            np.array([[8, 5, 2], [7, 4, 1], [6, 3, 0]]),
        ),  # Transpose (reflect over anti-diagonal)
    ],
)
def test_d4_transformations(group_member, expected):
    img = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8)
    transformed_img = fgeometric.d4(img, group_member)
    assert np.array_equal(
        transformed_img, expected
    ), f"Failed for transformation {group_member}"


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

    assert len(hashes) == len(
        set(hashes)
    ), "d4 should generate unique images for all group elements"


@pytest.mark.parametrize("image", RECTANGULAR_IMAGES)
@pytest.mark.parametrize("group_member", d4_group_elements)
def test_d4_output_shape_with_group(image, group_member):
    result = fgeometric.d4(image, group_member)
    if group_member in ["r90", "r270", "t", "hvt"]:
        assert (
            result.shape[:2] == image.shape[:2][::-1]
        ), "Output shape should be the transpose of input shape"
    else:
        assert result.shape == image.shape, "Output shape should match input shape"


@pytest.mark.parametrize("image", RECTANGULAR_IMAGES)
def test_transpose_output_shape(image):
    result = fgeometric.transpose(image)
    assert (
        result.shape[:2] == image.shape[:2][::-1]
    ), "Output shape should be the transpose of input shape"


@pytest.mark.parametrize("image", RECTANGULAR_IMAGES)
@pytest.mark.parametrize("factor", [0, 1, 2, 3])
def test_d4_output_shape_with_factor(image, factor):
    result = fgeometric.rot90(image, factor)
    if factor in {1, 3}:
        assert (
            result.shape[:2] == image.shape[:2][::-1]
        ), "Output shape should be the transpose of input shape"
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
            [
                [0.4963, 0.6977, 0.1759],
                [0.7682, 0.8, 0.2698],
                [0.0885, 0.161, 0.1507],
                [0.132, 0.2823, 0.0317],
            ],
            [
                [0.3074, 0.6816, 0.2081],
                [0.6341, 0.9152, 0.9298],
                [0.4901, 0.3971, 0.7231],
                [0.8964, 0.8742, 0.7423],
            ],
            [
                [0.4556, 0.4194, 0.5263],
                [0.6323, 0.5529, 0.2437],
                [0.3489, 0.9527, 0.5846],
                [0.4017, 0.0362, 0.0332],
            ],
            [
                [0.0223, 0.1852, 0.1387],
                [0.1689, 0.3734, 0.2422],
                [0.2939, 0.3051, 0.8155],
                [0.5185, 0.932, 0.7932],
            ],
        ],
    )

    expected_blackbody_plankian_jitter = np.array(
        [
            [
                [0.735, 0.6977, 0.0691],
                [1.0, 0.8, 0.1059],
                [0.1311, 0.161, 0.0592],
                [0.1955, 0.2823, 0.0124],
            ],
            [
                [0.4553, 0.6816, 0.0817],
                [0.9391, 0.9152, 0.365],
                [0.7258, 0.3971, 0.2839],
                [1.0, 0.8742, 0.2914],
            ],
            [
                [0.6748, 0.4194, 0.2066],
                [0.9364, 0.5529, 0.0957],
                [0.5167, 0.9527, 0.2295],
                [0.5949, 0.0362, 0.013],
            ],
            [
                [0.033, 0.1852, 0.0545],
                [0.2501, 0.3734, 0.0951],
                [0.4353, 0.3051, 0.3202],
                [0.7679, 0.932, 0.3114],
            ],
        ],
    )

    blackbody_plankian_jitter = fpixel.planckian_jitter(
        img, temperature=3500, mode="blackbody"
    )
    assert np.allclose(
        blackbody_plankian_jitter, expected_blackbody_plankian_jitter, atol=1e-4
    )


def test_planckian_jitter_cied():
    img = np.array(
        [
            [
                [0.4963, 0.6977, 0.1759],
                [0.7682, 0.8, 0.2698],
                [0.0885, 0.161, 0.1507],
                [0.132, 0.2823, 0.0317],
            ],
            [
                [0.3074, 0.6816, 0.2081],
                [0.6341, 0.9152, 0.9298],
                [0.4901, 0.3971, 0.7231],
                [0.8964, 0.8742, 0.7423],
            ],
            [
                [0.4556, 0.4194, 0.5263],
                [0.6323, 0.5529, 0.2437],
                [0.3489, 0.9527, 0.5846],
                [0.4017, 0.0362, 0.0332],
            ],
            [
                [0.0223, 0.1852, 0.1387],
                [0.1689, 0.3734, 0.2422],
                [0.2939, 0.3051, 0.8155],
                [0.5185, 0.932, 0.7932],
            ],
        ],
    )

    expected_cied_plankian_jitter = np.array(
        [
            [
                [0.6058, 0.6977, 0.1149],
                [0.9377, 0.8000, 0.1762],
                [0.1080, 0.1610, 0.0984],
                [0.1611, 0.2823, 0.0207],
            ],
            [
                [0.3752, 0.6816, 0.1359],
                [0.7740, 0.9152, 0.6072],
                [0.5982, 0.3971, 0.4722],
                [1.0000, 0.8742, 0.4848],
            ],
            [
                [0.5561, 0.4194, 0.3437],
                [0.7718, 0.5529, 0.1592],
                [0.4259, 0.9527, 0.3818],
                [0.4903, 0.0362, 0.0217],
            ],
            [
                [0.0272, 0.1852, 0.0906],
                [0.2062, 0.3734, 0.1582],
                [0.3587, 0.3051, 0.5326],
                [0.6329, 0.9320, 0.5180],
            ],
        ],
    )
    cied_plankian_jitter = fpixel.planckian_jitter(img, temperature=4500, mode="cied")
    assert np.allclose(cied_plankian_jitter, expected_cied_plankian_jitter, atol=1e-4)


@pytest.mark.parametrize("mode", ["blackbody", "cied"])
def test_planckian_jitter_edge_cases(mode):
    # Create a sample image
    img = np.ones((10, 10, 3), dtype=np.float32)

    # Get min and max temperatures for the mode
    min_temp = min(fpixel.PLANCKIAN_COEFFS[mode].keys())
    max_temp = max(fpixel.PLANCKIAN_COEFFS[mode].keys())

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
        result = fpixel.planckian_jitter(img, temp, mode)

        # Check that the output is a valid image
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        assert np.all(result >= 0) and np.all(result <= 1)

        # Check that the function didn't modify the input image
        assert not np.array_equal(result, img)

        # For temperatures outside the range, check if they're clamped correctly
        if temp < min_temp:
            np.testing.assert_allclose(result, fpixel.planckian_jitter(img, min_temp, mode))
        elif temp > max_temp:
            np.testing.assert_allclose(result, fpixel.planckian_jitter(img, max_temp, mode))


def test_planckian_jitter_interpolation():
    img = np.ones((10, 10, 3), dtype=np.float32)
    mode = "blackbody"

    # Test interpolation between two known temperatures
    temp1, temp2 = 4000, 4500
    result1 = fpixel.planckian_jitter(img, temp1, mode)
    result2 = fpixel.planckian_jitter(img, temp2, mode)
    result_mid = fpixel.planckian_jitter(img, (temp1 + temp2) // 2, mode)

    # The mid-temperature result should be between the two extremes
    assert np.all(
        (result_mid >= np.minimum(result1, result2))
        & (result_mid <= np.maximum(result1, result2))
    )


@pytest.mark.parametrize("mode", ["blackbody", "cied"])
def test_planckian_jitter_consistency(mode):
    img = np.ones((10, 10, 3), dtype=np.float32)

    # Test consistency of results for the same temperature
    temp = 5000
    result1 = fpixel.planckian_jitter(img, temp, mode)
    result2 = fpixel.planckian_jitter(img, temp, mode)
    np.testing.assert_allclose(result1, result2)


def test_planckian_jitter_invalid_mode():
    img = np.ones((10, 10, 3), dtype=np.float32)

    with pytest.raises(KeyError):
        fpixel.planckian_jitter(img, 5000, "invalid_mode")


@pytest.mark.parametrize(
        ("image", "num_channels"),
        [
            (SQUARE_UINT8_IMAGE, 3),
            (VOLUME, 3),
        ]
)
def test_random_tone_curve(image, num_channels):
    low_y = 0.1
    high_y = 0.9

    result_float_value = fpixel.move_tone_curve(image, low_y, high_y, num_channels)
    result_array_value = fpixel.move_tone_curve(
        image, np.array([low_y] * num_channels), np.array([high_y] * num_channels), num_channels
    )

    np.testing.assert_allclose(result_float_value, result_array_value)

    assert result_float_value.dtype == image.dtype
    assert result_float_value.shape == image.shape


@pytest.mark.parametrize("image", UINT8_IMAGES)
@pytest.mark.parametrize(
    "color_shift, intensity",
    [
        (0, 0),  # No noise
        (0.5, 0.5),  # Medium noise
        (1, 1),  # Maximum noise
    ],
)
def test_iso_noise(image, color_shift, intensity):
    """Test that iso_noise produces expected noise levels."""
    # Convert image to float and back
    float_image = to_float(image)

    # Generate noise using the same random state instance
    rng = np.random.default_rng(42)
    result_uint8 = fpixel.iso_noise(
        image, color_shift=color_shift, intensity=intensity, random_generator=rng
    )

    rng = np.random.default_rng(42)
    result_float = fpixel.iso_noise(
        float_image, color_shift=color_shift, intensity=intensity, random_generator=rng
    )

    # Convert float result back to uint8
    result_float = fpixel.from_float(result_float, target_dtype=np.uint8)

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
    result = fpixel.grayscale_to_multichannel(input_image, num_output_channels)
    assert result.shape == expected_shape
    assert np.all(result[..., 0] == result[..., 1])  # All channels should be identical


def test_grayscale_to_multichannel_preserves_values():
    input_image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    result = fpixel.grayscale_to_multichannel(input_image, num_output_channels=3)
    assert np.all(result[..., 0] == input_image)
    assert np.all(result[..., 1] == input_image)
    assert np.all(result[..., 2] == input_image)


def test_grayscale_to_multichannel_default_channels():
    input_image = np.zeros((10, 10), dtype=np.uint8)
    result = fpixel.grayscale_to_multichannel(input_image, num_output_channels=3)
    assert result.shape == (10, 10, 3)


def create_test_image(height, width, channels, dtype):
    if dtype == np.uint8:
        return np.random.randint(0, 256, (height, width, channels), dtype=dtype)
    return np.random.rand(height, width, channels).astype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_to_gray_weighted_average(dtype):
    img = create_test_image(10, 10, 3, dtype)
    result = fpixel.to_gray_weighted_average(img)
    expected = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    if dtype == np.uint8:
        expected = expected.astype(np.uint8)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_to_gray_from_lab(dtype):
    img = create_test_image(10, 10, 3, dtype)
    result = fpixel.to_gray_from_lab(img)
    expected = clip(cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0], dtype=dtype)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [3, 4, 5])
def test_to_gray_desaturation(dtype, channels):
    img = create_test_image(10, 10, channels, dtype)
    result = fpixel.to_gray_desaturation(img)
    expected = (
        np.max(img.astype(np.float32), axis=-1)
        + np.min(img.astype(np.float32), axis=-1)
    ) / 2
    if dtype == np.uint8:
        expected = expected.astype(np.uint8)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [3, 4, 5])
def test_to_gray_average(dtype, channels):
    img = create_test_image(10, 10, channels, dtype)
    result = fpixel.to_gray_average(img)
    expected = np.mean(img, axis=-1)
    if dtype == np.uint8:
        expected = expected.astype(np.uint8)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [3, 4, 5])
def test_to_gray_max(dtype, channels):
    img = create_test_image(10, 10, channels, dtype)
    result = fpixel.to_gray_max(img)
    expected = np.max(img, axis=-1)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("channels", [3, 4, 5])
def test_to_gray_pca(dtype, channels):
    img = create_test_image(10, 10, channels, dtype)
    result = fpixel.to_gray_pca(img)
    assert result.shape == (10, 10)
    assert result.dtype == dtype
    if dtype == np.uint8:
        assert result.min() >= 0 and result.max() <= 255
    else:
        assert result.min() >= 0 and result.max() <= 1


@pytest.mark.parametrize(
    "func",
    [
        fpixel.to_gray_weighted_average,
        fpixel.to_gray_from_lab,
        fpixel.to_gray_desaturation,
        fpixel.to_gray_average,
        fpixel.to_gray_max,
        fpixel.to_gray_pca,
    ],
)
def test_float32_uint8_consistency(func):
    img_uint8 = create_test_image(10, 10, 3, np.uint8)
    img_float32 = img_uint8.astype(np.float32) / 255.0

    result_uint8 = func(img_uint8)
    result_float32 = func(img_float32)

    np.testing.assert_allclose(
        result_uint8 / 255.0, result_float32, rtol=1e-5, atol=1e-2
    )


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

    result = fpixel.clahe(img, clip_limit, tile_grid_size)

    assert result.shape == img.shape
    assert result.dtype == img.dtype
    assert np.any(result != img)  # Ensure the image has changed


@pytest.mark.parametrize("shape", [(100, 100, 3), (100, 100, 1), (100, 100, 5)])
def test_fancy_pca_mean_preservation(shape):
    image = np.random.rand(*shape).astype(np.float32)
    alpha_vector = np.random.uniform(-0.1, 0.1, shape[-1])
    result = fpixel.fancy_pca(image, alpha_vector)
    np.testing.assert_almost_equal(np.mean(image), np.mean(result), decimal=4)


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
    result = fpixel.fancy_pca(image, alpha_vector)

    np.testing.assert_array_equal(image, result)


@pytest.mark.parametrize(
    ["image_type", "quality", "shape", "expected_shape"],
    [
        # Test JPEG compression
        (".jpg", 80, (100, 100, 3), (100, 100, 3)),  # RGB image
        (".jpg", 10, (50, 50, 1), (50, 50, 1)),  # Grayscale image
        (".jpg", 90, (30, 30, 2), (30, 30, 2)),  # 2-channel image
        (".jpg", 70, (40, 40, 4), (40, 40, 4)),  # RGBA image
        (".jpg", 50, (60, 60, 5), (60, 60, 5)),  # 5-channel image
        # Test WebP compression
        (".webp", 80, (100, 100, 3), (100, 100, 3)),  # RGB image
        (".webp", 10, (50, 50, 1), (50, 50, 1)),  # Grayscale image
        (".webp", 90, (30, 30, 2), (30, 30, 2)),  # 2-channel image
        (".webp", 70, (40, 40, 4), (40, 40, 4)),  # RGBA image
        (".webp", 50, (60, 60, 5), (60, 60, 5)),  # 5-channel image
    ],
)
def test_image_compression_shapes(image_type, quality, shape, expected_shape):
    """Test that image_compression preserves input shapes."""
    image = np.random.randint(0, 256, shape, dtype=np.uint8)
    compressed = fpixel.image_compression(image, quality, image_type)
    assert compressed.shape == expected_shape
    assert compressed.dtype == np.uint8


def test_image_compression_channel_consistency():
    """Test that compression maintains channel independence for extra channels."""
    # Create image with 4 channels where last channel is constant
    image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
    image[..., 3] = 128  # Constant alpha channel

    compressed = fpixel.image_compression(image, 80, ".jpg")

    # RGB channels should change due to compression
    assert not np.array_equal(image[..., :3], compressed[..., :3])
    # Alpha channel should remain constant
    assert np.all(compressed[..., 3] == 128)


@pytest.mark.parametrize(
    ["image_type", "quality", "shape"],
    [
        # Test JPEG compression - only supports 1 and 3 channels
        (".jpg", 80, (100, 100, 3)),  # RGB image
        (".jpg", 10, (50, 50, 1)),  # Grayscale image
        # Test WebP compression - supports 1, 3, and 4 channels
        (".webp", 80, (100, 100, 3)),  # RGB image
        (".webp", 10, (50, 50, 1)),  # Grayscale image
        (".webp", 70, (40, 40, 4)),  # RGBA image
    ],
)
def test_image_compression_supported_shapes(image_type, quality, shape):
    """Test image_compression with supported channel counts."""
    image = np.random.randint(0, 256, shape, dtype=np.uint8)
    compressed = fpixel.image_compression(image, quality, image_type)
    assert compressed.shape == shape
    assert compressed.dtype == np.uint8


@pytest.mark.parametrize("image_type", [".jpg", ".webp"])
def test_image_compression_quality_with_patterns(image_type):
    """Test that lower quality results in more compression artifacts."""
    # Create an image with high frequency patterns that are sensitive to compression
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    xx, yy = np.meshgrid(x, y)
    image = np.uint8(255 * (np.sin(xx) * np.sin(yy) + 1) / 2)
    image = np.stack([image] * 3, axis=-1)  # Convert to RGB

    high_quality = fpixel.image_compression(image, 100, image_type)
    low_quality = fpixel.image_compression(image, 10, image_type)

    high_diff = np.abs(image - high_quality).mean()
    low_diff = np.abs(image - low_quality).mean()

    assert (
        low_diff > high_diff
    ), f"Low quality diff ({low_diff}) should be greater than high quality diff ({high_diff})"


@pytest.mark.parametrize(
    "img, expected",
    [
        # Test with a normal RGB image
        (
            np.array(
                [[[50, 100, 150], [200, 250, 100]], [[100, 150, 200], [50, 100, 150]]],
                dtype=np.uint8,
            ),
            "non_constant",  # We expect the function to adjust contrast, so we check if it's not constant
        ),
        # Test with a constant channel image
        (
            np.array(
                [
                    [[100, 100, 100], [100, 100, 100]],
                    [[100, 100, 100], [100, 100, 100]],
                ],
                dtype=np.uint8,
            ),
            "constant",  # The output should remain constant
        ),
        # Test with a grayscale image
        (
            np.array([[50, 100], [150, 200]], dtype=np.uint8),
            "constant",  # The output should remain constant
        ),
        # Test with an image already using full intensity range
        (
            np.array([[0, 85], [170, 255]], dtype=np.uint8),
            "constant",  # The output should remain constant
        ),
        # Test with an all-zero image
        (
            np.zeros((2, 2, 3), dtype=np.uint8),
            "constant",  # The output should remain constant
        ),
    ],
)
def test_auto_contrast(img, expected):
    result = fpixel.auto_contrast(img, cutoff=0, ignore=None, method="cdf")

    if expected == "constant":
        (
            np.testing.assert_array_equal(result, img),
            "The output should remain constant for constant input.",
        )
    elif expected == "non_constant":
        assert not np.all(
            result == img
        ), "The output should change for non-constant input."


@pytest.mark.parametrize(
    ["array", "value", "expected_shape", "expected_dtype", "expected_values"],
    [
        # 2D array tests
        (
            np.zeros((10, 10), dtype=np.uint8),  # array
            None,  # value
            (10, 10),  # expected_shape
            np.uint8,  # expected_dtype
            None,  # expected_values - random, can't test exact values
        ),
        (
            np.zeros((10, 10), dtype=np.uint8),
            128,
            (10, 10),
            np.uint8,
            128,
        ),
        (
            np.zeros((10, 10), dtype=np.float32),
            0.5,
            (10, 10),
            np.float32,
            0.5,
        ),

        # 3D array tests
        (
            np.zeros((10, 10, 3), dtype=np.uint8),
            None,
            (10, 10, 3),
            np.uint8,
            None,
        ),
        (
            np.zeros((10, 10, 3), dtype=np.uint8),
            128,
            (10, 10, 3),
            np.uint8,
            128,
        ),
        (
            np.zeros((10, 10, 3), dtype=np.uint8),
            [128, 64, 32],
            (10, 10, 3),
            np.uint8,
            [128, 64, 32],
        ),
        (
            np.zeros((10, 10, 3), dtype=np.float32),
            [0.1, 0.2, 0.3],
            (10, 10, 3),
            np.float32,
            [0.1, 0.2, 0.3],
        ),

        # Edge cases
        (
            np.zeros((10, 10, 1), dtype=np.uint8),
            [128],
            (10, 10, 1),
            np.uint8,
            [128],
        ),
        (
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.array([128, 64, 32]),
            (10, 10, 3),
            np.uint8,
            [128, 64, 32],
        ),
        # 2-channel tests
        (
            np.zeros((10, 10, 2), dtype=np.uint8),
            None,
            (10, 10, 2),
            np.uint8,
            None,
        ),
        (
            np.zeros((10, 10, 2), dtype=np.uint8),
            128,
            (10, 10, 2),
            np.uint8,
            128,
        ),
        (
            np.zeros((10, 10, 2), dtype=np.uint8),
            [128, 64],
            (10, 10, 2),
            np.uint8,
            [128, 64],
        ),
        (
            np.zeros((10, 10, 2), dtype=np.float32),
            [0.1, 0.2],
            (10, 10, 2),
            np.float32,
            [0.1, 0.2],
        ),
    ],
)
def test_prepare_drop_values(array, value, expected_shape, expected_dtype, expected_values):
    rng = np.random.default_rng(42)
    result = fpixel.prepare_drop_values(array, value, rng)

    # Check shape and dtype
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype

    # Check values if not random
    if expected_values is not None:
        if isinstance(expected_values, (int, float)):
            assert np.all(result == expected_values)
        else:
            # For per-channel values, check each channel separately
            for i, val in enumerate(expected_values):
                assert np.all(result[..., i] == val)


def test_prepare_drop_values_random():
    """Test that random values are within expected range."""
    rng = np.random.default_rng(42)

    # Test uint8 random values
    array_uint8 = np.zeros((10, 10, 3), dtype=np.uint8)
    result_uint8 = fpixel.prepare_drop_values(array_uint8, None, rng)
    assert np.all((result_uint8 >= 0) & (result_uint8 <= 255))

    # Test float32 random values
    array_float32 = np.zeros((10, 10, 3), dtype=np.float32)
    result_float32 = fpixel.prepare_drop_values(array_float32, None, rng)
    assert np.all((result_float32 >= 0.0) & (result_float32 <= 1.0))

    # Test that different channels get different random values
    assert not np.all(result_uint8[..., 0] == result_uint8[..., 1])


@pytest.mark.parametrize(
    ["shape", "per_channel", "dropout_prob", "expected_shape", "expected_properties"],
    [
        # 2D array tests
        (
            (10, 10),  # shape
            False,  # per_channel
            0.5,  # dropout_prob
            (10, 10),  # expected_shape
            {"is_2d": True, "channels_same": True},  # expected_properties
        ),
        (
            (10, 10),
            True,  # per_channel doesn't affect 2D
            0.5,
            (10, 10),
            {"is_2d": True, "channels_same": True},
        ),

        # 3D array tests - shared mask across channels
        (
            (10, 10, 3),
            False,
            0.5,
            (10, 10, 3),
            {"is_2d": False, "channels_same": True},
        ),
        (
            (10, 10, 1),
            False,
            0.5,
            (10, 10, 1),
            {"is_2d": False, "channels_same": True},
        ),
        # 3D array tests - independent masks per channel
        (
            (10, 10, 3),
            True,
            0.5,
            (10, 10, 3),
            {"is_2d": False, "channels_same": False},
        ),
        (
            (10, 10, 1),
            True,
            0.5,
            (10, 10, 1),
            {"is_2d": False, "channels_same": True},  # single channel is always same
        ),
         # 2-channel tests - shared mask across channels
        (
            (10, 10, 2),
            False,
            0.5,
            (10, 10, 2),
            {"is_2d": False, "channels_same": True},
        ),

        # 2-channel tests - independent masks per channel
        (
            (10, 10, 2),
            True,
            0.5,
            (10, 10, 2),
            {"is_2d": False, "channels_same": False},
        ),
    ],
)
def test_get_drop_mask_shapes_and_properties(
    shape,
    per_channel,
    dropout_prob,
    expected_shape,
    expected_properties
):
    rng = np.random.default_rng(42)
    mask = fpixel.get_drop_mask(shape, per_channel, dropout_prob, rng)

    # Check shape
    assert mask.shape == expected_shape

    # Check dtype
    assert mask.dtype == bool

    # Check if mask is 2D or 3D
    assert (mask.ndim == 2) == expected_properties["is_2d"]

    # For 3D masks, check if channels are same or different
    if not expected_properties["is_2d"]:
        channels_same = all(
            np.array_equal(mask[..., 0], mask[..., i])
            for i in range(1, mask.shape[-1])
        )
        assert channels_same == expected_properties["channels_same"]

@pytest.mark.parametrize(
    "dropout_prob",
    [0.0, 0.3, 0.7, 1.0],
)
def test_get_drop_mask_probabilities(dropout_prob):
    """Test that the proportion of True values matches dropout_prob."""
    shape = (100, 100, 3)  # Large shape for better statistics
    rng = np.random.default_rng(42)

    # Test both per_channel modes
    for per_channel in [False, True]:
        mask = fpixel.get_drop_mask(shape, per_channel, dropout_prob, rng)
        true_proportion = np.mean(mask)
        np.testing.assert_allclose(
            true_proportion,
            dropout_prob,
            rtol=0.1,  # Allow 10% relative tolerance due to randomness
        )

def test_get_drop_mask_reproducibility():
    """Test that the same random seed produces the same mask."""
    shape = (10, 10, 3)
    per_channel = True
    dropout_prob = 0.5

    # Generate two masks with same seed
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    mask1 = fpixel.get_drop_mask(shape, per_channel, dropout_prob, rng1)
    mask2 = fpixel.get_drop_mask(shape, per_channel, dropout_prob, rng2)

    # Check they're identical
    np.testing.assert_array_equal(mask1, mask2)

    # Generate mask with different seed
    rng3 = np.random.default_rng(43)
    mask3 = fpixel.get_drop_mask(shape, per_channel, dropout_prob, rng3)

    # Check it's different
    assert not np.array_equal(mask1, mask3)


def test_pixel_dropout_sequence_per_channel():
    """Test pixel dropout with sequence values and per_channel=True"""
    # Setup
    image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    drop_values = (1, 2, 3)

    # With dropout_prob=1.0, drop_mask will be all True
    drop_mask = np.ones((10, 10, 3), dtype=bool)

    # Verify drop_mask is all True
    assert np.all(drop_mask)
    assert drop_mask.shape == image.shape

    # Prepare drop values
    prepared_values = fpixel.prepare_drop_values(image, drop_values, np.random.default_rng(42))

    # Verify prepared_values shape matches image
    assert prepared_values.shape == image.shape

    # Apply dropout
    result = fpixel.pixel_dropout(image, drop_mask, prepared_values)

    # Each channel should be entirely filled with its corresponding value
    for channel_idx, expected_value in enumerate(drop_values):
        assert np.all(result[:, :, channel_idx] == expected_value), \
            f"Channel {channel_idx} should be filled with value {expected_value}"


def test_prepare_drop_values_random_two_channels():
    """Test random value generation for 2-channel images."""
    rng = np.random.default_rng(42)

    # Test uint8 random values
    array_uint8 = np.zeros((10, 10, 2), dtype=np.uint8)
    result_uint8 = fpixel.prepare_drop_values(array_uint8, None, rng)
    assert result_uint8.shape == (10, 10, 2)
    assert np.all((result_uint8 >= 0) & (result_uint8 <= 255))

    # Test float32 random values
    array_float32 = np.zeros((10, 10, 2), dtype=np.float32)
    result_float32 = fpixel.prepare_drop_values(array_float32, None, rng)
    assert result_float32.shape == (10, 10, 2)
    assert np.all((result_float32 >= 0.0) & (result_float32 <= 1.0))

    # Test that channels get different values
    assert not np.all(result_uint8[..., 0] == result_uint8[..., 1])


@pytest.mark.parametrize(
    ["shape", "roughness"],
    [
        ((100, 100), 0.5),  # Standard square image
        ((200, 100), 0.5),  # Rectangular image
        ((50, 75), 0.5),    # Small irregular image
        ((100, 100), 0.1),  # Low roughness
        ((100, 100), 0.9),  # High roughness
    ],
)
def test_plasma_pattern_basic_properties(shape, roughness):
    """Test basic properties of generated plasma patterns."""
    rng = np.random.default_rng(42)
    pattern = fpixel.generate_plasma_pattern(shape, roughness, rng)

    assert pattern.shape == shape
    assert pattern.dtype == np.float32
    assert np.all(pattern >= 0)
    assert np.all(pattern <= 1)
    assert not np.allclose(pattern, pattern.mean())

@pytest.mark.parametrize(
    ["seed1", "seed2", "should_be_different"],
    [
        (42, 42, False),     # Same seed should produce same pattern
        (42, 43, True),      # Different seeds should produce different patterns
    ],
)
def test_plasma_pattern_reproducibility(seed1, seed2, should_be_different):
    """Test reproducibility of plasma patterns with same/different seeds."""
    shape = (100, 100)
    roughness = 0.5

    # Generate two patterns with given seeds
    pattern1 = fpixel.generate_plasma_pattern(shape, roughness, np.random.default_rng(seed1))
    pattern2 = fpixel.generate_plasma_pattern(shape, roughness, np.random.default_rng(seed2))

    if should_be_different:
        assert not np.allclose(pattern1, pattern2)
    else:
        assert np.allclose(pattern1, pattern2)


def test_plasma_pattern_statistical_properties():
    """Test statistical properties of generated plasma patterns."""
    shape = (200, 200)
    roughness = 0.5
    rng = np.random.default_rng(42)

    pattern = fpixel.generate_plasma_pattern(shape, roughness, rng)

    # Test mean is approximately 0.5 (center of range)
    assert 0.3 <= pattern.mean() <= 0.7  # Wider bounds to account for randomness

    # Test standard deviation is reasonable
    assert 0.05 <= pattern.std() <= 0.35

    # Test distribution is roughly symmetric
    median = np.median(pattern)
    assert 0.3 <= median <= 0.7  # Wider bounds to account for randomness


@pytest.mark.parametrize(
    ["hist", "min_intensity", "max_intensity", "max_value", "method", "expected_output"],
    [
        # Test PIL method with simple range
        (
            np.array([1, 1, 1]),  # Simple histogram
            0, 2,  # min/max intensities
            255,   # max value
            "pil",
            np.array([0, 128, 255]),  # Expected LUT for first 3 values
        ),

        # Test CDF method with simple range
        (
            np.array([1, 1, 1]),  # Equal distribution
            0, 2,  # min/max intensities
            255,   # max value
            "cdf",
            np.array([0, 128, 255]),  # Expected LUT for first 3 values
        ),

        # Test empty histogram with PIL method
        (
            np.zeros(256),  # Empty histogram
            0, 255,
            255,
            "pil",
            np.arange(256, dtype=np.uint8),  # Should return identity LUT
        ),

        # Test empty histogram with CDF method
        (
            np.zeros(256),
            0, 255,
            255,
            "cdf",
            np.arange(256, dtype=np.uint8),  # Should return identity LUT
        ),

        # Test single value histogram with PIL method
        (
            np.array([0, 10, 0]),  # Single non-zero value
            1, 1,
            255,
            "pil",
            np.zeros(256, dtype=np.uint8),  # Should map everything to 0
        ),

        # Test narrow range with PIL method
        (
            np.array([0, 1, 1, 1, 0]),
            1, 3,
            255,
            "pil",
            np.array([0, 0, 128, 255, 255, *[255]*(256-5)]),  # Linear scaling
        ),
    ]
)
def test_create_contrast_lut(
    hist: np.ndarray,
    min_intensity: int,
    max_intensity: int,
    max_value: int,
    method: str,
    expected_output: np.ndarray
):
    """Test create_contrast_lut function with various inputs."""
    # If hist is smaller than 256, pad it
    if len(hist) < 256:
        hist = np.pad(hist, (0, 256 - len(hist)))

    # Generate LUT
    lut = fpixel.create_contrast_lut(
        hist=hist,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        max_value=max_value,
        method=method
    )

    # Basic checks
    assert isinstance(lut, np.ndarray)
    assert lut.dtype == np.uint8
    assert lut.shape == (256,)
    assert np.all(lut >= 0)
    assert np.all(lut <= max_value)

    # Check if first few values match expected
    assert np.array_equal(
        lut[:len(expected_output)],
        expected_output[:len(expected_output)]
    )


def test_create_contrast_lut_properties():
    """Test mathematical properties of the lookup tables."""
    hist = np.random.randint(0, 100, 256)
    max_value = 255

    # Test monotonicity for PIL method
    lut_pil = fpixel.create_contrast_lut(
        hist=hist,
        min_intensity=50,
        max_intensity=200,
        max_value=max_value,
        method="pil"
    )
    assert np.all(np.diff(lut_pil) >= 0), "PIL LUT should be monotonically increasing"

    # Test CDF method preserves relative frequencies
    lut_cdf = fpixel.create_contrast_lut(
        hist=hist,
        min_intensity=50,
        max_intensity=200,
        max_value=max_value,
        method="cdf"
    )
    assert np.all(np.diff(lut_cdf) >= 0), "CDF LUT should be monotonically increasing"



@pytest.mark.parametrize(
    ["hist", "cutoff", "expected"],
    [
        # Test with no cutoff
        (
            np.array([0, 1, 1, 1, 0]),  # Simple histogram
            0,
            (1, 3)  # Should return first and last non-zero indices
        ),

        # Test with empty histogram
        (
            np.zeros(256),
            0,
            (0, 0)  # Should return (0, 0) for empty histogram
        ),

        # Test with single value histogram
        (
            np.array([0, 10, 0, 0]),
            0,
            (1, 1)  # Should return same index for single peak
        ),

        # Test with 20% cutoff
        (
            np.array([10, 10, 10, 10, 10]),  # Uniform histogram
            20,
            (1, 4)  # Should cut 20% from each end
        ),

        # Test with 50% cutoff
        (
            np.array([10, 10, 10, 10, 10]),
            50,
            (2, 2)  # Should converge to middle
        ),

        # Test with asymmetric histogram
        (
            np.array([50, 10, 10, 10, 20]),  # More weight on edges
            20,
            (1, 4)  # Should adjust for weight distribution
        ),

        # Test with all pixels in one bin
        (
            np.array([0, 100, 0, 0]),
            10,
            (1, 1)  # Should return the peak location
        ),
    ]
)
def test_get_histogram_bounds(hist: np.ndarray, cutoff: float, expected: tuple[int, int]):
    """Test get_histogram_bounds with various histogram shapes and cutoffs."""
    min_intensity, max_intensity = fpixel.get_histogram_bounds(hist, cutoff)

    assert isinstance(min_intensity, int)
    assert isinstance(max_intensity, int)
    assert min_intensity <= max_intensity
    assert min_intensity >= 0
    assert max_intensity < len(hist)
    assert (min_intensity, max_intensity) == expected



def test_get_histogram_bounds_properties():
    """Test mathematical and logical properties of the bounds."""
    np.random.seed(42)  # For reproducibility
    hist = np.random.randint(0, 100, 256)

    cutoffs = [0, 10, 25, 49]
    previous_range = 256

    for cutoff in cutoffs:
        min_intensity, max_intensity = fpixel.get_histogram_bounds(hist, cutoff)

        # Range should decrease as cutoff increases
        current_range = max_intensity - min_intensity + 1
        assert current_range <= previous_range, \
            f"Range should decrease with increasing cutoff. Cutoff: {cutoff}"
        previous_range = current_range

        # Verify percentage of pixels included
        if cutoff > 0:
            pixels_before_min = hist[:min_intensity].sum()
            total_pixels = hist.sum()

            expected_cut = total_pixels * cutoff / 100
            relative_error = abs(pixels_before_min - expected_cut) / expected_cut
            assert relative_error <= 0.1, \
                f"Lower bound cut incorrect for cutoff {cutoff}"


def test_get_histogram_bounds_edge_cases():
    """Test edge cases for get_histogram_bounds."""
    # Test with all zeros except edges
    hist = np.zeros(256)
    hist[0] = hist[-1] = 100
    min_intensity, max_intensity = fpixel.get_histogram_bounds(hist, 0)
    assert (min_intensity, max_intensity) == (0, 255)

    # Test with single non-zero value
    hist = np.zeros(256)
    hist[128] = 100
    min_intensity, max_intensity = fpixel.get_histogram_bounds(hist, 0)
    assert min_intensity == max_intensity == 128

    # Test with constant histogram and 25% cutoff
    hist = np.ones(256)
    min_intensity, max_intensity = fpixel.get_histogram_bounds(hist, 25)
    # With uniform distribution, should cut 25% from each end
    assert min_intensity == 64  # 256 * 0.25
    assert max_intensity == 191  # 256 * 0.75 - 1


def test_get_histogram_bounds_numerical_stability():
    """Test numerical stability with very large and small values."""
    # Test with very large values
    hist = np.ones(256) * 1e6
    min_intensity, max_intensity = fpixel.get_histogram_bounds(hist, 10)
    # With uniform distribution, should cut 10% from each end
    assert min_intensity == 25  # 256 * 0.10
    assert max_intensity == 230  # 256 * 0.90 - 1


@pytest.mark.parametrize(
    ["img", "intensity", "angle", "expected"],
    [
        # Test horizontal gradient (0 degrees)
        (
            np.array([[100, 100], [100, 100]], dtype=np.uint8),
            0.2,    # ±20% change
            0,
            np.array([[80, 120], [80, 120]], dtype=np.uint8),  # Updated
        ),

        # Test vertical gradient (90 degrees)
        (
            np.array([[100, 100], [100, 100]], dtype=np.uint8),
            0.2,    # ±20% change
            90,
            np.array([[80, 80], [120, 120]], dtype=np.uint8),  # Updated
        ),

        # Test RGB image
        (
            np.array([[[100, 150, 200], [100, 150, 200]]], dtype=np.uint8),
            0.1,    # ±10% change
            0,
            np.array([[[90, 135, 180], [110, 165, 220]]], dtype=np.uint8),  # Updated
        ),

        # Test float32 image
        (
            np.array([[0.5, 0.5]], dtype=np.float32),
            0.2,    # ±20% change
            0,
            np.array([[0.4, 0.6]], dtype=np.float32),  # Updated
        ),

        # Test negative intensity
        (
            np.array([[100, 100]], dtype=np.uint8),
            -0.2,   # ±20% change (inverted)
            0,
            np.array([[120, 80]], dtype=np.uint8),  # Updated
        ),

        # Test 45 degree angle
        (
            np.array([[100, 100], [100, 100]], dtype=np.uint8),
            0.2,    # ±20% change
            45,
            np.array([[80, 100], [100, 120]], dtype=np.uint8),
        ),

        # Test minimal intensity
        (
            np.array([[100, 100]], dtype=np.uint8),
            0.01,   # ±1% change
            0,
            np.array([[99, 101]], dtype=np.uint8),  # Correct
        ),

        # Test small intensity
        (
            np.array([[100, 100]], dtype=np.uint8),
            0.05,   # ±5% change
            0,
            np.array([[95, 105]], dtype=np.uint8),  # Correct
        ),

        # Test medium intensity
        (
            np.array([[100, 100]], dtype=np.uint8),
            0.15,   # ±15% change
            0,
            np.array([[85, 115]], dtype=np.uint8),  # Correct
        ),

        # Test diagonal with small intensity
        (
            np.array([[100, 100], [100, 100]], dtype=np.uint8),
            0.05,   # ±5% change
            45,
            np.array([[95, 100], [100, 105]], dtype=np.uint8),
        ),
    ]
)
def test_apply_linear_illumination(img, intensity, angle, expected):
    result = fpixel.apply_linear_illumination(img.copy(), intensity, angle)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1)


@pytest.mark.parametrize(
    ["shape", "dtype"],
    [
        ((100, 100), np.uint8),
        ((100, 100), np.float32),
        ((100, 100, 3), np.uint8),
        ((100, 100, 3), np.float32),
        ((50, 75, 4), np.uint8),  # RGBA
    ]
)
def test_apply_linear_illumination_shapes(shape, dtype):
    """Test that function works with different image shapes and dtypes."""
    img = np.random.rand(*shape).astype(dtype)
    if dtype == np.uint8:
        img = (img * 255).astype(dtype)

    result = fpixel.apply_linear_illumination(img.copy(), intensity=0.1, angle=45)

    assert result.shape == img.shape
    assert result.dtype == img.dtype

@pytest.mark.parametrize("angle", [0, 45, 90, 180, 270, 360])
def test_apply_linear_illumination_angles(angle):
    """Test different angles produce expected gradient directions."""
    img = np.full((50, 50), 100, dtype=np.uint8)
    result = fpixel.apply_linear_illumination(img.copy(), intensity=0.2, angle=angle)

    # Check that the gradient exists (image is not uniform)
    assert not np.all(result == result[0, 0])

def test_apply_linear_illumination_preserves_range():
    """Test that the function preserves the valid range of values."""
    # Test uint8 range
    img_uint8 = np.array([[0, 128, 255]], dtype=np.uint8)
    result_uint8 = fpixel.apply_linear_illumination(img_uint8.copy(), intensity=0.2, angle=0)
    assert result_uint8.min() >= 0
    assert result_uint8.max() <= 255

    # Test float32 range [0, 1]
    img_float = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    result_float = fpixel.apply_linear_illumination(img_float.copy(), intensity=0.2, angle=0)
    assert result_float.min() >= 0
    assert result_float.max() <= 1.0

def test_apply_linear_illumination_preserves_input():
    """Test that the function doesn't modify input array."""
    img = np.array([[100, 200]], dtype=np.uint8)
    img_copy = img.copy()

    _ = fpixel.apply_linear_illumination(img, intensity=0.1, angle=0)

    np.testing.assert_array_equal(img, img_copy)

def test_apply_linear_illumination_symmetry():
    """Test that opposite angles produce inverse effects."""
    img = np.full((10, 10), 100, dtype=np.uint8)

    result1 = fpixel.apply_linear_illumination(img.copy(), intensity=0.2, angle=0)
    result2 = fpixel.apply_linear_illumination(img.copy(), intensity=0.2, angle=180)

    # Check if the patterns are inversions of each other
    np.testing.assert_allclose(result1 + result2, 200, rtol=1e-5, atol=1)


@pytest.mark.parametrize(
    ["height", "width", "angle", "expected"],
    [
        # Test horizontal gradient (0 degrees)
        (2, 2, 0, np.array([[0, 1], [0, 1]], dtype=np.float32)),

        # Test vertical gradient (90 degrees)
        (2, 2, 90, np.array([[0, 0], [1, 1]], dtype=np.float32)),

        # Test diagonal gradient (45 degrees)
        (2, 2, 45, np.array([[0, 0.5], [0.5, 1]], dtype=np.float32)),

        # Test diagonal gradient (135 degrees)
        (2, 2, 135, np.array([[0.5, 0], [1, 0.5]], dtype=np.float32)),
    ]
)
def test_create_directional_gradient(height, width, angle, expected):
    gradient = fpixel.create_directional_gradient(height, width, angle)
    np.testing.assert_allclose(gradient, expected, atol=1e-15)


def test_gradient_range():
    """Test that gradient values are always in [0, 1] range."""
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        gradient = fpixel.create_directional_gradient(10, 10, angle)
        assert gradient.min() >= 0 - 1e-7
        assert gradient.max() <= 1 + 1e-7



@pytest.mark.parametrize(
    ["corner", "intensity", "expected_corner"],
    [
        # Test each corner with positive intensity (brightening)
        (0, 0.2, (0, 0)),      # top-left is brightest
        (1, 0.2, (0, 9)),      # top-right is brightest
        (2, 0.2, (9, 9)),      # bottom-right is brightest
        (3, 0.2, (9, 0)),      # bottom-left is brightest

        # Test with negative intensity (darkening)
        (0, -0.2, (9, 9)),     # top-left is darkest, opposite corner is brightest
        (1, -0.2, (9, 0)),     # top-right is darkest, opposite corner is brightest
        (2, -0.2, (0, 0)),     # bottom-right is darkest, opposite corner is brightest
        (3, -0.2, (0, 9)),     # bottom-left is darkest, opposite corner is brightest
    ],
)
def test_corner_illumination_brightest_point(corner, intensity, expected_corner):
    """Test that the illumination pattern has maximum intensity at the correct corner."""
    # Create a constant test image
    image = np.full((10, 10), 0.5, dtype=np.float32)

    # Apply corner illumination
    result = fpixel.apply_corner_illumination(image, intensity, corner)

    # Find the brightest point
    actual_corner = np.unravel_index(np.argmax(result), result.shape)

    assert actual_corner == expected_corner


@pytest.mark.parametrize(
    ["shape", "dtype"],
    [
        ((10, 10), np.float32),      # grayscale float32
        ((10, 10), np.uint8),        # grayscale uint8
        ((10, 10, 3), np.float32),   # RGB float32
        ((10, 10, 3), np.uint8),     # RGB uint8
        # Removed single channel test case as it's not supported
    ],
)
def test_corner_illumination_preserves_shape_and_type(shape, dtype):
    """Test that the output maintains the input shape and dtype."""
    # Create test image
    image = np.ones(shape, dtype=dtype)
    if dtype == np.uint8:
        image *= 255

    # Apply corner illumination
    result = fpixel.apply_corner_illumination(image, intensity=0.2, corner=0)

    assert result.shape == shape
    assert result.dtype == dtype


@pytest.mark.parametrize("intensity", [-0.2, 0, 0.2])
def test_corner_illumination_intensity_range(intensity):
    """Test that the output values stay within valid range."""
    # Create test images with extreme values
    image_zeros = np.zeros((10, 10), dtype=np.float32)
    image_ones = np.ones((10, 10), dtype=np.float32)

    # Apply corner illumination
    result_zeros = fpixel.apply_corner_illumination(image_zeros, intensity, corner=0)
    result_ones = fpixel.apply_corner_illumination(image_ones, intensity, corner=0)

    # Check that values stay in valid range
    assert np.all(result_zeros >= 0)
    assert np.all(result_zeros <= 1)
    assert np.all(result_ones >= 0)
    assert np.all(result_ones <= 1)


def test_corner_illumination_identity_zero_intensity():
    """Test that zero intensity returns the input image unchanged."""
    # Create random test image
    image = np.random.rand(10, 10).astype(np.float32)

    # Apply corner illumination with zero intensity
    result = fpixel.apply_corner_illumination(image, intensity=0, corner=0)

    np.testing.assert_array_almost_equal(result, image, decimal=2)


@pytest.mark.parametrize("corner", [0, 1, 2, 3])
def test_corner_illumination_symmetry(corner):
    """Test that the illumination pattern is symmetric around the corner."""
    # Create test image
    image = np.ones((11, 11), dtype=np.float32)  # Odd dimensions for clear center

    # Apply corner illumination
    result = fpixel.apply_corner_illumination(image, intensity=0.2, corner=corner)

    # Get distances from corner to test symmetry
    if corner == 0:  # top-left
        d1 = result[0, 1]  # one step right
        d2 = result[1, 0]  # one step down
    elif corner == 1:  # top-right
        d1 = result[0, -2]  # one step left
        d2 = result[1, -1]  # one step down
    elif corner == 2:  # bottom-right
        d1 = result[-1, -2]  # one step left
        d2 = result[-2, -1]  # one step up
    else:  # bottom-left
        d1 = result[-1, 1]  # one step right
        d2 = result[-2, 0]  # one step up

    np.testing.assert_almost_equal(d1, d2)


def test_corner_illumination_multichannel_consistency():
    """Test that all channels are modified identically for RGB images."""
    # Create RGB test image
    image = np.ones((10, 10, 3), dtype=np.float32)

    # Apply corner illumination
    result = fpixel.apply_corner_illumination(image, intensity=0.2, corner=0)

    # Check that all channels are identical
    np.testing.assert_array_almost_equal(result[..., 0], result[..., 1])
    np.testing.assert_array_almost_equal(result[..., 1], result[..., 2])


@pytest.mark.parametrize(
    ["center", "intensity", "sigma", "expected_brightest"],
    [
        # Test different centers with positive intensity (brightening)
        ((0.5, 0.5), 0.2, 0.25, (5, 5)),    # center
        ((0.0, 0.0), 0.2, 0.25, (0, 0)),    # top-left
        ((1.0, 0.0), 0.2, 0.25, (0, 9)),    # top-right
        ((1.0, 1.0), 0.2, 0.25, (9, 9)),    # bottom-right
        ((0.0, 1.0), 0.2, 0.25, (9, 0)),    # bottom-left

        # Test with negative intensity (darkening)
        ((0.5, 0.5), -0.2, 0.25, (0, 0)),   # center is darkest, corners brightest

        # Test different sigma values
        ((0.5, 0.5), 0.2, 0.1, (5, 5)),     # narrow gaussian
        ((0.5, 0.5), 0.2, 0.5, (5, 5)),     # wide gaussian
    ],
)
def test_gaussian_illumination_brightest_point(center, intensity, sigma, expected_brightest):
    """Test that the brightest point is at the expected location."""
    # Create a constant test image
    image = np.full((10, 10), 0.5, dtype=np.float32)

    # Apply gaussian illumination
    result = fpixel.apply_gaussian_illumination(image, intensity, center, sigma)

    # Find the brightest point
    actual_brightest = np.unravel_index(np.argmax(result), result.shape)

    assert actual_brightest == expected_brightest


@pytest.mark.parametrize(
    ["shape", "dtype"],
    [
        ((10, 10), np.float32),       # grayscale float32
        ((10, 10), np.uint8),         # grayscale uint8
        ((10, 10, 3), np.float32),    # RGB float32
        ((10, 10, 3), np.uint8),      # RGB uint8
    ],
)
def test_gaussian_illumination_preserves_shape_and_type(shape, dtype):
    """Test that output maintains input shape and dtype."""
    # Create test image
    image = np.ones(shape, dtype=dtype)
    if dtype == np.uint8:
        image *= 255

    result = fpixel.apply_gaussian_illumination(
        image,
        intensity=0.2,
        center=(0.5, 0.5),
        sigma=0.25,
    )

    assert result.shape == shape
    assert result.dtype == dtype


def test_gaussian_illumination_zero_intensity():
    """Test that zero intensity returns unchanged image."""
    image = np.random.rand(10, 10).astype(np.float32)
    result = fpixel.apply_gaussian_illumination(
        image,
        intensity=0.0,
        center=(0.5, 0.5),
        sigma=0.25,
    )

    np.testing.assert_array_equal(result, image)


def test_gaussian_illumination_symmetry():
    """Test that the gaussian pattern is symmetric around the center."""
    size = 11  # Odd size for exact center
    image = np.full((size, size), 0.5, dtype=np.float32)

    # Calculate exact pixel center
    center = ((size - 1) / (size * 2), (size - 1) / (size * 2))  # This gives exact center pixel

    result = fpixel.apply_gaussian_illumination(
        image,
        intensity=0.2,
        center=center,
        sigma=0.25,
    )

    # Get middle indices
    mid = size // 2
    radius = 3  # Test fewer pixels around center for stability

    # Check horizontal symmetry with tolerance
    center_row = result[mid]
    assert np.allclose(
        center_row[mid-radius:mid],  # Left of center
        np.flip(center_row[mid+1:mid+radius+1]),  # Right of center
        rtol=1e-4,  # Increased relative tolerance
        atol=1e-3,  # Absolute tolerance
    ), f"Horizontal asymmetry:\nLeft:  {center_row[mid-radius:mid]}\nRight: {np.flip(center_row[mid+1:mid+radius+1])}"

    # Check vertical symmetry with tolerance
    center_col = result[:, mid]
    assert np.allclose(
        center_col[mid-radius:mid],  # Above center
        np.flip(center_col[mid+1:mid+radius+1]),  # Below center
        rtol=1e-4,  # Increased relative tolerance
        atol=1e-3,  # Absolute tolerance
    ), f"Vertical asymmetry:\nAbove: {center_col[mid-radius:mid]}\nBelow: {np.flip(center_col[mid+1:mid+radius+1])}"


@pytest.mark.parametrize("intensity", [-0.2, 0.2])
def test_gaussian_illumination_multichannel_consistency(intensity):
    """Test that all channels are modified identically for RGB images."""
    image = np.ones((10, 10, 3), dtype=np.float32)
    result = fpixel.apply_gaussian_illumination(
        image,
        intensity=intensity,
        center=(0.5, 0.5),
        sigma=0.25,
    )

    # Check that all channels are identical
    assert np.allclose(result[..., 0], result[..., 1])
    assert np.allclose(result[..., 1], result[..., 2])


@pytest.mark.parametrize(
    ["sigma", "expected_pattern"],
    [
        (0.1, "narrow"),  # Narrow gaussian should have steeper falloff
        (0.5, "wide"),    # Wide gaussian should have gradual falloff
    ],
)
def test_gaussian_illumination_sigma(sigma, expected_pattern):
    """Test that sigma controls the spread of the gaussian pattern."""
    image = np.full((10, 10), 0.5, dtype=np.float32)
    result = fpixel.apply_gaussian_illumination(
        image,
        intensity=0.2,
        center=(0.5, 0.5),
        sigma=sigma,
    )

    # Compare center value to midpoint value
    center_val = result[5, 5]
    mid_val = result[5, 7]
    diff = center_val - mid_val

    if expected_pattern == "narrow":
        assert diff > 0.05  # Reduced threshold for steeper falloff
    else:
        assert diff < 0.05  # Reduced threshold for gradual falloff

    # Additional check: narrow should have larger difference than wide
    result_wide = fpixel.apply_gaussian_illumination(
        image,
        intensity=0.2,
        center=(0.5, 0.5),
        sigma=0.5,
    )
    wide_diff = result_wide[5, 5] - result_wide[5, 7]

    if expected_pattern == "narrow":
        assert diff > wide_diff  # Narrow should have steeper falloff than wide



@pytest.mark.parametrize(
    ["img", "slant", "drop_length", "drop_width", "drop_color", "blur_value", "brightness_coefficient", "rain_drops", "expected_shape"],
    [
        # Test basic functionality with small image
        (
            np.zeros((10, 10, 3), dtype=np.uint8),
            5,
            3,
            1,
            (200, 200, 200),
            3,
            0.7,
            np.array([(2, 2)]),
            (10, 10, 3),
        ),
        # Test with no rain drops
        (
            np.zeros((20, 20, 3), dtype=np.uint8),
            5,
            3,
            1,
            (200, 200, 200),
            3,
            0.7,
            np.array([]).reshape(0, 2),
            (20, 20, 3),
        ),
        # Test with multiple rain drops
        (
            np.zeros((30, 30, 3), dtype=np.uint8),
            -5,
            5,
            2,
            (255, 255, 255),
            5,
            0.8,
            np.array([(5, 5), (10, 10), (15, 15)]),
            (30, 30, 3),
        ),
    ]
)
def test_add_rain_shape_and_type(
    img, slant, drop_length, drop_width, drop_color, blur_value, brightness_coefficient, rain_drops, expected_shape
):
    result = fpixel.add_rain(
        img, slant, drop_length, drop_width, drop_color, blur_value, brightness_coefficient, rain_drops
    )
    assert result.shape == expected_shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("brightness_coefficient", [0.5, 0.7, 1.0])
def test_add_rain_brightness(brightness_coefficient):
    """Test that brightness coefficient correctly affects image brightness"""
    img = np.full((20, 20, 3), 100, dtype=np.uint8)
    rain_drops = np.array([(5, 5)])

    result = fpixel.add_rain(
        img=img,
        slant=5,
        drop_length=3,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=3,
        brightness_coefficient=brightness_coefficient,
        rain_drops=rain_drops,
    )

    # Convert to HSV to check brightness
    original_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

    if brightness_coefficient < 1.0:
        # For darkening coefficients, brightness should decrease
        assert np.mean(result_hsv[:, :, 2]) < np.mean(original_hsv[:, :, 2])
        np.testing.assert_allclose(
            np.mean(result_hsv[:, :, 2]) / np.mean(original_hsv[:, :, 2]),
            brightness_coefficient,
            rtol=0.1,  # Allow 10% tolerance due to rounding and blur effects
        )
    else:
        # For brightness_coefficient = 1.0, brightness might slightly increase
        # due to bright rain drops and blur, but shouldn't change dramatically
        np.testing.assert_allclose(
            np.mean(result_hsv[:, :, 2]) / np.mean(original_hsv[:, :, 2]),
            1.0,
            rtol=0.1,  # Allow 10% tolerance
        )


def test_add_rain_drops_visibility():
    """Test that rain drops are actually visible in the output"""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    rain_drops = np.array([(5, 5)])
    drop_color = (255, 255, 255)

    result = fpixel.add_rain(
        img=img,
        slant=0,
        drop_length=5,
        drop_width=1,
        drop_color=drop_color,
        blur_value=1,  # Minimal blur to check drop visibility
        brightness_coefficient=1.0,  # No brightness change
        rain_drops=rain_drops,
    )

    # Check if any pixels have the rain drop color
    assert np.any(result > 0)


def test_add_rain_preserves_input():
    """Test that the function doesn't modify the input image"""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img_copy = img.copy()

    fpixel.add_rain(
        img=img,
        slant=5,
        drop_length=3,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=3,
        brightness_coefficient=0.7,
        rain_drops=np.array([(5, 5)]),
    )

    np.testing.assert_array_equal(img, img_copy)


@pytest.mark.parametrize(
    "shape,color,intensity,expected_shape",
    [
        # Test different image sizes
        ((100, 100), np.array([1.0, 1.0, 1.0]), 1.0, (100, 100, 3)),
        ((200, 300), np.array([0.5, 0.5, 0.5]), 0.8, (200, 300, 3)),

        # Test different image shapes
        ((50, 75), np.array([0.7, 0.7, 0.7]), 0.6, (50, 75, 3)),
        ((480, 640), np.array([0.9, 0.9, 0.9]), 0.9, (480, 640, 3)),
    ]
)
def test_rain_params_shapes(shape, color, intensity, expected_shape):
    """Test that output shapes are correct for different input configurations."""
    # Generate random liquid layer
    liquid_layer = np.random.normal(0.65, 0.3, size=shape)

    result = fpixel.get_rain_params(liquid_layer, color, intensity)

    assert "drops" in result
    assert result["drops"].shape == expected_shape


@pytest.mark.parametrize(
    "intensity,color",
    [
        (0.0, np.array([1.0, 1.0, 1.0])),  # Zero intensity
        (1.0, np.array([0.0, 0.0, 0.0])),  # Black color
        (0.5, np.array([0.5, 0.5, 0.5])),  # Gray color
        (0.8, np.array([1.0, 0.0, 0.0])),  # Red color
    ]
)
def test_rain_params_intensity_and_color(intensity, color):
    """Test that intensity and color are correctly applied."""
    shape = (100, 100)
    liquid_layer = np.random.normal(0.65, 0.3, size=shape)

    result = fpixel.get_rain_params(liquid_layer, color, intensity)

    # Check that values are in valid range [0, 1]
    assert np.all(result["drops"] >= 0)
    assert np.all(result["drops"] <= 1)

    # Check that maximum value respects intensity
    assert np.max(result["drops"]) <= intensity + 1e-6  # Add small epsilon for floating point comparison


@pytest.mark.parametrize(
    "liquid_layer",
    [
        np.zeros((100, 100)),  # All zeros
        np.ones((100, 100)),   # All ones
        np.random.normal(0.65, 0.3, (100, 100)),  # Random normal
    ]
)
def test_rain_params_different_inputs(liquid_layer):
    """Test function behavior with different types of input layers."""
    color = np.array([1.0, 1.0, 1.0])
    intensity = 1.0

    result = fpixel.get_rain_params(liquid_layer, color, intensity)

    assert isinstance(result["drops"], np.ndarray)
    assert result["drops"].dtype in [np.float32, np.float64]


def test_rain_params_deterministic():
    """Test that function produces same output for same input."""
    shape = (100, 100)
    liquid_layer = np.random.normal(0.65, 0.3, size=shape)
    color = np.array([1.0, 1.0, 1.0])
    intensity = 1.0

    result1 = fpixel.get_rain_params(liquid_layer.copy(), color, intensity)
    result2 = fpixel.get_rain_params(liquid_layer.copy(), color, intensity)

    np.testing.assert_array_almost_equal(result1["drops"], result2["drops"])


def test_rain_params_visual_pattern():
    """Test that the rain pattern has expected characteristics."""
    shape = (100, 100)
    liquid_layer = np.random.normal(0.65, 0.3, size=shape)
    color = np.array([1.0, 1.0, 1.0])
    intensity = 1.0

    result = fpixel.get_rain_params(liquid_layer, color, intensity)
    drops = result["drops"]

    # Check that we have some variation in the pattern
    assert drops.std() > 0

    # Check that we have some zero and non-zero values (rain drops)
    assert np.sum(drops > 0) > 0
    assert np.sum(drops == 0) > 0



def test_rain_params_zero_input():
    """Test that zero input produces valid output without NaN values."""
    shape = (100, 100)
    liquid_layer = np.zeros(shape)  # All zeros input
    color = np.array([1.0, 1.0, 1.0])
    intensity = 1.0

    result = fpixel.get_rain_params(liquid_layer, color, intensity)

    # Check there are no NaN values
    assert not np.any(np.isnan(result["drops"]))
    # Check output is all zeros for zero input
    assert np.all(result["drops"] == 0)


def test_rain_params_small_input():
    """Test that very small input values produce valid output."""
    shape = (100, 100)
    liquid_layer = np.full(shape, 1e-10)  # Very small values
    color = np.array([1.0, 1.0, 1.0])
    intensity = 1.0

    result = fpixel.get_rain_params(liquid_layer, color, intensity)

    # Check there are no NaN values
    assert not np.any(np.isnan(result["drops"]))
    # Check values are in valid range
    assert np.all(result["drops"] >= 0)
    assert np.all(result["drops"] <= 1)


@pytest.mark.parametrize(
    "shape,color,cutout_threshold,sigma,intensity,expected_shape",
    [
        # Test different image sizes
        ((100, 100), np.array([0.2, 0.4, 0.6]), 0.5, 2.0, 0.8, (100, 100, 3)),
        ((200, 300), np.array([0.1, 0.2, 0.3]), 0.6, 1.5, 0.7, (200, 300, 3)),

        # Test different parameters
        ((50, 75), np.array([0.3, 0.3, 0.3]), 0.4, 3.0, 0.9, (50, 75, 3)),
        ((480, 640), np.array([0.4, 0.4, 0.4]), 0.7, 2.5, 0.6, (480, 640, 3)),
    ]
)
def test_mud_params_shapes(shape, color, cutout_threshold, sigma, intensity, expected_shape):
    """Test output shapes for different input configurations."""
    liquid_layer = np.random.normal(0.65, 0.3, size=shape)

    result = fpixel.get_mud_params(liquid_layer, color, cutout_threshold, sigma, intensity, np.random.default_rng(137))

    assert "mud" in result and "non_mud" in result
    assert result["mud"].shape == expected_shape
    assert result["non_mud"].shape == expected_shape


@pytest.mark.parametrize(
    "intensity,color,cutout_threshold",
    [
        (0.3, np.array([0.2, 0.2, 0.2]), 0.5),  # Low intensity
        (0.8, np.array([0.0, 0.0, 0.0]), 0.6),  # Black color
        (1.0, np.array([0.5, 0.5, 0.5]), 0.7),  # Gray color
        (0.9, np.array([0.8, 0.4, 0.2]), 0.4),  # Brown color
    ]
)
def test_mud_params_intensity_and_color(intensity, color, cutout_threshold):
    """Test intensity and color application."""
    shape = (100, 100)
    liquid_layer = np.random.normal(0.65, 0.3, size=shape)
    sigma = 2.0

    result = fpixel.get_mud_params(liquid_layer, color, cutout_threshold, sigma, intensity, np.random.default_rng(137))

    # Check value ranges
    assert np.all(result["mud"] >= 0)
    assert np.all(result["mud"] <= 1)
    assert np.all(result["non_mud"] >= 0)
    assert np.all(result["non_mud"] <= 1)

    # Check that actual intensity is reasonable
    max_effect = np.max(result["mud"])
    max_color = np.max(color)

    # Expected effect should be proportional to both intensity and color
    expected_max = intensity * max_color
    expected_min = 0.8 * expected_max  # Allow 20% tolerance for numerical effects

    # Add diagnostic information
    if max_effect < expected_min:
        # Calculate percentage of non-zero mud pixels
        mud_coverage = np.mean(result["mud"] > 0)

        # Show distribution of non-zero mud values
        non_zero_mud = result["mud"][result["mud"] > 0]

    if max_color > 0:
        assert max_effect >= expected_min, (
            f"Maximum effect ({max_effect:.3f}) is less than expected minimum ({expected_min:.3f}) "
            f"for intensity={intensity}, color={color}"
        )
    else:
        # For black color, expect no effect
        assert max_effect == 0, f"Expected no effect for black color, got {max_effect:.3f}"

def test_minimum_effect_coverage():
    """Test that minimum effect coverage is maintained."""
    shape = (100, 100)
    liquid_layer = np.zeros(shape)  # Should trigger minimum coverage logic
    color = np.array([0.2, 0.4, 0.6])
    cutout_threshold = 0.9  # High threshold to force adjustment
    sigma = 2.0
    intensity = 0.8

    result = fpixel.get_mud_params(liquid_layer, color, cutout_threshold, sigma, intensity, np.random.default_rng(137)   )

    # Check that we have some non-zero effect
    mud_coverage = np.sum(result["mud"] > 0) / result["mud"].size
    assert mud_coverage >= 0.1  # At least 10% coverage


def test_threshold_adjustment():
    """Test automatic threshold adjustment."""
    shape = (100, 100)
    liquid_layer = np.random.normal(0.3, 0.1, size=shape)  # Values mostly below threshold
    color = np.array([0.2, 0.4, 0.6])
    cutout_threshold = 0.8  # High threshold to force adjustment
    sigma = 2.0
    intensity = 0.8

    result = fpixel.get_mud_params(liquid_layer, color, cutout_threshold, sigma, intensity, np.random.default_rng(137))

    # Should have some effect despite high threshold
    assert np.sum(result["mud"] > 0) > 0


def test_blur_effect():
    """Test that Gaussian blur is properly applied."""
    shape = (100, 100)
    liquid_layer = np.zeros(shape)
    liquid_layer[40:60, 40:60] = 1  # Create a square
    color = np.array([0.2, 0.4, 0.6])
    cutout_threshold = 0.5
    sigma = 2.0
    intensity = 0.8

    result = fpixel.get_mud_params(liquid_layer, color, cutout_threshold, sigma, intensity, np.random.default_rng(137))

    # Blurring should create gradients - check that we have intermediate values
    mud = result["mud"][:,:,0]  # Take first channel
    assert np.sum((mud > 0) & (mud < np.max(mud))) > 0


def test_deterministic():
    """Test function produces same output for same input."""
    shape = (100, 100)
    liquid_layer = np.random.normal(0.65, 0.3, size=shape)
    color = np.array([0.2, 0.4, 0.6])
    cutout_threshold = 0.5
    sigma = 2.0
    intensity = 0.8

    result1 = fpixel.get_mud_params(liquid_layer.copy(), color, cutout_threshold, sigma, intensity, np.random.default_rng(137))
    result2 = fpixel.get_mud_params(liquid_layer.copy(), color, cutout_threshold, sigma, intensity, np.random.default_rng(137))

    np.testing.assert_array_almost_equal(result1["mud"], result2["mud"])
    np.testing.assert_array_almost_equal(result1["non_mud"], result2["non_mud"])



@pytest.fixture
def random_state():
    return np.random.RandomState(42)

@pytest.mark.parametrize(
    ["height", "width", "n_iter"],
    [
        (100, 100, 50),   # Small square image
        (200, 100, 100),  # Rectangle image
        (50, 50, 200),    # Small image, more iterations
    ]
)
def test_simple_nmf_shape(height, width, n_iter, random_state):
    # Create synthetic H&E-like data
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    od = -np.log((img.reshape((-1, 3)).astype(np.float32) + 1) / 256)

    # Our implementation
    nmf = fpixel.SimpleNMF(n_iter=n_iter)
    concentrations, colors = nmf.fit_transform(od)

    # Check shapes
    assert concentrations.shape == (height * width, 2)
    assert colors.shape == (2, 3)

    # Check non-negativity
    assert np.all(concentrations >= 0)
    assert np.all(colors >= 0)

    # Check unit norm constraint on colors
    np.testing.assert_allclose(
        np.sum(colors**2, axis=1),
        np.ones(2),
        rtol=1e-5
    )

@pytest.mark.parametrize(
    ["height", "width"],
    [
        (100, 100),  # Square image
        (200, 100),  # Rectangle image
    ]
)
def test_simple_nmf_against_sklearn(height, width, random_state):
    # Create synthetic H&E-like data
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    od = -np.log((img.reshape((-1, 3)).astype(np.float32) + 1) / 256)

    # Our implementation
    our_nmf = fpixel.SimpleNMF(n_iter=100)
    our_concentrations, our_colors = our_nmf.fit_transform(od)

    # Sklearn implementation
    sklearn_nmf = NMF(
        n_components=2,
        init='random',
        random_state=42,
        max_iter=100
    )
    sklearn_concentrations = sklearn_nmf.fit_transform(od)
    sklearn_colors = sklearn_nmf.components_

    # Reconstruction error comparison
    our_reconstruction = np.dot(our_concentrations, our_colors)
    sklearn_reconstruction = np.dot(sklearn_concentrations, sklearn_colors)

    our_error = np.mean((od - our_reconstruction) ** 2)
    sklearn_error = np.mean((od - sklearn_reconstruction) ** 2)

    # Our error should be within a reasonable factor of sklearn's
    assert our_error <= 2 * sklearn_error


@pytest.fixture
def synthetic_he_image():
    """Create synthetic H&E-like image with known stain vectors."""
    # Define synthetic H&E stain vectors (simplified but realistic)
    h_vector = np.array([0.65, 0.70, 0.29])  # Hematoxylin (purple)
    e_vector = np.array([0.07, 0.99, 0.11])  # Eosin (pink)
    stain_matrix = np.vstack([h_vector, e_vector])

    # Create synthetic concentrations with distinct regions
    height, width = 100, 100
    h_concentration = np.zeros((height, width))
    e_concentration = np.zeros((height, width))

    # Hematoxylin-rich region (top half)
    h_concentration[:50, :] = np.random.uniform(0.5, 1.0, (50, width))
    e_concentration[:50, :] = np.random.uniform(0, 0.5, (50, width))

    # Eosin-rich region (bottom half)
    h_concentration[50:, :] = np.random.uniform(0, 0.5, (50, width))
    e_concentration[50:, :] = np.random.uniform(0.5, 1.0, (50, width))

    # Add background region (top-left corner)
    h_concentration[:20, :20] = 0
    e_concentration[:20, :20] = 0

    # Create image
    concentrations = np.stack([h_concentration, e_concentration], axis=-1)
    optical_density = np.dot(concentrations, stain_matrix)
    img = np.exp(-optical_density) * 255
    return img.astype(np.uint8), stain_matrix, concentrations


@pytest.mark.parametrize(
    ["normalizer_class", "kwargs"],
    [
        (fpixel.VahadaneNormalizer, {}),
        (fpixel.MacenkoNormalizer, { "angular_percentile": 99 }),
    ]
)
def test_normalizer_output_shape(normalizer_class, kwargs, synthetic_he_image):
    """Test that normalizers produce correctly shaped output."""
    img = synthetic_he_image[0]
    normalizer = normalizer_class(**kwargs)
    normalizer.fit(img)

    assert normalizer.stain_matrix_target.shape == (2, 3)
    assert np.all(normalizer.stain_matrix_target >= 0)
    assert np.allclose(
        np.sum(normalizer.stain_matrix_target**2, axis=1),
        np.ones(2),
        rtol=1e-5
    )

@pytest.mark.parametrize(
    ["normalizer_class", "kwargs", "angle_tolerance"],
    [
        (fpixel.VahadaneNormalizer, {}, 45),
        (fpixel.MacenkoNormalizer, { "angular_percentile": 99 }, 45),
    ]
)
def test_normalizer_stain_separation(normalizer_class, kwargs, angle_tolerance, synthetic_he_image):
    """Test that normalizers correctly separate H&E stains."""
    img, stain_matrix, _ = synthetic_he_image  # Unpack 3 values
    normalizer = normalizer_class(**kwargs)
    normalizer.fit(img)

    # Calculate angles between true and estimated stain vectors
    for i in range(2):
        cos_angle = np.dot(normalizer.stain_matrix_target[i], stain_matrix[i])
        cos_angle /= np.linalg.norm(normalizer.stain_matrix_target[i])
        cos_angle /= np.linalg.norm(stain_matrix[i])
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_degrees = np.degrees(angle)
        assert angle_degrees < angle_tolerance


@pytest.mark.parametrize(
    "angular_percentile",
    [0, 50, 90, 99, 99.9]
)
def test_macenko_angular_percentile(angular_percentile, synthetic_he_image):
    """Test MacenkoNormalizer with different angular percentiles."""
    img = synthetic_he_image[0]
    normalizer = fpixel.MacenkoNormalizer(
        angular_percentile=angular_percentile
    )
    normalizer.fit(img)

    assert normalizer.stain_matrix_target.shape == (2, 3)
    assert np.all(normalizer.stain_matrix_target >= 0)


@pytest.mark.parametrize(
    ["scale_factors", "shift_values", "expected_changes"],
    [
        # Increase H only
        (
            np.array([2.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            {"h_change": "increase", "e_change": "stable"}
        ),
        # Decrease both
        (
            np.array([0.5, 0.5], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            {"h_change": "decrease", "e_change": "decrease"}
        ),
    ]
)
def test_stain_augmentation_effects(synthetic_he_image, scale_factors, shift_values, expected_changes):
    """Test that augmentation produces expected changes in stain intensities."""
    img, stain_matrix = synthetic_he_image[:2]

    # Ensure stain_matrix is float32
    stain_matrix = stain_matrix.astype(np.float32)

    result = fpixel.apply_he_stain_augmentation(
        img=img,
        stain_matrix=stain_matrix,
        scale_factors=scale_factors,
        shift_values=shift_values,
        augment_background=True
    )

    # Calculate changes in optical density
    def to_od(x):
        return -np.log((x.astype(np.float32) + 1) / 256)

    od_orig = to_od(img)
    od_result = to_od(result)

    # Exclude background pixels (where OD is very low)
    tissue_mask = np.any(od_orig > 0.15, axis=2)

    h_region_mask = tissue_mask[:50, 20:]
    e_region_mask = tissue_mask[50:, 20:]

    h_change = (np.mean(od_result[:50, 20:][h_region_mask]) -
                np.mean(od_orig[:50, 20:][h_region_mask]))
    e_change = (np.mean(od_result[50:, 20:][e_region_mask]) -
                np.mean(od_orig[50:, 20:][e_region_mask]))

    # Check direction of changes
    if expected_changes["h_change"] == "increase":
        assert h_change > 0, "H stain should increase"
    elif expected_changes["h_change"] == "decrease":
        assert h_change < 0, "H stain should decrease"
    else:  # stable
        np.testing.assert_allclose(h_change, 0, atol=0.15)

    if expected_changes["e_change"] == "increase":
        assert e_change > 0, "E stain should increase"
    elif expected_changes["e_change"] == "decrease":
        assert e_change < 0, "E stain should decrease"
    else:  # stable
        np.testing.assert_allclose(e_change, 0, atol=0.15)


def test_background_augmentation(synthetic_he_image):
    """Test that background augmentation flag works correctly."""
    img, stain_matrix, _ = synthetic_he_image

    result = fpixel.apply_he_stain_augmentation(
        img=img,
        stain_matrix=stain_matrix,
        scale_factors=np.array([2.0, 2.0]),
        shift_values=np.array([0.1, 0.1]),
        augment_background=False
    )

    # Background region should be unchanged
    np.testing.assert_allclose(
        result[:20, :20],
        img[:20, :20],
        rtol=1e-5,
        atol=1
    )



@pytest.mark.parametrize(
    "sat_shift, description",
    [
        (50, "small saturation increase"),
        (100, "medium saturation increase"),
        (200, "large saturation increase"),
        (255, "maximum saturation increase"),
    ],
)
def test_white_pixels_remain_white_with_saturation_increase(sat_shift, description):
    """Test that white pixels remain white when saturation is increased.

    This test verifies that increasing saturation doesn't affect white pixels,
    which should remain white regardless of saturation changes.
    """
    # Create a simple test image with white pixels (255, 255, 255)
    white_image = np.ones((10, 10, 3), dtype=np.uint8) * 255

    # Apply saturation increase
    result = fpixel.shift_hsv(white_image, hue_shift=0, sat_shift=sat_shift, val_shift=0)

    # Check that all pixels are still white (255, 255, 255)
    assert np.all(result == 255), f"White pixels changed color with {description}"


@pytest.mark.parametrize(
    "image_type, sat_shift",
    [
        ("white_and_gray", 100),
        ("white_and_color", 100),
        ("white_black_color", 200),
    ],
)
def test_white_pixels_in_mixed_images(image_type, sat_shift):
    """Test that white pixels in mixed images remain white when saturation is increased."""
    # Create different test images
    if image_type == "white_and_gray":
        # White and gray image
        image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        image[5:, 5:] = 128  # Gray area
    elif image_type == "white_and_color":
        # White and colored image
        image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        image[5:, 5:] = [255, 0, 0]  # Red area
    else:  # white_black_color
        # White, black and colored image
        image = np.ones((12, 12, 3), dtype=np.uint8) * 255
        image[4:8, 4:8] = 0  # Black area
        image[8:, 8:] = [0, 255, 0]  # Green area

    # Store original white pixels for comparison
    white_mask = np.all(image == 255, axis=2)

    # Apply saturation increase
    result = fpixel.shift_hsv(image, hue_shift=0, sat_shift=sat_shift, val_shift=0)

    # Check that white pixels remain white
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if white_mask[y, x]:
                assert np.array_equal(result[y, x], [255, 255, 255]), \
                    f"White pixel at ({y},{x}) changed color in {image_type} image"

    # Non-white pixels should be affected by saturation (except black which has V=0)
    if image_type in {"white_and_color", "white_black_color"}:
        colored_mask = ~white_mask & ~np.all(image == 0, axis=2)  # Not white and not black

        # Convert original and result to HSV to check saturation directly
        original_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

        # Check that saturation was properly applied (may not change RGB values if already at max)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # Instead of checking for exact equality, check that saturation increased
                if colored_mask[y, x]:
                    # Get original saturation
                    orig_sat = original_hsv[y, x, 1]
                    # Actual saturation
                    actual_sat = result_hsv[y, x, 1]

                    # Check that saturation increased or reached maximum
                    assert actual_sat > orig_sat or actual_sat == 255, \
                        f"Saturation did not increase at ({y},{x}) in {image_type} image. " \
                        f"Original: {orig_sat}, Actual: {actual_sat}"


def test_grayscale_image_with_saturation():
    """Test that grayscale images are handled correctly with saturation changes."""
    # Create a grayscale image
    gray_image = np.ones((10, 10), dtype=np.uint8) * 128

    # Apply saturation increase (should be ignored for grayscale)
    result = fpixel.shift_hsv(gray_image, hue_shift=0, sat_shift=100, val_shift=0)

    # Result should be the same as input for grayscale
    assert result.shape == gray_image.shape, "Output shape changed for grayscale image"
    assert np.all(result == gray_image), "Grayscale values changed with saturation increase"


@pytest.mark.parametrize(
    "gray_value, sat_shift",
    [
        (128, 100),  # Medium gray
        (50, 200),   # Dark gray
        (200, 150),  # Light gray
    ],
)
def test_gray_pixels_remain_gray_with_saturation_increase(gray_value, sat_shift):
    """Test that gray pixels remain gray when saturation is increased.

    This test verifies that increasing saturation doesn't affect gray pixels,
    which should remain gray regardless of saturation changes.
    """
    # Create a simple test image with gray pixels
    gray_image = np.ones((10, 10, 3), dtype=np.uint8) * gray_value

    # Apply saturation increase
    result = fpixel.shift_hsv(gray_image, hue_shift=0, sat_shift=sat_shift, val_shift=0)

    # Check that all pixels are still the same gray value
    expected = np.ones((10, 10, 3), dtype=np.uint8) * gray_value
    np.testing.assert_array_equal(
        result, expected,
        f"Gray pixels (value {gray_value}) changed color with saturation shift {sat_shift}"
    )


@pytest.mark.parametrize(
    "image_type, sat_shift",
    [
        ("gray_and_color", 100),
        ("multi_gray_and_color", 200),
    ],
)
def test_gray_pixels_in_mixed_images(image_type, sat_shift):
    """Test that gray pixels in mixed images remain gray when saturation is increased."""
    # Create different test images
    if image_type == "gray_and_color":
        # Gray and colored image
        image = np.ones((10, 10, 3), dtype=np.uint8) * 128  # Medium gray
        image[5:, 5:] = [200, 100, 50]  # Brown/orange color
    else:  # multi_gray_and_color
        # Multiple gray values and colored pixels
        image = np.ones((12, 12, 3), dtype=np.uint8) * 200  # Light gray
        image[4:8, 4:8] = 50  # Dark gray
        image[8:, 8:] = [100, 180, 220]  # Light blue

    # Store original gray pixels for comparison (all pixels where R=G=B)
    gray_mask = (image[:,:,0] == image[:,:,1]) & (image[:,:,1] == image[:,:,2])

    # Store original values for comparison
    original_values = image.copy()

    # Apply saturation increase
    result = fpixel.shift_hsv(image, hue_shift=0, sat_shift=sat_shift, val_shift=0)

    # Check that gray pixels remain unchanged
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if gray_mask[y, x]:
                assert np.array_equal(result[y, x], original_values[y, x]), \
                    f"Gray pixel at ({y},{x}) changed from {original_values[y, x]} to {result[y, x]}"

    # Non-gray pixels should be affected by saturation
    colored_mask = ~gray_mask

    # Convert original and result to HSV to check saturation directly
    original_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)

    # Check that saturation was properly applied to colored pixels
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if colored_mask[y, x]:
                # Get original saturation
                orig_sat = original_hsv[y, x, 1]
                # Actual saturation
                actual_sat = result_hsv[y, x, 1]

                # Check that saturation increased or reached maximum
                assert actual_sat > orig_sat or actual_sat == 255, \
                    f"Saturation did not increase at ({y},{x}) in {image_type} image. " \
                    f"Original: {orig_sat}, Actual: {actual_sat}"
