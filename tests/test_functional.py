from __future__ import absolute_import

import cv2
import numpy as np
import pytest

import albumentations.augmentations.functional as F
from .utils import convert_2d_to_3d


@pytest.mark.parametrize('target', ['image', 'mask'])
def test_vflip(target):
    img = np.array(
        [[1, 1, 1],
         [0, 1, 1],
         [0, 0, 1]], dtype=np.uint8)
    expected_output = np.array(
        [[0, 0, 1],
         [0, 1, 1],
         [1, 1, 1]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
        expected_output = convert_2d_to_3d(expected_output)
    flipped_img = F.vflip(img)
    assert np.array_equal(flipped_img, expected_output)


@pytest.mark.parametrize('target', ['image', 'mask'])
def test_hflip(target):
    img = np.array(
        [[1, 1, 1],
         [0, 1, 1],
         [0, 0, 1]], dtype=np.uint8)
    expected_output = np.array(
        [[1, 1, 1],
         [1, 1, 0],
         [1, 0, 0]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
        expected_output = convert_2d_to_3d(expected_output)
    flipped_img = F.hflip(img)
    assert np.array_equal(flipped_img, expected_output)


@pytest.mark.parametrize('target', ['image', 'mask'])
@pytest.mark.parametrize(['code', 'func'], [
    [0, F.vflip],
    [1, F.hflip],
    [-1, lambda img: F.vflip(F.hflip(img))],

])
def test_random_flip(code, func, target):
    img = np.array(
        [[1, 1, 1],
         [0, 1, 1],
         [0, 0, 1]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
    assert np.array_equal(F.random_flip(img, code), func(img))


@pytest.mark.parametrize(['input_shape', 'expected_shape'], [
    [(128, 64), (64, 128)],
    [(128, 64, 3), (64, 128, 3)],
])
def test_transpose(input_shape, expected_shape):
    img = np.random.randint(low=0, high=256, size=input_shape, dtype=np.uint8)
    transposed = F.transpose(img)
    assert transposed.shape == expected_shape


@pytest.mark.parametrize('target', ['image', 'mask'])
def test_rot90(target):
    img = np.array(
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 1]], dtype=np.uint8)
    expected = np.array(
        [[1, 1, 1],
         [0, 0, 0],
         [0, 0, 0]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
        expected = convert_2d_to_3d(expected)
    rotated = F.rot90(img, factor=1)
    assert np.array_equal(rotated, expected)


def test_normalize():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    normalized = F.normalize(img, mean=50, std=3)
    expected = (np.ones((100, 100, 3), dtype=np.float32) * 127 / 255 - 50) / 3
    assert np.allclose(normalized, expected)


def test_compare_rotate_and_shift_scale_rotate(image):
    rotated_img_1 = F.rotate(image, angle=60)
    rotated_img_2 = F.shift_scale_rotate(image, angle=60, scale=1, dx=0, dy=0)
    assert np.array_equal(rotated_img_1, rotated_img_2)


@pytest.mark.parametrize('target', ['image', 'mask'])
def test_center_crop(target):
    img = np.array(
        [[1, 1, 1, 1],
         [0, 1, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 1]], dtype=np.uint8)
    expected_output = np.array(
        [[1, 1],
         [0, 1]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
        expected_output = convert_2d_to_3d(expected_output)
    cropped_img = F.center_crop(img, 2, 2)
    assert np.array_equal(cropped_img, expected_output)


def test_center_crop_with_incorrectly_large_crop_size():
    img = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, message='Requested crop size (8, 8) is larger than the image size (4, 4)'):
        F.center_crop(img, 8, 8)


@pytest.mark.parametrize('target', ['image', 'mask'])
def test_random_crop(target):
    img = np.array(
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]], dtype=np.uint8)
    expected = np.array(
        [[5, 6],
         [9, 10]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
        expected = convert_2d_to_3d(expected)
    cropped_img = F.random_crop(img, crop_height=2, crop_width=2, h_start=0.5, w_start=0)
    assert np.array_equal(cropped_img, expected)


def test_random_crop_with_incorrectly_large_crop_size():
    img = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, message='Requested crop size (8, 8) is larger than the image size (4, 4)'):
        F.random_crop(img, crop_height=8, crop_width=8, h_start=0, w_start=0)


def test_clip():
    img = np.array(
        [[-300, 0],
         [100, 400]], dtype=np.float32)
    expected = np.array(
        [[0, 0],
         [100, 255]], dtype=np.float32)
    clipped = F.clip(img, dtype=np.uint8, maxval=255)
    assert np.array_equal(clipped, expected)


@pytest.mark.parametrize(['bbox', 'expected_output'], [
    [(1, 2, 5, 5), (194, 2, 5, 5)],
    [(45, 67, 35, 24), (120, 67, 35, 24)],
])
@pytest.mark.parametrize('input_type', [tuple, list, np.array])
def test_bbox_vflip(bbox, expected_output, input_type):
    bbox = input_type(bbox)
    flipped_bbox = F.bbox_vflip(bbox, cols=200, rows=100)
    assert isinstance(flipped_bbox, tuple)
    assert np.array_equal(flipped_bbox, expected_output)


@pytest.mark.parametrize(['bbox', 'expected_output'], [
    [(1, 2, 5, 5), (1, 93, 5, 5)],
    [(45, 67, 35, 24), (45, 9, 35, 24)],
])
@pytest.mark.parametrize('input_type', [tuple, list, np.array])
def test_bbox_hflip(bbox, expected_output, input_type):
    bbox = input_type(bbox)
    flipped_bbox = F.bbox_hflip(bbox, cols=200, rows=100)
    assert isinstance(flipped_bbox, tuple)
    assert np.array_equal(flipped_bbox, expected_output)


@pytest.mark.parametrize('target', ['image', 'mask'])
def test_pad(target):
    img = np.array(
        [[1, 2],
         [3, 4]], dtype=np.uint8)
    expected = np.array(
        [[4, 3, 4, 3],
         [2, 1, 2, 1],
         [4, 3, 4, 3],
         [2, 1, 2, 1]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
        expected = convert_2d_to_3d(expected)
    padded = F.pad(img, min_height=4, min_width=4)
    assert np.array_equal(padded, expected)


@pytest.mark.parametrize('target', ['image', 'mask'])
def test_rotate_from_shift_scale_rotate(target):
    img = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]], dtype=np.uint8)
    expected = np.array([
        [0, 0, 0, 0],
        [4, 8, 12, 16],
        [3, 7, 11, 15],
        [2, 6, 10, 14]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
        expected = convert_2d_to_3d(expected)
    rotated_img = F.shift_scale_rotate(img, angle=90, scale=1, dx=0, dy=0, interpolation=cv2.INTER_NEAREST,
                                       border_mode=cv2.BORDER_CONSTANT)
    assert np.array_equal(rotated_img, expected)


@pytest.mark.parametrize('target', ['image', 'mask'])
def test_scale_from_shift_scale_rotate(target):
    img = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]], dtype=np.uint8)
    expected = np.array([
        [6, 7, 7, 8],
        [10, 11, 11, 12],
        [10, 11, 11, 12],
        [14, 15, 15, 16]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
        expected = convert_2d_to_3d(expected)
    scaled_img = F.shift_scale_rotate(img, angle=0, scale=2, dx=0, dy=0, interpolation=cv2.INTER_NEAREST,
                                      border_mode=cv2.BORDER_CONSTANT)
    assert np.array_equal(scaled_img, expected)


@pytest.mark.parametrize('target', ['image', 'mask'])
def test_shift_x_from_shift_scale_rotate(target):
    img = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]], dtype=np.uint8)
    expected = np.array([
        [0, 0, 1, 2],
        [0, 0, 5, 6],
        [0, 0, 9, 10],
        [0, 0, 13, 14]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
        expected = convert_2d_to_3d(expected)
    shifted_along_x_img = F.shift_scale_rotate(img, angle=0, scale=1, dx=0.5, dy=0, interpolation=cv2.INTER_NEAREST,
                                               border_mode=cv2.BORDER_CONSTANT)
    assert np.array_equal(shifted_along_x_img, expected)


@pytest.mark.parametrize('target', ['image', 'mask'])
def test_shift_y_from_shift_scale_rotate(target):
    img = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]], dtype=np.uint8)
    expected = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 2, 3, 4],
        [5, 6, 7, 8]], dtype=np.uint8)
    if target == 'image':
        img = convert_2d_to_3d(img)
        expected = convert_2d_to_3d(expected)
    shifted_along_y_img = F.shift_scale_rotate(img, angle=0, scale=1, dx=0, dy=0.5, interpolation=cv2.INTER_NEAREST,
                                               border_mode=cv2.BORDER_CONSTANT)
    assert np.array_equal(shifted_along_y_img, expected)


@pytest.mark.parametrize(['shift_params', 'expected'], [
    [(-10, 0, 10), (117, 127, 137)],
    [(-200, 0, 200), (0, 127, 255)],
])
def test_shift_rgb(shift_params, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    r_shift, g_shift, b_shift = shift_params
    img = F.shift_rgb(img, r_shift=r_shift, g_shift=g_shift, b_shift=b_shift)
    expected_r, expected_g, expected_b = expected
    assert img.dtype == np.dtype('uint8')
    assert (img[:, :, 0] == expected_r).all()
    assert (img[:, :, 1] == expected_g).all()
    assert (img[:, :, 2] == expected_b).all()


@pytest.mark.parametrize(['alpha', 'expected'], [(1.5, 190), (3, 255)])
def test_random_brigtness(alpha, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    img = F.random_brightness(img, alpha)
    assert img.dtype == np.dtype('uint8')
    assert (img == expected).all()


@pytest.mark.parametrize(['alpha', 'expected'], [(1.2, 76), (0.1, 255), (10, 0)])
def test_random_contrast(alpha, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    img = F.random_contrast(img, alpha)
    assert img.dtype == np.dtype('uint8')
    assert (img == expected).all()


@pytest.mark.parametrize(['gamma', 'expected'], [(1, 1), (10, 146)])
def test_gamma_transform(gamma, expected):
    img = np.ones((100, 100, 3), dtype=np.uint8)
    img = F.gamma_transform(img, gamma=gamma)
    assert img.dtype == np.dtype('uint8')
    assert (img == expected).all()
