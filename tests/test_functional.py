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
