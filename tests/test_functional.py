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
