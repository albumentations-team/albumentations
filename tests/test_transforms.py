import numpy as np
import cv2
import pytest

from albumentations import Transpose, Rotate, ShiftScaleRotate
import albumentations.augmentations.functional as F


def test_transpose_both_image_and_mask():
    image = np.ones((8, 6, 3))
    mask = np.ones((8, 6))
    augmentation = Transpose(p=1)
    augmented = augmentation(image=image, mask=mask)
    assert augmented['image'].shape == (6, 8, 3)
    assert augmented['mask'].shape == (6, 8)


@pytest.mark.parametrize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = Rotate(limit=(45, 45), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = F.rotate(image, 45, interpolation=interpolation, border_mode=cv2.BORDER_REFLECT_101)
    expected_mask = F.rotate(mask, 45, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(data['image'], expected_image)
    assert np.array_equal(data['mask'], expected_mask)


@pytest.mark.parametrize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_shift_scale_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = ShiftScaleRotate(shift_limit=(0.2, 0.2), scale_limit=(1.1, 1.1), rotate_limit=(45, 45),
                           interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = F.shift_scale_rotate(image, angle=45, scale=2.1, dx=0.2, dy=0.2, interpolation=interpolation,
                                          border_mode=cv2.BORDER_REFLECT_101)
    expected_mask = F.shift_scale_rotate(mask, angle=45, scale=2.1, dx=0.2, dy=0.2, interpolation=cv2.INTER_NEAREST,
                                         border_mode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(data['image'], expected_image)
    assert np.array_equal(data['mask'], expected_mask)
