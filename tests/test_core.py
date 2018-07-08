from __future__ import absolute_import

import cv2
import numpy as np

from albumentations.core.transforms_interface import to_tuple, ImageOnlyTransform, DualTransform
from albumentations.core.composition import OneOrOther, Compose, OneOf
from .compat import mock, MagicMock, Mock, call


def test_one_or_other():
    first = MagicMock()
    second = MagicMock()
    augmentation = OneOrOther(first, second, p=1)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert first.called != second.called


def test_compose():
    first = MagicMock()
    second = MagicMock()
    augmentation = Compose([first, second], p=1)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert first.called
    assert second.called


def test_one_of():
    transforms = [Mock(p=1) for _ in range(10)]
    augmentation = OneOf(transforms, p=1)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert len([transform for transform in transforms if transform.called]) == 1


def test_to_tuple():
    assert to_tuple(10) == (-10, 10)
    assert to_tuple(0.5) == (-0.5, 0.5)
    assert to_tuple((-20, 20)) == (-20, 20)
    assert to_tuple(100, low=30) == (30, 100)


def test_image_only_transform(image, mask):
    height, width = image.shape[:2]
    with mock.patch.object(ImageOnlyTransform, 'apply') as mocked_apply:
        with mock.patch.object(ImageOnlyTransform, 'get_params', return_value={'interpolation': cv2.INTER_LINEAR}):
            aug = ImageOnlyTransform(p=1)
            data = aug(image=image, mask=mask)
            mocked_apply.assert_called_once_with(image, interpolation=cv2.INTER_LINEAR, cols=width, rows=height)
            assert np.array_equal(data['mask'], mask)


def test_dual_transform(image, mask):
    image_call = call(image, interpolation=cv2.INTER_LINEAR, cols=image.shape[1], rows=image.shape[0])
    mask_call = call(mask, interpolation=cv2.INTER_NEAREST, cols=mask.shape[1], rows=mask.shape[0])
    with mock.patch.object(DualTransform, 'apply') as mocked_apply:
        with mock.patch.object(DualTransform, 'get_params', return_value={'interpolation': cv2.INTER_LINEAR}):
            aug = DualTransform(p=1)
            aug(image=image, mask=mask)
            mocked_apply.assert_has_calls([image_call, mask_call], any_order=True)
