from __future__ import absolute_import

import cv2
import numpy as np
import pytest

from albumentations.core.transforms_interface import to_tuple, ImageOnlyTransform, DualTransform
from albumentations.augmentations.bbox_utils import check_bboxes
from albumentations.core.composition import OneOrOther, Compose, OneOf
from albumentations.augmentations.transforms import HorizontalFlip, Rotate, Blur, MedianBlur
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


def test_compose_to_tensor():
    first = MagicMock()
    second = MagicMock()
    to_tensor = MagicMock()
    augmentation = Compose([first, second], to_tensor=to_tensor, p=0)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert to_tensor.called


def oneof_always_apply_crash():
    aug = Compose([
        HorizontalFlip(),
        Rotate(),
        OneOf([
            Blur(),
            MedianBlur(),
        ], p=1)], p=1)
    image = np.ones((8, 8))
    data = aug(image=image)
    assert data


def test_always_apply():
    first = MagicMock(always_apply=True)
    second = MagicMock(always_apply=False)
    augmentation = Compose([first, second], p=0)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert first.called
    assert not second.called


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
    assert to_tuple([-20, 20]) == (-20, 20)
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


def test_additional_targets(image, mask):
    image_call = call(image, interpolation=cv2.INTER_LINEAR, cols=image.shape[1], rows=image.shape[0])
    image2_call = call(mask, interpolation=cv2.INTER_LINEAR, cols=mask.shape[1], rows=mask.shape[0])
    with mock.patch.object(DualTransform, 'apply') as mocked_apply:
        with mock.patch.object(DualTransform, 'get_params', return_value={'interpolation': cv2.INTER_LINEAR}):
            aug = DualTransform(p=1)
            aug.add_targets({'image2': 'image'})
            aug(image=image, image2=mask)
            mocked_apply.assert_has_calls([image_call, image2_call], any_order=True)


def test_check_bboxes_with_correct_values():
    try:
        check_bboxes([[0.1, 0.5, 0.8, 1.0], [0.2, 0.5, 0.5, 0.6, 99]])
    except Exception as e:
        pytest.fail('Unexpected Exception {!r}'.format(e))


def test_check_bboxes_with_values_less_than_zero():
    with pytest.raises(ValueError) as exc_info:
        check_bboxes([[0.2, 0.5, 0.5, 0.6, 99], [-0.1, 0.5, 0.8, 1.0]])
    message = 'Expected x_min for bbox [-0.1, 0.5, 0.8, 1.0] to be in the range [0.0, 1.0], got -0.1.'
    assert str(exc_info.value) == message


def test_check_bboxes_with_values_greater_than_one():
    with pytest.raises(ValueError) as exc_info:
        check_bboxes([[0.2, 0.5, 1.5, 0.6, 99], [0.1, 0.5, 0.8, 1.0]])
    message = 'Expected x_max for bbox [0.2, 0.5, 1.5, 0.6, 99] to be in the range [0.0, 1.0], got 1.5.'
    assert str(exc_info.value) == message


def test_check_bboxes_with_end_greater_that_start():
    with pytest.raises(ValueError) as exc_info:
        check_bboxes([[0.8, 0.5, 0.7, 0.6, 99], [0.1, 0.5, 0.8, 1.0]])
    message = 'x_max is less than or equal to x_min for bbox [0.8, 0.5, 0.7, 0.6, 99].'
    assert str(exc_info.value) == message
