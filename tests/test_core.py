from unittest import mock
from unittest.mock import Mock, MagicMock, call

import cv2
import numpy as np
import pytest
from hypothesis import given, example
from hypothesis.strategies import floats as h_float
from hypothesis.strategies import integers as h_int

from albumentations.augmentations.bbox_utils import check_bboxes
from albumentations.augmentations.transforms import HorizontalFlip, Rotate, Blur, MedianBlur
from albumentations.core.composition import OneOrOther, Compose, OneOf, PerChannel, ReplayCompose
from albumentations.core.transforms_interface import to_tuple, ImageOnlyTransform, DualTransform
from tests.utils import h_mask, h_image


def test_one_or_other(image):
    first = MagicMock()
    second = MagicMock()
    augmentation = OneOrOther(first, second, p=1)
    augmentation(image=image)
    assert first.called != second.called


def test_compose(image):
    first = MagicMock()
    second = MagicMock()
    augmentation = Compose([first, second], p=1)
    augmentation(image=image)
    assert first.called
    assert second.called


def oneof_always_apply_crash(image):
    aug = Compose([HorizontalFlip(), Rotate(), OneOf([Blur(), MedianBlur()], p=1)], p=1)
    data = aug(image=image)
    assert data


def test_always_apply(image):
    first = MagicMock(always_apply=True)
    second = MagicMock(always_apply=False)
    augmentation = Compose([first, second], p=0)
    augmentation(image=image)
    assert first.called
    assert not second.called


def test_one_of(image):
    transforms = [Mock(p=1) for _ in range(10)]
    augmentation = OneOf(transforms, p=1)
    augmentation(image=image)
    assert len([transform for transform in transforms if transform.called]) == 1


@given(
    int_value=h_int(min_value=0, max_value=255),
    float_value=h_float(min_value=0, max_value=1),
    int_value2=h_int(min_value=0, max_value=255),
)
@example(int_value=20, float_value=0.5, int_value2=60)
def test_to_tuple(int_value, float_value, int_value2):
    assert to_tuple(int_value) == (-int_value, int_value)
    assert to_tuple(float_value) == (-float_value, float_value)
    assert to_tuple((-int_value, int_value)) == (-int_value, int_value)
    assert to_tuple([-int_value, int_value]) == (-int_value, int_value)
    assert to_tuple(int_value, bias=1) == (-(int_value - 1), int_value + 1)
    assert to_tuple(int_value, bias=2) == (-(int_value - 2), int_value + 2)

    min_int = min(int_value, int_value2)
    max_int = max(int_value, int_value2)
    assert to_tuple(max_int, low=min_int) == (min_int, max_int)


def test_image_only_transform(image, mask):
    height, width = image.shape[:2]
    with mock.patch.object(ImageOnlyTransform, "apply") as mocked_apply:
        with mock.patch.object(ImageOnlyTransform, "get_params", return_value={"interpolation": cv2.INTER_LINEAR}):
            aug = ImageOnlyTransform(p=1)
            data = aug(image=image, mask=mask)
            mocked_apply.assert_called_once_with(image, interpolation=cv2.INTER_LINEAR, cols=width, rows=height)
            assert np.array_equal(data["mask"], mask)


def test_dual_transform(image, mask):
    image_call = call(image, interpolation=cv2.INTER_LINEAR, cols=image.shape[1], rows=image.shape[0])
    mask_call = call(mask, interpolation=cv2.INTER_NEAREST, cols=mask.shape[1], rows=mask.shape[0])
    with mock.patch.object(DualTransform, "apply") as mocked_apply:
        with mock.patch.object(DualTransform, "get_params", return_value={"interpolation": cv2.INTER_LINEAR}):
            aug = DualTransform(p=1)
            aug(image=image, mask=mask)
            mocked_apply.assert_has_calls([image_call, mask_call], any_order=True)


@given(image=h_image(), mask=h_mask())
def test_additional_targets(image, mask):
    image_call = call(image, interpolation=cv2.INTER_LINEAR, cols=image.shape[1], rows=image.shape[0])
    image2_call = call(mask, interpolation=cv2.INTER_LINEAR, cols=mask.shape[1], rows=mask.shape[0])
    with mock.patch.object(DualTransform, "apply") as mocked_apply:
        with mock.patch.object(DualTransform, "get_params", return_value={"interpolation": cv2.INTER_LINEAR}):
            aug = DualTransform(p=1)
            aug.add_targets({"image2": "image"})
            aug(image=image, image2=mask)
            mocked_apply.assert_has_calls([image_call, image2_call], any_order=True)


def test_check_bboxes_with_correct_values():
    try:
        check_bboxes([[0.1, 0.5, 0.8, 1.0], [0.2, 0.5, 0.5, 0.6, 99]])
    except Exception as e:
        pytest.fail("Unexpected Exception {!r}".format(e))


def test_check_bboxes_with_values_less_than_zero():
    with pytest.raises(ValueError) as exc_info:
        check_bboxes([[0.2, 0.5, 0.5, 0.6, 99], [-0.1, 0.5, 0.8, 1.0]])
    message = "Expected x_min for bbox [-0.1, 0.5, 0.8, 1.0] to be in the range [0.0, 1.0], got -0.1."
    assert str(exc_info.value) == message


def test_check_bboxes_with_values_greater_than_one():
    with pytest.raises(ValueError) as exc_info:
        check_bboxes([[0.2, 0.5, 1.5, 0.6, 99], [0.1, 0.5, 0.8, 1.0]])
    message = "Expected x_max for bbox [0.2, 0.5, 1.5, 0.6, 99] to be in the range [0.0, 1.0], got 1.5."
    assert str(exc_info.value) == message


def test_check_bboxes_with_end_greater_that_start():
    with pytest.raises(ValueError) as exc_info:
        check_bboxes([[0.8, 0.5, 0.7, 0.6, 99], [0.1, 0.5, 0.8, 1.0]])
    message = "x_max is less than or equal to x_min for bbox [0.8, 0.5, 0.7, 0.6, 99]."
    assert str(exc_info.value) == message


def test_per_channel_mono(mask):
    transforms = [Blur(), Rotate()]
    augmentation = PerChannel(transforms, p=1)
    data = augmentation(image=mask)
    assert data


def test_per_channel_multi(image):
    transforms = [Blur(), Rotate()]
    augmentation = PerChannel(transforms, p=1)
    data = augmentation(image=image)
    assert data


def test_deterministic_oneof(image):
    aug = ReplayCompose([OneOf([HorizontalFlip(), Blur()])], p=1)
    image2 = np.copy(image)
    data = aug(image=image)
    assert "replay" in data
    data2 = ReplayCompose.replay(data["replay"], image=image2)
    assert np.array_equal(data["image"], data2["image"])


def test_deterministic_one_or_other(image):
    aug = ReplayCompose([OneOrOther(HorizontalFlip(), Blur())], p=1)
    image2 = np.copy(image)
    data = aug(image=image)
    assert "replay" in data
    data2 = ReplayCompose.replay(data["replay"], image=image2)
    assert np.array_equal(data["image"], data2["image"])
