from __future__ import annotations

import typing
from unittest import mock
from unittest.mock import MagicMock, Mock, call, patch
import warnings
import torch
import cv2
import numpy as np
import pytest
from typing import Any

import albumentations as A
from albumentations.core.bbox_utils import check_bboxes
from albumentations.core.composition import (
    BaseCompose,
    BboxParams,
    Compose,
    KeypointParams,
    OneOf,
    OneOrOther,
    RandomOrder,
    ReplayCompose,
    Sequential,
    SomeOf,
)
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform, NoOp
from albumentations.core.utils import to_tuple, get_shape
from tests.conftest import (
    IMAGES,
    SQUARE_UINT8_IMAGE,
)

from .aug_definitions import transforms2metadata_key

from .utils import get_2d_transforms, get_dual_transforms, get_filtered_transforms, get_image_only_transforms, get_transforms, set_seed


def test_one_or_other():
    first = MagicMock()
    second = MagicMock()
    augmentation = OneOrOther(first, second, p=1)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert first.called != second.called


def test_compose():
    first = MagicMock(available_keys={"image"})
    second = MagicMock(available_keys={"image"})
    augmentation = Compose([first, second], p=1)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert first.called
    assert second.called


@pytest.mark.parametrize("target_as_params", ([], ["image"], ["image", "mask"], ["image", "mask", "keypoints"]))
def test_one_of(target_as_params):
    # Create a simple transform-like class for testing
    class DummyTransform:
        def __init__(self):
            self.p = 1
            self.available_keys = {"image"}
            self.targets_as_params = target_as_params
            self.params = {}

        def __call__(self, **kwargs):
            return kwargs

    transforms = [DummyTransform() for _ in range(10)]
    augmentation = OneOf(transforms, p=1)
    image = np.ones((8, 8))
    augmentation(image=image)


@pytest.mark.parametrize("N", [0, 1, 2, 5, 10, 12])
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("target_as_params", ([], ["image"], ["image", "mask"], ["image", "mask", "keypoints"]))
@pytest.mark.parametrize("aug", [SomeOf, RandomOrder])
def test_n_of(N, replace, target_as_params, aug):
    """Test for SomeOf and RandomOrder"""
    transforms = [
        Mock(
            p=1,
            side_effect=lambda **kw: {"image": kw["image"]},
            available_keys={"image"},
            targets_as_params=target_as_params,
        )
        for _ in range(10)
    ]
    augmentation = aug(transforms, N, p=1, replace=replace)
    image = np.ones((8, 8))
    augmentation(image=image)
    call_count = sum([transform.call_count for transform in transforms])
    if not replace:
        expected_count = min(N, 10)
        assert call_count == expected_count
        assert len([transform for transform in transforms if transform.called]) == call_count
    else:
        assert call_count == N


@pytest.mark.parametrize("target_as_params", ([], ["image"], ["image", "mask"], ["image", "mask", "keypoints"]))
def test_sequential(target_as_params):
    transforms = [
        Mock(side_effect=lambda **kw: kw, available_keys={"image"}, targets_as_params=target_as_params)
        for _ in range(10)
    ]
    augmentation = Sequential(transforms, p=1)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert len([transform for transform in transforms if transform.called]) == len(transforms)


@pytest.mark.parametrize(
    "input,kwargs,expected",
    [
        (10, {}, (-10, 10)),
        (0.5, {}, (-0.5, 0.5)),
        ((-20, 20), {}, (-20, 20)),
        ([-20, 20], {}, (-20, 20)),
        ((1, 2), {"low": 1}, (1, 2)),
        (100, {"low": 30}, (30, 100)),
        (10, {"bias": 1}, (-9, 11)),
        (100, {"bias": 2}, (-98, 102)),
    ],
)
def test_to_tuple(input, kwargs, expected):
    assert to_tuple(input, **kwargs) == expected


@pytest.mark.parametrize("image", IMAGES)
def test_image_only_transform(image):
    mask = image.copy()
    height, width = image.shape[:2]
    with mock.patch.object(ImageOnlyTransform, "apply") as mocked_apply:
        with mock.patch.object(ImageOnlyTransform, "get_params", return_value={"interpolation": cv2.INTER_LINEAR}):
            aug = ImageOnlyTransform(p=1)
            data = aug(image=image, mask=mask)
            mocked_apply.assert_called_once_with(
                image,
                interpolation=cv2.INTER_LINEAR,
                shape=image.shape,
            )
            np.testing.assert_array_equal(data["mask"], mask)


@pytest.mark.parametrize("image", IMAGES)
def test_dual_transform(image):
    mask = image.copy()

    with mock.patch.object(DualTransform, "apply") as mocked_apply:
        with mock.patch.object(DualTransform, "get_params", return_value={}):  # Empty params
            aug = DualTransform(p=1)
            aug(image=image, mask=mask)

            # Get the actual calls
            calls = mocked_apply.call_args_list
            assert len(calls) == 2  # Should be called twice

            # Check each call has correct structure
            for call_args in calls:
                args, kwargs = call_args

                # Check kwargs contain correct keys and values
                assert "shape" in kwargs
                assert kwargs["shape"] == image.shape

                # Check input array is either image or mask
                input_array = args[0]
                assert np.array_equal(input_array, image) or np.array_equal(input_array, mask)


@pytest.mark.parametrize("image", IMAGES)
def test_additional_targets(image):
    mask = image.copy()
    image_call = call(
        image,
        interpolation=cv2.INTER_LINEAR,
        shape=image.shape,
    )
    image2_call = call(
        mask,
        interpolation=cv2.INTER_LINEAR,
        shape=mask.shape,
    )
    with mock.patch.object(DualTransform, "apply") as mocked_apply:
        with mock.patch.object(DualTransform, "get_params", return_value={"interpolation": cv2.INTER_LINEAR}):
            aug = DualTransform(p=1)
            aug.add_targets({"image2": "image"})
            aug(image=image, image2=mask)
            mocked_apply.assert_has_calls([image_call, image2_call], any_order=True)


def test_check_bboxes_with_correct_values():
    try:
        check_bboxes(np.array([[0.1, 0.5, 0.8, 1.0, 1], [0.2, 0.5, 0.5, 0.6, 99]]))
    except Exception as e:
        pytest.fail(f"Unexpected Exception {e!r}")


def test_check_bboxes_with_values_less_than_zero():
    with pytest.raises(ValueError) as exc_info:
        check_bboxes(np.array([[0.2, 0.5, 0.5, 0.6, 99], [-0.1, 0.5, 0.8, 1.0, 0]]))
    message = "Expected x_min for bbox [-0.1  0.5  0.8  1.   0. ] to be in the range [0.0, 1.0], got -0.1."
    assert str(exc_info.value) == message


def test_check_bboxes_with_values_greater_than_one():
    with pytest.raises(ValueError) as exc_info:
        check_bboxes(np.array([[0.2, 0.5, 1.5, 0.6, 99], [0.1, 0.5, 0.8, 1.0, 0]]))
    message = "Expected x_max for bbox [ 0.2  0.5  1.5  0.6 99. ] to be in the range [0.0, 1.0], got 1.5."
    assert str(exc_info.value) == message


def test_check_bboxes_with_end_greater_that_start():
    with pytest.raises(ValueError) as exc_info:
        check_bboxes(np.array([[0.8, 0.5, 0.7, 0.6, 99], [0.1, 0.5, 0.8, 1.0, 0]]))
    message = "x_max is less than or equal to x_min for bbox [ 0.8  0.5  0.7  0.6 99. ]."
    assert str(exc_info.value) == message


def test_deterministic_oneof() -> None:
    aug = ReplayCompose([OneOf([A.HorizontalFlip(p=1), A.Blur(p=1)])], p=1)
    for _ in range(10):
        image = (np.random.random((8, 8)) * 255).astype(np.uint8)
        image2 = np.copy(image)
        data = aug(image=image)
        assert "replay" in data
        data2 = ReplayCompose.replay(data["replay"], image=image2)
        assert np.array_equal(data["image"], data2["image"])


def test_deterministic_one_or_other() -> None:
    aug = ReplayCompose([OneOrOther(A.HorizontalFlip(p=1), A.Blur(p=1))], p=1)
    for _ in range(10):
        image = (np.random.random((8, 8)) * 255).astype(np.uint8)
        image2 = np.copy(image)
        data = aug(image=image)
        assert "replay" in data
        data2 = ReplayCompose.replay(data["replay"], image=image2)
        assert np.array_equal(data["image"], data2["image"])


def test_deterministic_sequential() -> None:
    aug = ReplayCompose([Sequential([A.HorizontalFlip(p=1), A.Blur(p=1)])], p=1)
    for _ in range(10):
        image = (np.random.random((8, 8)) * 255).astype(np.uint8)
        image2 = np.copy(image)
        data = aug(image=image)
        assert "replay" in data
        data2 = ReplayCompose.replay(data["replay"], image=image2)
        assert np.array_equal(data["image"], data2["image"])


def test_named_args():
    image = np.empty([100, 100, 3], dtype=np.uint8)
    aug = A.HorizontalFlip(p=1)

    with pytest.raises(KeyError) as exc_info:
        aug(image)
    assert str(exc_info.value) == (
        "'You have to pass data to augmentations as named arguments, for example: aug(image=image)'"
    )


@pytest.mark.parametrize(
    ["targets", "additional_targets", "err_message"],
    [
        [{"image": None}, None, "image must be numpy array type"],
        [{"image": np.empty([100, 100, 3], np.uint8), "mask": None}, None, "mask must be numpy array type"],
        [
            {"image": np.empty([100, 100, 3], np.uint8), "image1": None},
            {"image1": "image"},
            "image1 must be numpy array type",
        ],
        [
            {"image": np.empty([100, 100, 3], np.uint8), "mask1": None},
            {"mask1": "mask"},
            "mask1 must be numpy array type",
        ],
    ],
)
def test_targets_type_check(targets, additional_targets, err_message):
    aug = Compose([A.NoOp()], additional_targets=additional_targets, strict=True)

    with pytest.raises(TypeError) as exc_info:
        aug(**targets)
    assert str(exc_info.value) == err_message

    aug = Compose([A.NoOp()], strict=True)
    aug.add_targets(additional_targets)
    with pytest.raises(TypeError) as exc_info:
        aug(**targets)
    assert str(exc_info.value) == err_message


@pytest.mark.parametrize(
    ["targets", "bbox_params", "keypoint_params", "expected"],
    [
        [
            {"keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]]},
            None,
            KeypointParams("xy", check_each_transform=False),
            {"keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25},
        ],
        [
            {"keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]]},
            None,
            KeypointParams("xy", check_each_transform=True),
            {"keypoints": np.array([[10, 10]]) + 25},
        ],
        [
            {"bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]]},
            BboxParams("pascal_voc", check_each_transform=False),
            None,
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]]},
        ],
        [
            {"bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]]},
            BboxParams("pascal_voc", check_each_transform=True),
            None,
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]]},
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=True),
            KeypointParams("xy", check_each_transform=True),
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]], "keypoints": np.array([[10, 10]]) + 25},
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=False),
            KeypointParams("xy", check_each_transform=True),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]],
                "keypoints": np.array([[10, 10]]) + 25,
            },
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=True),
            KeypointParams("xy", check_each_transform=False),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]],
                "keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25,
            },
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=False),
            KeypointParams("xy", check_each_transform=False),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]],
                "keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25,
            },
        ],
    ],
)
def test_check_each_transform(targets, bbox_params, keypoint_params, expected):
    image = np.empty([100, 100], dtype=np.uint8)
    augs = Compose(
        [A.Crop(0, 0, 50, 50), A.PadIfNeeded(100, 100, border_mode=cv2.BORDER_CONSTANT, fill=0)],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        seed=137
    )
    res = augs(image=image, **targets)

    for key, item in expected.items():
        np.testing.assert_allclose(np.array(item), np.array(res[key]), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ["targets", "bbox_params", "keypoint_params", "expected"],
    [
        [
            {"keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]]},
            None,
            KeypointParams("xy", check_each_transform=False),
            {"keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25},
        ],
        [
            {"keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]]},
            None,
            KeypointParams("xy", check_each_transform=True),
            {"keypoints": np.array([[10, 10]]) + 25},
        ],
        [
            {"bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]]},
            BboxParams("pascal_voc", check_each_transform=False),
            None,
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]]},
        ],
        [
            {"bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]]},
            BboxParams("pascal_voc", check_each_transform=True),
            None,
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]]},
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=True),
            KeypointParams("xy", check_each_transform=True),
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]], "keypoints": np.array([[10, 10]]) + 25},
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=False),
            KeypointParams("xy", check_each_transform=True),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]],
                "keypoints": np.array([[10, 10]]) + 25,
            },
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=True),
            KeypointParams("xy", check_each_transform=False),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]],
                "keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25,
            },
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=False),
            KeypointParams("xy", check_each_transform=False),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]],
                "keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25,
            },
        ],
    ],
)
def test_check_each_transform_compose(targets, bbox_params, keypoint_params, expected):
    """Test if compose inside compose"""
    image = np.empty([100, 100], dtype=np.uint8)

    augs = Compose(
        [Compose([A.Crop(0, 0, 50, 50), A.PadIfNeeded(100, 100, border_mode=cv2.BORDER_CONSTANT, fill=0)])],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        seed=137
    )
    res = augs(image=image, **targets)

    for key, item in expected.items():
        np.testing.assert_allclose(np.array(item), np.array(res[key]), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ["targets", "bbox_params", "keypoint_params", "expected"],
    [
        [
            {"keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]]},
            None,
            KeypointParams("xy", check_each_transform=False),
            {"keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25},
        ],
        [
            {"keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]]},
            None,
            KeypointParams("xy", check_each_transform=True),
            {"keypoints": np.array([[10, 10]]) + 25},
        ],
        [
            {"bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]]},
            BboxParams("pascal_voc", check_each_transform=False),
            None,
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]]},
        ],
        [
            {"bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]]},
            BboxParams("pascal_voc", check_each_transform=True),
            None,
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]]},
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=True),
            KeypointParams("xy", check_each_transform=True),
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]], "keypoints": np.array([[10, 10]]) + 25},
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=False),
            KeypointParams("xy", check_each_transform=True),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]],
                "keypoints": np.array([[10, 10]]) + 25,
            },
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=True),
            KeypointParams("xy", check_each_transform=False),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]],
                "keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25,
            },
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=False),
            KeypointParams("xy", check_each_transform=False),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]],
                "keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25,
            },
        ],
    ],
)
def test_check_each_transform_sequential(targets, bbox_params, keypoint_params, expected):
    """Test if sequential inside compose"""
    image = np.empty([100, 100], dtype=np.uint8)

    augs = Compose(
        [Sequential([A.Crop(0, 0, 50, 50), A.PadIfNeeded(100, 100, border_mode=cv2.BORDER_CONSTANT, fill=0)], p=1.0)],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
    )
    res = augs(image=image, **targets)

    for key, item in expected.items():
        np.testing.assert_allclose(np.array(item), np.array(res[key]), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ["targets", "bbox_params", "keypoint_params", "expected"],
    [
        [
            {"keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]]},
            None,
            KeypointParams("xy", check_each_transform=False),
            {"keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25},
        ],
        [
            {"keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]]},
            None,
            KeypointParams("xy", check_each_transform=True),
            {"keypoints": np.array([[10, 10]]) + 25},
        ],
        [
            {"bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]]},
            BboxParams("pascal_voc", check_each_transform=False),
            None,
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]]},
        ],
        [
            {"bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]]},
            BboxParams("pascal_voc", check_each_transform=True),
            None,
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]]},
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=True),
            KeypointParams("xy", check_each_transform=True),
            {"bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]], "keypoints": np.array([[10, 10]]) + 25},
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=False),
            KeypointParams("xy", check_each_transform=True),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]],
                "keypoints": np.array([[10, 10]]) + 25,
            },
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=True),
            KeypointParams("xy", check_each_transform=False),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 75, 75, 0]],
                "keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25,
            },
        ],
        [
            {
                "bboxes": [[0, 0, 10, 10, 0], [5, 5, 70, 70, 0], [60, 60, 70, 70, 0]],
                "keypoints": [[10, 10], [70, 70], [10, 70], [70, 10]],
            },
            BboxParams("pascal_voc", check_each_transform=False),
            KeypointParams("xy", check_each_transform=False),
            {
                "bboxes": [[25, 25, 35, 35, 0], [30, 30, 95, 95, 0], [85, 85, 95, 95, 0]],
                "keypoints": np.array([[10, 10], [70, 70], [10, 70], [70, 10]]) + 25,
            },
        ],
    ],
)
def test_check_each_transform_someof(targets, bbox_params, keypoint_params, expected):
    """Test if someof inside compose"""
    image = np.empty([100, 100], dtype=np.uint8)

    augs = Compose(
        [
            SomeOf([A.Crop(0, 0, 50, 50)], n=1, replace=False, p=1.0),
            SomeOf([A.PadIfNeeded(100, 100, border_mode=cv2.BORDER_CONSTANT, fill=0)], n=1, replace=False, p=1.0),
        ],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
    )
    res = augs(image=image, **targets)

    for key, item in expected.items():
        np.testing.assert_allclose(np.array(item), np.array(res[key]), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("image", IMAGES)
def test_bbox_params_is_not_set(image, bboxes):
    t = Compose([A.NoOp(p=1.0)], strict=True)
    with pytest.raises(ValueError) as exc_info:
        t(image=image, bboxes=bboxes)
    assert str(exc_info.value) == "bbox_params must be specified for bbox transformations"


@pytest.mark.parametrize(
    "compose_transform",
    get_filtered_transforms((BaseCompose,), custom_arguments={SomeOf: {"n": 1}}),
)
@pytest.mark.parametrize(
    "inner_transform",
    [(A.Normalize, {}), (A.Resize, {"height": 100, "width": 100})]
    + get_filtered_transforms((BaseCompose,), custom_arguments={SomeOf: {"n": 1}}),  # type: ignore
)
def test_single_transform_compose(
    compose_transform: tuple[type[BaseCompose], dict],
    inner_transform: tuple[typing.Union[type[BaseCompose], type[A.BasicTransform]], dict],
):
    compose_cls, compose_kwargs = compose_transform
    cls, kwargs = inner_transform
    transform = cls(transforms=[], **kwargs) if issubclass(cls, BaseCompose) else cls(**kwargs)

    with pytest.warns(UserWarning):
        res_transform = compose_cls(transforms=transform, **compose_kwargs)  # type: ignore
    assert isinstance(res_transform.transforms, list)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
        },
        except_augmentations={
            A.FDA,
            A.HistogramMatching,
            A.Lambda,
            A.RandomSizedBBoxSafeCrop,
            A.CropNonEmptyMaskIfExists,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.TextImage,
            A.RandomCropNearBBox,
            A.Mosaic,
        },
    ),
)
def test_contiguous_output_dual(augmentation_cls, params):
    set_seed(42)
    image = np.ones([3, 100, 100], dtype=np.uint8).transpose(1, 2, 0)
    mask = np.ones([2, 100, 100], dtype=np.uint8).transpose(1, 2, 0)

    # check preconditions
    assert not image.flags["C_CONTIGUOUS"]
    assert not mask.flags["C_CONTIGUOUS"]

    transform = augmentation_cls(p=1, **params)

    data = {"image": image, "mask": mask}
    if augmentation_cls == A.MaskDropout or augmentation_cls == A.ConstrainedCoarseDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask

    # pipeline always outputs contiguous results
    data = transform(**data)

    # confirm output contiguous
    # assert data["image"].flags["C_CONTIGUOUS"]
    assert data["mask"].flags["C_CONTIGUOUS"]
    assert data["image"].flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        except_augmentations={
            A.Lambda,
            A.RandomSizedBBoxSafeCrop,
            A.CropNonEmptyMaskIfExists,
            A.OverlayElements,
            A.TextImage,
            A.FromFloat,
            A.Mosaic,
        },
    ),
)
def test_contiguous_output_imageonly(augmentation_cls, params):
    set_seed(137)
    image = np.zeros([3, 100, 100], dtype=np.uint8).transpose(1, 2, 0)

    # check preconditions
    assert not image.flags["C_CONTIGUOUS"]

    transform = augmentation_cls(p=1, **params)

    data = {
        "image": image,
    }
    if augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]

    # pipeline always outputs contiguous results
    data = transform(**data)

    im = data["image"]

    # confirm output contiguous
    assert im.flags["C_CONTIGUOUS"], f"{(im.flags, im.strides, im.shape)!s}"


@pytest.mark.parametrize(
    "targets",
    [
        {"image": np.ones((20, 20, 3), dtype=np.uint8), "mask": np.ones((30, 20))},
        {"image": np.ones((20, 20, 3), dtype=np.uint8), "masks": [np.ones((30, 20))]},
    ],
)
def test_compose_image_mask_equal_size(targets):
    transforms = Compose([A.NoOp()])

    with pytest.raises(ValueError) as exc_info:
        transforms(**targets)

    assert str(exc_info.value).startswith(
        "Height and Width of image, mask or masks should be equal. "
        "You can disable shapes check by setting a parameter is_check_shapes=False "
        "of Compose class (do it only if you are sure about your data consistency).",
    )
    # test after disabling shapes check
    transforms = Compose([A.NoOp()], is_check_shapes=False)
    transforms(**targets)


def test_additional_targets_overwrite():
    """Check add_target rises error if trying add existing target."""
    transforms = Compose([], additional_targets={"image2": "image"})
    # add same name, same target, OK
    transforms.add_targets({"image2": "image"})
    with pytest.raises(ValueError) as exc_info:
        transforms.add_targets({"image2": "mask"})
    assert (
        str(exc_info.value) == "Trying to overwrite existed additional targets. Key=image2 Exists=image New value: mask"
    )


# Test 1: Probability 1 with HorizontalFlip
@pytest.mark.parametrize("image", IMAGES)
def test_sequential_with_horizontal_flip_prob_1(image):
    mask = image.copy()
    # Setup transformations
    transform = Sequential([A.HorizontalFlip(p=1)], p=1)
    expected_transform = Compose([A.HorizontalFlip(p=1)])

    with patch("random.random", return_value=0.1):  # Mocking probability less than 1
        result = transform(image=image, mask=mask)
        expected = expected_transform(image=image, mask=mask)

    assert np.array_equal(result["image"], expected["image"])
    assert np.array_equal(result["mask"], expected["mask"])


# Test 2: Probability 0 with HorizontalFlip
@pytest.mark.parametrize("image", IMAGES)
def test_sequential_with_horizontal_flip_prob_0(image):
    mask = image.copy()
    transform = Sequential([A.HorizontalFlip(p=1)], p=0)

    with patch("random.random", return_value=0.99):  # Mocking probability greater than 0
        result = transform(image=image, mask=mask)

    assert np.array_equal(result["image"], image)
    assert np.array_equal(result["mask"], mask)


# Test 3: Multiple flips and Transpose with probability 1
@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize("aug", [A.HorizontalFlip, A.VerticalFlip, A.Transpose])
def test_sequential_multiple_transformations(image, aug):
    mask = image.copy()

    transform = A.Sequential(
        [
            aug(p=1),
            aug(p=1),
        ],
        p=1,
    )

    with patch("random.random", return_value=0.1):  # Ensuring all transforms are applied
        result = transform(image=image, mask=mask)

    # Since HorizontalFlip, VerticalFlip, and Transpose are all applied twice, the image should be the same
    assert np.array_equal(result["image"], image)
    assert np.array_equal(result["mask"], mask)


@pytest.mark.parametrize(
    "transforms",
    [
        [  # image only
            A.Blur(p=1),
            A.MedianBlur(p=1),
            A.ToGray(p=1),
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.ImageCompression(quality_range=(75, 100), p=1),
        ],
        [  # with dual
            A.Blur(p=1),
            A.MedianBlur(p=1),
            A.ToGray(p=1),
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.ImageCompression(quality_range=(75, 100), p=1),
            A.Crop(x_max=50, y_max=50),
        ],
        [],  # empty
    ],
)
@pytest.mark.parametrize(
    ["compose_args", "args"],
    [
        [
            {},
            {"image": np.empty([100, 100, 3], dtype=np.uint8)},
        ],
        [
            {},
            {
                "image": np.empty([100, 100, 3], dtype=np.uint8),
                "mask": np.empty([100, 100, 3], dtype=np.uint8),
            },
        ],
        [
            {},
            {
                "image": np.empty([100, 100, 3], dtype=np.uint8),
                "masks": [np.empty([100, 100, 3], dtype=np.uint8)] * 3,
            },
        ],
        [
            dict(bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])),
            {
                "image": np.empty([100, 100, 3], dtype=np.uint8),
                "bboxes": [[0.5, 0.5, 0.1, 0.1]],
                "class_labels": [1],
            },
        ],
        [
            dict(keypoint_params=A.KeypointParams(format="xy", label_fields=["class_labels"])),
            {
                "image": np.empty([100, 100, 3], dtype=np.uint8),
                "keypoints": [[10, 20]],
                "class_labels": [1],
            },
        ],
        [
            dict(
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels_1"]),
                keypoint_params=A.KeypointParams(format="xy", label_fields=["class_labels_2"]),
            ),
            {
                "image": np.empty([100, 100, 3], dtype=np.uint8),
                "mask": np.empty([100, 100, 3], dtype=np.uint8),
                "bboxes": [[0.5, 0.5, 0.1, 0.1]],
                "class_labels_1": [1],
                "keypoints": [[10, 20]],
                "class_labels_2": [1],
            },
        ],
    ],
)
def test_common_pipeline_validity(transforms: list, compose_args: dict, args: dict):
    # Just check that everything is fine - no errors

    pipeline = A.Compose(transforms, **compose_args)

    res = pipeline(**args)
    for k in args:
        assert k in res


def test_compose_non_available_keys() -> None:
    """Check that non available keys raises error, except `mask` and `masks`"""
    mock_transform = MagicMock()
    mock_transform.available_keys = {"image"}
    mock_transform.invalid_args = []  # Add this line to set up _invalid_args

    transform = A.Compose(
        [mock_transform],
        strict=True,
        seed=137
    )

    image = np.empty([10, 10, 3], dtype=np.uint8)
    mask = np.empty([10, 10], dtype=np.uint8)
    _ = transform(image=image, mask=mask)
    _ = transform(image=image, masks=[mask])
    with pytest.raises(ValueError) as exc_info:
        _ = transform(image=image, image_2=mask)

    expected_msg = "Key image_2 is not in available keys."
    assert str(exc_info.value) == expected_msg

    # strict=False should not raise error
    transform = A.Compose(
        [MagicMock(available_keys={"image"})],
        strict=False,
    )
    _ = transform(image=image, mask=mask)
    _ = transform(image=image, masks=[mask])
    _ = transform(image=image, image_2=mask)


def test_compose_additional_targets_in_available_keys() -> None:
    """Check whether `available_keys` always contains everything in `additional_targets`"""
    first = MagicMock(available_keys={"image"})
    second = MagicMock(available_keys={"image"})
    image = np.ones((8, 8))

    # non-empty `transforms`
    augmentation = Compose(
        [first, second],
        p=1,
        additional_targets={"additional_target_1": "image", "additional_target_2": "image"},
        strict=False,
    )
    augmentation(image=image, additional_target_1=image, additional_target_2=image)  # will raise exception if not
    # strict=False should not raise error without additional_targets
    augmentation = Compose([first, second], p=1, strict=False)
    augmentation(image=image, additional_target_1=image, additional_target_2=image)

    # empty `transforms`
    augmentation = Compose([], p=1, additional_targets={"additional_target_1": "image", "additional_target_2": "image"}, strict=True)
    augmentation(image=image, additional_target_1=image, additional_target_2=image)  # will raise exception if not
    # strict=False should not raise error without additional_targets
    augmentation = Compose([], p=1, strict=False)
    augmentation(image=image, additional_target_1=image, additional_target_2=image)



@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        custom_arguments={
        },
        except_augmentations={
            A.Lambda,
            A.RandomSizedBBoxSafeCrop,
            A.CropNonEmptyMaskIfExists,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.TextImage,
            A.RandomCropNearBBox,
            A.Pad,
            A.Mosaic,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
        },
    ),
)
@pytest.mark.parametrize("as_array", [True, False])
@pytest.mark.parametrize("shape", [(101, 99, 3), (101, 99)])
def test_images_as_target(augmentation_cls, params, as_array, shape):
    if len(shape) == 2:
        if augmentation_cls in {A.ChannelDropout, A.Spatter, A.ISONoise,
                                A.RandomGravel, A.ChromaticAberration, A.PlanckianJitter, A.PixelDistributionAdaptation,
                                A.MaskDropout, A.ConstrainedCoarseDropout, A.ChannelShuffle, A.ToRGB, A.RandomSunFlare,
                                A.RandomFog, A.RandomSnow, A.RandomRain, A.HEStain}:
            pytest.skip(f"{augmentation_cls.__name__} is not applicable to grayscale images")

        if "fill" in params and not np.isscalar(params["fill"]):
            params["fill"] = params["fill"][0]

    image = np.random.uniform(0, 255, shape).astype(np.float32) if augmentation_cls == A.FromFloat else np.random.randint(0, 255, shape, dtype=np.uint8)


    if as_array:
        # Stack images into a single array
        images = np.stack([image] * 2)
        data = {"images": images}
    else:
        # Original list format
        data = {"images": [image] * 2}

    if augmentation_cls == A.MaskDropout or augmentation_cls == A.ConstrainedCoarseDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask

    aug = A.Compose(
        [augmentation_cls(p=1, **params)],
        p=1,
        strict=True,
        seed=137,
    )

    transformed = aug(**data)


    # Check both images were transformed identically
    np.testing.assert_allclose(transformed["images"][0], transformed["images"][1])

    # Check output format matches input format
    if as_array:
        assert isinstance(transformed["images"], np.ndarray)

        assert transformed["images"].ndim == len(shape) + 1, f"Expected {len(shape) + 1} dimensions, got {transformed['images'].ndim}"

        assert transformed["images"].flags["C_CONTIGUOUS"]  # Ensure memory is contiguous

        # Verify exact shape matches expected dimensions
        N, H, W = transformed["images"].shape[:3]
        assert N == 2  # Two images as input
        if len(shape) == 3:
            assert transformed["images"].shape[-1] == image.shape[2]  # Channels match input

        if augmentation_cls not in [A.RandomCrop, A.AtLeastOneBBoxRandomCrop, A.RandomResizedCrop, A.Resize, A.RandomSizedCrop, A.RandomSizedBBoxSafeCrop,
                                    A.BBoxSafeRandomCrop, A.Transpose, A.RandomCropNearBBox, A.CenterCrop, A.Crop, A.CropAndPad,
                                    A.LongestMaxSize, A.RandomScale, A.PadIfNeeded, A.SmallestMaxSize, A.RandomCropFromBorders,
                                    A.RandomRotate90, A.D4, A.SquareSymmetry]:
            assert H == image.shape[0]  # Height matches input
            assert W == image.shape[1]  # Width matches input
    else:
        assert isinstance(transformed["images"], list)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        except_augmentations={
            A.RandomCropNearBBox,
        },
    ),
)
def test_non_contiguous_input_with_compose(augmentation_cls, params, bboxes):
    image = np.ones([3, 100, 100], dtype=np.uint8).transpose(1, 2, 0)
    mask = np.zeros([3, 100, 100], dtype=np.uint8).transpose(1, 2, 0)

    # check preconditions
    assert not image.flags["C_CONTIGUOUS"]
    assert not mask.flags["C_CONTIGUOUS"]

    data = {
        "image": image,
        "mask": mask,
    }

    if augmentation_cls == A.RandomCropNearBBox:
        # requires "cropping_bbox" arg
        aug = A.Compose([augmentation_cls(p=1, **params)], strict=True, seed=137)

        data["cropping_bbox"] = bboxes[0]
    elif augmentation_cls in [A.RandomSizedBBoxSafeCrop, A.BBoxSafeRandomCrop]:
        # requires "bboxes" arg
        aug = A.Compose([augmentation_cls(p=1, **params)], bbox_params=A.BboxParams(format="pascal_voc"), strict=True, seed=137)
        data["bboxes"] = bboxes
    elif augmentation_cls == A.TextImage:
        aug = A.Compose([augmentation_cls(p=1, **params)], bbox_params=A.BboxParams(format="pascal_voc"), strict=True, seed=137)
        data["textimage_metadata"] = {"text": "Hello, world!", "bbox": (0.1, 0.1, 0.9, 0.2)}
    elif augmentation_cls == A.OverlayElements:
        # requires "metadata" arg
        aug = A.Compose([augmentation_cls(p=1, **params)], strict=True, seed=137)
        data["overlay_metadata"] = []
    elif augmentation_cls == A.Mosaic:
        aug = A.Compose([augmentation_cls(p=1, **params)], strict=True, seed=137)
        data["mosaic_metadata"] = [
                {
                    "image": image,
                    "mask": mask,
                }
            ]
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]
        aug = A.Compose([augmentation_cls(p=1, **params)], p=1, strict=True, seed=137)
    else:
        # standard args: image and mask
        if augmentation_cls == A.FromFloat:
            # requires float image
            image = (image / 255).astype(np.float32)
            assert not image.flags["C_CONTIGUOUS"]
        elif augmentation_cls == A.MaskDropout or augmentation_cls == A.ConstrainedCoarseDropout:
            # requires single channel mask
            mask = mask[:, :, 0]

        aug = A.Compose([augmentation_cls(p=1, **params)], p=1, strict=True, seed=137)

    transformed = aug(**data)

    assert transformed["image"].flags["C_CONTIGUOUS"], f"{augmentation_cls.__name__} did not return a C_CONTIGUOUS image"

    # Check if the augmentation is not an ImageOnlyTransform and mask is in the output
    if not issubclass(augmentation_cls, ImageOnlyTransform) and "mask" in transformed:
        assert transformed["mask"].flags[
            "C_CONTIGUOUS"
        ], f"{augmentation_cls.__name__} did not return a C_CONTIGUOUS mask"


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        except_augmentations={
            A.Lambda,
            A.RandomSizedBBoxSafeCrop,
            A.CropNonEmptyMaskIfExists,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.TextImage,
            A.FromFloat,
            A.MaskDropout,
            A.RandomCropNearBBox,
            A.PadIfNeeded,
            A.Mosaic,
        },
    ),
)
@pytest.mark.parametrize(
    "masks",
    [
        [np.random.randint(0, 2, [100, 100], dtype=np.uint8)] * 3,
        [np.random.randint(0, 2, [100, 100, 3], dtype=np.uint8)] * 3,
        np.stack([np.random.randint(0, 2, [100, 100], dtype=np.uint8)] * 3),
    ],
)
def test_masks_as_target(augmentation_cls, params, masks):
    image = SQUARE_UINT8_IMAGE

    data = {
        "image": image,
        "masks": masks,
    }

    aug = A.Compose(
        [augmentation_cls(p=1, **params)],
        seed=42,
        strict=True,
    )

    transformed = aug(**data)

    np.testing.assert_array_equal(transformed["masks"][0], transformed["masks"][1])

    assert transformed["masks"][0].dtype == masks[0].dtype


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
        },
        except_augmentations={
            A.PixelDropout,
            A.RandomCrop,
            A.Crop,
            A.CenterCrop,
            A.FDA,
            A.HistogramMatching,
            A.Lambda,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.TextImage,
            A.FromFloat,
            A.MaskDropout,
            A.XYMasking,
            A.TimeMasking,
            A.FrequencyMasking,
            A.Erasing,
            A.RandomCropNearBBox,
            A.GridDropout,
            A.CoarseDropout,
            A.ConstrainedCoarseDropout,
            A.RandomRotate90,
            A.D4,
            A.HorizontalFlip,
            A.VerticalFlip,
            A.Transpose,
            A.NoOp,
            A.RandomSizedBBoxSafeCrop,
            A.RandomRotate90,
            A.TimeReverse,
            A.TimeMasking,
            A.Mosaic,
        },
    ),
)
@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT
            ])
def test_mask_interpolation(augmentation_cls, params, interpolation, image):
    mask = image.copy()
    if augmentation_cls in {A.Affine, A.GridElasticDeform,
    A.SafeRotate,
    A.ShiftScaleRotate,
    A.OpticalDistortion,
    A.ThinPlateSpline,
    A.Perspective,
    A.ElasticTransform,
    A.GridDistortion,
    A.PiecewiseAffine,
    A.CropAndPad,
    A.LongestMaxSize,
    A.SmallestMaxSize,
    A.RandomResizedCrop,
    A.RandomScale,
    A.Rotate
    } and interpolation in {cv2.INTER_NEAREST_EXACT, cv2.INTER_LINEAR_EXACT}:
        return

    params["interpolation"] = interpolation
    params["mask_interpolation"] = interpolation
    params["border_mode"] = cv2.BORDER_CONSTANT
    params["fill"] = 10
    params["fill_mask"] = 10

    aug = A.Compose([augmentation_cls(**params, p=1)], seed=137, strict=False)

    transformed = aug(image=image, mask=mask)

    assert transformed["mask"].flags["C_CONTIGUOUS"]

    np.testing.assert_array_equal(transformed["mask"], transformed["image"])



@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST,
                                                cv2.INTER_LINEAR,
                                                cv2.INTER_CUBIC,
                                                cv2.INTER_AREA
                                                ])
@pytest.mark.parametrize("compose", [A.Compose, A.OneOf, A.Sequential, A.SomeOf])
def test_mask_interpolation_someof(interpolation, compose):
    transform = A.Compose([compose([A.Affine(p=1), A.RandomSizedCrop(min_max_height=(4, 8), size= (113, 103), p=1)], p=1)], mask_interpolation=interpolation, strict=True)

    image = SQUARE_UINT8_IMAGE
    mask = image.copy()

    transformed = transform(image=image, mask=mask)

    assert transformed["mask"].flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ["transform", "expected_param_keys"],
    [
        (A.HorizontalFlip(p=1), {"shape"}),
        (A.VerticalFlip(p=1), {"shape"}),
        (
            A.RandomBrightnessContrast(p=1),
            {"shape", "alpha", "beta"}
        ),
        (
            A.Rotate(p=1),
            {'shape', 'x_min', 'x_max', 'y_min', 'y_max', 'matrix', 'bbox_matrix', 'interpolation', "fill", "fill_mask"}
        ),
    ],
)
def test_transform_returns_params(transform, expected_param_keys):
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    transform(image=image)
    params = transform.get_applied_params()
    assert isinstance(params, dict)
    assert set(params.keys()) == expected_param_keys


@pytest.mark.parametrize(
    ["transforms", "expected_names"],
    [
        # Simple sequential transforms
        (
            [A.HorizontalFlip(p=1), A.Blur(p=1)],
            ["HorizontalFlip", "Blur"]
        ),
        # OneOf inside Compose
        (
            [
                A.OneOf([
                    A.HorizontalFlip(p=1),
                    A.VerticalFlip(p=1)
                ], p=1),
                A.Blur(p=1)
            ],
            ["HorizontalFlip|VerticalFlip", "Blur"]  # One of these will be applied
        ),
        # Nested Sequential
        (
            [
                A.Sequential([
                    A.HorizontalFlip(p=1),
                    A.Blur(p=1)
                ], p=1),
                A.RandomBrightnessContrast(p=1)
            ],
            ["HorizontalFlip", "Blur", "RandomBrightnessContrast"]
        ),
        # Complex nesting
        (
            [
                A.OneOf([
                    A.Sequential([
                        A.HorizontalFlip(p=1),
                        A.Blur(p=1)
                    ], p=1),
                    A.Sequential([
                        A.VerticalFlip(p=1),
                        A.RandomBrightnessContrast(p=1)
                    ], p=1)
                ], p=1)
            ],
            ["HorizontalFlip,Blur|VerticalFlip,RandomBrightnessContrast"]  # One sequence will be applied
        ),
    ]
)
def test_transform_tracking(image, transforms, expected_names):
    transform = A.Compose(transforms, p=1, save_applied_params=True, strict=True)
    result = transform(image=image)

    assert "applied_transforms" in result
    applied_names = [t[0] for t in result["applied_transforms"]]

    if "|" in expected_names[0]:
        # Handle OneOf case where one of multiple possibilities will be applied
        possible_names = expected_names[0].split("|")
        if "," in possible_names[0]:
            # Handle nested sequence case
            possible_sequences = [sequence.split(",") for sequence in possible_names]
            assert applied_names in possible_sequences
        else:
            assert applied_names[0] in possible_names
            assert len(applied_names) == len(expected_names)
    else:
        assert applied_names == expected_names

@pytest.mark.parametrize(
    ["transform_class", "transform_params"],
    [
        (A.Blur, {"blur_limit": 3}),
        (A.RandomBrightnessContrast, {"brightness_limit": 0.2, "contrast_limit": 0.2}),
        (A.HorizontalFlip, {}),
    ]
)
def test_params_content(image, transform_class, transform_params):
    transform = A.Compose([transform_class(p=1, **transform_params)], save_applied_params=True, strict=True)
    result = transform(image=image)

    assert len(result["applied_transforms"]) == 1
    transform_name, params = result["applied_transforms"][0]

    assert transform_name == transform_class.__name__

def test_no_param_tracking():
    """Test that params are not tracked when save_applied_params=False"""
    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.Blur(p=1)
    ], p=1, save_applied_params=False, strict=True)

    result = transform(image=np.zeros((100, 100, 3), dtype=np.uint8))
    assert "applied_transforms" not in result

def test_probability_control():
    """Test that transforms are only tracked when they are actually applied"""
    transform = A.Compose([
        A.HorizontalFlip(p=0),  # Will not be applied
        A.Blur(p=1)  # Will be applied
    ], p=1, save_applied_params=True, strict=True)

    result = transform(image=np.zeros((100, 100, 3), dtype=np.uint8))
    applied_names = [t[0] for t in result["applied_transforms"]]
    assert "HorizontalFlip" not in applied_names
    assert "Blur" in applied_names

def test_compose_probability():
    """Test that no transforms are tracked when compose probability is 0"""
    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.Blur(p=1)
    ], p=0, save_applied_params=True, strict=True)

    result = transform(image=np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(result["applied_transforms"]) == 0


@pytest.mark.parametrize(
    ["data", "expected_shape"],
    [
        # Test numpy image formats
        (
            {"image": np.zeros((100, 200, 3))},  # HWC
            {"height": 100, "width": 200},
        ),
        (
            {"image": np.zeros((100, 200))},  # HW
            {"height": 100, "width": 200},
        ),
        (
            {"images": [np.zeros((100, 200, 3)), np.zeros((100, 200, 3))]},  # N×HWC
            {"height": 100, "width": 200},
        ),

        # Test torch image formats
        (
            {"image": torch.zeros(3, 100, 200)},  # CHW
            {"height": 100, "width": 200},
        ),
        (
            {"image": torch.zeros(1, 100, 200)},  # 1HW
            {"height": 100, "width": 200},
        ),
        (
            {"images": torch.zeros(5, 3, 100, 200)},  # NCHW
            {"height": 100, "width": 200},
        ),

        # Test numpy volume formats
        (
            {"volume": np.zeros((50, 100, 200, 3))},  # DHWC
            {"depth": 50, "height": 100, "width": 200},
        ),
        (
            {"volume": np.zeros((50, 100, 200))},  # DHW
            {"depth": 50, "height": 100, "width": 200},
        ),

        # Test torch volume formats
        (
            {"volume": torch.zeros(3, 50, 100, 200)},  # CDHW
            {"depth": 50, "height": 100, "width": 200},
        ),
        (
            {"volume": torch.zeros(1, 50, 100, 200)},  # 1DHW
            {"depth": 50, "height": 100, "width": 200},
        ),
    ],
)
def test_get_shape(data, expected_shape):
    assert get_shape(data) == expected_shape


@pytest.mark.parametrize(
    ["data", "error_type", "error_message"],
    [
        (
            {},
            ValueError,
            "No image or volume found in data",
        ),
        (
            {"wrong_key": np.zeros((100, 200))},
            ValueError,
            "No image or volume found in data",
        ),
        (
            {"image": "not_an_array"},
            RuntimeError,
            "Unsupported image type: <class 'str'>",
        ),
        (
            {"volume": "not_an_array"},
            RuntimeError,
            "Unsupported volume type: <class 'str'>",
        ),
    ],
)
def test_get_shape_errors(data, error_type, error_message):
    with pytest.raises(error_type, match=error_message):
        get_shape(data)


@pytest.mark.parametrize(
    "key", ["image", "images", "volume"]
)
def test_get_shape_empty_arrays(key):
    # Test that empty arrays don't cause issues
    if key == "images":
        data = {key: [np.zeros((0, 0, 3))]}
    else:
        data = {key: np.zeros((0, 0, 3))}

    shape = get_shape(data)
    assert isinstance(shape, dict)
    assert all(isinstance(v, int) for v in shape.values())


def test_transform_strict_mode_raises_error():
    # Test that strict=True raises error for invalid parameters
    with pytest.raises(ValueError, match="Argument\\(s\\) 'invalid_param' are not valid for transform Blur"):
       A.Blur(strict=True, invalid_param=123)

def test_transform_non_strict_mode_shows_warning():
    # Test that strict=False (default) shows warning for invalid parameters
    with pytest.warns(UserWarning, match="Argument\\(s\\) 'invalid_param' are not valid for transform Blur"):
        transform = A.Blur(invalid_param=123)
        assert transform.p == 0.5  # Check that transform was still created with default values

def test_transform_valid_params_no_warning():
    # Test that no warning/error is raised for valid parameters
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Convert warnings to errors to ensure none are raised
        transform = A.Blur(p=0.7, blur_limit=(3, 5))
        assert transform.p == 0.7
        assert transform.blur_limit == (3, 5)

def test_transform_multiple_invalid_params():
    # Test handling of multiple invalid parameters
    with pytest.raises(ValueError, match="Argument\\(s\\) 'invalid1, invalid2' are not valid for transform Blur"):
        A.Blur(strict=True, invalid1=123, invalid2=456)

def test_transform_strict_with_valid_params():
    # Test that strict mode doesn't affect valid parameters
    transform = A.Blur(strict=True, p=0.7, blur_limit=(3, 5))
    assert transform.p == 0.7
    assert transform.blur_limit == (3, 5)


@pytest.mark.parametrize(
    ["labels", "expected_type", "expected_dtype"],
    [
        # Numpy arrays should stay numpy arrays
        (np.array([1, 2, 3], dtype=np.int32), np.ndarray, np.int32),
        (np.array([1, 2, 3], dtype=np.int64), np.ndarray, np.int64),
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), np.ndarray, np.float32),
        (np.array([1.0, 2.0, 3.0], dtype=np.float64), np.ndarray, np.float64),
        # Lists should stay lists
        ([1, 2, 3], list, None),
        ([1.0, 2.0, 3.0], list, None),
    ],
)
def test_label_type_preservation(labels, expected_type, expected_dtype):
    """Test that both type (list/ndarray) and dtype are preserved."""
    transform = Compose(
        [NoOp(p=1.0)],
        bbox_params=BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ),
        strict=True,
    )

    transformed = transform(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        bboxes=[(0, 0, 10, 10), (10, 10, 20, 20), (20, 20, 30, 30)],
        labels=labels
    )

    result_labels = transformed['labels']
    assert isinstance(result_labels, expected_type)
    if expected_dtype is not None:
        assert result_labels.dtype == expected_dtype
    if expected_type == list:
        assert result_labels == labels
    else:
        np.testing.assert_array_equal(result_labels, labels)


def test_string_labels():
    # Create sample data
    bboxes = [(0, 0, 10, 10), (10, 10, 20, 20), (20, 20, 30, 30)]
    labels = ['cat', 'dog', 'bird']

    transform = Compose(
        [NoOp(p=1.0)],
        bbox_params=BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ),
        strict=True,
    )

    transformed = transform(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        bboxes=bboxes,
        labels=labels
    )

    # Check that string labels are preserved exactly
    assert transformed['labels'] == labels


def test_empty_labels():
    transform = Compose(
        [NoOp(p=1.0)],
        bbox_params=BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        ),
        strict=True,
    )

    transformed = transform(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        bboxes=[],
        labels=[]
    )

    assert transformed['labels'] == []



@pytest.mark.parametrize(
    ["transforms_config", "strict", "should_raise"],
    [
        # Valid parameters, no error expected
        (
            [
                NoOp(p=0.5),
                OneOf([NoOp(p=0.7)], p=1.0),
                Sequential([NoOp(p=0.3)], p=1.0),
            ],
            True,
            False,
        ),
        # Invalid param in root level, should raise with strict=True
        (
            [
                NoOp(p=0.5, invalid_param=123),
                OneOf([NoOp(p=0.7)], p=1.0),
                Sequential([NoOp(p=0.3)], p=1.0),
            ],
            True,
            True,
        ),
        # Invalid param in OneOf, should raise with strict=True
        (
            [
                NoOp(p=0.5),
                OneOf([NoOp(p=0.7, invalid_param=123)], p=1.0),
                Sequential([NoOp(p=0.3)], p=1.0),
            ],
            True,
            True,
        ),
        # Multiple invalid params, should raise with strict=True
        (
            [
                NoOp(p=0.5, invalid1=123),
                OneOf([NoOp(p=0.7, invalid2=456)], p=1.0),
                Sequential([NoOp(p=0.3, invalid3=789)], p=1.0),
            ],
            True,
            True,
        ),
        # Invalid params but strict=False, should only warn
        (
            [
                NoOp(p=0.5, invalid1=123),
                OneOf([NoOp(p=0.7, invalid2=456)], p=1.0),
                Sequential([NoOp(p=0.3, invalid3=789)], p=1.0),
            ],
            False,
            False,
        ),
    ],
)
def test_strict_validation_in_compose(
    transforms_config: list[Any],
    strict: bool,
    should_raise: bool,
) -> None:
    """Test that strict parameter properly validates unknown parameters."""
    if should_raise:
        with pytest.raises(ValueError, match="are not valid for transform"):
            Compose(transforms_config, strict=strict)
    else:
        with warnings.catch_warnings(record=True) as w:
            transform = Compose(transforms_config, strict=strict)
            if not strict and any("invalid" in str(t) for t in transforms_config):
                assert len(w) > 0
                assert any("are not valid for transform" in str(warn.message) for warn in w)


def test_transform_strict_mode_raises_error():
    # Test that strict=True raises error for invalid parameters
    with pytest.raises(ValueError, match="Argument\\(s\\) 'invalid_param' are not valid for transform Blur"):
       A.Blur(strict=True, invalid_param=123)

def test_transform_non_strict_mode_shows_warning():
    # Test that strict=False (default) shows warning for invalid parameters
    with pytest.warns(UserWarning, match="Argument\\(s\\) 'invalid_param' are not valid for transform Blur"):
        transform = A.Blur(invalid_param=123)
        assert transform.p == 0.5  # Check that transform was still created with default values

def test_transform_valid_params_no_warning():
    # Test that no warning/error is raised for valid parameters
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Convert warnings to errors to ensure none are raised
        transform = A.Blur(p=0.7, blur_limit=(3, 5))
        assert transform.p == 0.7
        assert transform.blur_limit == (3, 5)

def test_transform_multiple_invalid_params():
    # Test handling of multiple invalid parameters
    with pytest.raises(ValueError, match="Argument\\(s\\) 'invalid1, invalid2' are not valid for transform Blur"):
        A.Blur(strict=True, invalid1=123, invalid2=456)

def test_transform_strict_with_valid_params():
    # Test that strict mode doesn't affect valid parameters
    transform = A.Blur(strict=True, p=0.7, blur_limit=(3, 5))
    assert transform.p == 0.7
    assert transform.blur_limit == (3, 5)



@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
        },
        except_augmentations={
            A.PixelDropout,
            A.RandomCrop,
            A.Crop,
            A.CenterCrop,
            A.FDA,
            A.HistogramMatching,
            A.Lambda,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.TextImage,
            A.FromFloat,
            A.MaskDropout,
            A.XYMasking,
            A.TimeMasking,
            A.FrequencyMasking,
            A.Erasing,
            A.RandomCropNearBBox,
            A.GridDropout,
            A.CoarseDropout,
            A.ConstrainedCoarseDropout,
            A.RandomRotate90,
            A.D4,
            A.HorizontalFlip,
            A.VerticalFlip,
            A.Transpose,
            A.NoOp,
            A.RandomSizedBBoxSafeCrop,
            A.RandomRotate90,
            A.TimeReverse,
            A.TimeMasking,
            A.ThinPlateSpline,
            A.ElasticTransform,
            A.PiecewiseAffine,
            A.ShiftScaleRotate,
            A.RandomScale,
            A.Resize,
            A.RandomResizedCrop,
            A.RandomGridShuffle,
            A.OpticalDistortion,
            A.Morphological,
            A.AtLeastOneBBoxRandomCrop,
            A.Mosaic,
        },
    ),
)
@pytest.mark.parametrize("border_mode", [
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
            cv2.BORDER_REFLECT101,
        ])
def test_mask_interpolation(augmentation_cls, params, border_mode, image):

    mask = image.copy()

    params["interpolation"] = cv2.INTER_LINEAR
    params["mask_interpolation"] = cv2.INTER_LINEAR
    params["border_mode"] = border_mode
    params["fill"] = 10
    params["fill_mask"] = 10

    transform = A.Compose([augmentation_cls(**params, p=1)], seed=137, strict=False)

    transform(image=image, mask=mask)


@pytest.mark.parametrize(
    "params, strict, expected_outcome, expected_error_params",
    [
        # Valid cases
        ({"rotate": 45}, False, "valid", []),
        ({"rotate": 45, "p": 0.5}, False, "valid", []),

        # Invalid parameter names (affected by strict)
        ({"rotate": 45, "invalid_param": 123}, False, "warning", []),
        ({"rotate": 45, "invalid_param": 123}, True, "error", ["invalid_param"]),
        ({"rotate": 45, "wrong_param": 0.5, "bad_param": 30}, False, "warning", []),

        # Invalid parameter values (always error, regardless of strict)
        ({"rotate": 45, "p": 1.5}, False, "value_error", ["p"]),
        ({"rotate": 45, "p": -0.5}, False, "value_error", ["p"]),
        # Multiple invalid values
        ({"interpolation": -1, "mask_interpolation": -1, "p": 1.5}, False, "value_error",
         ["interpolation", "mask_interpolation", "p"]),
    ]
)
def test_affine_invalid_parameters(params, strict, expected_outcome, expected_error_params):
    if expected_outcome == "valid":
        transform = A.Affine(**params)
        assert transform is not None
        assert not hasattr(transform, 'invalid_args') or not transform.invalid_args

    elif expected_outcome == "warning":
        transform = A.Affine(strict=strict, **params)
        assert hasattr(transform, 'invalid_args')
        invalid_params = set(params.keys()) - {"rotate", "p", "scale", "translate_percent",
                                             "translate_px", "interpolation", "mask_interpolation",
                                             "mode", "fit_output", "keep_ratio"}
        assert set(transform.invalid_args) == invalid_params

    elif expected_outcome == "error":
        with pytest.raises(ValueError) as excinfo:
            A.Affine(strict=strict, **params)
        error_msg = str(excinfo.value)
        for param in expected_error_params:
            assert param in error_msg

    elif expected_outcome == "value_error":
        with pytest.raises(ValueError) as excinfo:
            A.Affine(strict=strict, **params)
        error_msg = str(excinfo.value)

        # Verify that ALL expected error parameters are in the message
        for param in expected_error_params:
            assert param in error_msg

        if len(expected_error_params) > 1:
            # Count unique parameters mentioned in the error
            error_params = {param for param in expected_error_params if param in error_msg}

            assert len(error_params) == len(expected_error_params), \
                f"Expected validation errors for {expected_error_params}, got errors for {error_params}"

@pytest.mark.parametrize(
    ["bbox_format", "bboxes"],
    [
        ("coco", [[15, 12, 30, 40], [50, 50, 15, 40]]),
        ("pascal_voc", [[15, 12, 45, 52], [50, 50, 65, 90]]),
        ("albumentations", [[0.15, 0.12, 0.45, 0.52], [0.5, 0.5, 0.65, 0.9]]),
        ("yolo", [[(15 + 30 / 2) / 100, (12 + 40 / 2) / 100, 30 / 100, 40 / 100],
                  [(50 + 15 / 2) / 100, (50 + 40 / 2) / 100, 15 / 100, 40 / 100]]),
    ],
)
def test_bbox_hflip_hflip_no_labels(bbox_format: str, bboxes: list[list[float]]):
    """Check applying HorizontalFlip twice returns the original bboxes without labels."""
    image = np.ones((100, 100, 3))
    original_bboxes = np.array(bboxes, dtype=np.float32)

    aug = A.Compose(
        [A.HorizontalFlip(p=1.0), A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format=bbox_format), # No label_fields specified
        strict=True,
    )
    transformed = aug(image=image, bboxes=original_bboxes)

    assert np.allclose(transformed["bboxes"], original_bboxes, atol=1e-6)


@pytest.mark.parametrize(
    ["kp_format", "keypoints"],
    [
        ("xy", [[15, 12], [50, 50]]),  # Standard (x, y)
        ("yx", [[12, 15], [50, 50]]),  # Reversed (y, x)
        ("xya", [[15, 12, 90], [50, 50, 45]]),  # With angle
        ("xys", [[15, 12, 1.5], [50, 50, 0.8]]),  # With scale
        ("xyz", [[15, 12, 5], [50, 50, 10]]), # With z-coordinate
    ],
)
def test_keypoint_hflip_hflip_no_labels(kp_format: str, keypoints: list[list[float]]):
    """Check applying HorizontalFlip twice returns the original keypoints without labels."""
    image = np.ones((100, 100, 3))
    original_keypoints = np.array(keypoints, dtype=np.float32)

    aug = A.Compose(
        [A.HorizontalFlip(p=1.0), A.HorizontalFlip(p=1.0)],
        keypoint_params=A.KeypointParams(format=kp_format), # No label_fields specified
        strict=True,
    )
    transformed = aug(image=image, keypoints=original_keypoints)

    assert np.allclose(transformed["keypoints"], original_keypoints, atol=1e-6)


def test_compose_with_empty_masks():
    """Test that Compose can handle empty masks list."""
    transform = Compose([
        A.Resize(288, 384),
        A.ToFloat(max_value=255)
    ])
    image = np.zeros((288, 384, 3), dtype=np.uint8)
    result = transform(image=image, masks=[])
    # Verify that the result contains an empty masks list
    assert "masks" in result
    assert isinstance(result["masks"], (list, tuple))
    assert len(result["masks"]) == 0
