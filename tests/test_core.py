import typing
from unittest import mock
from unittest.mock import MagicMock, Mock, call, patch

import cv2
import numpy as np
import pytest

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
from albumentations.core.utils import to_tuple
from tests.conftest import (
    IMAGES,
    RECTANGULAR_FLOAT_IMAGE,
    RECTANGULAR_UINT8_IMAGE,
    SQUARE_FLOAT_IMAGE,
    SQUARE_UINT8_IMAGE,
)

from .utils import get_dual_transforms, get_filtered_transforms, get_image_only_transforms, get_transforms, set_seed


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


def oneof_always_apply_crash():
    aug = Compose([A.HorizontalFlip(p=1), A.Rotate(p=1), A.OneOf([A.Blur(p=1), A.MedianBlur(p=1)], p=1)], p=1)
    image = np.ones((8, 8))
    data = aug(image=image)
    assert data


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
                cols=width,
                rows=height,
                shape=image.shape,
            )
            np.testing.assert_array_equal(data["mask"], mask)


@pytest.mark.parametrize("image", IMAGES)
def test_dual_transform(image):
    mask = image.copy()
    image_call = call(
        image,
        interpolation=cv2.INTER_LINEAR,
        cols=image.shape[1],
        rows=image.shape[0],
        shape=image.shape,
    )
    mask_call = call(
        mask,
        interpolation=cv2.INTER_NEAREST,
        cols=mask.shape[1],
        rows=mask.shape[0],
        shape=mask.shape,
    )
    with mock.patch.object(DualTransform, "apply") as mocked_apply:
        with mock.patch.object(DualTransform, "get_params", return_value={"interpolation": cv2.INTER_LINEAR}):
            aug = DualTransform(p=1)
            aug(image=image, mask=mask)
            mocked_apply.assert_has_calls([image_call, mask_call], any_order=True)


@pytest.mark.parametrize("image", IMAGES)
def test_additional_targets(image):
    mask = image.copy()
    image_call = call(
        image,
        interpolation=cv2.INTER_LINEAR,
        cols=image.shape[1],
        rows=image.shape[0],
        shape=image.shape,
    )
    image2_call = call(
        mask,
        interpolation=cv2.INTER_LINEAR,
        cols=mask.shape[1],
        rows=mask.shape[0],
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
    aug = Compose([A.NoOp()], additional_targets=additional_targets)

    with pytest.raises(TypeError) as exc_info:
        aug(**targets)
    assert str(exc_info.value) == err_message

    aug = Compose([A.NoOp()])
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
        [A.Crop(0, 0, 50, 50), A.PadIfNeeded(100, 100, border_mode=cv2.BORDER_CONSTANT, value=0)],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
    )
    res = augs(image=image, **targets)

    for key, item in expected.items():
        assert np.all(np.array(item) == np.array(res[key]))


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
        [Compose([A.Crop(0, 0, 50, 50), A.PadIfNeeded(100, 100, border_mode=cv2.BORDER_CONSTANT, value=0)])],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
    )
    res = augs(image=image, **targets)

    for key, item in expected.items():
        assert np.all(np.array(item) == np.array(res[key]))


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
        [Sequential([A.Crop(0, 0, 50, 50), A.PadIfNeeded(100, 100, border_mode=cv2.BORDER_CONSTANT, value=0)], p=1.0)],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
    )
    res = augs(image=image, **targets)

    for key, item in expected.items():
        assert np.all(np.array(item) == np.array(res[key]))


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
            SomeOf([A.PadIfNeeded(100, 100, border_mode=cv2.BORDER_CONSTANT, value=0)], n=1, replace=False, p=1.0),
        ],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
    )
    res = augs(image=image, **targets)

    for key, item in expected.items():
        assert np.all(np.array(item) == np.array(res[key]))


@pytest.mark.parametrize("image", IMAGES)
def test_bbox_params_is_not_set(image, bboxes):
    t = Compose([A.NoOp(p=1.0)])
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
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"size": (10, 10)},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "size": (10, 10)},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": 3,
                "mask_x_length": (10, 20),
                "mask_y_length": 10,
                "fill_value": 0,
                "mask_fill_value": 1,
            },
            A.PadIfNeeded: {
                "min_height": 512,
                "min_width": 512,
                "border_mode": cv2.BORDER_CONSTANT,
                "value": [124, 116, 104],
                "position": "top_left",
            },
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
        },
        except_augmentations={
            A.FDA,
            A.HistogramMatching,
            A.Lambda,
            A.TemplateTransform,
            A.RandomSizedBBoxSafeCrop,
            A.CropNonEmptyMaskIfExists,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.TextImage,
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
    if augmentation_cls == A.MaskDropout:
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
            A.FDA,
            A.HistogramMatching,
            A.Lambda,
            A.RandomSizedBBoxSafeCrop,
            A.CropNonEmptyMaskIfExists,
            A.OverlayElements,
            A.TextImage,
            A.FromFloat,
        },
        custom_arguments={
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
            A.TemplateTransform: {
                "templates": np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
            },
        },
    ),
)
def test_contiguous_output_imageonly(augmentation_cls, params):
    set_seed(42)
    image = np.zeros([3, 100, 100], dtype=np.uint8).transpose(1, 2, 0)

    # check preconditions
    assert not image.flags["C_CONTIGUOUS"]

    transform = augmentation_cls(p=1, **params)

    # pipeline always outputs contiguous results
    data = transform(image=image)

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
    transform = A.Compose(
        [MagicMock(available_keys={"image"})],
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
    )
    augmentation(image=image, additional_target_1=image, additional_target_2=image)  # will raise exception if not
    # strict=False should not raise error without additional_targets
    augmentation = Compose([first, second], p=1, strict=False)
    augmentation(image=image, additional_target_1=image, additional_target_2=image)

    # empty `transforms`
    augmentation = Compose([], p=1, additional_targets={"additional_target_1": "image", "additional_target_2": "image"})
    augmentation(image=image, additional_target_1=image, additional_target_2=image)  # will raise exception if not
    # strict=False should not raise error without additional_targets
    augmentation = Compose([], p=1, strict=False)
    augmentation(image=image, additional_target_1=image, additional_target_2=image)


def test_transform_always_apply_warning() -> None:
    """Check that warning is raised if always_apply argument is used"""
    warning_expected = (
        "always_apply is deprecated. Use `p=1` if you want to always apply the transform. self.p will be set to 1."
    )

    with pytest.warns(DeprecationWarning) as record:
        transform = A.NoOp(always_apply=True, p=0.5)

    assert len(record) == 1

    assert record[0].message.args[0] == warning_expected
    assert transform.p == 1

    with pytest.warns(DeprecationWarning) as record:
        aug = A.Compose([A.NoOp(always_apply=True, p=0.5)], p=1)

    assert len(record) == 1

    assert record[0].message.args[0] == warning_expected
    assert aug.transforms[0].p == 1

    warning_expected_2 = "always_apply is deprecated."

    with pytest.warns(DeprecationWarning) as record:
        transform = A.NoOp(always_apply=False, p=0.5)

    assert len(record) == 1

    assert record[0].message.args[0] == warning_expected_2
    assert transform.p == 0.5

    with pytest.warns(DeprecationWarning) as record:
        aug = A.Compose([A.NoOp(always_apply=False, p=0.5)], p=1)

    assert len(record) == 1

    assert record[0].message.args[0] == warning_expected_2
    assert aug.transforms[0].p == 0.5


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"size": (10, 10)},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "size": (10, 10)},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": 3,
                "mask_x_length": (10, 20),
                "mask_y_length": 10,
                "fill_value": 0,
                "mask_fill_value": 1,
            },
            A.PadIfNeeded: {
                "min_height": 512,
                "min_width": 512,
                "border_mode": 0,
                "value": [124, 116, 104],
                "position": "top_left",
            },
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
        },
        except_augmentations={
            A.FDA,
            A.HistogramMatching,
            A.Lambda,
            A.TemplateTransform,
            A.RandomSizedBBoxSafeCrop,
            A.CropNonEmptyMaskIfExists,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.TextImage,
        },
    ),
)
def test_images_as_target(augmentation_cls, params):
    image = RECTANGULAR_FLOAT_IMAGE if augmentation_cls == A.FromFloat else RECTANGULAR_UINT8_IMAGE
    image2 = image.copy()

    data = {"images": [image, image2]}
    if augmentation_cls == A.MaskDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask

    aug = A.Compose(
        [augmentation_cls(p=1, **params)],
    )

    transformed2 = aug(**data)

    np.testing.assert_array_equal(transformed2["images"][0], transformed2["images"][1])


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            # only image
            A.HistogramMatching: {
                "reference_images": [SQUARE_UINT8_IMAGE],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [SQUARE_FLOAT_IMAGE],
                "read_fn": lambda x: x,
            },
            A.PixelDistributionAdaptation: {
                "reference_images": [SQUARE_UINT8_IMAGE],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
            A.MedianBlur: {"blur_limit": (3, 5)},
            A.TemplateTransform: {
                "templates": SQUARE_UINT8_IMAGE,
            },
            A.RingingOvershoot: {"blur_limit": (3, 5)},
            # dual
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.RandomSizedBBoxSafeCrop: {"height": 10, "width": 10},
            A.BBoxSafeRandomCrop: {"erosion_rate": 0.5},
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
        },
    ),
)
def test_non_contiguous_input_with_compose(augmentation_cls, params, bboxes):
    image = np.empty([3, 100, 100], dtype=np.uint8).transpose(1, 2, 0)
    mask = np.empty([3, 100, 100], dtype=np.uint8).transpose(1, 2, 0)

    # check preconditions
    assert not image.flags["C_CONTIGUOUS"]
    assert not mask.flags["C_CONTIGUOUS"]

    if augmentation_cls == A.RandomCropNearBBox:
        # requires "cropping_bbox" arg
        aug = A.Compose([augmentation_cls(p=1, **params)])
        transformed = aug(image=image, mask=mask, cropping_bbox=bboxes[0])
    elif augmentation_cls in [A.RandomSizedBBoxSafeCrop, A.BBoxSafeRandomCrop]:
        # requires "bboxes" arg
        aug = A.Compose([augmentation_cls(p=1, **params)], bbox_params=A.BboxParams(format="pascal_voc"))
        transformed = aug(image=image, mask=mask, bboxes=bboxes)
    elif augmentation_cls == A.TextImage:
        aug = A.Compose([augmentation_cls(p=1, **params)], bbox_params=A.BboxParams(format="pascal_voc"))
        transformed = aug(
            image=image,
            mask=mask,
            bboxes=bboxes,
            textimage_metadata={"text": "Hello, world!", "bbox": (0.1, 0.1, 0.9, 0.2)},
        )
    elif augmentation_cls == A.OverlayElements:
        # requires "metadata" arg
        aug = A.Compose([augmentation_cls(p=1, **params)])
        transformed = aug(image=image, overlay_metadata=[], mask=mask)
    else:
        # standard args: image and mask
        if augmentation_cls == A.FromFloat:
            # requires float image
            image = (image / 255).astype(np.float32)
            assert not image.flags["C_CONTIGUOUS"]
        elif augmentation_cls == A.MaskDropout:
            # requires single channel mask
            mask = mask[:, :, 0]

        aug = augmentation_cls(p=1, **params)
        transformed = aug(image=image, mask=mask)

    assert transformed["image"].flags["C_CONTIGUOUS"]

    # Check if the augmentation is not an ImageOnlyTransform and mask is in the output
    if not issubclass(augmentation_cls, ImageOnlyTransform) and "mask" in transformed:
        assert transformed["mask"].flags[
            "C_CONTIGUOUS"
        ], f"{augmentation_cls.__name__} did not return a C_CONTIGUOUS mask"


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"size": (10, 10)},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "size": (10, 10)},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": 3,
                "mask_x_length": (10, 20),
                "mask_y_length": 10,
                "fill_value": 0,
                "mask_fill_value": 1,
            },
            A.PadIfNeeded: {
                "min_height": 512,
                "min_width": 512,
                "border_mode": 0,
                "value": [124, 116, 104],
                "position": "top_left",
            },
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
        },
        except_augmentations={
            A.FDA,
            A.HistogramMatching,
            A.Lambda,
            A.TemplateTransform,
            A.RandomSizedBBoxSafeCrop,
            A.CropNonEmptyMaskIfExists,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.TextImage,
            A.FromFloat,
            A.MaskDropout,
        },
    ),
)
@pytest.mark.parametrize(
    "masks",
    [
        [np.random.randint(0, 2, [100, 100], dtype=np.uint8)] * 2,
        [np.random.randint(0, 2, [100, 100, 3], dtype=np.uint8)] * 2,
        np.stack([np.random.randint(0, 2, [100, 100], dtype=np.uint8)] * 2),
    ],
)
def test_masks_as_target(augmentation_cls, params, masks):
    image = SQUARE_UINT8_IMAGE

    aug = A.Compose(
        [augmentation_cls(p=1, **params)],
    )

    transformed = aug(image=image, masks=masks)

    np.testing.assert_array_equal(transformed["masks"][0], transformed["masks"][1])

    assert transformed["masks"][0].dtype == masks[0].dtype


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
            A.RandomResizedCrop: {"size": (113, 103)},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "size": (113, 103)},
            A.Resize: {"height": 113, "width": 113},
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.CropAndPad: {"px": 10},
        },
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.PixelDropout,
            A.CropNonEmptyMaskIfExists,
            A.PixelDistributionAdaptation,
            A.PadIfNeeded,
            A.RandomCrop,
            A.Crop,
            A.CenterCrop,
            A.FDA,
            A.HistogramMatching,
            A.Lambda,
            A.TemplateTransform,
            A.CropNonEmptyMaskIfExists,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.TextImage,
            A.FromFloat,
            A.MaskDropout,
            A.XYMasking,
            A.TimeMasking,
            A.FrequencyMasking,
            A.RandomPerspective,
        },
    ),
)
@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST,
                                                cv2.INTER_LINEAR,
                                                cv2.INTER_CUBIC,
                                                cv2.INTER_AREA
                                                ])
def test_mask_interpolation(augmentation_cls, params, interpolation):
    image = SQUARE_UINT8_IMAGE
    mask = image.copy()

    aug = A.Compose([augmentation_cls(p=1, interpolation=interpolation, mask_interpolation=interpolation, seed=42,**params)])

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
    transform = A.Compose([compose([A.Affine(p=1), A.RandomSizedCrop(min_max_height=(4, 8), size= (113, 103), p=1)], p=1)], mask_interpolation=interpolation)

    image = SQUARE_UINT8_IMAGE
    mask = image.copy()

    transformed = transform(image=image, mask=mask)

    assert transformed["mask"].flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ["transform", "expected_param_keys"],
    [
        (A.HorizontalFlip(p=1), {"cols", "rows", "shape"}),
        (A.VerticalFlip(p=1), {"cols", "rows", "shape"}),
        (
            A.RandomBrightnessContrast(p=1),
            {"cols", "rows", "shape", "alpha", "beta"}
        ),
        (
            A.Rotate(p=1),
            {'shape', 'cols', 'rows', 'x_min', 'x_max', 'y_min', 'y_max', 'matrix', 'bbox_matrix', 'interpolation'}
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
    transform = A.Compose(transforms, p=1, save_applied_params=True)
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
    transform = A.Compose([transform_class(p=1, **transform_params)], save_applied_params=True)
    result = transform(image=image)

    assert len(result["applied_transforms"]) == 1
    transform_name, params = result["applied_transforms"][0]

    assert transform_name == transform_class.__name__

def test_no_param_tracking():
    """Test that params are not tracked when save_applied_params=False"""
    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.Blur(p=1)
    ], p=1, save_applied_params=False)

    result = transform(image=np.zeros((100, 100, 3), dtype=np.uint8))
    assert "applied_transforms" not in result

def test_probability_control():
    """Test that transforms are only tracked when they are actually applied"""
    transform = A.Compose([
        A.HorizontalFlip(p=0),  # Will not be applied
        A.Blur(p=1)  # Will be applied
    ], p=1, save_applied_params=True)

    result = transform(image=np.zeros((100, 100, 3), dtype=np.uint8))
    applied_names = [t[0] for t in result["applied_transforms"]]
    assert "HorizontalFlip" not in applied_names
    assert "Blur" in applied_names

def test_compose_probability():
    """Test that no transforms are tracked when compose probability is 0"""
    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.Blur(p=1)
    ], p=0, save_applied_params=True)

    result = transform(image=np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(result["applied_transforms"]) == 0
