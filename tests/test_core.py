from __future__ import absolute_import

import typing
from unittest import mock
from unittest.mock import MagicMock, Mock, call

import cv2
import numpy as np
import pytest

from albumentations import (
    BasicTransform,
    Blur,
    ChannelShuffle,
    Crop,
    HorizontalFlip,
    MedianBlur,
    Normalize,
    PadIfNeeded,
    Resize,
    Rotate,
)
from albumentations.augmentations.bbox_utils import check_bboxes
from albumentations.core.composition import (
    BaseCompose,
    BboxParams,
    Compose,
    KeypointParams,
    OneOf,
    OneOrOther,
    PerChannel,
    ReplayCompose,
    Sequential,
    SomeOf,
)
from albumentations.core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    to_tuple,
)

from .utils import get_filtered_transforms


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


def oneof_always_apply_crash():
    aug = Compose([HorizontalFlip(), Rotate(), OneOf([Blur(), MedianBlur()], p=1)], p=1)
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


@pytest.mark.parametrize("N", [1, 2, 5, 10])
@pytest.mark.parametrize("replace", [True, False])
def test_n_of(N, replace):
    transforms = [Mock(p=1, side_effect=lambda **kw: {"image": kw["image"]}) for _ in range(10)]
    augmentation = SomeOf(transforms, N, p=1, replace=replace)
    print(augmentation.n)
    image = np.ones((8, 8))
    augmentation(image=image)
    if not replace:
        assert len([transform for transform in transforms if transform.called]) == N
    assert sum([transform.call_count for transform in transforms]) == N


def test_sequential():
    transforms = [Mock(side_effect=lambda **kw: kw) for _ in range(10)]
    augmentation = Sequential(transforms, p=1)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert len([transform for transform in transforms if transform.called]) == len(transforms)


def test_to_tuple():
    assert to_tuple(10) == (-10, 10)
    assert to_tuple(0.5) == (-0.5, 0.5)
    assert to_tuple((-20, 20)) == (-20, 20)
    assert to_tuple([-20, 20]) == (-20, 20)
    assert to_tuple(100, low=30) == (30, 100)
    assert to_tuple(10, bias=1) == (-9, 11)
    assert to_tuple(100, bias=2) == (-98, 102)


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
    except Exception as e:  # skipcq: PYL-W0703
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


def test_per_channel_mono():
    transforms = [Blur(), Rotate()]
    augmentation = PerChannel(transforms, p=1)
    image = np.ones((8, 8))
    data = augmentation(image=image)
    assert data


def test_per_channel_multi():
    transforms = [Blur(), Rotate()]
    augmentation = PerChannel(transforms, p=1)
    image = np.ones((8, 8, 5))
    data = augmentation(image=image)
    assert data


def test_deterministic_oneof():
    aug = ReplayCompose([OneOf([HorizontalFlip(), Blur()])], p=1)
    for _ in range(10):
        image = (np.random.random((8, 8)) * 255).astype(np.uint8)
        image2 = np.copy(image)
        data = aug(image=image)
        assert "replay" in data
        data2 = ReplayCompose.replay(data["replay"], image=image2)
        assert np.array_equal(data["image"], data2["image"])


def test_deterministic_one_or_other():
    aug = ReplayCompose([OneOrOther(HorizontalFlip(), Blur())], p=1)
    for _ in range(10):
        image = (np.random.random((8, 8)) * 255).astype(np.uint8)
        image2 = np.copy(image)
        data = aug(image=image)
        assert "replay" in data
        data2 = ReplayCompose.replay(data["replay"], image=image2)
        assert np.array_equal(data["image"], data2["image"])


def test_deterministic_sequential():
    aug = ReplayCompose([Sequential([HorizontalFlip(), Blur()])], p=1)
    for _ in range(10):
        image = (np.random.random((8, 8)) * 255).astype(np.uint8)
        image2 = np.copy(image)
        data = aug(image=image)
        assert "replay" in data
        data2 = ReplayCompose.replay(data["replay"], image=image2)
        assert np.array_equal(data["image"], data2["image"])


def test_named_args():
    image = np.empty([100, 100, 3], dtype=np.uint8)
    aug = HorizontalFlip(p=1)

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
    aug = Compose([], additional_targets=additional_targets)

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
        [Crop(0, 0, 50, 50), PadIfNeeded(100, 100)], bbox_params=bbox_params, keypoint_params=keypoint_params
    )
    res = augs(image=image, **targets)

    for key, item in expected.items():
        assert np.all(np.array(item) == np.array(res[key]))


def test_bbox_params_is_not_set(image, bboxes):
    t = Compose([])
    with pytest.raises(ValueError) as exc_info:
        t(image=image, bboxes=bboxes)
    assert str(exc_info.value) == "bbox_params must be specified for bbox transformations"


@pytest.mark.parametrize(
    "compose_transform", get_filtered_transforms((BaseCompose,), custom_arguments={SomeOf: {"n": 1}})
)
@pytest.mark.parametrize(
    "inner_transform",
    [(Normalize, {}), (Resize, {"height": 100, "width": 100})]
    + get_filtered_transforms((BaseCompose,), custom_arguments={SomeOf: {"n": 1}}),  # type: ignore
)
def test_single_transform_compose(
    compose_transform: typing.Tuple[typing.Type[BaseCompose], dict],
    inner_transform: typing.Tuple[typing.Union[typing.Type[BaseCompose], typing.Type[BasicTransform]], dict],
):
    compose_cls, compose_kwargs = compose_transform
    cls, kwargs = inner_transform
    transform = cls(transforms=[], **kwargs) if issubclass(cls, BaseCompose) else cls(**kwargs)

    with pytest.warns(UserWarning):
        res_transform = compose_cls(transforms=transform, **compose_kwargs)  # type: ignore
    assert isinstance(res_transform.transforms, list)


@pytest.mark.parametrize(
    "transforms",
    [OneOf([Sequential([HorizontalFlip(p=1)])], p=1), SomeOf([Sequential([HorizontalFlip(p=1)])], n=1, p=1)],
)
def test_choice_inner_compositions(transforms):
    """Check that the inner composition is selected without errors."""
    image = np.empty([10, 10, 3], dtype=np.uint8)
    transforms(image=image)


@pytest.mark.parametrize(
    "transforms",
    [
        Compose([ChannelShuffle(p=1)], p=1),
        Compose([ChannelShuffle(p=0)], p=0),
    ],
)
def test_contiguous_output(transforms):
    image = np.empty([3, 24, 24], dtype=np.uint8).transpose(1, 2, 0)
    mask = np.empty([3, 24, 24], dtype=np.uint8).transpose(1, 2, 0)

    # check preconditions
    assert not image.flags["C_CONTIGUOUS"]
    assert not mask.flags["C_CONTIGUOUS"]

    # pipeline always outputs contiguous results
    data = transforms(image=image, mask=mask)

    # confirm output contiguous
    assert data["image"].flags["C_CONTIGUOUS"]
    assert data["mask"].flags["C_CONTIGUOUS"]
