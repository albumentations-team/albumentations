import numpy as np
import pytest

from albumentations import Crop, RandomCrop, RandomResizedCrop, RandomSizedCrop, Rotate
from albumentations.core.bbox_utils import (
    calculate_bboxes_area,
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
    denormalize_bboxes_np,
    normalize_bboxes_np,
)
from albumentations.core.composition import BboxParams, Compose, ReplayCompose
from albumentations.core.transforms_interface import BBoxesInternalType, NoOp

from .utils import bboxes_internal_type_to_list, bboxes_list_to_internal_type


def test_normalize_bboxes():
    bboxes = [
        (15, 25, 100, 200),
        (15, 25, 100, 200, 99),
    ]
    expected = [
        (0.0375, 0.125, 0.25, 1.0),
        (0.0375, 0.125, 0.25, 1.0, 99),
    ]

    bboxes = bboxes_list_to_internal_type(bboxes)
    expected = bboxes_list_to_internal_type(expected)

    bboxes = normalize_bboxes_np(bboxes, rows=200, cols=400)
    assert bboxes == expected


def test_denormalize_bboxes():
    bboxes = [
        (0.0375, 0.125, 0.25, 1.0),
        (0.0375, 0.125, 0.25, 1.0, 99),
    ]
    expected = [
        (15.0, 25.0, 100.0, 200.0),
        (15.0, 25.0, 100.0, 200.0, 99),
    ]

    bboxes = bboxes_list_to_internal_type(bboxes)
    expected = bboxes_list_to_internal_type(expected)

    bboxes = denormalize_bboxes_np(bboxes, rows=200, cols=400)

    assert bboxes == expected


def test_normalize_denormalize_bboxes():
    bboxes = [(15, 25, 100, 200), (15, 25, 100, 200, 99)]
    bboxes_in = bboxes_list_to_internal_type(bboxes)
    bboxes_in = normalize_bboxes_np(bboxes_in, rows=200, cols=400)
    bboxes_in = denormalize_bboxes_np(bboxes_in, rows=200, cols=400)
    assert bboxes_internal_type_to_list(bboxes_in) == bboxes


def test_denormalize_normalize():
    bboxes = [(0.0375, 0.125, 0.25, 1.0), (0.0375, 0.125, 0.25, 1.0, 99)]
    bboxes_in = bboxes_list_to_internal_type(bboxes)
    bboxes_in = denormalize_bboxes_np(bboxes_in, rows=200, cols=400)
    bboxes_in = normalize_bboxes_np(bboxes_in, rows=200, cols=400)
    assert bboxes_internal_type_to_list(bboxes_in) == bboxes


@pytest.mark.parametrize(
    "bboxes, rows, cols, expected",
    [
        (np.array([(0, 0, 1, 1), (0.2, 0.2, 1, 1, 99)]), 50, 100, np.array([5000, 3200])),
    ],
)
def test_calculate_bboxes_area(bboxes, rows, cols, expected):
    bboxes = bboxes_list_to_internal_type(bboxes)
    areas = calculate_bboxes_area(bboxes, rows, cols).astype(int)
    assert np.array_equal(areas, expected)


@pytest.mark.parametrize(
    ["bboxes", "source_format", "expected"],
    [
        ([(20, 30, 40, 50), (20, 30, 40, 50, 99)], "coco", [(0.2, 0.3, 0.6, 0.8), (0.2, 0.3, 0.6, 0.8, 99)]),
        ([(20, 30, 60, 80), (20, 30, 60, 80, 99)], "pascal_voc", [(0.2, 0.3, 0.6, 0.8), (0.2, 0.3, 0.6, 0.8, 99)]),
        (
            [
                (0.2, 0.3, 0.4, 0.5),
                (0.2, 0.3, 0.4, 0.5, 99),
                (0.1, 0.1, 0.2, 0.2),
                (0.99662423, 0.7520255, 0.00675154, 0.01446759),
                (0.9375, 0.510416, 0.1234375, 0.97638),
            ],
            "yolo",
            [
                (0.00, 0.05, 0.40, 0.55),
                (0.00, 0.05, 0.40, 0.55, 99),
                (0.0, 0.0, 0.2, 0.2),
                (0.99324846, 0.744791705, 1.0, 0.759259295),
                (0.87578125, 0.022226, 0.999218749, 0.998606),
            ],
        ),
    ],
)
def test_convert_bboxes_to_albumentations(bboxes, source_format, expected):
    bboxes = bboxes_list_to_internal_type(bboxes)
    expected = bboxes_list_to_internal_type(expected)
    image = np.ones((100, 100, 3), dtype=np.uint8)
    converted_bboxes = convert_bboxes_to_albumentations(
        bboxes, rows=image.shape[0], cols=image.shape[1], source_format=source_format
    )

    assert np.allclose(converted_bboxes.array, expected.array)
    assert converted_bboxes.targets == expected.targets


@pytest.mark.parametrize(
    ["bboxes", "target_format", "expected"],
    [
        ([(0.2, 0.3, 0.6, 0.8), (0.2, 0.3, 0.6, 0.8, 99)], "coco", [(20, 30, 40, 50), (20, 30, 40, 50, 99)]),
        ([(0.2, 0.3, 0.6, 0.8), (0.2, 0.3, 0.6, 0.8, 99)], "pascal_voc", [(20, 30, 60, 80), (20, 30, 60, 80, 99)]),
        (
            [(0.00, 0.05, 0.40, 0.55), (0.00, 0.05, 0.40, 0.55, 99)],
            "yolo",
            [(0.2, 0.3, 0.4, 0.5), (0.2, 0.3, 0.4, 0.5, 99)],
        ),
    ],
)
def test_convert_bboxes_from_albumentations(bboxes, target_format, expected):
    bboxes = bboxes_list_to_internal_type(bboxes)
    expected = bboxes_list_to_internal_type(expected)
    image = np.ones((100, 100, 3), dtype=np.uint8)
    converted_bboxes = convert_bboxes_from_albumentations(
        bboxes, rows=image.shape[0], cols=image.shape[1], target_format=target_format
    )
    assert np.allclose(converted_bboxes.array, expected.array)
    assert converted_bboxes.targets == expected.targets


@pytest.mark.parametrize(
    ["bboxes", "bbox_format"],
    [
        (
            [
                (20, 30, 40, 50),
                (20, 30, 40, 50, 99),
                (20, 30, 41, 51, 99),
                (21, 31, 40, 50, 99),
                (21, 31, 41, 51, 99),
            ],
            "coco",
        ),
        (
            [
                (20, 30, 60, 80),
                (20, 30, 60, 80, 99),
                (20, 30, 61, 81, 99),
                (21, 31, 60, 80, 99),
                (21, 31, 61, 81, 99),
            ],
            "pascal_voc",
        ),
        (
            [
                (0.01, 0.06, 0.41, 0.56),
                (0.01, 0.06, 0.41, 0.56, 99),
                (0.02, 0.06, 0.42, 0.56, 99),
                (0.01, 0.05, 0.41, 0.55, 99),
                (0.02, 0.06, 0.41, 0.55, 99),
            ],
            "yolo",
        ),
    ],
)
def test_convert_bboxes_to_albumentations_and_back(bboxes, bbox_format):
    bboxes = bboxes_list_to_internal_type(bboxes)
    image = np.ones((100, 100, 3), dtype=np.uint8)
    converted_bboxes = convert_bboxes_to_albumentations(
        bboxes, rows=image.shape[0], cols=image.shape[1], source_format=bbox_format
    )
    converted_back_bboxes = convert_bboxes_from_albumentations(
        converted_bboxes, rows=image.shape[0], cols=image.shape[1], target_format=bbox_format
    )

    assert np.allclose(converted_back_bboxes.array, bboxes.array)
    assert converted_back_bboxes.targets == bboxes.targets


@pytest.mark.parametrize(
    ["bboxes", "bbox_format", "labels"],
    [
        ([(20, 30, 40, 50)], "coco", [1]),
        ([(20, 30, 40, 50, 99), (10, 40, 30, 20, 9)], "coco", None),
        ([(20, 30, 60, 80)], "pascal_voc", [2]),
        ([(20, 30, 60, 80, 99)], "pascal_voc", None),
        ([(0.1, 0.2, 0.1, 0.2)], "yolo", [2]),
        ([(0.1, 0.2, 0.1, 0.2, 99)], "yolo", None),
    ],
)
def test_compose_with_bbox_noop(bboxes, bbox_format, labels):
    image = np.ones((100, 100, 3))
    if labels is not None:
        aug = Compose([NoOp(p=1.0)], bbox_params={"format": bbox_format, "label_fields": ["labels"]})
        transformed = aug(image=image, bboxes=bboxes, labels=labels)
    else:
        aug = Compose([NoOp(p=1.0)], bbox_params={"format": bbox_format})
        transformed = aug(image=image, bboxes=bboxes)
    assert np.array_equal(transformed["image"], image)
    assert np.all(np.isclose(transformed["bboxes"], bboxes))


@pytest.mark.parametrize(["bboxes", "bbox_format"], [[[[20, 30, 40, 50]], "coco"]])
def test_compose_with_bbox_noop_error_label_fields(bboxes, bbox_format):
    image = np.ones((100, 100, 3))
    aug = Compose([NoOp(p=1.0)], bbox_params={"format": bbox_format})
    with pytest.raises(Exception):
        aug(image=image, bboxes=bboxes)


@pytest.mark.parametrize(
    ["bboxes", "bbox_format", "labels"],
    [
        [[(20, 30, 60, 80)], "pascal_voc", {"label": [1]}],
        [[], "pascal_voc", {}],
        [[], "pascal_voc", {"label": []}],
        [[(20, 30, 60, 80)], "pascal_voc", {"id": [3]}],
        [[(20, 30, 60, 80), (30, 40, 40, 50)], "pascal_voc", {"id": [3, 1]}],
        [[(20, 30, 60, 80, 1, 11), (30, 40, 40, 50, 2, 22)], "pascal_voc", {"id": [3, 1]}],
        [[(20, 30, 60, 80, 1, 11), (30, 40, 40, 50, 2, 22)], "pascal_voc", {}],
        [[(20, 30, 60, 80, 1, 11), (30, 40, 40, 50, 2, 21)], "pascal_voc", {"id": [31, 32], "subclass": [311, 321]}],
    ],
)
def test_compose_with_bbox_noop_label_outside(bboxes, bbox_format, labels):
    image = np.ones((100, 100, 3))
    aug = Compose([NoOp(p=1.0)], bbox_params={"format": bbox_format, "label_fields": list(labels.keys())})
    transformed = aug(image=image, bboxes=bboxes, **labels)
    assert np.array_equal(transformed["image"], image)
    assert transformed["bboxes"] == bboxes
    for k, v in labels.items():
        assert transformed[k] == v


def test_random_sized_crop_size():
    image = np.ones((100, 100, 3))
    bboxes = [(0.2, 0.3, 0.6, 0.8), (0.3, 0.4, 0.7, 0.9, 99)]
    bboxes_in = bboxes_list_to_internal_type(bboxes)
    aug = RandomSizedCrop(min_max_height=(70, 90), height=50, width=50, p=1.0)
    transformed = aug(image=image, bboxes=bboxes_in)
    assert transformed["image"].shape == (50, 50, 3)
    assert len(bboxes) == len(transformed["bboxes"])


def test_random_resized_crop_size():
    image = np.ones((100, 100, 3))
    bboxes = [(0.2, 0.3, 0.6, 0.8), (0.3, 0.4, 0.7, 0.9, 99)]
    bboxes_in = bboxes_list_to_internal_type(bboxes)
    aug = RandomResizedCrop(height=50, width=50, p=1.0)
    transformed = aug(image=image, bboxes=bboxes_in)
    assert transformed["image"].shape == (50, 50, 3)
    assert len(bboxes) == len(transformed["bboxes"])


def test_random_rotate():
    image = np.ones((192, 192, 3))
    bboxes = [(78, 42, 142, 80)]
    bboxes_in = bboxes_list_to_internal_type(bboxes)
    aug = Rotate(limit=15, p=1.0)
    transformed = aug(image=image, bboxes=bboxes_in)
    assert len(bboxes) == len(transformed["bboxes"])


def test_crop_boxes_replay_compose():
    image = np.ones((512, 384, 3))
    bboxes = [(78, 42, 142, 80), (32, 12, 42, 72), (200, 100, 300, 200)]
    labels = [0, 1, 2]
    transform = ReplayCompose(
        [RandomCrop(256, 256, p=1.0)],
        bbox_params=BboxParams(format="pascal_voc", min_area=16, label_fields=["labels"]),
    )

    input_data = dict(image=image, bboxes=bboxes, labels=labels)
    transformed = transform(**input_data)
    transformed2 = ReplayCompose.replay(transformed["replay"], **input_data)

    np.testing.assert_almost_equal(transformed["bboxes"], transformed2["bboxes"])


@pytest.mark.parametrize(
    ["transforms", "bboxes", "result_bboxes", "min_area", "min_visibility"],
    [
        [[Crop(10, 10, 20, 20)], [[0, 0, 10, 10, 0]], [], 0, 0],
        [[Crop(0, 0, 90, 90)], [[0, 0, 91, 91, 0], [0, 0, 90, 90, 0]], [[0, 0, 90, 90, 0]], 0, 1],
        [[Crop(0, 0, 90, 90)], [[0, 0, 1, 10, 0], [0, 0, 1, 11, 0]], [[0, 0, 1, 10, 0], [0, 0, 1, 11, 0]], 10, 0],
    ],
)
def test_bbox_params_edges(transforms, bboxes, result_bboxes, min_area, min_visibility):
    image = np.empty([100, 100, 3], dtype=np.uint8)
    aug = Compose(transforms, bbox_params=BboxParams("pascal_voc", min_area=min_area, min_visibility=min_visibility))
    res = aug(image=image, bboxes=bboxes)["bboxes"]
    assert np.allclose(res, result_bboxes)
