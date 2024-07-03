from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pytest

from albumentations import Crop, RandomCrop, RandomResizedCrop, RandomSizedCrop, Rotate
from albumentations.core.bbox_utils import (
    calculate_bbox_area,
    convert_bbox_from_albumentations,
    convert_bbox_to_albumentations,
    convert_bboxes_to_albumentations,
    denormalize_bbox,
    denormalize_bboxes,
    filter_bboxes,
    normalize_bbox,
    normalize_bboxes,
    union_of_bboxes,
)
from albumentations.core.composition import BboxParams, Compose, ReplayCompose
from albumentations.core.transforms_interface import NoOp, BasicTransform
from albumentations.core.types import BoxType
import albumentations as A
from .utils import set_seed


@pytest.mark.parametrize(
    ["bbox", "expected"],
    [
        ((15, 25, 100, 200), (0.0375, 0.125, 0.25, 1.0)),
        ((15, 25, 100, 200, 99), (0.0375, 0.125, 0.25, 1.0, 99)),
    ],
)
def test_normalize_bbox(bbox: BoxType, expected: BoxType) -> None:
    normalized_bbox = normalize_bbox(bbox, 200, 400)
    assert normalized_bbox == expected


@pytest.mark.parametrize(
    ["bbox", "expected"],
    [
        ((0.0375, 0.125, 0.25, 1.0), (15, 25, 100, 200)),
        ((0.0375, 0.125, 0.25, 1.0, 99), (15, 25, 100, 200, 99)),
    ],
)
def test_denormalize_bbox(bbox: BoxType, expected: BoxType) -> None:
    denormalized_bbox = denormalize_bbox(bbox, 200, 400)
    assert denormalized_bbox == expected


@pytest.mark.parametrize("bbox", [(15, 25, 100, 200), (15, 25, 100, 200, 99)])
def test_normalize_denormalize_bbox(bbox: BoxType) -> None:
    normalized_bbox = normalize_bbox(bbox, 200, 400)
    denormalized_bbox = denormalize_bbox(normalized_bbox, 200, 400)
    assert denormalized_bbox == bbox


@pytest.mark.parametrize(
    "bbox", [(0.0375, 0.125, 0.25, 1.0), (0.0375, 0.125, 0.25, 1.0, 99)]
)
def test_denormalize_normalize_bbox(bbox: BoxType) -> None:
    denormalized_bbox = denormalize_bbox(bbox, 200, 400)
    normalized_bbox = normalize_bbox(denormalized_bbox, 200, 400)
    assert normalized_bbox == bbox


def test_normalize_bboxes() -> None:
    bboxes = [(15, 25, 100, 200), (15, 25, 100, 200, 99)]
    normalized_bboxes_1 = normalize_bboxes(bboxes, 200, 400)
    normalized_bboxes_2 = [
        normalize_bbox(bboxes[0], 200, 400),
        normalize_bbox(bboxes[1], 200, 400),
    ]
    assert normalized_bboxes_1 == normalized_bboxes_2


def test_denormalize_bboxes() -> None:
    bboxes = [(0.0375, 0.125, 0.25, 1.0), (0.0375, 0.125, 0.25, 1.0, 99)]
    denormalized_bboxes_1 = denormalize_bboxes(bboxes, 200, 400)
    denormalized_bboxes_2 = [
        denormalize_bbox(bboxes[0], 200, 400),
        denormalize_bbox(bboxes[1], 200, 400),
    ]
    assert denormalized_bboxes_1 == denormalized_bboxes_2


@pytest.mark.parametrize(
    ["bbox", "rows", "cols", "expected"],
    [((0, 0, 1, 1), 50, 100, 5000), ((0.2, 0.2, 1, 1, 99), 50, 50, 1600)],
)
def test_calculate_bbox_area(
    bbox: BoxType, rows: int, cols: int, expected: int
) -> None:
    area = calculate_bbox_area(bbox, rows, cols)
    assert area == expected


@pytest.mark.parametrize(
    ["bbox", "source_format", "expected"],
    [
        ((20, 30, 40, 50), "coco", (0.2, 0.3, 0.6, 0.8)),
        ((20, 30, 40, 50, 99), "coco", (0.2, 0.3, 0.6, 0.8, 99)),
        ((20, 30, 60, 80), "pascal_voc", (0.2, 0.3, 0.6, 0.8)),
        ((20, 30, 60, 80, 99), "pascal_voc", (0.2, 0.3, 0.6, 0.8, 99)),
        ((0.2, 0.3, 0.4, 0.5), "yolo", (0.00, 0.05, 0.40, 0.55)),
        ((0.2, 0.3, 0.4, 0.5, 99), "yolo", (0.00, 0.05, 0.40, 0.55, 99)),
        ((0.1, 0.1, 0.2, 0.2), "yolo", (0.0, 0.0, 0.2, 0.2)),
        (
            (0.99662423, 0.7520255, 0.00675154, 0.01446759),
            "yolo",
            (0.99324846, 0.744791705, 1.0, 0.759259295),
        ),
        (
            (0.9375, 0.510416, 0.1234375, 0.97638),
            "yolo",
            (0.87578125, 0.022226, 0.999218749, 0.998606),
        ),
    ],
)
def test_convert_bbox_to_albumentations(
    bbox: BoxType, source_format: str, expected: BoxType
) -> None:
    image = np.ones((100, 100, 3))

    converted_bbox = convert_bbox_to_albumentations(
        bbox, rows=image.shape[0], cols=image.shape[1], source_format=source_format
    )
    assert np.all(np.isclose(converted_bbox, expected))


@pytest.mark.parametrize(
    ["bbox", "target_format", "expected"],
    [
        ((0.2, 0.3, 0.6, 0.8), "coco", (20, 30, 40, 50)),
        ((0.2, 0.3, 0.6, 0.8, 99), "coco", (20, 30, 40, 50, 99)),
        ((0.2, 0.3, 0.6, 0.8), "pascal_voc", (20, 30, 60, 80)),
        ((0.2, 0.3, 0.6, 0.8, 99), "pascal_voc", (20, 30, 60, 80, 99)),
        ((0.00, 0.05, 0.40, 0.55), "yolo", (0.2, 0.3, 0.4, 0.5)),
        ((0.00, 0.05, 0.40, 0.55, 99), "yolo", (0.2, 0.3, 0.4, 0.5, 99)),
    ],
)
def test_convert_bbox_from_albumentations(
    bbox: BoxType, target_format: str, expected: BoxType
) -> None:
    image = np.ones((100, 100, 3))
    converted_bbox = convert_bbox_from_albumentations(
        bbox, rows=image.shape[0], cols=image.shape[1], target_format=target_format
    )
    assert np.all(np.isclose(converted_bbox, expected))


@pytest.mark.parametrize(
    ["bbox", "bbox_format"],
    [
        ((20, 30, 40, 50), "coco"),
        ((20, 30, 40, 50, 99), "coco"),
        ((20, 30, 41, 51, 99), "coco"),
        ((21, 31, 40, 50, 99), "coco"),
        ((21, 31, 41, 51, 99), "coco"),
        ((20, 30, 60, 80), "pascal_voc"),
        ((20, 30, 60, 80, 99), "pascal_voc"),
        ((20, 30, 61, 81, 99), "pascal_voc"),
        ((21, 31, 60, 80, 99), "pascal_voc"),
        ((21, 31, 61, 81, 99), "pascal_voc"),
        ((0.01, 0.06, 0.41, 0.56), "yolo"),
        ((0.01, 0.06, 0.41, 0.56, 99), "yolo"),
        ((0.02, 0.06, 0.42, 0.56, 99), "yolo"),
        ((0.01, 0.05, 0.41, 0.55, 99), "yolo"),
        ((0.02, 0.06, 0.41, 0.55, 99), "yolo"),
    ],
)
def test_convert_bbox_to_albumentations_and_back(
    bbox: BoxType, bbox_format: str
) -> None:
    image = np.ones((100, 100, 3))
    converted_bbox = convert_bbox_to_albumentations(
        bbox, rows=image.shape[0], cols=image.shape[1], source_format=bbox_format
    )
    converted_back_bbox = convert_bbox_from_albumentations(
        converted_bbox,
        rows=image.shape[0],
        cols=image.shape[1],
        target_format=bbox_format,
    )
    assert np.all(np.isclose(converted_back_bbox, bbox))


def test_convert_bboxes_to_albumentations() -> None:
    bboxes = [(20, 30, 40, 50), (30, 40, 50, 60, 99)]
    image = np.ones((100, 100, 3))
    converted_bboxes = convert_bboxes_to_albumentations(
        bboxes, rows=image.shape[0], cols=image.shape[1], source_format="coco"
    )
    converted_bbox_1 = convert_bbox_to_albumentations(
        bboxes[0], rows=image.shape[0], cols=image.shape[1], source_format="coco"
    )
    converted_bbox_2 = convert_bbox_to_albumentations(
        bboxes[1], rows=image.shape[0], cols=image.shape[1], source_format="coco"
    )
    assert converted_bboxes == [converted_bbox_1, converted_bbox_2]


def test_convert_bboxes_from_albumentations() -> None:
    bboxes = [(0.2, 0.3, 0.6, 0.8), (0.3, 0.4, 0.7, 0.9, 99)]
    image = np.ones((100, 100, 3))
    converted_bboxes = convert_bboxes_to_albumentations(
        bboxes, rows=image.shape[0], cols=image.shape[1], source_format="coco"
    )
    converted_bbox_1 = convert_bbox_to_albumentations(
        bboxes[0], rows=image.shape[0], cols=image.shape[1], source_format="coco"
    )
    converted_bbox_2 = convert_bbox_to_albumentations(
        bboxes[1], rows=image.shape[0], cols=image.shape[1], source_format="coco"
    )
    assert converted_bboxes == [converted_bbox_1, converted_bbox_2]


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
def test_compose_with_bbox_noop(
    bboxes: BoxType, bbox_format: str, labels: Optional[List[int]]
) -> None:
    image = np.ones((100, 100, 3))
    if labels is not None:
        aug = Compose(
            [NoOp(p=1.0)],
            bbox_params={"format": bbox_format, "label_fields": ["labels"]},
        )
        transformed = aug(image=image, bboxes=bboxes, labels=labels)
    else:
        aug = Compose([NoOp(p=1.0)], bbox_params={"format": bbox_format})
        transformed = aug(image=image, bboxes=bboxes)
    assert np.array_equal(transformed["image"], image)
    assert np.all(np.isclose(transformed["bboxes"], bboxes))


@pytest.mark.parametrize(["bboxes", "bbox_format"], [[[[20, 30, 40, 50]], "coco"]])
def test_compose_with_bbox_noop_error_label_fields(
    bboxes: BoxType, bbox_format: str
) -> None:
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
        [
            [(20, 30, 60, 80, 1, 11), (30, 40, 40, 50, 2, 22)],
            "pascal_voc",
            {"id": [3, 1]},
        ],
        [[(20, 30, 60, 80, 1, 11), (30, 40, 40, 50, 2, 22)], "pascal_voc", {}],
        [
            [(20, 30, 60, 80, 1, 11), (30, 40, 40, 50, 2, 21)],
            "pascal_voc",
            {"id": [31, 32], "subclass": [311, 321]},
        ],
    ],
)
def test_compose_with_bbox_noop_label_outside(
    bboxes: BoxType, bbox_format: str, labels: Dict[str, List[int]]
) -> None:
    image = np.ones((100, 100, 3))
    aug = Compose(
        [NoOp(p=1.0)],
        bbox_params={"format": bbox_format, "label_fields": list(labels.keys())},
    )
    transformed = aug(image=image, bboxes=bboxes, **labels)
    assert np.array_equal(transformed["image"], image)
    assert transformed["bboxes"] == bboxes
    for k, v in labels.items():
        assert transformed[k] == v


def test_random_sized_crop_size() -> None:
    image = np.ones((100, 100, 3))
    bboxes = [(0.2, 0.3, 0.6, 0.8), (0.3, 0.4, 0.7, 0.9, 99)]
    aug = RandomSizedCrop(min_max_height=(70, 90), size=(50, 50), p=1.0)
    transformed = aug(image=image, bboxes=bboxes)
    assert transformed["image"].shape == (50, 50, 3)
    assert len(bboxes) == len(transformed["bboxes"])


def test_random_resized_crop_size() -> None:
    image = np.ones((100, 100, 3))
    bboxes = [(0.2, 0.3, 0.6, 0.8), (0.3, 0.4, 0.7, 0.9, 99)]
    aug = RandomResizedCrop(size=(50, 50), p=1.0)
    transformed = aug(image=image, bboxes=bboxes)
    assert transformed["image"].shape == (50, 50, 3)
    assert len(bboxes) == len(transformed["bboxes"])


def test_random_rotate() -> None:
    image = np.ones((192, 192, 3))
    bboxes = [(78, 42, 142, 80)]
    aug = Rotate(limit=15, p=1.0)
    transformed = aug(image=image, bboxes=bboxes)
    assert len(bboxes) == len(transformed["bboxes"])


def test_crop_boxes_replay_compose() -> None:
    image = np.ones((512, 384, 3))
    bboxes = [(78, 42, 142, 80), (32, 12, 42, 72), (200, 100, 300, 200)]
    labels = [0, 1, 2]
    transform = ReplayCompose(
        [RandomCrop(256, 256, p=1.0)],
        bbox_params=BboxParams(
            format="pascal_voc", min_area=16, label_fields=["labels"]
        ),
    )

    input_data = dict(image=image, bboxes=bboxes, labels=labels)
    transformed = transform(**input_data)
    transformed2 = ReplayCompose.replay(transformed["replay"], **input_data)

    np.testing.assert_almost_equal(transformed["bboxes"], transformed2["bboxes"])


def test_crop_boxes_return_params() -> None:
    image = np.ones((512, 384, 3))
    bboxes = [(78, 42, 142, 80), (32, 12, 42, 72), (200, 100, 300, 200)]
    labels = [0, 1, 2]
    transform = Compose(
        [RandomCrop(256, 256, p=1.0)],
        bbox_params=BboxParams(
            format="pascal_voc", min_area=16, label_fields=["labels"]
        ),
        return_params=True,
    )

    input_data = dict(image=image, bboxes=bboxes, labels=labels)
    transformed = transform(**input_data)
    transformed2 = transform.run_with_params(
        params=transformed["applied_params"], **input_data
    )

    np.testing.assert_almost_equal(transformed["bboxes"], transformed2["bboxes"])


def test_bounding_box_partially_outside_no_clip() -> None:
    """
    Test error is raised when bounding box exceeds image boundaries without clipping.
    """
    # Define a transformation with NoOp
    transform = Compose(
        [NoOp()], bbox_params={"format": "pascal_voc", "label_fields": ["labels"]}
    )

    # Bounding box that exceeds the image dimensions
    bbox = (110, 50, 140, 90)  # x_min, y_min, x_max, y_max in pixel values
    labels = [1]

    # Test should raise an error since bbox is out of image bounds and clipping is not enabled
    with pytest.raises(ValueError):
        transform(
            image=np.zeros((100, 100, 3), dtype=np.uint8), bboxes=[bbox], labels=labels
        )


@pytest.mark.parametrize(
    "image_size, bbox, expected_bbox",
    [
        ((100, 100), (-10, -10, 110, 110), (0, 0, 100, 100)),
        ((200, 200), (-20, -20, 220, 220), (0, 0, 200, 200)),
        ((50, 50), (-5, -5, 55, 55), (0, 0, 50, 50)),
    ],
)
def test_bounding_box_outside_clip(
    image_size: Tuple[int, int], bbox: BoxType, expected_bbox: BoxType
) -> None:
    transform = Compose(
        [A.NoOp()],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"], "clip": True},
    )
    transformed = transform(
        image=np.zeros((*image_size, 3), dtype=np.uint8), bboxes=[bbox], labels=[1]
    )
    assert transformed["bboxes"][0] == expected_bbox


@pytest.mark.parametrize(
    "bbox, expected_bbox",
    [
        ((1, 1, 1, 1), (2, 1, 1, 1)),
        ((0, 1, 1, 1), (3, 1, 1, 1)),
        ((1, 0, 1, 1), (2, 0, 1, 1)),
        ((0, 0, 1, 1), (3, 0, 1, 1)),
    ],
)
def test_bounding_box_hflip(bbox: BoxType, expected_bbox: BoxType) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="coco", label_fields=[]),
    )

    transformed = transform(image=image, bboxes=[bbox])

    assert transformed["bboxes"][0] == expected_bbox


@pytest.mark.parametrize(
    "bbox, expected_bbox",
    [
        ((1, 1, 1, 1), (1, 2, 1, 1)),
        ((0, 1, 1, 1), (0, 2, 1, 1)),
        ((1, 0, 1, 1), (1, 3, 1, 1)),
        ((0, 0, 1, 1), (0, 3, 1, 1)),
    ],
)
def test_bounding_box_vflip(bbox: BoxType, expected_bbox: BoxType) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    transform = A.Compose(
        [A.VerticalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="coco", label_fields=[]),
    )

    transformed = transform(image=image, bboxes=[bbox])

    assert transformed["bboxes"][0] == expected_bbox


@pytest.mark.parametrize(
    ["bboxes", "min_area", "min_visibility", "target"],
    [
        (
            [(0.1, 0.5, 1.1, 0.9), (-0.1, 0.5, 0.8, 0.9), (0.1, 0.5, 0.8, 0.9)],
            0,
            0,
            [(0.1, 0.5, 1, 0.9), (0.0, 0.5, 0.8, 0.9), (0.1, 0.5, 0.8, 0.9)],
        ),
        ([(0.1, 0.5, 0.8, 0.9), (0.4, 0.5, 0.5, 0.6)], 150, 0, [(0.1, 0.5, 0.8, 0.9)]),
        ([(0.1, 0.5, 0.8, 0.9), (0.4, 0.9, 0.5, 1.6)], 0, 0.75, [(0.1, 0.5, 0.8, 0.9)]),
        (
            [(0.1, 0.5, 0.8, 0.9), (0.4, 0.7, 0.5, 1.1)],
            0,
            0.7,
            [(0.1, 0.5, 0.8, 0.9), (0.4, 0.7, 0.5, 1)],
        ),
    ],
)
def test_filter_bboxes(
    bboxes: List[BoxType], min_area: float, min_visibility: float, target: List[BoxType]
) -> None:
    filtered_bboxes = filter_bboxes(
        bboxes, min_area=min_area, min_visibility=min_visibility, rows=100, cols=100
    )
    assert filtered_bboxes == target


@pytest.mark.parametrize(
    ["bboxes", "img_width", "img_height", "min_width", "min_height", "target"],
    [
        [
            [
                (0.1, 0.1, 0.9, 0.9),
                (0.1, 0.1, 0.2, 0.9),
                (0.1, 0.1, 0.9, 0.2),
                (0.1, 0.1, 0.2, 0.2),
            ],
            100,
            100,
            20,
            20,
            [(0.1, 0.1, 0.9, 0.9)],
        ],
        [
            [
                (0.1, 0.1, 0.9, 0.9),
                (0.1, 0.1, 0.2, 0.9),
                (0.1, 0.1, 0.9, 0.2),
                (0.1, 0.1, 0.2, 0.2),
            ],
            100,
            100,
            20,
            0,
            [(0.1, 0.1, 0.9, 0.9), (0.1, 0.1, 0.9, 0.2)],
        ],
        [
            [
                (0.1, 0.1, 0.9, 0.9),
                (0.1, 0.1, 0.2, 0.9),
                (0.1, 0.1, 0.9, 0.2),
                (0.1, 0.1, 0.2, 0.2),
            ],
            100,
            100,
            0,
            20,
            [(0.1, 0.1, 0.9, 0.9), (0.1, 0.1, 0.2, 0.9)],
        ],
    ],
)
def test_filter_bboxes_by_min_width_height(
    bboxes: List[BoxType],
    img_width: int,
    img_height: int,
    min_width: int,
    min_height: int,
    target: List[BoxType],
) -> None:
    filtered_bboxes = filter_bboxes(
        bboxes,
        cols=img_width,
        rows=img_height,
        min_width=min_width,
        min_height=min_height,
    )
    assert filtered_bboxes == target


@pytest.mark.parametrize(
    "get_transform",
    [
        lambda sign: A.Affine(translate_px=sign * 2),
        lambda sign: A.ShiftScaleRotate(
            shift_limit=(sign * 0.02, sign * 0.02), scale_limit=0, rotate_limit=0
        ),
    ],
)
@pytest.mark.parametrize(
    ["bboxes", "expected", "min_visibility", "sign"],
    [
        [[(0, 0, 10, 10, 1)], [], 0.9, -1],
        [[(0, 0, 10, 10, 1)], [(0, 0, 8, 8, 1)], 0.6, -1],
        [[(90, 90, 100, 100, 1)], [], 0.9, 1],
        [[(90, 90, 100, 100, 1)], [(92, 92, 100, 100, 1)], 0.49, 1],
    ],
)
def test_bbox_clipping(
    get_transform: Callable[[int], BasicTransform],
    bboxes: List[BoxType],
    expected: List[BoxType],
    min_visibility: float,
    sign: int,
) -> None:
    image = np.zeros([100, 100, 3], dtype=np.uint8)
    transform = get_transform(sign)
    transform.p = 1
    aug = A.Compose(
        [transform],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=min_visibility),
    )

    res = aug(image=image, bboxes=bboxes)["bboxes"]
    assert res == expected


def test_bbox_clipping_perspective() -> None:
    set_seed(0)
    transform = A.Compose(
        [A.Perspective(scale=(0.05, 0.05), p=1)],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.6),
    )

    image = np.empty([1000, 1000, 3], dtype=np.uint8)
    bboxes = np.array([[0, 0, 100, 100, 1]])
    res = transform(image=image, bboxes=bboxes)["bboxes"]
    assert len(res) == 0


@pytest.mark.parametrize(
    "bboxes, erosion_rate, expected",
    [
        # No bboxes
        ([], 0.0, None),

        # Single bbox, no erosion
        ([(0.1, 0.1, 0.3, 0.3)], 0.0, (0.1, 0.1, 0.3, 0.3)),
        ([(0.0, 0.0, 0.16759776536312848, 0.8333333333333334, 1)], 0, (0.0, 0.0, 0.16759776536312848, 0.8333333333333334) ),

        # Multiple bboxes, no erosion
        ([(0.1, 0.1, 0.3, 0.3), (0.2, 0.2, 0.4, 0.4)], 0.0, (0.1, 0.1, 0.4, 0.4)),

        # Single bbox with erosion
        ([(0.1, 0.1, 0.3, 0.3)], 0.5, (0.15, 0.15, 0.25, 0.25)),

        # Multiple bboxes with erosion
        ([(0.1, 0.1, 0.3, 0.3), (0.2, 0.2, 0.4, 0.4)], 0.5, (0.15, 0.15, 0.35, 0.35)),

        # Edge case with maximum erosion
        ([(0.1, 0.1, 0.3, 0.3), (0.2, 0.2, 0.4, 0.4)], 1.0, (0.2, 0.2, 0.3, 0.3)),

        # Bboxes touching edges of normalized space
        ([(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)], 0.0, (0.0, 0.0, 1.0, 1.0)),

        # Mixed size bboxes with no erosion
        ([(0.1, 0.1, 0.2, 0.2), (0.2, 0.2, 0.4, 0.4), (0.3, 0.3, 0.5, 0.5)], 0.0, (0.1, 0.1, 0.5, 0.5)),

        # Mixed size bboxes with erosion
        ([(0.1, 0.1, 0.2, 0.2), (0.2, 0.2, 0.4, 0.4), (0.3, 0.3, 0.5, 0.5)], 0.3, (0.115, 0.115, 0.47, 0.47)),

        # Single bbox with maximum erosion
        ([(0.0, 0.0, 0.16759776536312848, 0.8333333333333334)], 1.0, None),
    ],
)
def test_union_of_bboxes(bboxes, erosion_rate, expected):
    result = union_of_bboxes(bboxes, erosion_rate)
    assert result == expected or np.testing.assert_almost_equal(result, expected, decimal=6) is None
