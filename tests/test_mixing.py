import random
from typing import Any

import numpy as np
import pytest
from deepdiff import DeepDiff

import albumentations as A
from tests.conftest import UINT8_IMAGES, FLOAT32_IMAGES, MULTI_IMAGES


def image_generator():
    yield {"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)}


def complex_image_generator():
    height = 100
    width = 100
    yield {"image": (height, width)}


def complex_read_fn_image(x):
    return {"image": np.random.randint(0, 256, (x["image"][0], x["image"][1], 3), dtype=np.uint8)}


# Mock random.randint to produce consistent results
@pytest.fixture(autouse=True)
def mock_random(monkeypatch):
    def mock_randint(start, end):
        return start  # always return the start value for consistency in tests

    monkeypatch.setattr(random, "randint", mock_randint)


@pytest.mark.parametrize(
    "metadata, img_shape, expected_output",
    [
        (
            # Image + bbox without label + mask + mask_id + label_id + no offset
            {
                "image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "bbox": [0.3, 0.3, 0.5, 0.5],
                "mask": np.ones((20, 20), dtype=np.uint8) * 127,
                "mask_id": 1,
                "bbox_id": 99,
            },
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((20, 20), dtype=np.uint8) * 127,
                "offset": (30, 30),
                "mask_id": 1,
                "bbox": [30, 30, 50, 50, 99],
            },
        ),
        # Image + bbox with label + mask_id + no mask
        (
            {"image": np.ones((20, 20, 3), dtype=np.uint8) * 255, "bbox": [0.3, 0.3, 0.5, 0.5, 99], "mask_id": 1},
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((20, 20), dtype=np.uint8),
                "offset": (30, 30),
                "mask_id": 1,
                "bbox": [30, 30, 50, 50, 99],
            },
        ),
        # Test case with triangular mask
        (
            {
                "image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "bbox": [0, 0, 0.2, 0.2],
                "mask": np.tri(20, 20, dtype=np.uint8) * 127,
                "mask_id": 2,
                "bbox_id": 100,
            },
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.tri(20, 20, dtype=np.uint8) * 127,
                "offset": (0, 0),
                "mask_id": 2,
                "bbox": [0, 0, 20, 20, 100],
            },
        ),
        # Test case with overlay_image having the same size as img_shape
        (
            {
                "image": np.ones((100, 100, 3), dtype=np.uint8) * 255,
                "bbox": [0, 0, 1, 1],
                "mask": np.ones((100, 100), dtype=np.uint8) * 127,
                "mask_id": 3,
                "bbox_id": 101,
            },
            (100, 100),
            {
                "overlay_image": np.ones((100, 100, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((100, 100), dtype=np.uint8) * 127,
                "offset": (0, 0),
                "mask_id": 3,
                "bbox": [0, 0, 100, 100, 101],
            },
        ),
    ],
)
def test_preprocess_metadata(metadata: dict[str, Any], img_shape: tuple[int, int], expected_output: dict[str, Any]):
    result = A.OverlayElements.preprocess_metadata(metadata, img_shape, random.Random(0))

    assert DeepDiff(result, expected_output, ignore_type_in_groups=[(tuple, list)]) == {}


@pytest.mark.parametrize(
    "metadata, expected_output",
    [
        (
            {
                "image": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "bbox": [0.1, 0.2, 0.2, 0.3],
            },
            {
                "expected_overlay": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "expected_bbox": [10, 20, 20, 30],
            },
        ),
        (
            {
                "image": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "bbox": [0.3, 0.4, 0.4, 0.5],
                "label_id": 99,
            },
            {
                "expected_overlay": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "expected_bbox": [30, 40, 40, 50, 99],
            },
        ),
        (
            {
                "image": np.ones((10, 10, 3), dtype=np.uint8) * 255,
            },
            {
                "expected_overlay": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "expected_bbox": [0, 0, 10, 10],
            },
        ),
    ],
)
def test_end_to_end(metadata, expected_output):
    transform = A.Compose([A.OverlayElements(p=1)], strict=True)

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    transformed = transform(image=img, overlay_metadata=metadata)

    expected_img = np.zeros((100, 100, 3), dtype=np.uint8)

    x_min, y_min, x_max, y_max = expected_output["expected_bbox"][:4]

    expected_img[y_min:y_max, x_min:x_max] = expected_output["expected_overlay"]

    if "bbox" in metadata:
        np.testing.assert_array_equal(transformed["image"], expected_img)
    else:
        assert expected_img.sum() == transformed["image"].sum()


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        (A.Mosaic, {
            "reference_data": [
                {
                    "image": UINT8_IMAGES[0],
                },
                {
                    "image": UINT8_IMAGES[1],
                },
                {
                    "image": UINT8_IMAGES[0],
                }],
            "read_fn": lambda x: x,
        }),
        (A.Mosaic, {"reference_data": [1, 2, 3],
                    "read_fn": lambda x: {"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)}}),
        (A.Mosaic, {"reference_data": np.array([1, 2, 3]),
                    "read_fn": lambda x: {"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)}}),
        (A.Mosaic, {"reference_data": None}),
        (A.Mosaic, {"reference_data": image_generator(),
                    "read_fn": lambda x: x}),
        (A.Mosaic, {"reference_data": complex_image_generator(),
                    "read_fn": complex_read_fn_image})
    ]
)
def test_2x2_mosaic_rgb_image_only(augmentation_cls, params):
    img = UINT8_IMAGES[0]
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=img)
    assert data["image"].dtype == np.uint8
    assert data["image"].ndim == 3 and data["image"].shape[-1] == 3

    img = UINT8_IMAGES[1]
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=img)
    assert data["image"].dtype == np.uint8
    assert data["image"].ndim == 3 and data["image"].shape[-1] == 3


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        (A.Mosaic, {
            "reference_data": [
                {
                    "image": UINT8_IMAGES[0][:, :, 0].copy(),
                },
                {
                    "image": UINT8_IMAGES[1][:, :, 0].copy(),
                },
                {
                    "image": UINT8_IMAGES[0][:, :, 0].copy(),
                }],
            "read_fn": lambda x: x,
        }),
        (A.Mosaic, {"reference_data": [1, 2, 3],
                    "read_fn": lambda x: {"image": np.random.randint(0, 256, [100, 100], dtype=np.uint8)}}),
        (A.Mosaic, {"reference_data": np.array([1, 2, 3]),
                    "read_fn": lambda x: {"image": np.random.randint(0, 256, [100, 100], dtype=np.uint8)}}),
        (A.Mosaic, {"reference_data": None})
    ]
)
def test_2x2_mosaic_gray_image_only(augmentation_cls, params):
    img = UINT8_IMAGES[0][:, :, 0].copy()
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=img)
    assert data["image"].dtype == np.uint8
    assert data["image"].ndim == 2

    img = UINT8_IMAGES[1][:, :, 0].copy()
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=img)
    assert data["image"].dtype == np.uint8
    assert data["image"].ndim == 2


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        (A.Mosaic, {
            "reference_data": [
                {
                    "image": FLOAT32_IMAGES[0],
                },
                {
                    "image": FLOAT32_IMAGES[1],
                },
                {
                    "image": FLOAT32_IMAGES[0],
                }],
            "read_fn": lambda x: x,
        }),
        (A.Mosaic, {"reference_data": [1, 2, 3],
                    "read_fn": lambda x: {"image": FLOAT32_IMAGES[0].copy()}}),
        (A.Mosaic, {"reference_data": np.array([1, 2, 3]),
                    "read_fn": lambda x: {"image": FLOAT32_IMAGES[0].copy()}}),
        (A.Mosaic, {"reference_data": None})
    ]
)
def test_2x2_mosaic_float_rgb_image_only(augmentation_cls, params):
    img = FLOAT32_IMAGES[0]
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=img)
    assert data["image"].dtype == np.float32
    assert data["image"].ndim == 3 and data["image"].shape[-1] == 3

    img = FLOAT32_IMAGES[1]
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=img)
    assert data["image"].dtype == np.float32
    assert data["image"].ndim == 3 and data["image"].shape[-1] == 3


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        (A.Mosaic, {
            "reference_data": [
                {
                    "image": FLOAT32_IMAGES[0][:, :, 0].copy(),
                },
                {
                    "image": FLOAT32_IMAGES[1][:, :, 0].copy(),
                },
                {
                    "image": FLOAT32_IMAGES[0][:, :, 0].copy(),
                }],
            "read_fn": lambda x: x,
        }),
        (A.Mosaic, {"reference_data": [1, 2, 3],
                    "read_fn": lambda x: {"image": FLOAT32_IMAGES[0][:, :, 0].copy()}}),
        (A.Mosaic, {"reference_data": np.array([1, 2, 3]),
                    "read_fn": lambda x: {"image": FLOAT32_IMAGES[0][:, :, 0].copy()}}),
        (A.Mosaic, {"reference_data": None})
    ]
)
def test_2x2_mosaic_float_gray_image_only(augmentation_cls, params):
    img = FLOAT32_IMAGES[0][:, :, 0].copy()
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=img)
    assert data["image"].dtype == np.float32
    assert data["image"].ndim == 2

    img = FLOAT32_IMAGES[1][:, :, 0].copy()
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=img)
    assert data["image"].dtype == np.float32
    assert data["image"].ndim == 2


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        (A.Mosaic, {
            "reference_data": [
                {
                    "image": MULTI_IMAGES[0],
                },
                {
                    "image": MULTI_IMAGES[0],
                },
                {
                    "image": MULTI_IMAGES[0],
                }],
            "read_fn": lambda x: x,
        }),
        (A.Mosaic, {"reference_data": [1, 2, 3],
                    "read_fn": lambda x: {"image": MULTI_IMAGES[0]}}),
        (A.Mosaic, {"reference_data": np.array([1, 2, 3]),
                    "read_fn": lambda x: {"image": MULTI_IMAGES[0]}}),
        (A.Mosaic, {"reference_data": None})
    ]
)
def test_2x2_mosaic_multidim_image_only(augmentation_cls, params):
    img = MULTI_IMAGES[0]
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=img)
    assert data["image"].dtype == np.uint8
    assert data["image"].ndim == img.ndim and data["image"].shape[-1] == img.shape[-1]


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        (A.Mosaic, {
            "reference_data": [
                {
                    "image": MULTI_IMAGES[1],
                },
                {
                    "image": MULTI_IMAGES[1],
                },
                {
                    "image": MULTI_IMAGES[1],
                }],
            "read_fn": lambda x: x,
        }),
        (A.Mosaic, {"reference_data": [1, 2, 3],
                    "read_fn": lambda x: {"image": MULTI_IMAGES[1]}}),
        (A.Mosaic, {"reference_data": np.array([1, 2, 3]),
                    "read_fn": lambda x: {"image": MULTI_IMAGES[1]}}),
        (A.Mosaic, {"reference_data": None})
    ]
)
def test_2x2_mosaic_float_multidim_image_only(augmentation_cls, params):
    img = MULTI_IMAGES[1]
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=img)
    assert data["image"].dtype == np.float32
    assert data["image"].ndim == img.ndim and data["image"].shape[-1] == img.shape[-1]


@pytest.mark.parametrize(
    ["params", "expected_output"],
    [
        # RGB images with different shapes, keep aspect ratio, output shape 100
        ({
            "reference_data": [
                {
                    "image": np.full((200, 100, 3), fill_value=(255, 0, 0), dtype=np.uint8),
                    "mask": np.full((200, 100), fill_value=175, dtype=np.uint8),
                    "bbox": np.array([[0, 0, 0.2, 0.2, 0], [0.9, 0.9, 1.0, 1.0, 0], [0.25, 0.25, 0.75, 0.75, 0]]),
                    "class_labels": ["dummy", "dummy", "dummy"],
                    "keypoints": np.array([[0, 80, 0, 0, 2.0], [199, 0, 0, 0, 2.0], [40, 80, 0, 0, 2.0]])
                },
                {
                    "image": np.full((50, 100, 3), fill_value=(0, 255, 0), dtype=np.uint8),
                    "mask": np.full((50, 100), fill_value=130, dtype=np.uint8),
                    "bbox": np.array([[0, 0, 0.2, 0.2, 0], [0.9, 0.9, 1.0, 1.0, 0], [0.25, 0.25, 0.75, 0.75, 0]]),
                    "class_labels": ["dummy", "dummy", "dummy"],
                    "keypoints": np.array([[0, 99, 0, 0, 2.0], [49, 20, 0, 0, 2.0], [40, 20, 0, 0, 2.0]])
                },
                {
                    "image": np.full((50, 50, 3), fill_value=(0, 0, 255), dtype=np.uint8),
                    "mask": np.full((50, 50), fill_value=60, dtype=np.uint8),
                    "bbox": np.array([[0, 0, 0.2, 0.2, 0], [0.9, 0.9, 1.0, 1.0, 0], [0.25, 0.25, 0.75, 0.75, 0]]),
                    "class_labels": ["dummy", "dummy", "dummy"],
                    "keypoints": np.array([[15, 15, 0, 0, 2.0], [49, 49, 0, 0, 2.0]])
                }],
            "mosaic_size": 100,
            "keep_aspect_ratio": True,
            "center_range": (0.5, 0.5)  # Fix for reproductible results
        },
        {
            "mosaic_size": (100, 100),
            "image": np.block([[[255, 255, 255], [255, 0, 0]], [[0, 255, 0], [0, 0, 255]]]).repeat(50, axis=0).repeat(50, axis=1).astype(np.uint8),
            "mask": np.repeat(np.repeat([[255, 175], [130, 60]], 50, axis=0), 50, axis=1).astype(np.uint8),
            "bboxes": np.array([[0.4, 0.4, 0.5, 0.5],
                                [0., 0., 0.25, 0.25],
                                [0.95, 0.4, 1., 0.5],
                                [0.625, 0., 0.875, 0.25],
                                [0.4, 0.95, 0.5, 1.],
                                [0., 0.625, 0.25, 0.875],
                                [0.5, 0.5, 0.7, 0.7],
                                [0.75, 0.75, 1., 1.]]),
            "class_labels": ["dummy"] * 8,
            "keypoints": np.array([[49., 49., 2.], [30., 30., 2.], [80., 80., 4.]])
        }),
    ],
)
def test_2x2_mosaic_end_to_end(params, expected_output):
    img = np.full((100, 100, 3), fill_value=(255, 255, 255), dtype=np.uint8)
    mask = np.full((100, 100), fill_value=255, dtype=np.uint8)
    bboxes = np.array([[0, 0, 0.2, 0.2], [0.9, 0.9, 1.0, 1.0], [0.25, 0.25, 0.75, 0.75]])
    keypoints = np.array([[0, 0, 2.0], [99, 99, 2.0], [80, 80, 2.0]])
    class_labels = ["dummy"] * len(bboxes)

    transform = A.Compose(
        [A.Mosaic(p=1, **params)],
        bbox_params=A.BboxParams(format="albumentations", label_fields=["class_labels"]),
        keypoint_params=A.KeypointParams(format="xys"),
        strict=True
    )
    transformed = transform(image=img, mask=mask, bboxes=bboxes, keypoints=keypoints, class_labels=class_labels)

    result_image = transformed["image"]
    np.testing.assert_array_equal(result_image, expected_output["image"])

    result_mask = transformed["mask"]
    np.testing.assert_array_equal(result_mask, expected_output["mask"])

    result_bboxes = transformed["bboxes"]
    np.testing.assert_array_equal(result_bboxes, expected_output["bboxes"])

    assert transformed["class_labels"] == expected_output["class_labels"]

    result_keypoints = transformed["keypoints"]
    np.testing.assert_array_equal(result_keypoints, expected_output["keypoints"])
