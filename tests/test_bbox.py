from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import pytest

import albumentations as A
from albumentations import RandomCrop, RandomResizedCrop, RandomSizedCrop, Rotate
from albumentations.augmentations.crops.functional import crop_bboxes_by_coords
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.bbox_utils import (
    BboxProcessor,
    bboxes_from_masks,
    calculate_bbox_areas_in_pixels,
    check_bboxes,
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
    denormalize_bboxes,
    filter_bboxes,
    masks_from_bboxes,
    normalize_bboxes,
    union_of_bboxes,
    bboxes_to_mask,
    mask_to_bboxes,
)
from albumentations.core.composition import BboxParams, Compose, ReplayCompose
from albumentations.core.transforms_interface import BasicTransform, NoOp

from albumentations.augmentations.dropout.functional import resize_boxes_to_visible_area



@pytest.mark.parametrize(
    "bboxes, image_shape, expected",
    [
        (
            np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
            {"height": 100, "width": 200},
            np.array([[0.05, 0.2, 0.15, 0.4], [0.25, 0.6, 0.35, 0.8]]),
        ),
        (
            np.array([[0, 0, 200, 100]]),
            {"height": 100, "width": 200},
            np.array([[0, 0, 1, 1]]),
        ),
        (
            np.array([[50, 50, 150, 150, 1], [25, 25, 75, 75, 2]]),
            {"height": 200, "width": 200},
            np.array([[0.25, 0.25, 0.75, 0.75, 1], [0.125, 0.125, 0.375, 0.375, 2]]),
        ),
        (
            np.array([]),
            {"height": 100, "width": 100},
            np.array([]),
        ),
    ],
)
def test_normalize_bboxes(bboxes, image_shape, expected):
    result = normalize_bboxes(bboxes, image_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_normalize_bboxes_preserves_input():
    bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    image_shape = {"height": 100, "width": 200}
    original_bboxes = bboxes.copy()

    normalize_bboxes(bboxes, image_shape)

    np.testing.assert_array_equal(bboxes, original_bboxes)


def test_normalize_bboxes_output_type():
    bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    image_shape = {"height": 100, "width": 200}

    result = normalize_bboxes(bboxes, image_shape)

    assert isinstance(result, np.ndarray)
    assert result.dtype == float


@pytest.mark.parametrize(
    "bboxes, image_shape, expected",
    [
        (
            np.array([[0.05, 0.2, 0.15, 0.4], [0.25, 0.6, 0.35, 0.8]]),
            {"height": 100, "width": 200},
            np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        ),
        (
            np.array([[0, 0, 1, 1]]),
            {"height": 100, "width": 200},
            np.array([[0, 0, 200, 100]]),
        ),
        (
            np.array([[0.25, 0.25, 0.75, 0.75, 1], [0.125, 0.125, 0.375, 0.375, 2]]),
            {"height": 200, "width": 200},
            np.array([[50, 50, 150, 150, 1], [25, 25, 75, 75, 2]]),
        ),
        (
            np.array([]),
            {"height": 100, "width": 100},
            np.array([]),
        ),
    ],
)
def test_denormalize_bboxes(bboxes, image_shape, expected):
    result = denormalize_bboxes(bboxes, image_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_denormalize_bboxes_preserves_input():
    bboxes = np.array([[0.05, 0.2, 0.15, 0.4], [0.25, 0.6, 0.35, 0.8]])
    image_shape = {"height": 100, "width": 200}
    original_bboxes = bboxes.copy()

    denormalize_bboxes(bboxes, image_shape)

    np.testing.assert_array_equal(bboxes, original_bboxes)


def test_denormalize_bboxes_output_type():
    bboxes = np.array([[0.05, 0.2, 0.15, 0.4], [0.25, 0.6, 0.35, 0.8]])
    image_shape = {"height": 100, "width": 200}

    result = denormalize_bboxes(bboxes, image_shape)

    assert isinstance(result, np.ndarray)
    assert result.dtype == float


@pytest.mark.parametrize(
    "bboxes, image_shape",
    [
        (np.array([[10, 20, 30, 40], [50, 60, 70, 80]]), {"height": 100, "width": 200}),
        (np.array([[0, 0, 200, 100]]), {"height": 100, "width": 200}),
        (np.array([[50, 50, 150, 150, 1], [25, 25, 75, 75, 2]]), {"height": 200, "width": 200}),
        (np.array([]), {"height": 100, "width": 100}),
    ],
)
def test_normalize_denormalize_roundtrip(bboxes, image_shape):
    normalized = normalize_bboxes(bboxes, image_shape)
    denormalized = denormalize_bboxes(normalized, image_shape)
    np.testing.assert_allclose(denormalized, bboxes, rtol=1e-5)


@pytest.mark.parametrize(
    "bboxes, image_shape",
    [
        (np.array([[0.05, 0.2, 0.15, 0.4], [0.25, 0.6, 0.35, 0.8]]), {"height": 100, "width": 200}),
        (np.array([[0, 0, 1, 1]]), {"height": 100, "width": 200}),
        (np.array([[0.25, 0.25, 0.75, 0.75, 1], [0.125, 0.125, 0.375, 0.375, 2]]), {"height": 200, "width": 200}),
        (np.array([]), {"height": 100, "width": 100}),
    ],
)
def test_denormalize_normalize_roundtrip(bboxes, image_shape):
    denormalized = denormalize_bboxes(bboxes, image_shape)
    normalized = normalize_bboxes(denormalized, image_shape)
    np.testing.assert_allclose(normalized, bboxes, rtol=1e-5)


@pytest.mark.parametrize(
    "bboxes, image_shape, expected",
    [
        (
            np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]),
            {"height": 100, "width": 100},
            np.array([1600, 3600]),
        ),
        (
            np.array([[0, 0, 1, 1]]),
            {"height": 200, "width": 300},
            np.array([60000]),
        ),
        (
            np.array([[0.25, 0.25, 0.75, 0.75, 1], [0.1, 0.1, 0.9, 0.9, 2]]),
            {"height": 100, "width": 100},
            np.array([2500, 6400]),
        ),
        (
            np.array([]),
            {"height": 100, "width": 100},
            np.array([]),
        ),
        (
            np.array([[0.1, 0.1, 0.3, 0.3], [0.4, 0.4, 0.6, 0.6]]),
            {"height": 50, "width": 200},
            np.array([400, 400]),
        ),
    ],
)
def test_calculate_bbox_areas(bboxes, image_shape, expected):
    result = calculate_bbox_areas_in_pixels(bboxes, image_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_calculate_bbox_areas_preserves_input():
    bboxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    image_shape = {"height": 100, "width": 100}
    original_bboxes = bboxes.copy()

    calculate_bbox_areas_in_pixels(bboxes, image_shape)

    np.testing.assert_array_equal(bboxes, original_bboxes)


def test_calculate_bbox_areas_output_type():
    bboxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    image_shape = {"height": 100, "width": 100}

    result = calculate_bbox_areas_in_pixels(bboxes, image_shape)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_calculate_bbox_areas_zero_area():
    bboxes = np.array([[0.1, 0.1, 0.1, 0.2], [0.3, 0.3, 0.4, 0.3]])  # Zero width and zero height
    image_shape = {"height": 100, "width": 100}
    result = calculate_bbox_areas_in_pixels(bboxes, image_shape)
    np.testing.assert_allclose(result, [0, 0], atol=1e-10)


@pytest.mark.parametrize(
    "bboxes, source_format, image_shape, expected",
    [
        # COCO format
        (
            np.array([[10, 20, 30, 40], [50, 60, 20, 30]]),
            "coco",
            {"height": 100, "width": 200},
            np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]]),
        ),
        # Pascal VOC format
        (
            np.array([[10, 20, 40, 60], [50, 60, 70, 90]]),
            "pascal_voc",
            {"height": 100, "width": 200},
            np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]]),
        ),
        # YOLO format
        (
            np.array([[0.25, 0.5, 0.2, 0.4], [0.7, 0.8, 0.2, 0.3]]),
            "yolo",
            {"height": 100, "width": 200},  # image shape doesn't matter for YOLO
            np.array([[0.15, 0.3, 0.35, 0.7], [0.6, 0.65, 0.8, 0.95]]),
        ),
        # With additional columns
        (
            np.array([[10, 20, 30, 40, 1], [50, 60, 20, 30, 2]]),
            "coco",
            {"height": 100, "width": 200},
            np.array([[0.05, 0.2, 0.2, 0.6, 1], [0.25, 0.6, 0.35, 0.9, 2]]),
        ),
        # Empty array
        (
            np.array([]).reshape(0, 4),
            "coco",
            {"height": 100, "width": 200},
            np.array([]).reshape(0, 4),
        ),
    ],
)
def test_convert_bboxes_to_albumentations(bboxes, source_format, image_shape, expected):
    result = convert_bboxes_to_albumentations(bboxes, source_format, image_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_convert_bboxes_to_albumentations_preserves_input():
    bboxes = np.array([[10, 20, 30, 40], [50, 60, 20, 30]])
    original_bboxes = bboxes.copy()
    convert_bboxes_to_albumentations(bboxes, "coco", {"height": 100, "width": 200})
    np.testing.assert_array_equal(bboxes, original_bboxes)


def test_convert_bboxes_to_albumentations_output_type():
    bboxes = np.array([[10, 20, 30, 40], [50, 60, 20, 30]], dtype=np.float32)
    result = convert_bboxes_to_albumentations(bboxes, "coco", {"height": 100, "width": 200})
    assert isinstance(result, np.ndarray)
    assert result.dtype == bboxes.dtype


@pytest.mark.parametrize("source_format", ["invalid_format", "COCO", "Pascal_VOC"])
def test_convert_bboxes_to_albumentations_invalid_format(source_format):
    bboxes = np.array([[10, 20, 30, 40]])
    with pytest.raises(ValueError, match="Unknown source_format"):
        convert_bboxes_to_albumentations(bboxes, source_format, (100, 200))


def test_convert_bboxes_to_albumentations_yolo_invalid_range():
    bboxes = np.array([[0, 0.5, 0.2, 0.4], [1.1, 0.8, 0.2, 0.3]])
    with pytest.raises(ValueError, match="In YOLO format all coordinates must be float and in range"):
        convert_bboxes_to_albumentations(bboxes, "yolo", (100, 200), check_validity=True)


@pytest.mark.parametrize("source_format", ["coco", "pascal_voc", "yolo"])
def test_convert_bboxes_to_albumentations_check_validity(source_format, mocker):
    bboxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    image_shape = (100, 200)
    mock_check_bboxes = mocker.patch("albumentations.core.bbox_utils.check_bboxes")

    convert_bboxes_to_albumentations(bboxes, source_format, image_shape, check_validity=True)

    mock_check_bboxes.assert_called_once()


@pytest.mark.parametrize("source_format", ["coco", "pascal_voc"])
def test_convert_bboxes_to_albumentations_calls_normalize(source_format, mocker):
    bboxes = np.array([[10, 20, 30, 40]])
    image_shape = (100, 200)
    mock_normalize_bboxes = mocker.patch("albumentations.core.bbox_utils.normalize_bboxes", return_value=bboxes)

    convert_bboxes_to_albumentations(bboxes, source_format, image_shape)

    mock_normalize_bboxes.assert_called_once()


def test_convert_bboxes_to_albumentations_yolo_does_not_call_normalize(mocker):
    bboxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    image_shape = (100, 200)
    mock_normalize_bboxes = mocker.patch("albumentations.core.bbox_utils.normalize_bboxes")

    convert_bboxes_to_albumentations(bboxes, "yolo", image_shape)

    mock_normalize_bboxes.assert_not_called()


@pytest.mark.parametrize(
    "bboxes, target_format, image_shape, expected",
    [
        # Albumentations to COCO format
        (
            np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]]),
            "coco",
            {"height": 100, "width": 200},
            np.array([[10, 20, 30, 40], [50, 60, 20, 30]]),
        ),
        # Albumentations to Pascal VOC format
        (
            np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]]),
            "pascal_voc",
            {"height": 100, "width": 200},
            np.array([[10, 20, 40, 60], [50, 60, 70, 90]]),
        ),
        # Albumentations to YOLO format
        (
            np.array([[0.15, 0.3, 0.35, 0.7], [0.6, 0.65, 0.8, 0.95]]),
            "yolo",
            {"height": 100, "width": 200},  # image shape doesn't matter for YOLO
            np.array([[0.25, 0.5, 0.2, 0.4], [0.7, 0.8, 0.2, 0.3]]),
        ),
        # With additional columns
        (
            np.array([[0.05, 0.2, 0.2, 0.6, 1], [0.25, 0.6, 0.35, 0.9, 2]]),
            "coco",
            {"height": 100, "width": 200},
            np.array([[10, 20, 30, 40, 1], [50, 60, 20, 30, 2]]),
        ),
        # Empty array
        (
            np.array([]).reshape(0, 4),
            "coco",
            {"height": 100, "width": 200},
            np.array([]).reshape(0, 4),
        ),
    ],
)
def test_convert_bboxes_from_albumentations(bboxes, target_format, image_shape, expected):
    result = convert_bboxes_from_albumentations(bboxes, target_format, image_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_convert_bboxes_from_albumentations_preserves_input():
    bboxes = np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]])
    original_bboxes = bboxes.copy()
    convert_bboxes_from_albumentations(bboxes, "coco", {"height": 100, "width": 200})
    np.testing.assert_array_equal(bboxes, original_bboxes)


def test_convert_bboxes_from_albumentations_output_type():
    bboxes = np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]])
    result = convert_bboxes_from_albumentations(bboxes, "coco", {"height": 100, "width": 200})
    assert isinstance(result, np.ndarray)
    assert result.dtype == bboxes.dtype


@pytest.mark.parametrize("target_format", ["invalid_format", "COCO", "Pascal_VOC"])
def test_convert_bboxes_from_albumentations_invalid_format(target_format):
    bboxes = np.array([[0.05, 0.2, 0.2, 0.6]])
    with pytest.raises(ValueError, match="Unknown target_format"):
        convert_bboxes_from_albumentations(bboxes, target_format, {"height": 100, "width": 200})


def test_convert_bboxes_from_albumentations_check_validity(mocker):
    bboxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    image_shape = (100, 200)
    mock_check_bboxes = mocker.patch("albumentations.core.bbox_utils.check_bboxes")

    convert_bboxes_from_albumentations(bboxes, "coco", image_shape, check_validity=True)

    mock_check_bboxes.assert_called_once()


@pytest.mark.parametrize("target_format", ["coco", "pascal_voc"])
def test_convert_bboxes_from_albumentations_calls_denormalize(target_format, mocker):
    bboxes = np.array([[0.05, 0.2, 0.2, 0.6]])
    image_shape = {"height": 100, "width": 200}
    mock_denormalize_bboxes = mocker.patch("albumentations.core.bbox_utils.denormalize_bboxes", return_value=bboxes)

    convert_bboxes_from_albumentations(bboxes, target_format, image_shape)

    mock_denormalize_bboxes.assert_called_once()


def test_convert_bboxes_from_albumentations_yolo_does_not_call_denormalize(mocker):
    bboxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    image_shape = {"height": 100, "width": 200}
    mock_denormalize_bboxes = mocker.patch("albumentations.core.bbox_utils.denormalize_bboxes")

    convert_bboxes_from_albumentations(bboxes, "yolo", image_shape)

    mock_denormalize_bboxes.assert_not_called()


@pytest.mark.parametrize(
    "original_format, image_shape",
    [
        ("coco", {"height": 100, "width": 200}),
        ("pascal_voc", {"height": 100, "width": 200}),
        ("yolo", {"height": 100, "width": 200}),
    ],
)
def test_round_trip_to_from_albumentations(original_format, image_shape):
    original_bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    if original_format == "yolo":
        original_bboxes = np.array([[0.25, 0.3, 0.2, 0.2], [0.6, 0.7, 0.2, 0.2]])

    # Convert to albumentations format
    albu_bboxes = convert_bboxes_to_albumentations(original_bboxes, original_format, image_shape)

    # Convert back to original format
    converted_bboxes = convert_bboxes_from_albumentations(albu_bboxes, original_format, image_shape)

    np.testing.assert_allclose(converted_bboxes, original_bboxes, rtol=1e-5)


@pytest.mark.parametrize(
    "target_format, image_shape",
    [
        ("coco", {"height": 100, "width": 200}),
        ("pascal_voc", {"height": 100, "width": 200}),
        ("yolo", {"height": 100, "width": 200}),
    ],
)
def test_round_trip_from_to_albumentations(target_format, image_shape):
    albu_bboxes = np.array([[0.05, 0.1, 0.15, 0.2], [0.25, 0.3, 0.35, 0.4]])

    # Convert from albumentations format
    converted_bboxes = convert_bboxes_from_albumentations(albu_bboxes, target_format, image_shape)

    # Convert back to albumentations format
    reconverted_bboxes = convert_bboxes_to_albumentations(converted_bboxes, target_format, image_shape)

    np.testing.assert_allclose(reconverted_bboxes, albu_bboxes, rtol=1e-5)


def test_check_bboxes_valid():
    valid_bboxes = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 2],
            [0.5, 0.6, 0.7, 0.8, 3],
            [0, 0, 1, 1, 0],
            [0.1, 0.1, 0.2, 0.2, 1],  # with additional column
        ],
    )
    check_bboxes(valid_bboxes)  # Should not raise any exception


@pytest.mark.parametrize(
    "invalid_bbox, error_message",
    [
        (np.array([[1.1, 0.2, 0.3, 0.4]]), "Expected x_min for bbox"),
        (np.array([[0.1, 1.2, 0.3, 0.4]]), "Expected y_min for bbox"),
        (np.array([[0.1, 0.2, 1.3, 0.4]]), "Expected x_max for bbox"),
        (np.array([[0.1, 0.2, 0.3, 1.4]]), "Expected y_max for bbox"),
        (np.array([[-0.1, 0.2, 0.3, 0.4]]), "Expected x_min for bbox"),
    ],
)
def test_check_bboxes_out_of_range(invalid_bbox, error_message):
    with pytest.raises(ValueError, match=error_message):
        check_bboxes(invalid_bbox)


def test_check_bboxes_x_max_less_than_x_min():
    invalid_bbox = np.array([[0.3, 0.2, 0.1, 0.4]])
    with pytest.raises(ValueError, match="x_max is less than or equal to x_min"):
        check_bboxes(invalid_bbox)


def test_check_bboxes_y_max_less_than_y_min():
    invalid_bbox = np.array([[0.1, 0.4, 0.3, 0.2]])
    with pytest.raises(ValueError, match="y_max is less than or equal to y_min"):
        check_bboxes(invalid_bbox)


def test_check_bboxes_close_to_zero():
    valid_bbox = np.array([[1e-8, 0.2, 0.3, 0.4]])
    check_bboxes(valid_bbox)  # Should not raise any exception


def test_check_bboxes_close_to_one():
    valid_bbox = np.array([[0.1, 0.2, 0.3, 1 - 1e-8]])
    check_bboxes(valid_bbox)  # Should not raise any exception


def test_check_bboxes_empty():
    empty_bboxes = np.array([]).reshape(0, 4)
    check_bboxes(empty_bboxes)  # Should not raise any exception


def test_check_bboxes_multiple_invalid():
    invalid_bboxes = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [1.1, 0.2, 0.3, 0.4],  # invalid x_min
            [0.5, 0.6, 0.7, 0.8],
            [0.3, 0.2, 0.1, 0.4],  # invalid x_max < x_min
        ],
    )
    with pytest.raises(ValueError, match="Expected x_min for bbox"):
        check_bboxes(invalid_bboxes)


@pytest.mark.parametrize(
    "bbox",
    [
        np.array([[0, 0.2, 0.3, 0.4]]),
        np.array([[0.1, 0, 0.3, 0.4]]),
        np.array([[0.1, 0.2, 1, 0.4]]),
        np.array([[0.1, 0.2, 0.3, 1]]),
    ],
)
def test_check_bboxes_exact_zero_and_one(bbox):
    check_bboxes(bbox)  # Should not raise any exception


def test_check_bboxes_additional_columns():
    valid_bboxes = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 1, 2, 3],
            [0.5, 0.6, 0.7, 0.8, 4, 5, 6],
        ],
    )
    check_bboxes(valid_bboxes)  # Should not raise any exception


@pytest.mark.parametrize(
    "bboxes, image_shape, min_area, min_visibility, min_width, min_height, expected",
    [
        (
            np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]),
            {"height": 100, "width": 100},
            0,
            0,
            0,
            0,
            np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]),
        ),
        (
            np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]),
            {"height": 100, "width": 100},
            200,
            0,
            0,
            0,
            np.array([]).reshape(0, 4),
        ),
        (
            np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]),
            {"height": 100, "width": 100},
            0,
            0.5,
            0,
            0,
            np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]),
        ),
        (
            np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.5, 0.4], [0.5, 0.5, 0.7, 0.6]]),
            {"height": 100, "width": 100},
            0,
            0,
            15,
            0,
            np.array([[0.3, 0.3, 0.5, 0.4], [0.5, 0.5, 0.7, 0.6]]),
        ),
        (
            np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.5], [0.5, 0.5, 0.6, 0.7]]),
            {"height": 100, "width": 100},
            0,
            0,
            0,
            15,
            np.array([[0.3, 0.3, 0.4, 0.5], [0.5, 0.5, 0.6, 0.7]]),
        ),
        (
            np.array([[0.1, 0.1, 0.2, 0.2, 1], [0.3, 0.3, 0.4, 0.4, 2], [0.5, 0.5, 0.6, 0.7, 3]]),
            {"height": 100, "width": 100},
            200,
            0,
            0,
            0,
            np.array([[0.5, 0.5, 0.6, 0.7, 3]]),
        ),
        (
            np.array([[0.1, 0.1, 0.2, 0.2, 1], [0.3, 0.3, 0.4, 0.4, 2], [0.5, 0.5, 0.6, 0.7, 3]]),
            {"height": 100, "width": 100},
            300,
            0,
            0,
            0,
            np.array([]).reshape(0, 4),
        ),
        (
            np.array([]),
            {"height": 100, "width": 100},
            0,
            0,
            0,
            0,
            np.array([]).reshape(0, 4),
        ),
        (
            np.array([[0.1, 0.1, 0.2, 0.2]]),
            {"height": 100, "width": 100},
            101,
            0,
            0,
            0,
            np.array([]).reshape(0, 4),
        ),
    ],
)
def test_filter_bboxes(bboxes, image_shape, min_area, min_visibility, min_width, min_height, expected):
    result = filter_bboxes(bboxes, image_shape, min_area, min_visibility, min_width, min_height)
    np.testing.assert_array_almost_equal(result, expected)


def test_filter_bboxes_preserves_input():
    bboxes = np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]])
    original_bboxes = bboxes.copy()
    image_shape = {"height": 100, "width": 100}

    filter_bboxes(bboxes, image_shape)

    np.testing.assert_array_equal(bboxes, original_bboxes)


def test_filter_bboxes_output_type():
    bboxes = np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]])
    image_shape = {"height": 100, "width": 100}

    result = filter_bboxes(bboxes, image_shape)

    assert isinstance(result, np.ndarray)
    assert result.dtype == bboxes.dtype


def test_filter_bboxes_clipping():
    bboxes = np.array([[-0.1, -0.1, 1.1, 1.1], [0.3, 0.3, 0.4, 0.4]])
    image_shape = {"height": 100, "width": 100}

    result = filter_bboxes(bboxes, image_shape)

    expected = np.array([[0.0, 0.0, 1.0, 1.0], [0.3, 0.3, 0.4, 0.4]])
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_filter_bboxes_noop():
    in_data = dict(
        image=np.ones((100, 100, 3)),
        bboxes=np.array([[0.1, 0.2, 1e-3, 1e-3]]),
        classes=np.array([1]),
    )
    bbox_conf = A.core.bbox_utils.BboxParams(format="yolo", label_fields=["classes"], min_area=1.0)
    transf = A.Compose([A.NoOp(p=1.0)], bbox_params=bbox_conf, is_check_shapes=False, strict=True)

    out_data = transf(**in_data)

    assert out_data["bboxes"].shape[0] == 0
    assert len(out_data["classes"]) == 0


@pytest.mark.parametrize(
    "bboxes, erosion_rate, expected",
    [
        (np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]), 0, np.array([0.1, 0.1, 0.6, 0.6])),
        (np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]), 0.5, np.array([0.225, 0.225, 0.475, 0.475])),
        (np.array([[0.1, 0.1, 0.5, 0.5]]), 0, np.array([0.1, 0.1, 0.5, 0.5])),
        (np.array([[0.1, 0.1, 0.5, 0.5]]), 1, None),
        (np.array([]), 0, None),
        (np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]), 1, None),
        (np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]), 0.9, np.array([0.325, 0.325, 0.375, 0.375])),
    ],
)
def test_union_of_bboxes(bboxes, erosion_rate, expected):
    result = union_of_bboxes(bboxes, erosion_rate)
    if expected is None:
        assert result is None
    else:
        np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_union_of_bboxes_single_bbox():
    bbox = np.array([[0.1, 0.1, 0.5, 0.5]])
    result = union_of_bboxes(bbox, 0.5)
    expected = np.array([0.1, 0.1, 0.5, 0.5])
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_union_of_bboxes_output_type():
    bboxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]])
    result = union_of_bboxes(bboxes, 0)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_union_of_bboxes_additional_columns():
    bboxes = np.array([[0.1, 0.1, 0.5, 0.5, 1], [0.2, 0.2, 0.6, 0.6, 2]])
    result = union_of_bboxes(bboxes, 0)
    expected = np.array([0.1, 0.1, 0.6, 0.6])
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_union_of_bboxes_edge_case():
    bboxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5]])
    result = union_of_bboxes(bboxes, 0.999999)
    assert result is None


def test_union_of_bboxes_precision():
    bboxes = np.array([[0.12345678, 0.12345678, 0.87654321, 0.87654321]])
    result = union_of_bboxes(bboxes, 0)
    np.testing.assert_allclose(result, bboxes[0], rtol=1e-7)


@pytest.mark.parametrize(
    "bbox_format, bboxes, labels",
    [
        ("coco", [[15, 12, 30, 40], [50, 50, 15, 40]], ["cat", "dog"]),
        ("pascal_voc", [[15, 12, 30, 40], [50, 50, 55, 60]], [1, 2]),
        ("albumentations", [[0.2, 0.3, 0.4, 0.5], [0.1, 0.1, 0.3, 0.3]], ["label1", "label2"]),
        ("yolo", [[0.15, 0.22, 0.3, 0.4], [0.5, 0.5, 0.15, 0.4]], [0, 3]),
    ],
)
def test_bbox_processor_roundtrip(bbox_format, bboxes, labels):
    params = BboxParams(format=bbox_format, label_fields=["labels"])
    processor = BboxProcessor(params)

    data = {
        "image": np.zeros((100, 100, 3)),
        "bboxes": bboxes,
        "labels": labels,
    }

    # Preprocess
    processor.preprocess(data)

    # Postprocess
    processed_data = processor.postprocess(data)

    # Check that the original bboxes and labels are recovered
    assert np.allclose(processed_data["bboxes"], bboxes, atol=1e-6)
    assert processed_data["labels"] == labels


@pytest.mark.parametrize(
    "bbox_format, bboxes, labels1, labels2",
    [
        ("coco", [[15, 12, 30, 40], [50, 50, 15, 40]], ["cat", "dog"], [1, 2]),
        ("pascal_voc", [[15, 12, 30, 40], [50, 50, 55, 60]], [1, 2], ["label1", "label2"]),
        ("albumentations", [[0.2, 0.3, 0.4, 0.5], [0.1, 0.1, 0.3, 0.3]], ["label1", "label2"], [0, 1]),
        ("yolo", [[0.15, 0.22, 0.3, 0.4], [0.5, 0.5, 0.15, 0.4]], [0, 1], ["type1", "type2"]),
    ],
)
def test_bbox_processor_roundtrip_multiple_labels(bbox_format, bboxes, labels1, labels2):
    params = BboxParams(format=bbox_format, label_fields=["labels1", "labels2"])
    processor = BboxProcessor(params)

    data = {
        "image": np.zeros((100, 100, 3)),
        "bboxes": bboxes,
        "labels1": labels1,
        "labels2": labels2,
    }

    # Preprocess
    processor.preprocess(data)

    # Postprocess
    processed_data = processor.postprocess(data)

    # Check that the original bboxes and labels are recovered
    assert np.allclose(processed_data["bboxes"], bboxes, atol=1e-6)
    assert processed_data["labels1"] == labels1
    assert processed_data["labels2"] == labels2


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
    bboxes,
    bbox_format: str,
    labels: list[int] | None,
) -> None:
    image = np.ones((100, 100, 3))
    if labels is not None:
        aug = Compose(
            [NoOp(p=1.0)],
            bbox_params={"format": bbox_format, "label_fields": ["labels"]},
            strict=True,
        )
        transformed = aug(image=image, bboxes=bboxes, labels=labels)
    else:
        aug = Compose([NoOp(p=1.0)], bbox_params={"format": bbox_format}, strict=True)
        transformed = aug(image=image, bboxes=bboxes)
    assert np.array_equal(transformed["image"], image)
    assert np.all(np.isclose(transformed["bboxes"], bboxes))


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
    bboxes,
    bbox_format: str,
    labels: dict[str, list[int]],
) -> None:
    image = np.ones((100, 100, 3))
    aug = Compose(
        [NoOp(p=1.0)],
        bbox_params={"format": bbox_format, "label_fields": list(labels.keys())},
        strict=True,
    )
    transformed = aug(image=image, bboxes=bboxes, **labels)
    assert np.array_equal(transformed["image"], image)
    assert np.allclose(transformed["bboxes"], bboxes)
    for k, v in labels.items():
        assert transformed[k] == v


def test_random_sized_crop_size() -> None:
    image = np.ones((100, 100, 3))
    bboxes = [(0.2, 0.3, 0.6, 0.8, 2), (0.3, 0.4, 0.7, 0.9, 99)]
    aug = A.Compose(
        [RandomSizedCrop(min_max_height=(70, 90), size=(50, 50), p=1.0)],
        bbox_params={"format": "albumentations"},
        seed=42,
        strict=True,
    )
    transformed = aug(image=image, bboxes=bboxes)
    assert transformed["image"].shape == (50, 50, 3)
    assert len(bboxes) == len(transformed["bboxes"])


def test_random_resized_crop_size() -> None:
    image = np.ones((100, 100, 3))
    bboxes = [(0.2, 0.3, 0.6, 0.8, 2), (0.3, 0.4, 0.7, 0.9, 99)]
    aug = A.Compose([RandomResizedCrop(size=(50, 50), p=1.0)], bbox_params={"format": "albumentations"}, seed=42, strict=True)
    transformed = aug(image=image, bboxes=bboxes)
    assert transformed["image"].shape == (50, 50, 3)
    assert len(bboxes) == len(transformed["bboxes"])


def test_random_rotate() -> None:
    image = np.ones((192, 192, 3))
    bboxes = [(78, 42, 142, 80, 1), (32, 12, 42, 72, 2)]
    aug = A.Compose([Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT)], bbox_params={"format": "pascal_voc"}, strict=True)
    transformed = aug(image=image, bboxes=bboxes)
    assert len(bboxes) == len(transformed["bboxes"])


def test_crop_boxes_replay_compose() -> None:
    image = np.ones((512, 384, 3))
    bboxes = [(78, 42, 142, 80), (32, 12, 42, 72), (200, 100, 300, 200)]
    labels = [0, 1, 2]
    transform = ReplayCompose(
        [RandomCrop(256, 256, p=1.0)],
        bbox_params=BboxParams(
            format="pascal_voc",
            min_area=16,
            label_fields=["labels"],
        ),
    )

    input_data = dict(image=image, bboxes=bboxes, labels=labels)
    transformed = transform(**input_data)
    transformed2 = ReplayCompose.replay(transformed["replay"], **input_data)

    np.testing.assert_almost_equal(transformed["bboxes"], transformed2["bboxes"])


def test_bounding_box_partially_outside_no_clip() -> None:
    """Test error is raised when bounding box exceeds image boundaries without clipping."""
    # Define a transformation with NoOp
    transform = Compose(
        [NoOp()],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
    )

    # Bounding box that exceeds the image dimensions
    bbox = (110, 50, 140, 90)  # x_min, y_min, x_max, y_max in pixel values
    labels = [1]

    # Test should raise an error since bbox is out of image bounds and clipping is not enabled
    with pytest.raises(ValueError):
        transform(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            bboxes=[bbox],
            labels=labels,
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
    image_size: tuple[int, int],
    bbox,
    expected_bbox,
) -> None:
    transform = Compose(
        [A.NoOp()],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"], "clip": True},
        strict=True,
    )
    transformed = transform(
        image=np.zeros((*image_size, 3), dtype=np.uint8),
        bboxes=[bbox],
        labels=[1],
    )
    np.testing.assert_almost_equal(transformed["bboxes"][0], expected_bbox)


@pytest.mark.parametrize(
    "bbox, expected_bbox",
    [
        ((1, 1, 1, 1), (2, 1, 1, 1)),
        ((0, 1, 1, 1), (3, 1, 1, 1)),
        ((1, 0, 1, 1), (2, 0, 1, 1)),
        ((0, 0, 1, 1), (3, 0, 1, 1)),
    ],
)
def test_bounding_box_hflip(bbox, expected_bbox) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="coco", label_fields=[]),
        strict=True,
    )

    transformed = transform(image=image, bboxes=[bbox])

    np.testing.assert_almost_equal(transformed["bboxes"][0], expected_bbox)


@pytest.mark.parametrize(
    "bbox, expected_bbox",
    [
        ((1, 1, 1, 1), (1, 2, 1, 1)),
        ((0, 1, 1, 1), (0, 2, 1, 1)),
        ((1, 0, 1, 1), (1, 3, 1, 1)),
        ((0, 0, 1, 1), (0, 3, 1, 1)),
    ],
)
def test_bounding_box_vflip(bbox, expected_bbox) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    transform = A.Compose(
        [A.VerticalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="coco", label_fields=[]),
        strict=True,
    )

    transformed = transform(image=image, bboxes=[bbox])

    np.testing.assert_almost_equal(transformed["bboxes"][0], expected_bbox)


@pytest.mark.parametrize(
    "get_transform",
    [
        lambda sign: A.Affine(translate_px=sign * 2, border_mode=cv2.BORDER_CONSTANT, fill=255),
        lambda sign: A.ShiftScaleRotate(
            shift_limit=(sign * 0.02, sign * 0.02),
            scale_limit=0,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
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
    bboxes,
    expected,
    min_visibility: float,
    sign: int,
) -> None:
    image = np.zeros([100, 100, 3], dtype=np.uint8)
    transform = get_transform(sign)
    transform.p = 1
    aug = A.Compose(
        [transform],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=min_visibility),
        strict=True,
    )

    res = aug(image=image, bboxes=bboxes)["bboxes"]
    # Use assert_allclose instead of assert_almost_equal
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)


def test_bbox_clipping_perspective() -> None:
    transform = A.Compose(
        [A.Perspective(scale=(0.05, 0.05), p=1)],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.6),
        seed=42,
        strict=True,
    )

    image = np.empty([1000, 1000, 3], dtype=np.uint8)
    bboxes = np.array([[0, 0, 100, 100, 1]])
    res = transform(image=image, bboxes=bboxes)["bboxes"]
    assert len(res) == 0


@pytest.mark.parametrize(
    "pad_top, pad_bottom, pad_left, pad_right, image_shape, expected",
    [
        # Symmetric padding
        (100, 100, 100, 100, (100, 100), {"grid_shape": (3, 3), "original_position": (1, 1)}),  # Exact multiple
        (150, 150, 150, 150, (100, 100), {"grid_shape": (5, 5), "original_position": (2, 2)}),  # Rounded up
        (50, 50, 50, 50, (100, 100), {"grid_shape": (3, 3), "original_position": (1, 1)}),  # Less than image size
        # Asymmetric padding
        (100, 0, 100, 0, (100, 100), {"grid_shape": (2, 2), "original_position": (1, 1)}),
        (0, 100, 0, 100, (100, 100), {"grid_shape": (2, 2), "original_position": (0, 0)}),
        (100, 50, 75, 25, (100, 100), {"grid_shape": (3, 3), "original_position": (1, 1)}),
        # Edge cases
        (0, 0, 0, 0, (100, 100), {"grid_shape": (1, 1), "original_position": (0, 0)}),  # No padding
        (1, 1, 1, 1, (100, 100), {"grid_shape": (3, 3), "original_position": (1, 1)}),  # Minimal padding
        # Different image dimensions
        (100, 100, 50, 50, (50, 100), {"grid_shape": (5, 3), "original_position": (2, 1)}),
        # Large padding
        (500, 500, 500, 500, (100, 100), {"grid_shape": (11, 11), "original_position": (5, 5)}),
        # Asymmetric image dimensions
        (100, 100, 100, 100, (200, 100), {"grid_shape": (3, 3), "original_position": (1, 1)}),
        # Very small image dimensions
        (10, 10, 10, 10, (5, 5), {"grid_shape": (5, 5), "original_position": (2, 2)}),
        # Very large image dimensions
        (1000, 1000, 1000, 1000, (10000, 10000), {"grid_shape": (3, 3), "original_position": (1, 1)}),
        # Zero padding on some sides
        (100, 0, 0, 100, (100, 100), {"grid_shape": (2, 2), "original_position": (1, 0)}),
        # Padding smaller than image on some sides, larger on others
        (50, 150, 25, 175, (100, 100), {"grid_shape": (4, 4), "original_position": (1, 1)}),
    ],
)
def test_get_pad_grid_dimensions(pad_top, pad_bottom, pad_left, pad_right, image_shape, expected):
    result = fgeometric.get_pad_grid_dimensions(pad_top, pad_bottom, pad_left, pad_right, image_shape)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_pad_grid_dimensions_float_values():
    result = fgeometric.get_pad_grid_dimensions(10.5, 10.5, 10.5, 10.5, (100, 100))
    assert result == {
        "grid_shape": (3, 3),
        "original_position": (1, 1),
    }, "Function should handle float inputs by implicit conversion to int"


@pytest.mark.parametrize(
    "image_shape, bboxes, pad_params, expected_bboxes",
    [
        (
            (128, 96, 3),
            np.array([[5, 60, 40, 110, 0]]),  # input bboxes (scaled down)
            (128, 128, 96, 96),  # (pad_top, pad_bottom, pad_left, pad_right)
            np.array(
                [
                    (56.0, 18.0, 91.0, 68.0, 0.0),
                    (101.0, 18.0, 136.0, 68.0, 0.0),
                    (248.0, 18.0, 283.0, 68.0, 0.0),
                    (56.0, 188.0, 91.0, 238.0, 0.0),
                    (101.0, 188.0, 136.0, 238.0, 0.0),
                    (248.0, 188.0, 283.0, 238.0, 0.0),
                    (56.0, 274.0, 91.0, 324.0, 0.0),
                    (101.0, 274.0, 136.0, 324.0, 0.0),
                    (248.0, 274.0, 283.0, 324.0, 0.0),
                ],
            ),
        ),
        # Add more test cases here
    ],
)
def test_pad_bboxes_with_reflection(image_shape, bboxes, pad_params, expected_bboxes):
    pad_top, pad_bottom, pad_left, pad_right = pad_params
    image_shape[:2]

    result = fgeometric.pad_bboxes(
        bboxes,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        border_mode=cv2.BORDER_REFLECT_101,
        image_shape=image_shape[:2],
    )

    np.testing.assert_array_almost_equal(result, expected_bboxes, decimal=1)


# Test case for non-reflect border mode
@pytest.mark.parametrize(
    "image_shape, bboxes, pad_params, expected_bboxes",
    [
        (
            (100, 100, 3),  # image shape
            np.array([[20, 20, 80, 80, 0]]),  # input bboxes
            (10, 10, 10, 10),  # (pad_top, pad_bottom, pad_left, pad_right)
            np.array([[30, 30, 90, 90, 0]]),
        ),
    ],
)
def test_pad_bboxes_constant_border(image_shape, bboxes, pad_params, expected_bboxes):
    pad_top, pad_bottom, pad_left, pad_right = pad_params

    result = fgeometric.pad_bboxes(
        bboxes,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        border_mode=cv2.BORDER_CONSTANT,
        image_shape=image_shape[:2],
    )

    np.testing.assert_array_almost_equal(result, expected_bboxes, decimal=1)


@pytest.mark.parametrize(
    "bboxes, expected",
    [
        (
            np.array([[0.1, 0.2, 0.3, 0.4]]),
            np.array([[0.1, 0.6, 0.3, 0.8]]),
        ),
        (
            np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            np.array([[0.1, 0.6, 0.3, 0.8], [0.5, 0.2, 0.7, 0.4]]),
        ),
        (
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8, 0.9]]),
            np.array([[0.1, 0.6, 0.3, 0.8, 0.5], [0.5, 0.2, 0.7, 0.4, 0.9]]),
        ),
        (
            np.array([[0, 0, 1, 1]]),
            np.array([[0, 0, 1, 1]]),
        ),
        (
            np.array([]),
            np.array([]),
        ),
    ],
)
def test_bboxes_vflip(bboxes, expected):
    flipped_bboxes = fgeometric.bboxes_vflip(bboxes)
    np.testing.assert_allclose(flipped_bboxes, expected, rtol=1e-5)


def test_bboxes_vflip_preserves_shape():
    bboxes = np.random.rand(10, 6)  # 10 bboxes with 2 extra columns
    flipped_bboxes = fgeometric.bboxes_vflip(bboxes)
    assert flipped_bboxes.shape == bboxes.shape


def test_bboxes_vflip_inplace():
    bboxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    original_bboxes = bboxes.copy()
    flipped_bboxes = fgeometric.bboxes_vflip(bboxes)
    assert not np.array_equal(flipped_bboxes, original_bboxes)
    assert np.array_equal(bboxes, original_bboxes)


@pytest.mark.parametrize(
    "bboxes, expected",
    [
        (
            np.array([[0.1, 0.2, 0.3, 0.4]]),
            np.array([[0.7, 0.2, 0.9, 0.4]]),
        ),
        (
            np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            np.array([[0.7, 0.2, 0.9, 0.4], [0.3, 0.6, 0.5, 0.8]]),
        ),
        (
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8, 0.9]]),
            np.array([[0.7, 0.2, 0.9, 0.4, 0.5], [0.3, 0.6, 0.5, 0.8, 0.9]]),
        ),
        (
            np.array([[0, 0, 1, 1]]),
            np.array([[0, 0, 1, 1]]),
        ),
        (
            np.array([]),
            np.array([]),
        ),
    ],
)
def test_bboxes_hflip(bboxes, expected):
    flipped_bboxes = fgeometric.bboxes_hflip(bboxes)
    np.testing.assert_allclose(flipped_bboxes, expected, rtol=1e-5)


def test_bboxes_hflip_preserves_shape():
    bboxes = np.random.rand(10, 6)  # 10 bboxes with 2 extra columns
    flipped_bboxes = fgeometric.bboxes_hflip(bboxes)
    assert flipped_bboxes.shape == bboxes.shape


def test_bboxes_hflip_inplace():
    bboxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    original_bboxes = bboxes.copy()
    flipped_bboxes = fgeometric.bboxes_hflip(bboxes)
    assert not np.array_equal(flipped_bboxes, original_bboxes)
    assert np.array_equal(bboxes, original_bboxes)  # Original array should not be modified


def test_bboxes_hflip_symmetry():
    bboxes = np.random.rand(5, 4)
    flipped_once = fgeometric.bboxes_hflip(bboxes)
    flipped_twice = fgeometric.bboxes_hflip(flipped_once)
    np.testing.assert_allclose(bboxes, flipped_twice, rtol=1e-5)


def test_bboxes_hflip_extreme_values():
    bboxes = np.array([[0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9]])
    expected = np.array([[0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9]])
    flipped_bboxes = fgeometric.bboxes_hflip(bboxes)
    np.testing.assert_allclose(flipped_bboxes, expected, rtol=1e-5)


@pytest.mark.parametrize(
    "bboxes, crop_coords, image_shape, expected_bboxes",
    [
        # Test case 1: Single bbox, partial crop
        (
            np.array([[0.1, 0.1, 0.6, 0.6]]),
            (50, 50, 150, 150),
            (200, 200),
            np.array([[-0.3, -0.3, 0.7, 0.7]]),
        ),
        # Test case 2: Multiple bboxes, some outside crop
        (
            np.array(
                [
                    [0.1, 0.1, 0.3, 0.3],
                    [0.4, 0.4, 0.8, 0.8],
                    [0.7, 0.7, 0.9, 0.9],
                ],
            ),
            (100, 100, 180, 180),
            (200, 200),
            np.array(
                [[-1.0, -1.0, -0.5, -0.5], [-0.25, -0.25, 0.75, 0.75], [0.5, 0.5, 1.0, 1.0]],
            ),
        ),
        # Test case 3: Bbox with additional columns
        (
            np.array([[0.2, 0.2, 0.7, 0.7, 1, 2, 3]]),
            (40, 40, 160, 160),
            (200, 200),
            np.array([[4.967054e-09, 4.967054e-09, 8.333333e-01, 8.333333e-01, 1.000000e00, 2.000000e00, 3.000000e00]]),
        ),
    ],
)
def test_crop_bboxes_by_coords(bboxes, crop_coords, image_shape, expected_bboxes):
    result = crop_bboxes_by_coords(bboxes, crop_coords, image_shape)
    np.testing.assert_array_almost_equal(result, expected_bboxes, decimal=6)


def test_crop_bboxes_by_coords_empty_input():
    result = crop_bboxes_by_coords(np.array([]), (50, 50, 150, 150), (200, 200))
    assert result.size == 0


def test_bboxes_rot90():
    bboxes = np.array([[0.1, 0.2, 0.3, 0.4]])

    np.testing.assert_array_almost_equal(fgeometric.bboxes_rot90(bboxes, 0)[0], (0.1, 0.2, 0.3, 0.4))
    np.testing.assert_array_almost_equal(fgeometric.bboxes_rot90(bboxes, 1)[0], (0.2, 0.7, 0.4, 0.9))
    np.testing.assert_array_almost_equal(fgeometric.bboxes_rot90(bboxes, 2)[0], (0.7, 0.6, 0.9, 0.8))
    np.testing.assert_array_almost_equal(fgeometric.bboxes_rot90(bboxes, 3)[0], (0.6, 0.1, 0.8, 0.3))


def test_bboxes_transpose():
    bboxes = np.array([[0.7, 0.1, 0.8, 0.4]])
    assert np.allclose(fgeometric.bboxes_transpose(bboxes), (0.1, 0.7, 0.4, 0.8))
    rot90 = fgeometric.bboxes_rot90(bboxes, 2)
    reflected_anti_diagonal = fgeometric.bboxes_transpose(rot90)
    assert np.allclose(reflected_anti_diagonal, (0.6, 0.2, 0.9, 0.3))


@pytest.mark.parametrize(
    "bbox, group_member, expected",
    [
        ((0.05, 0.1, 0.55, 0.6), "e", (0.05, 0.1, 0.55, 0.6)),  # Identity
        ((0.05, 0.1, 0.55, 0.6), "r90", (0.1, 0.45, 0.6, 0.95)),  # Rotate 90 degrees CCW
        ((0.05, 0.1, 0.55, 0.6), "r180", (0.45, 0.4, 0.95, 0.9)),  # Rotate 180 degrees
        ((0.05, 0.1, 0.55, 0.6), "r270", (0.4, 0.05, 0.9, 0.55)),  # Rotate 270 degrees CCW
        ((0.05, 0.1, 0.55, 0.6), "v", (0.05, 0.4, 0.55, 0.9)),  # Vertical flip
        ((0.05, 0.1, 0.55, 0.6), "t", (0.1, 0.05, 0.6, 0.55)),  # Transpose around main diagonal
        ((0.05, 0.1, 0.55, 0.6), "h", (0.45, 0.1, 0.95, 0.6)),  # Horizontal flip
        ((0.05, 0.1, 0.55, 0.6), "hvt", (1 - 0.6, 1 - 0.55, 1 - 0.1, 1 - 0.05)),  # Transpose around second diagonal
    ],
)
def test_bbox_d4(bbox, group_member, expected):
    bboxes = np.array([bbox])
    result = fgeometric.bboxes_d4(bboxes, group_member)[0]
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "bbox_format, bbox, expected",
    [
        ("coco", [[1.0, 1.0, 0.75, 0.75]], [[1.0, 1.0, 0.75, 0.75]]),
        ("coco", [[1.0, 1.0, 1e-3, 1e-3]], [[1.0, 1.0, 1e-3, 1e-3]]),
        ("yolo", [[0.25, 0.25, 0.1875, 0.1875]], [[0.25, 0.25, 0.1875, 0.1875]]),
        ("yolo", [[0.25, 0.25, 1e-3, 1e-3]], [[0.25, 0.25, 1e-3, 1e-3]]),
        ("yolo", [[0.1, 0.2, 1e-3, 1e-3]], [[0.1, 0.2, 1e-3, 1e-3]]),
        ("pascal_voc", [[1, 1, 2, 2]], [[1, 1, 2, 2]]),
        ("pascal_voc", [[1, 1, 1.004, 1.004]], [[1, 1, 1.004, 1.004]]),
    ],
)
def test_small_bbox(bbox_format, bbox, expected):
    transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format=bbox_format, label_fields=["category_id"]),
        strict=True,
    )
    transformed = transform(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        bboxes=bbox,
        category_id=[1] * len(bbox),
    )

    np.testing.assert_array_almost_equal(transformed["bboxes"], expected)


@pytest.mark.parametrize(
    "bbox_format, bboxes, expected",
    [
        ("coco", np.array([[0.1, 0.2, 1e-3, 1e-3]]), np.array([[0.1, 0.2, 1e-3, 1e-3]])),
        ("yolo", np.array([[0.1, 0.2, 1e-3, 1e-3]]), np.array([[0.1, 0.2, 1e-3, 1e-3]])),
        ("pascal_voc", np.array([[1, 1, 1.001, 1.001]]), np.array([[1, 1, 1.001, 1.001]])),
    ],
)
def test_very_small_bbox(bbox_format, bboxes, expected):
    transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format=bbox_format, label_fields=["category_id"]),
        strict=True,
    )

    categories = [1]

    transformed = transform(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        bboxes=bboxes,
        category_id=categories,
    )

    np.testing.assert_array_almost_equal(transformed["bboxes"], expected)
    np.testing.assert_array_almost_equal(transformed["category_id"], categories)


@pytest.mark.parametrize(
    "masks, expected_bboxes",
    [
        # Original 3D test cases
        (np.array([[[0, 1, 1], [1, 1, 0], [0, 1, 0]]]), np.array([[0, 0, 3, 3]])),
        (np.array([[[1, 1], [1, 1]], [[0, 1], [1, 0]]]), np.array([[0, 0, 2, 2], [0, 0, 2, 2]])),
        (np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]]]), np.array([[-1, -1, -1, -1], [0, 0, 1, 1]])),

        # New 2D test cases
        (np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]), np.array([[0, 0, 3, 3]])),
        (np.array([[1, 1], [1, 1]]), np.array([[0, 0, 2, 2]])),
        (np.array([[0, 0], [0, 0]]), np.array([[-1, -1, -1, -1]])),
        (np.array([[1, 0], [0, 0]]), np.array([[0, 0, 1, 1]])),
        (np.array([[0, 1], [0, 1]]), np.array([[1, 0, 2, 2]])),
    ],
)
def test_bboxes_from_masks(masks, expected_bboxes):
    result = bboxes_from_masks(masks)
    np.testing.assert_array_equal(result, expected_bboxes)


@pytest.mark.parametrize(
    "bboxes, img_shape, expected_masks",
    [
        (np.array([[0, 0, 2, 2]]), (3, 3), np.array([[[1, 1, 0], [1, 1, 0], [0, 0, 0]]])),
        (
            np.array([[1, 1, 3, 3], [0, 0, 2, 2]]),
            (4, 4),
            np.array(
                [
                    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                    [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                ],
            ),
        ),
    ],
)
def test_masks_from_bboxes(bboxes, img_shape, expected_masks):
    result = masks_from_bboxes(bboxes, img_shape)
    np.testing.assert_array_equal(result, expected_masks)


@pytest.mark.parametrize(
    "original_masks",
    [
        np.array([[[0, 1, 1], [1, 1, 0], [0, 1, 0]]]),
        np.array([[[1, 1, 1], [1, 1, 1]], [[0, 1, 0], [1, 1, 1]]]),
        np.array([[[0, 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 0, 0]]]),
    ],
)
def test_inverse_relationship(original_masks):
    img_shape = original_masks.shape[1:]
    bboxes = bboxes_from_masks(original_masks)
    reconstructed_masks = masks_from_bboxes(bboxes, img_shape)

    for original, reconstructed in zip(original_masks, reconstructed_masks):
        # Check if the reconstructed mask fully contains the original mask
        assert np.all(original <= reconstructed)

        # Check if the bounding box of the reconstructed mask matches the original bounding box
        original_bbox = bboxes_from_masks(original[np.newaxis, ...])[0]
        reconstructed_bbox = bboxes_from_masks(reconstructed[np.newaxis, ...])[0]
        np.testing.assert_array_equal(original_bbox, reconstructed_bbox)


def test_bboxes_from_masks_preserves_input():
    original_masks = np.array([[[0, 1, 1], [1, 1, 0], [0, 1, 0]]])
    original_masks_copy = original_masks.copy()
    bboxes_from_masks(original_masks)
    np.testing.assert_array_equal(original_masks, original_masks_copy)


def test_masks_from_bboxes_output_type():
    bboxes = np.array([[0, 0, 2, 2]])
    img_shape = (3, 3)
    result = masks_from_bboxes(bboxes, img_shape)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8


def test_bboxes_from_masks_output_type():
    masks = np.array([[[0, 1, 1], [1, 1, 0], [0, 1, 0]]])
    result = bboxes_from_masks(masks)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32


def test_random_resized_crop():
    transform = A.Compose(
    [
        A.RandomResizedCrop((100, 100), scale=(0.01, 0.1), ratio=(1, 1)),
    ],
    bbox_params=A.BboxParams(
        format="coco",
            label_fields=["label"],
        ),
        strict=True,
    )
    boxes = [[10,10,20,20], [5,5,10,10], [450, 450, 5,5], [250,250,5,5]]
    labels = [1,2,3,4]
    res = transform(image=np.zeros((500,500,3), dtype='uint8'), bboxes=boxes, label=labels)
    assert len(res['bboxes']) == len(res['label'])


@pytest.mark.parametrize(
    ["bboxes", "map_x", "map_y", "image_shape", "expected"],
    [
        # Test case 1: Identity mapping (no distortion)
        (
            np.array([[10, 20, 30, 40]]),  # single bbox
            np.tile(np.arange(100), (100, 1)),  # identity map_x
            np.tile(np.arange(100).reshape(-1, 1), (1, 100)),  # identity map_y
            (100, 100),
            np.array([[10, 20, 30, 40]]),
        ),

        # Test case 2: Simple translation
        (
            np.array([[10, 20, 30, 40]]),
            np.tile(np.arange(100) - 5, (100, 1)),  # shift right by 5
            np.tile(np.arange(100).reshape(-1, 1) - 5, (1, 100)),  # shift down by 5
            (100, 100),
            np.array([[15, 25, 35, 45]]),
        ),

        # Test case 3: Multiple bboxes with additional attributes
        (
            np.array([
                [10, 20, 30, 40, 1],  # bbox with class label
                [50, 60, 70, 80, 2],
            ]),
            np.tile(np.arange(100), (100, 1)),  # identity map_x
            np.tile(np.arange(100).reshape(-1, 1), (1, 100)),  # identity map_y
            (100, 100),
            np.array([
                [10, 20, 30, 40, 1],
                [50, 60, 70, 80, 2],
            ]),
        ),

        # Test case 4: Boundary conditions
        (
            np.array([[0, 0, 10, 10]]),  # bbox at image corner
            np.tile(np.arange(100), (100, 1)),
            np.tile(np.arange(100).reshape(-1, 1), (1, 100)),
            (100, 100),
            np.array([[0, 0, 10, 10]]),
        ),

        # Test case 5: Empty array
        (
            np.zeros((0, 4)),  # empty bbox array
            np.tile(np.arange(100), (100, 1)),
            np.tile(np.arange(100).reshape(-1, 1), (1, 100)),
            (100, 100),
            np.zeros((0, 4)),
        ),
    ],
)
def test_distortion_bboxes(bboxes, map_x, map_y, image_shape, expected):
    result = fgeometric.remap_bboxes(bboxes, map_x, map_y, image_shape)
    np.testing.assert_array_almost_equal(result, expected)


def test_distortion_bboxes_complex_distortion():
    # Test with a more complex distortion pattern
    bboxes = np.array([[25, 25, 75, 75]])  # center box
    image_shape = (100, 100)

    # Create a radial distortion pattern
    y, x = np.mgrid[0:100, 0:100]
    c_x, c_y = 50, 50  # distortion center
    r = np.sqrt((x - c_x)**2 + (y - c_y)**2)
    factor = 1 + r/100  # increasing distortion with radius

    map_x = x + (x - c_x) / factor
    map_y = y + (y - c_y) / factor

    result = fgeometric.remap_bboxes(bboxes, map_x, map_y, image_shape)

    # Check that the result is different from input but still valid
    assert not np.array_equal(result, bboxes)
    assert np.all(result >= 0)
    assert np.all(result[:, [0, 2]] <= image_shape[1])  # x coordinates
    assert np.all(result[:, [1, 3]] <= image_shape[0])  # y coordinates
    assert np.all(result[:, [0, 1]] <= result[:, [2, 3]])  # min <= max


import numpy as np
import pytest

from albumentations.augmentations.geometric.functional import bboxes_grid_shuffle


def test_bboxes_grid_shuffle_basic():
    """Test basic functionality of bboxes_grid_shuffle."""
    # Create a simple test case with one bbox covering a 2x2 grid
    image_shape = (100, 100)
    bboxes = np.array([[25, 25, 75, 75]])  # Single bbox in the middle
    tiles = np.array([
        [0, 0, 50, 50],     # top-left
        [0, 50, 50, 100],   # top-right
        [50, 0, 100, 50],   # bottom-left
        [50, 50, 100, 100], # bottom-right
    ])
    mapping = [3, 2, 1, 0]  # Rotate tiles counter-clockwise

    result = bboxes_grid_shuffle(bboxes, tiles, mapping, image_shape, None, None)

    assert len(result) > 0  # Should have at least one bbox
    assert result.shape[1] == 4  # Each bbox should have 4 coordinates
    assert np.all(result >= 0) and np.all(result[:, [0,2]] <= image_shape[1]) and np.all(result[:, [1,3]] <= image_shape[0])
    assert np.all(result[:, 0] < result[:, 2]) and np.all(result[:, 1] < result[:, 3])


def test_bboxes_grid_shuffle_with_min_area():
    """Test bboxes_grid_shuffle with min_area filter."""
    image_shape = (100, 100)
    bboxes = np.array([[10, 10, 30, 30], [0, 0, 60, 60]])
    tiles = np.array([
        [0, 0, 50, 50],
        [0, 50, 50, 100],
        [50, 0, 100, 50],
        [50, 50, 100, 100],
    ])
    mapping = [3, 2, 1, 0]
    min_area = 2500
    result = bboxes_grid_shuffle(bboxes, tiles, mapping, image_shape, min_area, None)
    assert len(result) == 1


def test_bboxes_grid_shuffle_with_min_visibility():
    """Test bboxes_grid_shuffle with min_visibility filter."""
    image_shape = (100, 100)
    # Create a bbox that crosses tile boundaries
    bboxes = np.array([[20, 20, 80, 80]])  # Center box crossing all four tiles
    tiles = np.array([
        [0, 0, 50, 50],
        [0, 50, 50, 100],
        [50, 0, 100, 50],
        [50, 50, 100, 100],
    ])
    # Move diagonal tiles to opposite corners to split the bbox
    mapping = [3, 1, 2, 0]  # This will definitely split the bbox
    min_visibility = 0.6  # Each component should be less than 60% of original

    result = bboxes_grid_shuffle(bboxes, tiles, mapping, image_shape, None, min_visibility)

    assert len(result) == 0  # All components should be filtered out due to low visibility

def test_bboxes_grid_shuffle_with_extra_fields():
    """Test bboxes_grid_shuffle with additional bbox fields."""
    image_shape = (100, 100)
    bboxes = np.array([[25, 25, 75, 75, 1, 0.9]])  # bbox with class_id and score
    tiles = np.array([
        [0, 0, 50, 50],
        [0, 50, 50, 100],
        [50, 0, 100, 50],
        [50, 50, 100, 100],
    ])
    mapping = [3, 2, 1, 0]

    result = fgeometric.bboxes_grid_shuffle(bboxes, tiles, mapping, image_shape, None, None)

    assert result.shape[1] == 6  # Should preserve extra fields
    assert np.all(result[:, 4:] == [1, 0.9])  # Extra fields should remain unchanged


def test_bboxes_grid_shuffle_empty_input():
    """Test bboxes_grid_shuffle with empty input."""
    image_shape = (100, 100)
    bboxes = np.zeros((0, 4))
    tiles = np.array([
        [0, 0, 50, 50],
        [0, 50, 50, 100],
        [50, 0, 100, 50],
        [50, 50, 100, 100],
    ])
    mapping = [3, 2, 1, 0]

    result = fgeometric.bboxes_grid_shuffle(bboxes, tiles, mapping, image_shape, None, None)

    assert len(result) == 0
    assert result.shape[1] == 4


def test_bboxes_grid_shuffle_multiple_components():
    """Test bboxes_grid_shuffle when bbox splits into multiple components."""
    image_shape = (100, 100)
    # Create a bbox that crosses tile boundaries
    bboxes = np.array([[20, 20, 80, 80]])  # Center box crossing all four tiles
    tiles = np.array([
        [0, 0, 50, 50],
        [0, 50, 50, 100],
        [50, 0, 100, 50],
        [50, 50, 100, 100],
    ])
    # Move diagonal tiles to opposite corners to split the bbox
    mapping = [3, 1, 2, 0]  # This will definitely split the bbox

    result = bboxes_grid_shuffle(bboxes, tiles, mapping, image_shape, None, None)

    assert len(result) > 1  # Should split into multiple components
    assert np.all(result >= 0)  # All coordinates should be valid
    assert np.all(result[:, [0, 2]] <= image_shape[1])  # x coordinates within image width
    assert np.all(result[:, [1, 3]] <= image_shape[0])  # y coordinates within image height



@pytest.mark.parametrize("test_case", [
    {
        "name": "single_bbox",
        "bboxes": np.array([[10, 20, 30, 40]]),
        "image_shape": (100, 100),
        "expected_mask_shape": (100, 100, 1),
        "expected_nonzero": [(20, 10, 0), (20, 30, 0), (40, 10, 0), (40, 30, 0)]  # y, x, channel
    },
    {
        "name": "multiple_bboxes",
        "bboxes": np.array([
            [10, 20, 30, 40],
            [50, 60, 70, 80]
        ]),
        "image_shape": (100, 100),
        "expected_mask_shape": (100, 100, 2),
        "expected_nonzero": [(20, 10, 0), (60, 50, 1)]  # y, x, channel
    },
    {
        "name": "bbox_with_extra_fields",
        "bboxes": np.array([[10, 20, 30, 40, 1, 0.8]]),
        "image_shape": (100, 100),
        "expected_mask_shape": (100, 100, 1),
        "expected_nonzero": [(20, 10, 0)]
    },
    {
        "name": "bbox_at_edges",
        "bboxes": np.array([[0, 0, 99, 99]]),
        "image_shape": (100, 100),
        "expected_mask_shape": (100, 100, 1),
        "expected_nonzero": [(0, 0, 0), (99, 99, 0)]
    }
])
def test_bboxes_to_mask(test_case):
    bboxes = test_case["bboxes"]
    image_shape = test_case["image_shape"]
    expected_shape = test_case["expected_mask_shape"]
    expected_nonzero = test_case["expected_nonzero"]

    masks = bboxes_to_mask(bboxes, image_shape)

    # Check shape
    assert masks.shape == expected_shape

    # Check dtype
    assert masks.dtype == np.uint8

    # Check if masks are binary
    assert np.all(np.isin(masks, [0, 1]))

    # Check specific points
    for y, x, c in expected_nonzero:
        assert masks[y, x, c] == 1, f"Expected 1 at position ({y}, {x}, {c})"

@pytest.mark.parametrize("test_case", [
    {
        "name": "single_mask",
        "masks": np.array([  # (3, 3, 1) mask
            [[0], [0], [0]],
            [[0], [1], [0]],
            [[0], [0], [0]]
        ], dtype=np.uint8),
        "original_bboxes": np.array([[0, 0, 2, 2]]),
        "expected_bboxes": np.array([[1, 1, 1, 1]])
    },
    {
        "name": "multiple_masks",
        "masks": np.array([  # (3, 3, 2) mask
            [[0, 0], [0, 1], [0, 0]],
            [[0, 1], [1, 1], [0, 1]],
            [[0, 0], [0, 1], [0, 0]]
        ], dtype=np.uint8),
        "original_bboxes": np.array([
            [0, 0, 2, 2],
            [0, 0, 2, 2]
        ]),
        "expected_bboxes": np.array([
            [1, 1, 1, 1],
            [0, 0, 2, 2]
        ])
    },
    {
        "name": "mask_with_extra_fields",
        "masks": np.array([  # (3, 3, 1) mask
            [[0], [0], [0]],
            [[0], [1], [0]],
            [[0], [0], [0]]
        ], dtype=np.uint8),
        "original_bboxes": np.array([[0, 0, 2, 2, 1, 0.8]]),
        "expected_bboxes": np.array([[1, 1, 1, 1, 1, 0.8]])
    },
    {
        "name": "empty_mask",
        "masks": np.zeros((3, 3, 1), dtype=np.uint8),
        "original_bboxes": np.array([[10, 20, 30, 40]]),
        "expected_bboxes": np.array([[10, 20, 30, 40]])  # Should preserve original bbox
    }
])
def test_mask_to_bboxes(test_case):
    masks = test_case["masks"]
    original_bboxes = test_case["original_bboxes"]
    expected_bboxes = test_case["expected_bboxes"]

    result = mask_to_bboxes(masks, original_bboxes)

    # Check shape and values
    assert result.shape == expected_bboxes.shape
    np.testing.assert_array_equal(result, expected_bboxes)

    # Check extra fields preservation
    if original_bboxes.shape[1] > 4:
        assert np.all(result[:, 4:] == original_bboxes[:, 4:])

def test_empty_bboxes():
    empty_bboxes = np.zeros((0, 4))
    image_shape = (100, 100)

    # Test bboxes_to_mask with empty input
    masks = bboxes_to_mask(empty_bboxes, image_shape)
    assert masks.shape == (100, 100, 0)

    # Test mask_to_bboxes with empty input
    result = mask_to_bboxes(masks, empty_bboxes)
    assert result.shape == (0, 4)


@pytest.mark.parametrize(
    "bbox_format, bboxes",
    [
        # Test with invalid bboxes that should raise error when filter_invalid_bboxes=False
        ("pascal_voc", [[10, 10, 5, 20]]),  # x_max < x_min
        ("pascal_voc", [[10, 10, 10, 5]]),  # y_max < y_min
        ("coco", [[10, 10, -5, 20]]),  # negative width
        ("coco", [[10, 10, 20, -5]]),  # negative height
        ("yolo", [[0.5, 0.5, -0.1, 0.2]]),  # negative width
    ]
)
def test_bbox_processor_invalid_no_filter(bbox_format, bboxes):
    """Test that BboxProcessor raises error on invalid bboxes when filter_invalid_bboxes=False."""
    params = BboxParams(format=bbox_format, filter_invalid_bboxes=False)
    processor = BboxProcessor(params)

    data = {
        "image": np.zeros((100, 100, 3)),
        "bboxes": bboxes,
    }

    # Should raise ValueError due to invalid bboxes
    with pytest.raises(ValueError):
        processor.preprocess(data)

@pytest.mark.parametrize(
    "bbox_format, bboxes, expected_bboxes",
    [
        # COCO format: [x_min, y_min, width, height]
        ("coco", [[10, 10, -5, 20], [10, 10, 20, 20]], [[10, 10, 20, 20]]),

        # Pascal VOC format: [x_min, y_min, x_max, y_max]
        ("pascal_voc", [[10, 10, 5, 20], [10, 10, 30, 30]], [[10, 10, 30, 30]]),

        # Albumentations format: normalized [x_min, y_min, x_max, y_max]
        ("albumentations", [[0.1, 0.1, 0.05, 0.2], [0.1, 0.1, 0.3, 0.3]], [[0.1, 0.1, 0.3, 0.3]]),

        # YOLO format: normalized [x_center, y_center, width, height]
        ("yolo", [[0.5, 0.5, -0.1, 0.2], [0.5, 0.5, 0.2, 0.2]], [[0.5, 0.5, 0.2, 0.2]]),

        # Test with empty bboxes array
        ("pascal_voc", [], []),

        # Test with additional columns (labels)
        ("pascal_voc", [[10, 10, 5, 20, 1], [10, 10, 30, 30, 2]], [[10, 10, 30, 30, 2]]),
    ]
)
def test_bbox_processor_filter_invalid(bbox_format, bboxes, expected_bboxes):
    """Test that BboxProcessor correctly filters invalid bboxes when filter_invalid_bboxes=True."""
    params = BboxParams(format=bbox_format, filter_invalid_bboxes=True)
    processor = BboxProcessor(params)

    data = {
        "image": np.zeros((100, 100, 3)),
        "bboxes": bboxes,
    }

    # Preprocess data (this is where filtering should occur)
    processor.preprocess(data)

    # Convert filtered bboxes back to original format for comparison
    if bbox_format != "albumentations":
        data["bboxes"] = convert_bboxes_from_albumentations(
            data["bboxes"],
            bbox_format,
            {"height": 100, "width": 100}
        )

    # Check that invalid bboxes were filtered out
    assert len(data["bboxes"]) == len(expected_bboxes)
    if len(expected_bboxes) > 0:
        np.testing.assert_allclose(data["bboxes"], expected_bboxes)


def test_bbox_processor_clip_and_filter():
    """Test that BboxProcessor correctly handles both clipping and filtering."""
    params = BboxParams(
        format="pascal_voc",
        filter_invalid_bboxes=True,
        clip=True
    )
    processor = BboxProcessor(params)

    # Create a bbox that extends beyond image boundaries and would be invalid
    # without clipping
    data = {
        "image": np.zeros((100, 100, 3)),
        "bboxes": [[80, 80, 120, 120]],  # Goes beyond image boundaries
    }

    processor.preprocess(data)

    # Convert back to pascal_voc format for comparison
    result = convert_bboxes_from_albumentations(
        data["bboxes"],
        "pascal_voc",
        {"height": 100, "width": 100}
    )

    # After clipping, the bbox should be valid and preserved
    expected = [[80, 80, 100, 100]]  # Clipped to image boundaries
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ["bboxes", "classes", "scores", "format", "expected_bboxes", "expected_classes", "expected_scores", "clip"],
    [
        # YOLO format tests
        (
            np.array([[0.3, 0.4, 0.2, 0.3], [0.5, 0.6, 0.1, 0.2]], dtype=np.float32),
            np.array([2, 3], dtype=np.int32),
            np.array([0.9, 0.8], dtype=np.float32),
            "yolo",
            np.array([[0.3, 0.4, 0.2, 0.3], [0.5, 0.6, 0.1, 0.2]], dtype=np.float32),
            np.array([2, 3], dtype=np.int32),
            np.array([0.9, 0.8], dtype=np.float32),
            False,
        ),
        (
            np.array([[0.9, 0.8, 0.3, 0.3], [-0.1, 0.6, 0.3, 0.2]], dtype=np.float32),
            np.array([2, 3], dtype=np.int32),
            np.array([0.9, 0.8], dtype=np.float32),
            "yolo",
            np.array([[0.875, 0.8, 0.25, 0.3], [0.025, 0.6, 0.05, 0.2]], dtype=np.float32),
            np.array([2, 3], dtype=np.int32),
            np.array([0.9, 0.8], dtype=np.float32),
            True,
        ),

        # COCO format tests [x_min, y_min, width, height]
        (
            np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.7, 0.8], dtype=np.float32),
            "coco",
            np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.7, 0.8], dtype=np.float32),
            False,
        ),
        (
            np.array([[-10, -10, 30, 40], [90, 80, 30, 40]], dtype=np.float32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.7, 0.8], dtype=np.float32),
            "coco",
            np.array([[0, 0, 20, 30], [90, 80, 10, 20]], dtype=np.float32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.7, 0.8], dtype=np.float32),
            True,
        ),

        # Pascal VOC format tests [x_min, y_min, x_max, y_max]
        (
            np.array([[10, 10, 30, 30], [40, 40, 60, 60]], dtype=np.float32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.7, 0.8], dtype=np.float32),
            "pascal_voc",
            np.array([[10, 10, 30, 30], [40, 40, 60, 60]], dtype=np.float32),
            np.array([1, 2], dtype=np.int32),
            np.array([0.7, 0.8], dtype=np.float32),
            False,
        ),
        (
            np.array([[-10, -10, 30, 30], [80, 80, 120, 120]], dtype=np.float32),
            np.array([4, 5], dtype=np.int32),
            np.array([0.7, 0.8], dtype=np.float32),
            "pascal_voc",
            np.array([[0, 0, 30, 30], [80, 80, 100, 100]], dtype=np.float32),
            np.array([4, 5], dtype=np.int32),
            np.array([0.7, 0.8], dtype=np.float32),
            True,
        ),

        # Empty arrays tests for each format
        (
            np.zeros((0, 4), dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            "yolo",
            np.zeros((0, 4), dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            False,
        ),
        (
            np.zeros((0, 4), dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            "coco",
            np.zeros((0, 4), dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            False,
        ),
        (
            np.zeros((0, 4), dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            "pascal_voc",
            np.zeros((0, 4), dtype=np.float32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            False,
        ),

        # Single bbox tests with high class id
        (
            np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            np.array([100], dtype=np.int32),
            np.array([0.95], dtype=np.float32),
            "yolo",
            np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            np.array([100], dtype=np.int32),
            np.array([0.95], dtype=np.float32),
            False,
        ),

        # Edge cases for each format
        (
            np.array([[0.999, 0.999, 0.002, 0.002]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.9], dtype=np.float32),
            "yolo",
            np.array([[0.999, 0.999, 0.002, 0.002]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.9], dtype=np.float32),
            False,
        ),
        (
            np.array([[98, 98, 4, 4]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.9], dtype=np.float32),
            "coco",
            np.array([[98, 98, 2, 2]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.9], dtype=np.float32),
            True,
        ),
        (
            np.array([[98, 98, 102, 102]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.9], dtype=np.float32),
            "pascal_voc",
            np.array([[98, 98, 100, 100]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.9], dtype=np.float32),
            True,
        ),
(
            np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.95], dtype=np.float32),  # Single float in numpy array
            "yolo",
            np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.95], dtype=np.float32),  # Should preserve float32 type and value
            False,
        ),
        (
            np.array([[0.3, 0.4, 0.2, 0.3], [0.5, 0.6, 0.1, 0.2]], dtype=np.float32),
            np.array([2, 3], dtype=np.int32),
            np.array([0.8, 0.7], dtype=np.float64),  # Test float64 dtype
            "yolo",
            np.array([[0.3, 0.4, 0.2, 0.3], [0.5, 0.6, 0.1, 0.2]], dtype=np.float32),
            np.array([2, 3], dtype=np.int32),
            np.array([0.8, 0.7], dtype=np.float64),  # Should preserve float64 type
            False,
        ),
        (
            np.array([[10, 20, 30, 40]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.99999], dtype=np.float32),  # Test very high confidence
            "pascal_voc",
            np.array([[10, 20, 30, 40]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.99999], dtype=np.float32),  # Should preserve exact value
            False,
        ),
        (
            np.array([[10, 20, 30, 40]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.00001], dtype=np.float32),  # Test very low confidence
            "pascal_voc",
            np.array([[10, 20, 30, 40]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            np.array([0.00001], dtype=np.float32),  # Should preserve exact value
            False,
        ),
        # Test multiple label fields with different numpy dtypes
        (
            np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            np.array([1], dtype=np.int64),  # Test int64 class labels
            np.array([0.95], dtype=np.float16),  # Test float16 scores
            "yolo",
            np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            np.array([1], dtype=np.int64),  # Should preserve int64
            np.array([0.95], dtype=np.float16),  # Should preserve float16
            False,
        ),
        # Test case specifically for class ID clipping bug
        (
            np.array([[0.3, 0.8, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=np.float32),
            np.array([2, 3], dtype=np.int32),  # Class IDs > 1
            np.array([0.9, 0.8], dtype=np.float32),
            "yolo",
            np.array([[0.3, 0.8, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]], dtype=np.float32),
            np.array([2, 3], dtype=np.int32),  # Should remain [2, 3], not [1, 1]
            np.array([0.9, 0.8], dtype=np.float32),
            True,  # This is key - we want clip=True to test the bug
        ),
        # Additional test with higher class IDs
        (
            np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            np.array([10], dtype=np.int32),  # Higher class ID
            np.array([0.95], dtype=np.float32),
            "yolo",
            np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            np.array([10], dtype=np.int32),  # Should remain 10, not 1
            np.array([0.95], dtype=np.float32),
            True,  # With clip=True
        ),
    ]
)
def test_compose_bbox_transform(
    bboxes, classes, scores, format, expected_bboxes, expected_classes, expected_scores, clip
):
    """Test bbox transformations with various formats and configurations."""
    transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format=format,
            label_fields=["classes", "scores"],
            clip=clip,
        ),
        strict=True,
    )

    transformed = transform(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        bboxes=bboxes,
        classes=classes,
        scores=scores,
    )

    if len(bboxes) > 0:
        np.testing.assert_array_almost_equal(np.array(transformed["bboxes"]), expected_bboxes, decimal=5)
        np.testing.assert_array_equal(np.array(transformed["classes"]), expected_classes)
        np.testing.assert_array_almost_equal(np.array(transformed["scores"]), expected_scores, decimal=5)

        if format == "yolo" and clip:
            assert np.all(np.array(transformed["bboxes"]) >= 0)
            assert np.all(np.array(transformed["bboxes"]) <= 1)
    else:
        assert len(transformed["bboxes"]) == 0
        assert len(transformed["classes"]) == 0
        assert len(transformed["scores"]) == 0


MAX_ACCEPT_RATIO_TEST_CASES = [
    # Normal aspect ratios (should pass)
    (
        np.array([[0.1, 0.1, 0.2, 0.2]], dtype=np.float32),  # 1:1 ratio
        {"height": 100, "width": 100},
        2.0,
        np.array([[0.1, 0.1, 0.2, 0.2]], dtype=np.float32),
    ),
    # Too wide box (should be filtered)
    (
        np.array([[0.1, 0.1, 0.9, 0.2]], dtype=np.float32),  # 8:1 ratio
        {"height": 100, "width": 100},
        2.0,
        np.zeros((0, 4), dtype=np.float32),
    ),
    # Too tall box (should be filtered)
    (
        np.array([[0.1, 0.1, 0.2, 0.9]], dtype=np.float32),  # 1:8 ratio
        {"height": 100, "width": 100},
        2.0,
        np.zeros((0, 4), dtype=np.float32),
    ),
    # Multiple boxes with mixed ratios
    (
        np.array([
            [0.1, 0.1, 0.2, 0.2],  # 1:1 ratio (keep)
            [0.3, 0.3, 0.9, 0.4],  # 6:1 ratio (filter)
            [0.5, 0.5, 0.6, 0.6],  # 1:1 ratio (keep)
        ], dtype=np.float32),
        {"height": 100, "width": 100},
        2.0,
        np.array([
            [0.1, 0.1, 0.2, 0.2],
            [0.5, 0.5, 0.6, 0.6],
        ], dtype=np.float32),
    ),
    # None max_ratio (should not filter)
    (
        np.array([[0.1, 0.1, 0.9, 0.2]], dtype=np.float32),
        {"height": 100, "width": 100},
        None,
        np.array([[0.1, 0.1, 0.9, 0.2]], dtype=np.float32),
    ),
]

@pytest.mark.parametrize(
    ["bboxes", "shape", "max_accept_ratio", "expected"],
    MAX_ACCEPT_RATIO_TEST_CASES
)
def test_filter_bboxes_aspect_ratio(bboxes, shape, max_accept_ratio, expected):
    filtered = filter_bboxes(bboxes, shape, max_accept_ratio=max_accept_ratio)
    np.testing.assert_array_almost_equal(filtered, expected)

@pytest.mark.parametrize(
    ["bboxes", "shape", "max_accept_ratio", "expected"],
    MAX_ACCEPT_RATIO_TEST_CASES
)
def test_bbox_processor_max_accept_ratio(bboxes, shape, max_accept_ratio, expected):
    data = {
        "image": np.zeros((shape["height"], shape["width"], 3), dtype=np.uint8),
        "bboxes": bboxes,
    }

    params = BboxParams(
        format="albumentations",
        max_accept_ratio=max_accept_ratio,
    )
    processor = BboxProcessor(params)

    # Preprocess
    processor.preprocess(data)

    # Postprocess
    processed_data = processor.postprocess(data)

    np.testing.assert_array_almost_equal(processed_data["bboxes"], expected, decimal=5)

@pytest.mark.parametrize(
    ["bboxes", "shape", "max_accept_ratio", "expected"],
    MAX_ACCEPT_RATIO_TEST_CASES
)
def test_compose_with_max_accept_ratio(bboxes, shape, max_accept_ratio, expected):

    transform = A.Compose(
        [A.NoOp(p=1.0)],
        bbox_params=BboxParams(
            format="albumentations",
            max_accept_ratio=max_accept_ratio,
            label_fields=[],
        ),
        strict=True,
    )

    data = {
        "image": np.zeros((shape["height"], shape["width"], 3), dtype=np.uint8),
        "bboxes": bboxes,
    }
    result = transform(**data)

    np.testing.assert_array_almost_equal(result["bboxes"], expected, decimal=5)

@pytest.mark.parametrize(
    "bbox_format, bboxes, shape, max_accept_ratio, expected",
    [
        # COCO = [x_min, y_min, width, height]
        # (ratio 5:1 should be filtered)
        (
            "coco",
            np.array([[10, 10, 50, 10]], dtype=np.float32),
            {"height": 100, "width": 100},
            2.0,
            np.zeros((0, 4), dtype=np.float32),
        ),
        # Pascal VOC = [x_min, y_min, x_max, y_max]
        # (ratio 5:1 should be filtered)
        (
            "pascal_voc",
            np.array([[10, 10, 60, 20]], dtype=np.float32),
            {"height": 100, "width": 100},
            2.0,
            np.zeros((0, 4), dtype=np.float32),
        ),
        # YOLO = [x_center, y_center, width, height], normalized [0, 1].
        # (ratio 5:1 should be filtered)
        (
            "yolo",
            np.array([[0.5, 0.5, 0.5, 0.1]], dtype=np.float32),
            {"height": 100, "width": 100},
            2.0,
            np.zeros((0, 4), dtype=np.float32),
        ),
        # Albumentations = [x_min, y_min, x_max, y_max], all normalized in [0, 1].
        # (ratio 5:1 should be filtered)
        (
            "albumentations",
            np.array([[0.1, 0.1, 0.6, 0.2]], dtype=np.float32),
            {"height": 100, "width": 100},
            2.0,
            np.zeros((0, 4), dtype=np.float32),
        ),
    ],
)
def test_compose_max_accept_ratio_all_formats(bbox_format, bboxes, shape, max_accept_ratio, expected):
    transform = A.Compose(
        [A.NoOp(p=1.0)],
        bbox_params=BboxParams(
            format=bbox_format,
            max_accept_ratio=max_accept_ratio,
            label_fields=[],
        ),
        strict=True,
    )

    data = {
        "image": np.zeros((shape["height"], shape["width"], 3), dtype=np.uint8),
        "bboxes": bboxes,
    }

    result = transform(**data)
    np.testing.assert_array_almost_equal(result["bboxes"], expected, decimal=5)


def test_resize_boxes_to_visible_area_removes_fully_covered_boxes():
    """Test that resize_boxes_to_visible_area removes boxes with zero visible area."""
    # Create bounding boxes with additional columns for encoded labels
    # Format: [x_min, y_min, x_max, y_max, encoded_label_1, encoded_label_2]
    boxes = np.array([
        [10, 10, 30, 30, 1, 2],  # Box 1 with encoded labels 1, 2 - will be fully covered
        [40, 40, 60, 60, 3, 4],  # Box 2 with encoded labels 3, 4 - will remain visible
    ], dtype=np.float32)

    # Create a hole mask that completely covers the first box
    hole_mask = np.zeros((100, 100), dtype=np.uint8)
    hole_mask[10:30, 10:30] = 1  # Fully cover first box

    # Update boxes using the function
    updated_boxes = resize_boxes_to_visible_area(boxes, hole_mask)

    # Assertions
    assert len(updated_boxes) == 1, "Should remove fully covered boxes"
    assert updated_boxes.shape == (1, 6), "Should preserve the shape with correct number of columns"

    # The remaining box should be the second one
    np.testing.assert_array_equal(updated_boxes[0, :4], boxes[1, :4], "Second box coordinates should be unchanged")
    assert updated_boxes[0, 4] == 3, "Second box should preserve first encoded label"
    assert updated_boxes[0, 5] == 4, "Second box should preserve second encoded label"

def test_resize_boxes_to_visible_area_with_partially_covered_boxes():
    """Test that resize_boxes_to_visible_area correctly handles partially covered boxes."""
    # Create bounding boxes with additional columns for encoded labels
    boxes = np.array([
        [10, 10, 30, 30, 1, 2],  # Box that will be partially covered
    ], dtype=np.float32)

    # Create a hole mask that covers the left half of the box
    hole_mask = np.zeros((100, 100), dtype=np.uint8)
    hole_mask[10:30, 10:20] = 1  # Cover left half of the box

    # Update boxes using the function
    updated_boxes = resize_boxes_to_visible_area(boxes, hole_mask)

    # Assertions
    assert len(updated_boxes) == 1, "Should keep partially covered boxes"
    assert updated_boxes.shape == (1, 6), "Should preserve the shape with correct number of columns"
    assert updated_boxes[0, 0] > boxes[0, 0], "X min should increase (left part covered)"
    assert updated_boxes[0, 2] == boxes[0, 2], "X max should remain the same"
    assert updated_boxes[0, 4] == 1, "Should preserve first encoded label"
    assert updated_boxes[0, 5] == 2, "Should preserve second encoded label"

def test_resize_boxes_to_visible_area_with_all_boxes_covered():
    """Test that resize_boxes_to_visible_area returns empty array when all boxes are covered."""
    # Create bounding boxes with additional columns
    boxes = np.array([
        [10, 10, 30, 30, 1, 2],  # Box that will be fully covered
    ], dtype=np.float32)

    # Create a hole mask that completely covers the box
    hole_mask = np.zeros((100, 100), dtype=np.uint8)
    hole_mask[10:30, 10:30] = 1  # Fully cover the box

    # Update boxes using the function
    updated_boxes = resize_boxes_to_visible_area(boxes, hole_mask)

    # Assertions
    assert len(updated_boxes) == 0, "Should return empty array when all boxes are covered"
    assert updated_boxes.shape == (0, 6), "Empty array should have correct shape with all columns"

def test_resize_boxes_to_visible_area_with_empty_input():
    """Test that resize_boxes_to_visible_area handles empty input correctly."""
    # Empty boxes array with correct shape (0 boxes, 6 columns)
    boxes = np.zeros((0, 6), dtype=np.float32)
    hole_mask = np.zeros((100, 100), dtype=np.uint8)

    # Update boxes using the function
    updated_boxes = resize_boxes_to_visible_area(boxes, hole_mask)

    # Assertions
    assert len(updated_boxes) == 0, "Should return empty array"
    assert updated_boxes.shape == (0, 6), "Should preserve the shape with correct number of columns"
