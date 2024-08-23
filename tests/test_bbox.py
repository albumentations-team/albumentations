from __future__ import annotations

from typing import Callable
import cv2

import numpy as np
import pytest

from albumentations import RandomCrop, RandomResizedCrop, RandomSizedCrop, Rotate
from albumentations.core.bbox_utils import (
    BboxProcessor,
    calculate_bbox_areas,
    check_bboxes,
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
    denormalize_bboxes,
    filter_bboxes,
    normalize_bboxes,
    union_of_bboxes,
)

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.composition import BboxParams, Compose, ReplayCompose
from albumentations.core.transforms_interface import NoOp, BasicTransform
import albumentations as A
from .utils import set_seed



@pytest.mark.parametrize("bboxes, image_shape, expected", [
    (
        np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        (100, 200),
        np.array([[0.05, 0.2, 0.15, 0.4], [0.25, 0.6, 0.35, 0.8]])
    ),
    (
        np.array([[0, 0, 200, 100]]),
        (100, 200),
        np.array([[0, 0, 1, 1]])
    ),
    (
        np.array([[50, 50, 150, 150, 1], [25, 25, 75, 75, 2]]),
        (200, 200),
        np.array([[0.25, 0.25, 0.75, 0.75, 1], [0.125, 0.125, 0.375, 0.375, 2]])
    ),
    (
        np.array([]),
        (100, 100),
        np.array([])
    ),
])
def test_normalize_bboxes(bboxes, image_shape, expected):
    result = normalize_bboxes(bboxes, image_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_normalize_bboxes_preserves_input():
    bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    image_shape = (100, 200)
    original_bboxes = bboxes.copy()

    normalize_bboxes(bboxes, image_shape)

    np.testing.assert_array_equal(bboxes, original_bboxes)

def test_normalize_bboxes_output_type():
    bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    image_shape = (100, 200)

    result = normalize_bboxes(bboxes, image_shape)

    assert isinstance(result, np.ndarray)
    assert result.dtype == float


@pytest.mark.parametrize("bboxes, image_shape, expected", [
    (
        np.array([[0.05, 0.2, 0.15, 0.4], [0.25, 0.6, 0.35, 0.8]]),
        (100, 200),
        np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    ),
    (
        np.array([[0, 0, 1, 1]]),
        (100, 200),
        np.array([[0, 0, 200, 100]])
    ),
    (
        np.array([[0.25, 0.25, 0.75, 0.75, 1], [0.125, 0.125, 0.375, 0.375, 2]]),
        (200, 200),
        np.array([[50, 50, 150, 150, 1], [25, 25, 75, 75, 2]])
    ),
    (
        np.array([]),
        (100, 100),
        np.array([])
    ),
])
def test_denormalize_bboxes(bboxes, image_shape, expected):
    result = denormalize_bboxes(bboxes, image_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_denormalize_bboxes_preserves_input():
    bboxes = np.array([[0.05, 0.2, 0.15, 0.4], [0.25, 0.6, 0.35, 0.8]])
    image_shape = (100, 200)
    original_bboxes = bboxes.copy()

    denormalize_bboxes(bboxes, image_shape)

    np.testing.assert_array_equal(bboxes, original_bboxes)

def test_denormalize_bboxes_output_type():
    bboxes = np.array([[0.05, 0.2, 0.15, 0.4], [0.25, 0.6, 0.35, 0.8]])
    image_shape = (100, 200)

    result = denormalize_bboxes(bboxes, image_shape)

    assert isinstance(result, np.ndarray)
    assert result.dtype == float


@pytest.mark.parametrize("bboxes, image_shape", [
    (np.array([[10, 20, 30, 40], [50, 60, 70, 80]]), (100, 200)),
    (np.array([[0, 0, 200, 100]]), (100, 200)),
    (np.array([[50, 50, 150, 150, 1], [25, 25, 75, 75, 2]]), (200, 200)),
    (np.array([]), (100, 100)),
])
def test_normalize_denormalize_roundtrip(bboxes, image_shape):
    normalized = normalize_bboxes(bboxes, image_shape)
    denormalized = denormalize_bboxes(normalized, image_shape)
    np.testing.assert_allclose(denormalized, bboxes, rtol=1e-5)

@pytest.mark.parametrize("bboxes, image_shape", [
    (np.array([[0.05, 0.2, 0.15, 0.4], [0.25, 0.6, 0.35, 0.8]]), (100, 200)),
    (np.array([[0, 0, 1, 1]]), (100, 200)),
    (np.array([[0.25, 0.25, 0.75, 0.75, 1], [0.125, 0.125, 0.375, 0.375, 2]]), (200, 200)),
    (np.array([]), (100, 100)),
])
def test_denormalize_normalize_roundtrip(bboxes, image_shape):
    denormalized = denormalize_bboxes(bboxes, image_shape)
    normalized = normalize_bboxes(denormalized, image_shape)
    np.testing.assert_allclose(normalized, bboxes, rtol=1e-5)


@pytest.mark.parametrize("bboxes, image_shape, expected", [
    (
        np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]]),
        (100, 100),
        np.array([1600, 3600])
    ),
    (
        np.array([[0, 0, 1, 1]]),
        (200, 300),
        np.array([60000])
    ),
    (
        np.array([[0.25, 0.25, 0.75, 0.75, 1], [0.1, 0.1, 0.9, 0.9, 2]]),
        (100, 100),
        np.array([2500, 6400])
    ),
    (
        np.array([]),
        (100, 100),
        np.array([])
    ),
    (
        np.array([[0.1, 0.1, 0.3, 0.3], [0.4, 0.4, 0.6, 0.6]]),
        (50, 200),
        np.array([400, 400])
    ),
])
def test_calculate_bbox_areas(bboxes, image_shape, expected):
    result = calculate_bbox_areas(bboxes, image_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_calculate_bbox_areas_preserves_input():
    bboxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    image_shape = (100, 100)
    original_bboxes = bboxes.copy()

    calculate_bbox_areas(bboxes, image_shape)

    np.testing.assert_array_equal(bboxes, original_bboxes)

def test_calculate_bbox_areas_output_type():
    bboxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    image_shape = (100, 100)

    result = calculate_bbox_areas(bboxes, image_shape)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_calculate_bbox_areas_zero_area():
    bboxes = np.array([[0.1, 0.1, 0.1, 0.2], [0.3, 0.3, 0.4, 0.3]])  # Zero width and zero height
    image_shape = (100, 100)
    result = calculate_bbox_areas(bboxes, image_shape)
    np.testing.assert_allclose(result, [0, 0], atol=1e-10)


@pytest.mark.parametrize("bboxes, source_format, image_shape, expected", [
    # COCO format
    (
        np.array([[10, 20, 30, 40], [50, 60, 20, 30]]),
        "coco",
        (100, 200),
        np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]])
    ),
    # Pascal VOC format
    (
        np.array([[10, 20, 40, 60], [50, 60, 70, 90]]),
        "pascal_voc",
        (100, 200),
        np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]])
    ),
    # YOLO format
    (
        np.array([[0.25, 0.5, 0.2, 0.4], [0.7, 0.8, 0.2, 0.3]]),
        "yolo",
        (100, 200),  # image shape doesn't matter for YOLO
        np.array([[0.15, 0.3, 0.35, 0.7], [0.6, 0.65, 0.8, 0.95]])
    ),
    # With additional columns
    (
        np.array([[10, 20, 30, 40, 1], [50, 60, 20, 30, 2]]),
        "coco",
        (100, 200),
        np.array([[0.05, 0.2, 0.2, 0.6, 1], [0.25, 0.6, 0.35, 0.9, 2]])
    ),
    # Empty array
    (
        np.array([]).reshape(0, 4),
        "coco",
        (100, 200),
        np.array([]).reshape(0, 4)
    ),
])
def test_convert_bboxes_to_albumentations(bboxes, source_format, image_shape, expected):
    result = convert_bboxes_to_albumentations(bboxes, source_format, image_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_convert_bboxes_to_albumentations_preserves_input():
    bboxes = np.array([[10, 20, 30, 40], [50, 60, 20, 30]])
    original_bboxes = bboxes.copy()
    convert_bboxes_to_albumentations(bboxes, "coco", (100, 200))
    np.testing.assert_array_equal(bboxes, original_bboxes)

def test_convert_bboxes_to_albumentations_output_type():
    bboxes = np.array([[10, 20, 30, 40], [50, 60, 20, 30]], dtype=np.float32)
    result = convert_bboxes_to_albumentations(bboxes, "coco", (100, 200))
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


@pytest.mark.parametrize("bboxes, target_format, image_shape, expected", [
    # Albumentations to COCO format
    (
        np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]]),
        "coco",
        (100, 200),
        np.array([[10, 20, 30, 40], [50, 60, 20, 30]])
    ),
    # Albumentations to Pascal VOC format
    (
        np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]]),
        "pascal_voc",
        (100, 200),
        np.array([[10, 20, 40, 60], [50, 60, 70, 90]])
    ),
    # Albumentations to YOLO format
    (
        np.array([[0.15, 0.3, 0.35, 0.7], [0.6, 0.65, 0.8, 0.95]]),
        "yolo",
        (100, 200),  # image shape doesn't matter for YOLO
        np.array([[0.25, 0.5, 0.2, 0.4], [0.7, 0.8, 0.2, 0.3]])
    ),
    # With additional columns
    (
        np.array([[0.05, 0.2, 0.2, 0.6, 1], [0.25, 0.6, 0.35, 0.9, 2]]),
        "coco",
        (100, 200),
        np.array([[10, 20, 30, 40, 1], [50, 60, 20, 30, 2]])
    ),
    # Empty array
    (
        np.array([]).reshape(0, 4),
        "coco",
        (100, 200),
        np.array([]).reshape(0, 4)
    ),
])
def test_convert_bboxes_from_albumentations(bboxes, target_format, image_shape, expected):
    result = convert_bboxes_from_albumentations(bboxes, target_format, image_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_convert_bboxes_from_albumentations_preserves_input():
    bboxes = np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]])
    original_bboxes = bboxes.copy()
    convert_bboxes_from_albumentations(bboxes, "coco", (100, 200))
    np.testing.assert_array_equal(bboxes, original_bboxes)

def test_convert_bboxes_from_albumentations_output_type():
    bboxes = np.array([[0.05, 0.2, 0.2, 0.6], [0.25, 0.6, 0.35, 0.9]])
    result = convert_bboxes_from_albumentations(bboxes, "coco", (100, 200))
    assert isinstance(result, np.ndarray)
    assert result.dtype == bboxes.dtype

@pytest.mark.parametrize("target_format", ["invalid_format", "COCO", "Pascal_VOC"])
def test_convert_bboxes_from_albumentations_invalid_format(target_format):
    bboxes = np.array([[0.05, 0.2, 0.2, 0.6]])
    with pytest.raises(ValueError, match="Unknown target_format"):
        convert_bboxes_from_albumentations(bboxes, target_format, (100, 200))

def test_convert_bboxes_from_albumentations_check_validity(mocker):
    bboxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    image_shape = (100, 200)
    mock_check_bboxes = mocker.patch("albumentations.core.bbox_utils.check_bboxes")

    convert_bboxes_from_albumentations(bboxes, "coco", image_shape, check_validity=True)

    mock_check_bboxes.assert_called_once()

@pytest.mark.parametrize("target_format", ["coco", "pascal_voc"])
def test_convert_bboxes_from_albumentations_calls_denormalize(target_format, mocker):
    bboxes = np.array([[0.05, 0.2, 0.2, 0.6]])
    image_shape = (100, 200)
    mock_denormalize_bboxes = mocker.patch("albumentations.core.bbox_utils.denormalize_bboxes", return_value=bboxes)

    convert_bboxes_from_albumentations(bboxes, target_format, image_shape)

    mock_denormalize_bboxes.assert_called_once()

def test_convert_bboxes_from_albumentations_yolo_does_not_call_denormalize(mocker):
    bboxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    image_shape = (100, 200)
    mock_denormalize_bboxes = mocker.patch("albumentations.core.bbox_utils.denormalize_bboxes")

    convert_bboxes_from_albumentations(bboxes, "yolo", image_shape)

    mock_denormalize_bboxes.assert_not_called()

@pytest.mark.parametrize("original_format, image_shape", [
    ("coco", (100, 200)),
    ("pascal_voc", (100, 200)),
    ("yolo", (100, 200)),
])
def test_round_trip_to_from_albumentations(original_format, image_shape):
    original_bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    if original_format == "yolo":
        original_bboxes = np.array([[0.25, 0.3, 0.2, 0.2], [0.6, 0.7, 0.2, 0.2]])

    # Convert to albumentations format
    albu_bboxes = convert_bboxes_to_albumentations(original_bboxes, original_format, image_shape)

    # Convert back to original format
    converted_bboxes = convert_bboxes_from_albumentations(albu_bboxes, original_format, image_shape)

    np.testing.assert_allclose(converted_bboxes, original_bboxes, rtol=1e-5)

@pytest.mark.parametrize("target_format, image_shape", [
    ("coco", (100, 200)),
    ("pascal_voc", (100, 200)),
    ("yolo", (100, 200)),
])
def test_round_trip_from_to_albumentations(target_format, image_shape):
    albu_bboxes = np.array([[0.05, 0.1, 0.15, 0.2], [0.25, 0.3, 0.35, 0.4]])

    # Convert from albumentations format
    converted_bboxes = convert_bboxes_from_albumentations(albu_bboxes, target_format, image_shape)

    # Convert back to albumentations format
    reconverted_bboxes = convert_bboxes_to_albumentations(converted_bboxes, target_format, image_shape)

    np.testing.assert_allclose(reconverted_bboxes, albu_bboxes, rtol=1e-5)

def test_check_bboxes_valid():
    valid_bboxes = np.array([
        [0.1, 0.2, 0.3, 0.4, 2],
        [0.5, 0.6, 0.7, 0.8, 3],
        [0, 0, 1, 1, 0],
        [0.1, 0.1, 0.2, 0.2, 1]  # with additional column
    ])
    check_bboxes(valid_bboxes)  # Should not raise any exception

@pytest.mark.parametrize("invalid_bbox, error_message", [
    (np.array([[1.1, 0.2, 0.3, 0.4]]), "Expected x_min for bbox"),
    (np.array([[0.1, 1.2, 0.3, 0.4]]), "Expected y_min for bbox"),
    (np.array([[0.1, 0.2, 1.3, 0.4]]), "Expected x_max for bbox"),
    (np.array([[0.1, 0.2, 0.3, 1.4]]), "Expected y_max for bbox"),
    (np.array([[-0.1, 0.2, 0.3, 0.4]]), "Expected x_min for bbox"),
])
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
    invalid_bboxes = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [1.1, 0.2, 0.3, 0.4],  # invalid x_min
        [0.5, 0.6, 0.7, 0.8],
        [0.3, 0.2, 0.1, 0.4],  # invalid x_max < x_min
    ])
    with pytest.raises(ValueError, match="Expected x_min for bbox"):
        check_bboxes(invalid_bboxes)

@pytest.mark.parametrize("bbox", [
    np.array([[0, 0.2, 0.3, 0.4]]),
    np.array([[0.1, 0, 0.3, 0.4]]),
    np.array([[0.1, 0.2, 1, 0.4]]),
    np.array([[0.1, 0.2, 0.3, 1]]),
])
def test_check_bboxes_exact_zero_and_one(bbox):
    check_bboxes(bbox)  # Should not raise any exception

def test_check_bboxes_additional_columns():
    valid_bboxes = np.array([
        [0.1, 0.2, 0.3, 0.4, 1, 2, 3],
        [0.5, 0.6, 0.7, 0.8, 4, 5, 6],
    ])
    check_bboxes(valid_bboxes)  # Should not raise any exception



@pytest.mark.parametrize("bboxes, image_shape, min_area, min_visibility, min_width, min_height, expected", [
    (
        np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]),
        (100, 100),
        0, 0, 0, 0,
        np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]])
    ),
    (
        np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]),
        (100, 100),
        200, 0, 0, 0,
        np.array([], dtype=np.float32).reshape(0, 4)
    ),
    (
        np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]),
        (100, 100),
        0, 0.5, 0, 0,
        np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]])
    ),
    (
        np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.5, 0.4], [0.5, 0.5, 0.7, 0.6]]),
        (100, 100),
        0, 0, 15, 0,
        np.array([[0.3, 0.3, 0.5, 0.4], [0.5, 0.5, 0.7, 0.6]])
    ),
    (
        np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.5], [0.5, 0.5, 0.6, 0.7]]),
        (100, 100),
        0, 0, 0, 15,
        np.array([[0.3, 0.3, 0.4, 0.5], [0.5, 0.5, 0.6, 0.7]])
    ),
    (
        np.array([[0.1, 0.1, 0.2, 0.2, 1], [0.3, 0.3, 0.4, 0.4, 2], [0.5, 0.5, 0.6, 0.7, 3]]),
        (100, 100),
        200, 0, 0, 0,
        np.array([[0.5, 0.5, 0.6, 0.7, 3]])
    ),
    (
        np.array([]),
        (100, 100),
        0, 0, 0, 0,
        np.array([])
    ),
])
def test_filter_bboxes(bboxes, image_shape, min_area, min_visibility, min_width, min_height, expected):
    result = filter_bboxes(bboxes, image_shape, min_area, min_visibility, min_width, min_height)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_filter_bboxes_preserves_input():
    bboxes = np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]])
    original_bboxes = bboxes.copy()
    image_shape = (100, 100)

    filter_bboxes(bboxes, image_shape)

    np.testing.assert_array_equal(bboxes, original_bboxes)

def test_filter_bboxes_output_type():
    bboxes = np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]])
    image_shape = (100, 100)

    result = filter_bboxes(bboxes, image_shape)

    assert isinstance(result, np.ndarray)
    assert result.dtype == bboxes.dtype

def test_filter_bboxes_clipping():
    bboxes = np.array([[-0.1, -0.1, 1.1, 1.1], [0.3, 0.3, 0.4, 0.4]])
    image_shape = (100, 100)

    result = filter_bboxes(bboxes, image_shape)

    expected = np.array([[0.0, 0.0, 1.0, 1.0], [0.3, 0.3, 0.4, 0.4]])
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize("bboxes, erosion_rate, expected", [
    (np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]), 0, np.array([0.1, 0.1, 0.6, 0.6])),
    (np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]), 0.5, np.array([0.225, 0.225, 0.475, 0.475])),
    (np.array([[0.1, 0.1, 0.5, 0.5]]), 0, np.array([0.1, 0.1, 0.5, 0.5])),
    (np.array([[0.1, 0.1, 0.5, 0.5]]), 1, None),
    (np.array([]), 0, None),
    (np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]), 1, None),
    (np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]]), 0.9, np.array([0.325, 0.325, 0.375, 0.375])),
])
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


@pytest.mark.parametrize("bbox_format, bboxes, labels", [
    ("coco", [[15, 12, 30, 40], [50, 50, 15, 40]], ["cat", "dog"]),
    ("pascal_voc", [[15, 12, 30, 40], [50, 50, 55, 60]], [1, 2]),
    ("albumentations", [[0.2, 0.3, 0.4, 0.5], [0.1, 0.1, 0.3, 0.3]], ["label1", "label2"]),
    ("yolo", [[0.15, 0.22, 0.3, 0.4], [0.5, 0.5, 0.15, 0.4]], [0, 3]),
])
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

@pytest.mark.parametrize("bbox_format, bboxes, labels1, labels2", [
    ("coco", [[15, 12, 30, 40], [50, 50, 15, 40]], ["cat", "dog"], [1, 2]),
    ("pascal_voc", [[15, 12, 30, 40], [50, 50, 55, 60]], [1, 2], ["label1", "label2"]),
    ("albumentations", [[0.2, 0.3, 0.4, 0.5], [0.1, 0.1, 0.3, 0.3]], ["label1", "label2"], [0, 1]),
    ("yolo",  [[0.15, 0.22, 0.3, 0.4], [0.5, 0.5, 0.15, 0.4]], [0, 1], ["type1", "type2"]),
])
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
    bboxes, bbox_format: str, labels: list[int] | None
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
    bboxes, bbox_format: str
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
    bboxes, bbox_format: str, labels: dict[str, list[int]]
) -> None:
    image = np.ones((100, 100, 3))
    aug = Compose(
        [NoOp(p=1.0)],
        bbox_params={"format": bbox_format, "label_fields": list(labels.keys())},
    )
    transformed = aug(image=image, bboxes=bboxes, **labels)
    assert np.array_equal(transformed["image"], image)
    assert np.allclose(transformed["bboxes"], bboxes)
    for k, v in labels.items():
        assert transformed[k] == v


def test_random_sized_crop_size() -> None:
    image = np.ones((100, 100, 3))
    bboxes = [(0.2, 0.3, 0.6, 0.8, 2), (0.3, 0.4, 0.7, 0.9, 99)]
    aug = A.Compose([RandomSizedCrop(min_max_height=(70, 90), size=(50, 50), p=1.0)], bbox_params={"format": "albumentations"})
    transformed = aug(image=image, bboxes=bboxes)
    assert transformed["image"].shape == (50, 50, 3)
    assert len(bboxes) == len(transformed["bboxes"])


def test_random_resized_crop_size() -> None:
    image = np.ones((100, 100, 3))
    bboxes = [(0.2, 0.3, 0.6, 0.8, 2), (0.3, 0.4, 0.7, 0.9, 99)]
    aug = A.Compose([RandomResizedCrop(size=(50, 50), p=1.0)], bbox_params={"format": "albumentations"})
    transformed = aug(image=image, bboxes=bboxes)
    assert transformed["image"].shape == (50, 50, 3)
    assert len(bboxes) == len(transformed["bboxes"])


def test_random_rotate() -> None:
    image = np.ones((192, 192, 3))
    bboxes = [(78, 42, 142, 80, 1), (32, 12, 42, 72, 2)]
    aug = A.Compose([Rotate(limit=15, p=1.0)], bbox_params={"format": "pascal_voc"})
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
    image_size: tuple[int, int], bbox, expected_bbox
) -> None:
    transform = Compose(
        [A.NoOp()],
        bbox_params={"format": "pascal_voc", "label_fields": ["labels"], "clip": True},
    )
    transformed = transform(
        image=np.zeros((*image_size, 3), dtype=np.uint8), bboxes=[bbox], labels=[1]
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
    )

    transformed = transform(image=image, bboxes=[bbox])

    np.testing.assert_almost_equal(transformed["bboxes"][0], expected_bbox)


@pytest.mark.parametrize(
    "get_transform",
    [
        lambda sign: A.Affine(translate_px=sign * 2, mode=cv2.BORDER_CONSTANT, cval=255),
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
    )

    res = aug(image=image, bboxes=bboxes)["bboxes"]
    np.testing.assert_almost_equal(res, expected)


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


@pytest.mark.parametrize("pad_top, pad_bottom, pad_left, pad_right, image_shape, expected", [
    # Symmetric padding
    (100, 100, 100, 100, (100, 100), {'grid_shape': (3, 3), 'original_position': (1, 1)}),  # Exact multiple
    (150, 150, 150, 150, (100, 100), {'grid_shape': (5, 5), 'original_position': (2, 2)}),  # Rounded up
    (50, 50, 50, 50, (100, 100), {'grid_shape': (3, 3), 'original_position': (1, 1)}),      # Less than image size

    # Asymmetric padding
    (100, 0, 100, 0, (100, 100), {'grid_shape': (2, 2), 'original_position': (1, 1)}),
    (0, 100, 0, 100, (100, 100), {'grid_shape': (2, 2), 'original_position': (0, 0)}),
    (100, 50, 75, 25, (100, 100), {'grid_shape': (3, 3), 'original_position': (1, 1)}),

    # Edge cases
    (0, 0, 0, 0, (100, 100), {'grid_shape': (1, 1), 'original_position': (0, 0)}),          # No padding
    (1, 1, 1, 1, (100, 100), {'grid_shape': (3, 3), 'original_position': (1, 1)}),          # Minimal padding

    # Different image dimensions
    (100, 100, 50, 50, (50, 100), {'grid_shape': (5, 3), 'original_position': (2, 1)}),

    # Large padding
    (500, 500, 500, 500, (100, 100), {'grid_shape': (11, 11), 'original_position': (5, 5)}),

    # Asymmetric image dimensions
    (100, 100, 100, 100, (200, 100), {'grid_shape': (3, 3), 'original_position': (1, 1)}),

    # Very small image dimensions
    (10, 10, 10, 10, (5, 5), {'grid_shape': (5, 5), 'original_position': (2, 2)}),

    # Very large image dimensions
    (1000, 1000, 1000, 1000, (10000, 10000), {'grid_shape': (3, 3), 'original_position': (1, 1)}),

    # Zero padding on some sides
    (100, 0, 0, 100, (100, 100), {'grid_shape': (2, 2), 'original_position': (1, 0)}),

    # Padding smaller than image on some sides, larger on others
    (50, 150, 25, 175, (100, 100), {'grid_shape': (4, 4), 'original_position': (1, 1)}),
])
def test_get_pad_grid_dimensions(pad_top, pad_bottom, pad_left, pad_right, image_shape, expected):
    result = fgeometric.get_pad_grid_dimensions(pad_top, pad_bottom, pad_left, pad_right, image_shape)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_get_pad_grid_dimensions_float_values():
    result = fgeometric.get_pad_grid_dimensions(10.5, 10.5, 10.5, 10.5, (100, 100))
    assert result == {'grid_shape': (3, 3), 'original_position': (1, 1)}, "Function should handle float inputs by implicit conversion to int"


@pytest.mark.parametrize("image_shape, bboxes, pad_params, expected_bboxes", [
    (
        (128, 96, 3),
        np.array([[5, 60, 40, 110, 0]]),  # input bboxes (scaled down)
        (128, 128, 96, 96),  # (pad_top, pad_bottom, pad_left, pad_right)
        np.array(
                [(56.0, 18.0, 91.0, 68.0, 0.0),
                (101.0, 18.0, 136.0, 68.0, 0.0),
                (248.0, 18.0, 283.0, 68.0, 0.0),
                (56.0, 188.0, 91.0, 238.0, 0.0),
                (101.0, 188.0, 136.0, 238.0, 0.0),
                (248.0, 188.0, 283.0, 238.0, 0.0),
                (56.0, 274.0, 91.0, 324.0, 0.0),
                (101.0, 274.0, 136.0, 324.0, 0.0),
                (248.0, 274.0, 283.0, 324.0, 0.0)]
    )
    ),
    # Add more test cases here
])
def test_pad_bboxes_with_reflection(image_shape, bboxes, pad_params, expected_bboxes):
    pad_top, pad_bottom, pad_left, pad_right = pad_params
    image_shape[:2]

    result = fgeometric.pad_bboxes(
        bboxes,
        pad_top, pad_bottom, pad_left, pad_right,
        border_mode=cv2.BORDER_REFLECT_101,
        image_shape=image_shape[:2]
    )

    np.testing.assert_array_almost_equal(result, expected_bboxes, decimal=1)


# Test case for non-reflect border mode
@pytest.mark.parametrize("image_shape, bboxes, pad_params, expected_bboxes", [
    (
        (100, 100, 3),  # image shape
        np.array([[20, 20, 80, 80, 0]]),  # input bboxes
        (10, 10, 10, 10),  # (pad_top, pad_bottom, pad_left, pad_right)
        np.array([[30, 30, 90, 90, 0]])
    ),
])
def test_pad_bboxes_constant_border(image_shape, bboxes, pad_params, expected_bboxes):
    pad_top, pad_bottom, pad_left, pad_right = pad_params

    result = fgeometric.pad_bboxes(
        bboxes,
        pad_top, pad_bottom, pad_left, pad_right,
        border_mode=cv2.BORDER_CONSTANT,
        image_shape=image_shape[:2]
    )

    np.testing.assert_array_almost_equal(result, expected_bboxes, decimal=1)


@pytest.mark.parametrize("bboxes, expected", [
    (
        np.array([[0.1, 0.2, 0.3, 0.4]]),
        np.array([[0.1, 0.6, 0.3, 0.8]])
    ),
    (
        np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
        np.array([[0.1, 0.6, 0.3, 0.8], [0.5, 0.2, 0.7, 0.4]])
    ),
    (
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8, 0.9]]),
        np.array([[0.1, 0.6, 0.3, 0.8, 0.5], [0.5, 0.2, 0.7, 0.4, 0.9]])
    ),
    (
        np.array([[0, 0, 1, 1]]),
        np.array([[0, 0, 1, 1]])
    ),
    (
        np.array([]),
        np.array([])
    ),
])
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


@pytest.mark.parametrize("bboxes, expected", [
    (
        np.array([[0.1, 0.2, 0.3, 0.4]]),
        np.array([[0.7, 0.2, 0.9, 0.4]])
    ),
    (
        np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
        np.array([[0.7, 0.2, 0.9, 0.4], [0.3, 0.6, 0.5, 0.8]])
    ),
    (
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8, 0.9]]),
        np.array([[0.7, 0.2, 0.9, 0.4, 0.5], [0.3, 0.6, 0.5, 0.8, 0.9]])
    ),
    (
        np.array([[0, 0, 1, 1]]),
        np.array([[0, 0, 1, 1]])
    ),
    (
        np.array([]),
        np.array([])
    ),
])
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
