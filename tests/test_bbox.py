import numpy as np
import pytest

from albumentations.augmentations.transforms import CenterCrop
from albumentations.augmentations.bbox import normalize_bbox, denormalize_bbox, normalize_bboxes, denormalize_bboxes, \
    calculate_bbox_area, filter_bboxes_by_visibility, convert_bbox_to_albumentations,\
    convert_bbox_from_albumentations, convert_bboxes_to_albumentations, convert_bboxes_from_albumentations


@pytest.mark.parametrize(['bbox', 'expected'], [
    [[15, 25, 100, 200], [0.0375, 0.125, 0.25, 1.0]],
    [[15, 25, 100, 200, 99], [0.0375, 0.125, 0.25, 1.0, 99]],
])
def test_normalize_bbox(bbox, expected):
    normalized_bbox = normalize_bbox(bbox, 200, 400)
    assert normalized_bbox == expected


@pytest.mark.parametrize(['bbox', 'expected'], [
    [[0.0375, 0.125, 0.25, 1.0], [15, 25, 100, 200]],
    [[0.0375, 0.125, 0.25, 1.0, 99], [15, 25, 100, 200, 99]],
])
def test_denormalize_bbox(bbox, expected):
    denormalized_bbox = denormalize_bbox(bbox, 200, 400)
    assert denormalized_bbox == expected


@pytest.mark.parametrize('bbox', [[15, 25, 100, 200], [15, 25, 100, 200, 99]])
def test_normalize_denormalize_bbox(bbox):
    normalized_bbox = normalize_bbox(bbox, 200, 400)
    denormalized_bbox = denormalize_bbox(normalized_bbox, 200, 400)
    assert denormalized_bbox == bbox


@pytest.mark.parametrize('bbox', [[0.0375, 0.125, 0.25, 1.0], [0.0375, 0.125, 0.25, 1.0, 99]])
def test_denormalize_normalize_bbox(bbox):
    denormalized_bbox = denormalize_bbox(bbox, 200, 400)
    normalized_bbox = normalize_bbox(denormalized_bbox, 200, 400)
    assert normalized_bbox == bbox


def test_normalize_bboxes():
    bboxes = [[15, 25, 100, 200], [15, 25, 100, 200, 99]]
    normalized_bboxes_1 = normalize_bboxes(bboxes, 200, 400)
    normalized_bboxes_2 = [normalize_bbox(bboxes[0], 200, 400), normalize_bbox(bboxes[1], 200, 400)]
    assert normalized_bboxes_1 == normalized_bboxes_2


def test_denormalize_bboxes():
    bboxes = [[0.0375, 0.125, 0.25, 1.0], [0.0375, 0.125, 0.25, 1.0, 99]]
    denormalized_bboxes_1 = denormalize_bboxes(bboxes, 200, 400)
    denormalized_bboxes_2 = [denormalize_bbox(bboxes[0], 200, 400), denormalize_bbox(bboxes[1], 200, 400)]
    assert denormalized_bboxes_1 == denormalized_bboxes_2


@pytest.mark.parametrize(['bbox', 'rows', 'cols', 'expected'], [
    [[0, 0, 1, 1], 50, 100, 5000],
    [[0.2, 0.2, 1, 1, 99], 50, 50, 1600],
])
def test_calculate_bbox_area(bbox, rows, cols, expected):
    area = calculate_bbox_area(bbox, rows, cols)
    assert area == expected


def test_filter_bboxes_by_visibility():
    image = np.ones((100, 100, 3))
    aug = CenterCrop(64, 64, p=1)
    bboxes = [[0.1, 0.1, 0.9, 0.8, 99], [0.7, 0.8, 0.9, 0.9]]
    augmented = aug(image=image, bboxes=bboxes)

    visible_bboxes_threshold_0_5 = filter_bboxes_by_visibility(
        image,
        bboxes,
        augmented['image'],
        augmented['bboxes'],
        threshold=0.5,
    )
    assert visible_bboxes_threshold_0_5 == [augmented['bboxes'][0]]

    visible_bboxes_threshold_0_1 = filter_bboxes_by_visibility(
        image,
        bboxes,
        augmented['image'],
        augmented['bboxes'],
        threshold=0.1,
    )
    assert visible_bboxes_threshold_0_1 == augmented['bboxes']


@pytest.mark.parametrize(['bbox', 'source_format', 'expected'], [
    [[20, 30, 40, 50], 'coco', [0.2, 0.3, 0.6, 0.8]],
    [[20, 30, 40, 50, 99], 'coco', [0.2, 0.3, 0.6, 0.8, 99]],
    [[20, 30, 60, 80], 'pascal_voc', [0.2, 0.3, 0.6, 0.8]],
    [[20, 30, 60, 80, 99], 'pascal_voc', [0.2, 0.3, 0.6, 0.8, 99]],
])
def test_convert_bbox_to_albumentations(bbox, source_format, expected):
    image = np.ones((100, 100, 3))
    converted_bbox = convert_bbox_to_albumentations(image.shape, bbox, source_format=source_format)
    assert converted_bbox == expected


@pytest.mark.parametrize(['bbox', 'target_format', 'expected'], [
    [[0.2, 0.3, 0.6, 0.8], 'coco', [20, 30, 40, 50]],
    [[0.2, 0.3, 0.6, 0.8, 99], 'coco', [20, 30, 40, 50, 99]],
    [[0.2, 0.3, 0.6, 0.8], 'pascal_voc', [20, 30, 60, 80]],
    [[0.2, 0.3, 0.6, 0.8, 99], 'pascal_voc', [20, 30, 60, 80, 99]],
])
def test_convert_bbox_from_albumentations(bbox, target_format, expected):
    image = np.ones((100, 100, 3))
    converted_bbox = convert_bbox_from_albumentations(image.shape, bbox, target_format=target_format)
    assert converted_bbox == expected


@pytest.mark.parametrize(['bbox', 'bbox_format'], [
    [[20, 30, 40, 50], 'coco'],
    [[20, 30, 40, 50, 99], 'coco'],
    [[20, 30, 60, 80], 'pascal_voc'],
    [[20, 30, 60, 80, 99], 'pascal_voc'],
])
def test_convert_bbox_to_albumentations_and_back(bbox, bbox_format):
    image = np.ones((100, 100, 3))
    converted_bbox = convert_bbox_to_albumentations(image.shape, bbox, source_format=bbox_format)
    converted_back_bbox = convert_bbox_from_albumentations(image.shape, converted_bbox, target_format=bbox_format)
    assert converted_back_bbox == bbox


def test_convert_bboxes_to_albumentations():
    bboxes = [[20, 30, 40, 50], [30, 40, 50, 60, 99]]
    image = np.ones((100, 100, 3))
    converted_bboxes = convert_bboxes_to_albumentations(image.shape, bboxes, source_format='coco')
    converted_bbox_1 = convert_bbox_to_albumentations(image.shape, bboxes[0], source_format='coco')
    converted_bbox_2 = convert_bbox_to_albumentations(image.shape, bboxes[1], source_format='coco')
    assert converted_bboxes == [converted_bbox_1, converted_bbox_2]


def test_convert_bboxes_from_albumentations():
    bboxes = [[0.2, 0.3, 0.6, 0.8], [0.3, 0.4, 0.7, 0.9, 99]]
    image = np.ones((100, 100, 3))
    converted_bboxes = convert_bboxes_from_albumentations(image.shape, bboxes, target_format='coco')
    converted_bbox_1 = convert_bbox_from_albumentations(image.shape, bboxes[0], target_format='coco')
    converted_bbox_2 = convert_bbox_from_albumentations(image.shape, bboxes[1], target_format='coco')
    assert converted_bboxes == [converted_bbox_1, converted_bbox_2]
