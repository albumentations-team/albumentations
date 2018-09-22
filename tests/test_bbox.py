import numpy as np
import pytest

from albumentations.augmentations.bbox_utils import normalize_bbox, denormalize_bbox, normalize_bboxes, \
    denormalize_bboxes, calculate_bbox_area, filter_bboxes_by_visibility, convert_bbox_to_albumentations,\
    convert_bbox_from_albumentations, convert_bboxes_to_albumentations, convert_bboxes_from_albumentations
from albumentations.core.composition import Compose
from albumentations.core.transforms_interface import NoOp
from albumentations.augmentations.transforms import RandomSizedCrop


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


@pytest.mark.parametrize(['bbox', 'source_format', 'expected'], [
    [[20, 30, 40, 50], 'coco', [0.2, 0.3, 0.6, 0.8]],
    [[20, 30, 40, 50, 99], 'coco', [0.2, 0.3, 0.6, 0.8, 99]],
    [[20, 30, 60, 80], 'pascal_voc', [0.2, 0.3, 0.6, 0.8]],
    [[20, 30, 60, 80, 99], 'pascal_voc', [0.2, 0.3, 0.6, 0.8, 99]],
])
def test_convert_bbox_to_albumentations(bbox, source_format, expected):
    image = np.ones((100, 100, 3))

    converted_bbox = convert_bbox_to_albumentations(bbox, rows=image.shape[0], cols=image.shape[1],
                                                    source_format=source_format)
    assert converted_bbox == expected


@pytest.mark.parametrize(['bbox', 'target_format', 'expected'], [
    [[0.2, 0.3, 0.6, 0.8], 'coco', [20, 30, 40, 50]],
    [[0.2, 0.3, 0.6, 0.8, 99], 'coco', [20, 30, 40, 50, 99]],
    [[0.2, 0.3, 0.6, 0.8], 'pascal_voc', [20, 30, 60, 80]],
    [[0.2, 0.3, 0.6, 0.8, 99], 'pascal_voc', [20, 30, 60, 80, 99]],
])
def test_convert_bbox_from_albumentations(bbox, target_format, expected):
    image = np.ones((100, 100, 3))
    converted_bbox = convert_bbox_from_albumentations(bbox, rows=image.shape[0], cols=image.shape[1],
                                                      target_format=target_format)
    assert converted_bbox == expected


@pytest.mark.parametrize(['bbox', 'bbox_format'], [
    [[20, 30, 40, 50], 'coco'],
    [[20, 30, 40, 50, 99], 'coco'],
    [[20, 30, 60, 80], 'pascal_voc'],
    [[20, 30, 60, 80, 99], 'pascal_voc'],
])
def test_convert_bbox_to_albumentations_and_back(bbox, bbox_format):
    image = np.ones((100, 100, 3))
    converted_bbox = convert_bbox_to_albumentations(bbox, rows=image.shape[0], cols=image.shape[1],
                                                    source_format=bbox_format)
    converted_back_bbox = convert_bbox_from_albumentations(converted_bbox, rows=image.shape[0], cols=image.shape[1],
                                                           target_format=bbox_format)
    assert converted_back_bbox == bbox


def test_convert_bboxes_to_albumentations():
    bboxes = [[20, 30, 40, 50], [30, 40, 50, 60, 99]]
    image = np.ones((100, 100, 3))
    converted_bboxes = convert_bboxes_to_albumentations(bboxes, rows=image.shape[0], cols=image.shape[1],
                                                        source_format='coco')
    converted_bbox_1 = convert_bbox_to_albumentations(bboxes[0], rows=image.shape[0], cols=image.shape[1],
                                                      source_format='coco')
    converted_bbox_2 = convert_bbox_to_albumentations(bboxes[1], rows=image.shape[0], cols=image.shape[1],
                                                      source_format='coco')
    assert converted_bboxes == [converted_bbox_1, converted_bbox_2]


def test_convert_bboxes_from_albumentations():
    bboxes = [[0.2, 0.3, 0.6, 0.8], [0.3, 0.4, 0.7, 0.9, 99]]
    image = np.ones((100, 100, 3))
    converted_bboxes = convert_bboxes_to_albumentations(bboxes, rows=image.shape[0], cols=image.shape[1],
                                                        source_format='coco')
    converted_bbox_1 = convert_bbox_to_albumentations(bboxes[0], rows=image.shape[0], cols=image.shape[1],
                                                      source_format='coco')
    converted_bbox_2 = convert_bbox_to_albumentations(bboxes[1], rows=image.shape[0], cols=image.shape[1],
                                                      source_format='coco')
    assert converted_bboxes == [converted_bbox_1, converted_bbox_2]


@pytest.mark.parametrize(['bboxes', 'bbox_format', 'labels'], [
    [[[20, 30, 40, 50]], 'coco', [1]],
    [[[20, 30, 40, 50, 99], [10, 40, 30, 20, 9]], 'coco', None],
    [[[20, 30, 60, 80]], 'pascal_voc', [2]],
    [[[20, 30, 60, 80, 99]], 'pascal_voc', None],
])
def test_compose_with_bbox_noop(bboxes, bbox_format, labels):
    image = np.ones((100, 100, 3))
    if labels is not None:
        aug = Compose([NoOp(p=1.)], bbox_params={'format': bbox_format, 'label_fields': ['labels']})
        transformed = aug(image=image, bboxes=bboxes, labels=labels)
    else:
        aug = Compose([NoOp(p=1.)], bbox_params={'format': bbox_format})
        transformed = aug(image=image, bboxes=bboxes)
    assert np.array_equal(transformed['image'], image)
    assert transformed['bboxes'] == bboxes


@pytest.mark.parametrize(['bboxes', 'bbox_format'], [
    [[[20, 30, 40, 50]], 'coco'],
])
def test_compose_with_bbox_noop_error_label_fields(bboxes, bbox_format):
    image = np.ones((100, 100, 3))
    aug = Compose([NoOp(p=1.)], bbox_params={'format': bbox_format})
    with pytest.raises(Exception):
        aug(image=image, bboxes=bboxes)


@pytest.mark.parametrize(['bboxes'], [
    [[[20, 30, 40, 50]]],
])
def test_compose_with_bbox_noop_error_format(bboxes):
    image = np.ones((100, 100, 3))
    aug = Compose([NoOp(p=1.)])
    with pytest.raises(Exception):
        aug(image=image, bboxes=bboxes)


@pytest.mark.parametrize(['bboxes', 'bbox_format', 'labels'], [
    [[[20, 30, 60, 80]], 'pascal_voc', {'label': [1]}],
    [[[20, 30, 60, 80]], 'pascal_voc', {'id': [3]}],
    [[[20, 30, 60, 80], [30, 40, 40, 50]], 'pascal_voc', {'id': [3, 1]}],
])
def test_compose_with_bbox_noop_label_outside(bboxes, bbox_format, labels):
    image = np.ones((100, 100, 3))
    aug = Compose([NoOp(p=1.)], bbox_params={'format': bbox_format, 'label_fields': list(labels.keys())})
    transformed = aug(image=image, bboxes=bboxes, **labels)
    assert np.array_equal(transformed['image'], image)
    assert transformed['bboxes'] == bboxes
    for k, v in labels.items():
        assert transformed[k] == v


def test_random_sized_crop_size():
    image = np.ones((100, 100, 3))
    bboxes = [[0.2, 0.3, 0.6, 0.8], [0.3, 0.4, 0.7, 0.9, 99]]
    aug = RandomSizedCrop((70, 90), 50, 50, p=1.)
    transformed = aug(image=image, bboxes=bboxes)
    assert transformed['image'].shape == (50, 50, 3)
    assert len(bboxes) == len(transformed['bboxes'])
