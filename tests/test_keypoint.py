import numpy as np
import pytest
from albumentations import Flip, HorizontalFlip

from albumentations.augmentations.keypoints_utils import normalize_keypoint, denormalize_keypoint, \
    convert_keypoint_from_albumentations, convert_keypoints_from_albumentations, filter_keypoints, normalize_keypoints, denormalize_keypoints, \
    convert_keypoint_to_albumentations, convert_keypoints_to_albumentations
from albumentations.core.composition import Compose
from albumentations.core.transforms_interface import NoOp
from albumentations.augmentations.transforms import RandomSizedCrop


@pytest.mark.parametrize(['kp', 'expected'], [
    [[15, 25], [0.0375, 0.125]],
    [[15, 25, 99], [0.0375, 0.125, 99]],
])
def test_normalize_keypoint(kp, expected):
    normalized_kp = normalize_keypoint(kp, 200, 400)
    assert normalized_kp == expected


@pytest.mark.parametrize(['kp', 'expected'], [
    [[0.0375, 0.125], [15, 25]],
    [[0.0375, 0.125, 99], [15, 25, 99]],
])
def test_denormalize_keypoint(kp, expected):
    denormalized_kp = denormalize_keypoint(kp, 200, 400)
    assert denormalized_kp == expected


@pytest.mark.parametrize('kp', [[15, 25], [15, 25, 99]])
def test_normalize_denormalize_keypoint(kp):
    normalized_bbox = normalize_keypoint(kp, 200, 400)
    denormalized_bbox = denormalize_keypoint(normalized_bbox, 200, 400)
    assert denormalized_bbox == kp


@pytest.mark.parametrize('kp', [[0.0375, 0.125], [0.0375, 0.125, 99]])
def test_denormalize_normalize_keypoint(kp):
    denormalized_bbox = denormalize_keypoint(kp, 200, 400)
    normalized_bbox = normalize_keypoint(denormalized_bbox, 200, 400)
    assert normalized_bbox == kp


def test_normalize_keypoints():
    bboxes = [[15, 25, 100, 200], [15, 25, 100, 200, 99]]
    normalized_bboxes_1 = normalize_keypoints(bboxes, 200, 400)
    normalized_bboxes_2 = [normalize_keypoint(bboxes[0], 200, 400), normalize_keypoint(bboxes[1], 200, 400)]
    assert normalized_bboxes_1 == normalized_bboxes_2


def test_denormalize_bboxes():
    keypoints = [[0.0375, 0.125],
                 [0.0375, 0.125, 99]]
    denormalized_bboxes_1 = denormalize_keypoints(keypoints, 200, 400)
    denormalized_bboxes_2 = [denormalize_keypoint(keypoints[0], 200, 400), denormalize_keypoint(keypoints[1], 200, 400)]
    assert denormalized_bboxes_1 == denormalized_bboxes_2


@pytest.mark.parametrize(['kp', 'source_format', 'expected'], [
    [[20, 30], 'xy', [0.2, 0.3, 0, 0]],
    [[20, 30], 'yx', [0.3, 0.2, 0, 0]],
    [[20, 30, 60], 'xys', [0.2, 0.3, 0, 60]],
    [[20, 30, 60], 'xya', [0.2, 0.3, 60, 0]],
    [[20, 30, 60, 80], 'xyas', [0.2, 0.3, 60, 80]],
])
def test_convert_keypoint_to_albumentations(kp, source_format, expected):
    image = np.ones((100, 100, 3))

    converted_bbox = convert_keypoint_to_albumentations(kp, rows=image.shape[0], cols=image.shape[1],
                                                        source_format=source_format)
    assert converted_bbox == expected


@pytest.mark.parametrize(['kp', 'target_format', 'expected'], [
    [[0.2, 0.3, 0, 0], 'xy', [20, 30]],
    [[0.2, 0.3, 0, 0], 'yx', [30, 20]],
    [[0.2, 0.3, 0.6, 0], 'xya', [20, 30, 0.6]],
    [[0.2, 0.3, 0, 0.6], 'xys', [20, 30, 0.6]],
    [[0.2, 0.3, 60, 80], 'xyas', [20, 30, 60, 80]],
])
def test_convert_keypoint_from_albumentations(kp, target_format, expected):
    image = np.ones((100, 100, 3))
    converted_bbox = convert_keypoint_from_albumentations(kp, rows=image.shape[0], cols=image.shape[1],
                                                          target_format=target_format)
    assert converted_bbox == expected


@pytest.mark.parametrize(['kp', 'keypoint_format'], [
    [[20, 30, 40, 50], 'xy'],
    [[20, 30, 40, 50, 99], 'xyas'],
    [[20, 30, 60, 80], 'xysa'],
    [[20, 30, 60, 80, 99], 'yx'],
])
def test_convert_keypoint_to_albumentations_and_back(kp, keypoint_format):
    image = np.ones((100, 100, 3))
    converted_kp = convert_keypoint_to_albumentations(kp, rows=image.shape[0], cols=image.shape[1],
                                                      source_format=keypoint_format)
    converted_back_kp = convert_keypoint_from_albumentations(converted_kp, rows=image.shape[0], cols=image.shape[1],
                                                             target_format=keypoint_format)
    assert converted_back_kp == kp


def test_convert_keypoints_to_albumentations():
    keypoints = [[20, 30, 40, 50],
                 [30, 40, 50, 60, 99]]
    image = np.ones((100, 100, 3))
    converted_bboxes = convert_keypoints_to_albumentations(keypoints, rows=image.shape[0], cols=image.shape[1],
                                                           source_format='xyas')
    converted_bbox_1 = convert_keypoint_to_albumentations(keypoints[0], rows=image.shape[0], cols=image.shape[1],
                                                          source_format='xyas')
    converted_bbox_2 = convert_keypoint_to_albumentations(keypoints[1], rows=image.shape[0], cols=image.shape[1],
                                                          source_format='xyas')
    assert converted_bboxes == [converted_bbox_1, converted_bbox_2]


def test_convert_keypoints_from_albumentations():
    keypoints = [[0.2, 0.3, 0.6, 0.8], [0.3, 0.4, 0.7, 0.9, 99]]
    image = np.ones((100, 100, 3))
    converted_bboxes = convert_keypoints_from_albumentations(keypoints, rows=image.shape[0], cols=image.shape[1],
                                                             target_format='xyas')
    converted_bbox_1 = convert_keypoint_from_albumentations(keypoints[0], rows=image.shape[0], cols=image.shape[1],
                                                            target_format='xyas')
    converted_bbox_2 = convert_keypoint_from_albumentations(keypoints[1], rows=image.shape[0], cols=image.shape[1],
                                                            target_format='xyas')
    assert converted_bboxes == [converted_bbox_1, converted_bbox_2]


@pytest.mark.parametrize(['keypoints', 'keypoint_format', 'labels'], [
    [[[20, 30, 40, 50]], 'xyas', [1]],
    [[[20, 30, 40, 50, 99], [10, 40, 30, 20, 9]], 'xy', None],
    [[[20, 30, 60, 80]], 'yx', [2]],
    [[[20, 30, 60, 80, 99]], 'xys', None],
])
def test_compose_with_keypoint_noop(keypoints, keypoint_format, labels):
    image = np.ones((100, 100, 3))
    if labels is not None:
        aug = Compose([NoOp(p=1.)], keypoint_params={'format': keypoint_format, 'label_fields': ['labels']})
        transformed = aug(image=image, keypoints=keypoints, labels=labels)
    else:
        aug = Compose([NoOp(p=1.)], keypoint_params={'format': keypoint_format})
        transformed = aug(image=image, keypoints=keypoints)
    assert np.array_equal(transformed['image'], image)
    assert transformed['keypoints'] == keypoints


@pytest.mark.parametrize(['keypoints', 'keypoint_format'], [
    [[[20, 30, 40, 50]], 'xyas'],
])
def test_compose_with_bbox_noop_error_label_fields(keypoints, keypoint_format):
    image = np.ones((100, 100, 3))
    aug = Compose([NoOp(p=1.)], keypoint_params={'format': keypoint_format})
    with pytest.raises(Exception):
        aug(image=image, keypoints=keypoints)


@pytest.mark.parametrize(['keypoints', 'keypoint_format', 'labels'], [
    [[[20, 30, 60, 80]], 'xy', {'label': [1]}],
    [[], 'xy', {}],
    [[], 'xy', {'label': []}],
    [[[20, 30, 60, 80]], 'xy', {'id': [3]}],
    [[[20, 30, 60, 80], [30, 40, 40, 50]], 'xy', {'id': [3, 1]}],
])
def test_compose_with_keypoint_noop_label_outside(keypoints, keypoint_format, labels):
    image = np.ones((100, 100, 3))
    aug = Compose([NoOp(p=1.)], keypoint_params={'format': keypoint_format, 'label_fields': list(labels.keys())})
    transformed = aug(image=image, keypoints=keypoints, **labels)
    assert np.array_equal(transformed['image'], image)
    assert transformed['keypoints'] == keypoints
    for k, v in labels.items():
        assert transformed[k] == v


def test_random_sized_crop_size():
    image = np.ones((100, 100, 3))
    keypoints = [[0.2, 0.3, 0.6, 0.8], [0.3, 0.4, 0.7, 0.9, 99]]
    aug = RandomSizedCrop((70, 90), 50, 50, p=1.)
    transformed = aug(image=image, keypoints=keypoints)
    assert transformed['image'].shape == (50, 50, 3)
    assert len(keypoints) == len(transformed['keypoints'])


@pytest.mark.parametrize(['aug', 'keypoints', 'expected'], [
    [HorizontalFlip, [[20, 30, 0, 0]], [[80, 30, 180, 0]]],
    [HorizontalFlip, [[20, 30, 45, 0]], [[80, 30, 135, 0]]],
    [HorizontalFlip, [[20, 30, 90, 0]], [[80, 30, 90, 0]]],
])
def test_keypoint_transform(aug, keypoints, expected):
    transform = Compose([aug(p=1)], keypoint_params={'format': 'xyas', 'angle_in_degrees': True, 'label_fields': ['labels']})

    image = np.ones((100, 100, 3))
    transformed = transform(image=image, keypoints=keypoints, labels=np.ones(len(keypoints)))
    assert np.allclose(expected, transformed['keypoints'])
