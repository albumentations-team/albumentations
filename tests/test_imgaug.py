import numpy as np
import pytest

from albumentations.imgaug.transforms import IAAPiecewiseAffine, IAAPerspective, IAAFliplr, IAAFlipud, IAACropAndPad
from albumentations.augmentations.bbox_utils import convert_bboxes_from_albumentations, \
    convert_bboxes_to_albumentations


@pytest.mark.parametrize('augmentation_cls', [IAAPiecewiseAffine, IAAPerspective])
def test_imagaug_dual_augmentations_are_deterministic(augmentation_cls, image):
    aug = augmentation_cls(p=1)
    mask = np.copy(image)
    data = aug(image=image, mask=mask)
    assert np.array_equal(data['image'], data['mask'])


def test_imagaug_fliplr_transform_bboxes(image):
    aug = IAAFliplr(p=1)
    mask = np.copy(image)
    bboxes = [(10, 10, 20, 20), (20, 10, 30, 40)]
    expect = [(79, 10, 89, 20), (69, 10, 79, 40)]
    bboxes = convert_bboxes_to_albumentations(bboxes, 'pascal_voc', rows=image.shape[0], cols=image.shape[1])
    data = aug(image=image, mask=mask, bboxes=bboxes)
    actual = convert_bboxes_from_albumentations(data['bboxes'], 'pascal_voc', rows=image.shape[0], cols=image.shape[1])
    assert np.array_equal(data['image'], data['mask'])
    assert np.allclose(actual, expect)


def test_imagaug_flipud_transform_bboxes(image):
    aug = IAAFlipud(p=1)
    mask = np.copy(image)
    dummy_class = 1234
    bboxes = [(10, 10, 20, 20, dummy_class), (20, 10, 30, 40, dummy_class)]
    expect = [(10, 79, 20, 89, dummy_class), (20, 59, 30, 89, dummy_class)]
    bboxes = convert_bboxes_to_albumentations(bboxes, 'pascal_voc', rows=image.shape[0], cols=image.shape[1])
    data = aug(image=image, mask=mask, bboxes=bboxes)
    actual = convert_bboxes_from_albumentations(data['bboxes'], 'pascal_voc', rows=image.shape[0], cols=image.shape[1])
    assert np.array_equal(data['image'], data['mask'])
    assert np.allclose(actual, expect)
