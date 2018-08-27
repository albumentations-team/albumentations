import numpy as np
import pytest

from albumentations.imgaug.transforms import IAAPiecewiseAffine, IAAPerspective, IAAFliplr, IAAFlipud, IAACropAndPad


@pytest.mark.parametrize('augmentation_cls', [IAAPiecewiseAffine, IAAPerspective])
def test_imagaug_dual_augmentations_are_deterministic(augmentation_cls, image):
    aug = augmentation_cls(p=1)
    mask = np.copy(image)
    data = aug(image=image, mask=mask)
    assert np.array_equal(data['image'], data['mask'])


@pytest.mark.parametrize('augmentation_cls', [IAAFliplr])
def test_imagaug_dual_augmentations_transform_bboxes(augmentation_cls, image):
    aug = augmentation_cls(p=1)
    mask = np.copy(image)
    data = aug(image=image, mask=mask, bboxes=[(10, 10, 20, 20), (20, 10, 30, 40)])
    assert np.array_equal(data['image'], data['mask'])
    assert np.array_equal(data['bboxes'], [(79, 10, 89, 20), (69, 10, 79, 40)])


@pytest.mark.parametrize('augmentation_cls', [IAAFlipud])
def test_imagaug_dual_augmentations_transform_bboxes(augmentation_cls, image):
    aug = augmentation_cls(p=1)
    mask = np.copy(image)
    data = aug(image=image, mask=mask, bboxes=[(10, 10, 20, 20), (20, 10, 30, 40)])
    assert np.array_equal(data['image'], data['mask'])
    assert np.array_equal(data['bboxes'], [(10, 79, 20, 89), (20, 59, 30, 89)])
