import numpy as np
import pytest

from albumentations.imgaug.transforms import IAAPiecewiseAffine, IAAPerspective


@pytest.mark.parametrize('augmentation_cls', [IAAPiecewiseAffine, IAAPerspective])
def test_imagaug_dual_augmentations_are_deterministic(augmentation_cls, image):
    aug = augmentation_cls(p=1)
    mask = np.copy(image)
    data = aug(image=image, mask=mask)
    assert np.array_equal(data['image'], data['mask'])
