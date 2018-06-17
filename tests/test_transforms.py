import numpy as np

from albumentations import Transpose


def test_transpose_both_image_and_mask():
    image = np.ones((8, 6, 3))
    mask = np.ones((8, 6))
    augmentation = Transpose(p=1)
    augmented = augmentation(image=image, mask=mask)
    assert augmented['image'].shape == (6, 8, 3)
    assert augmented['mask'].shape == (6, 8)
