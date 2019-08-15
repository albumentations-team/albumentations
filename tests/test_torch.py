import numpy as np
import torch

import albumentations as A
from albumentations.pytorch.transforms import ImgToTensor


def test_torch_to_tensor_augmentations(image, mask):
    aug = ImgToTensor()
    data = aug(image=image, force_apply=True)
    assert data['image'].dtype == torch.uint8


def test_additional_targets_for_totensor():
    aug = A.Compose(
        [ImgToTensor()], additional_targets={'image2': 'image'})
    for i in range(10):
        image1 = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        image2 = image1.copy()
        res = aug(image=image1, image2=image2)
        assert np.array_equal(res['image'], res['image2'])
