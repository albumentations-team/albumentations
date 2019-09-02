import numpy as np
import torch

import albumentations as A
from albumentations.pytorch.transforms import ToTensor


def test_torch_to_tensor_augmentations(image, mask):
    aug = ToTensor()
    data = aug(image=image, mask=mask, force_apply=True)
    assert isinstance(data['image'], torch.Tensor) and data['image'].shape == image.shape[::-1]
    assert isinstance(data['mask'], torch.Tensor) and data['mask'].shape == mask.shape
    assert data['image'].dtype == torch.uint8
    assert data['mask'].dtype == torch.uint8


def test_additional_targets_for_totensor():
    aug = A.Compose(
        [ToTensor()], additional_targets={'image2': 'image', 'mask2': 'mask'})
    for i in range(10):
        image1 = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        image2 = image1.copy()
        mask1 = np.random.randint(low=0, high=256, size=(100, 100, 4), dtype=np.uint8)
        mask2 = mask1.copy()
        res = aug(image=image1, image2=image2, mask=mask1, mask2=mask2)
        assert isinstance(res['image'], torch.Tensor) and res['image'].shape == image1.shape[::-1]
        assert isinstance(res['image2'], torch.Tensor) and res['image2'].shape == image2.shape[::-1]
        assert isinstance(res['mask'], torch.Tensor) and res['mask'].shape == mask1.shape
        assert isinstance(res['mask2'], torch.Tensor) and res['mask2'].shape == mask2.shape
        assert np.array_equal(res['image'], res['image2'])
        assert np.array_equal(res['mask'], res['mask2'])
