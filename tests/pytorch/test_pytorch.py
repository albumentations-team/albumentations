import pytest

import cv2
import torch
import random
import itertools
import numpy as np

import albumentations as A
import albumentations.pytorch as ATorch

import albumentations.augmentations.functional as F
import albumentations.pytorch.augmentations.image_only.functional as FTorch


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def test_torch_to_tensor_v2_augmentations(image, mask):
    aug = ATorch.ToTensorV2()
    data = aug(image=image, mask=mask, force_apply=True)
    assert isinstance(data["image"], torch.Tensor) and data["image"].shape == image.shape[::-1]
    assert isinstance(data["mask"], torch.Tensor) and data["mask"].shape == mask.shape
    assert data["image"].dtype == torch.uint8
    assert data["mask"].dtype == torch.uint8


def test_additional_targets_for_totensorv2():
    aug = A.Compose([ATorch.ToTensorV2()], additional_targets={"image2": "image", "mask2": "mask"})
    for _i in range(10):
        image1 = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        image2 = image1.copy()
        mask1 = np.random.randint(low=0, high=256, size=(100, 100, 4), dtype=np.uint8)
        mask2 = mask1.copy()
        res = aug(image=image1, image2=image2, mask=mask1, mask2=mask2)
        assert isinstance(res["image"], torch.Tensor) and res["image"].shape == image1.shape[::-1]
        assert isinstance(res["image2"], torch.Tensor) and res["image2"].shape == image2.shape[::-1]
        assert isinstance(res["mask"], torch.Tensor) and res["mask"].shape == mask1.shape
        assert isinstance(res["mask2"], torch.Tensor) and res["mask2"].shape == mask2.shape
        assert np.array_equal(res["image"], res["image2"])
        assert np.array_equal(res["mask"], res["mask2"])


def test_torch_to_tensor_augmentations(image, mask):
    with pytest.warns(DeprecationWarning):
        aug = ATorch.ToTensor()
    data = aug(image=image, mask=mask, force_apply=True)
    assert data["image"].dtype == torch.float32
    assert data["mask"].dtype == torch.float32


def test_additional_targets_for_totensor():
    with pytest.warns(DeprecationWarning):
        aug = A.Compose([ATorch.ToTensor(num_classes=4)], additional_targets={"image2": "image", "mask2": "mask"})
    for _i in range(10):
        image1 = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        image2 = image1.copy()
        mask1 = np.random.randint(low=0, high=256, size=(100, 100, 4), dtype=np.uint8)
        mask2 = mask1.copy()
        res = aug(image=image1, image2=image2, mask=mask1, mask2=mask2)
        assert np.array_equal(res["image"], res["image2"])
        assert np.array_equal(res["mask"], res["mask2"])


def test_with_replaycompose():
    aug = A.ReplayCompose([ATorch.ToTensorV2()])
    kwargs = {
        "image": np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
        "mask": np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
    }
    res = aug(**kwargs)
    res2 = A.ReplayCompose.replay(res["replay"], **kwargs)
    assert np.array_equal(res["image"], res2["image"])
    assert np.array_equal(res["mask"], res2["mask"])
    assert res["image"].dtype == torch.uint8
    assert res["mask"].dtype == torch.uint8
    assert res2["image"].dtype == torch.uint8
    assert res2["mask"].dtype == torch.uint8
