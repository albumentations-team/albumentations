import pytest

import random
import numpy as np
import torch

import albumentations as A
import albumentations.pytorch as ATorch

import itertools


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


@pytest.mark.parametrize(
    ["image", "augs"],
    itertools.product(
        [np.random.randint(0, 256, (128, 323, 3), np.uint8), np.random.random([256, 111, 3]).astype(np.float32)],
        [
            [A.Normalize(p=1), ATorch.NormalizeTorch(p=1)],
            [A.RandomSnow(p=1), ATorch.RandomSnowTorch(p=1)],
            [A.CoarseDropout(p=1), ATorch.CoarseDropoutTorch(p=1)],
        ],
    ),
)
def test_image_transforms_rgb(image, augs):
    aug_cpu = A.Compose([augs[0]])
    aug_cuda = A.Compose([augs[1]])

    image_cuda = torch.from_numpy(np.transpose(image, [2, 0, 1]))

    set_seed(0)
    image_cpu = aug_cpu(image=image)["image"]
    set_seed(0)
    image_cuda = aug_cuda(image=image_cuda)["image"]
    image_cuda = np.transpose(image_cuda.numpy(), [1, 2, 0])
    assert np.allclose(image_cpu, image_cuda)


@pytest.mark.parametrize(
    ["image", "augs"],
    itertools.product(
        [np.random.randint(0, 256, (128, 323), np.uint8), np.random.random([256, 111]).astype(np.float32)],
        [
            [A.Normalize(mean=0.5, std=0.1, p=1), ATorch.NormalizeTorch(mean=0.5, std=0.1, p=1)],
            [A.RandomSnow(p=1), ATorch.RandomSnowTorch(p=1)],
            [A.CoarseDropout(p=1), ATorch.CoarseDropoutTorch(p=1)],
        ],
    ),
)
def test_image_transforms_grayscale(image, augs):
    aug_cpu = A.Compose([augs[0]])
    aug_cuda = A.Compose([augs[1]])

    image_cuda = torch.from_numpy(image.reshape((1,) + image.shape))

    set_seed(0)
    image_cpu = aug_cpu(image=image)["image"]
    set_seed(0)
    image_cuda = aug_cuda(image=image_cuda)["image"]
    assert np.allclose(image_cpu, image_cuda.detach().numpy().squeeze())


args = itertools.product(
    [np.random.randint(0, 256, (128, 323, 3), np.uint8), np.random.random([256, 111, 3]).astype(np.float32)],
    [
        [A.Normalize(p=1), ATorch.NormalizeTorch(p=1)],
        [A.RandomSnow(p=1), ATorch.RandomSnowTorch(p=1)],
        [A.CoarseDropout(p=1), ATorch.CoarseDropoutTorch(p=1)],
    ],
)

for a in args:
    test_image_transforms_rgb(*a)


args = itertools.product(
    [np.random.randint(0, 256, (128, 323), np.uint8), np.random.random([256, 111]).astype(np.float32)],
    [
        [A.Normalize(mean=0.5, std=0.1, p=1), ATorch.NormalizeTorch(mean=0.5, std=0.1, p=1)],
        [A.RandomSnow(p=1), ATorch.RandomSnowTorch(p=1)],
        [A.CoarseDropout(p=1), ATorch.CoarseDropoutTorch(p=1)],
    ],
)

for a in args:
    test_image_transforms_grayscale(*a)
