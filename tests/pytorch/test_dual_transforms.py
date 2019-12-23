import cv2
import torch
import pytest
import itertools
import numpy as np

import albumentations as A
import albumentations.pytorch as ATorch

import albumentations.augmentations.functional as F
import albumentations.pytorch.augmentations.image_only.functional as FTorch

from torch.testing import assert_allclose

from tests.utils import set_seed, to_tensor, from_tensor


def get_images_and_masks(shape=(512, 512, 3), dtype=np.uint8):
    if dtype == np.uint8:
        image = np.random.randint(0, 256, shape, dtype=dtype)
    else:
        image = np.random.random(shape).astype(np.float32)

    mask = np.random.randint(0, 1, image.shape, np.uint8)

    return image, to_tensor(image), mask, to_tensor(mask)


def assert_images_and_masks(result, torch_result, rtol=None):
    img, mask = result["image"], result["mask"]
    torch_img, torch_mask = torch_result["image"], torch_result["mask"]

    torch_img = torch_img.permute(1, 2, 0).squeeze()
    torch_mask = torch_mask.permute(1, 2, 0).squeeze()

    assert_allclose(img, torch_img, rtol=rtol, atol=rtol)
    assert_allclose(mask, torch_mask, rtol=rtol, atol=rtol)


@pytest.mark.parametrize(
    ["images_and_masks", "augs"],
    itertools.product(
        [
            get_images_and_masks((128, 323, 3), np.uint8),
            get_images_and_masks([256, 111, 3], np.float32),
            get_images_and_masks((128, 323), np.uint8),
            get_images_and_masks([256, 111], np.float32),
        ],
        [
            [A.PadIfNeeded(333, 512), ATorch.PadIfNeededTorch(333, 512)],
            [A.Crop(11, 5, 72, 36), ATorch.CropTorch(11, 5, 72, 36)],
            [A.VerticalFlip(), ATorch.VerticalFlipTorch()],
            [A.HorizontalFlip(), ATorch.HorizontalFlipTorch()],
            [A.Flip(), ATorch.FlipTorch()],
            [A.Transpose(), ATorch.TransposeTorch()],
            [
                A.LongestMaxSize(interpolation=cv2.INTER_NEAREST),
                ATorch.LongestMaxSizeTorch(interpolation=cv2.INTER_NEAREST),
            ],
            [
                A.SmallestMaxSize(interpolation=cv2.INTER_NEAREST),
                ATorch.SmallestMaxSizeTorch(interpolation=cv2.INTER_NEAREST),
            ],
            [
                A.Resize(100, 100, interpolation=cv2.INTER_NEAREST),
                ATorch.ResizeTorch(100, 100, interpolation=cv2.INTER_NEAREST),
            ],
            [A.RandomRotate90(), ATorch.RandomRotate90Torch()],
            [A.RandomScale(interpolation=cv2.INTER_NEAREST), ATorch.RandomScaleTorch(interpolation=cv2.INTER_NEAREST)],
            [A.CenterCrop(100, 100), ATorch.CenterCropTorch(100, 100)],
            [A.RandomCrop(50, 50), ATorch.RandomCropTorch(50, 50)],
            [A.RandomSizedCrop([40, 50], 40, 50), ATorch.RandomSizedCropTorch([40, 50], 40, 50)],
            [
                A.RandomResizedCrop(40, 50, interpolation=cv2.INTER_NEAREST),
                ATorch.RandomResizedCropTorch(40, 50, interpolation=cv2.INTER_NEAREST),
            ],
            [A.CropNonEmptyMaskIfExists(50, 50), ATorch.CropNonEmptyMaskIfExistsTorch(50, 50)],
        ],
    ),
)
def test_image_transforms(images_and_masks, augs):
    image, torch_image, mask, torch_mask = images_and_masks
    dtype = image.dtype

    aug_cpu, aug_torch = augs
    aug_cpu.p = 1
    aug_torch.p = 1

    aug_cpu = A.Compose([aug_cpu])
    aug_torch = A.Compose([aug_torch])

    seed = 0
    set_seed(seed)
    result = aug_cpu(image=image, mask=mask)
    set_seed(seed)
    torch_result = aug_torch(image=torch_image, mask=torch_mask)

    assert_images_and_masks(result, torch_result, 1 if dtype == np.uint8 else None)
