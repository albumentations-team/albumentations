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


@pytest.mark.parametrize("images_and_masks", [get_images_and_masks(), get_images_and_masks(dtype=np.float32)])
def test_pad_if_needed(images_and_masks):
    img, torch_img, mask, torch_mask = images_and_masks

    set_seed(0)
    result = A.PadIfNeeded(p=1)(image=img, mask=mask)
    set_seed(0)
    result_torch = ATorch.PadIfNeededTorch(p=1)(image=torch_img, mask=torch_mask)

    assert_images_and_masks(result, result_torch)
