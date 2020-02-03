import cv2
import pytest
import itertools
import numpy as np

import albumentations as A
import albumentations.pytorch as ATorch

import albumentations.augmentations.functional as F
import albumentations.pytorch.augmentations.image_only.functional as FTorch

from torch.testing import assert_allclose

from tests.utils import set_seed, to_tensor, from_tensor


def get_images(shape=(512, 512, 3), dtype=np.uint8):
    if dtype == np.uint8:
        image = np.random.randint(0, 256, shape, dtype=dtype)
    else:
        image = np.random.random(shape).astype(np.float32)

    return image, to_tensor(image)


def assert_images(img, torch_img, rtol=None):
    torch_img = torch_img.permute(1, 2, 0).squeeze()
    assert_allclose(img, torch_img, rtol=rtol, atol=rtol)


@pytest.mark.parametrize(
    ["images", "augs"],
    itertools.product(
        [get_images((128, 323, 3), np.uint8), get_images([256, 111, 3], np.float32)],
        [[A.Normalize(), ATorch.NormalizeTorch()]],
    ),
)
def test_image_transforms_rgb(images, augs):
    image, image_torch = images
    dtype = image.dtype

    aug_cpu, aug_torch = augs
    aug_cpu.p = 1
    aug_torch.p = 1

    aug_cpu = A.Compose([aug_cpu])
    aug_torch = A.Compose([aug_torch])

    set_seed(0)
    image = aug_cpu(image=image)["image"]
    set_seed(0)
    image_torch = aug_torch(image=image_torch)["image"]

    assert_images(image, image_torch, 1 if dtype == np.uint8 else None)


@pytest.mark.parametrize(
    ["images", "augs"],
    itertools.product(
        [get_images((128, 323), np.uint8), get_images([256, 111], np.float32)],
        [[A.Normalize(mean=0.5, std=0.1, p=1), ATorch.NormalizeTorch(mean=0.5, std=0.1, p=1)]],
    ),
)
def test_image_transforms_grayscale(images, augs):
    image, image_torch = images
    dtype = image.dtype

    aug_cpu, aug_torch = augs
    aug_cpu.p = 1
    aug_torch.p = 1

    aug_cpu = A.Compose([aug_cpu])
    aug_torch = A.Compose([aug_torch])

    set_seed(0)
    image = aug_cpu(image=image)["image"]
    set_seed(0)
    image_torch = aug_torch(image=image_torch)["image"]

    assert_images(image, image_torch, 1 if dtype == np.uint8 else None)


def test_rgb_to_hls_float():
    image = np.random.random([1000, 1000, 3]).astype(np.float32)
    torch_img = to_tensor(image)

    cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    torch_img = FTorch.rgb_to_hls(torch_img)

    torch_img = from_tensor(torch_img)
    assert_allclose(cv_img, torch_img, 1e-3, 1e-4)


def test_rgb_to_hls_uint8():
    image = np.random.randint(0, 256, [1000, 1000, 3], np.uint8)
    torch_img = to_tensor(image)

    cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    torch_img = FTorch.rgb_to_hls(torch_img)

    torch_img = from_tensor(torch_img)
    assert np.abs(cv_img.astype(int) == torch_img.astype(int)).max() <= 1


def test_hls_to_rgb_float():
    image = np.random.random([1000, 1000, 3]).astype(np.float32)
    image[..., 0] *= 360

    torch_img = to_tensor(image)

    cv_img = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    torch_img = FTorch.hls_to_rgb(torch_img)

    torch_img = from_tensor(torch_img)
    assert_allclose(cv_img, torch_img)


def test_hls_to_rgb_uint8():
    image = np.random.randint(0, 256, [1000, 1000, 3], np.uint8)

    torch_img = to_tensor(image)

    cv_img = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    torch_img = FTorch.hls_to_rgb(torch_img)

    torch_img = from_tensor(torch_img)
    assert np.all(cv_img == torch_img)
