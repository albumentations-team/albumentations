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

from ..utils import set_seed, to_tensor, from_tensor


def get_images(shape=(512, 512, 3), dtype=np.uint8):
    if dtype == np.uint8:
        image = np.random.randint(0, 256, shape, dtype=dtype)
    else:
        image = np.random.random(shape).astype(np.float32)

    return image, to_tensor(image)


def assert_images(cv_img, torch_img):
    cv_img = cv_img.transpose(2, 0, 1)
    assert_allclose(cv_img, torch_img)


@pytest.mark.parametrize(
    ["image", "augs"],
    itertools.product(
        [np.random.randint(0, 256, (128, 323, 3), np.uint8), np.random.random([256, 111, 3]).astype(np.float32)],
        [[A.Normalize(p=1), ATorch.NormalizeTorch(p=1)], [A.CoarseDropout(p=1), ATorch.CoarseDropoutTorch(p=1)]],
    ),
)
def test_image_transforms_rgb(image, augs):
    image_torch = to_tensor(image)

    aug_cpu = A.Compose([augs[0]])
    aug_torch = A.Compose([augs[1]])

    set_seed(0)
    image_cpu = aug_cpu(image=image)["image"]
    set_seed(0)
    image_torch = aug_torch(image=image_torch)["image"]

    image_torch = from_tensor(image_torch)
    assert_allclose(image_cpu, image_torch)


@pytest.mark.parametrize(
    ["image", "augs"],
    itertools.product(
        [np.random.randint(0, 256, (128, 323), np.uint8), np.random.random([256, 111]).astype(np.float32)],
        [
            [A.Normalize(mean=0.5, std=0.1, p=1), ATorch.NormalizeTorch(mean=0.5, std=0.1, p=1)],
            [A.CoarseDropout(p=1), ATorch.CoarseDropoutTorch(p=1)],
        ],
    ),
)
def test_image_transforms_grayscale(image, augs):
    aug_cpu = A.Compose([augs[0]])
    aug_torch = A.Compose([augs[1]])

    image_torch = to_tensor(image)

    set_seed(0)
    image_cpu = aug_cpu(image=image)["image"]
    set_seed(0)
    image_torch = aug_torch(image=image_torch)["image"]

    image_torch = from_tensor(image_torch)
    assert_allclose(image_cpu, image_torch)


def test_rgb_to_hls_float():
    image = np.random.random([1000, 1000, 3]).astype(np.float32)
    torch_img = to_tensor(image)

    cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    torch_img = FTorch.rgb_to_hls(torch_img)

    torch_img = from_tensor(torch_img)
    assert_allclose(cv_img, torch_img)


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


def test_rgb_to_hsv_float():
    image, torch_img = get_images(dtype=np.float32)

    cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    torch_img = FTorch.rgb_to_hsv(torch_img)

    torch_img = from_tensor(torch_img)

    cv_img = (cv_img * 255).astype(int)
    torch_img = (torch_img * 255).astype(int)

    assert np.abs(cv_img - torch_img).max() <= 3


def test_rgb_to_hsv_uint8():
    image, torch_img = get_images()

    cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    torch_img = FTorch.rgb_to_hsv(torch_img)

    torch_img = from_tensor(torch_img)

    val = np.abs(cv_img.astype(int) - torch_img.astype(int)).max()
    assert val <= 3 or val == 179


def test_hsv_to_rgb_float():
    image = np.random.random([1000, 1000, 3]).astype(np.float32)
    image[..., 0] *= 360

    torch_img = to_tensor(image)

    cv_img = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    torch_img = FTorch.hsv_to_rgb(torch_img)

    assert_images(cv_img, torch_img)


def test_hsv_to_rgb_uint8():
    image, torch_img = get_images()

    cv_img = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    torch_img = FTorch.hls_to_rgb(torch_img)

    assert_images(cv_img, torch_img)


def test_blur_float():
    image, torch_image = get_images(dtype=np.float32)
    cv_img = F.blur(image, 3)
    torch_image = FTorch.blur(torch_image, [3, 3])
    assert_images(cv_img, torch_image)


def test_blur_uint8():
    image, torch_image = get_images()
    cv_img = F.blur(image, 3)
    torch_image = FTorch.blur(torch_image, [3, 3])
    assert_images(cv_img, torch_image)


@pytest.mark.parametrize("images", [get_images(), get_images(dtype=np.float32)])
def test_solarize(images):
    image, torch_image = images

    image = F.solarize(image, 33)
    torch_image = FTorch.solarize(torch_image, 33)

    assert_images(image, torch_image)
