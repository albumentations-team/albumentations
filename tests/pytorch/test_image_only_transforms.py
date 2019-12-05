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
        [
            [A.Normalize(), ATorch.NormalizeTorch()],
            [A.CoarseDropout(), ATorch.CoarseDropoutTorch()],
            [A.Blur([3, 3]), ATorch.BlurTorch([3, 3])],
            [A.Solarize(33), ATorch.SolarizeTorch(33)],
            [
                A.RandomBrightnessContrast((1.33, 1.33), (0.77, 0.77), True),
                ATorch.RandomBrightnessContrastTorch((1.33, 1.33), (0.77, 0.77), True),
            ],
            [A.ChannelDropout(), ATorch.ChannelDropoutTorch()],
            [A.RandomGamma(), ATorch.RandomGammaTorch()],
            [A.ChannelShuffle(), ATorch.ChannelShuffleTorch()],
            [A.ToGray(), ATorch.ToGrayTorch()],
            [A.ToFloat(), ATorch.ToFloatTorch()],
            [A.FromFloat("uint8"), ATorch.FromFloatTorch()],
        ],
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
        [
            [A.Normalize(mean=0.5, std=0.1, p=1), ATorch.NormalizeTorch(mean=0.5, std=0.1, p=1)],
            [A.CoarseDropout(p=1), ATorch.CoarseDropoutTorch(p=1)],
            [A.Blur([3, 3], p=1), ATorch.BlurTorch([3, 3], p=1)],
            [A.Solarize(33), ATorch.SolarizeTorch(33)],
            [
                A.RandomBrightnessContrast((1.33, 1.33), (0.77, 0.77), True),
                ATorch.RandomBrightnessContrastTorch((1.33, 1.33), (0.77, 0.77), True),
            ],
            [A.RandomGamma(), ATorch.RandomGammaTorch()],
            [A.ToFloat(), ATorch.ToFloatTorch()],
            [A.FromFloat("uint8"), ATorch.FromFloatTorch()],
        ],
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


@pytest.mark.parametrize(
    ["images", "shifts"], [[get_images(), [12, 250, 133]], [get_images(dtype=np.float32), [0.11, 0.3, 0.77]]]
)
def test_rgb_shift(images, shifts):
    image, torch_image = images

    image = F.shift_rgb(image, *shifts)
    torch_image = FTorch.shift_rgb(torch_image, *shifts)

    assert_images(image, torch_image)


def test_motion_blur():
    image, torch_image = get_images()
    kernel = 5

    set_seed(0)
    image = A.MotionBlur(blur_limit=[kernel, kernel], p=1)(image=image)["image"]
    set_seed(0)
    torch_image = ATorch.MotionBlurTorch(blur_limit=[kernel, kernel], p=1)(image=torch_image)["image"]

    assert_images(image, torch_image)


def test_median_blur():
    image, torch_image = get_images(dtype=np.float32)
    kernel = 5

    set_seed(0)
    image = A.MedianBlur(blur_limit=kernel, p=1)(image=image)["image"]
    set_seed(0)
    torch_image = ATorch.MedianBlurTorch(blur_limit=kernel, p=1)(image=torch_image)["image"]

    assert_images(image, torch_image, 1)
