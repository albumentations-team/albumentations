import random
from functools import partial
from typing import Any, Dict, Optional, Tuple, Type

import cv2
import numpy as np

import pytest
import warnings
from torchvision import transforms as torch_transforms

from albumentations.core.pydantic import valid_interpolations, valid_border_modes

from albucore.utils import clip
import albumentations as A
import albumentations.augmentations.functional as F
import albumentations.augmentations.geometric.functional as FGeometric
from albumentations.augmentations.transforms import ImageCompression, RandomRain
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.types import ImageCompressionType
from albumentations.random_utils import get_random_seed
from albumentations.augmentations.transforms import RandomSnow
from tests.conftest import IMAGES, RECTANGULAR_FLOAT_IMAGE, SQUARE_FLOAT_IMAGE, SQUARE_MULTI_UINT8_IMAGE, SQUARE_UINT8_IMAGE

from .utils import get_dual_transforms, get_image_only_transforms, get_transforms, set_seed



def test_transpose_both_image_and_mask():
    image = np.ones((8, 6, 3))
    mask = np.ones((8, 6))
    augmentation = A.Transpose(p=1)
    augmented = augmentation(image=image, mask=mask)
    assert augmented["image"].shape == (6, 8, 3)
    assert augmented["mask"].shape == (6, 8)


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = A.Rotate(limit=(45, 45), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = FGeometric.rotate(image, 45, interpolation=interpolation, border_mode=cv2.BORDER_REFLECT_101)
    expected_mask = FGeometric.rotate(mask, 45, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(data["image"], expected_image)
    assert np.array_equal(data["mask"], expected_mask)


def test_rotate_crop_border():
    image = np.random.randint(low=100, high=256, size=(100, 100, 3), dtype=np.uint8)
    border_value = 13
    aug = A.Rotate(limit=(45, 45), p=1, value=border_value, border_mode=cv2.BORDER_CONSTANT, crop_border=True)
    aug_img = aug(image=image)["image"]
    expected_size = int(np.round(100 / np.sqrt(2)))
    assert aug_img.shape[0] == expected_size
    assert (aug_img == border_value).sum() == 0


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_optical_distortion_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = A.OpticalDistortion(distort_limit=(0.05, 0.05), shift_limit=(0, 0), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = FGeometric.optical_distortion(
        image, k=0.05, dx=0, dy=0, interpolation=interpolation, border_mode=cv2.BORDER_REFLECT_101
    )
    expected_mask = FGeometric.optical_distortion(
        mask, k=0.05, dx=0, dy=0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101
    )
    assert np.array_equal(data["image"], expected_image)
    assert np.array_equal(data["mask"], expected_mask)


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_grid_distortion_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = A.GridDistortion(num_steps=1, distort_limit=(0.3, 0.3), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = FGeometric.grid_distortion(
        image, num_steps=1, xsteps=[1.3], ysteps=[1.3], interpolation=interpolation, border_mode=cv2.BORDER_REFLECT_101
    )
    expected_mask = FGeometric.grid_distortion(
        mask,
        num_steps=1,
        xsteps=[1.3],
        ysteps=[1.3],
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
    )
    assert np.array_equal(data["image"], expected_image)
    assert np.array_equal(data["mask"], expected_mask)


@pytest.mark.parametrize("size", [17, 21, 33])
def test_grid_distortion_steps(size):
    image = np.random.rand(size, size, 3)
    aug = A.GridDistortion(num_steps=size - 2, p=1)
    data = aug(image=image)
    assert np.array_equal(data["image"].shape, (size, size, 3))


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_elastic_transform_interpolation(monkeypatch, interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)

    random_seed = get_random_seed()

    monkeypatch.setattr(
        "albumentations.augmentations.geometric.ElasticTransform.get_params", lambda *_: {"random_seed": random_seed}
    )

    aug = A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=interpolation, p=1)

    data = aug(image=image, mask=mask)
    expected_image = FGeometric.elastic_transform(
        image,
        alpha=1,
        sigma=50,
        alpha_affine=50,
        interpolation=interpolation,
        border_mode=cv2.BORDER_REFLECT_101,
        random_state=np.random.RandomState(random_seed),
    )
    expected_mask = FGeometric.elastic_transform(
        mask,
        alpha=1,
        sigma=50,
        alpha_affine=50,
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        random_state=np.random.RandomState(random_seed),
    )
    assert np.array_equal(data["image"], expected_image)
    assert np.array_equal(data["mask"], expected_mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.PixelDropout: {"dropout_prob": 0.5, "mask_drop_value": 10, "drop_value": 20},
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.D4: {},
        },
        except_augmentations={A.RandomCropNearBBox, A.RandomSizedBBoxSafeCrop, A.BBoxSafeRandomCrop, A.PixelDropout,
                              A.MixUp},
    ),
)
def test_binary_mask_interpolation(augmentation_cls, params):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts"""
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    data = aug(image=image, mask=mask)
    assert np.array_equal(np.unique(data["mask"]), np.array([0, 1]))


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.Resize: {"height": 10, "width": 10},
            A.PixelDropout: {"dropout_prob": 0.5, "mask_drop_value": 10, "drop_value": 20},
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropAndPad,
            A.PixelDropout,
            A.MixUp,
            A.XYMasking
        },
    ),
)
def test_semantic_mask_interpolation(augmentation_cls, params):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts."""
    aug = augmentation_cls(p=1, **params)
    image = SQUARE_UINT8_IMAGE
    mask = np.random.randint(low=0, high=4, size=(100, 100), dtype=np.uint8) * 64

    data = aug(image=image, mask=mask)
    assert np.array_equal(np.unique(data["mask"]), np.array([0, 64, 128, 192]))


def __test_multiprocessing_support_proc(args):
    x, transform = args
    return transform(image=x)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.TemplateTransform: {
                "templates": np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
            },
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.MaskDropout,
            A.MixUp
        },
    ),
)
def test_multiprocessing_support(mp_pool, augmentation_cls, params):
    """Checks whether we can use augmentations in multiprocessing environments"""
    image = SQUARE_UINT8_IMAGE
    aug = augmentation_cls(p=1, **params)

    mp_pool.map(__test_multiprocessing_support_proc, map(lambda x: (x, aug), [image] * 10))


def test_force_apply():
    """
    Unit test for https://github.com/albumentations-team/albumentations/issues/189
    """
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose(
                    [
                        A.RandomSizedCrop(min_max_height=(256, 1025), height=512, width=512, p=1),
                        A.OneOf(
                            [
                                A.RandomSizedCrop(min_max_height=(256, 512), height=384, width=384, p=0.5),
                                A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=0.5),
                            ]
                        ),
                    ]
                ),
                A.Compose(
                    [
                        A.RandomSizedCrop(min_max_height=(256, 1025), height=256, width=256, p=1),
                        A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1),
                    ]
                ),
            ),
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(p=0.5),
        ]
    )

    res = aug(image=np.zeros((1248, 1248, 3), dtype=np.uint8))
    assert res["image"].shape[0] in (256, 384, 512)
    assert res["image"].shape[1] in (256, 384, 512)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [SQUARE_UINT8_IMAGE],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [SQUARE_UINT8_IMAGE],
                "read_fn": lambda x: x,
            },
            A.PixelDistributionAdaptation: {
                "reference_images": [SQUARE_UINT8_IMAGE],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
            A.TemplateTransform: {
                "templates": SQUARE_UINT8_IMAGE,
            },
        },
    ),
)
def test_additional_targets_for_image_only(augmentation_cls, params):
    aug = A.Compose([augmentation_cls(p=1, **params)], additional_targets={"image2": "image"})
    for _ in range(10):
        image1 = SQUARE_UINT8_IMAGE
        image2 = image1.copy()
        res = aug(image=image1, image2=image2)
        aug1 = res["image"]
        aug2 = res["image2"]
        assert np.array_equal(aug1, aug2)

    aug = A.Compose([augmentation_cls(p=1, **params)])
    aug.add_targets(additional_targets={"image2": "image"})
    for _ in range(10):
        image1 = SQUARE_UINT8_IMAGE
        image2 = image1.copy()
        res = aug(image=image1, image2=image2)
        aug1 = res["image"]
        aug2 = res["image2"]
        assert np.array_equal(aug1, aug2)

def test_image_invert():
    for _ in range(10):
        # test for np.uint8 dtype
        image1 = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        image2 = A.to_float(image1)
        r_int = F.invert(F.invert(image1))
        r_float = F.invert(F.invert(image2))
        r_to_float = A.to_float(r_int)
        assert np.allclose(r_float, r_to_float, atol=0.01)


def test_lambda_transform():
    def negate_image(image, **kwargs):
        return -image

    def one_hot_mask(mask, num_channels, **kwargs):
        new_mask = np.eye(num_channels, dtype=np.uint8)[mask]
        return new_mask

    def vflip_bbox(bbox, **kwargs):
        return FGeometric.bbox_vflip(bbox, **kwargs)

    def vflip_keypoint(keypoint, **kwargs):
        return FGeometric.keypoint_vflip(keypoint, **kwargs)

    aug = A.Lambda(
        image=negate_image, mask=partial(one_hot_mask, num_channels=16), bbox=vflip_bbox, keypoint=vflip_keypoint, p=1
    )

    output = aug(
        image=np.ones((10, 10, 3), dtype=np.float32),
        mask=np.tile(np.arange(0, 10), (10, 1)),
        bboxes=[(10, 15, 25, 35)],
        keypoints=[(20, 30, 40, 50)],
    )
    assert (output["image"] < 0).all()
    assert output["mask"].shape[2] == 16  # num_channels
    assert output["bboxes"] == [FGeometric.bbox_vflip((10, 15, 25, 35), 10, 10)]
    assert output["keypoints"] == [FGeometric.keypoint_vflip((20, 30, 40, 50), 10, 10)]


def test_channel_droput():
    img = np.ones((10, 10, 3), dtype=np.float32)

    aug = A.ChannelDropout(channel_drop_range=(1, 1), p=1)  # Drop one channel

    transformed = aug(image=img)["image"]

    assert sum(transformed[:, :, c].max() for c in range(img.shape[2])) == 2

    aug = A.ChannelDropout(channel_drop_range=(2, 2), p=1)  # Drop two channels
    transformed = aug(image=img)["image"]

    assert sum(transformed[:, :, c].max() for c in range(img.shape[2])) == 1


def test_equalize():
    aug = A.Equalize(p=1)

    img = np.random.randint(0, 256, 256 * 256 * 3, np.uint8).reshape((256, 256, 3))
    a = aug(image=img)["image"]
    b = F.equalize(img)
    assert np.all(a == b)

    mask = np.random.randint(0, 2, 256 * 256, np.uint8).reshape((256, 256))
    aug = A.Equalize(mask=mask, p=1)
    a = aug(image=img)["image"]
    b = F.equalize(img, mask=mask)
    assert np.all(a == b)

    def mask_func(image, test):
        return mask

    aug = A.Equalize(mask=mask_func, mask_params=["test"], p=1)
    assert np.all(aug(image=img, test=mask)["image"] == F.equalize(img, mask=mask))


def test_crop_non_empty_mask():
    def _test_crop(mask, crop, aug, n=1):
        for _ in range(n):
            augmented = aug(image=mask, mask=mask)
            np.testing.assert_array_equal(augmented["image"], crop)
            np.testing.assert_array_equal(augmented["mask"], crop)

    def _test_crops(masks, crops, aug, n=1):
        for _ in range(n):
            augmented = aug(image=masks[0], masks=masks)
            for crop, augment in zip(crops, augmented["masks"]):
                np.testing.assert_array_equal(augment, crop)

    # test general case
    mask_1 = np.zeros([10, 10], dtype=np.uint8)  # uint8 required for passing mask_1 as `masks` (which uses bitwise or)
    mask_1[0, 0] = 1
    crop_1 = np.array([[1]])
    aug_1 = A.CropNonEmptyMaskIfExists(1, 1)

    # test empty mask
    mask_2 = np.zeros([10, 10], dtype=np.uint8)  # uint8 required for passing mask_2 as `masks` (which uses bitwise or)
    crop_2 = np.array([[0]])
    aug_2 = A.CropNonEmptyMaskIfExists(1, 1)

    # test ignore values
    mask_3 = np.ones([2, 2])
    mask_3[0, 0] = 2
    crop_3 = np.array([[2]])
    aug_3 = A.CropNonEmptyMaskIfExists(1, 1, ignore_values=[1])

    # test ignore channels
    mask_4 = np.zeros([2, 2, 2])
    mask_4[0, 0, 0] = 1
    mask_4[1, 1, 1] = 2
    crop_4 = np.array([[[1, 0]]])
    aug_4 = A.CropNonEmptyMaskIfExists(1, 1, ignore_channels=[1])

    # test full size crop
    mask_5 = np.random.random([10, 10, 3])
    crop_5 = mask_5
    aug_5 = A.CropNonEmptyMaskIfExists(10, 10)

    mask_6 = np.zeros([10, 10, 3])
    mask_6[0, 0, 0] = 0
    crop_6 = mask_6
    aug_6 = A.CropNonEmptyMaskIfExists(10, 10, ignore_values=[1])

    _test_crop(mask_1, crop_1, aug_1, n=1)
    _test_crop(mask_2, crop_2, aug_2, n=1)
    _test_crop(mask_3, crop_3, aug_3, n=5)
    _test_crop(mask_4, crop_4, aug_4, n=5)
    _test_crop(mask_5, crop_5, aug_5, n=1)
    _test_crop(mask_6, crop_6, aug_6, n=10)
    _test_crops([mask_2, mask_1], [crop_2, crop_1], aug_1, n=1)


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_downscale(interpolation):
    img_float = SQUARE_FLOAT_IMAGE
    img_uint = (img_float * 255).astype("uint8")

    aug = A.Downscale(scale_min=0.5, scale_max=0.5, interpolation=interpolation, p=1)

    for img in (img_float, img_uint):
        transformed = aug(image=img)["image"]
        func_applied = F.downscale(img, scale=0.5, down_interpolation=interpolation, up_interpolation=interpolation)
        np.testing.assert_almost_equal(transformed, func_applied)


def test_crop_keypoints():
    image = np.random.randint(0, 256, (100, 100), np.uint8)
    keypoints = [(50, 50, 0, 0)]

    aug = A.Crop(0, 0, 80, 80, p=1)
    result = aug(image=image, keypoints=keypoints)
    assert result["keypoints"] == keypoints

    aug = A.Crop(50, 50, 100, 100, p=1)
    result = aug(image=image, keypoints=keypoints)
    assert result["keypoints"] == [(0, 0, 0, 0)]


def test_longest_max_size_keypoints():
    img = np.random.randint(0, 256, [50, 10], np.uint8)
    keypoints = [(9, 5, 0, 0)]

    aug = A.LongestMaxSize(max_size=100, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(18, 10, 0, 0)]

    aug = A.LongestMaxSize(max_size=5, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(0.9, 0.5, 0, 0)]

    aug = A.LongestMaxSize(max_size=50, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(9, 5, 0, 0)]


def test_smallest_max_size_keypoints():
    img = np.random.randint(0, 256, [50, 10], np.uint8)
    keypoints = [(9, 5, 0, 0)]

    aug = A.SmallestMaxSize(max_size=100, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(90, 50, 0, 0)]

    aug = A.SmallestMaxSize(max_size=5, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(4.5, 2.5, 0, 0)]

    aug = A.SmallestMaxSize(max_size=10, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(9, 5, 0, 0)]


def test_resize_keypoints():
    img = np.random.randint(0, 256, [50, 10], np.uint8)
    keypoints = [(9, 5, 0, 0)]

    aug = A.Resize(height=100, width=5, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(4.5, 10, 0, 0)]

    aug = A.Resize(height=50, width=10, p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["keypoints"] == [(9, 5, 0, 0)]


@pytest.mark.parametrize(
    "image",
    [
        np.random.randint(0, 256, [256, 320], np.uint8),
        np.random.random([256, 320]).astype(np.float32),
        np.random.randint(0, 256, [256, 320, 1], np.uint8),
        np.random.random([256, 320, 1]).astype(np.float32),
    ],
)
def test_multiplicative_noise_grayscale(image):
    m = 0.5
    aug = A.MultiplicativeNoise((m, m), elementwise=False, p=1)
    params = aug.get_params_dependent_on_targets({"image": image})
    assert m == params["multiplier"]
    result_e = aug(image=image)["image"]

    expected = image.astype(np.float32) * params["multiplier"]

    assert np.allclose(clip(expected, image.dtype), result_e)

    aug = A.MultiplicativeNoise((m, m), elementwise=True, p=1)
    params = aug.get_params_dependent_on_targets({"image": image})
    result_ne = aug.apply(image, params["multiplier"])

    expected = image.astype(np.float32) * params["multiplier"]

    assert np.allclose(clip(expected, image.dtype), result_ne)

@pytest.mark.parametrize(
    "image", IMAGES
)
@pytest.mark.parametrize(
    "elementwise", ( True, False )
)
def test_multiplicative_noise_rgb(image, elementwise):
    dtype = image.dtype

    aug = A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=elementwise, p=1)
    params = aug.get_params_dependent_on_targets({"image": image})
    mul = params["multiplier"]

    if elementwise:
        assert mul.shape == image.shape
    else:
        assert mul.shape == (image.shape[-1],)

    result = aug.apply(image, mul)

    expected = image.astype(np.float32) * mul

    assert np.allclose(clip(expected, dtype), result, atol=1e-5)


def test_mask_dropout():
    # In this case we have mask with all ones, so MaskDropout wipe entire mask and image
    img = np.random.randint(0, 256, [50, 10], np.uint8)
    mask = np.ones([50, 10], dtype=np.int64)

    aug = A.MaskDropout(p=1)
    result = aug(image=img, mask=mask)
    assert np.all(result["image"] == 0)
    assert np.all(result["mask"] == 0)

    # In this case we have mask with zeros , so MaskDropout will make no changes
    img = np.random.randint(0, 256, [50, 10], np.uint8)
    mask = np.zeros([50, 10], dtype=np.int64)

    aug = A.MaskDropout(p=1)
    result = aug(image=img, mask=mask)
    assert np.all(result["image"] == img)
    assert np.all(result["mask"] == 0)


@pytest.mark.parametrize( "image", IMAGES )
def test_grid_dropout_mask(image):
    height, width = image.shape[:2]
    mask = np.ones([height, width], dtype=np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=0)
    result = aug(image=image, mask=mask)
    # with mask on ones and fill_value = 0 the sum of pixels is smaller
    assert result["image"].sum() < image.sum()
    assert result["image"].shape == image.shape
    assert result["mask"].sum() < mask.sum()
    assert result["mask"].shape == mask.shape

    # with mask of zeros and fill_value = 0 mask should not change
    mask = np.zeros([height, width], dtype=np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=0)
    result = aug(image=image, mask=mask)
    assert result["image"].sum() < image.sum()
    assert np.all(result["mask"] == 0)

    # with mask mask_fill_value=100, mask sum is larger
    mask = np.random.randint(0, 10, [height, width], np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=100)
    result = aug(image=image, mask=mask)
    assert result["image"].sum() < image.sum()
    assert result["mask"].sum() > mask.sum()

    # with mask mask_fill_value=None, mask is not changed
    mask = np.ones([height, width], dtype=np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=None)
    result = aug(image=image, mask=mask)
    assert result["image"].sum() < image.sum()
    assert result["mask"].sum() == mask.sum()


@pytest.mark.parametrize(
    ["ratio", "holes_number_x", "holes_number_y", "unit_size_min", "unit_size_max", "shift_x", "shift_y"],
    [
        (0.00001, 10, 10, 100, 100, 50, 50),
        (0.9, 100, None, 200, None, 0, 0),
        (0.4556, 10, 20, None, 200, 0, 0),
        (0.00004, None, None, 2, 100, 0, 0),
    ],
)
def test_grid_dropout_params(ratio, holes_number_x, holes_number_y, unit_size_min, unit_size_max, shift_x, shift_y):
    img = np.random.randint(0, 256, [256, 320], np.uint8)

    aug = A.GridDropout(
        ratio=ratio,
        unit_size_min=unit_size_min,
        unit_size_max=unit_size_max,
        holes_number_x=holes_number_x,
        holes_number_y=holes_number_y,
        shift_x=shift_x,
        shift_y=shift_y,
        random_offset=False,
        fill_value=0,
        p=1,
    )
    result = aug(image=img)["image"]
    # with fill_value = 0 the sum of pixels is smaller
    assert result.sum() < img.sum()
    assert result.shape == img.shape
    params = aug.get_params_dependent_on_targets({"image": img})
    holes = params["holes"]
    assert len(holes[0]) == 4
    # check grid offsets
    if shift_x:
        assert holes[0][0] == shift_x
    else:
        assert holes[0][0] == 0
    if shift_y:
        assert holes[0][1] == shift_y
    else:
        assert holes[0][1] == 0
    # for grid set with limits
    if unit_size_min and unit_size_max:
        assert max(1, unit_size_min * ratio) <= (holes[0][2] - holes[0][0]) <= min(max(1, unit_size_max * ratio), 256)
    elif holes_number_x and holes_number_y:
        assert (holes[0][2] - holes[0][0]) == max(1, int(ratio * 320 // holes_number_x))
        assert (holes[0][3] - holes[0][1]) == max(1, int(ratio * 256 // holes_number_y))


@pytest.mark.parametrize(
    ["blur_limit", "sigma", "result_blur", "result_sigma"],
    [
        [[0, 0], [1, 1], 0, 1],
        [[1, 1], [0, 0], 1, 0],
        [[1, 1], [1, 1], 1, 1],
    ],
)
def test_unsharp_mask_limits(blur_limit, sigma, result_blur, result_sigma):
    img = np.zeros([100, 100, 3], dtype=np.uint8)

    aug = A.Compose([A.UnsharpMask(blur_limit=blur_limit, sigma_limit=sigma, p=1)])

    res = aug(image=img)["image"]
    assert np.allclose(res, F.unsharp_mask(img, result_blur, result_sigma))


@pytest.mark.parametrize(["val_uint8"], [[0], [1], [128], [255]])
def test_unsharp_mask_float_uint8_diff_less_than_two(val_uint8):
    x_uint8 = np.zeros((5, 5)).astype(np.uint8)
    x_uint8[2, 2] = val_uint8

    x_float32 = np.zeros((5, 5)).astype(np.float32)
    x_float32[2, 2] = val_uint8 / 255.0

    unsharpmask = A.UnsharpMask(blur_limit=3, p=1)

    set_seed(0)
    usm_uint8 = unsharpmask(image=x_uint8)["image"]

    set_seed(0)
    usm_float32 = unsharpmask(image=x_float32)["image"]

    # Before comparison, rescale the usm_float32 to [0, 255]
    diff = np.abs(usm_uint8 - usm_float32 * 255)

    # The difference between the results of float32 and uint8 will be at most 2.
    assert np.all(diff <= 2.0)


@pytest.mark.parametrize(
    ["brightness", "contrast", "saturation", "hue"],
    [
        [1, 1, 1, 0],
        [0.123, 1, 1, 0],
        [1.321, 1, 1, 0],
        [1, 0.234, 1, 0],
        [1, 1.432, 1, 0],
        [1, 1, 0.345, 0],
        [1, 1, 1.543, 0],
        [1, 1, 1, 0.456],
        [1, 1, 1, -0.432],
    ],
)
def test_color_jitter_float_uint8_equal(brightness, contrast, saturation, hue):
    img = SQUARE_UINT8_IMAGE

    transform = A.Compose(
        [
            A.ColorJitter(
                brightness=[brightness, brightness],
                contrast=[contrast, contrast],
                saturation=[saturation, saturation],
                hue=[hue, hue],
                p=1,
            )
        ]
    )

    res1 = transform(image=img)["image"]
    res2 = (transform(image=img.astype(np.float32) / 255.0)["image"] * 255).astype(np.uint8)

    _max = np.abs(res1.astype(np.int16) - res2.astype(np.int16)).max()

    if hue != 0:
        assert _max <= 10, f"Max: {_max}"
    else:
        assert _max <= 2, f"Max: {_max}"


@pytest.mark.parametrize(["hue", "sat", "val"], [[13, 17, 23], [14, 18, 24], [131, 143, 151], [132, 144, 152]])
def test_hue_saturation_value_float_uint8_equal(hue, sat, val):
    img = SQUARE_UINT8_IMAGE

    for i in range(2):
        sign = 1 if i == 0 else -1
        for i in range(4):
            if i == 0:
                _hue = hue * sign
                _sat = 0
                _val = 0
            elif i == 1:
                _hue = 0
                _sat = sat * sign
                _val = 0
            elif i == 2:
                _hue = 0
                _sat = 0
                _val = val * sign
            else:
                _hue = hue * sign
                _sat = sat * sign
                _val = val * sign

            t1 = A.Compose(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=[_hue, _hue], sat_shift_limit=[_sat, _sat], val_shift_limit=[_val, _val], p=1
                    )
                ]
            )
            t2 = A.Compose(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=[_hue / 180 * 360, _hue / 180 * 360],
                        sat_shift_limit=[_sat / 255, _sat / 255],
                        val_shift_limit=[_val / 255, _val / 255],
                        p=1,
                    )
                ]
            )

            res1 = t1(image=img)["image"]
            res2 = (t2(image=img.astype(np.float32) / 255.0)["image"] * 255).astype(np.uint8)

            _max = np.abs(res1.astype(np.int32) - res2).max()
            assert _max <= 10, f"Max value: {_max}"


def test_perspective_keep_size():
    h, w = 100, 100
    img = np.zeros([h, w, 3], dtype=np.uint8)
    bboxes = []
    for _ in range(10):
        x1 = np.random.randint(0, w - 1)
        y1 = np.random.randint(0, h - 1)
        x2 = np.random.randint(x1 + 1, w)
        y2 = np.random.randint(y1 + 1, h)
        bboxes.append([x1, y1, x2, y2])
    keypoints = [(np.random.randint(0, w), np.random.randint(0, h), np.random.random()) for _ in range(10)]

    transform_1 = A.Compose(
        [A.Perspective(keep_size=True, p=1)],
        keypoint_params=A.KeypointParams("xys"),
        bbox_params=A.BboxParams("pascal_voc", label_fields=["labels"]),
    )
    transform_2 = A.Compose(
        [A.Perspective(keep_size=False, p=1), A.Resize(h, w, p=1)],
        keypoint_params=A.KeypointParams("xys"),
        bbox_params=A.BboxParams("pascal_voc", label_fields=["labels"]),
    )

    set_seed(0)
    res_1 = transform_1(image=img, bboxes=bboxes, keypoints=keypoints, labels=[0] * len(bboxes))
    set_seed(0)
    res_2 = transform_2(image=img, bboxes=bboxes, keypoints=keypoints, labels=[0] * len(bboxes))

    assert np.allclose(res_1["bboxes"], res_2["bboxes"], atol=0.2)
    assert np.allclose(res_1["keypoints"], res_2["keypoints"])


def test_longest_max_size_list():
    img = np.random.randint(0, 256, [50, 10], np.uint8)
    keypoints = [(9, 5, 0, 0)]

    aug = A.LongestMaxSize(max_size=[5, 10], p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["image"].shape in [(10, 2), (5, 1)]
    assert result["keypoints"] in [[(0.9, 0.5, 0, 0)], [(1.8, 1, 0, 0)]]


def test_smallest_max_size_list():
    img = np.random.randint(0, 256, [50, 10], np.uint8)
    keypoints = [(9, 5, 0, 0)]

    aug = A.SmallestMaxSize(max_size=[50, 100], p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["image"].shape in [(250, 50), (500, 100)]
    assert result["keypoints"] in [[(45, 25, 0, 0)], [(90, 50, 0, 0)]]


@pytest.mark.parametrize(
    ["img_weight", "template_weight", "template_transform", "image_size", "template_size"],
    [
        (0.5, 0.5, A.RandomSizedCrop((50, 200), size=(513, 450), p=1.), (513, 450), (224, 224)),
        (0.3, 0.5, A.RandomResizedCrop(size=(513, 450), p=1.), (513, 450), (224, 224)),
        (1.0, 0.5, A.CenterCrop(500, 450,  p=1.), (500, 450, 3), (512, 512, 3)),
        (0.5, 0.8, A.Resize(513, 450, p=1.), (513, 450), (512, 512)),
        (0.5, 0.2, A.NoOp(), (224, 224), (224, 224)),
        (0.5, 0.9, A.NoOp(), (512, 512, 3), (512, 512, 3)),
        (0.5, 0.5, None, (512, 512), (512, 512)),
        (0.8, 0.7, None, (512, 512, 3), (512, 512, 3)),
        (
            0.5,
            0.5,
            A.Compose([A.Blur(p=1.), A.RandomSizedCrop((50, 200), size=(512, 512), p=1.), A.HorizontalFlip(p=1.)]),
            (512, 512),
            (512, 512),
        ),
    ],
)
def test_template_transform(img_weight, template_weight, template_transform, image_size, template_size):
    img = np.random.randint(0, 256, image_size, np.uint8)
    template = np.random.randint(0, 256, template_size, np.uint8)

    aug = A.TemplateTransform(template, img_weight, template_weight, template_transform)
    result = aug(image=img)["image"]

    assert result.shape == img.shape

    params = aug.get_params_dependent_on_targets({"image": img})
    template = params["template"]
    assert template.shape == img.shape
    assert template.dtype == img.dtype


def test_template_transform_incorrect_size(template):
    image = np.random.randint(0, 256, (512, 512, 3), np.uint8)
    with pytest.raises(ValueError) as exc_info:
        transform = A.TemplateTransform(template, p=1.0)
        transform(image=image)

    message = f"Image and template must be the same size, got {image.shape[:2]} and {template.shape[:2]}"
    assert str(exc_info.value) == message


@pytest.mark.parametrize(["img_channels", "template_channels"], [(1, 3), (6, 3)])
def test_template_transform_incorrect_channels(img_channels, template_channels):
    img = np.random.randint(0, 255, [100, 100, img_channels], np.uint8)
    template = np.random.randint(0, 255, [100, 100, template_channels], np.uint8)

    with pytest.raises(ValueError) as exc_info:
        transform = A.TemplateTransform(template, p=1.)
        transform(image=img)

    message = (
        "Template must be a single channel or has the same number of channels "
        "as input image ({}), got {}".format(img_channels, template.shape[-1])
    )
    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ["params"],
    [
        [{"scale": (0.5, 1.0)}],
        [{"scale": (0.5, 1.0), "keep_ratio": False}],
        [{"scale": (0.5, 1.0), "keep_ratio": True}],
    ],
)
def test_affine_scale_ratio(params):
    set_seed(0)
    aug = A.Affine(**params, p=1.0)
    image = SQUARE_UINT8_IMAGE
    target = {"image": image}
    apply_params = aug.get_params_dependent_on_targets(target)

    if "keep_ratio" not in params:
        # default(keep_ratio=False)
        assert apply_params["scale"]["x"] != apply_params["scale"]["y"]
    elif not params["keep_ratio"]:
        # keep_ratio=False
        assert apply_params["scale"]["x"] != apply_params["scale"]["y"]
    else:
        # keep_ratio=True
        assert apply_params["scale"]["x"] == apply_params["scale"]["y"]


@pytest.mark.parametrize(
    ["params"],
    [
        [{"scale": {"x": (0.5, 1.0), "y": (1.0, 1.5)}, "keep_ratio": True}],
        [{"scale": {"x": 0.5, "y": 1.0}, "keep_ratio": True}],
    ],
)
def test_affine_incorrect_scale_range(params):
    with pytest.raises(ValueError):
        A.Affine(**params)


@pytest.mark.parametrize(
    ["angle", "targets", "expected"],
    [
        [
            -10,
            {
                "bboxes": [
                    [0, 0, 5, 5, 0],
                    [195, 0, 200, 5, 0],
                    [195, 95, 200, 100, 0],
                    [0, 95, 5, 99, 0],
                ],
                "keypoints": [
                    [0, 0, 0, 0],
                    [199, 0, 10, 10],
                    [199, 99, 20, 20],
                    [0, 99, 30, 30],
                ],
            },
            {
                "bboxes": [
                    [(16.036253471129026, 0.7268824985344293, 21.42442059056688, 5.049479254799872, 0), (194.61183288056216, 25.996579994841458, 200.0, 30.319176751106898, 0), (179.33014645626594, 95.67740324373456, 184.71831357570377, 100.0, 0), (0.8521337495555823, 70.54534260014618, 6.078767680466058, 74.1330081974473, 0)]
                ],
                "keypoints": [
                    [(16.84391941376591, 0.7268824985344293, 147.04220486917677, 0.0), (199.0, 26.514932763996473, 157.04220486917674, 9.30232558139535), (183.15608058623408, 99.0, 167.04220486917674, 18.6046511627907), (0.8521337495555823, 73.48506723600353, 177.04220486917674, 27.906976744186046)]
                ],
            },
        ],
        [
            10,
            {
                "bboxes": [
                    [0, 0, 5, 5, 0],
                    [195, 0, 200, 5, 0],
                    [195, 95, 200, 100, 0],
                    [0, 95, 5, 99, 0],
                ],
                "keypoints": [
                    [0, 0, 0, 0],
                    [199, 0, 10, 10],
                    [199, 99, 20, 20],
                    [0, 99, 30, 30],
                ],
            },
            {
                "bboxes": [
                    [(0.8521337495555819, 25.866991802552704, 6.240300868993435, 30.18958855881814, 0), (179.4916796447933, 0.5972943062456757, 184.87984676423113, 4.919891062511116, 0), (194.61183288056216, 70.41575440785743, 200.0, 74.73835116412288, 0), (16.1977866596564, 95.68545190416447, 21.424420590566875, 99.27311750146558, 0)]
                ],
                "keypoints": [
                    [(0.852133749555582, 26.514932763996473, 212.95779513082323, 0.0), (183.15608058623408, 0.7268824985344295, 222.95779513082323, 9.30232558139535), (199.0, 73.48506723600353, 232.9577951308232, 18.6046511627907), (16.84391941376591, 99.0, 242.9577951308232, 27.906976744186046)]
                ],
            },
        ],
    ],
)
def test_safe_rotate(angle: float, targets: dict, expected: dict):
    image = np.empty([100, 200, 3], dtype=np.uint8)
    t = A.Compose(
        [
            A.SafeRotate(limit=(angle, angle), border_mode=0, value=0, p=1),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.0),
        keypoint_params=A.KeypointParams("xyas"),
        p=1,
    )
    res = t(image=image, **targets)

    for key, value in expected.items():
        assert np.allclose(np.array(value), np.array(res[key])), key


@pytest.mark.parametrize(
    "aug_cls",
    [
        (lambda rotate: A.Affine(rotate=rotate, p=1, mode=cv2.BORDER_CONSTANT, cval=0)),
        (
            lambda rotate: A.ShiftScaleRotate(
                shift_limit=(0, 0),
                scale_limit=(0, 0),
                rotate_limit=rotate,
                p=1,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            )
        ),
    ],
)
@pytest.mark.parametrize(
    "img",
    [
        SQUARE_UINT8_IMAGE,
        np.random.randint(0, 256, [25, 100, 3], np.uint8),
        np.random.randint(0, 256, [100, 25, 3], np.uint8),
    ],
)
@pytest.mark.parametrize("angle", list(range(-360, 360, 15)))
def test_rotate_equal(img, aug_cls, angle):
    set_seed(0)

    height, width = img.shape[:2]
    kp = [[random.randint(0, width - 1), random.randint(0, height - 1), random.randint(0, 360)] for _ in range(50)]
    kp += [
        [round(width * 0.2), int(height * 0.3), 90],
        [int(width * 0.2), int(height * 0.3), 90],
        [int(width * 0.2), int(height * 0.3), 90],
        [int(width * 0.2), int(height * 0.3), 90],
        [0, 0, 0],
        [width - 1, height - 1, 0],
    ]
    keypoint_params = A.KeypointParams("xya", remove_invisible=False)

    a = A.Compose([aug_cls(rotate=(angle, angle))], keypoint_params=keypoint_params)
    b = A.Compose(
        [A.Rotate((angle, angle), border_mode=cv2.BORDER_CONSTANT, value=0, p=1)], keypoint_params=keypoint_params
    )

    res_a = a(image=img, keypoints=kp)
    res_b = b(image=img, keypoints=kp)
    assert np.allclose(res_a["image"], res_b["image"])
    res_a = np.array(res_a["keypoints"])
    res_b = np.array(res_b["keypoints"])
    diff = np.round(np.abs(res_a - res_b))
    assert diff[:, :2].max() <= 2
    assert (diff[:, -1] % 360).max() <= 1

@pytest.mark.parametrize("seed", list(range(10)))
def test_motion_blur_allow_shifted(seed):
    set_seed(seed)

    transform = A.MotionBlur(allow_shifted=False)
    kernel = transform.get_params()["kernel"]

    center = kernel.shape[0] / 2 - 0.5

    def check_center(vector):
        start = None
        end = None

        for i, v in enumerate(vector):
            if start is None and v != 0:
                start = i
            elif start is not None and v == 0:
                end = i
                break
        if end is None:
            end = len(vector)

        assert (end + start - 1) / 2 == center

    check_center(kernel.sum(axis=0))
    check_center(kernel.sum(axis=1))


@pytest.mark.parametrize(
    "augmentation",
    [
        A.RandomGravel(),
        A.RandomSnow(),
        A.RandomRain(),
        A.RandomFog(),
        A.RandomSunFlare(),
        A.RandomShadow(),
        A.Spatter(),
        A.ChromaticAberration(),
    ],
)
@pytest.mark.parametrize("img_channels", [1, 6])
def test_non_rgb_transform_warning(augmentation, img_channels):
    img = np.random.randint(0, 255, (100, 100, img_channels), dtype=np.uint8)

    with pytest.raises(ValueError) as exc_info:
        augmentation(image=img, force_apply=True)

    message = "This transformation expects 3-channel images"
    assert str(exc_info.value).startswith(message)

@pytest.mark.parametrize("height, width", [(100, 200), (200, 100)])
@pytest.mark.parametrize("scale", [(0.08, 1.0), (0.5, 1.0)])
@pytest.mark.parametrize("ratio", [(0.75, 1.33), (1.0, 1.0)])
def test_random_crop_interfaces_vs_torchvision(height, width, scale, ratio):
    # NOTE: below will fail when height, width is no longer expected as first two positional arguments
    transform_albu = A.RandomResizedCrop(height, width, scale=scale, ratio=ratio, p=1)
    transform_albu_new = A.RandomResizedCrop(size=(height, width), scale=scale, ratio=ratio, p=1)

    image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    transformed_image_albu = transform_albu(image=image)['image']
    transformed_image_albu_new = transform_albu_new(image=image)['image']

    # PyTorch equivalent operation
    transform_pt = torch_transforms.RandomResizedCrop(size=(height, width), scale=scale, ratio=ratio)
    image_pil = torch_transforms.functional.to_pil_image(image)
    transformed_image_pt = transform_pt(image_pil)

    transformed_image_pt_np = np.array(transformed_image_pt)
    assert transformed_image_albu.shape == transformed_image_pt_np.shape
    assert transformed_image_albu_new.shape == transformed_image_pt_np.shape

    # NOTE: below will fail when height, width is no longer expected as second and third positional arguments
    transform_albu = A.RandomSizedCrop((128, 224), height, width, p=1.0)
    transform_albu_new = A.RandomSizedCrop(min_max_height=(128, 224), size=(height, width), p=1.0)
    transformed_image_albu = transform_albu(image=image)['image']
    transformed_image_albu_new = transform_albu_new(image=image)['image']
    assert transformed_image_albu.shape == transformed_image_pt_np.shape
    assert transformed_image_albu_new.shape == transformed_image_pt_np.shape

    # NOTE: below will fail when height, width is no longer expected as first two positional arguments
    transform_albu = A.RandomResizedCrop(height, width, scale=scale, ratio=ratio, p=1)
    transform_albu_height_is_size = A.RandomResizedCrop(size=height, width=width, scale=scale, ratio=ratio, p=1)

    image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    transformed_image_albu = transform_albu(image=image)['image']
    transform_albu_height_is_size = transform_albu_new(image=image)['image']
    assert transformed_image_albu.shape == transformed_image_pt_np.shape
    assert transform_albu_height_is_size.shape == transformed_image_pt_np.shape

@pytest.mark.parametrize("num_shadows_limit, num_shadows_lower, num_shadows_upper, expected_warning", [
    ((1, 1), None, None, None),
    ((1, 2), None, None, None),
    ((2, 3), None, None, None),
    ((1, 2), 1, None, DeprecationWarning),
    ((1, 2), None, 2, DeprecationWarning),
    ((1, 2), 1, 2, DeprecationWarning),
    ((2, 1), None, None, ValueError),
])
def test_deprecation_warnings_random_shadow(
    num_shadows_limit: Tuple[int, int],
    num_shadows_lower: Optional[int],
    num_shadows_upper: Optional[int],
    expected_warning: Optional[Type[Warning]],
) -> None:
    """
    Test deprecation warnings for RandomShadow
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Change the filter to capture all warnings
        if expected_warning == ValueError:
            with pytest.raises(ValueError):
                A.RandomShadow(num_shadows_limit=num_shadows_limit, num_shadows_lower=num_shadows_lower,
                               num_shadows_upper=num_shadows_upper, p=1)
        elif expected_warning is DeprecationWarning:
            A.RandomShadow(num_shadows_limit=num_shadows_limit, num_shadows_lower=num_shadows_lower,
                           num_shadows_upper=num_shadows_upper, p=1)
            for warning in w:
                print(f"Warning captured: {warning.category.__name__}, Message: '{warning.message}'")

                if warning.category is DeprecationWarning:
                    print(f"Deprecation Warning: {warning.message}")
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w), \
                "No DeprecationWarning found"
        else:
            assert not w, "Unexpected warnings raised"

@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize("grid", [
    (3, 3), (4, 4), (5, 7)
])
def test_grid_shuffle(image, grid):
    """
    As we reshuffle the grid, the mean and sum of the image and mask should remain the same,
    while the reshuffled image and mask should not be equal to the original image and mask.
    """
    set_seed(0)

    mask = image.copy()

    aug = A.Compose([A.RandomGridShuffle(grid=grid, p=1)])

    res = aug(image=image, mask=mask)
    assert res["image"].shape == image.shape
    assert res["mask"].shape == mask.shape

    assert not np.array_equal(res["image"], image)
    assert not np.array_equal(res["mask"], mask)

    np.testing.assert_allclose(res["image"].sum(axis=(0, 1)), image.sum(axis=(0, 1)), atol=0.04)
    np.testing.assert_allclose(res["mask"].sum(axis=(0, 1)), mask.sum(axis=(0, 1)), atol=0.03)

@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize("crop_left, crop_right, crop_top, crop_bottom", [
    (0, 0, 0, 0),
    (0, 1, 0, 1),
    (1, 0, 1, 0),
    (0.5, 0.5, 0.5, 0.5),
    ( 0.1, 0.1, 0.1, 0.1 ),
                                                                          ( 0.3, 0.3, 0.3, 0.3 )])
def test_random_crop_from_borders(image, bboxes, keypoints, crop_left, crop_right, crop_top, crop_bottom):
    set_seed(0)
    aug = A.Compose([A.RandomCropFromBorders(crop_left=crop_left,
                                             crop_right=crop_right,
                                             crop_top=crop_top,
                                             crop_bottom=crop_bottom,
                                             p=1)],
                    bbox_params=A.BboxParams("pascal_voc"),
                    keypoint_params=A.KeypointParams("xy"))

    assert aug(image=image, mask=image, bboxes=bboxes, keypoints=keypoints)

@pytest.mark.parametrize("params, expected", [
    # Test default initialization values
    ({}, {"quality_range": (99, 100), "compression_type": ImageCompressionType.JPEG}),
    # Test custom quality range and compression type
    ({"quality_range": (10, 90), "compression_type": ImageCompressionType.WEBP},
     {"quality_range": (10, 90), "compression_type": ImageCompressionType.WEBP}),
    # Deprecated quality values handling
    ({"quality_lower": 75}, {"quality_range": (75, 100)}),
])
def test_image_compression_initialization(params, expected):
    img_comp = ImageCompression(**params)
    for key, value in expected.items():
        assert getattr(img_comp, key) == value, f"Failed on {key} with value {value}"

@pytest.mark.parametrize("params", [
    ({"quality_range": (101, 105)}),  # Invalid quality range
    ({"quality_range": (0, 0)}),  # Invalid range for JPEG
    ({"compression_type": "unknown"})  # Invalid compression type
])
def test_image_compression_invalid_input(params):
    with pytest.raises(Exception):
        ImageCompression(**params)


@pytest.mark.parametrize("params, expected", [
    # Default values
    ({}, {"num_holes_range": (1, 1), "hole_height_range": (8, 8), "hole_width_range": (8, 8)}),
    # Boundary values
    ({"num_holes_range": (2, 3)}, {"num_holes_range": (2, 3)}),
    ({"hole_height_range": (0.1, 0.1)}, {"hole_height_range": (0.1, 0.1)}),
    ({"hole_width_range": (0.1, 0.1)}, {"hole_width_range": (0.1, 0.1)}),
    # Random fill value
    ({"fill_value": 'random'}, {"fill_value": 'random'}),
    ({"fill_value": (255, 255, 255)}, {"fill_value": (255, 255, 255)}),
    # Deprecated values handling
    ({"min_holes": 1, "max_holes": 5}, {"num_holes_range": (1, 5)}),
    ({"min_height": 2, "max_height": 6}, {"hole_height_range": (2, 6)}),
    ({"min_width": 3, "max_width": 7}, {"hole_width_range": (3, 7)}),
])
def test_coarse_dropout_functionality(params, expected):
    aug = A.CoarseDropout(**params, p=1)
    aug_dict = aug.to_dict()["transform"]
    for key, value in expected.items():
        assert aug_dict[key] == value, f"Failed on {key} with value {value}"


@pytest.mark.parametrize("params", [
    ({"num_holes_range": (5, 1)}),  # Invalid range
    ({"num_holes_range": (0, 3)}),  # Invalid range
    ({"hole_height_range": (2.1, 3)}),  # Invalid type
    ({"hole_height_range": ('a', 'b')}),  # Invalid type
])
def test_coarse_dropout_invalid_input(params):
    with pytest.raises(Exception):
        aug = A.CoarseDropout(**params, p=1)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.TemplateTransform: {
                "templates": clip(SQUARE_UINT8_IMAGE + 2, np.uint8),
            },
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.Superpixels: {"p_replace": (1, 1),
                             "n_segments": (10, 10),
                             "max_size": 10
                            },
            A.ZoomBlur: {"max_factor": (1.05, 3)},
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.MaskDropout,
            A.MixUp,
            A.NoOp,
            A.Lambda,
            A.ToRGB,
            A.RandomRotate90,
            A.FancyPCA
        },
    ),
)
def test_change_image(augmentation_cls, params):
    """Checks whether transform performs changes to the image."""
    aug = A.Compose([augmentation_cls(p=1, **params)])
    image = SQUARE_UINT8_IMAGE
    assert not np.array_equal(aug(image=image)["image"], image)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.Superpixels: {
                "p_replace": (1, 1),
                "n_segments": (10, 10),
                "max_size": 10
            },
            A.FancyPCA: {"alpha": 1}
        },
        except_augmentations={
            A.Crop,
            A.CenterCrop,
            A.CropNonEmptyMaskIfExists,
            A.RandomCrop,
            A.RandomResizedCrop,
            A.RandomSizedCrop,
            A.CropAndPad,
            A.Resize,
            A.TemplateTransform,
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.MaskDropout,
            A.MixUp,
            A.NoOp,
            A.Lambda,
            A.ToRGB,
            A.ChannelDropout,
            A.LongestMaxSize,
            A.PadIfNeeded,
            A.RandomCropFromBorders,
            A.SmallestMaxSize,
            A.RandomScale,
            A.ChannelShuffle,
            A.ChromaticAberration,
            A.RandomRotate90,
            A.FancyPCA,
            A.PlanckianJitter
        },
    ),
)
def test_selective_channel(augmentation_cls: BasicTransform, params: Dict[str, Any]) -> None:
    set_seed(3)

    image = SQUARE_MULTI_UINT8_IMAGE
    channels = [3, 2, 4]

    aug = A.Compose(
        [A.SelectiveChannelTransform(transforms=[augmentation_cls(**params, p=1)], channels=channels, p=1)],
    )

    transformed_image = aug(image=image)["image"]

    for channel in range(image.shape[-1]):
        if channel in channels:
            assert not np.array_equal(image[..., channel], transformed_image[..., channel])
        else:
            assert np.array_equal(image[..., channel], transformed_image[..., channel])


@pytest.mark.parametrize("params, expected", [
    # Default values
    ({}, {"scale_range": (0.25, 0.25), "interpolation_pair": {"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_NEAREST}}),
    # Boundary values
    ({"scale_range": (0.1, 0.9)}, {"scale_range": (0.1, 0.9)}),
    ({"interpolation_pair": {"downscale": cv2.INTER_LINEAR, "upscale": cv2.INTER_CUBIC}}, {"interpolation_pair": {"downscale": cv2.INTER_LINEAR, "upscale": cv2.INTER_CUBIC}}),
    # Deprecated values handling
    ({"scale_min": 0.1, "scale_max": 0.9}, {"scale_range": (0.1, 0.9)}),
    ({"interpolation": cv2.INTER_AREA}, {"interpolation_pair": {"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_AREA}}),
])
def test_downscale_functionality(params, expected):
    aug = A.Downscale(**params, p=1)
    aug_dict = aug.get_transform_init_args()
    for key, value in expected.items():
        assert aug_dict[key] == value, f"Failed on {key} with value {value}"

@pytest.mark.parametrize("params", [
    ({"scale_range": (0.9, 0.1)}),  # Invalid range, max < min
    ({"scale_range": (1.1, 1.2)}),  # Values outside valid scale range (0, 1)
    ({"interpolation_pair": {"downscale": 9999, "upscale": 9999}}),  # Invalid interpolation method
])
def test_downscale_invalid_input(params):
    with pytest.raises(Exception):
        A.Downscale(**params, p=1)


@pytest.mark.parametrize("params, expected", [
    # Default values
    ({}, {"min_height": 1024, "min_width": 1024, "position": A.PadIfNeeded.PositionType.CENTER, "border_mode": cv2.BORDER_REFLECT_101}),
    # Boundary values
    ({"min_height": 800, "min_width": 800}, {"min_height": 800, "min_width": 800}),
    ({"pad_height_divisor": 10, "min_height": None, "pad_width_divisor": 10, "min_width": None},
     {"pad_height_divisor": 10, "min_height": None, "pad_width_divisor": 10, "min_width": None}),
    ({"position": "top_left"}, {"position": A.PadIfNeeded.PositionType.TOP_LEFT}),
    # Value handling when border_mode is BORDER_CONSTANT
    ({"border_mode": cv2.BORDER_CONSTANT, "value": 255}, {"border_mode": cv2.BORDER_CONSTANT, "value": 255}),
    ({"border_mode": cv2.BORDER_REFLECT_101, "value": 255}, {"border_mode": cv2.BORDER_CONSTANT, "value": 255}),
    ({"border_mode": cv2.BORDER_CONSTANT, "value": [0, 0, 0]}, {"border_mode": cv2.BORDER_CONSTANT, "value": [0, 0, 0]}),
    # Mask value handling
    ({"border_mode": cv2.BORDER_CONSTANT, "value": [0, 0, 0], "mask_value": 128}, {"border_mode": cv2.BORDER_CONSTANT, "mask_value": 128, "value": [0, 0, 0]}),
])
def test_pad_if_needed_functionality(params, expected):
    # Setup the augmentation with the provided parameters
    aug = A.PadIfNeeded(**params, p=1)
    # Get the initialization arguments to check against expected
    aug_dict = {key: getattr(aug, key) for key in expected.keys()}

    # Assert each expected key/value pair
    for key, value in expected.items():
        assert aug_dict[key] == value, f"Failed on {key} with value {value}"

@pytest.mark.parametrize("params, expected", [
    # Test default initialization values
    ({}, {"slant_range": (-10, 10)}),
    ({"slant_range": (-7, 4)},
     {"slant_range": (-7, 4)}),
    ({"slant_lower": 2}, {"slant_range": (2, 10)}),
    ({"slant_upper": 2}, {"slant_range": (-10, 2)}),
])
def test_random_rain_initialization(params, expected):
    img_rain = RandomRain(**params)
    for key, value in expected.items():
        assert getattr(img_rain, key) == value, f"Failed on {key} with value {value}"

@pytest.mark.parametrize("params", [
    ({"slant_range": (12, 8)}),  # Invalid slant range -> decreasing
    ({"slant_range": (-8, 62)}),  # invalid slant range -> 62 out of upper bound
])
def test_random_rain_invalid_input(params):
    with pytest.raises(Exception):
        RandomRain(**params)

@pytest.mark.parametrize("params, expected", [
    # Test default initialization values
    ({}, {"snow_point_range": (0.1, 0.3)}),
    # Test snow point range
    ({"snow_point_range": (0.2, 0.6)},
     {"snow_point_range": (0.2, 0.6)}),
    # Deprecated quality values handling
    ({"snow_point_lower": 0.15}, {"snow_point_range": (0.15, 0.3)}),
    ({"snow_point_upper": 0.4}, {"snow_point_range": (0.1, 0.4)}),
])
def test_random_snow_initialization(params, expected):
    img_comp = RandomSnow(**params)
    for key, value in expected.items():
        assert getattr(img_comp, key) == value, f"Failed on {key} with value {value}"

@pytest.mark.parametrize("params", [
    ({"snow_point_range": (1.2, 1.5)}),  # Invalid quality range -> upper bound
    ({"snow_point_range": (0.9, 0.7)}),  # Invalid range  -> decreasing
])
def test_random_snow_invalid_input(params):
    with pytest.raises(Exception):
        a = RandomSnow(**params)
        print(a.snow_point_range)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.TemplateTransform: {
                "templates": np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
            },
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
        },
        except_augmentations={
        #     A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.MaskDropout,
            A.MixUp
        },
    ),
)
def test_dual_transforms_methods(augmentation_cls, params):
    """Checks whether transformations based on DualTransform dont has abstract methods."""
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=4, size=(100, 100), dtype=np.uint8) * 64

    arg = {
        "masks": mask,
        "masks": [mask],
        "bboxes": [[0, 0, 0.1, 0.1, 1]],
        "keypoints": [(0, 0, 0, 0), (1, 1, 0, 0)],
    }

    for target in aug.targets:
        if target in arg:
            kwarg = {target: arg[target]}
            try:
                _res = aug(image=image, **kwarg)
            except Exception as e:
                if isinstance(e, NotImplementedError):
                    raise NotImplementedError(f"{target} error at: {augmentation_cls},  {e}")
                raise e


@pytest.mark.parametrize("px", [
    10,
    (10, 20),
    (-10, 20, -30, 40),
    ((10, 20), (20, 30), (30, 40), (40, 50)),
    ([1, 2, 3, 4],  [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]),
    None
])
@pytest.mark.parametrize("percent", [
    0.1,
    (0.1, 0.2),
    (0.1, 0.2, 0.3, 0.4),
    ((-0.1, -0.2), (-0.2, -0.3), (0.3, 0.4), (0.4, 0.5)),
    ([0.1, 0.2, 0.3, 0.4],  [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]),
    None
])
@pytest.mark.parametrize("pad_cval", [
    0,
    (0, 255),
    [0, 255]
])
@pytest.mark.parametrize("keep_size", [
    True,
    False
])
@pytest.mark.parametrize("sample_independently", [
    True,
    False
])
@pytest.mark.parametrize("image", IMAGES)
def test_crop_and_pad(px, percent, pad_cval, keep_size, sample_independently, image):
    pad_cval_mask = 255 if isinstance(pad_cval, list) else pad_cval
    interpolation = cv2.INTER_LINEAR
    pad_mode = cv2.BORDER_CONSTANT
    if (px is None) ==  (percent is None):
        # Skip the test case where both px and percent are None or both are not None
        return

    transform = A.Compose([A.CropAndPad(
        px=px,
        percent=percent,
        pad_mode=pad_mode,
        pad_cval=pad_cval,
        pad_cval_mask=pad_cval_mask,
        keep_size=keep_size,
        sample_independently=sample_independently,
        interpolation=interpolation,
        p=1
    )])

    transformed_image = transform(image=image)["image"]

    if keep_size:
        assert transformed_image.shape == image.shape

    assert transformed_image is not None
    assert transformed_image.shape[0] > 0
    assert transformed_image.shape[1] > 0
    assert transformed_image.shape[2] == image.shape[2]


@pytest.mark.parametrize("percent, expected_shape", [
    (0.1, (12, 12, 3)),  # Padding 10% of image size on each side
    (-0.1, (8, 8, 3)),  # Cropping 10% of image size from each side
    ((0.1, 0.2, 0.3, 0.4), (14, 16, 3)),  # Padding: top=10%, right=20%, bottom=30%, left=40%
    ((-0.1, -0.2, -0.3, -0.4), (6, 4, 3)),  # Cropping: top=10%, right=20%, bottom=30%, left=40%
])
def test_crop_and_pad_percent(percent, expected_shape):
    transform = A.Compose([A.CropAndPad(px=None, percent=percent, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False)])

    image = np.ones((10, 10, 3), dtype=np.uint8)

    transformed_image = transform(image=image)["image"]

    assert transformed_image.shape == expected_shape
    if percent is not None and all(p >= 0 for p in np.array(percent).flatten()):
        assert transformed_image.sum() == image.sum()

@pytest.mark.parametrize("px, expected_shape", [
    (2, (14, 14, 3)),  # Padding 2 pixels on each side
    (-2, (6, 6, 3)),  # Cropping 2 pixels from each side
    ((1, 2, 3, 4), (14, 16, 3)),  # Padding: top=1, right=2, bottom=3, left=4
    ((-1, -2, -3, -4), (6, 4, 3)),  # Cropping: top=1, right=2, bottom=3, left=4
])
def test_crop_and_pad_px_pixel_values(px, expected_shape):
    transform = A.Compose([A.CropAndPad(px=px, percent=None, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, keep_size=False)])

    image = np.ones((10, 10, 3), dtype=np.uint8) * 255

    transformed_image = transform(image=image)["image"]

    if isinstance(px, int):
        px = [px] * 4  # Convert to list of 4 elements
    if isinstance(px, tuple) and len(px) == 2:
        px = [px[0], px[1], px[0], px[1]]  # Convert to 4 elements for padding

    if px is not None:
        if all(p >= 0 for p in px):  # Padding
            pad_top, pad_right, pad_bottom, pad_left = px
            central_region = transformed_image[pad_top:pad_top + image.shape[0], pad_left:pad_left + image.shape[1], :]
            assert np.all(central_region == 255)
        elif all(p <= 0 for p in px):  # Cropping
            crop_top, crop_right, crop_bottom, crop_left = [-p for p in px]
            cropped_region = image[crop_top:image.shape[0] - crop_bottom, crop_left:image.shape[1] - crop_right, :]
            assert np.all(transformed_image == cropped_region)


@pytest.mark.parametrize("params, expected", [
    # Test default initialization values
    ({}, {"fog_coef_range": (0.3, 1)}),
    # Test fog coefficient range
    ({"fog_coef_range": (0.4, 0.7)}, {"fog_coef_range": (0.4, 0.7)}),
    # Deprecated fog coefficient values handling
    ({"fog_coef_lower": 0.2}, {"fog_coef_range": (0.2, 1)}),
    ({"fog_coef_upper": 0.6}, {"fog_coef_range": (0.3, 0.6)}),
])
def test_random_fog_initialization(params, expected):
    img_fog = A.RandomFog(**params)
    for key, value in expected.items():
        assert getattr(img_fog, key) == value, f"Failed on {key} with value {value}"

@pytest.mark.parametrize("params", [
    ({"fog_coef_range": (1.2, 1.5)}),  # Invalid fog coefficient range -> upper bound
    ({"fog_coef_range": (0.9, 0.7)}),  # Invalid range  -> decreasing
])
def test_random_fog_invalid_input(params):
    with pytest.raises(Exception):
        A.RandomFog(**params)


@pytest.mark.parametrize("image", IMAGES + [np.full((10, 10), 128, dtype=np.uint8)])
@pytest.mark.parametrize("mean", (0, 10, -10))
def test_gauss_noise(mean, image):
    set_seed(42)
    aug = A.GaussNoise(p=1, noise_scale_factor=1.0, mean=mean)

    apply_params = aug.get_params_dependent_on_targets(params = {"image":image })

    assert np.abs(mean - apply_params["gauss"].mean()) < 0.5
    result = A.Compose([aug])(image=image)

    assert not (result["image"] >= image).all()
