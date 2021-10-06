import random
from typing import Type, Dict, Tuple

import cv2
import numpy as np
import pytest

import albumentations as A

from .utils import get_image_only_transforms, get_dual_transforms, get_transforms


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
            A.TemplateTransform: {
                "templates": np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
            },
        },
        except_augmentations={A.FromFloat, A.Normalize, A.ToFloat},
    ),
)
def test_image_only_augmentations(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image, mask=mask)
    assert data["image"].dtype == np.uint8
    assert data["mask"].dtype == np.uint8
    assert np.array_equal(data["mask"], mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
            A.MedianBlur: {"blur_limit": (3, 5)},
            A.TemplateTransform: {
                "templates": np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32),
            },
        },
        except_augmentations={
            A.CLAHE,
            A.Equalize,
            A.FancyPCA,
            A.FromFloat,
            A.ISONoise,
            A.Posterize,
            A.RandomToneCurve,
        },
    ),
)
def test_image_only_augmentations_with_float_values(augmentation_cls, params, float_image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=float_image, mask=mask)
    assert data["image"].dtype == np.float32
    assert data["mask"].dtype == np.uint8
    assert np.array_equal(data["mask"], mask)


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
        },
        except_augmentations={A.RandomCropNearBBox, A.RandomSizedBBoxSafeCrop},
    ),
)
def test_dual_augmentations(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image, mask=mask)
    assert data["image"].dtype == np.uint8
    assert data["mask"].dtype == np.uint8


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
        },
        except_augmentations={A.RandomCropNearBBox, A.RandomSizedBBoxSafeCrop},
    ),
)
def test_dual_augmentations_with_float_values(augmentation_cls, params, float_image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=float_image, mask=mask)
    assert data["image"].dtype == np.float32
    assert data["mask"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
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
        },
        except_augmentations={A.RandomCropNearBBox, A.RandomSizedBBoxSafeCrop},
    ),
)
def test_augmentations_wont_change_input(augmentation_cls, params, image, mask):
    image_copy = image.copy()
    mask_copy = mask.copy()
    aug = augmentation_cls(p=1, **params)
    aug(image=image, mask=mask)
    assert np.array_equal(image, image_copy)
    assert np.array_equal(mask, mask_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
            A.MedianBlur: {"blur_limit": (3, 5)},
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.TemplateTransform: {
                "templates": np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32),
            },
        },
        except_augmentations={
            A.CLAHE,
            A.Equalize,
            A.FancyPCA,
            A.ISONoise,
            A.Posterize,
            A.RandomToneCurve,
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.CropNonEmptyMaskIfExists,
            A.MaskDropout,
        },
    ),
)
def test_augmentations_wont_change_float_input(augmentation_cls, params, float_image):
    float_image_copy = float_image.copy()
    aug = augmentation_cls(p=1, **params)
    aug(image=float_image)
    assert np.array_equal(float_image, float_image_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [np.random.randint(0, 256, [100, 100], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [np.random.randint(0, 256, [100, 100], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.Normalize: {"mean": 0, "std": 1},
            A.TemplateTransform: {
                "templates": np.random.randint(low=0, high=256, size=(224, 224), dtype=np.uint8),
            },
        },
        except_augmentations={
            A.ChannelDropout,
            A.ChannelShuffle,
            A.FancyPCA,
            A.ISONoise,
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.CenterCrop,
            A.Crop,
            A.CropNonEmptyMaskIfExists,
            A.RandomCrop,
            A.RandomResizedCrop,
            A.RandomSizedCrop,
            A.CropAndPad,
            A.Resize,
            A.LongestMaxSize,
            A.SmallestMaxSize,
            A.PadIfNeeded,
            A.RGBShift,
            A.RandomFog,
            A.RandomRain,
            A.RandomScale,
            A.RandomShadow,
            A.RandomSnow,
            A.RandomSunFlare,
            A.ToSepia,
            A.PixelDistributionAdaptation,
        },
    ),
)
def test_augmentations_wont_change_shape_grayscale(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)

    # Test for grayscale image
    image = np.zeros((224, 224), dtype=np.uint8)
    mask = np.zeros((224, 224))
    result = aug(image=image, mask=mask)
    assert np.array_equal(image.shape, result["image"].shape)
    assert np.array_equal(mask.shape, result["mask"].shape)

    # Test for grayscale image with dummy dim
    image_1ch = np.zeros((224, 224, 1), dtype=np.uint8)
    mask_1ch = np.zeros((224, 224, 1))

    result = aug(image=image_1ch, mask=mask_1ch)
    assert np.array_equal(image_1ch.shape, result["image"].shape)
    assert np.array_equal(mask_1ch.shape, result["mask"].shape)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
            A.TemplateTransform: {
                "templates": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            },
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.CenterCrop,
            A.Crop,
            A.CropNonEmptyMaskIfExists,
            A.RandomCrop,
            A.RandomResizedCrop,
            A.RandomSizedCrop,
            A.CropAndPad,
            A.Resize,
            A.LongestMaxSize,
            A.SmallestMaxSize,
            A.PadIfNeeded,
            A.RandomScale,
        },
    ),
)
def test_augmentations_wont_change_shape_rgb(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)

    # Test for RGB image
    image_3ch = np.zeros((224, 224, 3), dtype=np.uint8)
    mask_3ch = np.zeros((224, 224, 3))

    result = aug(image=image_3ch, mask=mask_3ch)
    assert np.array_equal(image_3ch.shape, result["image"].shape)
    assert np.array_equal(mask_3ch.shape, result["mask"].shape)


@pytest.mark.parametrize(["augmentation_cls", "params"], [[A.RandomCropNearBBox, {"max_part_shift": 0.15}]])
def test_image_only_crop_around_bbox_augmentation(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)
    annotations = {"image": image, "cropping_bbox": [-59, 77, 177, 231]}
    data = aug(**annotations)
    assert data["image"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [
            A.PadIfNeeded,
            {"min_height": 514, "min_width": 514, "border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1},
        ],
        [A.Rotate, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [A.SafeRotate, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [A.ShiftScaleRotate, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [A.OpticalDistortion, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [A.ElasticTransform, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [A.GridDistortion, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [A.Affine, {"mode": cv2.BORDER_CONSTANT, "cval_mask": 1, "cval": 100}],
        [A.PiecewiseAffine, {"mode": "constant", "cval_mask": 1, "cval": 100}],
    ],
)
def test_mask_fill_value(augmentation_cls, params):
    random.seed(42)
    aug = augmentation_cls(p=1, **params)
    input = {"image": np.zeros((512, 512), dtype=np.uint8) + 100, "mask": np.ones((512, 512))}
    output = aug(**input)
    assert (output["image"] == 100).all()
    assert (output["mask"] == 1).all()


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 6], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 6], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.TemplateTransform: {
                "templates": np.random.randint(0, 256, (100, 100, 6), dtype=np.uint8),
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.ImageCompression,
            A.MaskDropout,
            A.Normalize,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomFog,
            A.RandomRain,
            A.RandomShadow,
            A.RandomSizedBBoxSafeCrop,
            A.RandomSnow,
            A.RandomSunFlare,
            A.ToFloat,
            A.ToGray,
            A.ToSepia,
            A.FancyPCA,
            A.PixelDistributionAdaptation,
        },
    ),
)
def test_multichannel_image_augmentations(augmentation_cls, params):
    image = np.zeros((100, 100, 6), dtype=np.uint8)
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image)
    assert data["image"].dtype == np.uint8
    assert data["image"].shape[2] == 6


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 6], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 6], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.Normalize: {"mean": 0, "std": 1},
            A.MedianBlur: {"blur_limit": (3, 5)},
            A.TemplateTransform: {
                "templates": np.random.uniform(0.0, 1.0, (100, 100, 1)).astype(np.float32),
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.ImageCompression,
            A.MaskDropout,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomFog,
            A.RandomRain,
            A.RandomShadow,
            A.RandomSizedBBoxSafeCrop,
            A.RandomSnow,
            A.RandomSunFlare,
            A.ToGray,
            A.ToSepia,
            A.Equalize,
            A.FancyPCA,
            A.Posterize,
            A.RandomToneCurve,
            A.PixelDistributionAdaptation,
        },
    ),
)
def test_float_multichannel_image_augmentations(augmentation_cls, params):
    image = np.zeros((100, 100, 6), dtype=np.float32)
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image)
    assert data["image"].dtype == np.float32
    assert data["image"].shape[2] == 6


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.TemplateTransform: {
                "templates": np.random.randint(0, 1, (100, 100), dtype=np.uint8),
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.ImageCompression,
            A.MaskDropout,
            A.Normalize,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomFog,
            A.RandomRain,
            A.RandomShadow,
            A.RandomSizedBBoxSafeCrop,
            A.RandomSnow,
            A.RandomSunFlare,
            A.ToFloat,
            A.ToGray,
            A.ToSepia,
            A.FancyPCA,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
        },
    ),
)
def test_multichannel_image_augmentations_diff_channels(augmentation_cls, params):
    for num_channels in range(3, 13):
        image = np.zeros((100, 100, num_channels), dtype=np.uint8)
        aug = augmentation_cls(p=1, **params)
        data = aug(image=image)
        assert data["image"].dtype == np.uint8
        assert data["image"].shape[2] == num_channels


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.Normalize: {"mean": 0, "std": 1},
            A.MedianBlur: {"blur_limit": (3, 5)},
            A.TemplateTransform: {
                "templates": np.random.uniform(0.0, 1.0, (100, 100, 1)).astype(np.float32),
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.ImageCompression,
            A.MaskDropout,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomFog,
            A.RandomRain,
            A.RandomShadow,
            A.RandomSizedBBoxSafeCrop,
            A.RandomSnow,
            A.RandomSunFlare,
            A.ToGray,
            A.ToSepia,
            A.Equalize,
            A.FancyPCA,
            A.Posterize,
            A.RandomToneCurve,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
        },
    ),
)
def test_float_multichannel_image_augmentations_diff_channels(augmentation_cls, params):
    for num_channels in range(3, 13):
        image = np.zeros((100, 100, num_channels), dtype=np.float32)
        aug = augmentation_cls(p=1, **params)
        data = aug(image=image)
        assert data["image"].dtype == np.float32
        assert data["image"].shape[2] == num_channels


@pytest.mark.parametrize(
    ["augmentation_cls", "params", "image_shape"],
    [
        [A.PadIfNeeded, {"min_height": 514, "min_width": 516}, (300, 200)],
        [A.PadIfNeeded, {"min_height": 514, "min_width": 516}, (512, 516)],
        [A.PadIfNeeded, {"min_height": 514, "min_width": 516}, (600, 600)],
        [
            A.PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 128, "pad_width_divisor": 128},
            (300, 200),
        ],
        [
            A.PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (72, 128),
        ],
        [
            A.PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (15, 15),
        ],
        [
            A.PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (144, 256),
        ],
        [
            A.PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (200, 300),
        ],
        [A.PadIfNeeded, {"min_height": 512, "min_width": None, "pad_width_divisor": 128}, (300, 200)],
        [A.PadIfNeeded, {"min_height": None, "min_width": 512, "pad_height_divisor": 128}, (300, 200)],
    ],
)
def test_pad_if_needed(augmentation_cls: Type[A.PadIfNeeded], params: Dict, image_shape: Tuple[int, int]):
    image = np.zeros(image_shape)
    pad = augmentation_cls(**params)

    image_padded = pad(image=image)["image"]

    if pad.min_width is not None:
        assert image_padded.shape[1] >= pad.min_width

    if pad.min_height is not None:
        assert image_padded.shape[0] >= pad.min_height

    if pad.pad_width_divisor is not None:
        assert image_padded.shape[1] % pad.pad_width_divisor == 0
        assert image_padded.shape[1] >= image.shape[1]
        assert image_padded.shape[1] - image.shape[1] <= pad.pad_width_divisor

    if pad.pad_height_divisor is not None:
        assert image_padded.shape[0] % pad.pad_height_divisor == 0
        assert image_padded.shape[0] >= image.shape[0]
        assert image_padded.shape[0] - image.shape[0] <= pad.pad_height_divisor


@pytest.mark.parametrize(
    ["params", "image_shape"],
    [
        [{"min_height": 10, "min_width": 12, "border_mode": 0, "value": 1, "position": "center"}, (5, 6)],
        [{"min_height": 10, "min_width": 12, "border_mode": 0, "value": 1, "position": "top_left"}, (5, 6)],
        [{"min_height": 10, "min_width": 12, "border_mode": 0, "value": 1, "position": "top_right"}, (5, 6)],
        [{"min_height": 10, "min_width": 12, "border_mode": 0, "value": 1, "position": "bottom_left"}, (5, 6)],
        [{"min_height": 10, "min_width": 12, "border_mode": 0, "value": 1, "position": "bottom_right"}, (5, 6)],
    ],
)
def test_pad_if_needed_position(params, image_shape):
    image = np.zeros(image_shape)
    pad = A.PadIfNeeded(**params)
    image_padded = pad(image=image)["image"]

    true_result = np.ones((max(image_shape[0], params["min_height"]), max(image_shape[1], params["min_width"])))

    if params["position"] == "center":
        x_start = image_shape[0] // 2
        y_start = image_shape[1] // 2
        true_result[x_start : x_start + image_shape[0], y_start : y_start + image_shape[1]] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "top_left":
        true_result[: image_shape[0], : image_shape[1]] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "top_right":
        true_result[: image_shape[0], -image_shape[1] :] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "bottom_left":
        true_result[-image_shape[0] :, : image_shape[1]] = 0
        assert (image_padded == true_result).all()

    if params["position"] == "bottom_right":
        true_result[-image_shape[0] :, -image_shape[1] :] = 0
        assert (image_padded == true_result).all()


@pytest.mark.parametrize(
    ["points"],
    [
        [
            [
                [37.25756906, 11.0567457],
                [514.03919117, 9.49484312],
                [585.66154354, 74.97413793],
                [63.60979494, 85.39815904],
            ]
        ],
        [
            [
                [37, 11],
                [514, 9],
                [585, 74],
                [63, 85],
            ]
        ],
        [
            [
                [10, 20],
                [719, 34],
                [613, 63],
                [91, 33],
            ]
        ],
    ],
)
def test_perspective_order_points(points):
    points = np.array(points)
    res = A.Perspective._order_points(points)
    assert len(points) == len(np.unique(res, axis=0))


@pytest.mark.parametrize(
    ["seed", "scale", "h", "w"],
    [
        [0, 0.08, 89, 628],
        [0, 0.15, 89, 628],
        [0, 0.15, 35, 190],
    ],
)
def test_perspective_valid_keypoints_after_transform(seed: int, scale: float, h: int, w: int):
    random.seed(seed)
    np.random.seed(seed)

    image = np.zeros([h, w, 3], dtype=np.uint8)
    keypoints = [
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0],
    ]

    transform = A.Compose(
        [A.Perspective(scale=(scale, scale), p=1)], keypoint_params={"format": "xy", "remove_invisible": False}
    )

    res = transform(image=image, keypoints=keypoints)["keypoints"]

    x1, y1 = res[0]
    x2, y2 = res[1]
    x3, y3 = res[2]
    x4, y4 = res[3]

    assert x1 < x3 and x1 < x4 and x2 < x3 and x2 < x4 and y1 < y2 and y1 < y3 and y4 < y2 and y4 < y3


@pytest.mark.parametrize("kind", ["pca", "minmax", "standard"])
def test_pixel_domain_adaptation(kind):
    img_uint8 = np.random.randint(low=100, high=200, size=(100, 100, 3), dtype=np.uint8)
    ref_img_uint8 = np.random.randint(low=0, high=100, size=(100, 100, 3), dtype=np.uint8)
    img_float, ref_img_float = [x.astype("float32") / 255.0 for x in (img_uint8, ref_img_uint8)]

    for img, ref_img in ((img_uint8, ref_img_uint8), (img_float, ref_img_float)):
        adapter = A.PixelDistributionAdaptation(
            reference_images=[ref_img],
            blend_ratio=(1, 1),
            read_fn=lambda x: x,
            always_apply=True,
            transform_type=kind,
        )
        adapted = adapter(image=img)["image"]
        np.testing.assert_allclose(
            adapted.mean(),
            ref_img.mean(),
            rtol=0,
            atol=2 if img.dtype == np.uint8 else 0.01,
            err_msg=f"{adapted.mean()} {img.mean()} {ref_img.mean()}",
        )
