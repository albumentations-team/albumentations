import random

import cv2
import numpy as np
import pytest

from albumentations import (
    RandomCrop,
    PadIfNeeded,
    VerticalFlip,
    HorizontalFlip,
    Flip,
    Transpose,
    RandomRotate90,
    Rotate,
    ShiftScaleRotate,
    CenterCrop,
    OpticalDistortion,
    GridDistortion,
    ElasticTransform,
    RandomGridShuffle,
    ToGray,
    RandomGamma,
    ImageCompression,
    HueSaturationValue,
    RGBShift,
    Blur,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    CLAHE,
    ChannelShuffle,
    InvertImg,
    IAAEmboss,
    IAASuperpixels,
    IAASharpen,
    IAAAdditiveGaussianNoise,
    IAAPiecewiseAffine,
    IAAPerspective,
    Cutout,
    CoarseDropout,
    Normalize,
    ToFloat,
    FromFloat,
    RandomBrightnessContrast,
    RandomSnow,
    RandomRain,
    RandomFog,
    RandomSunFlare,
    RandomCropNearBBox,
    RandomShadow,
    RandomSizedCrop,
    RandomResizedCrop,
    ChannelDropout,
    ISONoise,
    Solarize,
    Posterize,
    Equalize,
    CropNonEmptyMaskIfExists,
    LongestMaxSize,
    Downscale,
    MultiplicativeNoise,
)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [ImageCompression, {}],
        [HueSaturationValue, {}],
        [RGBShift, {}],
        [RandomBrightnessContrast, {}],
        [Blur, {}],
        [MotionBlur, {}],
        [MedianBlur, {}],
        [GaussianBlur, {}],
        [GaussNoise, {}],
        [CLAHE, {}],
        [ChannelShuffle, {}],
        [InvertImg, {}],
        [RandomGamma, {}],
        [ToGray, {}],
        [Cutout, {}],
        [CoarseDropout, {}],
        [GaussNoise, {}],
        [RandomSnow, {}],
        [RandomRain, {}],
        [RandomFog, {}],
        [RandomSunFlare, {}],
        [RandomShadow, {}],
        [ChannelDropout, {}],
        [ISONoise, {}],
        [Solarize, {}],
        [Posterize, {}],
        [Equalize, {}],
        [Downscale, {}],
        [MultiplicativeNoise, {}],
    ],
)
def test_image_only_augmentations(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image, mask=mask)
    assert data["image"].dtype == np.uint8
    assert data["mask"].dtype == np.uint8
    assert np.array_equal(data["mask"], mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [HueSaturationValue, {}],
        [RGBShift, {}],
        [RandomBrightnessContrast, {}],
        [Blur, {}],
        [MotionBlur, {}],
        [MedianBlur, {"blur_limit": (3, 5)}],
        [GaussianBlur, {}],
        [GaussNoise, {}],
        [ChannelShuffle, {}],
        [InvertImg, {}],
        [RandomGamma, {}],
        [ImageCompression, {}],
        [ToGray, {}],
        [Cutout, {}],
        [CoarseDropout, {}],
        [GaussNoise, {}],
        [RandomSnow, {}],
        [RandomRain, {}],
        [RandomFog, {}],
        [RandomSunFlare, {}],
        [RandomShadow, {}],
        [ChannelDropout, {}],
        [Solarize, {}],
        [MultiplicativeNoise, {}],
    ],
)
def test_image_only_augmentations_with_float_values(augmentation_cls, params, float_image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=float_image, mask=mask)
    assert data["image"].dtype == np.float32
    assert data["mask"].dtype == np.uint8
    assert np.array_equal(data["mask"], mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [PadIfNeeded, {}],
        [VerticalFlip, {}],
        [HorizontalFlip, {}],
        [Flip, {}],
        [Transpose, {}],
        [RandomRotate90, {}],
        [Rotate, {}],
        [ShiftScaleRotate, {}],
        [OpticalDistortion, {}],
        [GridDistortion, {}],
        [ElasticTransform, {}],
        [CenterCrop, {"height": 10, "width": 10}],
        [RandomCrop, {"height": 10, "width": 10}],
        [CropNonEmptyMaskIfExists, {"height": 10, "width": 10}],
        [RandomResizedCrop, {"height": 10, "width": 10}],
        [RandomSizedCrop, {"min_max_height": (4, 8), "height": 10, "width": 10}],
        [ISONoise, {}],
        [RandomGridShuffle, {}],
    ],
)
def test_dual_augmentations(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image, mask=mask)
    assert data["image"].dtype == np.uint8
    assert data["mask"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [PadIfNeeded, {}],
        [VerticalFlip, {}],
        [HorizontalFlip, {}],
        [Flip, {}],
        [Transpose, {}],
        [RandomRotate90, {}],
        [Rotate, {}],
        [ShiftScaleRotate, {}],
        [OpticalDistortion, {}],
        [GridDistortion, {}],
        [ElasticTransform, {}],
        [CenterCrop, {"height": 10, "width": 10}],
        [RandomCrop, {"height": 10, "width": 10}],
        [CropNonEmptyMaskIfExists, {"height": 10, "width": 10}],
        [RandomResizedCrop, {"height": 10, "width": 10}],
        [RandomSizedCrop, {"min_max_height": (4, 8), "height": 10, "width": 10}],
        [RandomGridShuffle, {}],
    ],
)
def test_dual_augmentations_with_float_values(augmentation_cls, params, float_image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=float_image, mask=mask)
    assert data["image"].dtype == np.float32
    assert data["mask"].dtype == np.uint8


@pytest.mark.parametrize("augmentation_cls", [IAAEmboss, IAASuperpixels, IAASharpen, IAAAdditiveGaussianNoise])
def test_imgaug_image_only_augmentations(augmentation_cls, image, mask):
    aug = augmentation_cls(p=1)
    data = aug(image=image, mask=mask)
    assert data["image"].dtype == np.uint8
    assert data["mask"].dtype == np.uint8
    assert np.array_equal(data["mask"], mask)


@pytest.mark.parametrize("augmentation_cls", [IAAPiecewiseAffine, IAAPerspective])
def test_imgaug_dual_augmentations(augmentation_cls, image, mask):
    aug = augmentation_cls(p=1)
    data = aug(image=image, mask=mask)
    assert data["image"].dtype == np.uint8
    assert data["mask"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [Cutout, {}],
        [ImageCompression, {}],
        [HueSaturationValue, {}],
        [RGBShift, {}],
        [RandomBrightnessContrast, {}],
        [RandomBrightnessContrast, {}],
        [Blur, {}],
        [MotionBlur, {}],
        [MedianBlur, {}],
        [GaussianBlur, {}],
        [GaussNoise, {}],
        [CLAHE, {}],
        [ChannelShuffle, {}],
        [InvertImg, {}],
        [RandomGamma, {}],
        [ToGray, {}],
        [Cutout, {}],
        [CoarseDropout, {}],
        [PadIfNeeded, {}],
        [VerticalFlip, {}],
        [HorizontalFlip, {}],
        [Flip, {}],
        [Transpose, {}],
        [RandomRotate90, {}],
        [Rotate, {}],
        [ShiftScaleRotate, {}],
        [OpticalDistortion, {}],
        [GridDistortion, {}],
        [ElasticTransform, {}],
        [CenterCrop, {"height": 10, "width": 10}],
        [RandomCrop, {"height": 10, "width": 10}],
        [CropNonEmptyMaskIfExists, {"height": 10, "width": 10}],
        [RandomResizedCrop, {"height": 10, "width": 10}],
        [RandomSizedCrop, {"min_max_height": (4, 8), "height": 10, "width": 10}],
        [Normalize, {}],
        [GaussNoise, {}],
        [ToFloat, {}],
        [FromFloat, {}],
        [RandomSnow, {}],
        [RandomRain, {}],
        [RandomFog, {}],
        [RandomSunFlare, {}],
        [RandomShadow, {}],
        [ChannelDropout, {}],
        [ISONoise, {}],
        [RandomGridShuffle, {}],
        [Solarize, {}],
        [Posterize, {}],
        [Equalize, {}],
        [MultiplicativeNoise, {}],
    ],
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
    [
        [Cutout, {}],
        [CoarseDropout, {}],
        [HueSaturationValue, {}],
        [RGBShift, {}],
        [RandomBrightnessContrast, {}],
        [RandomBrightnessContrast, {}],
        [Blur, {}],
        [MotionBlur, {}],
        [MedianBlur, {"blur_limit": (3, 5)}],
        [GaussianBlur, {}],
        [GaussNoise, {}],
        [ChannelShuffle, {}],
        [InvertImg, {}],
        [RandomGamma, {}],
        [ToGray, {}],
        [PadIfNeeded, {}],
        [VerticalFlip, {}],
        [HorizontalFlip, {}],
        [Flip, {}],
        [Transpose, {}],
        [RandomRotate90, {}],
        [Rotate, {}],
        [ShiftScaleRotate, {}],
        [OpticalDistortion, {}],
        [GridDistortion, {}],
        [ElasticTransform, {}],
        [CenterCrop, {"height": 10, "width": 10}],
        [RandomCrop, {"height": 10, "width": 10}],
        [RandomResizedCrop, {"height": 10, "width": 10}],
        [RandomSizedCrop, {"min_max_height": (4, 8), "height": 10, "width": 10}],
        [Normalize, {}],
        [GaussNoise, {}],
        [ToFloat, {}],
        [FromFloat, {}],
        [RandomSnow, {}],
        [RandomRain, {}],
        [RandomFog, {}],
        [RandomSunFlare, {}],
        [RandomShadow, {}],
        [ChannelDropout, {}],
        [RandomGridShuffle, {}],
        [Solarize, {}],
        [MultiplicativeNoise, {}],
    ],
)
def test_augmentations_wont_change_float_input(augmentation_cls, params, float_image):
    float_image_copy = float_image.copy()
    aug = augmentation_cls(p=1, **params)
    aug(image=float_image)
    assert np.array_equal(float_image, float_image_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [Cutout, {}],
        [CoarseDropout, {}],
        [ImageCompression, {}],
        [RandomBrightnessContrast, {}],
        [Blur, {}],
        [MotionBlur, {}],
        [MedianBlur, {}],
        [GaussianBlur, {}],
        [GaussNoise, {}],
        [InvertImg, {}],
        [RandomGamma, {}],
        [VerticalFlip, {}],
        [HorizontalFlip, {}],
        [Flip, {}],
        [Transpose, {}],
        [RandomRotate90, {}],
        [Rotate, {}],
        [OpticalDistortion, {}],
        [GridDistortion, {}],
        [ElasticTransform, {}],
        [GaussNoise, {}],
        [ToFloat, {}],
        [FromFloat, {}],
        [RandomGridShuffle, {}],
        [Solarize, {}],
        [Posterize, {}],
        [Equalize, {}],
        [MultiplicativeNoise, {}],
    ],
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

    # Test for RGB image
    image_3ch = np.zeros((224, 224, 3), dtype=np.uint8)
    mask_3ch = np.zeros((224, 224, 3))

    result = aug(image=image_3ch, mask=mask_3ch)
    assert np.array_equal(image_3ch.shape, result["image"].shape)
    assert np.array_equal(mask_3ch.shape, result["mask"].shape)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [Cutout, {}],
        [CoarseDropout, {}],
        [ImageCompression, {}],
        [HueSaturationValue, {}],
        [RGBShift, {}],
        [RandomBrightnessContrast, {}],
        [Blur, {}],
        [MotionBlur, {}],
        [MedianBlur, {}],
        [GaussianBlur, {}],
        [GaussNoise, {}],
        [CLAHE, {}],
        [ChannelShuffle, {}],
        [InvertImg, {}],
        [RandomGamma, {}],
        [ToGray, {}],
        [VerticalFlip, {}],
        [HorizontalFlip, {}],
        [Flip, {}],
        [Transpose, {}],
        [RandomRotate90, {}],
        [Rotate, {}],
        [OpticalDistortion, {}],
        [GridDistortion, {}],
        [ElasticTransform, {}],
        [Normalize, {}],
        [GaussNoise, {}],
        [ToFloat, {}],
        [FromFloat, {}],
        [RandomSnow, {}],
        [RandomRain, {}],
        [RandomFog, {}],
        [RandomSunFlare, {}],
        [RandomShadow, {}],
        [ChannelDropout, {}],
        [ISONoise, {}],
        [RandomGridShuffle, {}],
        [Solarize, {}],
        [Posterize, {}],
        [Equalize, {}],
        [MultiplicativeNoise, {}],
    ],
)
def test_augmentations_wont_change_shape_rgb(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)

    # Test for RGB image
    image_3ch = np.zeros((224, 224, 3), dtype=np.uint8)
    mask_3ch = np.zeros((224, 224, 3))

    result = aug(image=image_3ch, mask=mask_3ch)
    assert np.array_equal(image_3ch.shape, result["image"].shape)
    assert np.array_equal(mask_3ch.shape, result["mask"].shape)


@pytest.mark.parametrize(["augmentation_cls", "params"], [[RandomCropNearBBox, {"max_part_shift": 0.15}]])
def test_image_only_crop_around_bbox_augmentation(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)
    annotations = {"image": image, "cropping_bbox": [-59, 77, 177, 231]}
    data = aug(**annotations)
    assert data["image"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [
            PadIfNeeded,
            {"min_height": 514, "min_width": 514, "border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1},
        ],
        [Rotate, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [ShiftScaleRotate, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [OpticalDistortion, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [ElasticTransform, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [GridDistortion, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
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
    [
        [Blur, {}],
        [MotionBlur, {}],
        [MedianBlur, {}],
        [GaussianBlur, {}],
        [GaussNoise, {}],
        [RandomSizedCrop, {"min_max_height": (384, 512), "height": 512, "width": 512}],
        [ShiftScaleRotate, {}],
        [PadIfNeeded, {"min_height": 514, "min_width": 516}],
        [LongestMaxSize, {"max_size": 256}],
        [GridDistortion, {}],
        [ElasticTransform, {}],
        [RandomBrightnessContrast, {}],
        [MultiplicativeNoise, {}],
    ],
)
def test_multichannel_image_augmentations(augmentation_cls, params):
    image = np.zeros((512, 512, 6), dtype=np.uint8)
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image)
    assert data["image"].dtype == np.uint8
    assert data["image"].shape[2] == 6
