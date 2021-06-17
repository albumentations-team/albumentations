import random
from typing import Type, Dict, Tuple

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
    SafeRotate,
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
    RandomToneCurve,
    ChannelDropout,
    ISONoise,
    Solarize,
    Posterize,
    Equalize,
    CropNonEmptyMaskIfExists,
    LongestMaxSize,
    Downscale,
    MultiplicativeNoise,
    GridDropout,
    ColorJitter,
    FDA,
    HistogramMatching,
    Perspective,
    Sharpen,
    Emboss,
    CropAndPad,
    Superpixels,
    Affine,
    PiecewiseAffine,
    Compose,
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
        [RandomToneCurve, {}],
        [ChannelDropout, {}],
        [ISONoise, {}],
        [Solarize, {}],
        [Posterize, {}],
        [Equalize, {}],
        [Downscale, {}],
        [MultiplicativeNoise, {}],
        [GridDropout, {}],
        [ColorJitter, {}],
        [
            HistogramMatching,
            {"reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [
            FDA,
            {"reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [Sharpen, {}],
        [Emboss, {}],
        [Superpixels, {}],
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
        [GridDropout, {}],
        [ColorJitter, {}],
        [
            HistogramMatching,
            {"reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [
            FDA,
            {"reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [Sharpen, {}],
        [Emboss, {}],
        [Superpixels, {}],
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
        [SafeRotate, {}],
        [CoarseDropout, {"fill_value": 0, "mask_fill_value": 0}],
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
        [GridDropout, {}],
        [Perspective, {}],
        [CropAndPad, {"px": 10}],
        [Affine, {}],
        [PiecewiseAffine, {}],
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
        [SafeRotate, {}],
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
        [GridDropout, {}],
        [Perspective, {}],
        [CropAndPad, {"px": 10}],
        [Affine, {}],
        [PiecewiseAffine, {}],
    ],
)
def test_dual_augmentations_with_float_values(augmentation_cls, params, float_image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=float_image, mask=mask)
    assert data["image"].dtype == np.float32
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
        [SafeRotate, {}],
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
        [RandomToneCurve, {}],
        [ChannelDropout, {}],
        [ISONoise, {}],
        [RandomGridShuffle, {}],
        [Solarize, {}],
        [Posterize, {}],
        [Equalize, {}],
        [MultiplicativeNoise, {}],
        [GridDropout, {}],
        [ColorJitter, {}],
        [
            HistogramMatching,
            {"reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [
            FDA,
            {"reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [Perspective, {}],
        [Sharpen, {}],
        [Emboss, {}],
        [CropAndPad, {"px": 10}],
        [Superpixels, {}],
        [Affine, {}],
        [PiecewiseAffine, {}],
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
        [SafeRotate, {}],
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
        [GridDropout, {}],
        [ColorJitter, {}],
        [
            HistogramMatching,
            {"reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [
            FDA,
            {"reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [Perspective, {}],
        [Sharpen, {}],
        [Emboss, {}],
        [CropAndPad, {"px": 10}],
        [Superpixels, {}],
        [Affine, {}],
        [PiecewiseAffine, {}],
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
        [SafeRotate, {}],
        [OpticalDistortion, {}],
        [GridDistortion, {}],
        [ElasticTransform, {}],
        [GaussNoise, {}],
        [ToFloat, {}],
        [FromFloat, {}],
        [RandomGridShuffle, {}],
        [RandomToneCurve, {}],
        [Solarize, {}],
        [Posterize, {}],
        [Equalize, {}],
        [MultiplicativeNoise, {}],
        [GridDropout, {}],
        [HueSaturationValue, {}],
        [ColorJitter, {}],
        [
            HistogramMatching,
            {"reference_images": [np.random.randint(0, 256, [100, 100], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [FDA, {"reference_images": [np.random.randint(0, 256, [100, 100], dtype=np.uint8)], "read_fn": lambda x: x}],
        [Perspective, {}],
        [Sharpen, {}],
        [Emboss, {}],
        [CropAndPad, {"px": 10}],
        [Superpixels, {}],
        [Affine, {}],
        [PiecewiseAffine, {}],
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
        [SafeRotate, {}],
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
        [RandomToneCurve, {}],
        [Solarize, {}],
        [Posterize, {}],
        [Equalize, {}],
        [MultiplicativeNoise, {}],
        [GridDropout, {}],
        [ColorJitter, {}],
        [
            HistogramMatching,
            {"reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [
            FDA,
            {"reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)], "read_fn": lambda x: x},
        ],
        [Perspective, {}],
        [Sharpen, {}],
        [Emboss, {}],
        [CropAndPad, {"px": 10}],
        [Superpixels, {}],
        [Affine, {}],
        [PiecewiseAffine, {}],
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
        [SafeRotate, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [ShiftScaleRotate, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [OpticalDistortion, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [ElasticTransform, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [GridDistortion, {"border_mode": cv2.BORDER_CONSTANT, "value": 100, "mask_value": 1}],
        [Affine, {"mode": cv2.BORDER_CONSTANT, "cval_mask": 1, "cval": 100}],
        [PiecewiseAffine, {"mode": "constant", "cval_mask": 1, "cval": 100}],
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
        [SafeRotate, {}],
        [PadIfNeeded, {"min_height": 514, "min_width": 516}],
        [LongestMaxSize, {"max_size": 256}],
        [GridDistortion, {}],
        [ElasticTransform, {}],
        [RandomBrightnessContrast, {}],
        [RandomToneCurve, {}],
        [MultiplicativeNoise, {}],
        [GridDropout, {}],
        [Perspective, {}],
        [CropAndPad, {"px": 10}],
        [Superpixels, {}],
        [Affine, {}],
        [PiecewiseAffine, {}],
    ],
)
def test_multichannel_image_augmentations(augmentation_cls, params):
    image = np.zeros((512, 512, 6), dtype=np.uint8)
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image)
    assert data["image"].dtype == np.uint8
    assert data["image"].shape[2] == 6


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [Blur, {}],
        [MotionBlur, {}],
        [MedianBlur, {"blur_limit": [7, 7]}],
        [GaussianBlur, {"blur_limit": [7, 7]}],
        [GaussNoise, {}],
        [RandomSizedCrop, {"min_max_height": (384, 512), "height": 512, "width": 512}],
        [ShiftScaleRotate, {}],
        [SafeRotate, {}],
        [PadIfNeeded, {"min_height": 514, "min_width": 516}],
        [LongestMaxSize, {"max_size": 256}],
        [GridDistortion, {}],
        [ElasticTransform, {}],
        [RandomBrightnessContrast, {}],
        [RandomToneCurve, {}],
        [MultiplicativeNoise, {}],
        [GridDropout, {}],
        [Perspective, {}],
        [CropAndPad, {"px": 10}],
        [Superpixels, {}],
        [Affine, {}],
        [PiecewiseAffine, {}],
    ],
)
def test_multichannel_image_augmentations_diff_channels(augmentation_cls, params):
    for num_channels in range(3, 13):
        image = np.zeros((512, 512, num_channels), dtype=np.uint8)
        aug = augmentation_cls(p=1, **params)
        data = aug(image=image)
        assert data["image"].dtype == np.uint8
        assert data["image"].shape[2] == num_channels


@pytest.mark.parametrize(
    ["augmentation_cls", "params", "image_shape"],
    [
        [PadIfNeeded, {"min_height": 514, "min_width": 516}, (300, 200)],
        [PadIfNeeded, {"min_height": 514, "min_width": 516}, (512, 516)],
        [PadIfNeeded, {"min_height": 514, "min_width": 516}, (600, 600)],
        [
            PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 128, "pad_width_divisor": 128},
            (300, 200),
        ],
        [
            PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (72, 128),
        ],
        [
            PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (15, 15),
        ],
        [
            PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (144, 256),
        ],
        [
            PadIfNeeded,
            {"min_height": None, "min_width": None, "pad_height_divisor": 72, "pad_width_divisor": 128},
            (200, 300),
        ],
        [PadIfNeeded, {"min_height": 512, "min_width": None, "pad_width_divisor": 128}, (300, 200)],
        [PadIfNeeded, {"min_height": None, "min_width": 512, "pad_height_divisor": 128}, (300, 200)],
    ],
)
def test_pad_if_needed(augmentation_cls: Type[PadIfNeeded], params: Dict, image_shape: Tuple[int, int]):
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
    res = Perspective._order_points(points)
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

    transform = Compose(
        [Perspective(scale=(scale, scale), p=1)], keypoint_params={"format": "xy", "remove_invisible": False}
    )

    res = transform(image=image, keypoints=keypoints)["keypoints"]

    x1, y1 = res[0]
    x2, y2 = res[1]
    x3, y3 = res[2]
    x4, y4 = res[3]

    assert x1 < x3 and x1 < x4 and x2 < x3 and x2 < x4 and y1 < y2 and y1 < y3 and y4 < y2 and y4 < y3
