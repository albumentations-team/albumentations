import json
import os
import random
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import albumentations as A
import albumentations.augmentations.functional as F
import albumentations.augmentations.geometric.functional as FGeometric
from albumentations.core.serialization import SERIALIZABLE_REGISTRY, shorten_class_name
from albumentations.core.transforms_interface import ImageOnlyTransform

from .conftest import skipif_no_torch
from .utils import (
    OpenMock,
    check_all_augs_exists,
    get_image_only_transforms,
    get_transforms,
    set_seed,
)

TEST_SEEDS = (0, 1, 42, 111, 9999)


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
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.Lambda,
            A.TemplateTransform,
            A.CutAndPaste,
        },
    ),
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_serialization(augmentation_cls, params, p, seed, image, mask, always_apply):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, mask=mask)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, mask=mask)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])


AUGMENTATION_CLS_PARAMS = [
    [
        A.ImageCompression,
        {
            "quality_lower": 10,
            "quality_upper": 80,
            "compression_type": A.ImageCompression.ImageCompressionType.WEBP,
        },
    ],
    [A.JpegCompression, {"quality_lower": 10, "quality_upper": 80}],
    [A.HueSaturationValue, {"hue_shift_limit": 70, "sat_shift_limit": 95, "val_shift_limit": 55}],
    [A.RGBShift, {"r_shift_limit": 70, "g_shift_limit": 80, "b_shift_limit": 40}],
    [A.RandomBrightnessContrast, {"brightness_limit": 0.5, "contrast_limit": 0.8}],
    [A.Blur, {"blur_limit": 3}],
    [A.MotionBlur, {"blur_limit": 3}],
    [A.MedianBlur, {"blur_limit": 3}],
    [A.GaussianBlur, {"blur_limit": 3}],
    [A.GaussNoise, {"var_limit": (20, 90), "mean": 10, "per_channel": False}],
    [A.CLAHE, {"clip_limit": 2, "tile_grid_size": (12, 12)}],
    [A.RandomGamma, {"gamma_limit": (10, 90)}],
    [A.Cutout, {"num_holes": 4, "max_h_size": 4, "max_w_size": 4}],
    [A.CoarseDropout, {"max_holes": 4, "max_height": 4, "max_width": 4}],
    [A.RandomSnow, {"snow_point_lower": 0.2, "snow_point_upper": 0.4, "brightness_coeff": 4}],
    [
        A.RandomRain,
        {
            "slant_lower": -5,
            "slant_upper": 5,
            "drop_length": 15,
            "drop_width": 2,
            "drop_color": (100, 100, 100),
            "blur_value": 3,
            "brightness_coefficient": 0.5,
            "rain_type": "heavy",
        },
    ],
    [A.RandomFog, {"fog_coef_lower": 0.2, "fog_coef_upper": 0.8, "alpha_coef": 0.11}],
    [
        A.RandomSunFlare,
        {
            "flare_roi": (0.1, 0.1, 0.9, 0.6),
            "angle_lower": 0.1,
            "angle_upper": 0.95,
            "num_flare_circles_lower": 7,
            "num_flare_circles_upper": 11,
            "src_radius": 300,
            "src_color": (200, 200, 200),
        },
    ],
    [
        A.RandomShadow,
        {
            "shadow_roi": (0.1, 0.4, 0.9, 0.9),
            "num_shadows_lower": 2,
            "num_shadows_upper": 4,
            "shadow_dimension": 8,
        },
    ],
    [
        A.PadIfNeeded,
        {"min_height": 512, "min_width": 512, "border_mode": cv2.BORDER_CONSTANT, "value": (10, 10, 10)},
    ],
    [
        A.Rotate,
        {
            "limit": 120,
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "value": (10, 10, 10),
            "crop_border": False,
        },
    ],
    [
        A.SafeRotate,
        {
            "limit": 120,
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "value": (10, 10, 10),
        },
    ],
    [
        A.ShiftScaleRotate,
        {
            "shift_limit": 0.2,
            "scale_limit": 0.2,
            "rotate_limit": 70,
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "value": (10, 10, 10),
        },
    ],
    [
        A.ShiftScaleRotate,
        {
            "shift_limit_x": 0.3,
            "shift_limit_y": 0.4,
            "scale_limit": 0.2,
            "rotate_limit": 70,
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "value": (10, 10, 10),
        },
    ],
    [
        A.OpticalDistortion,
        {
            "distort_limit": 0.2,
            "shift_limit": 0.2,
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "value": (10, 10, 10),
        },
    ],
    [
        A.GridDistortion,
        {
            "num_steps": 10,
            "distort_limit": 0.5,
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "value": (10, 10, 10),
        },
    ],
    [
        A.ElasticTransform,
        {
            "alpha": 2,
            "sigma": 25,
            "alpha_affine": 40,
            "interpolation": cv2.INTER_CUBIC,
            "border_mode": cv2.BORDER_CONSTANT,
            "value": (10, 10, 10),
        },
    ],
    [A.CenterCrop, {"height": 10, "width": 10}],
    [A.RandomCrop, {"height": 10, "width": 10}],
    [A.CropNonEmptyMaskIfExists, {"height": 10, "width": 10}],
    [A.RandomSizedCrop, {"min_max_height": (4, 8), "height": 10, "width": 10}],
    [A.Crop, {"x_max": 64, "y_max": 64}],
    [A.ToFloat, {"max_value": 16536}],
    [A.Normalize, {"mean": (0.385, 0.356, 0.306), "std": (0.129, 0.124, 0.125), "max_pixel_value": 100.0}],
    [A.RandomBrightness, {"limit": 0.4}],
    [A.RandomContrast, {"limit": 0.4}],
    [A.RandomScale, {"scale_limit": 0.2, "interpolation": cv2.INTER_CUBIC}],
    [A.Resize, {"height": 64, "width": 64}],
    [A.SmallestMaxSize, {"max_size": 64, "interpolation": cv2.INTER_CUBIC}],
    [A.LongestMaxSize, {"max_size": 128, "interpolation": cv2.INTER_CUBIC}],
    [A.RandomGridShuffle, {"grid": (5, 5)}],
    [A.Solarize, {"threshold": 32}],
    [A.Posterize, {"num_bits": 1}],
    [A.Equalize, {"mode": "pil", "by_channels": False}],
    [A.MultiplicativeNoise, {"multiplier": (0.7, 2.3), "per_channel": True, "elementwise": True}],
    [
        A.ColorJitter,
        {"brightness": [0.2, 0.3], "contrast": [0.7, 0.9], "saturation": [1.2, 1.7], "hue": [-0.2, 0.1]},
    ],
    [
        A.Perspective,
        {
            "scale": 0.5,
            "keep_size": False,
            "pad_mode": cv2.BORDER_REFLECT_101,
            "pad_val": 10,
            "mask_pad_val": 100,
            "fit_output": True,
            "interpolation": cv2.INTER_CUBIC,
        },
    ],
    [A.Sharpen, {"alpha": [0.2, 0.5], "lightness": [0.5, 1.0]}],
    [A.Emboss, {"alpha": [0.2, 0.5], "strength": [0.5, 1.0]}],
    [A.RandomToneCurve, {"scale": 0.2}],
    [
        A.CropAndPad,
        {
            "px": 10,
            "keep_size": False,
            "sample_independently": False,
            "interpolation": cv2.INTER_CUBIC,
            "pad_cval_mask": [10, 20, 30],
            "pad_cval": [11, 12, 13],
            "pad_mode": cv2.BORDER_REFLECT101,
        },
    ],
    [
        A.Superpixels,
        {"p_replace": (0.5, 0.7), "n_segments": (20, 30), "max_size": 25, "interpolation": cv2.INTER_CUBIC},
    ],
    [
        A.Affine,
        {
            "scale": 0.5,
            "translate_percent": 0.7,
            "translate_px": None,
            "rotate": 33,
            "shear": 21,
            "interpolation": cv2.INTER_CUBIC,
            "cval": 25,
            "cval_mask": 1,
            "mode": cv2.BORDER_REFLECT,
            "fit_output": True,
        },
    ],
    [
        A.Affine,
        {
            "scale": {"x": [0.3, 0.5], "y": [0.1, 0.2]},
            "translate_percent": None,
            "translate_px": {"x": [10, 200], "y": [5, 101]},
            "rotate": [333, 360],
            "shear": {"x": [31, 38], "y": [41, 48]},
            "interpolation": 3,
            "cval": [10, 20, 30],
            "cval_mask": 1,
            "mode": cv2.BORDER_REFLECT,
            "fit_output": True,
        },
    ],
    [
        A.PiecewiseAffine,
        {
            "scale": 0.33,
            "nb_rows": (10, 20),
            "nb_cols": 33,
            "interpolation": 2,
            "mask_interpolation": 1,
            "cval": 10,
            "cval_mask": 20,
            "mode": "edge",
            "absolute_scale": True,
            "keypoints_threshold": 0.1,
        },
    ],
    [A.ChannelDropout, dict(channel_drop_range=(1, 2), fill_value=1)],
    [A.ChannelShuffle, {}],
    [A.Downscale, dict(scale_min=0.5, scale_max=0.75, interpolation=cv2.INTER_LINEAR)],
    [A.Flip, {}],
    [A.FromFloat, dict(dtype="uint8", max_value=1)],
    [A.HorizontalFlip, {}],
    [A.ISONoise, dict(color_shift=(0.2, 0.3), intensity=(0.7, 0.9))],
    [A.InvertImg, {}],
    [A.MaskDropout, dict(max_objects=2, image_fill_value=10, mask_fill_value=20)],
    [A.NoOp, {}],
    [A.RandomResizedCrop, dict(height=20, width=30, scale=(0.5, 0.6), ratio=(0.8, 0.9))],
    [A.FancyPCA, dict(alpha=0.3)],
    [A.RandomRotate90, {}],
    [A.ToGray, {}],
    [A.ToSepia, {}],
    [A.Transpose, {}],
    [A.VerticalFlip, {}],
    [A.RingingOvershoot, dict(blur_limit=(7, 15), cutoff=(np.pi / 5, np.pi / 2))],
    [A.UnsharpMask, {"blur_limit": 3, "sigma_limit": 0.5, "alpha": 0.2, "threshold": 15}],
    [A.AdvancedBlur, dict(blur_limit=(3, 5), rotate_limit=(60, 90))],
    [A.PixelDropout, {"dropout_prob": 0.1, "per_channel": True, "drop_value": None}],
    [A.PixelDropout, {"dropout_prob": 0.1, "per_channel": False, "drop_value": None, "mask_drop_value": 15}],
    [A.RandomCropFromBorders, dict(crop_left=0.2, crop_right=0.3, crop_top=0.05, crop_bottom=0.5)],
    [A.Spatter, dict(mean=0.2, std=0.1, gauss_sigma=3, cutout_threshold=0.4, intensity=0.7, mode="mud")],
    [A.Defocus, {"radius": (5, 7), "alias_blur": (0.2, 0.6)}],
    [A.ZoomBlur, {"max_factor": (1.56, 1.7), "step_factor": (0.02, 0.04)}],
]

AUGMENTATION_CLS_EXCEPT = {
    A.FDA,
    A.HistogramMatching,
    A.PixelDistributionAdaptation,
    A.Lambda,
    A.RandomCropNearBBox,
    A.RandomSizedBBoxSafeCrop,
    A.BBoxSafeRandomCrop,
    A.GridDropout,
    A.GlassBlur,
    A.TemplateTransform,
    A.CutAndPaste,
}


@pytest.mark.parametrize(
    ["augmentation_cls", "params"], check_all_augs_exists(AUGMENTATION_CLS_PARAMS, AUGMENTATION_CLS_EXCEPT)
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_serialization_with_custom_parameters(
    augmentation_cls, params, p, seed, image, mask, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, mask=mask)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, mask=mask)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params"], check_all_augs_exists(AUGMENTATION_CLS_PARAMS, AUGMENTATION_CLS_EXCEPT)
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
@pytest.mark.parametrize("data_format", ("yaml",))
def test_augmentations_serialization_to_file_with_custom_parameters(
    augmentation_cls, params, p, seed, image, mask, always_apply, data_format
):
    with patch("builtins.open", OpenMock()):
        aug = augmentation_cls(p=p, always_apply=always_apply, **params)
        filepath = "serialized.{}".format(data_format)
        A.save(aug, filepath, data_format=data_format)
        deserialized_aug = A.load(filepath, data_format=data_format)
        set_seed(seed)
        aug_data = aug(image=image, mask=mask)
        set_seed(seed)
        deserialized_aug_data = deserialized_aug(image=image, mask=mask)
        assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
        assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])


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
            A.RandomSizedBBoxSafeCrop: {"height": 10, "width": 10},
            A.BBoxSafeRandomCrop: {"erosion_rate": 0.6},
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.Lambda,
            A.CoarseDropout,
            A.CropNonEmptyMaskIfExists,
            A.ElasticTransform,
            A.GridDistortion,
            A.RandomGridShuffle,
            A.GridDropout,
            A.MaskDropout,
            A.OpticalDistortion,
            A.TemplateTransform,
            A.CutAndPaste,
        },
    ),
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_for_bboxes_serialization(
    augmentation_cls, params, p, seed, image, albumentations_bboxes, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, bboxes=albumentations_bboxes)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, bboxes=albumentations_bboxes)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["bboxes"], deserialized_aug_data["bboxes"])


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
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.Lambda,
            A.CoarseDropout,
            A.CropNonEmptyMaskIfExists,
            A.ElasticTransform,
            A.GridDistortion,
            A.RandomGridShuffle,
            A.GridDropout,
            A.MaskDropout,
            A.OpticalDistortion,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.TemplateTransform,
            A.CutAndPaste,
        },
    ),
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_for_keypoints_serialization(augmentation_cls, params, p, seed, image, keypoints, always_apply):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, keypoints=keypoints)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, keypoints=keypoints)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["keypoints"], deserialized_aug_data["keypoints"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params", "call_params"],
    [[A.RandomCropNearBBox, {"max_part_shift": 0.15}, {"cropping_bbox": [-59, 77, 177, 231]}]],
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_serialization_with_call_params(
    augmentation_cls, params, call_params, p, seed, image, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    annotations = {"image": image, **call_params}
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(**annotations)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(**annotations)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])


def test_from_float_serialization(float_image):
    aug = A.FromFloat(p=1, dtype="uint8")
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    aug_data = aug(image=float_image)
    deserialized_aug_data = deserialized_aug(image=float_image)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])


@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_transform_pipeline_serialization(seed, image, mask):
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose(
                    [
                        A.Resize(1024, 1024),
                        A.RandomSizedCrop(min_max_height=(256, 1024), height=512, width=512, p=1),
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
                        A.Resize(1024, 1024),
                        A.RandomSizedCrop(min_max_height=(256, 1025), height=256, width=256, p=1),
                        A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1),
                    ]
                ),
            ),
            A.SomeOf(
                [
                    A.HorizontalFlip(p=1),
                    A.Transpose(p=1),
                    A.HueSaturationValue(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                2,
                replace=False,
            ),
        ]
    )
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, mask=mask)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, mask=mask)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])


@pytest.mark.parametrize(
    ["bboxes", "bbox_format", "labels"],
    [
        ([(20, 30, 40, 50)], "coco", [1]),
        ([(20, 30, 40, 50, 99), (10, 40, 30, 20, 9)], "coco", [1, 2]),
        ([(20, 30, 60, 80)], "pascal_voc", [2]),
        ([(20, 30, 60, 80, 99)], "pascal_voc", [1]),
        ([(0.2, 0.3, 0.4, 0.5)], "yolo", [2]),
        ([(0.2, 0.3, 0.4, 0.5, 99)], "yolo", [1]),
    ],
)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_transform_pipeline_serialization_with_bboxes(seed, image, bboxes, bbox_format, labels):
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose([A.RandomRotate90(), A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])]),
                A.Compose([A.Rotate(p=0.5), A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1)]),
            ),
            A.SomeOf(
                [
                    A.HorizontalFlip(p=1),
                    A.Transpose(p=1),
                    A.HueSaturationValue(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                n=5,
            ),
        ],
        bbox_params={"format": bbox_format, "label_fields": ["labels"]},
    )
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, bboxes=bboxes, labels=labels)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, bboxes=bboxes, labels=labels)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["bboxes"], deserialized_aug_data["bboxes"])


@pytest.mark.parametrize(
    ["keypoints", "keypoint_format", "labels"],
    [
        ([(20, 30, 40, 50)], "xyas", [1]),
        ([(20, 30, 40, 50, 99), (10, 40, 30, 20, 9)], "xy", [1, 2]),
        ([(20, 30, 60, 80)], "yx", [2]),
        ([(20, 30, 60, 80, 99)], "xys", [1]),
    ],
)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_transform_pipeline_serialization_with_keypoints(seed, image, keypoints, keypoint_format, labels):
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose([A.RandomRotate90(), A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])]),
                A.Compose([A.Rotate(p=0.5), A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1)]),
            ),
            A.SomeOf(
                n=2,
                transforms=[
                    A.HorizontalFlip(p=1),
                    A.Transpose(p=1),
                    A.HueSaturationValue(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ],
                replace=False,
            ),
        ],
        keypoint_params={"format": keypoint_format, "label_fields": ["labels"]},
    )
    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, keypoints=keypoints, labels=labels)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, keypoints=keypoints, labels=labels)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["keypoints"], deserialized_aug_data["keypoints"])


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        except_augmentations={A.HistogramMatching, A.FDA, A.PixelDistributionAdaptation, A.TemplateTransform},
    ),
)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_additional_targets_for_image_only_serialization(augmentation_cls, params, image, seed):
    aug = A.Compose([augmentation_cls(always_apply=True, **params)], additional_targets={"image2": "image"})
    image2 = image.copy()

    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(image=image, image2=image2)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, image2=image2)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["image2"], deserialized_aug_data["image2"])


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("p", [1])
def test_lambda_serialization(image, mask, albumentations_bboxes, keypoints, seed, p):
    def vflip_image(image, **kwargs):
        return FGeometric.vflip(image)

    def vflip_mask(mask, **kwargs):
        return FGeometric.vflip(mask)

    def vflip_bbox(bbox, **kwargs):
        return FGeometric.bbox_vflip(bbox, **kwargs)

    def vflip_keypoint(keypoint, **kwargs):
        return FGeometric.keypoint_vflip(keypoint, **kwargs)

    aug = A.Lambda(name="vflip", image=vflip_image, mask=vflip_mask, bbox=vflip_bbox, keypoint=vflip_keypoint, p=p)

    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug, lambda_transforms={"vflip": aug})
    set_seed(seed)
    aug_data = aug(image=image, mask=mask, bboxes=albumentations_bboxes, keypoints=keypoints)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image, mask=mask, bboxes=albumentations_bboxes, keypoints=keypoints)
    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
    assert np.array_equal(aug_data["mask"], deserialized_aug_data["mask"])
    assert np.array_equal(aug_data["bboxes"], deserialized_aug_data["bboxes"])
    assert np.array_equal(aug_data["keypoints"], deserialized_aug_data["keypoints"])


def test_serialization_v2_conversion_without_totensor():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    files_directory = os.path.join(current_directory, "files")
    transform_1_1_0 = A.load(os.path.join(files_directory, "transform_v1.1.0_without_totensor.json"))
    with open(os.path.join(files_directory, "output_v1.1.0_without_totensor.json")) as f:
        output_1_1_0 = json.load(f)
    np.random.seed(42)
    image = np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
    random.seed(42)
    transformed_image = transform_1_1_0(image=image)["image"]
    assert transformed_image.tolist() == output_1_1_0


@skipif_no_torch
def test_serialization_v2_conversion_with_totensor():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    files_directory = os.path.join(current_directory, "files")
    transform_1_1_0 = A.load(os.path.join(files_directory, "transform_v1.1.0_with_totensor.json"))
    with open(os.path.join(files_directory, "output_v1.1.0_with_totensor.json")) as f:
        output_1_1_0 = json.load(f)
    np.random.seed(42)
    image = np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
    random.seed(42)
    transformed_image = transform_1_1_0(image=image)["image"]
    assert transformed_image.numpy().tolist() == output_1_1_0


def test_serialization_v2_without_totensor():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    files_directory = os.path.join(current_directory, "files")
    transform = A.load(os.path.join(files_directory, "transform_serialization_v2_without_totensor.json"))
    with open(os.path.join(files_directory, "output_v1.1.0_without_totensor.json")) as f:
        output_1_1_0 = json.load(f)
    np.random.seed(42)
    image = np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
    random.seed(42)
    transformed_image = transform(image=image)["image"]
    assert transformed_image.tolist() == output_1_1_0


@skipif_no_torch
def test_serialization_v2_with_totensor():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    files_directory = os.path.join(current_directory, "files")
    transform = A.load(os.path.join(files_directory, "transform_serialization_v2_with_totensor.json"))
    with open(os.path.join(files_directory, "output_v1.1.0_with_totensor.json")) as f:
        output_1_1_0 = json.load(f)
    np.random.seed(42)
    image = np.random.randint(low=0, high=255, size=(256, 256, 3), dtype=np.uint8)
    random.seed(42)
    transformed_image = transform(image=image)["image"]
    assert transformed_image.numpy().tolist() == output_1_1_0


def test_custom_transform_with_overlapping_name():
    class HorizontalFlip(ImageOnlyTransform):
        pass

    assert SERIALIZABLE_REGISTRY["HorizontalFlip"] == A.HorizontalFlip
    assert SERIALIZABLE_REGISTRY["tests.test_serialization.HorizontalFlip"] == HorizontalFlip


def test_serialization_v2_to_dict():
    transform = A.Compose([A.HorizontalFlip()])
    transform_dict = A.to_dict(transform)["transform"]
    assert transform_dict == {
        "__class_fullname__": "Compose",
        "p": 1.0,
        "transforms": [{"__class_fullname__": "HorizontalFlip", "always_apply": False, "p": 0.5}],
        "bbox_params": None,
        "keypoint_params": None,
        "additional_targets": {},
    }


@pytest.mark.parametrize(
    ["class_fullname", "expected_short_class_name"],
    [
        ["albumentations.augmentations.transforms.HorizontalFlip", "HorizontalFlip"],
        ["HorizontalFlip", "HorizontalFlip"],
        ["some_module.HorizontalFlip", "some_module.HorizontalFlip"],
    ],
)
def test_shorten_class_name(class_fullname, expected_short_class_name):
    assert shorten_class_name(class_fullname) == expected_short_class_name


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("p", [1])
def test_template_transform_serialization(image, template, seed, p):
    template_transform = A.TemplateTransform(name="template", templates=template, p=p)

    aug = A.Compose([A.Flip(), template_transform, A.Blur()])

    serialized_aug = A.to_dict(aug)
    deserialized_aug = A.from_dict(serialized_aug, lambda_transforms={"template": template_transform})

    set_seed(seed)
    aug_data = aug(image=image)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(image=image)

    assert np.array_equal(aug_data["image"], deserialized_aug_data["image"])
