from typing import Type

import cv2
import numpy as np
import pytest
from albucore import to_float

import albumentations as A
from tests.conftest import (
    IMAGES,
    RECTANGULAR_UINT8_IMAGE,
    SQUARE_FLOAT_IMAGE,
    SQUARE_MULTI_FLOAT_IMAGE,
    SQUARE_MULTI_UINT8_IMAGE,
    SQUARE_UINT8_IMAGE,
)

from .utils import get_dual_transforms, get_image_only_transforms, get_transforms, set_seed


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
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
        },
        except_augmentations={
            A.FromFloat,
            A.Normalize,
            A.ToFloat,
        },
    ),
)
def test_image_only_augmentations_mask_persists(augmentation_cls, params):
    image = SQUARE_UINT8_IMAGE
    mask = image.copy()
    if augmentation_cls == A.TextImage:
        aug = A.Compose([augmentation_cls(p=1, **params)], bbox_params=A.BboxParams(format="pascal_voc"))
        data = aug(
            image=image,
            mask=mask,
            textimage_metadata={"text": "May the transformations be ever in your favor!", "bbox": (0.1, 0.1, 0.9, 0.2)},
        )
    else:
        aug = A.Compose([augmentation_cls(p=1, **params)])
        data = aug(image=image, mask=mask)

    assert data["image"].dtype == image.dtype
    assert data["mask"].dtype == mask.dtype
    assert np.array_equal(data["mask"], mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [SQUARE_FLOAT_IMAGE],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [SQUARE_FLOAT_IMAGE],
                "read_fn": lambda x: x,
            },
            A.PixelDistributionAdaptation: {
                "reference_images": [SQUARE_FLOAT_IMAGE],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
            A.MedianBlur: {"blur_limit": (3, 5)},
            A.TemplateTransform: {
                "templates": SQUARE_FLOAT_IMAGE,
            },
            A.RingingOvershoot: {"blur_limit": (3, 5)},
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
        },
        except_augmentations={
            A.FromFloat,
        },
    ),
)
def test_image_only_augmentations(augmentation_cls, params):
    image = SQUARE_FLOAT_IMAGE
    mask = image[:, :, 0].copy().astype(np.uint8)
    if augmentation_cls == A.TextImage:
        aug = A.Compose([augmentation_cls(p=1, **params)], bbox_params=A.BboxParams(format="pascal_voc"))
        data = aug(image=image, mask=mask, textimage_metadata={"text": "Hello, world!", "bbox": (0.1, 0.1, 0.9, 0.2)})
    else:
        aug = augmentation_cls(p=1, **params)
        data = aug(image=image, mask=mask)
    assert data["image"].dtype == image.dtype
    assert data["mask"].dtype == mask.dtype
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
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "fill_value": 0,
                "mask_fill_value": 1,
            },
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
        },
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
        },
    ),
)
def test_dual_augmentations(augmentation_cls, params):
    image = SQUARE_UINT8_IMAGE
    mask = image[:, :, 0].copy()
    aug = A.Compose([augmentation_cls(p=1, **params)])
    if augmentation_cls == A.OverlayElements:
        data = aug(image=image, mask=mask, overlay_metadata=[])
    else:
        data = aug(image=image, mask=mask)
    assert data["image"].dtype == image.dtype
    assert data["mask"].dtype == mask.dtype


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
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
        },
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
        },
    ),
)
def test_dual_augmentations_with_float_values(augmentation_cls, params):
    image = SQUARE_FLOAT_IMAGE
    mask = image.copy()[:, :, 0].astype(np.uint8)
    aug = augmentation_cls(p=1, **params)
    if augmentation_cls == A.OverlayElements:
        data = aug(image=image, mask=mask, overlay_metadata=[])
    else:
        data = aug(image=image, mask=mask)

    assert data["image"].dtype == np.float32
    assert data["mask"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
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
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.TemplateTransform: {
                "templates": SQUARE_UINT8_IMAGE,
            },
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
        },
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
        },
    ),
)
def test_augmentations_wont_change_input(augmentation_cls, params):
    image = SQUARE_FLOAT_IMAGE if augmentation_cls == A.FromFloat else SQUARE_UINT8_IMAGE
    mask = image[:, :, 0].copy()
    image_copy = image.copy()
    mask_copy = mask.copy()
    aug = augmentation_cls(p=1, **params)

    if augmentation_cls == A.OverlayElements:
        aug(image=image, mask=mask, overlay_metadata=[])
    elif augmentation_cls == A.TextImage:
        aug(
            image=image,
            mask=mask,
            textimage_metadata={"text": "May the transformations be ever in your favor!", "bbox": (0.1, 0.1, 0.9, 0.2)},
        )
    else:
        aug(image=image, mask=mask)

    assert np.array_equal(image, image_copy)
    assert np.array_equal(mask, mask_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [SQUARE_FLOAT_IMAGE],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [SQUARE_FLOAT_IMAGE],
                "read_fn": lambda x: x,
            },
            A.PixelDistributionAdaptation: {
                "reference_images": [SQUARE_FLOAT_IMAGE],
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
                "templates": SQUARE_FLOAT_IMAGE,
            },
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
        },
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
        },
    ),
)
def test_augmentations_wont_change_float_input(augmentation_cls, params):
    image = SQUARE_FLOAT_IMAGE
    float_image_copy = image.copy()

    aug = augmentation_cls(p=1, **params)

    data = {"image": image}

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.TextImage:
        data["textimage_metadata"] = {
            "text": "May the transformations be ever in your favor!",
            "bbox": (0.1, 0.1, 0.9, 0.2),
        }
    elif augmentation_cls == A.MaskDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask

    aug(**data)

    assert np.array_equal(image, float_image_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [np.random.randint(0, 255, [100, 100], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [np.random.randint(0, 255, [100, 100], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.Normalize: {"mean": 0, "std": 1},
            A.TemplateTransform: {
                "templates": np.random.randint(low=0, high=255, size=(100, 100), dtype=np.uint8),
            },
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 1], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
        },
        except_augmentations={
            A.ChannelDropout,
            A.ChannelShuffle,
            A.ISONoise,
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
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
            A.RandomScale,
            A.RandomSnow,
            A.ToRGB,
            A.ToSepia,
            A.RandomCropFromBorders,
            A.Spatter,
            A.ChromaticAberration,
            A.PlanckianJitter,
            A.RandomRain,
            A.RandomGravel,
            A.RandomChannelDropout,
            A.RandomPlanckianJitter,
        },
    ),
)
@pytest.mark.parametrize("shape", [(100, 100), (100, 100, 1)])
def test_augmentations_wont_change_shape_grayscale(augmentation_cls, params, shape):
    aug = augmentation_cls(p=1, **params)

    # Test for grayscale image
    image = np.zeros(shape, dtype=np.float32) if augmentation_cls == A.FromFloat else np.zeros(shape, dtype=np.uint8)
    mask = np.zeros(shape)

    data = {
        "image": image,
        "mask": mask,
    }
    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.TextImage:
        data["textimage_metadata"] = {
            "text": "May the transformations be ever in your favor!",
            "bbox": (0.1, 0.1, 0.9, 0.2),
        }
    result = aug(**data)

    np.testing.assert_array_equal(image.shape, result["image"].shape)
    np.testing.assert_array_equal(mask.shape, result["mask"].shape)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
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
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
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
            A.RandomCropFromBorders,
        },
    ),
)
def test_augmentations_wont_change_shape_rgb(augmentation_cls, params):
    image_3ch = SQUARE_UINT8_IMAGE
    mask_3ch = np.zeros_like(image_3ch)

    aug = augmentation_cls(p=1, **params)

    if augmentation_cls == A.OverlayElements:
        data = {
            "image": image_3ch,
            "overlay_metadata": [],
            "mask": mask_3ch,
        }
    elif augmentation_cls == A.TextImage:
        data = {
            "image": image_3ch,
            "textimage_metadata": {
                "text": "May the transformations be ever in your favor!",
                "bbox": (0.1, 0.1, 0.9, 0.2),
            },
            "mask": mask_3ch,
        }
    elif augmentation_cls == A.FromFloat:
        data = {
            "image": SQUARE_FLOAT_IMAGE,
            "mask": mask_3ch,
        }
    else:
        data = {
            "image": image_3ch,
            "mask": mask_3ch,
        }
    result = aug(**data)

    np.testing.assert_array_equal(image_3ch.shape, result["image"].shape)
    np.testing.assert_array_equal(mask_3ch.shape, result["mask"].shape)


@pytest.mark.parametrize(["augmentation_cls", "params"], [[A.RandomCropNearBBox, {"max_part_shift": 0.15}]])
@pytest.mark.parametrize("image", IMAGES)
def test_image_only_crop_around_bbox_augmentation(augmentation_cls, params, image):
    aug = augmentation_cls(p=1, **params)
    annotations = {"image": image, "cropping_bbox": [-59, 77, 177, 231]}
    data = aug(**annotations)
    assert data["image"].dtype == image.dtype


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
    set_seed(42)
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
                "reference_images": [SQUARE_MULTI_UINT8_IMAGE],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [SQUARE_MULTI_UINT8_IMAGE],
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
                "templates": SQUARE_MULTI_UINT8_IMAGE,
            },
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.ToGray: {"method": "desaturation", "num_output_channels": 5},
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 5], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.Normalize,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomGravel,
            A.RandomRain,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.RandomSnow,
            A.ToFloat,
            A.ToRGB,
            A.ToSepia,
            A.Spatter,
            A.ChromaticAberration,
            A.PlanckianJitter,
            A.RandomGrayscale,
            A.RandomHue,
            A.RandomClahe,
        },
    ),
)
def test_multichannel_image_augmentations(augmentation_cls, params):
    image = SQUARE_MULTI_UINT8_IMAGE
    aug = augmentation_cls(p=1, **params)

    data = {
        "image": image,
    }

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.TextImage:
        data["textimage_metadata"] = {
            "text": "May the transformations be ever in your favor!",
            "bbox": (0.1, 0.1, 0.9, 0.2),
        }
    elif augmentation_cls == A.MaskDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask

    data = aug(**data)
    assert data["image"].dtype == np.uint8
    assert data["image"].shape[2] == image.shape[-1]


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [SQUARE_MULTI_FLOAT_IMAGE],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [SQUARE_MULTI_UINT8_IMAGE],
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
                "templates": SQUARE_MULTI_FLOAT_IMAGE,
            },
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.ToGray: {"method": "max", "num_output_channels": 5},
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.uniform(size=(100, 100, 5)).astype(np.float32)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomGravel,
            A.RandomRain,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.RandomSnow,
            A.ToRGB,
            A.ToSepia,
            A.Equalize,
            A.Spatter,
            A.ChromaticAberration,
            A.PlanckianJitter,
            A.RandomGrayscale,
            A.RandomHue,
            A.RandomClahe,
        },
    ),
)
def test_float_multichannel_image_augmentations(augmentation_cls, params):
    image = SQUARE_MULTI_FLOAT_IMAGE
    aug = augmentation_cls(p=1, **params)
    data = {
        "image": image,
    }

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.TextImage:
        data["textimage_metadata"] = {
            "text": "May the transformations be ever in your favor!",
            "bbox": (0.1, 0.1, 0.9, 0.2),
        }
    elif augmentation_cls == A.MaskDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask

    data = aug(**data)

    assert data["image"].dtype == np.float32
    assert data["image"].shape[-1] == image.shape[-1]


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
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.ToGray: {"method": "pca", "num_output_channels": 5},
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 5], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.Normalize,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomGravel,
            A.RandomRain,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.RandomSnow,
            A.ToFloat,
            A.ToRGB,
            A.ToSepia,
            A.FancyPCA,
            A.FDA,
            A.HistogramMatching,
            A.Spatter,
            A.ChromaticAberration,
            A.PlanckianJitter,
            A.RandomGrayscale,
            A.RandomHue,
            A.RandomClahe,
        },
    ),
)
def test_multichannel_image_augmentations_diff_channels(augmentation_cls, params):
    image = SQUARE_MULTI_UINT8_IMAGE

    aug = augmentation_cls(p=1, **params)

    data = {
        "image": image,
    }

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.TextImage:
        data["textimage_metadata"] = {
            "text": "May the transformations be ever in your favor!",
            "bbox": (0.1, 0.1, 0.9, 0.2),
        }
    elif augmentation_cls == A.MaskDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask

    data = aug(**data)

    assert data["image"].dtype == np.uint8
    assert data["image"].shape[-1] == image.shape[-1]


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
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.ToGray: {"method": "pca", "num_output_channels": 5},
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.uniform(size=(100, 100, 5)).astype(np.float32)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
        },
        except_augmentations={
            A.CLAHE,
            A.ColorJitter,
            A.CropNonEmptyMaskIfExists,
            A.FromFloat,
            A.HueSaturationValue,
            A.ISONoise,
            A.RGBShift,
            A.RandomCropNearBBox,
            A.RandomGravel,
            A.RandomRain,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.RandomSnow,
            A.ToRGB,
            A.ToSepia,
            A.Equalize,
            A.FDA,
            A.HistogramMatching,
            A.Spatter,
            A.ChromaticAberration,
            A.PlanckianJitter,
            A.RandomGrayscale,
            A.RandomHue,
            A.RandomClahe,
        },
    ),
)
def test_float_multichannel_image_augmentations_diff_channels(augmentation_cls, params):
    image = SQUARE_MULTI_FLOAT_IMAGE
    aug = A.Compose([augmentation_cls(p=1, **params)])

    data = {
        "image": image,
    }

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.TextImage:
        data["textimage_metadata"] = {
            "text": "May the transformations be ever in your favor!",
            "bbox": (0.1, 0.1, 0.9, 0.2),
        }
    elif augmentation_cls == A.MaskDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask

    data = aug(**data)

    assert data["image"].dtype == np.float32
    assert data["image"].shape[2] == image.shape[-1]


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
def test_pad_if_needed(augmentation_cls: Type[A.PadIfNeeded], params: dict, image_shape: tuple[int, int]):
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
        [
            {"min_height": 10, "min_width": 12, "border_mode": cv2.BORDER_CONSTANT, "value": 1, "position": "center"},
            (5, 6),
        ],
        [
            {"min_height": 10, "min_width": 12, "border_mode": cv2.BORDER_CONSTANT, "value": 1, "position": "top_left"},
            (5, 6),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "border_mode": cv2.BORDER_CONSTANT,
                "value": 1,
                "position": "top_right",
            },
            (5, 6),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "border_mode": cv2.BORDER_CONSTANT,
                "value": 1,
                "position": "bottom_left",
            },
            (5, 6),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "border_mode": cv2.BORDER_CONSTANT,
                "value": 1,
                "position": "bottom_right",
            },
            (5, 6),
        ],
        [
            {"min_height": 10, "min_width": 12, "border_mode": cv2.BORDER_CONSTANT, "value": 1, "position": "random"},
            (5, 6),
        ],
    ],
)
def test_pad_if_needed_position(params, image_shape):
    image = np.zeros(image_shape)
    pad = A.PadIfNeeded(**params)
    pad.set_random_seed(0)

    transformed = pad(image=image)
    image_padded = transformed["image"]

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

    elif params["position"] == "bottom_right":
        true_result[-image_shape[0] :, -image_shape[1] :] = 0
        assert (image_padded == true_result).all()

    elif params["position"] == "random":
        # Find where the original image was placed (where pixels are 0)
        zero_mask = (image_padded == 0)

        # Get the bounds of the zero region
        zero_rows = np.where(zero_mask.any(axis=1))[0]
        zero_cols = np.where(zero_mask.any(axis=0))[0]

        # Check that the zero region is contiguous and of correct size
        assert len(zero_rows) == image_shape[0], "Height of placed image incorrect"
        assert len(zero_cols) == image_shape[1], "Width of placed image incorrect"
        assert np.all(np.diff(zero_rows) == 1), "Image placement not contiguous in height"
        assert np.all(np.diff(zero_cols) == 1), "Image placement not contiguous in width"

        # Verify the rest of the image is filled with ones
        padded_mask = np.ones_like(true_result)
        padded_mask[zero_rows[0]:zero_rows[-1]+1, zero_cols[0]:zero_cols[-1]+1] = 0
        assert np.all(image_padded[padded_mask == 1] == 1), "Padding value incorrect"


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
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
            A.TextImage: dict(font_path="./tests/files/LiberationSerif-Bold.ttf"),
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.MedianBlur: {"blur_limit": (3, 5)},
        },
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.FromFloat,
            A.ToFloat,
            A.Normalize,
            A.CropNonEmptyMaskIfExists,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.TemplateTransform,
            A.OverlayElements,
            A.TextImage,
            A.Solarize,
            A.RGBShift,
            A.HueSaturationValue,
            A.GaussNoise,
            A.ColorJitter,
        },
    ),
)
def test_augmentations_match_uint8_float32(augmentation_cls, params):
    image_uint8 = RECTANGULAR_UINT8_IMAGE
    image_float32 = to_float(image_uint8)

    transform = A.Compose([augmentation_cls(p=1, **params)])


    data = {"image": image_uint8}
    if augmentation_cls == A.MaskDropout:
        mask = np.zeros_like(image_uint8)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask

    set_seed(42)
    transform.set_random_seed(42)
    transformed_uint8 = transform(**data)["image"]

    set_seed(42)
    data["image"] = image_float32

    transform.set_random_seed(42)
    transformed_float32 = transform(**data)["image"]

    np.testing.assert_array_almost_equal(to_float(transformed_uint8), transformed_float32, decimal=2)


def test_solarize_threshold():
    image = SQUARE_UINT8_IMAGE
    image[20:40, 20:40] = 255
    transform = A.Solarize(threshold=128, p=1)
    transformed_image = transform(image=image)["image"]
    assert (transformed_image[20:40, 20:40] == 0).all()

    transform = A.Solarize(threshold=0.5, p=1)

    float_image = SQUARE_FLOAT_IMAGE
    float_image[20:40, 20:40] = 1
    transformed_image = transform(image=float_image)["image"]
    assert (transformed_image[20:40, 20:40] == 0).all()
