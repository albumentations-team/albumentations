from typing import Type

import cv2
import numpy as np
import pytest
from albucore import to_float

from .aug_definitions import transforms2metadata_key

import albumentations as A
from tests.conftest import (
    IMAGES,
    RECTANGULAR_UINT8_IMAGE,
    SQUARE_FLOAT_IMAGE,
    SQUARE_MULTI_FLOAT_IMAGE,
    SQUARE_MULTI_UINT8_IMAGE,
    SQUARE_UINT8_IMAGE,
)

from .utils import get_2d_transforms, get_dual_transforms, get_image_only_transforms, get_transforms, set_seed


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
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

    data = {
        "image": image,
        "mask": mask,
    }
    if augmentation_cls == A.TextImage:
        aug = A.Compose([augmentation_cls(p=1, **params)], bbox_params=A.BboxParams(format="pascal_voc"), strict=True)
        data = aug(**data, textimage_metadata={"text": "May the transformations be ever in your favor!", "bbox": (0.1, 0.1, 0.9, 0.2)})
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]
        aug = A.Compose([augmentation_cls(p=1, **params)], strict=True)
        data = aug(**data)
    else:
        aug = A.Compose([augmentation_cls(p=1, **params)], strict=True)
        data = aug(**data)

    assert data["image"].dtype == image.dtype
    assert data["mask"].dtype == mask.dtype
    assert np.array_equal(data["mask"], mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        except_augmentations={
            A.FromFloat,
        },
    ),
)
def test_image_only_augmentations(augmentation_cls, params):
    image = SQUARE_FLOAT_IMAGE
    mask = image[:, :, 0].copy().astype(np.uint8)


    data = {
        "image": image,
        "mask": mask,
    }
    if augmentation_cls == A.TextImage:
        aug = A.Compose([augmentation_cls(p=1, **params)], bbox_params=A.BboxParams(format="pascal_voc"), strict=True)
        data = aug(**data, textimage_metadata={"text": "Hello, world!", "bbox": (0.1, 0.1, 0.9, 0.2)})
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": SQUARE_FLOAT_IMAGE,
                "mask": mask
            }
        ]
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]
        aug = A.Compose([augmentation_cls(p=1, **params)], strict=True)
        data = aug(**data)
    else:
        aug = augmentation_cls(p=1, **params)
        data = aug(**data)

    assert data["image"].dtype == image.dtype
    assert data["mask"].dtype == mask.dtype
    assert np.array_equal(data["mask"], mask)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={ },
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
        },
    ),
)
def test_dual_augmentations(augmentation_cls, params):
    image = SQUARE_UINT8_IMAGE
    mask = image[:, :, 0].copy()
    aug = A.Compose([augmentation_cls(p=1, **params)], strict=True)
    data = {"image": image, "mask": mask}
    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.RandomCropNearBBox:
        data["cropping_bbox"] = [0, 0, 10, 10]
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": SQUARE_UINT8_IMAGE,
                "mask": mask
            }
        ]
    data = aug(**data)
    assert data["image"].dtype == image.dtype
    assert data["mask"].dtype == mask.dtype


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={ },
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

    data = {"image": image, "mask": mask}

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.RandomCropNearBBox:
        data["cropping_bbox"] = [0, 0, 10, 10]
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": SQUARE_FLOAT_IMAGE,
                "mask": mask
            }
        ]

    data = aug(**data)

    assert data["image"].dtype == np.float32
    assert data["mask"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
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

    data = {"image": image, "mask": mask}

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.TextImage:
        data["textimage_metadata"] = {
            "text": "May the transformations be ever in your favor!",
            "bbox": (0.1, 0.1, 0.9, 0.2),
        }
    elif augmentation_cls == A.RandomCropNearBBox:
        data["cropping_bbox"] = [0, 0, 10, 10]
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": SQUARE_UINT8_IMAGE,
                "mask": mask
            }
        ]
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]

    aug(**data)

    np.testing.assert_array_equal(image, image_copy)
    np.testing.assert_array_equal(mask, mask_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
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
    elif augmentation_cls == A.MaskDropout or augmentation_cls == A.ConstrainedCoarseDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.RandomCropNearBBox:
        data["cropping_bbox"] = [0, 0, 10, 10]
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            }
        ]
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]

    aug(**data)

    np.testing.assert_array_equal(image, float_image_copy)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        custom_arguments={
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
            A.AtLeastOneBBoxRandomCrop,
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
            A.RandomSunFlare,
            A.RandomFog,
            A.Pad,
            A.HEStain,
            A.Mosaic,
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
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
                "mask": mask
            }
        ]
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]

    result = aug(**data)

    np.testing.assert_array_equal(image.shape, result["image"].shape)
    np.testing.assert_array_equal(mask.shape, result["mask"].shape)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CenterCrop,
            A.Crop,
            A.CropNonEmptyMaskIfExists,
            A.RandomCrop,
            A.AtLeastOneBBoxRandomCrop,
            A.RandomResizedCrop,
            A.RandomSizedCrop,
            A.CropAndPad,
            A.Resize,
            A.LongestMaxSize,
            A.SmallestMaxSize,
            A.PadIfNeeded,
            A.RandomScale,
            A.RandomCropFromBorders,
            A.ConstrainedCoarseDropout,
            A.Pad,
            A.Mosaic,
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
    elif augmentation_cls == A.Mosaic:
        data = {
            "image": image_3ch,
            "mask": mask_3ch,
            "mosaic_metadata": [
                {
                    "image": image_3ch,
                    "mask": mask_3ch,
                }
            ]
        }
    elif augmentation_cls in transforms2metadata_key:
        data = {
            "image": image_3ch,
            "mask": mask_3ch,
            transforms2metadata_key[augmentation_cls]: [image_3ch]
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
            {"min_height": 514, "min_width": 514, "border_mode": cv2.BORDER_CONSTANT, "fill": 100, "fill_mask": 1},
        ],
        [A.Rotate, {"border_mode": cv2.BORDER_CONSTANT, "fill": 100, "fill_mask": 1}],
        [A.SafeRotate, {"border_mode": cv2.BORDER_CONSTANT, "fill": 100, "fill_mask": 1}],
        [A.ShiftScaleRotate, {"border_mode": cv2.BORDER_CONSTANT, "fill": 100, "fill_mask": 1}],
        [A.Affine, {"border_mode": cv2.BORDER_CONSTANT, "fill_mask": 1, "fill": 100}],
    ],
)
def test_mask_fill_value(augmentation_cls, params):
    set_seed(137)
    aug = augmentation_cls(p=1, **params)
    input = {"image": np.zeros((512, 512), dtype=np.uint8) + 100, "mask": np.ones((512, 512))}
    output = aug(**input)
    assert (output["image"] == 100).all()
    assert (output["mask"] == 1).all()


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        custom_arguments={
            A.ToGray: {
                "method": "pca",
                "num_output_channels": 5,
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
            A.RandomSunFlare,
            A.RandomFog,
            A.Equalize,
            A.GridElasticDeform,
            A.HEStain,
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
    elif augmentation_cls == A.MaskDropout or augmentation_cls == A.ConstrainedCoarseDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            }
        ]
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]

    data = aug(**data)
    assert data["image"].dtype == np.uint8
    assert data["image"].shape[2] == image.shape[-1]


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        custom_arguments={
            A.ToGray: {
                "method": "pca",
                "num_output_channels": 5,
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
            A.RandomSunFlare,
            A.RandomFog,
            A.GridElasticDeform,
            A.HEStain,
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
    elif augmentation_cls == A.MaskDropout or augmentation_cls == A.ConstrainedCoarseDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            }
        ]
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]

    data = aug(**data)

    assert data["image"].dtype == np.float32
    assert data["image"].shape[-1] == image.shape[-1]


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        custom_arguments={
            A.ToGray: {
                "method": "pca",
                "num_output_channels": 5,
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
            A.Spatter,
            A.ChromaticAberration,
            A.PlanckianJitter,
            A.RandomSunFlare,
            A.RandomFog,
            A.Equalize,
            A.GridElasticDeform,
            A.HEStain,
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
    elif augmentation_cls == A.MaskDropout or augmentation_cls == A.ConstrainedCoarseDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            }
        ]
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]

    data = aug(**data)

    assert data["image"].dtype == np.uint8
    assert data["image"].shape[-1] == image.shape[-1]


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        custom_arguments={
            A.ToGray: {
                "method": "pca",
                "num_output_channels": 5,
            },
            A.ToGray: {
                "method": "pca",
                "num_output_channels": 5,
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
            A.RandomSunFlare,
            A.RandomFog,
            A.GridElasticDeform,
            A.HEStain,
        },
    ),
)
def test_float_multichannel_image_augmentations_diff_channels(augmentation_cls, params):
    image = SQUARE_MULTI_FLOAT_IMAGE
    aug = A.Compose([augmentation_cls(p=1, **params)], strict=True)

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
    elif augmentation_cls == A.MaskDropout or augmentation_cls == A.ConstrainedCoarseDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            }
        ]
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [image]

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
            {"min_height": 10, "min_width": 12, "border_mode": cv2.BORDER_CONSTANT, "fill": 1, "position": "center"},
            (5, 6),
        ],
        [
            {"min_height": 10, "min_width": 12, "border_mode": cv2.BORDER_CONSTANT, "fill": 1, "position": "top_left"},
            (5, 6),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 1,
                "position": "top_right",
            },
            (5, 6),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 1,
                "position": "bottom_left",
            },
            (5, 6),
        ],
        [
            {
                "min_height": 10,
                "min_width": 12,
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 1,
                "position": "bottom_right",
            },
            (5, 6),
        ],
        [
            {"min_height": 10, "min_width": 12, "border_mode": cv2.BORDER_CONSTANT, "fill": 1, "position": "random"},
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
    get_2d_transforms(
        custom_arguments={
            A.ShiftScaleRotate: {
                "fill": 0,
                "interpolation": cv2.INTER_NEAREST,
            },
            A.SafeRotate: {
                "interpolation": cv2.INTER_NEAREST,
                "fill": 0,
            },
            A.Rotate: {
                "interpolation": cv2.INTER_NEAREST,
                "fill": 0,
            },
            A.RandomScale: {
                "scale_limit": 0.2,
                "interpolation": cv2.INTER_NEAREST,
            },
            A.Affine: {
                "interpolation": cv2.INTER_NEAREST,
                "fill": 0,
            },
            A.PixelDropout: {
                "drop_value": 0,
            },
            A.PadIfNeeded: {
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 0,
            },
            A.ChannelDropout: {
                "fill": 0,
            },
            A.PiecewiseAffine: {
                "interpolation": cv2.INTER_NEAREST,
            },
            A.Perspective: {
                "interpolation": cv2.INTER_NEAREST,
            },
            A.GridDropout: {
                "fill": 0,
            },
            A.GridDistortion: {
                "interpolation": cv2.INTER_NEAREST,
            },
            A.ElasticTransform: {
                "interpolation": cv2.INTER_NEAREST,
            },
            A.Pad : {
                "fill": 0,
            },
            A.Resize: {
                "interpolation": cv2.INTER_NEAREST,
                "height": 50,
                "width": 50,
            },
            A.CropAndPad: {
                "fill": 0,
                "px": 10,
            },
            A.OpticalDistortion: {
                "interpolation": cv2.INTER_NEAREST,
            },
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
            A.OverlayElements,
            A.TextImage,
            A.RGBShift,
            A.HueSaturationValue,
            A.ColorJitter,
            A.Mosaic,
        },
    ),
)
def test_augmentations_match_uint8_float32(augmentation_cls, params):
    image_uint8 = RECTANGULAR_UINT8_IMAGE
    image_float32 = to_float(image_uint8)

    transform = A.Compose([augmentation_cls(p=1, **params)], seed=137, strict=True)

    data = {"image": image_uint8}
    if augmentation_cls == A.MaskDropout or augmentation_cls == A.ConstrainedCoarseDropout:
        mask = np.zeros_like(image_uint8)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.RandomCropNearBBox:
        data["cropping_bbox"] = [12, 77, 177, 231]

    transformed_uint8 = transform(**data)["image"]

    data["image"] = image_float32

    transform.set_random_seed(137)
    transformed_float32 = transform(**data)["image"]

    np.testing.assert_array_almost_equal(to_float(transformed_uint8), transformed_float32, decimal=2)


def test_solarize_threshold():
    image = SQUARE_UINT8_IMAGE
    image[20:40, 20:40] = 255
    transform = A.Solarize(threshold_range = (0.5, 0.5), p=1)
    transformed_image = transform(image=image)["image"]
    assert (transformed_image[20:40, 20:40] == 0).all()

    transform = A.Solarize(threshold_range=(0.5, 0.5), p=1)

    float_image = SQUARE_FLOAT_IMAGE
    float_image[20:40, 20:40] = 1
    transformed_image = transform(image=float_image)["image"]
    assert (transformed_image[20:40, 20:40] == 0).all()


def test_constrained_coarse_dropout_with_mask():
    """Test ConstrainedCoarseDropout with segmentation mask."""
    # Create test data
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)

    # Create objects in mask
    mask[10:30, 10:30] = 1  # First object (class 1)
    mask[40:60, 40:60] = 2  # Second object (class 2)
    mask[70:90, 70:90] = 2  # Third object (class 2)

    transform = A.ConstrainedCoarseDropout(
        num_holes_range=(2, 2),  # Fixed 2 holes per object
        hole_height_range=(0.3, 0.3),  # Fixed 30% of object height
        hole_width_range=(0.3, 0.3),  # Fixed 30% of object width
        mask_indices=[1, 2],
        p=1.0,
    )
    transform.set_random_seed(137)

    # Apply transform
    _ = transform(image=image, mask=mask)

    # Get holes
    params = transform.get_params_dependent_on_data({}, {"image": image, "mask": mask})
    holes = params["holes"]

    # Verify number of holes (2 per object, 3 objects)
    assert len(holes) == 6, f"Expected 6 holes (2 per object), got {len(holes)}"

    # Verify holes are within image bounds
    for hole in holes:
        x1, y1, x2, y2 = hole
        assert 0 <= x1 < x2 <= 100, f"Invalid hole x coordinates: {x1}, {x2}"
        assert 0 <= y1 < y2 <= 100, f"Invalid hole y coordinates: {y1}, {y2}"


@pytest.mark.parametrize(
    ["bbox_labels", "bboxes",  "expected_num_objects"],
    [
        # Case 1: String labels
        (
            ["Billy The Cat", "dog"],
            [
                [10, 10, 20, 20, "Billy The Cat"],
                [30, 30, 40, 40, "dog"],
                [50, 50, 60, 60, "bird"],  # Should be ignored
            ],
            2,  # 2 objects: one cat, one dog
        ),
        # Case 2: Numeric labels
        (
            [1, 2],
            [
                [10, 10, 20, 20, 1],
                [30, 30, 40, 40, 2],
                [50, 50, 60, 60, 3],  # Should be ignored
            ],
            2,  # 2 objects: class 1 and class 2
        ),
    ],
)
def test_constrained_coarse_dropout_with_bboxes(bbox_labels, bboxes, expected_num_objects):
    """Test ConstrainedCoarseDropout with bounding boxes."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    transform = A.Compose([
        A.ConstrainedCoarseDropout(
            num_holes_range=(2, 2),  # Fixed 2 holes per object
            hole_height_range=(0.3, 0.3),  # Fixed 30% of object height
            hole_width_range=(0.3, 0.3),  # Fixed 30% of object width
            bbox_labels=bbox_labels,
            p=1.0,
        )
    ],
    strict=True,
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']), seed=137, save_applied_params=True)


    # Extract labels for bbox_params
    labels = [bbox[4] for bbox in bboxes]
    bboxes_without_labels = [bbox[:4] for bbox in bboxes]

    # Apply transform
    transformed = transform(image=image, bboxes=bboxes_without_labels, class_labels=labels)

    # Get applied parameters
    applied_params = transformed['applied_transforms'][0][1]  # First transform's params
    holes = applied_params['holes']

    # Verify number of holes (2 per object)
    assert len(holes) == expected_num_objects * 2, \
        f"Expected {expected_num_objects * 2} holes (2 per object), got {len(holes)}"

    # Verify holes are within image bounds
    for hole in holes:
        x1, y1, x2, y2 = hole
        assert 0 <= x1 < x2 <= 100, f"Invalid hole x coordinates: {x1}, {x2}"
        assert 0 <= y1 < y2 <= 100, f"Invalid hole y coordinates: {y1}, {y2}"

    # Verify holes overlap with target boxes
    target_boxes = [
        bbox[:4] for bbox, label in zip(bboxes, labels)
        if label in bbox_labels
    ]

    for hole in holes:
        overlaps_any = False
        for box in target_boxes:
            # Check for overlap
            if not (hole[2] <= box[0] or  # hole right < box left
                   hole[0] >= box[2] or  # hole left > box right
                   hole[3] <= box[1] or  # hole bottom < box top
                   hole[1] >= box[3]):   # hole top > box bottom
                overlaps_any = True
                break
        assert overlaps_any, f"Hole {hole} doesn't overlap with any target box"



@pytest.mark.parametrize(
    ["drop_value", "expected_values"],
    [
        (None, None),  # Random values will be generated
        (0, 0),  # Single value
        ((1, 2, 3), np.array([1, 2, 3])),  # Sequence of values
    ],
)
def test_pixel_dropout_drop_values(drop_value, expected_values):
    image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    transform = A.PixelDropout(dropout_prob=1.0, drop_value=drop_value, p=1.0)

    result = transform(image=image)["image"]

    if drop_value is None:
        # For None, we just verify values are within valid range
        assert result.dtype == np.uint8
        assert np.all((result >= 0) & (result <= 255))
    else:
        if isinstance(drop_value, (int, float)):
            # For single value, all channels should have same value
            assert np.all(result == expected_values)
        else:
            # For sequence, each channel should have corresponding value
            for channel_idx, expected_value in enumerate(expected_values):
                assert np.all(result[:, :, channel_idx] == expected_value)


def test_pixel_dropout_per_channel():
    """Test that per_channel=True works correctly with different drop_values"""
    image = np.ones((10, 10, 3), dtype=np.uint8) * 255

    # Test with single value
    transform = A.PixelDropout(
        dropout_prob=0.5,
        drop_value=0,
        per_channel=True,
        p=1.0
    )
    result = transform(image=image)["image"]
    assert np.any(result == 0)  # Should have some dropped pixels

    # Test with sequence
    transform = A.PixelDropout(
        dropout_prob=0.5,
        drop_value=(1, 2, 3),
        per_channel=True,
        p=1.0
    )
    result = transform(image=image)["image"]
    # Each channel should only contain original values or its drop value
    for channel_idx, drop_val in enumerate((1, 2, 3)):
        unique_values = np.unique(result[:, :, channel_idx])
        assert len(unique_values) == 2  # Should only have original value and drop value
        assert drop_val in unique_values
        assert 255 in unique_values



def test_salt_and_pepper_noise():
    # Test image setup - create all gray image instead of black with gray square
    image = np.full((100, 100, 3), 128, dtype=np.uint8)  # All gray image

    # Fixed parameters for deterministic testing
    amount = (0.05, 0.05)  # Exactly 5% of pixels
    salt_vs_pepper = (0.6, 0.6)  # Exactly 60% salt, 40% pepper

    transform = A.SaltAndPepper(
        amount=amount,
        salt_vs_pepper=salt_vs_pepper,
        p=1.0
    )
    transform.set_random_seed(137)

    # Apply transform
    transformed = transform(image=image)["image"]

    # Count all changes
    salt_pixels = (transformed == 255).all(axis=2)
    pepper_pixels = (transformed == 0).all(axis=2)

    total_changes = salt_pixels.sum() + pepper_pixels.sum()

    expected_pixels = int(image.shape[0] * image.shape[1] * amount[0])
    assert total_changes == expected_pixels, \
        f"Expected {expected_pixels} noisy pixels, got {total_changes}"

    # Verify salt vs pepper ratio
    expected_salt = int(expected_pixels * salt_vs_pepper[0])
    assert salt_pixels.sum() == expected_salt, \
        f"Expected {expected_salt} salt pixels, got {salt_pixels.sum()}"


def test_salt_and_pepper_float_image():
    """Test salt and pepper noise on float32 images"""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    image[25:75, 25:75] = 0.5  # Gray square

    transform = A.SaltAndPepper(
        amount=(0.05, 0.05),
        salt_vs_pepper=(0.6, 0.6),
        p=1.0
    )
    transform.set_random_seed(137)

    transformed = transform(image=image)["image"]

    # Check that salt pixels are 1.0 and pepper pixels are 0.0
    changed_mask = (transformed != image).any(axis=2)
    assert np.allclose(transformed[transformed > 0.9], 1.0), \
        "Salt pixels should be exactly 1.0 for float images"
    assert np.allclose(transformed[transformed < 0.1], 0.0), \
        "Pepper pixels should be exactly 0.0 for float images"

def test_salt_and_pepper_grayscale():
    """Test salt and pepper noise on single-channel images"""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[25:75, 25:75] = 128

    transform = A.SaltAndPepper(
        amount=(0.05, 0.05),
        salt_vs_pepper=(0.6, 0.6),
        p=1.0
    )
    transform.set_random_seed(137)

    transformed = transform(image=image)["image"]

    # Verify shape is preserved
    assert transformed.shape == image.shape, \
        "Transform should preserve single-channel image shape"

    # Check noise values
    changed_mask = transformed != image
    salt_pixels = (transformed == 255) & changed_mask
    pepper_pixels = (transformed == 0) & changed_mask

    assert (salt_pixels | pepper_pixels | ~changed_mask).all(), \
        "Changed pixels should only be salt (255) or pepper (0)"



@pytest.mark.parametrize(
    ["slant_range", "expected_slant_range"],
    [
        ((-10, -5), (-10, -5)),  # negative slant range
        ((5, 10), (5, 10)),      # positive slant range
        ((-5, 5), (-5, 5)),      # range crossing zero
    ]
)
def test_random_rain_slant(slant_range, expected_slant_range):
    # Create a deterministic image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create transform with 100% probability and specific slant range
    transform = A.RandomRain(
        slant_range=slant_range,
        p=1.0,
        rain_type="heavy"  # Use heavy to ensure enough rain drops
    )
    transform.set_random_seed(137)
    # Run multiple times to ensure slant stays within range
    slants = []
    for _ in range(50):  # Run 50 times to get a good sample
        # Get params without actually applying transform
        params = transform.get_params_dependent_on_data(
            {"shape": image.shape},
            {"image": image}
        )
        slants.append(params["slant"])

    # Assert all slants are within the expected range
    assert all(expected_slant_range[0] <= s <= expected_slant_range[1] for s in slants), \
        f"Slants {slants} not within range {expected_slant_range}"

    # Assert we get at least some variation in slants
    assert len(set(slants)) > 1, "Slant values show no variation"

    # Assert we get values from both halves of the range
    slant_mid = (expected_slant_range[0] + expected_slant_range[1]) / 2
    has_lower = any(s < slant_mid for s in slants)
    has_upper = any(s > slant_mid for s in slants)
    if expected_slant_range[0] != expected_slant_range[1]:  # Skip if range is single value
        assert has_lower and has_upper, \
            f"Slant values {slants} don't cover both halves of range {expected_slant_range}"

@pytest.mark.parametrize("slant", [-10, 0, 10])
def test_random_rain_visual_effect(slant):
    # Create a white image
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    # Create transform with fixed slant for visual verification
    transform = A.RandomRain(
        slant_range=(slant, slant),  # Force specific slant
        drop_length=20,
        drop_width=2,
        drop_color=(0, 0, 0),  # Black rain drops for contrast
        blur_value=1,          # Minimal blur for clearer lines
        brightness_coefficient=1.0,  # No brightness change
        p=1.0,
        rain_type="heavy"
    )

    transform.set_random_seed(137)

    # Apply transform
    result = transform(image=image)["image"]

    # Find non-white pixels (rain drops)
    rain_pixels = np.where(result != 255)

    if len(rain_pixels[0]) > 0:  # If we found rain drops
        # Calculate the average slope of rain drops
        # We can do this by looking at the leftmost and rightmost points
        left_x = min(rain_pixels[1])
        right_x = max(rain_pixels[1])
        if right_x > left_x:  # Ensure we have horizontal spread
            left_y = rain_pixels[0][rain_pixels[1] == left_x].mean()
            right_y = rain_pixels[0][rain_pixels[1] == right_x].mean()

            # Calculate observed slant direction
            observed_slant = 1 if right_y > left_y else -1 if right_y < left_y else 0
            expected_slant = 1 if slant > 0 else -1 if slant < 0 else 0

            # The slant direction should match the expected direction
            assert observed_slant == expected_slant, \
                f"Rain drops slant direction {observed_slant} doesn't match expected {expected_slant}"
