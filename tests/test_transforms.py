from __future__ import annotations
import random
from functools import partial
from typing import Any

import cv2
import numpy as np
import pytest
from albucore import to_float, clip, MAX_VALUES_BY_DTYPE

import albumentations as A
import albumentations.augmentations.pixel.functional as fpixel
import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.core.transforms_interface import BasicTransform
from tests.conftest import (
    IMAGES,
    SQUARE_FLOAT_IMAGE,
    SQUARE_MULTI_UINT8_IMAGE,
    SQUARE_UINT8_IMAGE,
    RECTANGULAR_UINT8_IMAGE,
)

from .aug_definitions import transforms2metadata_key

from .utils import get_2d_transforms, get_dual_transforms, get_image_only_transforms


def test_transpose_both_image_and_mask():
    image = np.ones((8, 6, 3))
    mask = np.ones((8, 6))
    augmentation = A.Transpose(p=1)
    augmented = augmentation(image=image, mask=mask)
    assert augmented["image"].shape == (6, 8, 3)
    assert augmented["mask"].shape == (6, 8)


def test_rotate_crop_border(image):
    height = image.shape[0]
    border_value = 256
    aug = A.Rotate(
        limit=(45, 45),
        p=1,
        fill=border_value,
        border_mode=cv2.BORDER_CONSTANT,
        crop_border=True,
    )
    aug_img = aug(image=image)["image"]
    expected_size = int(np.round(height / np.sqrt(2)))
    assert aug_img.shape[0] == expected_size
    assert (aug_img == border_value).sum() == 0


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
            A.GridDropout: {"fill_mask": 0},
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.PixelDropout,
            A.Lambda,
            A.RandomRotate90,
            A.D4,
            A.VerticalFlip,
            A.HorizontalFlip,
            A.Transpose,
        },
    ),
)
def test_binary_mask_interpolation(augmentation_cls, params, image):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts"""
    params["mask_interpolation"] = cv2.INTER_NEAREST
    params["fill_mask"] = 0

    aug = augmentation_cls(p=1, **params)
    mask = cv2.randu(np.zeros((100, 100), dtype=np.uint8), 0, 2)
    data = {
        "image": image,
        "mask": mask,
    }
    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = []
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
                "mask": mask,
            }
        ]

    result = aug(**data)
    np.testing.assert_array_equal(np.unique(result["mask"]), np.array([0, 1]))


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
            A.GridDropout: {"holes_number_xy": (10, 10), "fill_mask": 64},
        },
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.PixelDropout,
            A.CropNonEmptyMaskIfExists,
            A.PixelDistributionAdaptation,
            A.PadIfNeeded,
            A.RandomCrop,
            A.AtLeastOneBBoxRandomCrop,
            A.Crop,
            A.CenterCrop,
            A.FDA,
            A.HistogramMatching,
            A.Lambda,
            A.CropNonEmptyMaskIfExists,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.TextImage,
            A.FromFloat,
            A.MaskDropout,
            A.XYMasking,
            A.RandomCropNearBBox,
            A.PiecewiseAffine,
            A.Lambda,
            A.RandomRotate90,
            A.D4,
            A.VerticalFlip,
            A.HorizontalFlip,
            A.Transpose,
            A.Mosaic,
        },
    ),
)
def test_semantic_mask_interpolation(augmentation_cls, params, image):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts."""

    seed = 137
    params["mask_interpolation"] = cv2.INTER_NEAREST
    params["fill_mask"] = 0

    mask = cv2.randu(np.zeros((100, 100), dtype=np.uint8), 0, 4) * 64

    data = A.Compose([augmentation_cls(p=1, **params)], seed=seed, strict=False)(image=image, mask=mask)

    np.testing.assert_array_equal(np.unique(data["mask"]), np.array([0, 64, 128, 192]))


def __test_multiprocessing_support_proc(args):
    x, transform = args
    return transform(image=x)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        custom_arguments={
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.OverlayElements,
            A.TextImage,
            A.MaskDropout,
            A.Mosaic,
        },
    ),
)
def test_multiprocessing_support(mp_pool, augmentation_cls, params):
    """Checks whether we can use augmentations in multiprocessing environments"""
    image = (
        SQUARE_FLOAT_IMAGE if augmentation_cls == A.FromFloat else SQUARE_UINT8_IMAGE
    )
    aug = augmentation_cls(p=1, **params)

    mp_pool.map(
        __test_multiprocessing_support_proc, map(lambda x: (x, aug), [image] * 10)
    )


def test_force_apply():
    """Unit test for https://github.com/albumentations-team/albumentations/issues/189"""
    aug = A.Compose(
        [
            A.OneOrOther(
                A.Compose(
                    [
                        A.RandomSizedCrop(
                            min_max_height=(256, 1025), size=(512, 512), p=1
                        ),
                        A.OneOf(
                            [
                                A.RandomSizedCrop(
                                    min_max_height=(256, 512),
                                    size=(384, 384),
                                    p=0.5,
                                ),
                                A.RandomSizedCrop(
                                    min_max_height=(256, 512),
                                    size=(512, 512),
                                    p=0.5,
                                ),
                            ],
                        ),
                    ],
                ),
                A.Compose(
                    [
                        A.RandomSizedCrop(
                            min_max_height=(256, 1025), size=(256, 256), p=1
                        ),
                        A.OneOf([A.HueSaturationValue(p=0.5), A.RGBShift(p=0.7)], p=1),
                    ],
                ),
            ),
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(p=0.5),
        ],
    )

    res = aug(image=np.zeros((1248, 1248, 3), dtype=np.uint8))
    assert res["image"].shape[0] in (256, 384, 512)
    assert res["image"].shape[1] in (256, 384, 512)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        except_augmentations={
            A.TextImage,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
        },
    ),
)
def test_additional_targets_for_image_only(augmentation_cls, params):
    aug = A.Compose(
        [augmentation_cls(p=1, **params)], additional_targets={"image2": "image"}, strict=True
    )
    for _ in range(10):
        image1 = (
            SQUARE_FLOAT_IMAGE
            if augmentation_cls == A.FromFloat
            else SQUARE_UINT8_IMAGE
        )
        image2 = image1.copy()
        res = aug(image=image1, image2=image2)
        aug1 = res["image"]
        aug2 = res["image2"]
        np.testing.assert_array_equal(aug1, aug2)

    aug = A.Compose([augmentation_cls(p=1, **params)], strict=True)
    aug.add_targets(additional_targets={"image2": "image"})
    for _ in range(10):
        image1 = (
            SQUARE_FLOAT_IMAGE
            if augmentation_cls == A.FromFloat
            else SQUARE_UINT8_IMAGE
        )
        image2 = image1.copy()
        res = aug(image=image1, image2=image2)
        aug1 = res["image"]
        aug2 = res["image2"]
        np.testing.assert_array_equal(aug1, aug2)


def test_image_invert():
    for _ in range(10):
        # test for np.uint8 dtype
        image1 = cv2.randu(np.zeros((100, 100, 3), dtype=np.uint8), 0, 255)
        image2 = to_float(image1)
        r_int = fpixel.invert(fpixel.invert(image1))
        r_float = fpixel.invert(fpixel.invert(image2))
        r_to_float = to_float(r_int)
        assert np.allclose(r_float, r_to_float, atol=0.01)


def test_lambda_transform():
    def negate_image(image, **kwargs):
        return -image

    def one_hot_mask(mask, num_channels, **kwargs):
        return np.eye(num_channels, dtype=np.uint8)[mask]

    def vflip_bboxes(bboxes, **kwargs):
        return fgeometric.bboxes_vflip(bboxes)

    def vflip_keypoints(keypoints, **kwargs):
        return fgeometric.keypoints_vflip(keypoints, kwargs["shape"][0])

    aug = A.Lambda(
        image=negate_image,
        mask=partial(one_hot_mask, num_channels=16),
        bboxes=vflip_bboxes,
        keypoints=vflip_keypoints,
        p=1,
    )

    bboxes = np.array([[0.1, 0.1, 0.4, 0.5]])
    keypoints = np.array([[2, 3, np.pi / 4, 5]])

    height, width = 10, 10

    image = np.ones((height, width, 3), dtype=np.float32)
    mask = np.tile(np.arange(0, height), (width, 1)).T

    output = aug(
        image=image,
        mask=mask,
        bboxes=bboxes,
        keypoints=keypoints,
    )

    assert (output["image"] < 0).all()
    assert output["mask"].shape[2] == 16  # num_channels
    np.testing.assert_array_almost_equal(
        output["bboxes"], fgeometric.bboxes_vflip(np.array(bboxes))
    )
    np.testing.assert_array_almost_equal(
        output["keypoints"], fgeometric.keypoints_vflip(np.array(keypoints), height)
    )


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

    img = cv2.randu(np.zeros((256, 256, 3), dtype=np.uint8), 0, 255)
    a = aug(image=img)["image"]
    b = fpixel.equalize(img)
    assert np.all(a == b)

    mask = cv2.randu(np.zeros((256, 256), dtype=np.uint8), 0, 2)
    aug = A.Equalize(mask=mask, p=1)
    a = aug(image=img)["image"]
    b = fpixel.equalize(img, mask=mask)
    assert np.all(a == b)

    def mask_func(image, test):
        return mask

    aug = A.Equalize(mask=mask_func, mask_params=["test"], p=1)
    assert np.all(aug(image=img, test=mask)["image"] == fpixel.equalize(img, mask=mask))


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
    mask_1 = np.zeros(
        [10, 10], dtype=np.uint8
    )  # uint8 required for passing mask_1 as `masks` (which uses bitwise or)
    mask_1[0, 0] = 1
    crop_1 = np.array([[1]])
    aug_1 = A.CropNonEmptyMaskIfExists(1, 1)

    # test empty mask
    mask_2 = np.zeros(
        [10, 10], dtype=np.uint8
    )  # uint8 required for passing mask_2 as `masks` (which uses bitwise or)
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
    _test_crops(np.stack([mask_2, mask_1]), np.stack([crop_2, crop_1]), aug_1, n=1)


@pytest.mark.parametrize(
    "image",
    [
        cv2.randu(np.zeros((256, 320), dtype=np.uint8), 0, 255),
        cv2.randu(np.zeros((256, 320), dtype=np.float32), 0, 1),
        cv2.randu(np.zeros((256, 320, 1), dtype=np.uint8), 0, 255),
        cv2.randu(np.zeros((256, 320, 1), dtype=np.float32), 0, 1),
    ],
)
def test_multiplicative_noise_grayscale(image):
    m = 0.5
    aug = A.MultiplicativeNoise((m, m), elementwise=False, p=1)
    params = aug.get_params_dependent_on_data(
        params={"shape": image.shape},
        data={"image": image},
    )
    assert m == params["multiplier"]
    result_e = aug(image=image)["image"]

    expected = image.astype(np.float32) * params["multiplier"]

    assert np.allclose(clip(expected, image.dtype), result_e)

    aug = A.MultiplicativeNoise((m, m), elementwise=True, p=1)
    params = aug.get_params_dependent_on_data(
        params={"shape": image.shape},
        data={"image": image},
    )
    result_ne = aug.apply(image, params["multiplier"])

    expected = image.astype(np.float32) * params["multiplier"]

    assert np.allclose(clip(expected, image.dtype), result_ne)


@pytest.mark.parametrize(
    "image",
    IMAGES,
)
@pytest.mark.parametrize(
    "elementwise",
    (True, False),
)
def test_multiplicative_noise_rgb(image, elementwise):
    dtype = image.dtype

    aug = A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=elementwise, p=1)
    params = aug.get_params_dependent_on_data(
        params={"shape": image.shape},
        data={"image": image},
    )
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
    img = cv2.randu(np.zeros((50, 10, 3), dtype=np.uint8), 0, 255)
    mask = np.ones([50, 10], dtype=np.int64)

    aug = A.MaskDropout(p=1)
    result = aug(image=img, mask=mask)
    assert np.all(result["image"] == 0)
    assert np.all(result["mask"] == 0)

    # In this case we have mask with zeros , so MaskDropout will make no changes
    img = cv2.randu(np.zeros((50, 10, 3), dtype=np.uint8), 0, 255)
    mask = np.zeros([50, 10], dtype=np.int64)

    aug = A.MaskDropout(p=1)
    result = aug(image=img, mask=mask)
    assert np.all(result["image"] == img)
    assert np.all(result["mask"] == 0)


@pytest.mark.parametrize("val_uint8", [0, 1, 128, 255])
def test_unsharp_mask_float_uint8_diff_less_than_two(val_uint8):
    x_uint8 = np.zeros((5, 5)).astype(np.uint8)
    x_uint8[2, 2] = val_uint8

    x_float32 = np.zeros((5, 5)).astype(np.float32)
    x_float32[2, 2] = val_uint8 / 255.0

    unsharpmask = A.UnsharpMask(blur_limit=3, p=1)
    unsharpmask.set_random_seed(0)

    usm_uint8 = unsharpmask(image=x_uint8)["image"]

    usm_float32 = unsharpmask(image=x_float32)["image"]

    # Before comparison, rescale the usm_float32 to [0, 255]
    diff = np.abs(usm_uint8 - usm_float32 * 255)

    # The difference between the results of float32 and uint8 will be at most 2.
    assert np.all(diff <= 2.0), f"Max difference: {diff.max()}"


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
            ),
        ],
        seed=137,
        strict=True,
    )

    res1 = transform(image=img)["image"]
    res2 = (transform(image=img.astype(np.float32) / 255.0)["image"] * 255).astype(
        np.uint8
    )

    _max = np.abs(res1.astype(np.int16) - res2.astype(np.int16)).max()

    if hue != 0:
        assert _max <= 10, f"Max: {_max}"
    else:
        assert _max <= 2, f"Max: {_max}"


def test_perspective_keep_size():
    height, width = 100, 100
    img = np.zeros([height, width, 3], dtype=np.uint8)
    bboxes = []
    for _ in range(10):
        x1 = np.random.randint(0, width - 1)
        y1 = np.random.randint(0, height - 1)
        x2 = np.random.randint(x1 + 1, width)
        y2 = np.random.randint(y1 + 1, height)
        bboxes.append([x1, y1, x2, y2])
    keypoints = [
        (np.random.randint(0, width), np.random.randint(0, height), np.random.random())
        for _ in range(10)
    ]

    transform_1 = A.Compose(
        [A.Perspective(keep_size=True, p=1)],
        keypoint_params=A.KeypointParams("xys"),
        bbox_params=A.BboxParams("pascal_voc", label_fields=["labels"]),
        seed=137,
        strict=True,
    )

    res_1 = transform_1(
        image=img, bboxes=bboxes, keypoints=keypoints, labels=[0] * len(bboxes)
    )

    transform_2 = A.Compose(
        [A.Perspective(keep_size=False, p=1), A.Resize(height, width, p=1)],
        keypoint_params=A.KeypointParams("xys"),
        bbox_params=A.BboxParams("pascal_voc", label_fields=["labels"]),
        seed=137,
        strict=True,
    )

    res_2 = transform_2(
        image=img, bboxes=bboxes, keypoints=keypoints, labels=[0] * len(bboxes)
    )

    assert np.allclose(res_1["bboxes"], res_2["bboxes"], atol=0.2)
    assert np.allclose(res_1["keypoints"], res_2["keypoints"])

    assert res_1["image"].shape == img.shape


def test_longest_max_size_list():
    img = cv2.randu(np.zeros((50, 10), dtype=np.uint8), 0, 255)
    keypoints = np.array([(9, 5, 33, 0, 0)])

    aug = A.LongestMaxSize(max_size=[5, 10], p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["image"].shape in [(10, 2), (5, 1)]
    assert tuple(result["keypoints"][0].tolist()) in [
        (0.9, 0.5, 33, 0, 0),
        (1.8, 1.0, 33, 0, 0),
    ]


def test_smallest_max_size_list():
    img = cv2.randu(np.zeros((50, 10), dtype=np.uint8), 0, 255)
    keypoints = np.array([(9, 5, 33, 0, 0)])

    aug = A.SmallestMaxSize(max_size=[50, 100], p=1)
    result = aug(image=img, keypoints=keypoints)
    assert result["image"].shape in [(250, 50), (500, 100)]
    assert tuple(result["keypoints"][0].tolist()) in [
        (45.0, 25.0, 33, 0, 0),
        (90.0, 50.0, 33, 0, 0),
    ]


@pytest.mark.parametrize(
    "params",
    [
        {"scale": (0.5, 1.0)},
        {"scale": (0.5, 1.0), "keep_ratio": False},
        {"scale": (0.5, 1.0), "keep_ratio": True},
    ],
)
def test_affine_scale_ratio(params):
    aug = A.Affine(**params, p=1.0)
    aug.set_random_seed(0)

    image = SQUARE_UINT8_IMAGE

    data = {"image": image}
    call_params = aug.get_params()
    call_params = aug.update_transform_params(call_params, data)

    apply_params = aug.get_params_dependent_on_data(params=call_params, data=data)

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
    "params",
    [
        {"scale": {"x": (0.5, 1.0), "y": (1.0, 1.5)}, "keep_ratio": True},
        {"scale": {"x": 0.5, "y": 1.0}, "keep_ratio": True},
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
                    [199, 0, 10, 8.72],
                    [199, 99, 20, 17.46],
                    [0, 99, 30, 26.19],
                ],
            },
            {
                "bboxes": [
                    [
                        15.264852523803711,
                        0.0,
                        20.678197860717773,
                        4.27599573135376,
                        0.0,
                    ],
                    [
                        194.73916625976562,
                        25.38059425354004,
                        200.0,
                        29.73569107055664,
                        0.0,
                    ],
                    [
                        179.32180786132812,
                        95.72400665283203,
                        184.7351531982422,
                        100.0,
                        0.0,
                    ],
                    [
                        0.009779278188943863,
                        70.2643051147461,
                        5.260837078094482,
                        73.87895202636719,
                        0.0,
                    ],
                ],
                "keypoints": [
                    [16.455339431762695, 0.35640794038772583, 351.9260559082031, 0.0],
                    [
                        199.61117553710938,
                        26.338354110717773,
                        1.9260603189468384,
                        8.149533,
                    ],
                    [
                        183.54466247558594,
                        99.64359283447266,
                        11.92605972290039,
                        16.317757,
                    ],
                    [
                        0.3888263702392578,
                        73.6616439819336,
                        21.92605972290039,
                        24.476635,
                    ],
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
                    [199, 0, 10, 8.72],
                    [199, 99, 20, 17.46],
                    [0, 99, 30, 26.19],
                ],
            },
            {
                "bboxes": [
                    [0.0, 25.38059425354004, 5.260837078094482, 29.73569107055664, 0.0],
                    [
                        179.32180786132812,
                        0.0,
                        184.7351531982422,
                        4.275995254516602,
                        0.0,
                    ],
                    [
                        194.73916625976562,
                        70.2643051147461,
                        200.0,
                        74.6194076538086,
                        0.0,
                    ],
                    [
                        15.264852523803711,
                        95.72400665283203,
                        20.515911102294922,
                        99.3386459350586,
                        0.0,
                    ],
                ],
                "keypoints": [
                    [0.3888259530067444, 26.338354110717773, 8.073939323425293, 0.0],
                    [
                        183.54466247558594,
                        0.35640716552734375,
                        18.073938369750977,
                        8.149533,
                    ],
                    [
                        199.61117553710938,
                        73.6616439819336,
                        28.073936462402344,
                        16.317757,
                    ],
                    [
                        16.455339431762695,
                        99.64359283447266,
                        38.073936462402344,
                        24.476635,
                    ],
                ],
            },
        ],
    ],
)
def test_safe_rotate(angle: float, targets: dict, expected: dict):
    image = np.empty([100, 200, 3], dtype=np.uint8)
    t = A.Compose(
        [
            A.SafeRotate(limit=(angle, angle), border_mode=0, fill=0, p=1),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.0),
        keypoint_params=A.KeypointParams("xyas", angle_in_degrees=True),
        p=1,
        seed=137,
        strict=True,
    )
    res = t(image=image, **targets)

    for key, value in expected.items():
        np.testing.assert_allclose(res[key], value, atol=1e-6, rtol=1e-6), key


@pytest.mark.parametrize(
    "aug_cls",
    [
        (lambda rotate: A.Affine(rotate=rotate, p=1, border_mode=cv2.BORDER_CONSTANT, fill=0)),
        (
            lambda rotate: A.ShiftScaleRotate(
                shift_limit=(0, 0),
                scale_limit=(0, 0),
                rotate_limit=rotate,
                p=1,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            )
        ),
    ],
)
@pytest.mark.parametrize(
    "img",
    [
        SQUARE_UINT8_IMAGE,
        cv2.randu(np.zeros((25, 100, 3), dtype=np.uint8), 0, 255),
        cv2.randu(np.zeros((100, 25, 3), dtype=np.uint8), 0, 255),
    ],
)
# @pytest.mark.parametrize("angle", list(range(-360, 360, 15)))
@pytest.mark.parametrize("angle", [15])
def test_rotate_equal(img, aug_cls, angle):
    height, width = img.shape[:2]
    kp = [
        [
            random.randint(0, width - 1),
            random.randint(0, height - 1),
            random.randint(0, 360),
        ]
        for _ in range(50)
    ]
    kp += [
        [round(width * 0.2), int(height * 0.3), 90],
        [int(width * 0.2), int(height * 0.3), 90],
        [int(width * 0.2), int(height * 0.3), 90],
        [int(width * 0.2), int(height * 0.3), 90],
        [0, 0, 0],
        [width - 1, height - 1, 0],
    ]
    keypoint_params = A.KeypointParams("xya", remove_invisible=False)

    a = A.Compose(
        [aug_cls(rotate=(angle, angle))], keypoint_params=keypoint_params, seed=137, strict=True
    )

    b = A.Compose(
        [A.Rotate((angle, angle), border_mode=cv2.BORDER_CONSTANT, fill=0, p=1)],
        keypoint_params=keypoint_params,
        seed=137,
        strict=True,
    )
    res_a = a(image=img, keypoints=kp)
    res_b = b(image=img, keypoints=kp)
    np.testing.assert_array_equal(res_a["image"], res_b["image"])

    res_a = np.array(res_a["keypoints"])
    res_b = np.array(res_b["keypoints"])
    diff = np.round(np.abs(res_a - res_b))
    assert diff[:, :2].max() <= 2
    assert (diff[:, -1] % 360).max() <= 1


def test_motion_blur_allow_shifted():
    # Use fixed parameters to ensure horizontal/vertical lines for center checking
    transform = A.MotionBlur(
        allow_shifted=False,
        angle_range=(0, 0),  # Fixed horizontal angle
        direction_range=(0, 0),  # Symmetric direction
        blur_limit=(7, 7)  # Fixed kernel size
    )
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


def test_motion_blur_allow_shifted_true():
    """Test that allow_shifted=True produces different kernels at transform level."""
    transform = A.MotionBlur(
        allow_shifted=True,
        angle_range=(0, 0),  # Fixed horizontal angle
        direction_range=(0, 0),  # Symmetric direction
        blur_limit=(7, 7),  # Fixed kernel size
        p=1.0
    )

    # Generate multiple kernels with same transform instance
    kernels = []
    for _ in range(10):
        kernel = transform.get_params()["kernel"]
        kernels.append(kernel)

    # Check that not all kernels are identical (shifting should cause variation)
    # We expect at least some kernels to be different due to random shifting
    unique_kernels = set()
    for kernel in kernels:
        # Convert to tuple for hashability
        kernel_tuple = tuple(kernel.flatten())
        unique_kernels.add(kernel_tuple)

    # With allow_shifted=True, we should get some variation in kernel positions
    # Even if not all are different, we should have more than 1 unique kernel
    assert len(unique_kernels) > 1, "allow_shifted=True should produce some variation in kernel positions"


@pytest.mark.parametrize(
    "augmentation",
    [
        A.RandomGravel(),
        A.RandomSnow(),
        A.RandomRain(),
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


@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize(
    "grid",
    [
        (3, 3),
        (4, 4),
        (5, 7),
    ],
)
def test_grid_shuffle(image, grid):
    """As we reshuffle the grid, the mean and sum of the image and mask should remain the same,
    while the reshuffled image and mask should not be equal to the original image and mask.
    """
    mask = image.copy()

    aug = A.Compose([A.RandomGridShuffle(grid=grid, p=1)], seed=137, strict=True)

    res = aug(image=image, mask=mask)
    assert res["image"].shape == image.shape
    assert res["mask"].shape == mask.shape

    assert not np.array_equal(res["image"], image)
    assert not np.array_equal(res["mask"], mask)

    np.testing.assert_allclose(
        res["image"].sum(axis=(0, 1)), image.sum(axis=(0, 1)), atol=0.04
    )
    np.testing.assert_allclose(
        res["mask"].sum(axis=(0, 1)), mask.sum(axis=(0, 1)), atol=0.03
    )


@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize(
    "crop_left, crop_right, crop_top, crop_bottom",
    [
        (0, 0, 0, 0),
        (0, 1, 0, 1),
        (1, 0, 1, 0),
        (0.5, 0.5, 0.5, 0.5),
        (0.1, 0.1, 0.1, 0.1),
        (0.3, 0.3, 0.3, 0.3),
    ],
)
def test_random_crop_from_borders(
    image, bboxes, keypoints, crop_left, crop_right, crop_top, crop_bottom
):
    aug = A.Compose(
        [
            A.RandomCropFromBorders(
                crop_left=crop_left,
                crop_right=crop_right,
                crop_top=crop_top,
                crop_bottom=crop_bottom,
                p=1,
            ),
        ],
        bbox_params=A.BboxParams("pascal_voc"),
        keypoint_params=A.KeypointParams("xy"),
        seed=137,
        strict=True,
    )

    assert aug(image=image, mask=image, bboxes=bboxes, keypoints=keypoints)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.NoOp,
            A.Lambda,
            A.ToRGB,
            A.FrequencyMasking,
            A.TimeMasking,
            A.RandomRotate90,
        },
    ),
)
def test_change_image(augmentation_cls, params, image):
    """Checks whether resulting image is different from the original one."""
    aug = A.Compose([augmentation_cls(p=1, **params)], seed=137, strict=False)

    original_image = image.copy()

    data = {
        "image": image,
    }

    if augmentation_cls == A.OverlayElements:
        data["overlay_metadata"] = {
            "image": clip(SQUARE_UINT8_IMAGE + 2, image.dtype),
            "bbox": (0.1, 0.12, 0.6, 0.3),
        }
    elif augmentation_cls == A.FromFloat:
        data["image"] = SQUARE_FLOAT_IMAGE
    elif augmentation_cls == A.TextImage:
        data["textimage_metadata"] = {
            "text": "May the transformations be ever in your favor!",
            "bbox": (0.1, 0.1, 0.9, 0.2),
        }
    elif augmentation_cls in {A.MaskDropout, A.ConstrainedCoarseDropout}:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.ConstrainedCoarseDropout:
        data["mask"] = np.zeros_like(image)[:, :, 0]
        data["mask"][:20, :20] = 1
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            }
        ]
    elif augmentation_cls in transforms2metadata_key:
        data[transforms2metadata_key[augmentation_cls]] = [np.random.randint(0, 255, image.shape, dtype=image.dtype)]


    transformed = aug(**data)

    np.testing.assert_array_equal(image, original_image)
    assert not np.array_equal(transformed["image"], image)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        custom_arguments={
            A.AdvancedBlur: {"blur_limit": (5, 7),
                             "sigma_x_limit": (1, 3),
                             "sigma_y_limit": (1, 3),
                             "rotate_limit": (0, 0),
                             },
        },
        except_augmentations={
            A.Crop,
            A.CenterCrop,
            A.CropNonEmptyMaskIfExists,
            A.RandomCrop,
            A.AtLeastOneBBoxRandomCrop,
            A.RandomResizedCrop,
            A.RandomSizedCrop,
            A.CropAndPad,
            A.Resize,
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.FDA,
            A.HistogramMatching,
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
            A.PlanckianJitter,
            A.OverlayElements,
            A.FromFloat,
            A.TextImage,
            A.PixelDistributionAdaptation,
            A.MaskDropout,
            A.Pad,
            A.ConstrainedCoarseDropout,
            A.RandomRotate90,
            A.FrequencyMasking,
            A.TimeMasking,
            A.Mosaic,
        },
    ),
)
def test_selective_channel(
    augmentation_cls: BasicTransform, params: dict[str, Any]
) -> None:
    image = SQUARE_MULTI_UINT8_IMAGE
    channels = [3, 2, 4]

    aug_cls = augmentation_cls(**params, p=1)

    aug = A.Compose(
        [
            A.SelectiveChannelTransform(
                transforms=[aug_cls], channels=channels, p=1
            )
        ],
        seed=137,
        strict=False,
    )

    data = {"image": image}

    transformed_image = aug(**data)["image"]

    for channel in range(image.shape[-1]):
        if channel in channels:
            assert not np.array_equal(
                image[..., channel], transformed_image[..., channel]
            )
        else:
            np.testing.assert_array_equal(
                image[..., channel], transformed_image[..., channel]
            )


@pytest.mark.parametrize(
    "params, expected",
    [
        # Default values
        (
            {},
            {
                "min_height": 1024,
                "min_width": 1024,
                "position": "center",
                "border_mode": cv2.BORDER_CONSTANT,
                "fill": 0,
                "fill_mask": 0,
            },
        ),
        # Boundary values
        ({"min_height": 800, "min_width": 800}, {"min_height": 800, "min_width": 800}),
        (
            {
                "pad_height_divisor": 10,
                "min_height": None,
                "pad_width_divisor": 10,
                "min_width": None,
            },
            {
                "pad_height_divisor": 10,
                "min_height": None,
                "pad_width_divisor": 10,
                "min_width": None,
            },
        ),
        ({"position": "top_left"}, {"position": "top_left"}),
        # Value handling when border_mode is BORDER_CONSTANT
        (
            {"border_mode": cv2.BORDER_CONSTANT, "fill": 255},
            {"border_mode": cv2.BORDER_CONSTANT, "fill": 255},
        ),
        (
            {"border_mode": cv2.BORDER_CONSTANT, "fill": (0, 0, 0)},
            {"border_mode": cv2.BORDER_CONSTANT, "fill": (0, 0, 0)},
        ),
        # Mask value handling
        (
            {"border_mode": cv2.BORDER_CONSTANT, "fill": (0, 0, 0), "fill_mask": 128},
            {"border_mode": cv2.BORDER_CONSTANT, "fill": (0, 0, 0), "fill_mask": 128},
        ),
    ],
)
def test_pad_if_needed_functionality(params, expected):
    # Setup the augmentation with the provided parameters
    aug = A.PadIfNeeded(**params, p=1)
    # Get the initialization arguments to check against expected
    aug_dict = {key: getattr(aug, key) for key in expected.keys()}

    # Assert each expected key/value pair
    for key, value in expected.items():
        assert aug_dict[key] == value, f"Failed on {key} with value {value}"



@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        except_augmentations={
            A.RandomSizedBBoxSafeCrop,
            A.RandomCropNearBBox,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.OverlayElements,
            A.MaskDropout,
            A.TextImage,
            A.Mosaic,
        },
    ),
)
def test_dual_transforms_methods(augmentation_cls, params):
    """Checks whether transformations based on DualTransform dont has abstract methods."""
    aug = augmentation_cls(p=1, **params)
    aug.set_random_seed(42)

    image = SQUARE_UINT8_IMAGE
    mask = cv2.randu(np.zeros((100, 100), dtype=np.uint8), 0, 4) * 64

    arg = {
        "images": np.stack([image] * 4),
        "mask": mask,
        "masks": np.stack([mask] * 2),
        "bboxes": np.array([[0, 0, 0.1, 0.1, 1]]),
        "keypoints": np.array([(0, 0, 1, 0, 0), (1, 1, 1, 0, 0)]),
    }

    for target in aug.targets:
        if target in arg:
            kwarg = {target: arg[target]}
            try:
                _ = aug(image=image.copy(), **kwarg)
            except Exception as e:
                if isinstance(e, NotImplementedError):
                    raise NotImplementedError(
                        f"{target} error at: {augmentation_cls},  {e}"
                    )
                raise e


@pytest.mark.parametrize(
    "px",
    [
        10,
        (10, 20),
        (-10, 20, -30, 40),
        ((10, 20), (20, 30), (30, 40), (40, 50)),
        ([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]),
        None,
    ],
)
@pytest.mark.parametrize(
    "percent",
    [
        0.1,
        (0.1, 0.2),
        (0.1, 0.2, 0.3, 0.4),
        ((-0.1, -0.2), (-0.2, -0.3), (0.3, 0.4), (0.4, 0.5)),
        (
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ),
        None,
    ],
)
@pytest.mark.parametrize(
    "fill",
    [
        0,
        (0, 255),
        [0, 255],
    ],
)
@pytest.mark.parametrize(
    "keep_size",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "sample_independently",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("image", IMAGES)
def test_crop_and_pad(px, percent, fill, keep_size, sample_independently, image):
    fill_mask = 255 if isinstance(fill, list) else fill

    interpolation = cv2.INTER_LINEAR
    border_mode = cv2.BORDER_CONSTANT
    if (px is None) == (percent is None):
        # Skip the test case where both px and percent are None or both are not None
        return

    transform = A.Compose(
        [
            A.CropAndPad(
                px=px,
                percent=percent,
                border_mode=border_mode,
                fill=fill,
                fill_mask=fill_mask,
                keep_size=keep_size,
                sample_independently=sample_independently,
                interpolation=interpolation,
                p=1,
            ),
        ],
        strict=True,
        seed=137,
    )

    transformed_image = transform(image=image)["image"]

    if keep_size:
        assert transformed_image.shape == image.shape

    assert transformed_image is not None
    assert transformed_image.shape[0] > 0
    assert transformed_image.shape[1] > 0
    assert transformed_image.shape[2] == image.shape[2]


@pytest.mark.parametrize(
    "percent, expected_shape",
    [
        (0.1, (12, 12, 3)),  # Padding 10% of image size on each side
        (-0.1, (8, 8, 3)),  # Cropping 10% of image size from each side
        (
            (0.1, 0.2, 0.3, 0.4),
            (14, 16, 3),
        ),  # Padding: top=10%, right=20%, bottom=30%, left=40%
        (
            (-0.1, -0.2, -0.3, -0.4),
            (6, 4, 3),
        ),  # Cropping: top=10%, right=20%, bottom=30%, left=40%
    ],
)
def test_crop_and_pad_percent(percent, expected_shape):
    transform = A.Compose(
        [
            A.CropAndPad(
                px=None,
                percent=percent,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                keep_size=False,
            )
        ],
        strict=True,
        seed=137,
    )

    image = np.ones((10, 10, 3), dtype=np.uint8)

    transformed_image = transform(image=image)["image"]

    assert transformed_image.shape == expected_shape
    if percent is not None and all(p >= 0 for p in np.array(percent).flatten()):
        assert transformed_image.sum() == image.sum()


@pytest.mark.parametrize(
    "px, expected_shape",
    [
        (2, (14, 14, 3)),  # Padding 2 pixels on each side
        (-2, (6, 6, 3)),  # Cropping 2 pixels from each side
        ((1, 2, 3, 4), (14, 16, 3)),  # Padding: top=1, right=2, bottom=3, left=4
        ((-1, -2, -3, -4), (6, 4, 3)),  # Cropping: top=1, right=2, bottom=3, left=4
    ],
)
def test_crop_and_pad_px_pixel_values(px, expected_shape):
    transform = A.Compose(
        [
            A.CropAndPad(
                px=px,
                percent=None,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                keep_size=False,
            )
        ],
        strict=True,
        seed=137,
    )

    image = np.ones((10, 10, 3), dtype=np.uint8) * 255

    transformed_image = transform(image=image)["image"]

    if isinstance(px, int):
        px = [px] * 4  # Convert to list of 4 elements
    if isinstance(px, tuple) and len(px) == 2:
        px = [px[0], px[1], px[0], px[1]]  # Convert to 4 elements for padding

    if px is not None:
        if all(p >= 0 for p in px):  # Padding
            pad_top, pad_right, pad_bottom, pad_left = px
            central_region = transformed_image[
                pad_top : pad_top + image.shape[0],
                pad_left : pad_left + image.shape[1],
                :,
            ]
            assert np.all(central_region == 255)
        elif all(p <= 0 for p in px):  # Cropping
            crop_top, crop_right, crop_bottom, crop_left = [-p for p in px]
            cropped_region = image[
                crop_top : image.shape[0] - crop_bottom,
                crop_left : image.shape[1] - crop_right,
                :,
            ]
            assert np.all(transformed_image == cropped_region)


@pytest.mark.parametrize(
    "params",
    [
        (
            {"fog_coef_range": (1.2, 1.5)}
        ),  # Invalid fog coefficient range -> upper bound
        ({"fog_coef_range": (0.9, 0.7)}),  # Invalid range  -> decreasing
    ],
)
def test_random_fog_invalid_input(params):
    with pytest.raises(Exception):
        A.RandomFog(**params)


@pytest.mark.parametrize("image", IMAGES + [np.full((100, 100), 128, dtype=np.uint8)])
@pytest.mark.parametrize("mean", (0, 0.1, -0.1))
def test_gauss_noise(mean, image):
    aug = A.GaussNoise(p=1, noise_scale_factor=1.0, mean_range=(mean, mean))
    aug.set_random_seed(42)

    apply_params = aug.get_params_dependent_on_data(
        params={"shape": image.shape},
        data={"image": image},
    )

    assert (
        np.abs(
            mean - apply_params["noise_map"].mean() / MAX_VALUES_BY_DTYPE[image.dtype]
        )
        < 0.5
    )
    result = A.Compose([aug], seed=137, strict=True)(image=image)

    assert not (result["image"] >= image).all()


@pytest.mark.parametrize(
    "params",
    [
        (
            {"flare_roi": (1.2, 0.2, 0.8, 0.9)}
        ),  # Invalid flare_roi -> x_min out of bounds
        (
            {"flare_roi": (0.2, -0.1, 0.8, 0.9)}
        ),  # Invalid flare_roi -> y_min out of bounds
        (
            {"flare_roi": (0.2, 0.3, 1.2, 0.9)}
        ),  # Invalid flare_roi -> x_max out of bounds
        (
            {"flare_roi": (0.2, 0.3, 0.8, 1.1)}
        ),  # Invalid flare_roi -> y_max out of bounds
        ({"flare_roi": (0.8, 0.2, 0.4, 0.9)}),  # Invalid flare_roi -> x_min > x_max
        ({"flare_roi": (0.2, 0.9, 0.8, 0.3)}),  # Invalid flare_roi -> y_min > y_max
        (
            {"angle_range": (1.2, 0.5)}
        ),  # Invalid angle range -> angle_upper out of bounds
        (
            {"angle_range": (0.5, 1.2)}
        ),  # Invalid angle range -> angle_upper out of bounds
        ({"angle_range": (0.7, 0.5)}),  # Invalid angle range -> non-decreasing
        (
            {"num_flare_circles_range": (12, 8)}
        ),  # Invalid num_flare_circles_range -> non-decreasing
        (
            {"num_flare_circles_range": (-1, 6)}
        ),  # Invalid num_flare_circles_range -> lower bound negative
    ],
)
def test_random_sun_flare_invalid_input(params):
    with pytest.raises(ValueError):
        A.RandomSunFlare(**params)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.OverlayElements,
            A.NoOp,
            A.Lambda,
        },
    ),
)
def test_return_nonzero(augmentation_cls, params):
    """Mistakes in clipping may lead to zero image, testing for that"""
    image = (
        SQUARE_FLOAT_IMAGE if augmentation_cls == A.FromFloat else SQUARE_UINT8_IMAGE
    )

    aug = A.Compose([augmentation_cls(**params, p=1)], seed=137, strict=False)

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
    elif augmentation_cls == A.ToRGB:
        data["image"] = cv2.randu(np.zeros((100, 100), dtype=np.uint8), 0, 255)
    elif augmentation_cls == A.MaskDropout:
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

    result = aug(**data)

    assert np.max(result["image"]) > 0


@pytest.mark.parametrize(
    "transform",
    [
        A.PadIfNeeded(
            min_height=6, min_width=6, fill=128, border_mode=cv2.BORDER_CONSTANT, p=1
        ),
        A.CropAndPad(
            px=2,
            border_mode=cv2.BORDER_CONSTANT,
            fill=128,
            p=1,
            interpolation=cv2.INTER_NEAREST_EXACT,
        ),
        A.CropAndPad(
            percent=(0, 0.3, 0, 0), fill=128, p=1, interpolation=cv2.INTER_NEAREST_EXACT
        ),
        A.Affine(
            translate_px={"x": -1, "y": -1},
            fill=128,
            p=1,
            interpolation=cv2.INTER_NEAREST,
        ),
        A.Rotate(
            p=1,
            limit=(45, 45),
            interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=128,
        ),
    ],
)
@pytest.mark.parametrize("num_channels", [1, 3, 5])
def test_padding_color(transform, num_channels):
    # Create an image with zeros
    if num_channels == 1:
        image = np.zeros((4, 4), dtype=np.uint8)
    else:
        image = np.zeros((4, 4, num_channels), dtype=np.uint8)

    pipeline = A.Compose([transform], seed=137, strict=True)

    # Apply the transform
    augmented = pipeline(image=image)["image"]

    # Check the unique values in each channel of the padded image
    if num_channels == 1:
        channels = [augmented]
    else:
        channels = [augmented[:, :, i] for i in range(num_channels)]

    for channel_id, channel in enumerate(channels):
        unique_values = np.unique(channel)
        assert set(unique_values) == {0, 128}, f"{channel_id}"


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.BBoxSafeRandomCrop,
            A.OverlayElements,
            A.GridElasticDeform,
            A.CropNonEmptyMaskIfExists,
        },
    ),
)
def test_empty_bboxes_keypoints(augmentation_cls, params):
    aug = A.Compose(
        [augmentation_cls(p=1, **params)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        keypoint_params=A.KeypointParams(format="xy"),
        seed=137,
        strict=False,
    )
    image = SQUARE_UINT8_IMAGE
    data = {
        "image": image,
        "bboxes": np.array([], dtype=np.float32).reshape(0, 4),
        "labels": [],
        "keypoints": np.array([], dtype=np.float32).reshape(0, 2),
    }

    if augmentation_cls == A.OverlayElements:
        data = {
            "image": image,
            "overlay_metadata": [],
        }
    elif augmentation_cls == A.MaskDropout:
        mask = np.zeros_like(image)[:, :, 0]
        mask[:20, :20] = 1
        data["mask"] = mask
    elif augmentation_cls == A.Mosaic:
        data["mosaic_metadata"] = [
            {
                "image": image,
            }
        ]

    data = aug(**data)

    np.testing.assert_array_equal(data["bboxes"], np.array([], dtype=np.float32).reshape(0, 4))
    np.testing.assert_array_equal(data["keypoints"], np.array([], dtype=np.float32).reshape(0, 2))


@pytest.mark.parametrize(
    "remove_invisible, expected_keypoints",
    [
        (True, np.array([], dtype=np.float32).reshape(0, 2)),
        (False, np.array([[10, 10]])),
    ],
)
def test_mask_dropout_bboxes(remove_invisible, expected_keypoints):
    image = SQUARE_UINT8_IMAGE
    mask = np.zeros_like(image)[:, :, 0]
    mask[:20, :20] = 1
    keypoints = np.array([[10, 10]])

    transform = A.Compose(
        [A.MaskDropout(p=1, max_objects=1, fill_mask=0, fill=1)],
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=remove_invisible
        ),
        seed=137,
        strict=True,
    )

    transformed = transform(image=image, mask=mask, keypoints=keypoints)
    np.testing.assert_array_equal(transformed["keypoints"], expected_keypoints)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_2d_transforms(
        custom_arguments={
        },
        except_augmentations={
            A.XYMasking,
            A.RandomSizedBBoxSafeCrop,
            A.RandomCropNearBBox,
            A.BBoxSafeRandomCrop,
            A.CropNonEmptyMaskIfExists,
            A.FDA,
            A.HistogramMatching,
            A.OverlayElements,
            A.MaskDropout,
            A.TextImage,
            A.VerticalFlip,
            A.HorizontalFlip,
            A.GridElasticDeform,
            A.PixelDistributionAdaptation,
            A.SafeRotate,
            A.Rotate,
            A.Affine,
            A.ShiftScaleRotate,
            A.D4,
            A.RandomRotate90,
            A.SquareSymmetry,
            A.PiecewiseAffine,
            A.Perspective,
            A.RandomGridShuffle,
            A.TimeReverse,
            A.Erasing,
            A.ConstrainedCoarseDropout,
            A.PixelDropout,
            A.CoarseDropout,
            A.ElasticTransform,
            A.GridDistortion,
            A.OpticalDistortion,
            A.ThinPlateSpline,
            A.Mosaic,
            A.FrequencyMasking,
        },
    ),
)
def test_keypoints_bboxes_match(augmentation_cls, params):
    """Checks whether transformations based on DualTransform dont has abstract methods."""
    aug = augmentation_cls(p=1, **params)

    image = RECTANGULAR_UINT8_IMAGE

    x_min, y_min, x_max, y_max = 40, 30, 50, 60

    bboxes = np.array([[x_min, y_min, x_max, y_max]])
    keypoints = np.array([[x_min, y_min], [x_max, y_max]])

    transform = A.Compose(
        [aug],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_area=0, min_visibility=0),
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        seed=137,
        strict=False,
    )

    transformed = transform(image=image, bboxes=bboxes, keypoints=keypoints, labels=[1])

    x_min_transformed, y_min_transformed, x_max_transformed, y_max_transformed = (
        transformed["bboxes"][0]
    )

    np.testing.assert_allclose(
        transformed["keypoints"][0], [x_min_transformed, y_min_transformed], atol=1
    )
    np.testing.assert_allclose(
        transformed["keypoints"][1], [x_max_transformed, y_max_transformed], atol=1.5
    )


@pytest.mark.parametrize(
    ["input_shape", "max_size", "max_size_hw", "expected_shape"],
    [
        # LongestMaxSize with max_size
        ((80, 60, 3), 40, None, (40, 30, 3)),  # landscape
        ((60, 80, 3), 40, None, (30, 40, 3)),  # portrait
        ((80, 80, 3), 40, None, (40, 40, 3)),  # square
        # LongestMaxSize with max_size_hw - both dimensions
        ((80, 60, 3), None, (40, 30), (40, 30, 3)),  # exact fit
        ((80, 60, 3), None, (30, 40), (30, 22, 3)),  # height constrains
        ((60, 80, 3), None, (40, 30), (22, 30, 3)),  # width constrains
        # LongestMaxSize with max_size_hw - single dimension
        ((80, 60, 3), None, (40, None), (40, 30, 3)),  # height only
        ((80, 60, 3), None, (None, 30), (40, 30, 3)),  # width only
    ],
)
def test_longest_max_size(input_shape, max_size, max_size_hw, expected_shape):
    image = np.zeros(input_shape, dtype=np.uint8)
    aug = A.LongestMaxSize(max_size=max_size, max_size_hw=max_size_hw)
    transformed = aug(image=image)["image"]
    assert transformed.shape == expected_shape


@pytest.mark.parametrize(
    ["input_shape", "max_size", "max_size_hw", "expected_shape"],
    [
        # SmallestMaxSize with max_size
        ((80, 60, 3), 40, None, (53, 40, 3)),  # landscape
        ((60, 80, 3), 40, None, (40, 53, 3)),  # portrait
        ((80, 80, 3), 40, None, (40, 40, 3)),  # square
        # SmallestMaxSize with max_size_hw - both dimensions
        ((80, 60, 3), None, (40, 30), (40, 30, 3)),  # height determines
        ((60, 80, 3), None, (30, 40), (30, 40, 3)),  # width determines
        ((80, 80, 3), None, (40, 40), (40, 40, 3)),  # square input
        # SmallestMaxSize with max_size_hw - single dimension
        ((80, 60, 3), None, (40, None), (40, 30, 3)),  # height only
        ((80, 60, 3), None, (None, 30), (40, 30, 3)),  # width only
    ],
)
def test_smallest_max_size(input_shape, max_size, max_size_hw, expected_shape):
    image = np.zeros(input_shape, dtype=np.uint8)
    aug = A.SmallestMaxSize(max_size=max_size, max_size_hw=max_size_hw)
    transformed = aug(image=image)["image"]
    assert transformed.shape == expected_shape
