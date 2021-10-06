from functools import partial

import cv2
import numpy as np
import pytest
import random

import albumentations as A
import albumentations.augmentations.functional as F
import albumentations.augmentations.geometric.functional as FGeometric

from .utils import get_transforms, get_image_only_transforms, get_dual_transforms


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def test_transpose_both_image_and_mask():
    image = np.ones((8, 6, 3))
    mask = np.ones((8, 6))
    augmentation = A.Transpose(p=1)
    augmented = augmentation(image=image, mask=mask)
    assert augmented["image"].shape == (6, 8, 3)
    assert augmented["mask"].shape == (6, 8)


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_safe_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = A.SafeRotate(limit=(45, 45), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = FGeometric.safe_rotate(image, 45, interpolation=interpolation, border_mode=cv2.BORDER_REFLECT_101)
    expected_mask = FGeometric.safe_rotate(
        mask, 45, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101
    )
    assert np.array_equal(data["image"], expected_image)
    assert np.array_equal(data["mask"], expected_mask)


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


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_shift_scale_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = A.ShiftScaleRotate(
        shift_limit=(0.2, 0.2), scale_limit=(1.1, 1.1), rotate_limit=(45, 45), interpolation=interpolation, p=1
    )
    data = aug(image=image, mask=mask)
    expected_image = FGeometric.shift_scale_rotate(
        image, angle=45, scale=2.1, dx=0.2, dy=0.2, interpolation=interpolation, border_mode=cv2.BORDER_REFLECT_101
    )
    expected_mask = FGeometric.shift_scale_rotate(
        mask, angle=45, scale=2.1, dx=0.2, dy=0.2, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101
    )
    assert np.array_equal(data["image"], expected_image)
    assert np.array_equal(data["mask"], expected_mask)


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_optical_distortion_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = A.OpticalDistortion(distort_limit=(0.05, 0.05), shift_limit=(0, 0), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = F.optical_distortion(
        image, k=0.05, dx=0, dy=0, interpolation=interpolation, border_mode=cv2.BORDER_REFLECT_101
    )
    expected_mask = F.optical_distortion(
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
    expected_image = F.grid_distortion(
        image, num_steps=1, xsteps=[1.3], ysteps=[1.3], interpolation=interpolation, border_mode=cv2.BORDER_REFLECT_101
    )
    expected_mask = F.grid_distortion(
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
    monkeypatch.setattr(
        "albumentations.augmentations.geometric.ElasticTransform.get_params", lambda *_: {"random_state": 1111}
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
        random_state=np.random.RandomState(1111),
    )
    expected_mask = FGeometric.elastic_transform(
        mask,
        alpha=1,
        sigma=50,
        alpha_affine=50,
        interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        random_state=np.random.RandomState(1111),
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
        },
        except_augmentations={A.RandomCropNearBBox, A.RandomSizedBBoxSafeCrop},
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
        },
        except_augmentations={A.RandomCropNearBBox, A.RandomSizedBBoxSafeCrop, A.CropAndPad},
    ),
)
def test_semantic_mask_interpolation(augmentation_cls, params):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts.
    Note: IAAAffine, IAAPiecewiseAffine, IAAPerspective does not properly operate if mask has values other than {0;1}
    """
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
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
        },
        except_augmentations={
            A.RandomCropNearBBox,
            A.RandomSizedBBoxSafeCrop,
            A.CropNonEmptyMaskIfExists,
            A.FDA,
            A.HistogramMatching,
            A.PixelDistributionAdaptation,
            A.MaskDropout,
        },
    ),
)
def test_multiprocessing_support(augmentation_cls, params, multiprocessing_context):
    """Checks whether we can use augmentations in multiprocessing environments"""
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)

    pool = multiprocessing_context.Pool(8)
    pool.map(__test_multiprocessing_support_proc, map(lambda x: (x, aug), [image] * 100))
    pool.close()
    pool.join()


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
    ),
)
def test_additional_targets_for_image_only(augmentation_cls, params):
    aug = A.Compose([augmentation_cls(always_apply=True, **params)], additional_targets={"image2": "image"})
    for _i in range(10):
        image1 = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        image2 = image1.copy()
        res = aug(image=image1, image2=image2)
        aug1 = res["image"]
        aug2 = res["image2"]
        assert np.array_equal(aug1, aug2)


def test_lambda_transform():
    def negate_image(image, **kwargs):
        return -image

    def one_hot_mask(mask, num_channels, **kwargs):
        new_mask = np.eye(num_channels, dtype=np.uint8)[mask]
        return new_mask

    def vflip_bbox(bbox, **kwargs):
        return F.bbox_vflip(bbox, **kwargs)

    def vflip_keypoint(keypoint, **kwargs):
        return F.keypoint_vflip(keypoint, **kwargs)

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
    assert output["bboxes"] == [F.bbox_vflip((10, 15, 25, 35), 10, 10)]
    assert output["keypoints"] == [F.keypoint_vflip((20, 30, 40, 50), 10, 10)]


def test_channel_droput():
    img = np.ones((10, 10, 3), dtype=np.float32)

    aug = A.ChannelDropout(channel_drop_range=(1, 1), always_apply=True)  # Drop one channel

    transformed = aug(image=img)["image"]

    assert sum(transformed[:, :, c].max() for c in range(img.shape[2])) == 2

    aug = A.ChannelDropout(channel_drop_range=(2, 2), always_apply=True)  # Drop two channels
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

    def mask_func(image, test):  # skipcq: PYL-W0613
        return mask

    aug = A.Equalize(mask=mask_func, mask_params=["test"], p=1)
    assert np.all(aug(image=img, test=mask)["image"] == F.equalize(img, mask=mask))


def test_crop_non_empty_mask():
    def _test_crop(mask, crop, aug, n=1):
        for _ in range(n):
            augmented = aug(image=mask, mask=mask)
            np.testing.assert_array_equal(augmented["image"], crop)
            np.testing.assert_array_equal(augmented["mask"], crop)

    # test general case
    mask_1 = np.zeros([10, 10])
    mask_1[0, 0] = 1
    crop_1 = np.array([[1]])
    aug_1 = A.CropNonEmptyMaskIfExists(1, 1)

    # test empty mask
    mask_2 = np.zeros([10, 10])
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


@pytest.mark.parametrize("interpolation", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_downscale(interpolation):
    img_float = np.random.rand(100, 100, 3)
    img_uint = (img_float * 255).astype("uint8")

    aug = A.Downscale(scale_min=0.5, scale_max=0.5, interpolation=interpolation, always_apply=True)

    for img in (img_float, img_uint):
        transformed = aug(image=img)["image"]
        func_applied = F.downscale(img, scale=0.5, interpolation=interpolation)
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
    aug = A.MultiplicativeNoise(m, p=1)
    result = aug(image=image)["image"]
    image = F.clip(image * m, image.dtype, F.MAX_VALUES_BY_DTYPE[image.dtype])
    assert np.allclose(image, result)

    aug = A.MultiplicativeNoise(elementwise=True, p=1)
    params = aug.get_params_dependent_on_targets({"image": image})
    mul = params["multiplier"]
    assert mul.shape == image.shape
    result = aug.apply(image, mul)
    dtype = image.dtype
    image = image.astype(np.float32) * mul
    image = F.clip(image, dtype, F.MAX_VALUES_BY_DTYPE[dtype])
    assert np.allclose(image, result)


@pytest.mark.parametrize(
    "image", [np.random.randint(0, 256, [256, 320, 3], np.uint8), np.random.random([256, 320, 3]).astype(np.float32)]
)
def test_multiplicative_noise_rgb(image):
    dtype = image.dtype

    m = 0.5
    aug = A.MultiplicativeNoise(m, p=1)
    result = aug(image=image)["image"]
    image = F.clip(image * m, dtype, F.MAX_VALUES_BY_DTYPE[dtype])
    assert np.allclose(image, result)

    aug = A.MultiplicativeNoise(elementwise=True, p=1)
    params = aug.get_params_dependent_on_targets({"image": image})
    mul = params["multiplier"]
    assert mul.shape == image.shape[:2] + (1,)
    result = aug.apply(image, mul)
    image = F.clip(image.astype(np.float32) * mul, dtype, F.MAX_VALUES_BY_DTYPE[dtype])
    assert np.allclose(image, result)

    aug = A.MultiplicativeNoise(per_channel=True, p=1)
    params = aug.get_params_dependent_on_targets({"image": image})
    mul = params["multiplier"]
    assert mul.shape == (3,)
    result = aug.apply(image, mul)
    image = F.clip(image.astype(np.float32) * mul, dtype, F.MAX_VALUES_BY_DTYPE[dtype])
    assert np.allclose(image, result)

    aug = A.MultiplicativeNoise(elementwise=True, per_channel=True, p=1)
    params = aug.get_params_dependent_on_targets({"image": image})
    mul = params["multiplier"]
    assert mul.shape == image.shape
    result = aug.apply(image, mul)
    image = F.clip(image.astype(np.float32) * mul, image.dtype, F.MAX_VALUES_BY_DTYPE[image.dtype])
    assert np.allclose(image, result)


def test_mask_dropout():
    # In this case we have mask with all ones, so MaskDropout wipe entire mask and image
    img = np.random.randint(0, 256, [50, 10], np.uint8)
    mask = np.ones([50, 10], dtype=np.long)

    aug = A.MaskDropout(p=1)
    result = aug(image=img, mask=mask)
    assert np.all(result["image"] == 0)
    assert np.all(result["mask"] == 0)

    # In this case we have mask with zeros , so MaskDropout will make no changes
    img = np.random.randint(0, 256, [50, 10], np.uint8)
    mask = np.zeros([50, 10], dtype=np.long)

    aug = A.MaskDropout(p=1)
    result = aug(image=img, mask=mask)
    assert np.all(result["image"] == img)
    assert np.all(result["mask"] == 0)


@pytest.mark.parametrize(
    "image", [np.random.randint(0, 256, [256, 320, 3], np.uint8), np.random.random([256, 320, 3]).astype(np.float32)]
)
def test_grid_dropout_mask(image):
    mask = np.ones([256, 320], dtype=np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=0)
    result = aug(image=image, mask=mask)
    # with mask on ones and fill_value = 0 the sum of pixels is smaller
    assert result["image"].sum() < image.sum()
    assert result["image"].shape == image.shape
    assert result["mask"].sum() < mask.sum()
    assert result["mask"].shape == mask.shape

    # with mask of zeros and fill_value = 0 mask should not change
    mask = np.zeros([256, 320], dtype=np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=0)
    result = aug(image=image, mask=mask)
    assert result["image"].sum() < image.sum()
    assert np.all(result["mask"] == 0)

    # with mask mask_fill_value=100, mask sum is larger
    mask = np.random.randint(0, 10, [256, 320], np.uint8)
    aug = A.GridDropout(p=1, mask_fill_value=100)
    result = aug(image=image, mask=mask)
    assert result["image"].sum() < image.sum()
    assert result["mask"].sum() > mask.sum()

    # with mask mask_fill_value=None, mask is not changed
    mask = np.ones([256, 320], dtype=np.uint8)
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
        (0.00004, None, None, 2, 100, None, None),
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


def test_gauss_noise_incorrect_var_limit_type():
    with pytest.raises(TypeError) as exc_info:
        A.GaussNoise(var_limit={"low": 70, "high": 90})
    message = "Expected var_limit type to be one of (int, float, tuple, list), got <class 'dict'>"
    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ["blur_limit", "sigma", "result_blur", "result_sigma"],
    [
        [[0, 0], [1, 1], 0, 1],
        [[1, 1], [0, 0], 1, 0],
        [[1, 1], [1, 1], 1, 1],
        [[0, 0], [0, 0], 3, 0],
        [[0, 3], [0, 0], 3, 0],
        [[0, 3], [0.1, 0.1], 3, 0.1],
    ],
)
def test_gaus_blur_limits(blur_limit, sigma, result_blur, result_sigma):
    img = np.zeros([100, 100, 3], dtype=np.uint8)

    aug = A.Compose([A.GaussianBlur(blur_limit=blur_limit, sigma_limit=sigma, p=1)])

    res = aug(image=img)["image"]
    assert np.allclose(res, F.gaussian_blur(img, result_blur, result_sigma))


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
    img = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

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
        assert _max <= 10, "Max: {}".format(_max)
    else:
        assert _max <= 2, "Max: {}".format(_max)


@pytest.mark.parametrize(["hue", "sat", "val"], [[13, 17, 23], [14, 18, 24], [131, 143, 151], [132, 144, 152]])
def test_hue_saturation_value_float_uint8_equal(hue, sat, val):
    img = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

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

            _max = np.abs(res1.astype(np.int) - res2).max()
            assert _max <= 10, "Max value: {}".format(_max)


def test_shift_scale_separate_shift_x_shift_y(image, mask):
    aug = A.ShiftScaleRotate(shift_limit=(0.3, 0.3), shift_limit_y=(0.4, 0.4), scale_limit=0, rotate_limit=0, p=1)
    data = aug(image=image, mask=mask)
    expected_image = FGeometric.shift_scale_rotate(
        image, angle=0, scale=1, dx=0.3, dy=0.4, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101
    )
    expected_mask = FGeometric.shift_scale_rotate(
        mask, angle=0, scale=1, dx=0.3, dy=0.4, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101
    )
    assert np.array_equal(data["image"], expected_image)
    assert np.array_equal(data["mask"], expected_mask)


@pytest.mark.parametrize(["val_uint8"], [[0], [1], [128], [255]])
def test_glass_blur_float_uint8_diff_less_than_two(val_uint8):

    x_uint8 = np.zeros((5, 5)).astype(np.uint8)
    x_uint8[2, 2] = val_uint8

    x_float32 = np.zeros((5, 5)).astype(np.float32)
    x_float32[2, 2] = val_uint8 / 255.0

    glassblur = A.GlassBlur(always_apply=True, max_delta=1)

    np.random.seed(0)
    blur_uint8 = glassblur(image=x_uint8)["image"]

    np.random.seed(0)
    blur_float32 = glassblur(image=x_float32)["image"]

    # Before comparison, rescale the blur_float32 to [0, 255]
    diff = np.abs(blur_uint8 - blur_float32 * 255)

    # The difference between the results of float32 and uint8 will be at most 2.
    assert np.all(diff <= 2.0)


def test_perspective_keep_size():
    h, w = 100, 100
    img = np.zeros([h, w, 3], dtype=np.uint8)
    h, w = img.shape[:2]
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
        [A.Perspective(keep_size=False, p=1), A.Resize(h, w)],
        keypoint_params=A.KeypointParams("xys"),
        bbox_params=A.BboxParams("pascal_voc", label_fields=["labels"]),
    )

    set_seed()
    res_1 = transform_1(image=img, bboxes=bboxes, keypoints=keypoints, labels=[0] * len(bboxes))
    set_seed()
    res_2 = transform_2(image=img, bboxes=bboxes, keypoints=keypoints, labels=[0] * len(bboxes))

    assert np.allclose(res_1["bboxes"], res_2["bboxes"])
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
        (0.5, 0.5, A.RandomSizedCrop((50, 200), 513, 450, always_apply=True), (513, 450), (224, 224)),
        (0.3, 0.5, A.RandomResizedCrop(513, 450, always_apply=True), (513, 450), (224, 224)),
        (1.0, 0.5, A.CenterCrop(500, 450, always_apply=True), (500, 450, 3), (512, 512, 3)),
        (0.5, 0.8, A.Resize(513, 450, always_apply=True), (513, 450), (512, 512)),
        (0.5, 0.2, A.NoOp(), (224, 224), (224, 224)),
        (0.5, 0.9, A.NoOp(), (512, 512, 3), (512, 512, 3)),
        (0.5, 0.5, None, (512, 512), (512, 512)),
        (0.8, 0.7, None, (512, 512, 3), (512, 512, 3)),
        (
            0.5,
            0.5,
            A.Compose([A.Blur(), A.RandomSizedCrop((50, 200), 512, 512, always_apply=True), A.HorizontalFlip()]),
            (512, 512),
            (512, 512),
        ),
    ],
)
def test_template_transform(image, img_weight, template_weight, template_transform, image_size, template_size):
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
        transform = A.TemplateTransform(template, always_apply=True)
        transform(image=image)

    message = "Image and template must be the same size, got {} and {}".format(image.shape[:2], template.shape[:2])
    assert str(exc_info.value) == message


@pytest.mark.parametrize(["img_channels", "template_channels"], [(1, 3), (6, 3)])
def test_template_transform_incorrect_channels(img_channels, template_channels):
    img = np.random.randint(0, 256, [512, 512, img_channels], np.uint8)
    template = np.random.randint(0, 256, [512, 512, template_channels], np.uint8)

    with pytest.raises(ValueError) as exc_info:
        transform = A.TemplateTransform(template, always_apply=True)
        transform(image=img)

    message = (
        "Template must be a single channel or has the same number of channels "
        "as input image ({}), got {}".format(img_channels, template.shape[-1])
    )
    assert str(exc_info.value) == message
