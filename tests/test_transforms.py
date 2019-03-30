from multiprocessing.pool import Pool

import numpy as np
import cv2
import pytest

import albumentations as A
import albumentations.augmentations.functional as F
from .conftest import skip_appveyor


def test_transpose_both_image_and_mask():
    image = np.ones((8, 6, 3))
    mask = np.ones((8, 6))
    augmentation = A.Transpose(p=1)
    augmented = augmentation(image=image, mask=mask)
    assert augmented['image'].shape == (6, 8, 3)
    assert augmented['mask'].shape == (6, 8)


@pytest.mark.parametrize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = A.Rotate(limit=(45, 45), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = F.rotate(image, 45, interpolation=interpolation, border_mode=cv2.BORDER_REFLECT_101)
    expected_mask = F.rotate(mask, 45, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(data['image'], expected_image)
    assert np.array_equal(data['mask'], expected_mask)


@pytest.mark.parametrize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_shift_scale_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = A.ShiftScaleRotate(shift_limit=(0.2, 0.2), scale_limit=(1.1, 1.1), rotate_limit=(45, 45),
                             interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = F.shift_scale_rotate(image, angle=45, scale=2.1, dx=0.2, dy=0.2, interpolation=interpolation,
                                          border_mode=cv2.BORDER_REFLECT_101)
    expected_mask = F.shift_scale_rotate(mask, angle=45, scale=2.1, dx=0.2, dy=0.2, interpolation=cv2.INTER_NEAREST,
                                         border_mode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(data['image'], expected_image)
    assert np.array_equal(data['mask'], expected_mask)


@pytest.mark.parametrize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_optical_distortion_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = A.OpticalDistortion(distort_limit=(0.05, 0.05), shift_limit=(0, 0), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = F.optical_distortion(image, k=0.05, dx=0, dy=0, interpolation=interpolation,
                                          border_mode=cv2.BORDER_REFLECT_101)
    expected_mask = F.optical_distortion(mask, k=0.05, dx=0, dy=0, interpolation=cv2.INTER_NEAREST,
                                         border_mode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(data['image'], expected_image)
    assert np.array_equal(data['mask'], expected_mask)


@pytest.mark.parametrize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_grid_distortion_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = A.GridDistortion(num_steps=1, distort_limit=(0.3, 0.3), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = F.grid_distortion(image, num_steps=1, xsteps=[1.3], ysteps=[1.3], interpolation=interpolation,
                                       border_mode=cv2.BORDER_REFLECT_101)
    expected_mask = F.grid_distortion(mask, num_steps=1, xsteps=[1.3], ysteps=[1.3], interpolation=cv2.INTER_NEAREST,
                                      border_mode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(data['image'], expected_image)
    assert np.array_equal(data['mask'], expected_mask)


@pytest.mark.parametrize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_elastic_transform_interpolation(monkeypatch, interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    monkeypatch.setattr('albumentations.augmentations.transforms.ElasticTransform.get_params',
                        lambda *_: {'random_state': 1111})
    aug = A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = F.elastic_transform(image, alpha=1, sigma=50, alpha_affine=50, interpolation=interpolation,
                                         border_mode=cv2.BORDER_REFLECT_101,
                                         random_state=np.random.RandomState(1111))
    expected_mask = F.elastic_transform(mask, alpha=1, sigma=50, alpha_affine=50,
                                        interpolation=cv2.INTER_NEAREST,
                                        border_mode=cv2.BORDER_REFLECT_101,
                                        random_state=np.random.RandomState(1111))
    assert np.array_equal(data['image'], expected_image)
    assert np.array_equal(data['mask'], expected_mask)


@pytest.mark.parametrize(['augmentation_cls', 'params'], [
    [A.ElasticTransform, {}],
    [A.GridDistortion, {}],
    [A.ShiftScaleRotate, {'rotate_limit': 45}],
    [A.RandomScale, {'scale_limit': 0.5}],
    [A.RandomSizedCrop, {'min_max_height': (80, 90), 'height': 100, 'width': 100}],
    [A.RandomSizedBBoxSafeCrop, {'height': 100, 'width': 100}],
    [A.LongestMaxSize, {'max_size': 50}],
    [A.Rotate, {}],
    [A.OpticalDistortion, {}],
    [A.IAAAffine, {'scale': 1.5}],
    [A.IAAPiecewiseAffine, {'scale': 1.5}],
    [A.IAAPerspective, {}],
])
def test_binary_mask_interpolation(augmentation_cls, params):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts"""
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    data = aug(image=image, mask=mask)
    assert np.array_equal(np.unique(data['mask']), np.array([0, 1]))


@pytest.mark.parametrize(['augmentation_cls', 'params'], [
    [A.ElasticTransform, {}],
    [A.GridDistortion, {}],
    [A.ShiftScaleRotate, {'rotate_limit': 45}],
    [A.RandomScale, {'scale_limit': 0.5}],
    [A.RandomSizedCrop, {'min_max_height': (80, 90), 'height': 100, 'width': 100}],
    [A.RandomSizedBBoxSafeCrop, {'height': 100, 'width': 100}],
    [A.LongestMaxSize, {'max_size': 50}],
    [A.Rotate, {}],
    [A.Resize, {'height': 80, 'width': 90}],
    [A.Resize, {'height': 120, 'width': 130}],
    [A.OpticalDistortion, {}]
])
def test_semantic_mask_interpolation(augmentation_cls, params):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts.
    Note: IAAAffine, IAAPiecewiseAffine, IAAPerspective does not properly operate if mask has values other than {0;1}
    """
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=4, size=(100, 100), dtype=np.uint8) * 64

    data = aug(image=image, mask=mask)
    assert np.array_equal(np.unique(data['mask']), np.array([0, 64, 128, 192]))


def __test_multiprocessing_support_proc(args):
    x, transform = args
    return transform(image=x)


@pytest.mark.parametrize(['augmentation_cls', 'params'], [
    [A.ElasticTransform, {}],
    [A.GridDistortion, {}],
    [A.ShiftScaleRotate, {'rotate_limit': 45}],
    [A.RandomScale, {'scale_limit': 0.5}],
    [A.RandomSizedCrop, {'min_max_height': (80, 90), 'height': 100, 'width': 100}],
    [A.RandomSizedBBoxSafeCrop, {'height': 100, 'width': 100}],
    [A.LongestMaxSize, {'max_size': 50}],
    [A.Rotate, {}],
    [A.OpticalDistortion, {}],
    [A.IAAAffine, {'scale': 1.5}],
    [A.IAAPiecewiseAffine, {'scale': 1.5}],
    [A.IAAPerspective, {}],
    [A.IAASharpen, {}]
])
@skip_appveyor
def test_multiprocessing_support(augmentation_cls, params):
    """Checks whether we can use augmetnations in multi-threaded environments"""
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)

    pool = Pool(8)
    pool.map(__test_multiprocessing_support_proc, map(lambda x: (x, aug), [image] * 100))


def test_force_apply():
    """
    Unit test for https://github.com/albu/albumentations/issues/189
    """
    aug = A.Compose([
        A.OneOrOther(
            A.Compose([
                A.RandomSizedCrop(min_max_height=(256, 1025), height=512, width=512, p=1),
                A.OneOf([
                    A.RandomSizedCrop(min_max_height=(256, 512), height=384, width=384, p=0.5),
                    A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=0.5),
                ])
            ]),
            A.Compose([
                A.RandomSizedCrop(min_max_height=(256, 1025), height=256, width=256, p=1),
                A.OneOf([
                    A.HueSaturationValue(p=0.5),
                    A.RGBShift(p=0.7)
                ], p=1),
            ])
        ),
        A.HorizontalFlip(p=1),
        A.RandomBrightnessContrast(p=0.5)
    ])

    res = aug(image=np.zeros((1248, 1248, 3), dtype=np.uint8))
    assert res['image'].shape[0] in (256, 384, 512)
    assert res['image'].shape[1] in (256, 384, 512)


@pytest.mark.parametrize(['augmentation_cls', 'params'], [
    [A.ChannelShuffle, {}],
    [A.GaussNoise, {}],
    [A.Cutout, {}],
    [A.JpegCompression, {}],
    [A.HueSaturationValue, {}],
    [A.RGBShift, {}],
    [A.RandomBrightnessContrast, {}],
    [A.Blur, {}],
    [A.MotionBlur, {}],
    [A.MedianBlur, {}],
    [A.CLAHE, {}],
    [A.InvertImg, {}],
    [A.RandomGamma, {}],
    [A.ToGray, {}],
    [A.VerticalFlip, {}],
    [A.HorizontalFlip, {}],
    [A.Flip, {}],
    [A.Transpose, {}],
    [A.RandomRotate90, {}],
    [A.Rotate, {}],
    [A.OpticalDistortion, {}],
    [A.GridDistortion, {}],
    [A.ElasticTransform, {}],
    [A.Normalize, {}],
    [A.ToFloat, {}],
    [A.FromFloat, {}],
])
def test_additional_targets_for_image_only(augmentation_cls, params):
    aug = A.Compose(
        [augmentation_cls(always_apply=True, **params)], additional_targets={'image2': 'image'})
    for i in range(10):
        image1 = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        image2 = image1.copy()
        res = aug(image=image1, image2=image2)
        aug1 = res['image']
        aug2 = res['image2']
        assert np.array_equal(aug1, aug2)
