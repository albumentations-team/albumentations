import numpy as np
import cv2
import pytest

from albumentations import Transpose, Rotate, ShiftScaleRotate, OpticalDistortion, GridDistortion, ElasticTransform, \
    VerticalFlip, HorizontalFlip, RandomSizedCrop, RandomScale, IAAPiecewiseAffine, IAAAffine, IAAPerspective, \
    LongestMaxSize, Resize
import albumentations.augmentations.functional as F


def test_transpose_both_image_and_mask():
    image = np.ones((8, 6, 3))
    mask = np.ones((8, 6))
    augmentation = Transpose(p=1)
    augmented = augmentation(image=image, mask=mask)
    assert augmented['image'].shape == (6, 8, 3)
    assert augmented['mask'].shape == (6, 8)


@pytest.mark.parametrize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = Rotate(limit=(45, 45), interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = F.rotate(image, 45, interpolation=interpolation, border_mode=cv2.BORDER_REFLECT_101)
    expected_mask = F.rotate(mask, 45, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(data['image'], expected_image)
    assert np.array_equal(data['mask'], expected_mask)


@pytest.mark.parametrize('interpolation', [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
def test_shift_scale_rotate_interpolation(interpolation):
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)
    aug = ShiftScaleRotate(shift_limit=(0.2, 0.2), scale_limit=(1.1, 1.1), rotate_limit=(45, 45),
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
    aug = OpticalDistortion(distort_limit=(0.05, 0.05), shift_limit=(0, 0), interpolation=interpolation, p=1)
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
    aug = GridDistortion(num_steps=1, distort_limit=(0.3, 0.3), interpolation=interpolation, p=1)
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
    aug = ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=interpolation, p=1)
    data = aug(image=image, mask=mask)
    expected_image = F.elastic_transform_fast(image, alpha=1, sigma=50, alpha_affine=50, interpolation=interpolation,
                                              border_mode=cv2.BORDER_REFLECT_101,
                                              random_state=np.random.RandomState(1111))
    expected_mask = F.elastic_transform_fast(mask, alpha=1, sigma=50, alpha_affine=50,
                                             interpolation=cv2.INTER_NEAREST,
                                             border_mode=cv2.BORDER_REFLECT_101,
                                             random_state=np.random.RandomState(1111))
    assert np.array_equal(data['image'], expected_image)
    assert np.array_equal(data['mask'], expected_mask)


@pytest.mark.parametrize(['augmentation_cls', 'params'], [
    [ElasticTransform, {}],
    [GridDistortion, {}],
    [ShiftScaleRotate, {'rotate_limit': 45}],
    [RandomScale, {'scale_limit': 0.5}],
    [RandomSizedCrop, {'min_max_height': (80, 90), 'height': 100, 'width': 100}],
    [LongestMaxSize, {'max_size': 50}],
    [Rotate, {}],
    [OpticalDistortion, {}],
    [IAAAffine, {'scale': 1.5}],
    [IAAPiecewiseAffine, {'scale': 1.5}],
    [IAAPerspective, {}],
])
def test_binary_mask_interpolation(augmentation_cls, params):
    """Checks whether transformations based on DualTransform does not introduce a mask interpolation artifacts"""
    aug = augmentation_cls(p=1, **params)
    image = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)

    data = aug(image=image, mask=mask)
    assert np.array_equal(np.unique(data['mask']), np.array([0, 1]))


@pytest.mark.parametrize(['augmentation_cls', 'params'], [
    [ElasticTransform, {}],
    [GridDistortion, {}],
    [ShiftScaleRotate, {'rotate_limit': 45}],
    [RandomScale, {'scale_limit': 0.5}],
    [RandomSizedCrop, {'min_max_height': (80, 90), 'height': 100, 'width': 100}],
    [LongestMaxSize, {'max_size': 50}],
    [Rotate, {}],
    [Resize, {'height': 80, 'width': 90}],
    [Resize, {'height': 120, 'width': 130}],
    [OpticalDistortion, {}]
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
