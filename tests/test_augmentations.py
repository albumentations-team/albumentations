import pytest
import numpy as np

try:
    import torch
    import torchvision
    from albumentations.torch import ToTensor
    torch_available = True
except ImportError:
    torch_available = False

from albumentations import RandomCrop, PadIfNeeded, VerticalFlip, HorizontalFlip, Flip, Transpose, RandomRotate90, \
    Rotate, ShiftScaleRotate, CenterCrop, OpticalDistortion, GridDistortion, ElasticTransform, ToGray, RandomGamma, \
    JpegCompression, HueSaturationValue, RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, \
    GaussNoise, CLAHE, ChannelShuffle, InvertImg, IAAEmboss, IAASuperpixels, IAASharpen, IAAAdditiveGaussianNoise, \
    IAAPiecewiseAffine, IAAPerspective


@pytest.mark.parametrize(['augmentation_cls', 'params'], [
    [JpegCompression, {}],
    [HueSaturationValue, {}],
    [RGBShift, {}],
    [RandomBrightness, {}],
    [RandomContrast, {}],
    [Blur, {}],
    [MotionBlur, {}],
    [MedianBlur, {}],
    [GaussNoise, {}],
    [CLAHE, {}],
    [ChannelShuffle, {}],
    [InvertImg, {}],
    [RandomGamma, {}],
    [ToGray, {}],
])
def test_image_only_augmentations(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image, mask=mask)
    assert data['image'].dtype == np.uint8
    assert data['mask'].dtype == np.uint8
    assert np.array_equal(data['mask'], mask)


@pytest.mark.parametrize(['augmentation_cls', 'params'], [
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
    [CenterCrop, {'height': 10, 'width': 10}],
    [RandomCrop, {'height': 10, 'width': 10}],
])
def test_dual_augmentations(augmentation_cls, params, image, mask):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image, mask=mask)
    assert data['image'].dtype == np.uint8
    assert data['mask'].dtype == np.uint8


@pytest.mark.parametrize('augmentation_cls', [IAAEmboss, IAASuperpixels, IAASharpen, IAAAdditiveGaussianNoise])
def test_imgaug_image_only_augmentations(augmentation_cls, image, mask):
    aug = augmentation_cls(p=1)
    data = aug(image=image, mask=mask)
    assert data['image'].dtype == np.uint8
    assert data['mask'].dtype == np.uint8
    assert np.array_equal(data['mask'], mask)


@pytest.mark.parametrize('augmentation_cls', [IAAPiecewiseAffine, IAAPerspective])
def test_imgaug_dual_augmentations(augmentation_cls, image, mask):
    aug = augmentation_cls(p=1)
    data = aug(image=image, mask=mask)
    assert data['image'].dtype == np.uint8
    assert data['mask'].dtype == np.uint8


@pytest.mark.skipif(not torch_available, reason='PyTorch and torchvision are not available')
def test_torch_to_tensor_augmentations(image, mask):
    aug = ToTensor()
    data = aug(image=image, mask=mask)
    assert data['image'].dtype == torch.float32
    assert data['mask'].dtype == torch.float32
