import numpy as np
from numpy.random import RandomState
import pytest

import albumentations as A


TEST_SEEDS = (0, 1, 42, 111, 9999)


@pytest.mark.parametrize(['augmentation_cls', 'params'], [
    [A.ChannelShuffle, {}],
    [A.GaussNoise, {}],
    [A.Cutout, {}],
    [A.CoarseDropout, {}],
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
    [A.ChannelDropout, {}]
])
@pytest.mark.parametrize('seed', TEST_SEEDS)
def test_repeatability(augmentation_cls, seed, params):
    aug = augmentation_cls(**params)
    random_state_1 = np.random.RandomState(seed)
    random_state_2 = np.random.RandomState(seed)
    for i in range(10):
        image = np.random.randint(low=0, high=256, size=(255, 255, 3), dtype=np.uint8)
        res_1 = aug(random_state=random_state_1, image=image)
        res_2 = aug(random_state=random_state_2, image=image)
        assert np.array_equal(res_1['image'], res_2['image'])


@pytest.mark.parametrize('seed', TEST_SEEDS)
def test_pipeline_repeatability(seed, image, mask):
    aug = A.Compose([
        A.OneOrOther(
            A.Compose([
                A.Resize(1024, 1024),
                A.RandomSizedCrop(min_max_height=(256, 1024), height=512, width=512, p=1),
                A.OneOf([
                    A.RandomSizedCrop(min_max_height=(256, 512), height=384, width=384, p=0.5),
                    A.RandomSizedCrop(min_max_height=(256, 512), height=512, width=512, p=0.5),
                ])
            ]),
            A.Compose([
                A.Resize(1024, 1024),
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
    random_state_1 = RandomState(seed)
    random_state_2 = RandomState(seed)
    for i in range(10):
        aug_data_1 = aug(random_state=random_state_1, image=image, mask=mask)
        aug_data_2 = aug(random_state=random_state_2, image=image, mask=mask)
        assert np.array_equal(aug_data_1['image'], aug_data_2['image'])
        assert np.array_equal(aug_data_1['mask'], aug_data_2['mask'])

@pytest.mark.parametrize(['augmentation_cls', 'params'], [
    [A.GaussNoise, {}],
    [A.Cutout, {}],
    [A.CoarseDropout, {}],
    [A.JpegCompression, {'quality_lower': 50, 'quality_upper': 100}],
    [A.HueSaturationValue, {}],
    [A.RGBShift, {}],
    [A.RandomBrightnessContrast, {}],
    [A.Blur, {'blur_limit': 50}],
    [A.MotionBlur, {}],
    [A.CLAHE, {}],
    [A.RandomGamma, {}],
    [A.Rotate, {}],
    [A.OpticalDistortion, {}],
    [A.GridDistortion, {}],
    [A.ElasticTransform, {}]
])
@pytest.mark.parametrize('seed', TEST_SEEDS)
def test_randomness(augmentation_cls, seed, params):
    aug = augmentation_cls(**params)
    gen_state = np.random.RandomState(seed)
    random_state_1 = RandomState(gen_state.randint(0, 65536))
    random_state_2 = RandomState(gen_state.randint(0, 65536))
    image = np.random.randint(low=0, high=256, size=(255, 255, 3), dtype=np.uint8)
    res_1 = aug(force_apply=True, random_state=random_state_1, image=image.copy())
    res_2 = aug(force_apply=True, random_state=random_state_2, image=image.copy())
    assert not np.array_equal(res_1['image'], res_2['image'])