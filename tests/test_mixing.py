import numpy as np
import pytest
import math

import albumentations as A
from .test_functional_mixing import find_mix_coef

def image_generator():
    yield {"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)}

def complex_image_generator():
    height = 100
    width = 100
    yield {"image": (height, width)}

def complex_read_fn_image(x):
    return {"image": np.random.randint(0, 256, (x["image"][0], x["image"][1], 3), dtype=np.uint8)}


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
             [(A.MixUp, {
                "reference_data": [{"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)}],
                "read_fn": lambda x: x}),
              (A.MixUp, {
            "reference_data": image_generator(),
            "read_fn": lambda x: x}),
              (A.MixUp, {
            "reference_data": complex_image_generator(),
            "read_fn": complex_read_fn_image})]

)
def test_image_only(augmentation_cls, params, image):
    aug = augmentation_cls(p=1, **params)
    data = aug(image=image)
    assert data["image"].dtype == np.uint8


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
             [(A.MixUp, {
                "reference_data": [{"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8),
                                   "global_label": np.array([0, 0, 1])}],
                "read_fn": lambda x: x})]
)
def test_image_global_label(augmentation_cls, params, image, global_label):
    aug = augmentation_cls(p=1, **params)

    data = aug(image=image, global_label=global_label)

    assert data["image"].dtype == np.uint8

    mix_coeff_image = find_mix_coef(data["image"], image, aug.reference_data[0]["image"])
    mix_coeff_label = find_mix_coef(data["global_label"], global_label, aug.reference_data[0]["global_label"])

    assert math.isclose(mix_coeff_image, mix_coeff_label, abs_tol=0.01)
    assert 0 <= mix_coeff_image <= 1

@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
             [(A.MixUp, {
                "reference_data": [{"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8),
                                    "mask": np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8),
                                   "global_label": np.array([0, 0, 1])}],

                "read_fn": lambda x: x})]
)
def test_image_mask_global_label(augmentation_cls, params, image, mask, global_label):
    aug = augmentation_cls(p=1, **params)

    data = aug(image=image, global_label=global_label, mask=mask)

    assert data["image"].dtype == np.uint8

    mix_coeff_image = find_mix_coef(data["image"], image, aug.reference_data[0]["image"])
    mix_coeff_mask = find_mix_coef(data["mask"], mask, aug.reference_data[0]["mask"])
    mix_coeff_label = find_mix_coef(data["global_label"], global_label, aug.reference_data[0]["global_label"])

    assert math.isclose(mix_coeff_image, mix_coeff_label, abs_tol=0.01)
    assert math.isclose(mix_coeff_image, mix_coeff_mask, abs_tol=0.01)
    assert 0 <= mix_coeff_image <= 1


def test_additional_targets(image, mask, global_label):
    reference_data = [{"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8),
                                    "mask": np.random.randint(0, 256, (100, 100), dtype=np.uint8),
                                   "global_label": np.array([0, 0, 1])}]


    aug = A.Compose([A.MixUp(p=1, reference_data=reference_data, read_fn = lambda x: x)], additional_targets={'image1': 'image', 'mask1': 'mask',
                                                                           'global_label1': 'global_label'})

    image1 = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
    mask1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    global_label1 = np.array([0, 1, 0])

    data = aug(image=image, global_label=global_label, mask=mask, image1=image1, global_label1=global_label1, mask1=mask1)

    assert data["image"].dtype == np.uint8

    mix_coeff_image = find_mix_coef(data["image"], image, reference_data[0]["image"])
    mix_coeff_mask = find_mix_coef(data["mask"], mask, reference_data[0]["mask"])
    mix_coeff_label = find_mix_coef(data["global_label"], global_label, reference_data[0]["global_label"])

    mix_coeff_image1 = find_mix_coef(data["image1"], image1, reference_data[0]["image"])
    mix_coeff_mask1 = find_mix_coef(data["mask1"], mask1, reference_data[0]["mask"])
    mix_coeff_label1 = find_mix_coef(data["global_label1"], global_label1, reference_data[0]["global_label"])

    assert math.isclose(mix_coeff_image, mix_coeff_label, abs_tol=0.01)

    assert math.isclose(mix_coeff_image, mix_coeff_mask, abs_tol=0.01)
    assert 0 <= mix_coeff_image <= 1

    assert math.isclose(mix_coeff_image, mix_coeff_image1, abs_tol=0.01)
    assert math.isclose(mix_coeff_image, mix_coeff_mask1, abs_tol=0.01)
    assert math.isclose(mix_coeff_image, mix_coeff_label1, abs_tol=0.01)
