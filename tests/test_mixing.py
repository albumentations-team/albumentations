import numpy as np
import pytest
import math

import albumentations as A
from tests.conftest import IMAGES, UINT8_IMAGES
from tests.utils import set_seed
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
                  "reference_data": [1],
                  "read_fn": lambda x: {"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)}},
              ),
               (A.MixUp, {
                  "reference_data": np.array([1]),
                  "read_fn": lambda x: {"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)}},
              ),
              (A.MixUp, {
                  "reference_data": None,
              }),
              (A.MixUp, {
            "reference_data": image_generator(),
            "read_fn": lambda x: x}),
              (A.MixUp, {
            "reference_data": complex_image_generator(),
            "read_fn": complex_read_fn_image})] )
def test_image_only(augmentation_cls, params):
    square_image = UINT8_IMAGES[0]
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)
    data = aug(image=square_image)
    assert data["image"].dtype == np.uint8

@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
             [(A.MixUp, {
                "reference_data": [{"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8),
                                   "global_label": np.array([0, 0, 1])}],
                "read_fn": lambda x: x}),
            (A.MixUp, {
                  "reference_data": [1],
                  "read_fn": lambda x: {"image": np.ones((100, 100, 3)).astype(np.uint8),
                                        "global_label": np.array([0, 0, 1])}},
              ),
              ]
)
def test_image_global_label(augmentation_cls, params, global_label):
    square_image = UINT8_IMAGES[0]
    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)

    data = aug(image=square_image, global_label=global_label)

    assert data["image"].dtype == np.uint8

    reference_data = params["reference_data"][0]

    reference_item = params["read_fn"](reference_data)

    reference_image = reference_item["image"]
    reference_global_label = reference_item["global_label"]

    mix_coef = data["mix_coef"]

    mix_coeff_image = find_mix_coef(data["image"], square_image, reference_image)
    mix_coeff_label = find_mix_coef(data["global_label"], global_label, reference_global_label)

    assert math.isclose(mix_coef, mix_coeff_image, abs_tol=0.01)
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
def test_image_mask_global_label(augmentation_cls, params, global_label):
    image = UINT8_IMAGES[0]
    mask = image[:, :, 0].copy()

    reference_data = params["reference_data"][0]

    aug = A.Compose([augmentation_cls(p=1, **params)], p=1)

    data = aug(image=image, global_label=global_label, mask=mask)


    mix_coef = data["mix_coef"]

    mix_coeff_image = find_mix_coef(data["image"], image, reference_data["image"])
    mix_coeff_mask = find_mix_coef(data["mask"], mask, reference_data["mask"])
    mix_coeff_label = find_mix_coef(data["global_label"], global_label, reference_data["global_label"])

    assert math.isclose(mix_coef, mix_coeff_image, abs_tol=0.01)
    assert math.isclose(mix_coeff_image, mix_coeff_label, abs_tol=0.01)
    assert math.isclose(mix_coeff_image, mix_coeff_mask, abs_tol=0.01)
    assert 0 <= mix_coeff_image <= 1

@pytest.mark.parametrize("image", IMAGES)
def test_additional_targets(image, global_label):
    set_seed(42)

    mask = image.copy()
    image1 = np.random.randint(0, 256, image.shape, dtype=np.uint8).astype(image.dtype)
    mask1 = np.random.randint(0, 256, mask.shape, dtype=np.uint8).astype(mask.dtype)
    reference_image = np.random.randint(0, 256, image.shape, dtype=np.uint8).astype(image.dtype)
    reference_mask = np.random.randint(0, 256, mask.shape, dtype=np.uint8).astype(mask.dtype)

    if image.dtype == np.float32:
        image /= 255
        mask1 /= 255
        image1 /= 255
        mask1 /= 255
        reference_image /= 255
        reference_mask /= 255

    reference_data = [{"image": reference_image,
                                    "mask": reference_mask,
                                   "global_label": np.array([0, 0, 1])}]


    aug = A.Compose([A.MixUp(p=1, reference_data=reference_data, read_fn = lambda x: x)], additional_targets={'image1': 'image', 'mask1': 'mask',
                                                                           'global_label1': 'global_label'})

    global_label1 = np.array([0, 1, 0])

    data = aug(image=image, global_label=global_label, mask=mask, image1=image1, global_label1=global_label1, mask1=mask1)

    mix_coef = data["mix_coef"]

    assert data["image"].dtype == image.dtype

    mix_coeff_image = find_mix_coef(data["image"], image, reference_data[0]["image"])
    mix_coeff_mask = find_mix_coef(data["mask"], mask, reference_data[0]["mask"])
    mix_coeff_label = find_mix_coef(data["global_label"], global_label, reference_data[0]["global_label"])

    mix_coeff_image1 = find_mix_coef(data["image1"], image1, reference_data[0]["image"])
    mix_coeff_mask1 = find_mix_coef(data["mask1"], mask1, reference_data[0]["mask"])
    mix_coeff_label1 = find_mix_coef(data["global_label1"], global_label1, reference_data[0]["global_label"])

    assert math.isclose(mix_coef, mix_coeff_image, abs_tol=0.01)
    assert math.isclose(mix_coeff_image, mix_coeff_label, abs_tol=0.01)

    assert math.isclose(mix_coeff_image, mix_coeff_mask, abs_tol=0.01)
    assert 0 <= mix_coeff_image <= 1

    assert math.isclose(mix_coeff_image, mix_coeff_image1, abs_tol=0.01)
    assert math.isclose(mix_coeff_image, mix_coeff_mask1, abs_tol=0.01)
    assert math.isclose(mix_coeff_image, mix_coeff_label1, abs_tol=0.01)

@pytest.mark.parametrize("image", IMAGES)
def test_bbox_error(image, global_label, bboxes):
    mask = image.copy()

    reference_data = [
        {"image": np.random.randint(0, 256, image.shape, dtype=np.uint8).astype(image.dtype),
         "mask": np.random.randint(0, 256, mask.shape, dtype=np.uint8).astype(mask.dtype),
         "bboxes": [[15, 12, 75, 30, 1], [55, 25, 90, 90, 2]],
         "global_label": np.array([0, 0, 1])}
        ]

    aug = A.Compose([A.MixUp(p=1, reference_data=reference_data, read_fn=lambda x: x)], p=1, bbox_params=A.BboxParams(format="pascal_voc", min_area=16))

    with pytest.raises(NotImplementedError):
        aug(image=image, global_label=global_label, mask=mask, bboxes=bboxes)

@pytest.mark.parametrize("image", IMAGES)
def test_keypoint_error(image, global_label, keypoints):
    mask = image.copy()

    reference_data = [
        {"image": np.random.randint(0, 256, image.shape, dtype=np.uint8).astype(image.dtype),
         "mask": np.random.randint(0, 256, mask.shape, dtype=np.uint8).astype(mask.dtype),
         "keypoints": [[20, 30, 40, 50, 1], [20, 30, 60, 80, 2]],
         "global_label": np.array([0, 0, 1])}
    ]

    aug = A.Compose([A.MixUp(p=1, reference_data=reference_data, read_fn=lambda x: x)], p=1, keypoint_params=A.KeypointParams(format="xy"))

    with pytest.raises(NotImplementedError):
        aug(image=image, global_label=global_label, mask=mask, keypoints=keypoints)

@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize( ["augmentation_cls", "params"], [(A.ColorJitter, {"p": 1}), (A.HorizontalFlip, {"p": 1})])
def test_pipeline(augmentation_cls, params, image, global_label):
    mask = image.copy()

    reference_data =[{"image": np.random.randint(0, 256, image.shape, dtype=np.uint8).astype(image.dtype),
                     "mask": np.random.randint(0, 256, mask.shape, dtype=np.uint8).astype(mask.dtype),
                     "global_label": np.array([0, 0, 1])}]

    mix_up = A.MixUp(p=1, reference_data=reference_data, read_fn=lambda x: x)

    aug = A.Compose([augmentation_cls(**params), mix_up], p=1)

    data = aug(image=image, global_label=global_label, mask=mask)

    assert data["image"].dtype == image.dtype

    mix_coef = data["mix_coef"]

    mix_coeff_label = find_mix_coef(data["global_label"], global_label, reference_data[0]["global_label"])

    assert math.isclose(mix_coef, mix_coeff_label, abs_tol=0.01)
    assert 0 <= mix_coeff_label <= 1
