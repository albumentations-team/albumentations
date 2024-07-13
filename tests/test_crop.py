import numpy as np

from tests.utils import set_seed
from .conftest import IMAGES, RECTANGULAR_UINT8_IMAGE
import albumentations as A
import pytest


def test_random_crop_vs_crop(bboxes, keypoints):
    image = RECTANGULAR_UINT8_IMAGE
    image_height, image_width = image.shape[:2]

    mask = np.random.randint(0, 2, image.shape[:2], dtype=np.uint8)

    random_crop_transform = A.Compose([A.RandomCrop(height=image_height, width=image_width, p=1.0)], bbox_params=A.BboxParams(format="pascal_voc"), keypoint_params=A.KeypointParams(format="xyas"))
    crop_transform = A.Compose([A.Crop(x_min=0, y_min=0, x_max=image_width, y_max=image_height, p=1.0)], bbox_params=A.BboxParams(format="pascal_voc"), keypoint_params=A.KeypointParams(format="xyas"))

    random_crop_result = random_crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
    crop_result = crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)

    np.testing.assert_array_equal(random_crop_result["image"], crop_result["image"])
    np.testing.assert_array_equal(random_crop_result["mask"], crop_result["mask"])
    assert random_crop_result["bboxes"] == crop_result["bboxes"]
    assert random_crop_result["keypoints"] == crop_result["keypoints"]


def test_center_crop_vs_crop(bboxes, keypoints):
    image = RECTANGULAR_UINT8_IMAGE
    height, width = 50, 50
    img_height, img_width = image.shape[:2]
    mask = np.random.randint(0, 2, image.shape[:2], dtype=np.uint8)

    center_crop_transform = A.Compose([A.CenterCrop(height=height, width=width, p=1.0)], bbox_params=A.BboxParams(format="pascal_voc"), keypoint_params=A.KeypointParams(format="xyas"))
    crop_transform = A.Compose([A.Crop(x_min=(img_width - width) // 2, y_min=(img_height - height) // 2,
                                   x_max=(img_width + width) // 2, y_max=(img_height + height) // 2, p=1.0)], bbox_params=A.BboxParams(format="pascal_voc"), keypoint_params=A.KeypointParams(format="xyas"))

    center_crop_result = center_crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
    crop_result = crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)

    np.testing.assert_array_equal(center_crop_result["image"], crop_result["image"])
    np.testing.assert_array_equal(center_crop_result["mask"], crop_result["mask"])
    assert center_crop_result["bboxes"] == crop_result["bboxes"]
    assert center_crop_result["keypoints"] == crop_result["keypoints"]


@pytest.mark.parametrize("image", IMAGES)
def test_crop_near_bbox(image, bboxes, keypoints):
    set_seed(42)
    bbox_key = "target_bbox"
    aug = A.Compose(
        [A.RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_bbox_key=bbox_key, p=1)],
        bbox_params=A.BboxParams("pascal_voc"),
        keypoint_params=A.KeypointParams(format="xyas")
    )

    aug(image=image, bboxes=bboxes, target_bbox=[0, 5, 10, 20], keypoints=keypoints)

    target_keys = {'image', "images", 'bboxes', "labels", "mask", "masks", "keypoints", bbox_key}

    assert aug._available_keys == target_keys

    aug2 = A.Compose(
        [A.Sequential([A.RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_bbox_key=bbox_key, p=1)])],
        bbox_params=A.BboxParams("pascal_voc"),
        keypoint_params=A.KeypointParams(format="xyas")
    )

    assert aug2._available_keys == target_keys


def test_crop_bbox_by_coords():
    cropped_bbox = A.crop_bbox_by_coords((0.5, 0.2, 0.9, 0.7), (18, 18, 82, 82), 100, 100)
    assert cropped_bbox == (0.5, 0.03125, 1.125, 0.8125)


@pytest.mark.parametrize(
    ["transforms", "bboxes", "result_bboxes", "min_area", "min_visibility"],
    [
        [[A.Crop(10, 10, 20, 20)], [[0, 0, 10, 10, 0]], [], 0, 0],
        [
            [A.Crop(0, 0, 90, 90)],
            [[0, 0, 91, 91, 0], [0, 0, 89, 89, 0]],
            [[0, 0, 90, 90, 0], [0, 0, 89, 89, 0]],
            0,
            0.9,
        ],
        [
            [A.Crop(0, 0, 90, 90)],
            [[0, 0, 1, 10, 0], [0, 0, 1, 11, 0]],
            [[0, 0, 1, 10, 0], [0, 0, 1, 11, 0]],
            10,
            0,
        ],
    ],
)
def test_bbox_params_edges(
    transforms,
    bboxes,
    result_bboxes,
    min_area: float,
    min_visibility: float,
) -> None:
    image = np.empty([100, 100, 3], dtype=np.uint8)
    aug = A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            "pascal_voc", min_area=min_area, min_visibility=min_visibility
        ),
    )
    res = aug(image=image, bboxes=bboxes)["bboxes"]

    assert np.array_equal(res, result_bboxes)
