import numpy as np
import pytest

import albumentations as A
from albumentations.core.types import PositionType
import cv2

from .conftest import IMAGES, RECTANGULAR_UINT8_IMAGE


def test_random_crop_vs_crop(bboxes, keypoints):
    image = RECTANGULAR_UINT8_IMAGE
    image_height, image_width = image.shape[:2]

    mask = np.random.randint(0, 2, image.shape[:2], dtype=np.uint8)

    random_crop_transform = A.Compose(
        [A.RandomCrop(height=image_height, width=image_width, p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc"),
        keypoint_params=A.KeypointParams(format="xyas"),
    )
    crop_transform = A.Compose(
        [A.Crop(x_min=0, y_min=0, x_max=image_width, y_max=image_height, p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc"),
        keypoint_params=A.KeypointParams(format="xyas"),
    )

    random_crop_result = random_crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
    crop_result = crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)

    np.testing.assert_array_equal(random_crop_result["image"], crop_result["image"])
    np.testing.assert_array_equal(random_crop_result["mask"], crop_result["mask"])

    np.testing.assert_equal(random_crop_result["bboxes"], crop_result["bboxes"])
    np.testing.assert_equal(random_crop_result["keypoints"], crop_result["keypoints"])


def test_center_crop_vs_crop(bboxes, keypoints):
    image = RECTANGULAR_UINT8_IMAGE
    height, width = 50, 50
    img_height, img_width = image.shape[:2]
    mask = np.random.randint(0, 2, image.shape[:2], dtype=np.uint8)

    center_crop_transform = A.Compose(
        [A.CenterCrop(height=height, width=width, p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc"),
        keypoint_params=A.KeypointParams(format="xyas"),
    )
    crop_transform = A.Compose(
        [
            A.Crop(
                x_min=(img_width - width) // 2,
                y_min=(img_height - height) // 2,
                x_max=(img_width + width) // 2,
                y_max=(img_height + height) // 2,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(format="pascal_voc"),
        keypoint_params=A.KeypointParams(format="xyas"),
    )

    center_crop_result = center_crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
    crop_result = crop_transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)

    np.testing.assert_array_equal(center_crop_result["image"], crop_result["image"])
    np.testing.assert_array_equal(center_crop_result["mask"], crop_result["mask"])

    np.testing.assert_equal(center_crop_result["bboxes"], crop_result["bboxes"])
    np.testing.assert_equal(center_crop_result["keypoints"], crop_result["keypoints"])


@pytest.mark.parametrize("image", IMAGES)
def test_crop_near_bbox(image, bboxes, keypoints):
    bbox_key = "target_bbox"
    aug = A.Compose(
        [A.RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_bbox_key=bbox_key, p=1)],
        bbox_params=A.BboxParams("pascal_voc"),
        keypoint_params=A.KeypointParams(format="xyas"),
    )

    aug(image=image, bboxes=bboxes, target_bbox=[0, 5, 10, 20], keypoints=keypoints)

    target_keys = {"image", "images", "bboxes", "labels", "mask", "masks", "keypoints", bbox_key}

    assert aug._available_keys == target_keys

    aug2 = A.Compose(
        [A.Sequential([A.RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_bbox_key=bbox_key, p=1)])],
        bbox_params=A.BboxParams("pascal_voc"),
        keypoint_params=A.KeypointParams(format="xyas"),
    )

    assert aug2._available_keys == target_keys


def test_crop_bbox_by_coords():
    cropped_bbox = A.crop_bboxes_by_coords(np.array([[0.5, 0.2, 0.9, 0.7]]), (18, 18, 82, 82), (100, 100))
    np.testing.assert_array_almost_equal(cropped_bbox, np.array([[0.5, 0.03125, 1.125, 0.8125]]))


@pytest.mark.parametrize(
    ["transforms", "bboxes", "expected_bboxes", "min_area", "min_visibility"],
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
    expected_bboxes,
    min_area: float,
    min_visibility: float,
) -> None:
    image = np.empty([100, 100, 3], dtype=np.uint8)
    aug = A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            "pascal_voc",
            min_area=min_area,
            min_visibility=min_visibility,
        ),
    )
    res = aug(image=image, bboxes=bboxes)["bboxes"]

    np.testing.assert_array_equal(res, expected_bboxes)

POSITIONS: list[PositionType] = ["center", "top_left", "top_right", "bottom_left", "bottom_right"]



@pytest.mark.parametrize(
    ["crop_cls", "crop_params"],
    [
        (A.RandomCrop, {"height": 150, "width": 150}),
        (A.CenterCrop, {"height": 150, "width": 150}),
    ]
)
@pytest.mark.parametrize("pad_position", POSITIONS)
@pytest.mark.parametrize("pad_mode", [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT_101, cv2.BORDER_REFLECT])
def test_pad_position_equivalence(
    image: np.ndarray,
    crop_cls: type[A.DualTransform],
    crop_params: dict[str, int],
    pad_position: PositionType,
    pad_mode: int,
    mask: np.ndarray,
    bboxes: np.ndarray,
    keypoints: np.ndarray,
):
    """Test that pad_position works identically for both padding approaches."""

    # Approach 1: Crop with built-in padding
    transform1 = A.Compose([
        crop_cls(
            **crop_params,
            pad_if_needed=True,
            pad_mode=pad_mode,
            pad_cval=0,
            pad_position=pad_position,
        )
    ], keypoint_params=A.KeypointParams(format="xyas"), bbox_params=A.BboxParams(format="pascal_voc"))

    # Approach 2: Separate pad and crop
    transform2 = A.Compose([
        A.PadIfNeeded(
            min_height=crop_params["height"],
            min_width=crop_params["width"],
            border_mode=pad_mode,
            value=0,
            position=pad_position,
        ),
        crop_cls(
            **crop_params,
            pad_if_needed=False,
        )
    ], keypoint_params=A.KeypointParams(format="xyas"), bbox_params=A.BboxParams(format="pascal_voc"))

    result1 = transform1(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
    result2 = transform2(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)

    np.testing.assert_array_equal(
        result1["image"],
        result2["image"],
        err_msg=f"Images don't match for position {pad_position}"
    )
    np.testing.assert_array_equal(
        result1["mask"],
        result2["mask"],
        err_msg=f"Masks don't match for position {pad_position}"
    )
    np.testing.assert_array_equal(
        result1["bboxes"],
        result2["bboxes"],
        err_msg=f"Bboxes don't match for position {pad_position}"
    )
    np.testing.assert_array_equal(
        result1["keypoints"],
        result2["keypoints"],
        err_msg=f"Keypoints don't match for position {pad_position}"
    )
