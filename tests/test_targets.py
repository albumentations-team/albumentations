import re

import numpy as np
import pytest

import albumentations as A
from albumentations.core.type_definitions import ALL_TARGETS, Targets

from tests.conftest import SQUARE_FLOAT_IMAGE
from .utils import get_dual_transforms, get_image_only_transforms


def get_targets_from_methods(cls):
    targets = {Targets.IMAGE, Targets.MASK, Targets.VOLUME, Targets.MASK3D}

    has_bboxes_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(A.DualTransform, attr, None)
        for attr in ["apply_to_bbox", "apply_to_bboxes"]
    )
    if has_bboxes_method:
        targets.add(Targets.BBOXES)

    has_keypoints_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(A.DualTransform, attr, None)
        for attr in ["apply_to_keypoint", "apply_to_keypoints"]
    )
    if has_keypoints_method:
        targets.add(Targets.KEYPOINTS)

    return targets


def extract_targets_from_docstring(cls):
    # Access the class's docstring
    if not (docstring := cls.__doc__):
        return []  # Return an empty list if there's no docstring

    # Regular expression to match the 'Targets:' section in the docstring
    targets_pattern = r"Targets:\s*([^\n]+)"

    # Search for the pattern in the docstring and extract targets if found
    if matches := re.search(targets_pattern, docstring):
        # Extract the targets string and split it by commas or spaces
        targets = re.split(r"[,\s]+", matches[1])  # Using subscript notation instead of group()
        return [target.strip() for target in targets if target.strip()]  # Remove any extra whitespace

    return []  # Return an empty list if the 'Targets:' section isn't found


DUAL_TARGETS = {
    A.OverlayElements: (Targets.IMAGE, Targets.MASK),
    A.Mosaic: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
}


str2target = {
    "image": Targets.IMAGE,
    "mask": Targets.MASK,
    "bboxes": Targets.BBOXES,
    "keypoints": Targets.KEYPOINTS,
    "volume": Targets.VOLUME,
    "mask3d": Targets.MASK3D,
}


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        custom_arguments={
            A.TextImage: dict(font_path="./tests/filesLiberationSerif-Bold.ttf"),
        },
    ),
)
def test_image_only(augmentation_cls, params):
    aug = augmentation_cls(p=1, **params)
    assert aug._targets == (Targets.IMAGE, Targets.VOLUME)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.AtLeastOneBBoxRandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"size": (10, 10)},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "size": (10, 10)},
            A.RandomSizedBBoxSafeCrop: {"height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "fill_mask": 1,
                "fill": 0,
            },
            A.GridElasticDeform: {"num_grid_xy": (10, 10), "magnitude": 10},
            A.Mosaic: {},
        },
    ),
)
def test_dual(augmentation_cls, params):
    aug = augmentation_cls(p=1, **params)
    assert set(aug._targets) == set(DUAL_TARGETS.get(augmentation_cls, ALL_TARGETS))
    assert set(aug._targets) <= get_targets_from_methods(augmentation_cls)

    targets_from_docstring = {str2target[target] for target in extract_targets_from_docstring(augmentation_cls)}

    assert set(aug._targets) == targets_from_docstring
