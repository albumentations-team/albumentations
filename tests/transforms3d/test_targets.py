import re

import numpy as np
import pytest

import albumentations as A
from albumentations.core.type_definitions import Targets
from tests.utils import get_3d_transforms


def extract_targets_from_docstring(cls):
    # Access the class's docstring
    docstring = cls.__doc__
    if not docstring:
        return []  # Return an empty list if there's no docstring

    # Regular expression to match the 'Targets:' section in the docstring
    targets_pattern = r"Targets:\s*([^\n]+)"

    # Search for the pattern in the docstring
    matches = re.search(targets_pattern, docstring)
    if matches:
        # Extract the targets string and split it by commas or spaces
        targets_str = matches.group(1)
        targets = re.split(r"[,\s]+", targets_str)  # Split by comma or whitespace
        return [target.strip() for target in targets if target.strip()]  # Remove any extra whitespace
    return []  # Return an empty list if the 'Targets:' section isn't found


def get_targets_from_methods(cls):
    targets = {Targets.VOLUME, Targets.MASK3D}

    has_volume_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(A.Transform3D, attr, None)
        for attr in ["apply_to_volume"]
    )
    if has_volume_method:
        targets.add(Targets.VOLUME)

    has_masks_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(A.Transform3D, attr, None)
        for attr in ["apply_to_mask"]
    )
    if has_masks_method:
        targets.add(Targets.MASK)

    has_masks3d_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(A.Transform3D, attr, None)
        for attr in ["apply_to_mask3d"]
    )
    if has_masks3d_method:
        targets.add(Targets.MASK3D)

    has_bboxes_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(A.Transform3D, attr, None)
        for attr in ["apply_to_bboxes"]
    )
    if has_bboxes_method:
        targets.add(Targets.BBOXES)

    has_keypoints_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(A.Transform3D, attr, None)
        for attr in ["apply_to_keypoints"]
    )
    if has_keypoints_method:
        targets.add(Targets.KEYPOINTS)

    return targets

TRASNFORM_3D_TARGETS = {
}

str2target = {
    "mask3d": Targets.MASK3D,
    "volume": Targets.VOLUME,
    "keypoints": Targets.KEYPOINTS,
}

@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_3d_transforms(custom_arguments={
        A.PadIfNeeded3D: {"min_zyx": (4, 250, 230), "position": "center", "fill": 0, "fill_mask": 0},
        A.Pad3D: {"padding": 10},
        A.RandomCrop3D: {"size": (2, 30, 30), "pad_if_needed": True},
        A.CenterCrop3D: {"size": (2, 30, 30), "pad_if_needed": True},
    })
)
def test_transform3d(augmentation_cls, params):
    aug = augmentation_cls(p=1, **params)
    assert set(aug._targets) == set(TRASNFORM_3D_TARGETS.get(augmentation_cls, {Targets.MASK3D, Targets.VOLUME, Targets.KEYPOINTS}))
    assert set(aug._targets) <= get_targets_from_methods(augmentation_cls)

    targets_from_docstring = {str2target[target] for target in extract_targets_from_docstring(augmentation_cls)}

    assert set(aug._targets) == targets_from_docstring
