from tests.conftest import SQUARE_FLOAT_IMAGE
from .utils import get_dual_transforms, get_image_only_transforms
import pytest
from albumentations.core.types import Targets
import albumentations as A
import numpy as np
import re

def get_targets_from_methods(cls):
    targets = {Targets.IMAGE, Targets.MASK}

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

    has_global_label_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(A.DualTransform, attr, None)
        for attr in ["apply_to_global_label", "apply_to_global_labels"]
    )
    if has_global_label_method:
        targets.add(Targets.GLOBAL_LABEL)

    return targets


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
        targets = re.split(r'[,\s]+', targets_str)  # Split by comma or whitespace
        return [target.strip() for target in targets if target.strip()]  # Remove any extra whitespace
    else:
        return []  # Return an empty list if the 'Targets:' section isn't found


DUAL_TARGETS = {
    A.Affine: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.BBoxSafeRandomCrop: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.CenterCrop: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.RandomCrop: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.RandomCropFromBorders: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.RandomResizedCrop: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.CropNonEmptyMaskIfExists: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.CoarseDropout: (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS),
    A.Crop: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.CropAndPad: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.Flip: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.VerticalFlip: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.HorizontalFlip: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.PadIfNeeded: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.RandomScale: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.XYMasking: (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS),
    A.NoOp: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS, Targets.GLOBAL_LABEL),
    A.Resize: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.Rotate: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.RandomRotate90: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.Transpose: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.SmallestMaxSize: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.ShiftScaleRotate: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.ElasticTransform: (Targets.IMAGE, Targets.MASK, Targets.BBOXES),
    A.GridDistortion: (Targets.IMAGE, Targets.MASK, Targets.BBOXES),
    A.LongestMaxSize: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.PiecewiseAffine: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.RandomSizedCrop: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.RandomCropFromBorders: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.RandomGridShuffle: (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS),
    A.OpticalDistortion: (Targets.IMAGE, Targets.MASK, Targets.BBOXES),
    A.SafeRotate: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.CropNonEmptyMaskIfExists: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.XYMasking: (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS),
    A.RandomCropNearBBox: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.Perspective: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.RandomSizedBBoxSafeCrop: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.MixUp: (Targets.IMAGE, Targets.MASK, Targets.GLOBAL_LABEL),
    A.Lambda: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS, Targets.GLOBAL_LABEL),
    A.D4: (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS),
    A.OverlayElements: (Targets.IMAGE, Targets.MASK),
}

str2target = {
    'image': Targets.IMAGE, 'mask': Targets.MASK, 'bboxes': Targets.BBOXES, 'keypoints': Targets.KEYPOINTS, 'global_label': Targets.GLOBAL_LABEL
}

@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_image_only_transforms(
        custom_arguments={
            A.HistogramMatching: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.FDA: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
            },
            A.PixelDistributionAdaptation: {
                "reference_images": [np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)],
                "read_fn": lambda x: x,
                "transform_type": "standard",
            },
            A.TemplateTransform: {
                "templates": np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8),
            },
            A.TextImage: dict(font_path="./tests/filesLiberationSerif-Bold.ttf")
        },
    ),
)
def test_image_only(augmentation_cls, params):
    aug = augmentation_cls(p=1, **params)

    assert aug._targets == (Targets.IMAGE)

@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    get_dual_transforms(
        custom_arguments={
            A.Crop: {"y_min": 0, "y_max": 10, "x_min": 0, "x_max": 10},
            A.CenterCrop: {"height": 10, "width": 10},
            A.CropNonEmptyMaskIfExists: {"height": 10, "width": 10},
            A.RandomCrop: {"height": 10, "width": 10},
            A.RandomResizedCrop: {"height": 10, "width": 10},
            A.RandomSizedCrop: {"min_max_height": (4, 8), "height": 10, "width": 10},
            A.RandomSizedBBoxSafeCrop: {"height": 10, "width": 10},
            A.CropAndPad: {"px": 10},
            A.Resize: {"height": 10, "width": 10},
            A.XYMasking: {
                "num_masks_x": (1, 3),
                "num_masks_y": (1, 3),
                "mask_x_length": 10,
                "mask_y_length": 10,
                "mask_fill_value": 1,
                "fill_value": 0,
            },
             A.MixUp: {
                "reference_data": [{"image": SQUARE_FLOAT_IMAGE,
                                    "mask": np.random.uniform(low=0, high=1, size=(100, 100)).astype(np.float32)
                                    }],
                "read_fn": lambda x: x,
            },
        },
    ),
)
def test_dual(augmentation_cls, params):
    aug = augmentation_cls(p=1, **params)
    assert set(aug._targets) == set(DUAL_TARGETS.get(augmentation_cls, {Targets.IMAGE, Targets.MASK}))
    assert set(aug._targets) <= get_targets_from_methods(augmentation_cls)

    targets_from_docstring = {str2target[target] for target in  extract_targets_from_docstring(augmentation_cls)}

    assert set(aug._targets) == targets_from_docstring
