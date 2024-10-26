from __future__ import annotations

from typing import Any, Literal, cast

import cv2
import numpy as np

import albumentations.augmentations.dropout.functional as fdropout
from albumentations.core.bbox_utils import BboxProcessor, denormalize_bboxes, normalize_bboxes
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.pydantic import OnePlusIntRangeType
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import ScalarType, ScaleIntType, Targets

__all__ = ["MaskDropout"]


class MaskDropout(DualTransform):
    """Apply dropout to random objects in a mask, zeroing out the corresponding regions in both the image and mask.

    This transform identifies objects in the mask (where each unique non-zero value represents a distinct object),
    randomly selects a number of these objects, and sets their corresponding regions to zero in both the image and mask.
    It can also handle bounding boxes and keypoints, removing or adjusting them based on the dropout regions.

    Args:
        max_objects (int | tuple[int, int]): Maximum number of objects to dropout. If a single int is provided,
            it's treated as the upper bound. If a tuple of two ints is provided, it's treated as a range [min, max].
        image_fill_value (float | str): Value to fill the dropped out regions in the image.
            If set to 'inpaint', it applies inpainting to the dropped out regions (works only for 3-channel images).
        mask_fill_value (float | int): Value to fill the dropped out regions in the mask.
        min_area (float): Minimum area (in pixels) of a bounding box that must remain visible after dropout to be kept.
            Only applicable if bounding box augmentation is enabled. Default: 0.0
        min_visibility (float): Minimum visibility ratio (visible area / total area) of a bounding box after dropout
            to be kept. Only applicable if bounding box augmentation is enabled. Default: 0.0
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The mask should be a single-channel image where 0 represents the background and non-zero values represent
          different object instances.
        - For bounding box and keypoint augmentation, make sure to set up the corresponding processors in the pipeline.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Define a sample image, mask, and bounding boxes
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[20:40, 20:40] = 1  # Object 1
        >>> mask[60:80, 60:80] = 2  # Object 2
        >>> bboxes = np.array([[20, 20, 40, 40], [60, 60, 80, 80]])
        >>>
        >>> # Define the transform
        >>> transform = A.Compose([
        ...     A.MaskDropout(max_objects=1, mask_fill_value=0, min_area=100, min_visibility=0.5, p=1.0),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', min_area=1, min_visibility=0.1))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes)
        >>>
        >>> # The result will have one of the objects dropped out in both image and mask,
        >>> # and the corresponding bounding box removed if it doesn't meet the area and visibility criteria
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        max_objects: OnePlusIntRangeType

        image_fill_value: float | Literal["inpaint"]
        mask_fill_value: ScalarType

    def __init__(
        self,
        max_objects: ScaleIntType = (1, 1),
        image_fill_value: float | Literal["inpaint"] = 0,
        mask_fill_value: ScalarType = 0,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.max_objects = cast(tuple[int, int], max_objects)
        self.image_fill_value = image_fill_value
        self.mask_fill_value = mask_fill_value

    @property
    def targets_as_params(self) -> list[str]:
        return ["mask"]

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        mask = data["mask"]

        label_image, num_labels = fdropout.label(mask, return_num=True)

        if num_labels == 0:
            dropout_mask = None
        else:
            objects_to_drop = self.py_random.randint(*self.max_objects)
            objects_to_drop = min(num_labels, objects_to_drop)

            if objects_to_drop == num_labels:
                dropout_mask = mask > 0
            else:
                labels_index = self.py_random.sample(range(1, num_labels + 1), objects_to_drop)
                dropout_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)
                for label_index in labels_index:
                    dropout_mask |= label_image == label_index

        return {"dropout_mask": dropout_mask}

    def apply(self, img: np.ndarray, dropout_mask: np.ndarray | None, **params: Any) -> np.ndarray:
        if dropout_mask is None:
            return img

        if self.image_fill_value == "inpaint":
            dropout_mask = dropout_mask.astype(np.uint8)
            _, _, width, height = cv2.boundingRect(dropout_mask)
            radius = min(3, max(width, height) // 2)
            return cv2.inpaint(img, dropout_mask, radius, cv2.INPAINT_NS)

        img = img.copy()
        img[dropout_mask] = self.image_fill_value

        return img

    def apply_to_mask(self, mask: np.ndarray, dropout_mask: np.ndarray | None, **params: Any) -> np.ndarray:
        if dropout_mask is None:
            return mask

        mask = mask.copy()
        mask[dropout_mask] = self.mask_fill_value
        return mask

    def apply_to_bboxes(self, bboxes: np.ndarray, dropout_mask: np.ndarray | None, **params: Any) -> np.ndarray:
        if dropout_mask is None:
            return bboxes

        processor = cast(BboxProcessor, self.get_processor("bboxes"))
        if processor is None:
            return bboxes

        image_shape = params["shape"][:2]

        denormalized_bboxes = denormalize_bboxes(bboxes, image_shape)

        result = fdropout.mask_dropout_bboxes(
            denormalized_bboxes,
            dropout_mask,
            image_shape,
            processor.params.min_area,
            processor.params.min_visibility,
        )

        return normalize_bboxes(result, image_shape)

    def apply_to_keypoints(self, keypoints: np.ndarray, dropout_mask: np.ndarray | None, **params: Any) -> np.ndarray:
        if dropout_mask is None:
            return keypoints

        processor = cast(KeypointsProcessor, self.get_processor("keypoints"))

        if processor is None or not processor.params.remove_invisible:
            return keypoints

        return fdropout.mask_dropout_keypoints(keypoints, dropout_mask)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "max_objects", "image_fill_value", "mask_fill_value"
