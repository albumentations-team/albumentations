"""Implementation of mask-based dropout augmentation.

This module provides the MaskDropout transform, which identifies objects in a segmentation mask
and drops out random objects completely. This augmentation is particularly useful for instance
segmentation and object detection tasks, as it simulates occlusions or missing objects in a
semantically meaningful way, rather than dropping out random pixels or regions.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import cv2
import numpy as np

import albumentations.augmentations.dropout.functional as fdropout
from albumentations.core.bbox_utils import BboxProcessor, denormalize_bboxes, normalize_bboxes
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.pydantic import OnePlusIntRangeType
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.type_definitions import ALL_TARGETS

__all__ = ["MaskDropout"]


class MaskDropout(DualTransform):
    """Apply dropout to random objects in a mask, zeroing out the corresponding regions in both the image and mask.

    This transform identifies objects in the mask (where each unique non-zero value represents a distinct object),
    randomly selects a number of these objects, and sets their corresponding regions to zero in both the image and mask.
    It can also handle bounding boxes and keypoints, removing or adjusting them based on the dropout regions.

    Args:
        max_objects (int | tuple[int, int]): Maximum number of objects to dropout. If a single int is provided,
            it's treated as the upper bound. If a tuple of two ints is provided, it's treated as a range [min, max].
        fill (float | Literal["inpaint_telea", "inpaint_ns"]): Value to fill dropped out regions in the image.
            Can be one of:
            - float: Constant value to fill the regions (e.g., 0 for black, 255 for white)
            - "inpaint_telea": Use Telea inpainting algorithm (for 3-channel images only)
            - "inpaint_ns": Use Navier-Stokes inpainting algorithm (for 3-channel images only)
        fill_mask (float): Value to fill the dropped out regions in the mask.
        min_area (float): Minimum area (in pixels) of a bounding box that must remain visible after dropout to be kept.
            Only applicable if bounding box augmentation is enabled. Default: 0.0
        min_visibility (float): Minimum visibility ratio (visible area / total area) of a bounding box after dropout
            to be kept. Only applicable if bounding box augmentation is enabled. Default: 0.0
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - The mask should be a single-channel image where 0 represents the background and non-zero values represent
          different object instances.
        - For bounding box and keypoint augmentation, make sure to set up the corresponding processors in the pipeline.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[20:40, 20:40] = 1  # Object 1
        >>> mask[60:80, 60:80] = 2  # Object 2
        >>> bboxes = np.array([[20, 20, 40, 40], [60, 60, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[30, 30], [70, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define the transform with tuple for max_objects
        >>> transform = A.Compose(
        ...     transforms=[
        ...         A.MaskDropout(
        ...             max_objects=(1, 2),  # Using tuple to specify min and max objects to drop
        ...             fill=0,  # Fill value for dropped regions in image
        ...             fill_mask=0,  # Fill value for dropped regions in mask
        ...             p=1.0
        ...         ),
        ...     ],
        ...     bbox_params=A.BboxParams(
        ...         format='pascal_voc',
        ...         label_fields=['bbox_labels'],
        ...         min_area=1,
        ...         min_visibility=0.1
        ...     ),
        ...     keypoint_params=A.KeypointParams(
        ...         format='xy',
        ...         label_fields=['keypoint_labels'],
        ...         remove_invisible=True
        ...     )
        ... )
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = transformed['image']  # Image with dropped out regions
        >>> transformed_mask = transformed['mask']    # Mask with dropped out regions
        >>> transformed_bboxes = transformed['bboxes']  # Remaining bboxes after dropout
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for remaining bboxes
        >>> transformed_keypoints = transformed['keypoints']  # Remaining keypoints after dropout
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for remaining keypoints

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        max_objects: OnePlusIntRangeType

        fill: float | Literal["inpaint_telea", "inpaint_ns"]
        fill_mask: float

    def __init__(
        self,
        max_objects: tuple[int, int] | int = (1, 1),
        fill: float | Literal["inpaint_telea", "inpaint_ns"] = 0,
        fill_mask: float = 0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.max_objects = cast("tuple[int, int]", max_objects)
        self.fill = fill  # type: ignore[assignment]
        self.fill_mask = fill_mask

    @property
    def targets_as_params(self) -> list[str]:
        """Get targets as parameters.

        Returns:
            list[str]: List of targets as parameters.

        """
        return ["mask"]

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Get parameters dependent on the data.

        Args:
            params (dict[str, Any]): Dictionary containing parameters.
            data (dict[str, Any]): Dictionary containing data.

        Returns:
            dict[str, Any]: Dictionary with parameters for transformation.

        """
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
                dropout_mask = np.zeros(mask.shape[:2], dtype=bool)
                for label_index in labels_index:
                    dropout_mask |= label_image == label_index

        return {"dropout_mask": dropout_mask}

    def apply(self, img: np.ndarray, dropout_mask: np.ndarray | None, **params: Any) -> np.ndarray:
        """Apply dropout to the image.

        Args:
            img (np.ndarray): The image to apply the transform to.
            dropout_mask (np.ndarray | None): The dropout mask for the image.
            **params (Any): Additional parameters for the transform.

        Returns:
            np.ndarray: The transformed image.

        """
        if dropout_mask is None:
            return img

        if self.fill in {"inpaint_telea", "inpaint_ns"}:
            dropout_mask = dropout_mask.astype(np.uint8)
            _, _, width, height = cv2.boundingRect(dropout_mask)
            radius = min(3, max(width, height) // 2)
            return cv2.inpaint(img, dropout_mask, radius, cast("Literal['inpaint_telea', 'inpaint_ns']", self.fill))

        img = img.copy()
        img[dropout_mask] = self.fill

        return img

    def apply_to_mask(self, mask: np.ndarray, dropout_mask: np.ndarray | None, **params: Any) -> np.ndarray:
        """Apply dropout to the mask.

        Args:
            mask (np.ndarray): The mask to apply the transform to.
            dropout_mask (np.ndarray | None): The dropout mask for the mask.
            **params (Any): Additional parameters for the transform.

        Returns:
            np.ndarray: The transformed mask.

        """
        if dropout_mask is None or self.fill_mask is None:
            return mask

        mask = mask.copy()
        mask[dropout_mask] = self.fill_mask
        return mask

    def apply_to_bboxes(self, bboxes: np.ndarray, dropout_mask: np.ndarray | None, **params: Any) -> np.ndarray:
        """Apply dropout to bounding boxes.

        Args:
            bboxes (np.ndarray): The bounding boxes to apply the transform to.
            dropout_mask (np.ndarray | None): The dropout mask for the bounding boxes.
            **params (Any): Additional parameters for the transform.

        Returns:
            np.ndarray: The transformed bounding boxes.

        """
        if dropout_mask is None:
            return bboxes

        processor = cast("BboxProcessor", self.get_processor("bboxes"))
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
        """Apply dropout to keypoints.

        Args:
            keypoints (np.ndarray): The keypoints to apply the transform to.
            dropout_mask (np.ndarray | None): The dropout mask for the keypoints.
            **params (Any): Additional parameters for the transform.

        Returns:
            np.ndarray: The transformed keypoints.

        """
        if dropout_mask is None:
            return keypoints
        processor = cast("KeypointsProcessor", self.get_processor("keypoints"))

        if processor is None or not processor.params.remove_invisible:
            return keypoints

        return fdropout.mask_dropout_keypoints(keypoints, dropout_mask)
