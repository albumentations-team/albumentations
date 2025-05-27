"""Transforms for resizing images and associated data.

This module provides transform classes for resizing operations, including uniform resizing,
scaling with aspect ratio preservation, and size-constrained transformations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import cv2
import numpy as np
from albucore import batch_transform
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.type_definitions import ALL_TARGETS
from albumentations.core.utils import to_tuple

from . import functional as fgeometric

__all__ = ["LongestMaxSize", "RandomScale", "Resize", "SmallestMaxSize"]


class RandomScale(DualTransform):
    """Randomly resize the input. Output image size is different from the input image size.

    Args:
        scale_limit (float or tuple[float, float]): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - The output image size is different from the input image size.
        - Scale factor is sampled independently per image side (width and height).
        - Bounding box coordinates are scaled accordingly.
        - Keypoint coordinates are scaled accordingly.
        - When area_for_downscale is set, INTER_AREA interpolation will be used automatically for
          downscaling (scale < 1.0), which provides better quality for size reduction.

    Mathematical formulation:
        Let (W, H) be the original image dimensions and (W', H') be the output dimensions.
        The scale factor s is sampled from the range [1 + scale_limit[0], 1 + scale_limit[1]].
        Then, W' = W * s and H' = H * s.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create sample data for demonstration
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize scaling effects
        >>> cv2.rectangle(image, (25, 25), (75, 75), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (50, 50), 10, (0, 255, 0), -1)  # Green circle
        >>>
        >>> # Create a mask for segmentation
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 1  # Mask covering the red square
        >>>
        >>> # Create bounding boxes and keypoints
        >>> bboxes = np.array([[25, 25, 75, 75]])  # Box around the red square
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[50, 50]])  # Center of circle
        >>> keypoint_labels = [0]
        >>>
        >>> # Apply RandomScale transform with comprehensive parameters
        >>> transform = A.Compose([
        ...     A.RandomScale(
        ...         scale_limit=(-0.3, 0.5),     # Scale between 0.7x and 1.5x
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",  # Use INTER_AREA for image downscaling
        ...         p=1.0                         # Always apply
        ...     )
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform to all targets
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed results
        >>> scaled_image = result['image']        # Dimensions will be between 70-150 pixels
        >>> scaled_mask = result['mask']          # Mask scaled proportionally to image
        >>> scaled_bboxes = result['bboxes']      # Bounding boxes adjusted to new dimensions
        >>> scaled_bbox_labels = result['bbox_labels']  # Labels remain unchanged
        >>> scaled_keypoints = result['keypoints']      # Keypoints adjusted to new dimensions
        >>> scaled_keypoint_labels = result['keypoint_labels']  # Labels remain unchanged
        >>>
        >>> # The image dimensions will vary based on the randomly sampled scale factor
        >>> # With scale_limit=(-0.3, 0.5), dimensions could be anywhere from 70% to 150% of original

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        scale_limit: tuple[float, float] | float
        area_for_downscale: Literal[None, "image", "image_mask"]
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]

        @field_validator("scale_limit")
        @classmethod
        def _check_scale_limit(cls, v: tuple[float, float] | float) -> tuple[float, float]:
            return to_tuple(v)

    def __init__(
        self,
        scale_limit: tuple[float, float] | float = (-0.1, 0.1),
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_NEAREST,
        area_for_downscale: Literal[None, "image", "image_mask"] = None,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.scale_limit = cast("tuple[float, float]", scale_limit)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.area_for_downscale = area_for_downscale

    def get_params(self) -> dict[str, float]:
        """Get parameters for the transform.

        Returns:
            dict[str, float]: Dictionary with parameters.

        """
        return {"scale": self.py_random.uniform(*self.scale_limit) + 1.0}

    def apply(
        self,
        img: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        """Apply scaling to the image.

        Args:
            img (np.ndarray): Image to scale.
            scale (float): Scaling factor.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Scaled image.

        """
        interpolation = self.interpolation
        if self.area_for_downscale in ["image", "image_mask"] and scale < 1.0:
            interpolation = cv2.INTER_AREA

        return fgeometric.scale(img, scale, interpolation)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        """Apply scaling to the mask.

        Args:
            mask (np.ndarray): Mask to scale.
            scale (float): Scaling factor.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Scaled mask.

        """
        interpolation = self.mask_interpolation
        if self.area_for_downscale == "image_mask" and scale < 1.0:
            interpolation = cv2.INTER_AREA

        return fgeometric.scale(mask, scale, interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the transform to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to transform.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transformed bounding boxes which are scale invariant.

        """
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        """Apply scaling to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to scale.
            scale (float): Scaling factor.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Scaled keypoints.

        """
        return fgeometric.keypoints_scale(keypoints, scale, scale)


class MaxSizeTransform(DualTransform):
    """Base class for transforms that resize based on maximum size constraints.

    This class provides common functionality for derived transforms like LongestMaxSize and
    SmallestMaxSize that resize images based on size constraints while preserving aspect ratio.

    Args:
        max_size (int, Sequence[int], optional): Maximum size constraint. The specific interpretation
            depends on the derived class. Default: None.
        max_size_hw (tuple[int | None, int | None], optional): Maximum (height, width) constraints.
            Either max_size or max_size_hw must be specified, but not both. Default: None.
        interpolation (OpenCV flag): Flag for the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): Flag for the mask interpolation algorithm.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): Probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - This is a base class that should be extended by concrete resize transforms.
        - The scaling calculation is implemented in derived classes.
        - Aspect ratio is preserved by applying the same scale factor to both dimensions.
        - When area_for_downscale is set, INTER_AREA interpolation will be used automatically for
          downscaling (scale < 1.0), which provides better quality for size reduction.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Example of creating a custom transform that extends MaxSizeTransform
        >>> class CustomMaxSize(A.MaxSizeTransform):
        ...     def get_params_dependent_on_data(self, params, data):
        ...         img_h, img_w = params["shape"][:2]
        ...         # Calculate scale factor - here we scale to make the image area constant
        ...         target_area = 300 * 300  # Target area of 300x300
        ...         current_area = img_h * img_w
        ...         scale = np.sqrt(target_area / current_area)
        ...         return {"scale": scale}
        >>>
        >>> # Prepare sample data
        >>> image = np.zeros((100, 200, 3), dtype=np.uint8)
        >>> # Add a rectangle to visualize the effect
        >>> cv2.rectangle(image, (50, 20), (150, 80), (255, 0, 0), -1)
        >>>
        >>> # Create a mask
        >>> mask = np.zeros((100, 200), dtype=np.uint8)
        >>> mask[20:80, 50:150] = 1
        >>>
        >>> # Create bounding boxes and keypoints
        >>> bboxes = np.array([[50, 20, 150, 80]])
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[100, 50]])
        >>> keypoint_labels = [0]
        >>>
        >>> # Apply the custom transform
        >>> transform = A.Compose([
        ...     CustomMaxSize(
        ...         max_size=None,
        ...         max_size_hw=(None, None),  # Not used in our custom implementation
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",  # Use INTER_AREA when downscaling images
        ...         p=1.0
        ...     )
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get results
        >>> transformed_image = result['image']  # Shape will be approximately (122, 245, 3)
        >>> transformed_mask = result['mask']    # Shape will be approximately (122, 245)
        >>> transformed_bboxes = result['bboxes']  # Bounding boxes are scale invariant
        >>> transformed_keypoints = result['keypoints']  # Keypoints scaled proportionally
        >>> transformed_bbox_labels = result['bbox_labels']  # Labels remain unchanged
        >>> transformed_keypoint_labels = result['keypoint_labels']  # Labels remain unchanged

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        max_size: int | list[int] | None
        max_size_hw: tuple[int | None, int | None] | None
        area_for_downscale: Literal[None, "image", "image_mask"]
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]

        @model_validator(mode="after")
        def validate_size_parameters(self) -> Self:
            if self.max_size is None and self.max_size_hw is None:
                raise ValueError("Either max_size or max_size_hw must be specified")
            if self.max_size is not None and self.max_size_hw is not None:
                raise ValueError("Only one of max_size or max_size_hw should be specified")
            return self

    def __init__(
        self,
        max_size: int | Sequence[int] | None = None,
        max_size_hw: tuple[int | None, int | None] | None = None,
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_NEAREST,
        area_for_downscale: Literal[None, "image", "image_mask"] = None,
        p: float = 1,
    ):
        super().__init__(p=p)
        self.max_size = max_size
        self.max_size_hw = max_size_hw
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.area_for_downscale = area_for_downscale

    def apply(
        self,
        img: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        height, width = img.shape[:2]
        new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))

        interpolation = self.interpolation
        if self.area_for_downscale in ["image", "image_mask"] and scale < 1.0:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(img, (new_height, new_width), interpolation=interpolation)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        height, width = mask.shape[:2]
        new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))

        interpolation = self.mask_interpolation
        if self.area_for_downscale == "image_mask" and scale < 1.0:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(mask, (new_height, new_width), interpolation=interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.keypoints_scale(keypoints, scale, scale)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(images, *args, **params)

    @batch_transform("spatial", has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(volume, *args, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(volumes, *args, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_mask3d(self, mask3d: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply_to_mask(mask3d, *args, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_masks3d(self, masks3d: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply_to_mask(masks3d, *args, **params)


class LongestMaxSize(MaxSizeTransform):
    """Rescale an image so that the longest side is equal to max_size or sides meet max_size_hw constraints,
        keeping the aspect ratio.

    Args:
        max_size (int, Sequence[int], optional): Maximum size of the longest side after the transformation.
            When using a list or tuple, the max size will be randomly selected from the values provided. Default: None.
        max_size_hw (tuple[int | None, int | None], optional): Maximum (height, width) constraints. Supports:
            - (height, width): Both dimensions must fit within these bounds
            - (height, None): Only height is constrained, width scales proportionally
            - (None, width): Only width is constrained, height scales proportionally
            If specified, max_size must be None. Default: None.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - If the longest side of the image is already equal to max_size, the image will not be resized.
        - This transform will not crop the image. The resulting image may be smaller than specified in both dimensions.
        - For non-square images, both sides will be scaled proportionally to maintain the aspect ratio.
        - Bounding boxes and keypoints are scaled accordingly.
        - When area_for_downscale is set, INTER_AREA will be used for downscaling, providing better quality.

    Mathematical Details:
        Let (W, H) be the original width and height of the image.

        When using max_size:
            1. The scaling factor s is calculated as:
               s = max_size / max(W, H)
            2. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

        When using max_size_hw=(H_target, W_target):
            1. For both dimensions specified:
               s = min(H_target/H, W_target/W)
               This ensures both dimensions fit within the specified bounds.

            2. For height only (W_target=None):
               s = H_target/H
               Width will scale proportionally.

            3. For width only (H_target=None):
               s = W_target/W
               Height will scale proportionally.

            4. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

    Examples:
        >>> import albumentations as A
        >>> import cv2
        >>> # Using max_size
        >>> transform1 = A.LongestMaxSize(max_size=1024, area_for_downscale="image")
        >>> # Input image (1500, 800) -> Output (1024, 546)
        >>>
        >>> # Using max_size_hw with both dimensions
        >>> transform2 = A.LongestMaxSize(max_size_hw=(800, 1024), area_for_downscale="image_mask")
        >>> # Input (1500, 800) -> Output (800, 427)
        >>> # Input (800, 1500) -> Output (546, 1024)
        >>>
        >>> # Using max_size_hw with only height
        >>> transform3 = A.LongestMaxSize(max_size_hw=(800, None))
        >>> # Input (1500, 800) -> Output (800, 427)
        >>>
        >>> # Common use case with padding
        >>> transform4 = A.Compose([
        ...     A.LongestMaxSize(max_size=1024, area_for_downscale="image"),
        ...     A.PadIfNeeded(min_height=1024, min_width=1024),
        ... ])

    """

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Calculate parameters that depend on the input data.

        Args:
            params (dict[str, Any]): Parameters dictionary.
            data (dict[str, Any]): Dictionary containing input data.

        Returns:
            dict[str, Any]: Dictionary with parameters calculated based on input data.

        """
        img_h, img_w = params["shape"][:2]

        if self.max_size is not None:
            if isinstance(self.max_size, (list, tuple)):
                max_size = self.py_random.choice(self.max_size)
            else:
                max_size = self.max_size
            scale = max_size / max(img_h, img_w)
        elif self.max_size_hw is not None:
            # We know max_size_hw is not None here due to model validator
            max_h, max_w = self.max_size_hw
            if max_h is not None and max_w is not None:
                # Scale based on longest side to maintain aspect ratio
                h_scale = max_h / img_h
                w_scale = max_w / img_w
                scale = min(h_scale, w_scale)
            elif max_h is not None:
                # Only height specified
                scale = max_h / img_h
            else:
                # Only width specified
                scale = max_w / img_w

        return {"scale": scale}


class SmallestMaxSize(MaxSizeTransform):
    """Rescale an image so that minimum side is equal to max_size or sides meet max_size_hw constraints,
    keeping the aspect ratio.

    Args:
        max_size (int, list of int, optional): Maximum size of smallest side of the image after the transformation.
            When using a list, max size will be randomly selected from the values in the list. Default: None.
        max_size_hw (tuple[int | None, int | None], optional): Maximum (height, width) constraints. Supports:
            - (height, width): Both dimensions must be at least these values
            - (height, None): Only height is constrained, width scales proportionally
            - (None, width): Only width is constrained, height scales proportionally
            If specified, max_size must be None. Default: None.
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): Probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - If the smallest side of the image is already equal to max_size, the image will not be resized.
        - This transform will not crop the image. The resulting image may be larger than specified in both dimensions.
        - For non-square images, both sides will be scaled proportionally to maintain the aspect ratio.
        - Bounding boxes and keypoints are scaled accordingly.
        - When area_for_downscale is set, INTER_AREA will be used for downscaling, providing better quality.

    Mathematical Details:
        Let (W, H) be the original width and height of the image.

        When using max_size:
            1. The scaling factor s is calculated as:
               s = max_size / min(W, H)
            2. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

        When using max_size_hw=(H_target, W_target):
            1. For both dimensions specified:
               s = max(H_target/H, W_target/W)
               This ensures both dimensions are at least as large as specified.

            2. For height only (W_target=None):
               s = H_target/H
               Width will scale proportionally.

            3. For width only (H_target=None):
               s = W_target/W
               Height will scale proportionally.

            4. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> # Using max_size
        >>> transform1 = A.SmallestMaxSize(max_size=120, area_for_downscale="image")
        >>> # Input image (100, 150) -> Output (120, 180)
        >>>
        >>> # Using max_size_hw with both dimensions
        >>> transform2 = A.SmallestMaxSize(max_size_hw=(100, 200), area_for_downscale="image_mask")
        >>> # Input (80, 160) -> Output (100, 200)
        >>> # Input (160, 80) -> Output (400, 200)
        >>>
        >>> # Using max_size_hw with only height
        >>> transform3 = A.SmallestMaxSize(max_size_hw=(100, None))
        >>> # Input (80, 160) -> Output (100, 200)

    """

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Calculate parameters that depend on the input data.

        Args:
            params (dict[str, Any]): Parameters dictionary.
            data (dict[str, Any]): Dictionary containing input data.

        Returns:
            dict[str, Any]: Dictionary with parameters calculated based on input data.

        """
        img_h, img_w = params["shape"][:2]

        if self.max_size is not None:
            if isinstance(self.max_size, (list, tuple)):
                max_size = self.py_random.choice(self.max_size)
            else:
                max_size = self.max_size
            scale = max_size / min(img_h, img_w)
        elif self.max_size_hw is not None:
            max_h, max_w = self.max_size_hw
            if max_h is not None and max_w is not None:
                # Scale based on smallest side to maintain aspect ratio
                h_scale = max_h / img_h
                w_scale = max_w / img_w
                scale = max(h_scale, w_scale)
            elif max_h is not None:
                # Only height specified
                scale = max_h / img_h
            else:
                # Only width specified
                scale = max_w / img_w

        return {"scale": scale}


class Resize(DualTransform):
    """Resize the input to the given height and width.

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        area_for_downscale (Literal[None, "image", "image_mask"]): Controls automatic use of INTER_AREA interpolation
            for downscaling. Options:
            - None: No automatic interpolation selection, always use the specified interpolation method
            - "image": Use INTER_AREA when downscaling images, retain specified interpolation for upscaling and masks
            - "image_mask": Use INTER_AREA when downscaling both images and masks
            Default: None.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create sample data for demonstration
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize resize effects
        >>> cv2.rectangle(image, (25, 25), (75, 75), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (50, 50), 10, (0, 255, 0), -1)  # Green circle
        >>>
        >>> # Create a mask for segmentation
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 1  # Mask covering the red square
        >>>
        >>> # Create bounding boxes and keypoints
        >>> bboxes = np.array([[25, 25, 75, 75]])  # Box around the red square
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[50, 50]])  # Center of circle
        >>> keypoint_labels = [0]
        >>>
        >>> # Resize all data to 224x224 (common input size for many CNNs)
        >>> transform = A.Compose([
        ...     A.Resize(
        ...         height=224,
        ...         width=224,
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         area_for_downscale="image",  # Use INTER_AREA when downscaling images
        ...         p=1.0
        ...     )
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform to all targets
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed results
        >>> resized_image = result['image']        # Shape will be (224, 224, 3)
        >>> resized_mask = result['mask']          # Shape will be (224, 224)
        >>> resized_bboxes = result['bboxes']      # Bounding boxes scaled to new dimensions
        >>> resized_bbox_labels = result['bbox_labels']  # Labels remain unchanged
        >>> resized_keypoints = result['keypoints']      # Keypoints scaled to new dimensions
        >>> resized_keypoint_labels = result['keypoint_labels']  # Labels remain unchanged
        >>>
        >>> # Note: When resizing from 100x100 to 224x224:
        >>> # - The red square will be scaled from (25-75) to approximately (56-168)
        >>> # - The keypoint at (50, 50) will move to approximately (112, 112)
        >>> # - All spatial relationships are preserved but coordinates are scaled

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        height: int = Field(ge=1)
        width: int = Field(ge=1)
        area_for_downscale: Literal[None, "image", "image_mask"]
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_NEAREST,
        area_for_downscale: Literal[None, "image", "image_mask"] = None,
        p: float = 1,
    ):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.area_for_downscale = area_for_downscale

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply resizing to the image.

        Args:
            img (np.ndarray): Image to resize.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Resized image.

        """
        height, width = img.shape[:2]
        is_downscale = (self.height < height) or (self.width < width)

        interpolation = self.interpolation
        if self.area_for_downscale in ["image", "image_mask"] and is_downscale:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(img, (self.height, self.width), interpolation=interpolation)

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        """Apply resizing to the mask.

        Args:
            mask (np.ndarray): Mask to resize.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Resized mask.

        """
        height, width = mask.shape[:2]
        is_downscale = (self.height < height) or (self.width < width)

        interpolation = self.mask_interpolation
        if self.area_for_downscale == "image_mask" and is_downscale:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(mask, (self.height, self.width), interpolation=interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the transform to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to transform.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transformed bounding boxes which are scale invariant.

        """
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        """Apply resizing to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to resize.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Resized keypoints.

        """
        height, width = params["shape"][:2]
        scale_x = self.width / width
        scale_y = self.height / height
        return fgeometric.keypoints_scale(keypoints, scale_x, scale_y)
