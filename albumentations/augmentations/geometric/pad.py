"""Padding transformations for images and related data.

This module provides transformations for padding images and associated data. Padding is the process
of adding pixels to the borders of an image to increase its dimensions. Common use cases include:

- Ensuring uniform sizes for model inputs in a batch
- Making image dimensions divisible by specific values (often required by CNNs)
- Creating space around an image for annotations or visual purposes
- Standardizing data dimensions for processing pipelines

Padding transformations in this module support various border modes (constant, reflection, replication)
and properly handle all target types including images, masks, bounding boxes, and keypoints.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import (
    Field,
    model_validator,
)
from typing_extensions import Self

from albumentations.core.bbox_utils import (
    denormalize_bboxes,
    normalize_bboxes,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import ALL_TARGETS

from . import functional as fgeometric

__all__ = [
    "Pad",
    "PadIfNeeded",
]

NUM_PADS_XY = 2
NUM_PADS_ALL_SIDES = 4


class Pad(DualTransform):
    """Pad the sides of an image by specified number of pixels.

    Args:
        padding (int, tuple[int, int] or tuple[int, int, int, int]): Padding values. Can be:
            * int - pad all sides by this value
            * tuple[int, int] - (pad_x, pad_y) to pad left/right by pad_x and top/bottom by pad_y
            * tuple[int, int, int, int] - (left, top, right, bottom) specific padding per side
        fill (tuple[float, ...] | float): Padding value if border_mode is cv2.BORDER_CONSTANT
        fill_mask (tuple[float, ...] | float): Padding value for mask if border_mode is cv2.BORDER_CONSTANT
        border_mode (OpenCV flag): OpenCV border mode
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    References:
        PyTorch Pad: https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Pad.html

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Example 1: Pad all sides by the same value
        >>> transform = A.Compose([
        ...     A.Pad(padding=20, border_mode=cv2.BORDER_CONSTANT, fill=0),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> padded = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the padded data
        >>> padded_image = padded['image']  # Shape will be (140, 140, 3)
        >>> padded_mask = padded['mask']    # Shape will be (140, 140)
        >>> padded_bboxes = padded['bboxes']  # Bounding boxes coordinates adjusted to the padded image
        >>> padded_keypoints = padded['keypoints']  # Keypoints coordinates adjusted to the padded image
        >>>
        >>> # Example 2: Different padding for sides using (pad_x, pad_y)
        >>> transform_xy = A.Compose([
        ...     A.Pad(
        ...         padding=(10, 30),  # 10px padding on left/right, 30px on top/bottom
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=128  # Gray padding color
        ...     ),
        ... ])
        >>>
        >>> padded_xy = transform_xy(image=image)
        >>> padded_xy_image = padded_xy['image']  # Shape will be (160, 120, 3)
        >>>
        >>> # Example 3: Different padding for each side
        >>> transform_sides = A.Compose([
        ...     A.Pad(
        ...         padding=(5, 10, 15, 20),  # (left, top, right, bottom)
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=0,
        ...         fill_mask=0
        ...     ),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']))
        >>>
        >>> padded_sides = transform_sides(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels
        ... )
        >>>
        >>> padded_sides_image = padded_sides['image']  # Shape will be (130, 120, 3)
        >>> padded_sides_bboxes = padded_sides['bboxes']  # Bounding boxes adjusted to the new coordinates
        >>>
        >>> # Example 4: Using different border_mode options
        >>> # Create a smaller image for better visualization of reflection/wrapping
        >>> small_image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        >>>
        >>> # Reflection padding
        >>> reflect_pad = A.Compose([
        ...     A.Pad(padding=5, border_mode=cv2.BORDER_REFLECT_101),
        ... ])
        >>> reflected = reflect_pad(image=small_image)
        >>> reflected_image = reflected['image']  # Shape will be (20, 20, 3) with reflected edges
        >>>
        >>> # Replicate padding
        >>> replicate_pad = A.Compose([
        ...     A.Pad(padding=5, border_mode=cv2.BORDER_REPLICATE),
        ... ])
        >>> replicated = replicate_pad(image=small_image)
        >>> replicated_image = replicated['image']  # Shape will be (20, 20, 3) with replicated edges
        >>>
        >>> # Example 5: Padding with masks and constant border mode
        >>> binary_mask = np.zeros((50, 50), dtype=np.uint8)
        >>> binary_mask[10:40, 10:40] = 1  # Set center region to 1
        >>>
        >>> mask_transform = A.Compose([
        ...     A.Pad(
        ...         padding=10,
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=0,          # Black padding for image
        ...         fill_mask=0      # Use 0 for mask padding (background)
        ...     ),
        ... ])
        >>>
        >>> padded_mask_result = mask_transform(image=image, mask=binary_mask)
        >>> padded_binary_mask = padded_mask_result['mask']  # Shape will be (70, 70)

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        padding: int | tuple[int, int] | tuple[int, int, int, int]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

    def __init__(
        self,
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.padding = padding
        self.fill = fill
        self.fill_mask = fill_mask
        self.border_mode = border_mode

    def apply(
        self,
        img: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Pad transform to an image.

        Args:
            img (np.ndarray): Image to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
        return fgeometric.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.fill,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Pad transform to a mask.

        Args:
            mask (np.ndarray): Mask to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
        return fgeometric.pad_with_params(
            mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.fill_mask,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Pad transform to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
        image_shape = params["shape"][:2]
        bboxes_np = denormalize_bboxes(bboxes, params["shape"])

        result = fgeometric.pad_bboxes(
            bboxes_np,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            self.border_mode,
            image_shape=image_shape,
        )

        rows, cols = params["shape"][:2]
        return normalize_bboxes(
            result,
            (rows + pad_top + pad_bottom, cols + pad_left + pad_right),
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Pad transform to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
        return fgeometric.pad_keypoints(
            keypoints,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            self.border_mode,
            image_shape=params["shape"][:2],
        )

    def apply_to_images(
        self,
        images: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Pad transform to a batch of images.

        Args:
            images (np.ndarray): Batch of images to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
        return fgeometric.pad_images_with_params(
            images,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.fill,
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): Parameters.
            data (dict[str, Any]): Data.

        Returns:
            dict[str, Any]: Parameters.

        """
        if isinstance(self.padding, Real):
            pad_top = pad_bottom = pad_left = pad_right = self.padding
        elif isinstance(self.padding, (tuple, list)):
            if len(self.padding) == NUM_PADS_XY:
                pad_left = pad_right = self.padding[0]
                pad_top = pad_bottom = self.padding[1]
            elif len(self.padding) == NUM_PADS_ALL_SIDES:
                pad_left, pad_top, pad_right, pad_bottom = self.padding  # type: ignore[misc]
            else:
                raise TypeError(
                    "Padding must be a single number, a pair of numbers, or a quadruple of numbers",
                )
        else:
            raise TypeError(
                "Padding must be a single number, a pair of numbers, or a quadruple of numbers",
            )

        return {
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
        }


class PadIfNeeded(Pad):
    """Pads the sides of an image if the image dimensions are less than the specified minimum dimensions.
    If the `pad_height_divisor` or `pad_width_divisor` is specified, the function additionally ensures
    that the image dimensions are divisible by these values.

    Args:
        min_height (int | None): Minimum desired height of the image. Ensures image height is at least this value.
            If not specified, pad_height_divisor must be provided.
        min_width (int | None): Minimum desired width of the image. Ensures image width is at least this value.
            If not specified, pad_width_divisor must be provided.
        pad_height_divisor (int | None): If set, pads the image height to make it divisible by this value.
            If not specified, min_height must be provided.
        pad_width_divisor (int | None): If set, pads the image width to make it divisible by this value.
            If not specified, min_width must be provided.
        position (Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"]):
            Position where the image is to be placed after padding. Default is 'center'.
        border_mode (int): Specifies the border mode to use if padding is required.
            The default is `cv2.BORDER_CONSTANT`.
        fill (tuple[float, ...] | float | None): Value to fill the border pixels if the border mode
            is `cv2.BORDER_CONSTANT`. Default is None.
        fill_mask (tuple[float, ...] | float | None): Similar to `fill` but used for padding masks. Default is None.
        p (float): Probability of applying the transform. Default is 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - Either `min_height` or `pad_height_divisor` must be set, but not both.
        - Either `min_width` or `pad_width_divisor` must be set, but not both.
        - If `border_mode` is set to `cv2.BORDER_CONSTANT`, `value` must be provided.
        - The transform will maintain consistency across all targets (image, mask, bboxes, keypoints, volume).
        - For bounding boxes, the coordinates will be adjusted to account for the padding.
        - For keypoints, their positions will be shifted according to the padding.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Example 1: Basic usage with min_height and min_width
        >>> transform = A.Compose([
        ...     A.PadIfNeeded(min_height=150, min_width=200, border_mode=cv2.BORDER_CONSTANT, fill=0),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> padded = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the padded data
        >>> padded_image = padded['image']  # Shape will be (150, 200, 3)
        >>> padded_mask = padded['mask']    # Shape will be (150, 200)
        >>> padded_bboxes = padded['bboxes']  # Bounding boxes adjusted for the padded image
        >>> padded_bbox_labels = padded['bbox_labels']  # Labels remain unchanged
        >>> padded_keypoints = padded['keypoints']  # Keypoints adjusted for the padded image
        >>> padded_keypoint_labels = padded['keypoint_labels']  # Labels remain unchanged
        >>>
        >>> # Example 2: Using pad_height_divisor and pad_width_divisor
        >>> # This ensures the output dimensions are divisible by the specified values
        >>> transform_divisor = A.Compose([
        ...     A.PadIfNeeded(
        ...         pad_height_divisor=32,
        ...         pad_width_divisor=32,
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=0
        ...     ),
        ... ])
        >>>
        >>> padded_divisor = transform_divisor(image=image)
        >>> padded_divisor_image = padded_divisor['image']  # Shape will be (128, 128, 3) - divisible by 32
        >>>
        >>> # Example 3: Different position options
        >>> # Create a small recognizable image for better visualization of positioning
        >>> small_image = np.zeros((50, 50, 3), dtype=np.uint8)
        >>> small_image[20:30, 20:30, :] = 255  # White square in the middle
        >>>
        >>> # Top-left positioning
        >>> top_left_pad = A.Compose([
        ...     A.PadIfNeeded(
        ...         min_height=100,
        ...         min_width=100,
        ...         position="top_left",
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=128  # Gray padding
        ...     ),
        ... ])
        >>> top_left_result = top_left_pad(image=small_image)
        >>> top_left_image = top_left_result['image']  # Image will be at top-left of 100x100 canvas
        >>>
        >>> # Center positioning (default)
        >>> center_pad = A.Compose([
        ...     A.PadIfNeeded(
        ...         min_height=100,
        ...         min_width=100,
        ...         position="center",
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=128
        ...     ),
        ... ])
        >>> center_result = center_pad(image=small_image)
        >>> center_image = center_result['image']  # Image will be centered in 100x100 canvas
        >>>
        >>> # Example 4: Different border_mode options
        >>> # Reflection padding
        >>> reflect_pad = A.Compose([
        ...     A.PadIfNeeded(
        ...         min_height=100,
        ...         min_width=100,
        ...         border_mode=cv2.BORDER_REFLECT_101
        ...     ),
        ... ])
        >>> reflected = reflect_pad(image=small_image)
        >>> reflected_image = reflected['image']  # Will use reflection for padding
        >>>
        >>> # Replication padding
        >>> replicate_pad = A.Compose([
        ...     A.PadIfNeeded(
        ...         min_height=100,
        ...         min_width=100,
        ...         border_mode=cv2.BORDER_REPLICATE
        ...     ),
        ... ])
        >>> replicated = replicate_pad(image=small_image)
        >>> replicated_image = replicated['image']  # Will use edge replication for padding
        >>>
        >>> # Example 5: Working with masks and custom fill values
        >>> binary_mask = np.zeros((50, 50), dtype=np.uint8)
        >>> binary_mask[10:40, 10:40] = 1  # Set center region to 1
        >>>
        >>> mask_transform = A.Compose([
        ...     A.PadIfNeeded(
        ...         min_height=100,
        ...         min_width=100,
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=0,          # Black padding for image
        ...         fill_mask=0      # Use 0 for mask padding (background)
        ...     ),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']))
        >>>
        >>> padded_mask_result = mask_transform(
        ...     image=image,
        ...     mask=binary_mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels
        ... )
        >>> padded_binary_mask = padded_mask_result['mask']  # Shape will be (100, 100)
        >>> padded_result_bboxes = padded_mask_result['bboxes']  # Adjusted for padding
        >>> padded_result_bbox_labels = padded_mask_result['bbox_labels']  # Labels remain unchanged

    """

    class InitSchema(BaseTransformInitSchema):
        min_height: int | None = Field(ge=1)
        min_width: int | None = Field(ge=1)
        pad_height_divisor: int | None = Field(ge=1)
        pad_width_divisor: int | None = Field(ge=1)
        position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"]
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

        @model_validator(mode="after")
        def _validate_divisibility(self) -> Self:
            if (self.min_height is None) == (self.pad_height_divisor is None):
                msg = "Only one of 'min_height' and 'pad_height_divisor' parameters must be set"
                raise ValueError(msg)
            if (self.min_width is None) == (self.pad_width_divisor is None):
                msg = "Only one of 'min_width' and 'pad_width_divisor' parameters must be set"
                raise ValueError(msg)

            if self.border_mode == cv2.BORDER_CONSTANT and self.fill is None:
                msg = "If 'border_mode' is set to 'BORDER_CONSTANT', 'fill' must be provided."
                raise ValueError(msg)

            return self

    def __init__(
        self,
        min_height: int | None = 1024,
        min_width: int | None = 1024,
        pad_height_divisor: int | None = None,
        pad_width_divisor: int | None = None,
        position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = "center",
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 1.0,
    ):
        # Initialize with dummy padding that will be calculated later
        super().__init__(
            padding=0,
            fill=fill,
            fill_mask=fill_mask,
            border_mode=border_mode,
            p=p,
        )
        self.min_height = min_height
        self.min_width = min_width
        self.pad_height_divisor = pad_height_divisor
        self.pad_width_divisor = pad_width_divisor
        self.position = position

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): Parameters.
            data (dict[str, Any]): Data.

        Returns:
            dict[str, Any]: Parameters.

        """
        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = fgeometric.get_padding_params(
            image_shape=params["shape"][:2],
            min_height=self.min_height,
            min_width=self.min_width,
            pad_height_divisor=self.pad_height_divisor,
            pad_width_divisor=self.pad_width_divisor,
        )

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = fgeometric.adjust_padding_by_position(
            h_top=h_pad_top,
            h_bottom=h_pad_bottom,
            w_left=w_pad_left,
            w_right=w_pad_right,
            position=self.position,
            py_random=self.py_random,
        )

        return {
            "pad_top": h_pad_top,
            "pad_bottom": h_pad_bottom,
            "pad_left": w_pad_left,
            "pad_right": w_pad_right,
        }
