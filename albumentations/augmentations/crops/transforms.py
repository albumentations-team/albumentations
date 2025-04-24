"""Transform classes for cropping operations on images and other data types.

This module provides various crop transforms that can be applied to images, masks,
bounding boxes, and keypoints. The transforms include simple cropping, random cropping,
center cropping, cropping near bounding boxes, and other specialized cropping operations
that maintain the integrity of bounding boxes. These transforms are designed to work within
the albumentations pipeline and can be used for data augmentation in computer vision tasks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Union, cast

import cv2
import numpy as np
from pydantic import AfterValidator, Field, model_validator
from typing_extensions import Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes, union_of_bboxes
from albumentations.core.pydantic import (
    OnePlusIntRangeType,
    ZeroOneRangeType,
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.type_definitions import (
    ALL_TARGETS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    PAIR,
    PercentType,
    PxType,
)

from . import functional as fcrops

__all__ = [
    "AtLeastOneBBoxRandomCrop",
    "BBoxSafeRandomCrop",
    "CenterCrop",
    "Crop",
    "CropAndPad",
    "CropNonEmptyMaskIfExists",
    "RandomCrop",
    "RandomCropFromBorders",
    "RandomCropNearBBox",
    "RandomResizedCrop",
    "RandomSizedBBoxSafeCrop",
    "RandomSizedCrop",
]


class CropSizeError(Exception):
    pass


class BaseCrop(DualTransform):
    """Base class for transforms that only perform cropping."""

    _targets = ALL_TARGETS

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop transform to an image.

        Args:
            img (np.ndarray): The image to apply the crop transform to.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The cropped image.

        """
        return fcrops.crop(img, x_min=crop_coords[0], y_min=crop_coords[1], x_max=crop_coords[2], y_max=crop_coords[3])

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop transform to bounding boxes.

        Args:
            bboxes (np.ndarray): The bounding boxes to apply the crop transform to.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The cropped bounding boxes.

        """
        return fcrops.crop_bboxes_by_coords(bboxes, crop_coords, params["shape"][:2])

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop transform to keypoints.

        Args:
            keypoints (np.ndarray): The keypoints to apply the crop transform to.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The cropped keypoints.

        """
        return fcrops.crop_keypoints_by_coords(keypoints, crop_coords)

    def apply_to_images(
        self,
        images: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return fcrops.volume_crop_yx(images, crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3])

    def apply_to_volume(
        self,
        volume: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return self.apply_to_images(volume, crop_coords, **params)

    def apply_to_volumes(
        self,
        volumes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return fcrops.volumes_crop_yx(volumes, crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3])

    def apply_to_mask3d(
        self,
        mask3d: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return self.apply_to_images(mask3d, crop_coords, **params)

    def apply_to_masks3d(
        self,
        masks3d: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return self.apply_to_volumes(masks3d, crop_coords, **params)

    @staticmethod
    def _clip_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        height, width = image_shape[:2]
        x_min, y_min, x_max, y_max = bbox
        x_min = np.clip(x_min, 0, width)
        y_min = np.clip(y_min, 0, height)

        x_max = np.clip(x_max, x_min, width)
        y_max = np.clip(y_max, y_min, height)
        return x_min, y_min, x_max, y_max


class BaseCropAndPad(BaseCrop):
    """Base class for transforms that need both cropping and padding."""

    class InitSchema(BaseTransformInitSchema):
        pad_if_needed: bool
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"]

    def __init__(
        self,
        pad_if_needed: bool,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ],
        fill: tuple[float, ...] | float,
        fill_mask: tuple[float, ...] | float,
        pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"],
        p: float,
    ):
        super().__init__(p=p)
        self.pad_if_needed = pad_if_needed
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask
        self.pad_position = pad_position

    def _get_pad_params(self, image_shape: tuple[int, int], target_shape: tuple[int, int]) -> dict[str, Any] | None:
        """Calculate padding parameters if needed."""
        if not self.pad_if_needed:
            return None

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = fgeometric.get_padding_params(
            image_shape=image_shape,
            min_height=target_shape[0],
            min_width=target_shape[1],
            pad_height_divisor=None,
            pad_width_divisor=None,
        )

        if h_pad_top == h_pad_bottom == w_pad_left == w_pad_right == 0:
            return None

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = fgeometric.adjust_padding_by_position(
            h_top=h_pad_top,
            h_bottom=h_pad_bottom,
            w_left=w_pad_left,
            w_right=w_pad_right,
            position=self.pad_position,
            py_random=self.py_random,
        )

        return {
            "pad_top": h_pad_top,
            "pad_bottom": h_pad_bottom,
            "pad_left": w_pad_left,
            "pad_right": w_pad_right,
        }

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to an image.

        Args:
            img (np.ndarray): The image to apply the crop and pad transform to.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The cropped and padded image.

        """
        pad_params = params.get("pad_params")
        if pad_params is not None:
            img = fgeometric.pad_with_params(
                img,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                border_mode=self.border_mode,
                value=self.fill,
            )
        return BaseCrop.apply(self, img, crop_coords, **params)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        crop_coords: Any,
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to a mask.

        Args:
            mask (np.ndarray): The mask to apply the crop and pad transform to.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The cropped and padded mask.

        """
        pad_params = params.get("pad_params")
        if pad_params is not None:
            mask = fgeometric.pad_with_params(
                mask,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                border_mode=self.border_mode,
                value=self.fill_mask,
            )
        # Note' that super().apply would apply the padding twice as it is looped to this.apply
        return BaseCrop.apply(self, mask, crop_coords=crop_coords, **params)

    def apply_to_images(
        self,
        images: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        pad_params = params.get("pad_params")
        if pad_params is not None:
            images = fcrops.pad_along_axes(
                images,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                h_axis=2,
                w_axis=3,
                border_mode=self.border_mode,
                pad_value=self.fill,
            )
        return BaseCrop.apply_to_images(self, images, crop_coords, **params)

    def apply_to_volume(
        self,
        volume: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return self.apply_to_images(volume, crop_coords, **params)

    def apply_to_volumes(
        self,
        volumes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        pad_params = params.get("pad_params")
        if pad_params is not None:
            volumes = fcrops.pad_along_axes(
                volumes,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                h_axis=3,
                w_axis=4,
                border_mode=self.border_mode,
                pad_value=self.fill,
            )
        return BaseCrop.apply_to_volumes(self, volumes, crop_coords, **params)

    def apply_to_mask3d(
        self,
        mask3d: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return self.apply_to_images(mask3d, crop_coords, **params)

    def apply_to_masks3d(
        self,
        masks3d: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        return self.apply_to_volumes(masks3d, crop_coords, **params)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to bounding boxes.

        Args:
            bboxes (np.ndarray): The bounding boxes to apply the crop and pad transform to.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The cropped and padded bounding boxes.

        """
        pad_params = params.get("pad_params")
        image_shape = params["shape"][:2]

        if pad_params is not None:
            # First denormalize bboxes to absolute coordinates
            bboxes_np = denormalize_bboxes(bboxes, image_shape)

            # Apply padding to bboxes (already works with absolute coordinates)
            bboxes_np = fgeometric.pad_bboxes(
                bboxes_np,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                self.border_mode,
                image_shape=image_shape,
            )

            # Update shape to padded dimensions
            padded_height = image_shape[0] + pad_params["pad_top"] + pad_params["pad_bottom"]
            padded_width = image_shape[1] + pad_params["pad_left"] + pad_params["pad_right"]
            padded_shape = (padded_height, padded_width)

            bboxes_np = normalize_bboxes(bboxes_np, padded_shape)

            params["shape"] = padded_shape

            return BaseCrop.apply_to_bboxes(self, bboxes_np, crop_coords, **params)

        # If no padding, use original function behavior
        return BaseCrop.apply_to_bboxes(self, bboxes, crop_coords, **params)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to keypoints.

        Args:
            keypoints (np.ndarray): The keypoints to apply the crop and pad transform to.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The cropped and padded keypoints.

        """
        pad_params = params.get("pad_params")
        image_shape = params["shape"][:2]

        if pad_params is not None:
            # Calculate padded dimensions
            padded_height = image_shape[0] + pad_params["pad_top"] + pad_params["pad_bottom"]
            padded_width = image_shape[1] + pad_params["pad_left"] + pad_params["pad_right"]

            # First apply padding to keypoints using original image shape
            keypoints = fgeometric.pad_keypoints(
                keypoints,
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
                self.border_mode,
                image_shape=image_shape,
            )

            # Update image shape for subsequent crop operation
            params = {**params, "shape": (padded_height, padded_width)}

        return BaseCrop.apply_to_keypoints(self, keypoints, crop_coords, **params)


class RandomCrop(BaseCropAndPad):
    """Crop a random part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        pad_if_needed (bool): Whether to pad if crop size exceeds image size. Default: False.
        border_mode (OpenCV flag): OpenCV border mode used for padding. Default: cv2.BORDER_CONSTANT.
        fill (tuple[float, ...] | float): Padding value for images if border_mode is
            cv2.BORDER_CONSTANT. Default: 0.
        fill_mask (tuple[float, ...] | float): Padding value for masks if border_mode is
            cv2.BORDER_CONSTANT. Default: 0.
        pad_position (Literal['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'random']):
            Position of padding. Default: 'center'.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        If pad_if_needed is True and crop size exceeds image dimensions, the image will be padded
        before applying the random crop.

    """

    class InitSchema(BaseCropAndPad.InitSchema):
        height: Annotated[int, Field(ge=1)]
        width: Annotated[int, Field(ge=1)]
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        height: int,
        width: int,
        pad_if_needed: bool = False,
        pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = "center",
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0.0,
        fill_mask: tuple[float, ...] | float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(
            pad_if_needed=pad_if_needed,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            pad_position=pad_position,
            p=p,
        )
        self.height = height
        self.width = width

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:  # Changed return type to be more flexible
        """Get parameters that depend on input data.

        Args:
            params (dict[str, Any]): Parameters.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, Any]: Dictionary with parameters.

        """
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        if not self.pad_if_needed and (self.height > image_height or self.width > image_width):
            raise CropSizeError(
                f"Crop size (height, width) exceeds image dimensions (height, width):"
                f" {(self.height, self.width)} vs {image_shape[:2]}",
            )

        # Get padding params first if needed
        pad_params = self._get_pad_params(image_shape, (self.height, self.width))

        # If padding is needed, adjust the image shape for crop calculation
        if pad_params is not None:
            pad_top = pad_params["pad_top"]
            pad_bottom = pad_params["pad_bottom"]
            pad_left = pad_params["pad_left"]
            pad_right = pad_params["pad_right"]

            padded_height = image_height + pad_top + pad_bottom
            padded_width = image_width + pad_left + pad_right
            padded_shape = (padded_height, padded_width)

            # Get random crop coordinates based on padded dimensions
            h_start = self.py_random.random()
            w_start = self.py_random.random()
            crop_coords = fcrops.get_crop_coords(padded_shape, (self.height, self.width), h_start, w_start)
        else:
            # Get random crop coordinates based on original dimensions
            h_start = self.py_random.random()
            w_start = self.py_random.random()
            crop_coords = fcrops.get_crop_coords(image_shape, (self.height, self.width), h_start, w_start)

        return {
            "crop_coords": crop_coords,
            "pad_params": pad_params,
        }


class CenterCrop(BaseCropAndPad):
    """Crop the central part of the input.

    This transform crops the center of the input image, mask, bounding boxes, and keypoints to the specified dimensions.
    It's useful when you want to focus on the central region of the input, discarding peripheral information.

    Args:
        height (int): The height of the crop. Must be greater than 0.
        width (int): The width of the crop. Must be greater than 0.
        pad_if_needed (bool): Whether to pad if crop size exceeds image size. Default: False.
        border_mode (OpenCV flag): OpenCV border mode used for padding. Default: cv2.BORDER_CONSTANT.
        fill (tuple[float, ...] | float): Padding value for images if border_mode is
            cv2.BORDER_CONSTANT. Default: 0.
        fill_mask (tuple[float, ...] | float): Padding value for masks if border_mode is
            cv2.BORDER_CONSTANT. Default: 0.
        pad_position (Literal['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'random']):
            Position of padding. Default: 'center'.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - If pad_if_needed is False and crop size exceeds image dimensions, it will raise a CropSizeError.
        - If pad_if_needed is True and crop size exceeds image dimensions, the image will be padded.
        - For bounding boxes and keypoints, coordinates are adjusted appropriately for both padding and cropping.

    """

    class InitSchema(BaseCropAndPad.InitSchema):
        height: Annotated[int, Field(ge=1)]
        width: Annotated[int, Field(ge=1)]
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        height: int,
        width: int,
        pad_if_needed: bool = False,
        pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = "center",
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0.0,
        fill_mask: tuple[float, ...] | float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(
            pad_if_needed=pad_if_needed,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            pad_position=pad_position,
            p=p,
        )
        self.height = height
        self.width = width

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data of the transform.

        """
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        if not self.pad_if_needed and (self.height > image_height or self.width > image_width):
            raise CropSizeError(
                f"Crop size (height, width) exceeds image dimensions (height, width):"
                f" {(self.height, self.width)} vs {image_shape[:2]}",
            )

        # Get padding params first if needed
        pad_params = self._get_pad_params(image_shape, (self.height, self.width))

        # If padding is needed, adjust the image shape for crop calculation
        if pad_params is not None:
            pad_top = pad_params["pad_top"]
            pad_bottom = pad_params["pad_bottom"]
            pad_left = pad_params["pad_left"]
            pad_right = pad_params["pad_right"]

            padded_height = image_height + pad_top + pad_bottom
            padded_width = image_width + pad_left + pad_right
            padded_shape = (padded_height, padded_width)

            # Get crop coordinates based on padded dimensions
            crop_coords = fcrops.get_center_crop_coords(padded_shape, (self.height, self.width))
        else:
            # Get crop coordinates based on original dimensions
            crop_coords = fcrops.get_center_crop_coords(image_shape, (self.height, self.width))

        return {
            "crop_coords": crop_coords,
            "pad_params": pad_params,
        }


class Crop(BaseCropAndPad):
    """Crop a specific region from the input image.

    This transform crops a rectangular region from the input image, mask, bounding boxes, and keypoints
    based on specified coordinates. It's useful when you want to extract a specific area of interest
    from your inputs.

    Args:
        x_min (int): Minimum x-coordinate of the crop region (left edge). Must be >= 0. Default: 0.
        y_min (int): Minimum y-coordinate of the crop region (top edge). Must be >= 0. Default: 0.
        x_max (int): Maximum x-coordinate of the crop region (right edge). Must be > x_min. Default: 1024.
        y_max (int): Maximum y-coordinate of the crop region (bottom edge). Must be > y_min. Default: 1024.
        pad_if_needed (bool): Whether to pad if crop coordinates exceed image dimensions. Default: False.
        border_mode (OpenCV flag): OpenCV border mode used for padding. Default: cv2.BORDER_CONSTANT.
        fill (tuple[float, ...] | float): Padding value if border_mode is cv2.BORDER_CONSTANT. Default: 0.
        fill_mask (tuple[float, ...] | float): Padding value for masks. Default: 0.
        pad_position (Literal['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'random']):
            Position of padding. Default: 'center'.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - The crop coordinates are applied as follows: x_min <= x < x_max and y_min <= y < y_max.
        - If pad_if_needed is False and crop region extends beyond image boundaries, it will be clipped.
        - If pad_if_needed is True, image will be padded to accommodate the full crop region.
        - For bounding boxes and keypoints, coordinates are adjusted appropriately for both padding and cropping.

    """

    class InitSchema(BaseCropAndPad.InitSchema):
        x_min: Annotated[int, Field(ge=0)]
        y_min: Annotated[int, Field(ge=0)]
        x_max: Annotated[int, Field(gt=0)]
        y_max: Annotated[int, Field(gt=0)]
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
        def _validate_coordinates(self) -> Self:
            if not self.x_min < self.x_max:
                msg = "x_max must be greater than x_min"
                raise ValueError(msg)
            if not self.y_min < self.y_max:
                msg = "y_max must be greater than y_min"
                raise ValueError(msg)

            return self

    def __init__(
        self,
        x_min: int = 0,
        y_min: int = 0,
        x_max: int = 1024,
        y_max: int = 1024,
        pad_if_needed: bool = False,
        pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = "center",
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
        super().__init__(
            pad_if_needed=pad_if_needed,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            pad_position=pad_position,
            p=p,
        )
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    # New helper function for computing minimum padding
    def _compute_min_padding(self, image_height: int, image_width: int) -> tuple[int, int, int, int]:
        pad_top = 0
        pad_bottom = max(0, self.y_max - image_height)
        pad_left = 0
        pad_right = max(0, self.x_max - image_width)
        return pad_top, pad_bottom, pad_left, pad_right

    # New helper function for distributing and adjusting padding
    def _compute_adjusted_padding(self, pad_top: int, pad_bottom: int, pad_left: int, pad_right: int) -> dict[str, int]:
        delta_h = pad_top + pad_bottom
        delta_w = pad_left + pad_right
        pad_top_dist = delta_h // 2
        pad_bottom_dist = delta_h - pad_top_dist
        pad_left_dist = delta_w // 2
        pad_right_dist = delta_w - pad_left_dist

        (pad_top_adj, pad_bottom_adj, pad_left_adj, pad_right_adj) = fgeometric.adjust_padding_by_position(
            h_top=pad_top_dist,
            h_bottom=pad_bottom_dist,
            w_left=pad_left_dist,
            w_right=pad_right_dist,
            position=self.pad_position,
            py_random=self.py_random,
        )

        final_top = max(pad_top_adj, pad_top)
        final_bottom = max(pad_bottom_adj, pad_bottom)
        final_left = max(pad_left_adj, pad_left)
        final_right = max(pad_right_adj, pad_right)

        return {
            "pad_top": final_top,
            "pad_bottom": final_bottom,
            "pad_left": final_left,
            "pad_right": final_right,
        }

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Get parameters for crop.

        Args:
            params (dict): Dictionary with parameters for crop.
            data (dict): Dictionary with data.

        Returns:
            dict: Dictionary with parameters for crop.

        """
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        if not self.pad_if_needed:
            return {"crop_coords": (self.x_min, self.y_min, self.x_max, self.y_max), "pad_params": None}

        pad_top, pad_bottom, pad_left, pad_right = self._compute_min_padding(image_height, image_width)
        pad_params = None

        if any([pad_top, pad_bottom, pad_left, pad_right]):
            pad_params = self._compute_adjusted_padding(pad_top, pad_bottom, pad_left, pad_right)

        return {"crop_coords": (self.x_min, self.y_min, self.x_max, self.y_max), "pad_params": pad_params}


class CropNonEmptyMaskIfExists(BaseCrop):
    """Crop area with mask if mask is non-empty, else make random crop.

    This transform attempts to crop a region containing a mask (non-zero pixels). If the mask is empty or not provided,
    it falls back to a random crop. This is particularly useful for segmentation tasks where you want to focus on
    regions of interest defined by the mask.

    Args:
        height (int): Vertical size of crop in pixels. Must be > 0.
        width (int): Horizontal size of crop in pixels. Must be > 0.
        ignore_values (list of int, optional): Values to ignore in mask, `0` values are always ignored.
            For example, if background value is 5, set `ignore_values=[5]` to ignore it. Default: None.
        ignore_channels (list of int, optional): Channels to ignore in mask.
            For example, if background is the first channel, set `ignore_channels=[0]` to ignore it. Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - If a mask is provided, the transform will try to crop an area containing non-zero (or non-ignored) pixels.
        - If no suitable area is found in the mask or no mask is provided, it will perform a random crop.
        - The crop size (height, width) must not exceed the original image dimensions.
        - Bounding boxes and keypoints are also cropped along with the image and mask.

    Raises:
        ValueError: If the specified crop size is larger than the input image dimensions.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 1  # Create a non-empty region in the mask
        >>> transform = A.Compose([
        ...     A.CropNonEmptyMaskIfExists(height=50, width=50, p=1.0),
        ... ])
        >>> transformed = transform(image=image, mask=mask)
        >>> transformed_image = transformed['image']
        >>> transformed_mask = transformed['mask']
        # The resulting crop will likely include part of the non-zero region in the mask

    """

    class InitSchema(BaseCrop.InitSchema):
        ignore_values: list[int] | None
        ignore_channels: list[int] | None
        height: Annotated[int, Field(ge=1)]
        width: Annotated[int, Field(ge=1)]

    def __init__(
        self,
        height: int,
        width: int,
        ignore_values: list[int] | None = None,
        ignore_channels: list[int] | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.height = height
        self.width = width
        self.ignore_values = ignore_values
        self.ignore_channels = ignore_channels

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_height, mask_width = mask.shape[:2]

        if self.ignore_values is not None:
            ignore_values_np = np.array(self.ignore_values)
            mask = np.where(np.isin(mask, ignore_values_np), 0, mask)

        if mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS and self.ignore_channels is not None:
            target_channels = np.array([ch for ch in range(mask.shape[-1]) if ch not in self.ignore_channels])
            mask = np.take(mask, target_channels, axis=-1)

        if self.height > mask_height or self.width > mask_width:
            raise ValueError(
                f"Crop size ({self.height},{self.width}) is larger than image ({mask_height},{mask_width})",
            )

        return mask

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get crop coordinates based on mask content.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data of the transform.

        """
        if "mask" in data:
            mask = self._preprocess_mask(data["mask"])
        elif "masks" in data and len(data["masks"]):
            masks = data["masks"]
            mask = self._preprocess_mask(np.copy(masks[0]))
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            msg = "Can not find mask for CropNonEmptyMaskIfExists"
            raise RuntimeError(msg)

        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            # Find non-zero regions in mask
            mask_sum = mask.sum(axis=-1) if mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS else mask
            non_zero_yx = np.argwhere(mask_sum)
            y, x = self.py_random.choice(non_zero_yx)

            # Calculate crop coordinates centered around chosen point
            x_min = x - self.py_random.randint(0, self.width - 1)
            y_min = y - self.py_random.randint(0, self.height - 1)
            x_min = np.clip(x_min, 0, mask_width - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
        else:
            # Random crop if no non-zero regions
            x_min = self.py_random.randint(0, mask_width - self.width)
            y_min = self.py_random.randint(0, mask_height - self.height)

        x_max = x_min + self.width
        y_max = y_min + self.height

        return {"crop_coords": (x_min, y_min, x_max, y_max)}


class BaseRandomSizedCropInitSchema(BaseTransformInitSchema):
    size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]


class _BaseRandomSizedCrop(DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    class InitSchema(BaseRandomSizedCropInitSchema):
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
        size: tuple[int, int],
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
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.size = size
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop to the image.

        Args:
            img (np.ndarray): The image to crop.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            **params (Any): Additional parameters.

        """
        crop = fcrops.crop(img, *crop_coords)
        return fgeometric.resize(crop, self.size, self.interpolation)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop to the mask.

        Args:
            mask (np.ndarray): The mask to crop.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            **params (Any): Additional parameters.

        """
        crop = fcrops.crop(mask, *crop_coords)
        return fgeometric.resize(crop, self.size, self.mask_interpolation)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop to the bounding boxes.

        Args:
            bboxes (np.ndarray): The bounding boxes to crop.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            **params (Any): Additional parameters.

        """
        return fcrops.crop_bboxes_by_coords(bboxes, crop_coords, params["shape"])

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop to the keypoints.

        Args:
            keypoints (np.ndarray): The keypoints to crop.
            crop_coords (tuple[int, int, int, int]): The coordinates of the crop.
            **params (Any): Additional parameters.

        """
        # First, crop the keypoints
        cropped_keypoints = fcrops.crop_keypoints_by_coords(keypoints, crop_coords)

        # Calculate the dimensions of the crop
        crop_height = crop_coords[3] - crop_coords[1]
        crop_width = crop_coords[2] - crop_coords[0]

        # Calculate scaling factors
        scale_x = self.size[1] / crop_width
        scale_y = self.size[0] / crop_height

        # Scale the cropped keypoints
        return fgeometric.keypoints_scale(cropped_keypoints, scale_x, scale_y)


class RandomSizedCrop(_BaseRandomSizedCrop):
    """Crop a random part of the input and rescale it to a specific size.

    This transform first crops a random portion of the input and then resizes it to a specified size.
    The size of the random crop is controlled by the 'min_max_height' parameter.

    Args:
        min_max_height (tuple[int, int]): Minimum and maximum height of the crop in pixels.
        size (tuple[int, int]): Target size for the output image, i.e. (height, width) after crop and resize.
        w2h_ratio (float): Aspect ratio (width/height) of crop. Default: 1.0
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - The crop size is randomly selected for each execution within the range specified by 'min_max_height'.
        - The aspect ratio of the crop is determined by the 'w2h_ratio' parameter.
        - After cropping, the result is resized to the specified 'size'.
        - Bounding boxes that end up fully outside the cropped area will be removed.
        - Keypoints that end up outside the cropped area will be removed.
        - This transform differs from RandomResizedCrop in that it allows more control over the crop size
          through the 'min_max_height' parameter, rather than using a scale parameter.

    Mathematical Details:
        1. A random crop height h is sampled from the range [min_max_height[0], min_max_height[1]].
        2. The crop width w is calculated as: w = h * w2h_ratio
        3. A random location for the crop is selected within the input image.
        4. The image is cropped to the size (h, w).
        5. The crop is then resized to the specified 'size'.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.RandomSizedCrop(
        ...     min_max_height=(50, 80),
        ...     size=(64, 64),
        ...     w2h_ratio=1.0,
        ...     interpolation=cv2.INTER_LINEAR,
        ...     p=1.0
        ... )
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        # transformed_image will be a 64x64 image, resulting from a crop with height
        # between 50 and 80 pixels, and the same aspect ratio as specified by w2h_ratio,
        # taken from a random location in the original image and then resized.

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
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
        min_max_height: OnePlusIntRangeType
        w2h_ratio: Annotated[float, Field(gt=0)]
        size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]

    def __init__(
        self,
        min_max_height: tuple[int, int],
        size: tuple[int, int],
        w2h_ratio: float = 1.0,
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
        p: float = 1.0,
    ):
        super().__init__(
            size=size,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            p=p,
        )
        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data of the transform.

        """
        image_shape = params["shape"][:2]

        crop_height = self.py_random.randint(*self.min_max_height)
        crop_width = int(crop_height * self.w2h_ratio)

        crop_shape = (crop_height, crop_width)

        h_start = self.py_random.random()
        w_start = self.py_random.random()

        crop_coords = fcrops.get_crop_coords(image_shape, crop_shape, h_start, w_start)

        return {"crop_coords": crop_coords}


class RandomResizedCrop(_BaseRandomSizedCrop):
    """Crop a random part of the input and rescale it to a specified size.

    This transform first crops a random portion of the input image (or mask, bounding boxes, keypoints)
    and then resizes the crop to a specified size. It's particularly useful for training neural networks
    on images of varying sizes and aspect ratios.

    Args:
        size (tuple[int, int]): Target size for the output image, i.e. (height, width) after crop and resize.
        scale (tuple[float, float]): Range of the random size of the crop relative to the input size.
            For example, (0.08, 1.0) means the crop size will be between 8% and 100% of the input size.
            Default: (0.08, 1.0)
        ratio (tuple[float, float]): Range of aspect ratios of the random crop.
            For example, (0.75, 1.3333) allows crop aspect ratios from 3:4 to 4:3.
            Default: (0.75, 1.3333333333333333)
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - This transform attempts to crop a random area with an aspect ratio and relative size
          specified by 'ratio' and 'scale' parameters. If it fails to find a suitable crop after
          10 attempts, it will return a crop from the center of the image.
        - The crop's aspect ratio is defined as width / height.
        - Bounding boxes that end up fully outside the cropped area will be removed.
        - Keypoints that end up outside the cropped area will be removed.
        - After cropping, the result is resized to the specified size.

    Mathematical Details:
        1. A target area A is sampled from the range [scale[0] * input_area, scale[1] * input_area].
        2. A target aspect ratio r is sampled from the range [ratio[0], ratio[1]].
        3. The crop width and height are computed as:
           w = sqrt(A * r)
           h = sqrt(A / r)
        4. If w and h are within the input image dimensions, the crop is accepted.
           Otherwise, steps 1-3 are repeated (up to 10 times).
        5. If no valid crop is found after 10 attempts, a centered crop is taken.
        6. The crop is then resized to the specified size.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.RandomResizedCrop(size=80, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0)
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        # transformed_image will be a 80x80 crop from a random location in the original image,
        # with the crop's size between 50% and 100% of the original image size,
        # and the crop's aspect ratio between 3:4 and 4:3.

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        scale: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1)), AfterValidator(nondecreasing)]
        ratio: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]
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
        size: tuple[int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
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
        p: float = 1.0,
    ):
        super().__init__(
            size=size,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            p=p,
        )
        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data of the transform.

        """
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        area = image_height * image_width

        for _ in range(10):
            target_area = self.py_random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(self.py_random.uniform(*log_ratio))

            width = round(math.sqrt(target_area * aspect_ratio))
            height = round(math.sqrt(target_area / aspect_ratio))

            if 0 < width <= image_width and 0 < height <= image_height:
                i = self.py_random.randint(0, image_height - height)
                j = self.py_random.randint(0, image_width - width)

                h_start = i * 1.0 / (image_height - height + 1e-10)
                w_start = j * 1.0 / (image_width - width + 1e-10)

                crop_shape = (height, width)

                crop_coords = fcrops.get_crop_coords(image_shape, crop_shape, h_start, w_start)

                return {"crop_coords": crop_coords}

        # Fallback to central crop
        in_ratio = image_width / image_height
        if in_ratio < min(self.ratio):
            width = image_width
            height = round(image_width / min(self.ratio))
        elif in_ratio > max(self.ratio):
            height = image_height
            width = round(height * max(self.ratio))
        else:  # whole image
            width = image_width
            height = image_height

        i = (image_height - height) // 2
        j = (image_width - width) // 2

        h_start = i * 1.0 / (image_height - height + 1e-10)
        w_start = j * 1.0 / (image_width - width + 1e-10)

        crop_shape = (height, width)

        crop_coords = fcrops.get_crop_coords(image_shape, crop_shape, h_start, w_start)

        return {"crop_coords": crop_coords}


class RandomCropNearBBox(BaseCrop):
    """Crop bbox from image with random shift by x,y coordinates

    Args:
        max_part_shift (float, (float, float)): Max shift in `height` and `width` dimensions relative
            to `cropping_bbox` dimension.
            If max_part_shift is a single float, the range will be (0, max_part_shift).
            Default (0, 0.3).
        cropping_bbox_key (str): Additional target key for cropping box. Default `cropping_bbox`.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Examples:
        >>> aug = Compose([RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_bbox_key='test_bbox')],
        >>>              bbox_params=BboxParams("pascal_voc"))
        >>> result = aug(image=image, bboxes=bboxes, test_bbox=[0, 5, 10, 20])

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        max_part_shift: ZeroOneRangeType
        cropping_bbox_key: str

    def __init__(
        self,
        max_part_shift: tuple[float, float] | float = (0, 0.3),
        cropping_bbox_key: str = "cropping_bbox",
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.max_part_shift = cast("tuple[float, float]", max_part_shift)
        self.cropping_bbox_key = cropping_bbox_key

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[float, ...]]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data of the transform.

        """
        bbox = data[self.cropping_bbox_key]

        image_shape = params["shape"][:2]

        bbox = self._clip_bbox(bbox, image_shape)

        h_max_shift = round((bbox[3] - bbox[1]) * self.max_part_shift[0])
        w_max_shift = round((bbox[2] - bbox[0]) * self.max_part_shift[1])

        x_min = bbox[0] - self.py_random.randint(-w_max_shift, w_max_shift)
        x_max = bbox[2] + self.py_random.randint(-w_max_shift, w_max_shift)

        y_min = bbox[1] - self.py_random.randint(-h_max_shift, h_max_shift)
        y_max = bbox[3] + self.py_random.randint(-h_max_shift, h_max_shift)

        crop_coords = self._clip_bbox((x_min, y_min, x_max, y_max), image_shape)

        if crop_coords[0] == crop_coords[2] or crop_coords[1] == crop_coords[3]:
            crop_shape = (bbox[3] - bbox[1], bbox[2] - bbox[0])
            crop_coords = fcrops.get_center_crop_coords(image_shape, crop_shape)

        return {"crop_coords": crop_coords}

    @property
    def targets_as_params(self) -> list[str]:
        """Get the targets as parameters.

        Returns:
            list[str]: The targets as parameters.

        """
        return [self.cropping_bbox_key]


class BBoxSafeRandomCrop(BaseCrop):
    """Crop an area from image while ensuring all bounding boxes are preserved in the crop.

    Similar to AtLeastOneBboxRandomCrop, but with a key difference:
    - BBoxSafeRandomCrop ensures ALL bounding boxes are preserved in the crop
    - AtLeastOneBboxRandomCrop ensures AT LEAST ONE bounding box is present in the crop

    This makes BBoxSafeRandomCrop more suitable for scenarios where:
    - You need to preserve all objects in the scene
    - Losing any bounding box would be problematic (e.g., rare object classes)
    - You're training a model that needs to detect multiple objects simultaneously

    The algorithm:
    1. If bounding boxes exist:
        - Computes the union of all bounding boxes
        - Applies erosion based on erosion_rate to this union
        - Clips the eroded union to valid image coordinates [0,1]
        - Randomly samples crop coordinates within the clipped union area
    2. If no bounding boxes exist:
        - Computes crop height based on erosion_rate
        - Sets crop width to maintain original aspect ratio
        - Randomly places the crop within the image

    Args:
        erosion_rate (float): Controls how much the valid crop region can deviate from the bbox union.
            Must be in range [0.0, 1.0].
            - 0.0: crop must contain the exact bbox union
            - 1.0: crop can deviate maximally from the bbox union while still containing all boxes
            Defaults to 0.0.
        p (float, optional): Probability of applying the transform. Defaults to 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d
    Image types:
        uint8, float32

    Raises:
        CropSizeError: If requested crop size exceeds image dimensions

    Example:
        >>> import albumentations as A
        >>> transform = A.BBoxSafeRandomCrop(erosion_rate=0.2)
        >>> result = transform(
        ...     image=image,
        ...     bboxes=[[0.1, 0.2, 0.5, 0.7, 'cat'], [0.3, 0.4, 0.6, 0.8, 'dog']],
        ...     bbox_format='yolo'  # or 'coco', 'pascal_voc'
        ... )
        >>> transformed_image = result['image']
        >>> transformed_bboxes = result['bboxes']

    Note:
        - All bounding boxes will be preserved in their entirety
        - Aspect ratio is preserved only when no bounding boxes are present
        - May be more restrictive in crop placement compared to AtLeastOneBboxRandomCrop
        - The crop size is determined by the bounding boxes when present

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        erosion_rate: float = Field(
            ge=0.0,
            le=1.0,
        )

    def __init__(self, erosion_rate: float = 0.0, p: float = 1.0):
        super().__init__(p=p)
        self.erosion_rate = erosion_rate

    def _get_coords_no_bbox(self, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        image_height, image_width = image_shape

        erosive_h = int(image_height * (1.0 - self.erosion_rate))
        crop_height = image_height if erosive_h >= image_height else self.py_random.randint(erosive_h, image_height)

        crop_width = int(crop_height * image_width / image_height)

        h_start = self.py_random.random()
        w_start = self.py_random.random()

        crop_shape = (crop_height, crop_width)

        return fcrops.get_crop_coords(image_shape, crop_shape, h_start, w_start)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data of the transform.

        """
        image_shape = params["shape"][:2]

        if len(data["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            crop_coords = self._get_coords_no_bbox(image_shape)
            return {"crop_coords": crop_coords}

        bbox_union = union_of_bboxes(bboxes=data["bboxes"], erosion_rate=self.erosion_rate)

        if bbox_union is None:
            crop_coords = self._get_coords_no_bbox(image_shape)
            return {"crop_coords": crop_coords}

        x_min, y_min, x_max, y_max = bbox_union

        x_min = np.clip(x_min, 0, 1)
        y_min = np.clip(y_min, 0, 1)
        x_max = np.clip(x_max, x_min, 1)
        y_max = np.clip(y_max, y_min, 1)

        image_height, image_width = image_shape

        crop_x_min = int(x_min * self.py_random.random() * image_width)
        crop_y_min = int(y_min * self.py_random.random() * image_height)

        bbox_xmax = x_max + (1 - x_max) * self.py_random.random()
        bbox_ymax = y_max + (1 - y_max) * self.py_random.random()
        crop_x_max = int(bbox_xmax * image_width)
        crop_y_max = int(bbox_ymax * image_height)

        return {"crop_coords": (crop_x_min, crop_y_min, crop_x_max, crop_y_max)}


class RandomSizedBBoxSafeCrop(BBoxSafeRandomCrop):
    """Crop a random part of the input and rescale it to a specific size without loss of bounding boxes.

    This transform first attempts to crop a random portion of the input image while ensuring that all bounding boxes
    remain within the cropped area. It then resizes the crop to the specified size. This is particularly useful for
    object detection tasks where preserving all objects in the image is crucial while also standardizing the image size.

    Args:
        height (int): Height of the output image after resizing.
        width (int): Width of the output image after resizing.
        erosion_rate (float): A value between 0.0 and 1.0 that determines the minimum allowable size of the crop
            as a fraction of the original image size. For example, an erosion_rate of 0.2 means the crop will be
            at least 80% of the original image height and width. Default: 0.0 (no minimum size).
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - This transform ensures that all bounding boxes in the original image are fully contained within the
          cropped area. If it's not possible to find such a crop (e.g., when bounding boxes are too spread out),
          it will default to cropping the entire image.
        - After cropping, the result is resized to the specified (height, width) size.
        - Bounding box coordinates are adjusted to match the new image size.
        - Keypoints are moved along with the crop and scaled to the new image size.
        - If there are no bounding boxes in the image, it will fall back to a random crop.

    Mathematical Details:
        1. A crop region is selected that includes all bounding boxes.
        2. The crop size is determined by the erosion_rate:
           min_crop_size = (1 - erosion_rate) * original_size
        3. If the selected crop is smaller than min_crop_size, it's expanded to meet this requirement.
        4. The crop is then resized to the specified (height, width) size.
        5. Bounding box coordinates are transformed to match the new image size:
           new_coord = (old_coord - crop_start) * (new_size / crop_size)

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        >>> bboxes = [(10, 10, 50, 50), (100, 100, 150, 150)]
        >>> transform = A.Compose([
        ...     A.RandomSizedBBoxSafeCrop(height=224, width=224, erosion_rate=0.2, p=1.0),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        >>> transformed = transform(image=image, bboxes=bboxes, labels=['cat', 'dog'])
        >>> transformed_image = transformed['image']
        >>> transformed_bboxes = transformed['bboxes']
        # transformed_image will be a 224x224 image containing all original bounding boxes,
        # with their coordinates adjusted to the new image size.

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        height: Annotated[int, Field(ge=1)]
        width: Annotated[int, Field(ge=1)]
        erosion_rate: float = Field(
            ge=0.0,
            le=1.0,
        )
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
        erosion_rate: float = 0.0,
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
        p: float = 1.0,
    ):
        super().__init__(erosion_rate=erosion_rate, p=p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to an image.

        Args:
            img (np.ndarray): The image to apply the crop and pad transform to.
            crop_coords (tuple[int, int, int, int]): The parameters for the crop.
            params (dict[str, Any]): Additional parameters for the transform.

        """
        crop = fcrops.crop(img, *crop_coords)
        return fgeometric.resize(crop, (self.height, self.width), self.interpolation)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to a mask.

        Args:
            mask (np.ndarray): The mask to apply the crop and pad transform to.
            crop_coords (tuple[int, int, int, int]): The parameters for the crop.
            params (dict[str, Any]): Additional parameters for the transform.

        """
        crop = fcrops.crop(mask, *crop_coords)
        return fgeometric.resize(crop, (self.height, self.width), self.mask_interpolation)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to keypoints.

        Args:
            keypoints (np.ndarray): The keypoints to apply the crop and pad transform to.
            crop_coords (tuple[int, int, int, int]): The parameters for the crop.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The keypoints after the crop and pad transform.

        """
        keypoints = fcrops.crop_keypoints_by_coords(keypoints, crop_coords)

        crop_height = crop_coords[3] - crop_coords[1]
        crop_width = crop_coords[2] - crop_coords[0]

        scale_y = self.height / crop_height
        scale_x = self.width / crop_width
        return fgeometric.keypoints_scale(keypoints, scale_x=scale_x, scale_y=scale_y)


class CropAndPad(DualTransform):
    """Crop and pad images by pixel amounts or fractions of image sizes.

    This transform allows for simultaneous cropping and padding of images. Cropping removes pixels from the sides
    (i.e., extracts a subimage), while padding adds pixels to the sides (e.g., black pixels). The amount of
    cropping/padding can be specified either in absolute pixels or as a fraction of the image size.

    Args:
        px (int, tuple of int, tuple of tuples of int, or None):
            The number of pixels to crop (negative values) or pad (positive values) on each side of the image.
            Either this or the parameter `percent` may be set, not both at the same time.
            - If int: crop/pad all sides by this value.
            - If tuple of 2 ints: crop/pad by (top/bottom, left/right).
            - If tuple of 4 ints: crop/pad by (top, right, bottom, left).
            - Each int can also be a tuple of 2 ints for a range, or a list of ints for discrete choices.
            Default: None.

        percent (float, tuple of float, tuple of tuples of float, or None):
            The fraction of the image size to crop (negative values) or pad (positive values) on each side.
            Either this or the parameter `px` may be set, not both at the same time.
            - If float: crop/pad all sides by this fraction.
            - If tuple of 2 floats: crop/pad by (top/bottom, left/right) fractions.
            - If tuple of 4 floats: crop/pad by (top, right, bottom, left) fractions.
            - Each float can also be a tuple of 2 floats for a range, or a list of floats for discrete choices.
            Default: None.

        border_mode (int):
            OpenCV border mode used for padding. Default: cv2.BORDER_CONSTANT.

        fill (tuple[float, ...] | float):
            The constant value to use for padding if border_mode is cv2.BORDER_CONSTANT.
            Default: 0.

        fill_mask (tuple[float, ...] | float):
            Same as fill but used for mask padding. Default: 0.

        keep_size (bool):
            If True, the output image will be resized to the input image size after cropping/padding.
            Default: True.

        sample_independently (bool):
            If True and ranges are used for px/percent, sample a value for each side independently.
            If False, sample one value and use it for all sides. Default: True.

        interpolation (int):
            OpenCV interpolation flag used for resizing if keep_size is True.
            Default: cv2.INTER_LINEAR.

        mask_interpolation (int):
            OpenCV interpolation flag used for resizing if keep_size is True.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.

        p (float):
            Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - This transform will never crop images below a height or width of 1.
        - When using pixel values (px), the image will be cropped/padded by exactly that many pixels.
        - When using percentages (percent), the amount of crop/pad will be calculated based on the image size.
        - Bounding boxes that end up fully outside the image after cropping will be removed.
        - Keypoints that end up outside the image after cropping will be removed.

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.CropAndPad(px=(-10, 20, 30, -40), border_mode=cv2.BORDER_REFLECT, fill=128, p=1.0),
        ... ])
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>> transformed_image = transformed['image']
        >>> transformed_mask = transformed['mask']
        >>> transformed_bboxes = transformed['bboxes']
        >>> transformed_keypoints = transformed['keypoints']

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        px: PxType | None
        percent: PercentType | None
        keep_size: bool
        sample_independently: bool
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
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

        @model_validator(mode="after")
        def _check_px_percent(self) -> Self:
            if self.px is None and self.percent is None:
                msg = "Both px and percent parameters cannot be None simultaneously."
                raise ValueError(msg)
            if self.px is not None and self.percent is not None:
                msg = "Only px or percent may be set!"
                raise ValueError(msg)

            return self

    def __init__(
        self,
        px: int | list[int] | None = None,
        percent: float | list[float] | None = None,
        keep_size: bool = True,
        sample_independently: bool = True,
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
        super().__init__(p=p)

        self.px = px
        self.percent = percent

        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask

        self.keep_size = keep_size
        self.sample_independently = sample_independently

        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_params: Sequence[int],
        pad_params: Sequence[int],
        fill: tuple[float, ...] | float,
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to an image.

        Args:
            img (np.ndarray): The image to apply the crop and pad transform to.
            crop_params (Sequence[int]): The parameters for the crop.
            pad_params (Sequence[int]): The parameters for the pad.
            fill (tuple[float, ...] | float): The value to fill the image with.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The image after the crop and pad transform.

        """
        return fcrops.crop_and_pad(
            img,
            crop_params,
            pad_params,
            fill,
            params["shape"][:2],
            self.interpolation,
            self.border_mode,
            self.keep_size,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        crop_params: Sequence[int],
        pad_params: Sequence[int],
        fill_mask: tuple[float, ...] | float,
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to a mask.

        Args:
            mask (np.ndarray): The mask to apply the crop and pad transform to.
            crop_params (Sequence[int]): The parameters for the crop.
            pad_params (Sequence[int]): The parameters for the pad.
            fill_mask (tuple[float, ...] | float): The value to fill the mask with.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The mask after the crop and pad transform.

        """
        return fcrops.crop_and_pad(
            mask,
            crop_params,
            pad_params,
            fill_mask,
            params["shape"][:2],
            self.mask_interpolation,
            self.border_mode,
            self.keep_size,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        crop_params: tuple[int, int, int, int],
        pad_params: tuple[int, int, int, int],
        result_shape: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to bounding boxes.

        Args:
            bboxes (np.ndarray): The bounding boxes to apply the crop and pad transform to.
            crop_params (tuple[int, int, int, int]): The parameters for the crop.
            pad_params (tuple[int, int, int, int]): The parameters for the pad.
            result_shape (tuple[int, int]): The shape of the result.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The bounding boxes after the crop and pad transform.

        """
        return fcrops.crop_and_pad_bboxes(bboxes, crop_params, pad_params, params["shape"][:2], result_shape)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_params: tuple[int, int, int, int],
        pad_params: tuple[int, int, int, int],
        result_shape: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the crop and pad transform to keypoints.

        Args:
            keypoints (np.ndarray): The keypoints to apply the crop and pad transform to.
            crop_params (tuple[int, int, int, int]): The parameters for the crop.
            pad_params (tuple[int, int, int, int]): The parameters for the pad.
            result_shape (tuple[int, int]): The shape of the result.
            params (dict[str, Any]): Additional parameters for the transform.

        Returns:
            np.ndarray: The keypoints after the crop and pad transform.

        """
        return fcrops.crop_and_pad_keypoints(
            keypoints,
            crop_params,
            pad_params,
            params["shape"][:2],
            result_shape,
            self.keep_size,
        )

    @staticmethod
    def __prevent_zero(val1: int, val2: int, max_val: int) -> tuple[int, int]:
        regain = abs(max_val) + 1
        regain1 = regain // 2
        regain2 = regain // 2
        if regain1 + regain2 < regain:
            regain1 += 1

        if regain1 > val1:
            diff = regain1 - val1
            regain1 = val1
            regain2 += diff
        elif regain2 > val2:
            diff = regain2 - val2
            regain2 = val2
            regain1 += diff

        return val1 - regain1, val2 - regain2

    @staticmethod
    def _prevent_zero(crop_params: list[int], height: int, width: int) -> list[int]:
        top, right, bottom, left = crop_params

        remaining_height = height - (top + bottom)
        remaining_width = width - (left + right)

        if remaining_height < 1:
            top, bottom = CropAndPad.__prevent_zero(top, bottom, height)
        if remaining_width < 1:
            left, right = CropAndPad.__prevent_zero(left, right, width)

        return [max(top, 0), max(right, 0), max(bottom, 0), max(left, 0)]

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Get the parameters for the crop.

        Args:
            params (dict[str, Any]): The parameters for the transform.
            data (dict[str, Any]): The data for the transform.

        Returns:
            dict[str, Any]: The parameters for the crop.

        """
        height, width = params["shape"][:2]

        if self.px is not None:
            new_params = self._get_px_params()
        else:
            percent_params = self._get_percent_params()
            new_params = [
                int(percent_params[0] * height),
                int(percent_params[1] * width),
                int(percent_params[2] * height),
                int(percent_params[3] * width),
            ]

        pad_params = [max(i, 0) for i in new_params]

        crop_params = self._prevent_zero([-min(i, 0) for i in new_params], height, width)

        top, right, bottom, left = crop_params
        crop_params = [left, top, width - right, height - bottom]
        result_rows = crop_params[3] - crop_params[1]
        result_cols = crop_params[2] - crop_params[0]
        if result_cols == width and result_rows == height:
            crop_params = []

        top, right, bottom, left = pad_params
        pad_params = [top, bottom, left, right]
        if any(pad_params):
            result_rows += top + bottom
            result_cols += left + right
        else:
            pad_params = []

        return {
            "crop_params": crop_params or None,
            "pad_params": pad_params or None,
            "fill": None if pad_params is None else self._get_pad_value(self.fill),
            "fill_mask": None
            if pad_params is None
            else self._get_pad_value(cast("Union[tuple[float, ...], float]", self.fill_mask)),
            "result_shape": (result_rows, result_cols),
        }

    def _get_px_params(self) -> list[int]:
        if self.px is None:
            msg = "px is not set"
            raise ValueError(msg)

        if isinstance(self.px, int):
            return [self.px] * 4
        if len(self.px) == PAIR:
            if self.sample_independently:
                return [self.py_random.randrange(*self.px) for _ in range(4)]
            px = self.py_random.randrange(*self.px)
            return [px] * 4
        if isinstance(self.px[0], int):
            return self.px
        if len(self.px[0]) == PAIR:
            return [self.py_random.randrange(*i) for i in self.px]

        return [self.py_random.choice(i) for i in self.px]

    def _get_percent_params(self) -> list[float]:
        if self.percent is None:
            msg = "percent is not set"
            raise ValueError(msg)

        if isinstance(self.percent, float):
            params = [self.percent] * 4
        elif len(self.percent) == PAIR:
            if self.sample_independently:
                params = [self.py_random.uniform(*self.percent) for _ in range(4)]
            else:
                px = self.py_random.uniform(*self.percent)
                params = [px] * 4
        elif isinstance(self.percent[0], (int, float)):
            params = self.percent
        elif len(self.percent[0]) == PAIR:
            params = [self.py_random.uniform(*i) for i in self.percent]
        else:
            params = [self.py_random.choice(i) for i in self.percent]

        return params  # params = [top, right, bottom, left]

    def _get_pad_value(
        self,
        fill: Sequence[float] | float,
    ) -> int | float:
        if isinstance(fill, (list, tuple)):
            if len(fill) == PAIR:
                a, b = fill
                if isinstance(a, int) and isinstance(b, int):
                    return self.py_random.randint(a, b)
                return self.py_random.uniform(a, b)
            return self.py_random.choice(fill)

        if isinstance(fill, (int, float)):
            return fill

        msg = "fill should be a number or list, or tuple of two numbers."
        raise ValueError(msg)


class RandomCropFromBorders(BaseCrop):
    """Randomly crops the input from its borders without resizing.

    This transform randomly crops parts of the input (image, mask, bounding boxes, or keypoints)
    from each of its borders. The amount of cropping is specified as a fraction of the input's
    dimensions for each side independently.

    Args:
        crop_left (float): The maximum fraction of width to crop from the left side.
            Must be in the range [0.0, 1.0]. Default: 0.1
        crop_right (float): The maximum fraction of width to crop from the right side.
            Must be in the range [0.0, 1.0]. Default: 0.1
        crop_top (float): The maximum fraction of height to crop from the top.
            Must be in the range [0.0, 1.0]. Default: 0.1
        crop_bottom (float): The maximum fraction of height to crop from the bottom.
            Must be in the range [0.0, 1.0]. Default: 0.1
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - The actual amount of cropping for each side is randomly chosen between 0 and
          the specified maximum for each application of the transform.
        - The sum of crop_left and crop_right must not exceed 1.0, and the sum of
          crop_top and crop_bottom must not exceed 1.0. Otherwise, a ValueError will be raised.
        - This transform does not resize the input after cropping, so the output dimensions
          will be smaller than the input dimensions.
        - Bounding boxes that end up fully outside the cropped area will be removed.
        - Keypoints that end up outside the cropped area will be removed.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.RandomCropFromBorders(
        ...     crop_left=0.1, crop_right=0.2, crop_top=0.2, crop_bottom=0.1, p=1.0
        ... )
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        # The resulting image will have random crops from each border, with the maximum
        # possible crops being 10% from the left, 20% from the right, 20% from the top,
        # and 10% from the bottom. The image size will be reduced accordingly.

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        crop_left: float = Field(
            ge=0.0,
            le=1.0,
        )
        crop_right: float = Field(
            ge=0.0,
            le=1.0,
        )
        crop_top: float = Field(
            ge=0.0,
            le=1.0,
        )
        crop_bottom: float = Field(
            ge=0.0,
            le=1.0,
        )

        @model_validator(mode="after")
        def _validate_crop_values(self) -> Self:
            if self.crop_left + self.crop_right > 1.0:
                msg = "The sum of crop_left and crop_right must be <= 1."
                raise ValueError(msg)
            if self.crop_top + self.crop_bottom > 1.0:
                msg = "The sum of crop_top and crop_bottom must be <= 1."
                raise ValueError(msg)
            return self

    def __init__(
        self,
        crop_left: float = 0.1,
        crop_right: float = 0.1,
        crop_top: float = 0.1,
        crop_bottom: float = 0.1,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Get the parameters for the crop.

        Args:
            params (dict[str, Any]): The parameters for the transform.
            data (dict[str, Any]): The data for the transform.

        Returns:
            dict[str, tuple[int, int, int, int]]: The parameters for the crop.

        """
        height, width = params["shape"][:2]

        x_min = self.py_random.randint(0, int(self.crop_left * width))
        x_max = self.py_random.randint(max(x_min + 1, int((1 - self.crop_right) * width)), width)

        y_min = self.py_random.randint(0, int(self.crop_top * height))
        y_max = self.py_random.randint(max(y_min + 1, int((1 - self.crop_bottom) * height)), height)

        crop_coords = x_min, y_min, x_max, y_max

        return {"crop_coords": crop_coords}


class AtLeastOneBBoxRandomCrop(BaseCrop):
    """Crop an area from image while ensuring at least one bounding box is present in the crop.

    Similar to BBoxSafeRandomCrop, but with a key difference:
    - BBoxSafeRandomCrop ensures ALL bounding boxes are preserved in the crop
    - AtLeastOneBBoxRandomCrop ensures AT LEAST ONE bounding box is present in the crop

    This makes AtLeastOneBBoxRandomCrop more flexible for scenarios where:
    - You want to focus on individual objects rather than all objects
    - You're willing to lose some bounding boxes to get more varied crops
    - The image has many bounding boxes and keeping all of them would be too restrictive

    The algorithm:
    1. If bounding boxes exist:
        - Randomly selects a reference bounding box from available boxes
        - Computes an eroded version of this box (shrunk by erosion_factor)
        - Calculates valid crop bounds that ensure overlap with the eroded box
        - Randomly samples crop coordinates within these bounds
    2. If no bounding boxes exist:
        - Uses full image dimensions as valid bounds
        - Randomly samples crop coordinates within these bounds

    Args:
        height (int): Fixed height of the crop
        width (int): Fixed width of the crop
        erosion_factor (float, optional): Factor by which to erode (shrink) the reference
            bounding box when computing valid crop regions. Must be in range [0.0, 1.0].
            - 0.0 means no erosion (crop must fully contain the reference box)
            - 1.0 means maximum erosion (crop can be anywhere that intersects the reference box)
            Defaults to 0.0.
        p (float, optional): Probability of applying the transform. Defaults to 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Raises:
        CropSizeError: If requested crop size exceeds image dimensions

    Example:
        >>> import albumentations as A
        >>> transform = A.AtLeastOneBBoxRandomCrop(height=100, width=100)
        >>> result = transform(
        ...     image=image,
        ...     bboxes=[[0.1, 0.2, 0.5, 0.7, 'cat']],
        ...     bbox_format='yolo'  # or 'coco', 'pascal_voc'
        ... )
        >>> transformed_image = result['image']
        >>> transformed_bboxes = result['bboxes']

    Note:
        - Uses fixed crop dimensions (height and width)
        - Bounding boxes that end up partially outside the crop will be adjusted
        - Bounding boxes that end up completely outside the crop will be removed
        - If no bounding boxes are provided, acts as a regular random crop

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseCrop.InitSchema):
        height: Annotated[int, Field(ge=1)]
        width: Annotated[int, Field(ge=1)]
        erosion_factor: Annotated[float, Field(ge=0.0, le=1.0)]

    def __init__(
        self,
        height: int,
        width: int,
        erosion_factor: float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.erosion_factor = erosion_factor

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Get the parameters for the crop.

        Args:
            params (dict[str, Any]): The parameters for the transform.
            data (dict[str, Any]): The data for the transform.

        """
        image_height, image_width = params["shape"][:2]
        bboxes = data.get("bboxes", [])

        if self.height > image_height or self.width > image_width:
            raise CropSizeError(
                f"Crop size (height, width) exceeds image dimensions (height, width):"
                f" {(self.height, self.width)} vs {image_height, image_width}",
            )

        if len(bboxes) > 0:
            bboxes = denormalize_bboxes(bboxes, shape=(image_height, image_width))

            # Pick a bbox amongst all possible as our reference bbox.
            reference_bbox = self.py_random.choice(bboxes)

            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = reference_bbox[:4]

            # Compute valid crop bounds:
            # erosion_factor = 0.0: crop must fully contain the bbox
            # erosion_factor = 1.0: crop can be anywhere that intersects the bbox
            if self.erosion_factor < 1.0:
                # Regular case: compute eroded box dimensions
                bbox_width = bbox_x2 - bbox_x1
                bbox_height = bbox_y2 - bbox_y1
                eroded_width = bbox_width * (1.0 - self.erosion_factor)
                eroded_height = bbox_height * (1.0 - self.erosion_factor)

                min_crop_x = np.clip(
                    a=bbox_x1 + eroded_width - self.width,
                    a_min=0.0,
                    a_max=image_width - self.width,
                )
                max_crop_x = np.clip(
                    a=bbox_x2 - eroded_width,
                    a_min=0.0,
                    a_max=image_width - self.width,
                )

                min_crop_y = np.clip(
                    a=bbox_y1 + eroded_height - self.height,
                    a_min=0.0,
                    a_max=image_height - self.height,
                )
                max_crop_y = np.clip(
                    a=bbox_y2 - eroded_height,
                    a_min=0.0,
                    a_max=image_height - self.height,
                )
            else:
                # Maximum erosion case: crop can be anywhere that intersects the bbox
                min_crop_x = np.clip(
                    a=bbox_x1 - self.width,  # leftmost position that still intersects
                    a_min=0.0,
                    a_max=image_width - self.width,
                )
                max_crop_x = np.clip(
                    a=bbox_x2,  # rightmost position that still intersects
                    a_min=0.0,
                    a_max=image_width - self.width,
                )

                min_crop_y = np.clip(
                    a=bbox_y1 - self.height,  # topmost position that still intersects
                    a_min=0.0,
                    a_max=image_height - self.height,
                )
                max_crop_y = np.clip(
                    a=bbox_y2,  # bottommost position that still intersects
                    a_min=0.0,
                    a_max=image_height - self.height,
                )
        else:
            # If there are no bboxes, just crop anywhere in the image.
            min_crop_x = 0.0
            max_crop_x = image_width - self.width

            min_crop_y = 0.0
            max_crop_y = image_height - self.height

        # Randomly draw the upper-left corner of the crop.
        crop_x1 = int(self.py_random.uniform(a=min_crop_x, b=max_crop_x))
        crop_y1 = int(self.py_random.uniform(a=min_crop_y, b=max_crop_y))

        crop_x2 = crop_x1 + self.width
        crop_y2 = crop_y1 + self.height

        return {"crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2)}
