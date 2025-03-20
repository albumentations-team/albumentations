"""Transform classes for dropout-based augmentations.

This module contains transform classes for various dropout techniques used in image
augmentation. It provides the base dropout class and specialized implementations like
PixelDropout. These transforms randomly remove or modify pixels, channels, or regions
in images, which can help models become more robust to occlusions and missing information.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from albucore import get_num_channels
from pydantic import Field

from albumentations.augmentations import functional as fmain
from albumentations.augmentations.dropout import functional as fdropout
from albumentations.augmentations.dropout.functional import (
    cutout,
    filter_bboxes_by_holes,
    filter_keypoints_in_holes,
)
from albumentations.core.bbox_utils import BboxProcessor, denormalize_bboxes, normalize_bboxes
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.type_definitions import ALL_TARGETS, Targets

__all__ = ["PixelDropout"]


class BaseDropout(DualTransform):
    """Base class for dropout-style transformations.

    This class provides common functionality for various dropout techniques,
    including applying cutouts to images and masks.

    Args:
        fill (tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]):
            Value to fill dropped regions.
        fill_mask (tuple[float, ...] | float | None): Value to fill
            dropped regions in the mask. If None, the mask is not modified.
        p (float): Probability of applying the transform.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32
    """

    _targets: tuple[Targets, ...] | Targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]
        fill_mask: tuple[float, ...] | float | None

    def __init__(
        self,
        fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"],
        fill_mask: tuple[float, ...] | float | None,
        p: float,
    ):
        super().__init__(p=p)
        self.fill = fill  # type: ignore[assignment]
        self.fill_mask = fill_mask

    def apply(self, img: np.ndarray, holes: np.ndarray, seed: int, **params: Any) -> np.ndarray:
        if holes.size == 0:
            return img
        if self.fill in {"inpaint_telea", "inpaint_ns"}:
            num_channels = get_num_channels(img)
            if num_channels not in {1, 3}:
                raise ValueError("Inpainting works only for 1 or 3 channel images")
        return cutout(img, holes, self.fill, np.random.default_rng(seed))

    def apply_to_mask(self, mask: np.ndarray, holes: np.ndarray, seed: int, **params: Any) -> np.ndarray:
        if self.fill_mask is None or holes.size == 0:
            return mask
        return cutout(mask, holes, self.fill_mask, np.random.default_rng(seed))

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        holes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        if holes.size == 0:
            return bboxes
        processor = cast("BboxProcessor", self.get_processor("bboxes"))
        if processor is None:
            return bboxes

        image_shape = params["shape"][:2]
        denormalized_bboxes = denormalize_bboxes(bboxes, image_shape)

        return normalize_bboxes(
            filter_bboxes_by_holes(
                denormalized_bboxes,
                holes,
                image_shape,
                min_area=processor.params.min_area,
                min_visibility=processor.params.min_visibility,
            ),
            image_shape,
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        holes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        if holes.size == 0:
            return keypoints
        processor = cast("KeypointsProcessor", self.get_processor("keypoints"))

        if processor is None or not processor.params.remove_invisible:
            return keypoints

        return filter_keypoints_in_holes(keypoints, holes)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method.")


class PixelDropout(DualTransform):
    """Drops random pixels from the image.

    This transform randomly sets pixels in the image to a specified value, effectively "dropping out" those pixels.
    It can be applied to both the image and its corresponding mask.

    Args:
        dropout_prob (float): Probability of dropping out each pixel. Should be in the range [0, 1].
            Default: 0.01

        per_channel (bool): If True, the dropout mask will be generated independently for each channel.
            If False, the same dropout mask will be applied to all channels.
            Default: False

        drop_value (float | tuple[float, ...] | None): Value to assign to the dropped pixels.
            If None, the value will be randomly sampled for each application:
                - For uint8 images: Random integer in [0, 255]
                - For float32 images: Random float in [0, 1]
            If a single number, that value will be used for all dropped pixels.
            If a sequence, it should contain one value per channel.
            Default: 0

        mask_drop_value (float | tuple[float, ...] | None): Value to assign to dropped pixels in the mask.
            If None, the mask will remain unchanged.
            If a single number, that value will be used for all dropped pixels in the mask.
            If a sequence, it should contain one value per channel.
            Default: None

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - When applied to bounding boxes, this transform may cause some boxes to have zero area
          if all pixels within the box are dropped. Such boxes will be removed.
        - When applied to keypoints, keypoints that fall on dropped pixels will be removed if
          the keypoint processor is configured to remove invisible keypoints.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> transform = A.PixelDropout(dropout_prob=0.1, per_channel=True, p=1.0)
        >>> result = transform(image=image, mask=mask)
        >>> dropped_image, dropped_mask = result['image'], result['mask']
    """

    class InitSchema(BaseTransformInitSchema):
        dropout_prob: float = Field(ge=0, le=1)
        per_channel: bool
        drop_value: tuple[float, ...] | float | None
        mask_drop_value: tuple[float, ...] | float | None

    _targets = ALL_TARGETS

    def __init__(
        self,
        dropout_prob: float = 0.01,
        per_channel: bool = False,
        drop_value: tuple[float, ...] | float | None = 0,
        mask_drop_value: tuple[float, ...] | float | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.drop_value = drop_value
        self.mask_drop_value = mask_drop_value

    def apply(
        self,
        img: np.ndarray,
        drop_mask: np.ndarray,
        drop_values: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        return fmain.pixel_dropout(img, drop_mask, drop_values)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        mask_drop_mask: np.ndarray,
        mask_drop_values: float | np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        if self.mask_drop_value is None:
            return mask

        return fmain.pixel_dropout(mask, mask_drop_mask, mask_drop_values)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        drop_mask: np.ndarray | None,
        **params: Any,
    ) -> np.ndarray:
        if drop_mask is None or self.per_channel:
            return bboxes

        processor = cast("BboxProcessor", self.get_processor("bboxes"))
        if processor is None:
            return bboxes

        image_shape = params["shape"][:2]

        denormalized_bboxes = denormalize_bboxes(bboxes, image_shape)

        # If per_channel is True, we need to create a single channel mask
        # by combining the multi-channel mask (considering a pixel dropped if it's dropped in any channel)
        if self.per_channel and len(drop_mask.shape) > 2:
            # Create a single channel mask where a pixel is considered dropped if it's dropped in any channel
            combined_mask = np.any(drop_mask, axis=-1 if drop_mask.shape[-1] <= 4 else 0)
            # Ensure the mask has the right shape for the bboxes function
            if combined_mask.ndim == 3 and combined_mask.shape[0] == 1:
                combined_mask = combined_mask[0]
        else:
            combined_mask = drop_mask

        result = fdropout.mask_dropout_bboxes(
            denormalized_bboxes,
            combined_mask,
            image_shape,
            processor.params.min_area,
            processor.params.min_visibility,
        )

        return normalize_bboxes(result, image_shape)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        drop_mask: np.ndarray | None,
        **params: Any,
    ) -> np.ndarray:
        return keypoints

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters for pixel dropout based on input data.

        Args:
            params (dict[str, Any]): Transform parameters
            data (dict[str, Any]): Input data dictionary

        Returns:
            dict[str, Any]: Dictionary of parameters for applying the transform
        """
        reference_array = data["image"] if "image" in data else data["images"][0]

        # Generate drop mask and values for all targets
        drop_mask = fmain.get_drop_mask(
            reference_array.shape,
            self.per_channel,
            self.dropout_prob,
            self.random_generator,
        )
        drop_values = fmain.prepare_drop_values(
            reference_array,
            self.drop_value,
            self.random_generator,
        )

        # Handle mask drop values if specified
        mask_drop_mask = None
        mask_drop_values = None
        mask = fmain.get_mask_array(data)
        if self.mask_drop_value is not None and mask is not None:
            mask_drop_mask = fmain.get_drop_mask(
                mask.shape,
                self.per_channel,
                self.dropout_prob,
                self.random_generator,
            )
            mask_drop_values = fmain.prepare_drop_values(
                mask,
                self.mask_drop_value,
                self.random_generator,
            )

        return {
            "drop_mask": drop_mask,
            "drop_values": drop_values,
            "mask_drop_mask": mask_drop_mask if mask_drop_mask is not None else None,
            "mask_drop_values": mask_drop_values if mask_drop_values is not None else None,
        }
