from __future__ import annotations

from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import AfterValidator, model_validator
from typing_extensions import Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.transforms3d import functional as f3d
from albumentations.core.pydantic import check_range_bounds_3d
from albumentations.core.transforms_interface import Transform3D
from albumentations.core.types import ColorType

__all__ = ["PadIfNeeded3D"]


class PadIfNeeded3D(Transform3D):
    """Pads the sides of a 3D volume if its dimensions are less than specified minimum dimensions.
    If the pad_divisor_zyx is specified, the function additionally ensures that the volume
    dimensions are divisible by these values.

    Args:
        min_zyx (tuple[int, int, int] | None): Minimum desired size as (depth, height, width).
            Ensures volume dimensions are at least these values.
            If not specified, pad_divisor_zyx must be provided.
        pad_divisor_zyx (tuple[int, int, int] | None): If set, pads each dimension to make it
            divisible by corresponding value in format (depth_div, height_div, width_div).
            If not specified, min_zyx must be provided.
        position (Literal["center", "random"]): Position where the volume is to be placed after padding.
            Default is 'center'.
        fill (ColorType): Value to fill the border voxels for images. Default: 0
        fill_mask (ColorType): Value to fill the border voxels for masks. Default: 0
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        image, mask

    Image types:
        uint8, float32

    Note:
        - Either min_zyx or pad_divisor_zyx must be set, but not both for each dimension.
        - The transform will maintain consistency across all targets (image and mask).
        - Input volumes can be either 3D arrays (depth, height, width) or
          4D arrays (depth, height, width, channels).
        - Padding is always applied using constant values specified by fill/fill_mask.

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.PadIfNeeded3D(
        ...         min_zyx=(64, 128, 128),  # Minimum size for each dimension
        ...         fill=0,  # Fill value for images
        ...         fill_mask=0,  # Fill value for masks
        ...     ),
        ... ])
        >>> # For divisible dimensions
        >>> transform = A.Compose([
        ...     A.PadIfNeeded3D(
        ...         pad_divisor_zyx=(16, 16, 16),  # Make dimensions divisible by 16
        ...         fill=0,
        ...     ),
        ... ])
        >>> transformed = transform(image=volume, mask=mask)
        >>> padded_volume = transformed['image']
        >>> padded_mask = transformed['mask']
    """

    class InitSchema(Transform3D.InitSchema):
        min_zyx: Annotated[tuple[int, int, int] | None, AfterValidator(check_range_bounds_3d(0, None))]
        pad_divisor_zyx: Annotated[tuple[int, int, int] | None, AfterValidator(check_range_bounds_3d(1, None))]
        position: Literal["center", "random"]
        fill: ColorType
        fill_mask: ColorType

        @model_validator(mode="after")
        def validate_params(self) -> Self:
            if self.min_zyx is None and self.pad_divisor_zyx is None:
                msg = "At least one of min_zyx or pad_divisor_zyx must be set"
                raise ValueError(msg)
            return self

    def __init__(
        self,
        min_zyx: tuple[int, int, int] | None = None,
        pad_divisor_zyx: tuple[int, int, int] | None = None,
        position: Literal["center", "random"] = "center",
        fill: ColorType = 0,
        fill_mask: ColorType = 0,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.min_zyx = min_zyx
        self.pad_divisor_zyx = pad_divisor_zyx
        self.position = position
        self.fill = fill
        self.fill_mask = fill_mask

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        depth, height, width = data["images"].shape[:3]
        sizes = (depth, height, width)

        paddings = [
            fgeometric.get_dimension_padding(
                current_size=size,
                min_size=self.min_zyx[i] if self.min_zyx else None,
                divisor=self.pad_divisor_zyx[i] if self.pad_divisor_zyx else None,
            )
            for i, size in enumerate(sizes)
        ]

        padding = f3d.adjust_padding_by_position3d(
            paddings=paddings,
            position=self.position,
            py_random=self.py_random,
        )

        return {"padding": padding}  # (d_front, d_back, h_top, h_bottom, w_left, w_right)

    def apply_to_images(
        self,
        images: np.ndarray,
        padding: tuple[int, int, int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        if padding == (0, 0, 0, 0, 0, 0):
            return images
        return f3d.pad_3d_with_params(
            img=images,
            padding=padding,  # (d_front, d_back, h_top, h_bottom, w_left, w_right)
            value=cast(ColorType, self.fill),
        )

    def apply_to_masks(
        self,
        masks: np.ndarray,
        padding: tuple[int, int, int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        if padding == (0, 0, 0, 0, 0, 0):
            return masks
        return f3d.pad_3d_with_params(
            img=masks,
            padding=padding,  # (d_front, d_back, h_top, h_bottom, w_left, w_right)
            value=cast(ColorType, self.fill_mask),
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "min_zyx",
            "pad_divisor_zyx",
            "position",
            "fill",
            "fill_mask",
        )
