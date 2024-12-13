from __future__ import annotations

from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import AfterValidator, model_validator
from typing_extensions import Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.transforms3d import functional as f3d
from albumentations.core.pydantic import check_range_bounds_3d
from albumentations.core.transforms_interface import Transform3D
from albumentations.core.types import ColorType, Targets

__all__ = ["Pad3D", "PadIfNeeded3D"]

NUM_DIMENSIONS = 3


class BasePad3D(Transform3D):
    """Base class for 3D padding transforms."""

    _targets = (Targets.IMAGE, Targets.MASK)

    class InitSchema(Transform3D.InitSchema):
        fill: ColorType
        fill_mask: ColorType

    def __init__(
        self,
        fill: ColorType = 0,
        fill_mask: ColorType = 0,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.fill = fill
        self.fill_mask = fill_mask

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
            padding=padding,
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
            padding=padding,
            value=cast(ColorType, self.fill_mask),
        )


class Pad3D(BasePad3D):
    """Pad the sides of a 3D volume by specified number of voxels.

    Args:
        padding (int, tuple[int, int, int] or tuple[int, int, int, int, int, int]): Padding values. Can be:
            * int - pad all sides by this value
            * tuple[int, int, int] - symmetric padding (pad_z, pad_y, pad_x) where:
                - pad_z: padding for depth/z-axis (front/back)
                - pad_y: padding for height/y-axis (top/bottom)
                - pad_x: padding for width/x-axis (left/right)
            * tuple[int, int, int, int, int, int] - explicit padding per side in order:
                (front, top, left, back, bottom, right) where:
                - front/back: padding along z-axis (depth)
                - top/bottom: padding along y-axis (height)
                - left/right: padding along x-axis (width)
        fill (ColorType): Padding value for image
        fill_mask (ColorType): Padding value for mask
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        images, masks

    Image types:
        uint8, float32

    Note:
        Input volume should be a numpy array with dimensions ordered as (z, y, x) or (depth, height, width),
        with optional channel dimension as the last axis.
    """

    class InitSchema(BasePad3D.InitSchema):
        padding: int | tuple[int, int, int] | tuple[int, int, int, int, int, int]

    def __init__(
        self,
        padding: int | tuple[int, int, int] | tuple[int, int, int, int, int, int],
        fill: ColorType = 0,
        fill_mask: ColorType = 0,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.padding = padding
        self.fill = fill
        self.fill_mask = fill_mask

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self.padding, int):
            pad_d = pad_h = pad_w = self.padding
            padding = (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w)
        elif len(self.padding) == NUM_DIMENSIONS:
            pad_d, pad_h, pad_w = self.padding  # type: ignore[misc]
            padding = (pad_d, pad_d, pad_h, pad_h, pad_w, pad_w)
        else:
            padding = self.padding  # type: ignore[assignment]

        return {"padding": padding}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("padding", "fill", "fill_mask")


class PadIfNeeded3D(BasePad3D):
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
        images, masks

    Image types:
        uint8, float32

    Note:
        Input volume should be a numpy array with dimensions ordered as (z, y, x) or (depth, height, width),
        with optional channel dimension as the last axis.
    """

    class InitSchema(BasePad3D.InitSchema):
        min_zyx: Annotated[tuple[int, int, int] | None, AfterValidator(check_range_bounds_3d(0, None))]
        pad_divisor_zyx: Annotated[tuple[int, int, int] | None, AfterValidator(check_range_bounds_3d(1, None))]
        position: Literal["center", "random"]

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
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.min_zyx = min_zyx
        self.pad_divisor_zyx = pad_divisor_zyx
        self.position = position

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

        return {"padding": padding}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "min_zyx",
            "pad_divisor_zyx",
            "position",
            "fill",
            "fill_mask",
        )
