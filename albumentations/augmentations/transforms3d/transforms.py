from __future__ import annotations

from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import AfterValidator, field_validator, model_validator
from typing_extensions import Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.transforms3d import functional as f3d
from albumentations.core.pydantic import check_range_bounds_3d
from albumentations.core.transforms_interface import BaseTransformInitSchema, Transform3D
from albumentations.core.types import ColorType, Targets

__all__ = ["CenterCrop3D", "Pad3D", "PadIfNeeded3D", "RandomCrop3D"]

NUM_DIMENSIONS = 3


class BasePad3D(Transform3D):
    """Base class for 3D padding transforms."""

    _targets = (Targets.VOLUME, Targets.MASK3D)

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

    def apply_to_volume(
        self,
        volume: np.ndarray,
        padding: tuple[int, int, int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        if padding == (0, 0, 0, 0, 0, 0):
            return volume
        return f3d.pad_3d_with_params(
            volume=volume,
            padding=padding,
            value=cast(ColorType, self.fill),
        )

    def apply_to_mask3d(
        self,
        mask3d: np.ndarray,
        padding: tuple[int, int, int, int, int, int],
        **params: Any,
    ) -> np.ndarray:
        if padding == (0, 0, 0, 0, 0, 0):
            return mask3d
        return f3d.pad_3d_with_params(
            volume=mask3d,
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
        volume, mask3d

    Image types:
        uint8, float32

    Note:
        Input volume should be a numpy array with dimensions ordered as (z, y, x) or (depth, height, width),
        with optional channel dimension as the last axis.
    """

    class InitSchema(BasePad3D.InitSchema):
        padding: int | tuple[int, int, int] | tuple[int, int, int, int, int, int]

        @field_validator("padding")
        @classmethod
        def validate_padding(
            cls,
            v: int | tuple[int, int, int] | tuple[int, int, int, int, int, int],
        ) -> int | tuple[int, int, int] | tuple[int, int, int, int, int, int]:
            if isinstance(v, int) and v < 0:
                raise ValueError("Padding value must be non-negative")
            if isinstance(v, tuple) and not all(isinstance(i, int) and i >= 0 for i in v):
                raise ValueError("Padding tuple must contain non-negative integers")

            return v

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
        return "padding", "fill", "fill_mask"


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
        fill (ColorType): Value to fill the border voxels for volume. Default: 0
        fill_mask (ColorType): Value to fill the border voxels for masks. Default: 0
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        volume, mask3d

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
        depth, height, width = data["volume"].shape[:3]
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


class BaseCropAndPad3D(Transform3D):
    """Base class for 3D transforms that need both cropping and padding."""

    _targets = (Targets.MASK3D, Targets.VOLUME)

    class InitSchema(Transform3D.InitSchema):
        pad_if_needed: bool
        fill: ColorType
        fill_mask: ColorType
        pad_position: Literal["center", "random"]

    def __init__(
        self,
        pad_if_needed: bool,
        fill: ColorType,
        fill_mask: ColorType,
        pad_position: Literal["center", "random"],
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.fill_mask = fill_mask
        self.pad_position = pad_position

    def _random_pad(self, pad: int) -> tuple[int, int]:
        """Helper function to calculate random padding for one dimension."""
        if pad > 0:
            pad_start = self.py_random.randint(0, pad)
            pad_end = pad - pad_start
        else:
            pad_start = pad_end = 0
        return pad_start, pad_end

    def _center_pad(self, pad: int) -> tuple[int, int]:
        """Helper function to calculate center padding for one dimension."""
        pad_start = pad // 2
        pad_end = pad - pad_start
        return pad_start, pad_end

    def _get_pad_params(
        self,
        image_shape: tuple[int, int, int],
        target_shape: tuple[int, int, int],
    ) -> dict[str, Any] | None:
        """Calculate padding parameters if needed for 3D volumes."""
        if not self.pad_if_needed:
            return None

        z, h, w = image_shape
        target_z, target_h, target_w = target_shape

        # Calculate total padding needed for each dimension
        z_pad = max(0, target_z - z)
        h_pad = max(0, target_h - h)
        w_pad = max(0, target_w - w)

        if z_pad == 0 and h_pad == 0 and w_pad == 0:
            return None

        # For center padding, split equally
        if self.pad_position == "center":
            z_front, z_back = self._center_pad(z_pad)
            h_top, h_bottom = self._center_pad(h_pad)
            w_left, w_right = self._center_pad(w_pad)
        # For random padding, randomly distribute the padding
        else:  # random
            z_front, z_back = self._random_pad(z_pad)
            h_top, h_bottom = self._random_pad(h_pad)
            w_left, w_right = self._random_pad(w_pad)

        return {
            "pad_front": z_front,
            "pad_back": z_back,
            "pad_top": h_top,
            "pad_bottom": h_bottom,
            "pad_left": w_left,
            "pad_right": w_right,
        }

    def apply_to_volume(
        self,
        volume: np.ndarray,
        crop_coords: tuple[int, int, int, int, int, int],
        pad_params: dict[str, int] | None,
        **params: Any,
    ) -> np.ndarray:
        # First crop
        cropped = f3d.crop(volume, crop_coords)

        # Then pad if needed
        if pad_params is not None:
            padding = (
                pad_params["pad_front"],
                pad_params["pad_back"],
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
            )
            return f3d.pad_3d_with_params(
                cropped,
                padding=padding,
                value=cast(ColorType, self.fill),
            )

        return cropped

    def apply_to_mask3d(
        self,
        mask3d: np.ndarray,
        crop_coords: tuple[int, int, int, int, int, int],
        pad_params: dict[str, int] | None,
        **params: Any,
    ) -> np.ndarray:
        # First crop
        cropped = f3d.crop(mask3d, crop_coords)

        # Then pad if needed
        if pad_params is not None:
            padding = (
                pad_params["pad_front"],
                pad_params["pad_back"],
                pad_params["pad_top"],
                pad_params["pad_bottom"],
                pad_params["pad_left"],
                pad_params["pad_right"],
            )
            return f3d.pad_3d_with_params(
                cropped,
                padding=padding,
                value=cast(ColorType, self.fill_mask),
            )

        return cropped


class CenterCrop3D(BaseCropAndPad3D):
    """Crop the center of 3D volume.

    Args:
        size (tuple[int, int, int]): Desired output size of the crop in format (depth, height, width)
        pad_if_needed (bool): Whether to pad if the volume is smaller than desired crop size. Default: False
        fill (ColorType): Padding value for image if pad_if_needed is True. Default: 0
        fill_mask (ColorType): Padding value for mask if pad_if_needed is True. Default: 0
        p (float): probability of applying the transform. Default: 1.0

    Targets:
        volume, mask3d

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        size: Annotated[tuple[int, int, int], AfterValidator(check_range_bounds_3d(1, None))]
        pad_if_needed: bool
        fill: ColorType
        fill_mask: ColorType

    def __init__(
        self,
        size: tuple[int, int, int],
        pad_if_needed: bool = False,
        fill: ColorType = 0,
        fill_mask: ColorType = 0,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(
            pad_if_needed=pad_if_needed,
            fill=fill,
            fill_mask=fill_mask,
            pad_position="center",  # Center crop always uses center padding
            p=p,
        )
        self.size = size

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        volume = data["volume"]
        z, h, w = volume.shape[:3]
        target_z, target_h, target_w = self.size

        # Get padding params if needed
        pad_params = self._get_pad_params(
            image_shape=(z, h, w),
            target_shape=self.size,
        )

        # Update dimensions if padding is applied
        if pad_params is not None:
            z = z + pad_params["pad_front"] + pad_params["pad_back"]
            h = h + pad_params["pad_top"] + pad_params["pad_bottom"]
            w = w + pad_params["pad_left"] + pad_params["pad_right"]

        # Validate dimensions after padding
        if z < target_z or h < target_h or w < target_w:
            msg = (
                f"Crop size {self.size} is larger than padded image size ({z}, {h}, {w}). "
                f"This should not happen - please report this as a bug."
            )
            raise ValueError(msg)

        # For CenterCrop3D:
        z_start = (z - target_z) // 2
        h_start = (h - target_h) // 2
        w_start = (w - target_w) // 2

        crop_coords = (
            z_start,
            z_start + target_z,
            h_start,
            h_start + target_h,
            w_start,
            w_start + target_w,
        )

        return {
            "crop_coords": crop_coords,
            "pad_params": pad_params,
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "size", "pad_if_needed", "fill", "fill_mask"


class RandomCrop3D(BaseCropAndPad3D):
    """Crop random part of 3D volume.

    Args:
        size (tuple[int, int, int]): Desired output size of the crop in format (depth, height, width)
        pad_if_needed (bool): Whether to pad if the volume is smaller than desired crop size. Default: False
        fill (ColorType): Padding value for image if pad_if_needed is True. Default: 0
        fill_mask (ColorType): Padding value for mask if pad_if_needed is True. Default: 0
        p (float): probability of applying the transform. Default: 1.0

    Targets:
        volume, mask3d

    Image types:
        uint8, float32
    """

    class InitSchema(BaseTransformInitSchema):
        size: Annotated[tuple[int, int, int], AfterValidator(check_range_bounds_3d(1, None))]
        pad_if_needed: bool
        fill: ColorType
        fill_mask: ColorType

    def __init__(
        self,
        size: tuple[int, int, int],
        pad_if_needed: bool = False,
        fill: ColorType = 0,
        fill_mask: ColorType = 0,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(
            pad_if_needed=pad_if_needed,
            fill=fill,
            fill_mask=fill_mask,
            pad_position="random",  # Random crop uses random padding position
            p=p,
        )
        self.size = size

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        volume = data["volume"]
        z, h, w = volume.shape[:3]
        target_z, target_h, target_w = self.size

        # Get padding params if needed
        pad_params = self._get_pad_params(
            image_shape=(z, h, w),
            target_shape=self.size,
        )

        # Update dimensions if padding is applied
        if pad_params is not None:
            z = z + pad_params["pad_front"] + pad_params["pad_back"]
            h = h + pad_params["pad_top"] + pad_params["pad_bottom"]
            w = w + pad_params["pad_left"] + pad_params["pad_right"]

        # Calculate random crop coordinates
        z_start = self.py_random.randint(0, max(0, z - target_z))
        h_start = self.py_random.randint(0, max(0, h - target_h))
        w_start = self.py_random.randint(0, max(0, w - target_w))

        crop_coords = (
            z_start,
            z_start + target_z,
            h_start,
            h_start + target_h,
            w_start,
            w_start + target_w,
        )

        return {
            "crop_coords": crop_coords,
            "pad_params": pad_params,
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "size", "pad_if_needed", "fill", "fill_mask"
