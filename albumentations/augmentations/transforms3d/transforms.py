from __future__ import annotations

from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import AfterValidator, field_validator, model_validator
from typing_extensions import Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.transforms3d import functional as f3d
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.pydantic import check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, Transform3D
from albumentations.core.types import ColorType, Targets

__all__ = ["CenterCrop3D", "CoarseDropout3D", "CubicSymmetry", "Pad3D", "PadIfNeeded3D", "RandomCrop3D"]

NUM_DIMENSIONS = 3


class BasePad3D(Transform3D):
    """Base class for 3D padding transforms."""

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(Transform3D.InitSchema):
        fill: ColorType
        fill_mask: ColorType

    def __init__(
        self,
        fill: ColorType = 0,
        fill_mask: ColorType = 0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
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

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        padding = params["padding"]
        shift_vector = np.array([padding[4], padding[2], padding[0]])
        return fgeometric.shift_keypoints(keypoints, shift_vector)


class Pad3D(BasePad3D):
    """Pad the sides of a 3D volume by specified number of voxels.

    Args:
        padding (int, tuple[int, int, int] or tuple[int, int, int, int, int, int]): Padding values. Can be:
            * int - pad all sides by this value
            * tuple[int, int, int] - symmetric padding (depth, height, width) where each value
              is applied to both sides of the corresponding dimension
            * tuple[int, int, int, int, int, int] - explicit padding per side in order:
              (depth_front, depth_back, height_top, height_bottom, width_left, width_right)

        fill (ColorType): Padding value for image
        fill_mask (ColorType): Padding value for mask
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        volume, mask3d, keypoints

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
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        Input volume should be a numpy array with dimensions ordered as (z, y, x) or (depth, height, width),
        with optional channel dimension as the last axis.
    """

    class InitSchema(BasePad3D.InitSchema):
        min_zyx: Annotated[tuple[int, int, int] | None, AfterValidator(check_range_bounds(0, None))]
        pad_divisor_zyx: Annotated[tuple[int, int, int] | None, AfterValidator(check_range_bounds(1, None))]
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

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

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
    ):
        super().__init__(p=p)
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
        cropped = f3d.crop3d(volume, crop_coords)

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
        cropped = f3d.crop3d(mask3d, crop_coords)

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

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        crop_coords: tuple[int, int, int, int, int, int],
        pad_params: dict[str, int] | None,
        **params: Any,
    ) -> np.ndarray:
        # Extract crop start coordinates (z1,y1,x1)
        crop_z1, _, crop_y1, _, crop_x1, _ = crop_coords

        # Initialize shift vector with negative crop coordinates
        shift = np.array(
            [
                -crop_x1,  # X shift
                -crop_y1,  # Y shift
                -crop_z1,  # Z shift
            ],
        )

        # Add padding shift if needed
        if pad_params is not None:
            shift += np.array(
                [
                    pad_params["pad_left"],  # X shift
                    pad_params["pad_top"],  # Y shift
                    pad_params["pad_front"],  # Z shift
                ],
            )

        # Apply combined shift
        return fgeometric.shift_keypoints(keypoints, shift)


class CenterCrop3D(BaseCropAndPad3D):
    """Crop the center of 3D volume.

    Args:
        size (tuple[int, int, int]): Desired output size of the crop in format (depth, height, width)
        pad_if_needed (bool): Whether to pad if the volume is smaller than desired crop size. Default: False
        fill (ColorType): Padding value for image if pad_if_needed is True. Default: 0
        fill_mask (ColorType): Padding value for mask if pad_if_needed is True. Default: 0
        p (float): probability of applying the transform. Default: 1.0

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        If you want to perform cropping only in the XY plane while preserving all slices along
        the Z axis, consider using CenterCrop instead. CenterCrop will apply the same XY crop
        to each slice independently, maintaining the full depth of the volume.
    """

    class InitSchema(BaseTransformInitSchema):
        size: Annotated[tuple[int, int, int], AfterValidator(check_range_bounds(1, None))]
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
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        If you want to perform random cropping only in the XY plane while preserving all slices along
        the Z axis, consider using RandomCrop instead. RandomCrop will apply the same XY crop
        to each slice independently, maintaining the full depth of the volume.
    """

    class InitSchema(BaseTransformInitSchema):
        size: Annotated[tuple[int, int, int], AfterValidator(check_range_bounds(1, None))]
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


class CoarseDropout3D(Transform3D):
    """CoarseDropout3D randomly drops out cuboid regions from a 3D volume and optionally,
    the corresponding regions in an associated 3D mask, to simulate occlusion and
    varied object sizes found in real-world volumetric data.

    Args:
        num_holes_range (tuple[int, int]): Range (min, max) for the number of cuboid
            regions to drop out. Default: (1, 1)
        hole_depth_range (tuple[float, float]): Range (min, max) for the depth
            of dropout regions as a fraction of the volume depth (between 0 and 1). Default: (0.1, 0.2)
        hole_height_range (tuple[float, float]): Range (min, max) for the height
            of dropout regions as a fraction of the volume height (between 0 and 1). Default: (0.1, 0.2)
        hole_width_range (tuple[float, float]): Range (min, max) for the width
            of dropout regions as a fraction of the volume width (between 0 and 1). Default: (0.1, 0.2)
        fill (ColorType): Value for the dropped voxels. Can be:
            - int or float: all channels are filled with this value
            - tuple: tuple of values for each channel
            Default: 0
        fill_mask (ColorType | None): Fill value for dropout regions in the 3D mask.
            If None, mask regions corresponding to volume dropouts are unchanged. Default: None
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        - The actual number and size of dropout regions are randomly chosen within the specified ranges.
        - All values in hole_depth_range, hole_height_range and hole_width_range must be between 0 and 1.
        - If you want to apply dropout only in the XY plane while preserving the full depth dimension,
          consider using CoarseDropout instead. CoarseDropout will apply the same rectangular dropout
          to each slice independently, effectively creating cylindrical dropout regions that extend
          through the entire depth of the volume.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> volume = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)    # (D, H, W)
        >>> aug = A.CoarseDropout3D(
        ...     num_holes_range=(3, 6),
        ...     hole_depth_range=(0.1, 0.2),
        ...     hole_height_range=(0.1, 0.2),
        ...     hole_width_range=(0.1, 0.2),
        ...     fill=0,
        ...     p=1.0
        ... )
        >>> transformed = aug(volume=volume, mask3d=mask3d)
        >>> transformed_volume, transformed_mask3d = transformed["volume"], transformed["mask3d"]
    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    class InitSchema(Transform3D.InitSchema):
        num_holes_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        hole_depth_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        hole_height_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        hole_width_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        fill: ColorType
        fill_mask: ColorType | None

        @staticmethod
        def validate_range(range_value: tuple[float, float], range_name: str) -> None:
            if not 0 <= range_value[0] <= range_value[1] <= 1:
                raise ValueError(
                    f"All values in {range_name} should be in [0, 1] range and first value "
                    f"should be less or equal than the second value. Got: {range_value}",
                )

        @model_validator(mode="after")
        def check_ranges(self) -> Self:
            self.validate_range(self.hole_depth_range, "hole_depth_range")
            self.validate_range(self.hole_height_range, "hole_height_range")
            self.validate_range(self.hole_width_range, "hole_width_range")
            return self

    def __init__(
        self,
        num_holes_range: tuple[int, int] = (1, 1),
        hole_depth_range: tuple[float, float] = (0.1, 0.2),
        hole_height_range: tuple[float, float] = (0.1, 0.2),
        hole_width_range: tuple[float, float] = (0.1, 0.2),
        fill: ColorType = 0,
        fill_mask: ColorType | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_holes_range = num_holes_range
        self.hole_depth_range = hole_depth_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range
        self.fill = fill
        self.fill_mask = fill_mask

    def calculate_hole_dimensions(
        self,
        volume_shape: tuple[int, int, int],
        depth_range: tuple[float, float],
        height_range: tuple[float, float],
        width_range: tuple[float, float],
        size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate random hole dimensions based on the provided ranges."""
        depth, height, width = volume_shape[:3]

        hole_depths = np.maximum(1, np.ceil(depth * self.random_generator.uniform(*depth_range, size=size))).astype(int)
        hole_heights = np.maximum(1, np.ceil(height * self.random_generator.uniform(*height_range, size=size))).astype(
            int,
        )
        hole_widths = np.maximum(1, np.ceil(width * self.random_generator.uniform(*width_range, size=size))).astype(int)

        return hole_depths, hole_heights, hole_widths

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        volume_shape = data["volume"].shape[:3]

        num_holes = self.py_random.randint(*self.num_holes_range)

        hole_depths, hole_heights, hole_widths = self.calculate_hole_dimensions(
            volume_shape,
            self.hole_depth_range,
            self.hole_height_range,
            self.hole_width_range,
            size=num_holes,
        )

        depth, height, width = volume_shape[:3]

        z_min = self.random_generator.integers(0, depth - hole_depths + 1, size=num_holes)
        y_min = self.random_generator.integers(0, height - hole_heights + 1, size=num_holes)
        x_min = self.random_generator.integers(0, width - hole_widths + 1, size=num_holes)
        z_max = z_min + hole_depths
        y_max = y_min + hole_heights
        x_max = x_min + hole_widths

        holes = np.stack([z_min, y_min, x_min, z_max, y_max, x_max], axis=-1)

        return {"holes": holes}

    def apply_to_volume(self, volume: np.ndarray, holes: np.ndarray, **params: Any) -> np.ndarray:
        if holes.size == 0:
            return volume

        return f3d.cutout3d(volume, holes, cast(ColorType, self.fill))

    def apply_to_mask(self, mask: np.ndarray, holes: np.ndarray, **params: Any) -> np.ndarray:
        if self.fill_mask is None or holes.size == 0:
            return mask

        return f3d.cutout3d(mask, holes, cast(ColorType, self.fill_mask))

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        holes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Remove keypoints that fall within dropout regions."""
        if holes.size == 0:
            return keypoints
        processor = cast(KeypointsProcessor, self.get_processor("keypoints"))

        if processor is None or not processor.params.remove_invisible:
            return keypoints
        return f3d.filter_keypoints_in_holes3d(keypoints, holes)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "num_holes_range",
            "hole_depth_range",
            "hole_height_range",
            "hole_width_range",
            "fill",
            "fill_mask",
        )


class CubicSymmetry(Transform3D):
    """Applies a random cubic symmetry transformation to a 3D volume.

    This transform is a 3D extension of D4. While D4 handles the 8 symmetries
    of a square (4 rotations x 2 reflections), CubicSymmetry handles all 48 symmetries of a cube.
    Like D4, this transform does not create any interpolation artifacts as it only remaps voxels
    from one position to another without any interpolation.

    The 48 transformations consist of:
    - 24 rotations (orientation-preserving):
        * 4 rotations around each face diagonal (6 face diagonals x 4 rotations = 24)
    - 24 rotoreflections (orientation-reversing):
        * Reflection through a plane followed by any of the 24 rotations

    For a cube, these transformations preserve:
    - All face centers (6)
    - All vertex positions (8)
    - All edge centers (12)

    works with 3D volumes and masks of the shape (D, H, W) or (D, H, W, C)

    Args:
        p (float): Probability of applying the transform. Default: 1.0

    Targets:
        volume, mask3d, keypoints

    Image types:
        uint8, float32

    Note:
        - This transform is particularly useful for data augmentation in 3D medical imaging,
          crystallography, and voxel-based 3D modeling where the object's orientation
          is arbitrary.
        - All transformations preserve the object's chirality (handedness) when using
          pure rotations (indices 0-23) and invert it when using rotoreflections
          (indices 24-47).

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> volume = np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)  # (D, H, W)
        >>> mask3d = np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)    # (D, H, W)
        >>> transform = A.CubicSymmetry(p=1.0)
        >>> transformed = transform(volume=volume, mask3d=mask3d)
        >>> transformed_volume = transformed["volume"]
        >>> transformed_mask3d = transformed["mask3d"]

    See Also:
        - D4: The 2D version that handles the 8 symmetries of a square
    """

    _targets = (Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS)

    def __init__(
        self,
        p: float = 1.0,
    ):
        super().__init__(p=p)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        # Randomly select one of 48 possible transformations

        volume_shape = data["volume"].shape
        return {"index": self.py_random.randint(0, 47), "volume_shape": volume_shape}

    def apply_to_volume(self, volume: np.ndarray, index: int, **params: Any) -> np.ndarray:
        return f3d.transform_cube(volume, index)

    def apply_to_keypoints(self, keypoints: np.ndarray, index: int, **params: Any) -> np.ndarray:
        return f3d.transform_cube_keypoints(keypoints, index, volume_shape=params["volume_shape"])

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()
