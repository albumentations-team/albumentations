from __future__ import annotations

from typing import Any, Callable, Iterable
from warnings import warn

import numpy as np
from pydantic import AfterValidator, Field, model_validator
from typing_extensions import Annotated, Literal, Self

from albumentations.core.pydantic import check_1plus, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import ColorType, NumericType, ScalarType, Targets
from albumentations.random_utils import randint, uniform

from .functional import cutout, filter_keypoints_in_holes

__all__ = ["CoarseDropout"]


class CoarseDropout(DualTransform):
    """CoarseDropout randomly drops out rectangular regions from the image and optionally,
    the corresponding regions in an associated mask, to simulate the occlusion and
    varied object sizes found in real-world settings. This transformation is an
    evolution of CutOut and RandomErasing, offering more flexibility in the size,
    number of dropout regions, and fill values.

    Args:
        num_holes_range (tuple[int, int]): Specifies the range (minimum and maximum)
            of the number of rectangular regions to zero out. This allows for dynamic
            variation in the number of regions removed per transformation instance.
        hole_height_range (tuple[ScalarType, ScalarType]): Defines the minimum and
            maximum heights of the dropout regions, providing variability in their vertical dimensions.
        hole_width_range (tuple[ScalarType, ScalarType]): Defines the minimum and
            maximum widths of the dropout regions, providing variability in their horizontal dimensions.
        fill_value (ColorType, Literal["random"]): Specifies the value used to fill the dropout regions.
            This can be a constant value, a tuple specifying pixel intensity across channels, or 'random'
            which fills the region with random noise.
        mask_fill_value (ColorType | None): Specifies the fill value for dropout regions in the mask.
            If set to `None`, the mask regions corresponding to the image dropout regions are left unchanged.


    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32

    Reference:
        https://arxiv.org/abs/1708.04552
        https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        min_holes: int | None = Field(
            default=None,
            ge=0,
            description="Minimum number of regions to zero out.",
        )
        max_holes: int | None = Field(
            default=8,
            ge=0,
            description="Maximum number of regions to zero out.",
        )
        num_holes_range: Annotated[tuple[int, int], AfterValidator(check_1plus), AfterValidator(nondecreasing)] = (1, 1)

        min_height: ScalarType | None = Field(
            default=None,
            ge=0,
            description="Minimum height of the hole.",
        )
        max_height: ScalarType | None = Field(
            default=8,
            ge=0,
            description="Maximum height of the hole.",
        )
        hole_height_range: tuple[ScalarType, ScalarType] = (8, 8)

        min_width: ScalarType | None = Field(
            default=None,
            ge=0,
            description="Minimum width of the hole.",
        )
        max_width: ScalarType | None = Field(
            default=8,
            ge=0,
            description="Maximum width of the hole.",
        )
        hole_width_range: tuple[ScalarType, ScalarType] = (8, 8)

        fill_value: ColorType | Literal["random"] = Field(default=0, description="Value for dropped pixels.")
        mask_fill_value: ColorType | None = Field(default=None, description="Fill value for dropped pixels in mask.")

        @staticmethod
        def update_range(
            min_value: NumericType | None,
            max_value: NumericType | None,
            default_range: tuple[NumericType, NumericType],
        ) -> tuple[NumericType, NumericType]:
            if max_value is not None:
                return (min_value or max_value, max_value)

            return default_range

        @staticmethod
        # Validation for hole dimensions ranges
        def validate_range(range_value: tuple[ScalarType, ScalarType], range_name: str, minimum: float = 0) -> None:
            if not minimum <= range_value[0] <= range_value[1]:
                raise ValueError(
                    f"First value in {range_name} should be less or equal than the second value "
                    f"and at least {minimum}. Got: {range_value}",
                )
            if isinstance(range_value[0], float) and not all(0 <= x <= 1 for x in range_value):
                raise ValueError(f"All values in {range_name} should be in [0, 1] range. Got: {range_value}")

        @model_validator(mode="after")
        def check_num_holes_and_dimensions(self) -> Self:
            if self.min_holes is not None:
                warn("`min_holes` is deprecated. Use num_holes_range instead.", DeprecationWarning, stacklevel=2)

            if self.max_holes is not None:
                warn("`max_holes` is deprecated. Use num_holes_range instead.", DeprecationWarning, stacklevel=2)

            if self.min_height is not None:
                warn("`min_height` is deprecated. Use hole_height_range instead.", DeprecationWarning, stacklevel=2)

            if self.max_height is not None:
                warn("`max_height` is deprecated. Use hole_height_range instead.", DeprecationWarning, stacklevel=2)

            if self.min_width is not None:
                warn("`min_width` is deprecated. Use hole_width_range instead.", DeprecationWarning, stacklevel=2)

            if self.max_width is not None:
                warn("`max_width` is deprecated. Use hole_width_range instead.", DeprecationWarning, stacklevel=2)

            if self.max_holes is not None:
                # Update ranges for holes, heights, and widths
                self.num_holes_range = self.update_range(self.min_holes, self.max_holes, self.num_holes_range)

            self.validate_range(self.num_holes_range, "num_holes_range", minimum=1)

            if self.max_height is not None:
                self.hole_height_range = self.update_range(self.min_height, self.max_height, self.hole_height_range)
            self.validate_range(self.hole_height_range, "hole_height_range")

            if self.max_width is not None:
                self.hole_width_range = self.update_range(self.min_width, self.max_width, self.hole_width_range)
            self.validate_range(self.hole_width_range, "hole_width_range")

            return self

    def __init__(
        self,
        max_holes: int | None = None,
        max_height: ScalarType | None = None,
        max_width: ScalarType | None = None,
        min_holes: int | None = None,
        min_height: ScalarType | None = None,
        min_width: ScalarType | None = None,
        fill_value: ColorType | Literal["random"] = 0,
        mask_fill_value: ColorType | None = None,
        num_holes_range: tuple[int, int] = (1, 1),
        hole_height_range: tuple[ScalarType, ScalarType] = (8, 8),
        hole_width_range: tuple[ScalarType, ScalarType] = (8, 8),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range

        self.fill_value = fill_value  # type: ignore[assignment]
        self.mask_fill_value = mask_fill_value

    def apply(
        self,
        img: np.ndarray,
        fill_value: ColorType | Literal["random"],
        holes: Iterable[tuple[int, int, int, int]],
        **params: Any,
    ) -> np.ndarray:
        return cutout(img, holes, fill_value)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        mask_fill_value: ScalarType,
        holes: Iterable[tuple[int, int, int, int]],
        **params: Any,
    ) -> np.ndarray:
        if mask_fill_value is None:
            return mask
        return cutout(mask, holes, mask_fill_value)

    @staticmethod
    def calculate_hole_dimensions(
        image_shape: tuple[int, int],
        height_range: tuple[ScalarType, ScalarType],
        width_range: tuple[ScalarType, ScalarType],
        size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate random hole dimensions based on the provided ranges."""
        height, width = image_shape[:2]

        if isinstance(height_range[0], int):
            min_height = height_range[0]
            max_height = min(height_range[1], height)

            min_width = width_range[0]
            max_width = min(width_range[1], width)

            hole_heights = randint(np.int64(min_height), np.int64(max_height + 1), size=size)
            hole_widths = randint(np.int64(min_width), np.int64(max_width + 1), size=size)

        else:  # Assume float
            hole_heights = (height * uniform(height_range[0], height_range[1], size=size)).astype(int)
            hole_widths = (width * uniform(width_range[0], width_range[1], size=size)).astype(int)

        return hole_heights, hole_widths

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"][:2]

        num_holes = randint(self.num_holes_range[0], self.num_holes_range[1] + 1)

        hole_heights, hole_widths = self.calculate_hole_dimensions(
            image_shape,
            self.hole_height_range,
            self.hole_width_range,
            size=num_holes,
        )

        height, width = image_shape[:2]

        y1 = randint(np.int8(0), height - hole_heights + 1, size=num_holes)
        x1 = randint(np.int8(0), width - hole_widths + 1, size=num_holes)
        y2 = y1 + hole_heights
        x2 = x1 + hole_widths

        holes = np.stack([x1, y1, x2, y2], axis=-1)

        return {"holes": holes}

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        holes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        return filter_keypoints_in_holes(keypoints, holes)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "num_holes_range",
            "hole_height_range",
            "hole_width_range",
            "fill_value",
            "mask_fill_value",
        )

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "keypoints": self.apply_to_keypoints,
        }
