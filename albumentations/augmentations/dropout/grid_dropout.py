from __future__ import annotations

import random
from typing import Any, Iterable, Sequence
from warnings import warn

from pydantic import AfterValidator, Field, model_validator
from typing_extensions import Annotated, Self

from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import MIN_UNIT_SIZE, PAIR, ColorType, Targets

from . import functional as fdropout


import numpy as np

from albumentations.core.pydantic import check_0plus, check_1plus, nondecreasing

__all__ = ["GridDropout"]


class GridDropout(DualTransform):
    """GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.

    Args:
        ratio (float): The ratio of the mask holes to the unit_size (same for horizontal and vertical directions).
            Must be between 0 and 1. Default: 0.5.
        random_offset (bool): Whether to offset the grid randomly between 0 and grid unit size - hole size.
            If True, entered shift_x and shift_y are ignored and set randomly. Default: False.
        fill_value (Optional[ColorType]): Value for the dropped pixels. Default: 0.
        mask_fill_value (Optional[ColorType]): Value for the dropped pixels in mask.
            If None, transformation is not applied to the mask. Default: None.
        unit_size_range (Optional[tuple[int, int]]): Range from which to sample grid size. Default: None.
             Must be between 2 and the image shorter edge.
        holes_number_xy (Optional[tuple[int, int]]): The number of grid units in x and y directions.
            First value should be between 1 and image width//2,
            Second value should be between 1 and image height//2.
            Default: None.
        shift_xy (tuple[int, int]): Offsets of the grid start in x and y directions.
            Offsets of the grid start in x and y directions from (0,0) coordinate.
            Default: (0, 0).

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
        https://arxiv.org/abs/2001.04086

    """

    _targets = (Targets.IMAGE, Targets.MASK)

    class InitSchema(BaseTransformInitSchema):
        ratio: float = Field(description="The ratio of the mask holes to the unit_size.", gt=0, le=1)

        unit_size_min: int | None = Field(None, description="Minimum size of the grid unit.", ge=2)
        unit_size_max: int | None = Field(None, description="Maximum size of the grid unit.", ge=2)

        holes_number_x: int | None = Field(None, description="The number of grid units in x direction.", ge=1)
        holes_number_y: int | None = Field(None, description="The number of grid units in y direction.", ge=1)

        shift_x: int | None = Field(0, description="Offsets of the grid start in x direction.", ge=0)
        shift_y: int | None = Field(0, description="Offsets of the grid start in y direction.", ge=0)

        random_offset: bool = Field(False, description="Whether to offset the grid randomly.")
        fill_value: ColorType | None = Field(0, description="Value for the dropped pixels.")
        mask_fill_value: ColorType | None = Field(None, description="Value for the dropped pixels in mask.")
        unit_size_range: (
            Annotated[tuple[int, int], AfterValidator(check_1plus), AfterValidator(nondecreasing)] | None
        ) = None
        shift_xy: Annotated[tuple[int, int], AfterValidator(check_0plus)] = Field(
            (0, 0),
            description="Offsets of the grid start in x and y directions.",
        )
        holes_number_xy: Annotated[tuple[int, int], AfterValidator(check_1plus)] | None = Field(
            None,
            description="The number of grid units in x and y directions.",
        )

        @model_validator(mode="after")
        def validate_normalization(self) -> Self:
            if self.unit_size_min is not None and self.unit_size_max is not None:
                self.unit_size_range = self.unit_size_min, self.unit_size_max
                warn(
                    "unit_size_min and unit_size_max are deprecated. Use unit_size_range instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if self.shift_x is not None and self.shift_y is not None:
                self.shift_xy = self.shift_x, self.shift_y
                warn("shift_x and shift_y are deprecated. Use shift_xy instead.", DeprecationWarning, stacklevel=2)

            if self.holes_number_x is not None and self.holes_number_y is not None:
                self.holes_number_xy = self.holes_number_x, self.holes_number_y
                warn(
                    "holes_number_x and holes_number_y are deprecated. Use holes_number_xy instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if self.unit_size_range and not MIN_UNIT_SIZE <= self.unit_size_range[0] <= self.unit_size_range[1]:
                raise ValueError("Max unit size should be >= min size, both at least 2 pixels.")

            return self

    def __init__(
        self,
        ratio: float = 0.5,
        unit_size_min: int | None = None,
        unit_size_max: int | None = None,
        holes_number_x: int | None = None,
        holes_number_y: int | None = None,
        shift_x: int | None = None,
        shift_y: int | None = None,
        random_offset: bool = False,
        fill_value: ColorType = 0,
        mask_fill_value: ColorType | None = None,
        unit_size_range: tuple[int, int] | None = None,
        holes_number_xy: tuple[int, int] | None = None,
        shift_xy: tuple[int, int] = (0, 0),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)
        self.ratio = ratio
        self.unit_size_range = unit_size_range
        self.holes_number_xy = holes_number_xy
        self.random_offset = random_offset
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        self.shift_xy = shift_xy

    def apply(self, img: np.ndarray, holes: Iterable[tuple[int, int, int, int]], **params: Any) -> np.ndarray:
        return fdropout.cutout(img, holes, self.fill_value)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        holes: Iterable[tuple[int, int, int, int]],
        **params: Any,
    ) -> np.ndarray:
        if self.mask_fill_value is None:
            return mask

        return fdropout.cutout(mask, holes, self.mask_fill_value)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]
        unit_width, unit_height = self._calculate_unit_dimensions(width, height)
        hole_width, hole_height = self._calculate_hole_dimensions(unit_width, unit_height)
        shift_x, shift_y = self._calculate_shifts(unit_width, unit_height, hole_width, hole_height)
        holes = self._generate_holes(width, height, unit_width, unit_height, hole_width, hole_height, shift_x, shift_y)
        return {"holes": holes}

    def _calculate_unit_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Calculates the dimensions of the grid units."""
        if self.unit_size_range is not None:
            self._validate_unit_sizes(height, width)
            unit_size = random.randint(*self.unit_size_range)
            return unit_size, unit_size

        return self._calculate_dimensions_based_on_holes(width, height)

    def _validate_unit_sizes(self, height: int, width: int) -> None:
        """Validates the minimum and maximum unit sizes."""
        if self.unit_size_range is None:
            raise ValueError("unit_size_range must not be None.")
        if self.unit_size_range[1] > min(height, width):
            msg = "Grid size limits must be within the shortest image edge."
            raise ValueError(msg)

    def _calculate_dimensions_based_on_holes(self, width: int, height: int) -> tuple[int, int]:
        """Calculates dimensions based on the number of holes specified."""
        holes_number_x, holes_number_y = self.holes_number_xy or (None, None)
        unit_width = self._calculate_dimension(width, holes_number_x, 10)
        unit_height = self._calculate_dimension(height, holes_number_y, unit_width)
        return unit_width, unit_height

    @staticmethod
    def _calculate_dimension(dimension: int, holes_number: int | None, fallback: int) -> int:
        """Helper function to calculate unit width or height."""
        if holes_number is None:
            return max(2, dimension // fallback)

        if not 1 <= holes_number <= dimension // 2:
            raise ValueError(f"The number of holes must be between 1 and {dimension // 2}.")
        return dimension // holes_number

    def _calculate_hole_dimensions(self, unit_width: int, unit_height: int) -> tuple[int, int]:
        """Calculates the dimensions of the holes to be dropped out."""
        hole_width = int(unit_width * self.ratio)
        hole_height = int(unit_height * self.ratio)
        hole_width = min(max(hole_width, 1), unit_width - 1)
        hole_height = min(max(hole_height, 1), unit_height - 1)
        return hole_width, hole_height

    def _calculate_shifts(
        self,
        unit_width: int,
        unit_height: int,
        hole_width: int,
        hole_height: int,
    ) -> tuple[int, int]:
        """Calculates the shifts for the grid start."""
        if self.random_offset:
            shift_x = random.randint(0, unit_width - hole_width)
            shift_y = random.randint(0, unit_height - hole_height)
            return shift_x, shift_y

        if isinstance(self.shift_xy, Sequence) and len(self.shift_xy) == PAIR:
            shift_x = min(max(0, self.shift_xy[0]), unit_width - hole_width)
            shift_y = min(max(0, self.shift_xy[1]), unit_height - hole_height)
            return shift_x, shift_y

        return 0, 0

    def _generate_holes(
        self,
        width: int,
        height: int,
        unit_width: int,
        unit_height: int,
        hole_width: int,
        hole_height: int,
        shift_x: int,
        shift_y: int,
    ) -> list[tuple[int, int, int, int]]:
        """Generates the list of holes to be dropped out."""
        holes = []
        for i in range(width // unit_width + 1):
            for j in range(height // unit_height + 1):
                x1 = min(shift_x + unit_width * i, width)
                y1 = min(shift_y + unit_height * j, height)
                x2 = min(x1 + hole_width, width)
                y2 = min(y1 + hole_height, height)
                holes.append((x1, y1, x2, y2))
        return holes

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "ratio",
            "unit_size_range",
            "holes_number_xy",
            "shift_xy",
            "random_offset",
            "fill_value",
            "mask_fill_value",
        )
