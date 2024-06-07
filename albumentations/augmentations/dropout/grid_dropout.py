import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from pydantic import Field

from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import TWO, ColorType, ScalarType, Targets

from . import functional as fdropout

__all__ = ["GridDropout"]


class GridDropout(DualTransform):
    """GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.

    Args:
        ratio: the ratio of the mask holes to the unit_size (same for horizontal and vertical directions).
            Must be between 0 and 1. Default: 0.5.
        unit_size_min (int): minimum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        unit_size_max (int): maximum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        holes_number_x (int): the number of grid units in x direction. Must be between 1 and image width//2.
            If 'None', grid unit width is set as image_width//10. Default: `None`.
        holes_number_y (int): the number of grid units in y direction. Must be between 1 and image height//2.
            If `None`, grid unit height is set equal to the grid unit width or image height, whatever is smaller.
        shift_x (int): offsets of the grid start in x direction from (0,0) coordinate.
            Clipped between 0 and grid unit_width - hole_width. Default: 0.
        shift_y (int): offsets of the grid start in y direction from (0,0) coordinate.
            Clipped between 0 and grid unit height - hole_height. Default: 0.
        random_offset (boolean): weather to offset the grid randomly between 0 and grid unit size - hole size
            If 'True', entered shift_x, shift_y are ignored and set randomly. Default: `False`.
        fill_value (int): value for the dropped pixels. Default = 0
        mask_fill_value (int): value for the dropped pixels in mask.
            If `None`, transformation is not applied to the mask. Default: `None`.

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
        https://arxiv.org/abs/2001.04086

    """

    _targets = (Targets.IMAGE, Targets.MASK)

    class InitSchema(BaseTransformInitSchema):
        ratio: float = Field(description="The ratio of the mask holes to the unit_size.", ge=0, le=1)
        unit_size_min: Optional[int] = Field(None, description="Minimum size of the grid unit.", ge=2)
        unit_size_max: Optional[int] = Field(None, description="Maximum size of the grid unit.", ge=2)
        holes_number_x: Optional[int] = Field(None, description="The number of grid units in x direction.", ge=1)
        holes_number_y: Optional[int] = Field(None, description="The number of grid units in y direction.", ge=1)
        shift_x: int = Field(0, description="Offsets of the grid start in x direction.", ge=0)
        shift_y: int = Field(0, description="Offsets of the grid start in y direction.", ge=0)
        random_offset: bool = Field(False, description="Whether to offset the grid randomly.")
        fill_value: Optional[ColorType] = Field(0, description="Value for the dropped pixels.")
        mask_fill_value: Optional[ColorType] = Field(None, description="Value for the dropped pixels in mask.")

    def __init__(
        self,
        ratio: float = 0.5,
        unit_size_min: Optional[int] = None,
        unit_size_max: Optional[int] = None,
        holes_number_x: Optional[int] = None,
        holes_number_y: Optional[int] = None,
        shift_x: int = 0,
        shift_y: int = 0,
        random_offset: bool = False,
        fill_value: float = 0,
        mask_fill_value: Optional[ScalarType] = None,
        always_apply: Optional[bool] = None,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.ratio = ratio
        self.unit_size_min = unit_size_min
        self.unit_size_max = unit_size_max
        self.holes_number_x = holes_number_x
        self.holes_number_y = holes_number_y
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.random_offset = random_offset
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value

    def apply(self, img: np.ndarray, holes: Iterable[Tuple[int, int, int, int]], **params: Any) -> np.ndarray:
        return fdropout.cutout(img, holes, self.fill_value)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        holes: Iterable[Tuple[int, int, int, int]],
        **params: Any,
    ) -> np.ndarray:
        if self.mask_fill_value is None:
            return mask

        return fdropout.cutout(mask, holes, self.mask_fill_value)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        height, width = img.shape[:2]
        unit_width, unit_height = self._calculate_unit_dimensions(width, height)
        hole_width, hole_height = self._calculate_hole_dimensions(unit_width, unit_height)
        shift_x, shift_y = self._calculate_shifts(unit_width, unit_height, hole_width, hole_height)
        holes = self._generate_holes(width, height, unit_width, unit_height, hole_width, hole_height, shift_x, shift_y)
        return {"holes": holes}

    def _calculate_unit_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """Calculates the dimensions of the grid units."""
        if self.unit_size_min is not None and self.unit_size_max is not None:
            self._validate_unit_sizes(height, width)
            unit_size = random.randint(self.unit_size_min, self.unit_size_max)
            return unit_size, unit_size

        return self._calculate_dimensions_based_on_holes(width, height)

    def _validate_unit_sizes(self, height: int, width: int) -> None:
        """Validates the minimum and maximum unit sizes."""
        if self.unit_size_min is not None and self.unit_size_max is not None:
            if not TWO <= self.unit_size_min <= self.unit_size_max:
                msg = "Max unit size should be >= min size, both at least 2 pixels."
                raise ValueError(msg)
            if self.unit_size_max > min(height, width):
                msg = "Grid size limits must be within the shortest image edge."
                raise ValueError(msg)
        else:
            msg = "unit_size_min and unit_size_max must not be None."
            raise ValueError(msg)

    def _calculate_dimensions_based_on_holes(self, width: int, height: int) -> Tuple[int, int]:
        """Calculates dimensions based on the number of holes specified."""
        unit_width = self._calculate_dimension(width, self.holes_number_x, 10)
        unit_height = self._calculate_dimension(height, self.holes_number_y, unit_width)
        return unit_width, unit_height

    def _calculate_dimension(self, dimension: int, holes_number: Optional[int], fallback: int) -> int:
        """Helper function to calculate unit width or height."""
        if holes_number is None:
            return max(2, dimension // fallback)

        if not 1 <= holes_number <= dimension // 2:
            raise ValueError(f"The number of holes must be between 1 and {dimension // 2}.")
        return dimension // holes_number

    def _calculate_hole_dimensions(self, unit_width: int, unit_height: int) -> Tuple[int, int]:
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
    ) -> Tuple[int, int]:
        """Calculates the shifts for the grid start."""
        if self.random_offset:
            shift_x = random.randint(0, unit_width - hole_width)
            shift_y = random.randint(0, unit_height - hole_height)
        else:
            shift_x = 0 if self.shift_x is None else min(max(0, self.shift_x), unit_width - hole_width)
            shift_y = 0 if self.shift_y is None else min(max(0, self.shift_y), unit_height - hole_height)
        return shift_x, shift_y

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
    ) -> List[Tuple[int, int, int, int]]:
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

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return (
            "ratio",
            "unit_size_min",
            "unit_size_max",
            "holes_number_x",
            "holes_number_y",
            "shift_x",
            "shift_y",
            "random_offset",
            "fill_value",
            "mask_fill_value",
        )

    @property
    def targets(self) -> Dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
        }
