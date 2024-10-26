from __future__ import annotations

from typing import Annotated, Any, Literal
from warnings import warn

from pydantic import AfterValidator, Field, model_validator
from typing_extensions import Self

import albumentations.augmentations.dropout.functional as fdropout
from albumentations.augmentations.dropout.transforms import BaseDropout
from albumentations.core.pydantic import check_0plus, check_1plus, nondecreasing
from albumentations.core.types import MIN_UNIT_SIZE, ColorType

__all__ = ["GridDropout"]


class GridDropout(BaseDropout):
    """Apply GridDropout augmentation to images, masks, bounding boxes, and keypoints.

    GridDropout drops out rectangular regions of an image and the corresponding mask in a grid fashion.
    This technique can help improve model robustness by forcing the network to rely on a broader context
    rather than specific local features.

    Args:
        ratio (float): The ratio of the mask holes to the unit size (same for horizontal and vertical directions).
            Must be between 0 and 1. Default: 0.5.
        unit_size_range (tuple[int, int] | None): Range from which to sample grid size. Default: None.
            Must be between 2 and the image's shorter edge. If None, grid size is calculated based on image size.
        holes_number_xy (tuple[int, int] | None): The number of grid units in x and y directions.
            First value should be between 1 and image width//2,
            Second value should be between 1 and image height//2.
            Default: None. If provided, overrides unit_size_range.
        random_offset (bool): Whether to offset the grid randomly between 0 and (grid unit size - hole size).
            If True, entered shift_xy is ignored and set randomly. Default: True.
        fill_value (int | float | Literal["random"] | tuple[int | float,...]): Value for the dropped pixels. Can be:
            - int or float: all channels are filled with this value.
            - tuple: tuple of values for each channel.
            - 'random': filled with random values.
            Default: 0.
        mask_fill_value (int | float | tuple[int | float,...] | None): Value for the dropped pixels in mask.
            If None, the mask is not modified. Default: None.
        shift_xy (tuple[int, int]): Offsets of the grid start in x and y directions from (0,0) coordinate.
            Only used when random_offset is False. Default: (0, 0).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - If both unit_size_range and holes_number_xy are None, the grid size is calculated based on the image size.
        - The actual number of dropped regions may differ slightly from holes_number_xy due to rounding.
        - This implementation includes deprecation warnings for older parameter names.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> augmentation = A.GridDropout(ratio=0.3, unit_size_range=(10, 20), random_offset=True, p=1.0)
        >>> transformed = augmentation(image=image, mask=mask)
        >>> transformed_image, transformed_mask = transformed["image"], transformed["mask"]

    Reference:
        - Paper: https://arxiv.org/abs/2001.04086
    """

    class InitSchema(BaseDropout.InitSchema):
        ratio: float = Field(gt=0, le=1)

        unit_size_min: int | None = Field(ge=2)
        unit_size_max: int | None = Field(ge=2)

        holes_number_x: int | None = Field(ge=1)
        holes_number_y: int | None = Field(ge=1)

        shift_x: int | None = Field(ge=0)
        shift_y: int | None = Field(ge=0)

        random_offset: bool
        fill_value: ColorType | Literal["random"]
        mask_fill_value: ColorType | None
        unit_size_range: Annotated[tuple[int, int], AfterValidator(check_1plus), AfterValidator(nondecreasing)] | None
        shift_xy: Annotated[tuple[int, int], AfterValidator(check_0plus)]

        holes_number_xy: Annotated[tuple[int, int], AfterValidator(check_1plus)] | None

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
        random_offset: bool = True,
        fill_value: ColorType | Literal["random"] = 0,
        mask_fill_value: ColorType | None = None,
        unit_size_range: tuple[int, int] | None = None,
        holes_number_xy: tuple[int, int] | None = None,
        shift_xy: tuple[int, int] = (0, 0),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(fill_value=fill_value, mask_fill_value=mask_fill_value, p=p, always_apply=always_apply)
        self.ratio = ratio
        self.unit_size_range = unit_size_range
        self.holes_number_xy = holes_number_xy
        self.random_offset = random_offset
        self.shift_xy = shift_xy

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"]
        if self.holes_number_xy:
            grid = self.holes_number_xy
        else:
            # Calculate grid based on unit_size_range or default
            unit_height, unit_width = fdropout.calculate_grid_dimensions(
                image_shape,
                self.unit_size_range,
                self.holes_number_xy,
                self.random_generator,
            )
            grid = (image_shape[0] // unit_height, image_shape[1] // unit_width)

        holes = fdropout.generate_grid_holes(
            image_shape,
            grid,
            self.ratio,
            self.random_offset,
            self.shift_xy,
            self.random_generator,
        )
        return {"holes": holes, "seed": self.random_generator.integers(0, 2**32 - 1)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            *super().get_transform_init_args_names(),
            "ratio",
            "unit_size_range",
            "holes_number_xy",
            "shift_xy",
            "random_offset",
        )
