from __future__ import annotations

from typing import Annotated, Any
from warnings import warn

import numpy as np
from pydantic import AfterValidator, Field, model_validator
from typing_extensions import Literal, Self

from albumentations.augmentations.dropout.transforms import BaseDropout
from albumentations.core.pydantic import check_1plus, nondecreasing
from albumentations.core.types import ColorType, NumericType, ScalarType

__all__ = ["CoarseDropout"]


class CoarseDropout(BaseDropout):
    """CoarseDropout randomly drops out rectangular regions from the image and optionally,
    the corresponding regions in an associated mask, to simulate occlusion and
    varied object sizes found in real-world settings.

    This transformation is an evolution of CutOut and RandomErasing, offering more
    flexibility in the size, number of dropout regions, and fill values.

    Args:
        num_holes_range (tuple[int, int]): Range (min, max) for the number of rectangular
            regions to drop out. Default: (1, 1)
        hole_height_range (tuple[ScalarType, ScalarType]): Range (min, max) for the height
            of dropout regions. If int, specifies absolute pixel values. If float,
            interpreted as a fraction of the image height. Default: (8, 8)
        hole_width_range (tuple[ScalarType, ScalarType]): Range (min, max) for the width
            of dropout regions. If int, specifies absolute pixel values. If float,
            interpreted as a fraction of the image width. Default: (8, 8)
        fill_value (int | float | Literal["random"] | tuple[int | float,...]): Value for the dropped pixels. Can be:
            - int or float: all channels are filled with this value.
            - tuple: tuple of values for each channel.
            - 'random': filled with random values.
            Default: 0.
        mask_fill_value (ColorType | None): Fill value for dropout regions in the mask.
            If None, mask regions corresponding to image dropouts are unchanged. Default: None
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The actual number and size of dropout regions are randomly chosen within the specified ranges for each
            application.
        - When using float values for hole_height_range and hole_width_range, ensure they are between 0 and 1.
        - This implementation includes deprecation warnings for older parameter names (min_holes, max_holes, etc.).

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> augmentation = A.CoarseDropout(num_holes_range=(3, 6),
        ...                                hole_height_range=(10, 20),
        ...                                hole_width_range=(10, 20),
        ...                                fill_value=0,
        ...                                p=1.0)
        >>> transformed = augmentation(image=image, mask=mask)
        >>> transformed_image, transformed_mask = transformed["image"], transformed["mask"]

    References:
        - CutOut: https://arxiv.org/abs/1708.04552
        - Random Erasing: https://arxiv.org/abs/1708.04896
    """

    class InitSchema(BaseDropout.InitSchema):
        min_holes: int | None = Field(ge=0)
        max_holes: int | None = Field(ge=0)
        num_holes_range: Annotated[tuple[int, int], AfterValidator(check_1plus), AfterValidator(nondecreasing)]

        min_height: ScalarType | None = Field(ge=0)
        max_height: ScalarType | None = Field(ge=0)
        hole_height_range: tuple[ScalarType, ScalarType]

        min_width: ScalarType | None = Field(ge=0)
        max_width: ScalarType | None = Field(ge=0)
        hole_width_range: tuple[ScalarType, ScalarType]

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
        super().__init__(fill_value=fill_value, mask_fill_value=mask_fill_value, p=p, always_apply=always_apply)
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range

    def calculate_hole_dimensions(
        self,
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

            hole_heights = self.random_generator.integers(int(min_height), int(max_height + 1), size=size)
            hole_widths = self.random_generator.integers(int(min_width), int(max_width + 1), size=size)

        else:  # Assume float
            hole_heights = (height * self.random_generator.uniform(*height_range, size=size)).astype(int)
            hole_widths = (width * self.random_generator.uniform(*width_range, size=size)).astype(int)

        return hole_heights, hole_widths

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"][:2]

        num_holes = self.py_random.randint(*self.num_holes_range)

        hole_heights, hole_widths = self.calculate_hole_dimensions(
            image_shape,
            self.hole_height_range,
            self.hole_width_range,
            size=num_holes,
        )

        height, width = image_shape[:2]

        y_min = self.random_generator.integers(0, height - hole_heights + 1, size=num_holes)
        x_min = self.random_generator.integers(0, width - hole_widths + 1, size=num_holes)
        y_max = y_min + hole_heights
        x_max = x_min + hole_widths

        holes = np.stack([x_min, y_min, x_max, y_max], axis=-1)

        return {"holes": holes, "seed": self.random_generator.integers(0, 2**32 - 1)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (*super().get_transform_init_args_names(), "num_holes_range", "hole_height_range", "hole_width_range")
