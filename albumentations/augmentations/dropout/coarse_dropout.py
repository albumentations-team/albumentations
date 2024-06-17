import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
from pydantic import AfterValidator, Field, model_validator
from typing_extensions import Annotated, Literal, Self

from albumentations.core.pydantic import check_1plus, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import ColorType, KeypointType, NumericType, ScalarType, Targets

from .functional import cutout, keypoint_in_hole

__all__ = ["CoarseDropout"]


class CoarseDropout(DualTransform):
    """CoarseDropout randomly drops out rectangular regions from the image and optionally,
    the corresponding regions in an associated mask, to simulate the occlusion and
    varied object sizes found in real-world settings. This transformation is an
    evolution of CutOut and RandomErasing, offering more flexibility in the size,
    number of dropout regions, and fill values.

    Args:
        num_holes_range (Tuple[int, int]): Specifies the range (minimum and maximum)
            of the number of rectangular regions to zero out. This allows for dynamic
            variation in the number of regions removed per transformation instance.
        hole_height_range (Tuple[ScalarType, ScalarType]): Defines the minimum and
            maximum heights of the dropout regions, providing variability in their vertical dimensions.
        hole_width_range (Tuple[ScalarType, ScalarType]): Defines the minimum and
            maximum widths of the dropout regions, providing variability in their horizontal dimensions.
        fill_value (ColorType, Literal["random"]): Specifies the value used to fill the dropout regions.
            This can be a constant value, a tuple specifying pixel intensity across channels, or 'random'
            which fills the region with random noise.
        mask_fill_value (Optional[ColorType]): Specifies the fill value for dropout regions in the mask.
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
        min_holes: Optional[int] = Field(
            default=None,
            ge=0,
            description="Minimum number of regions to zero out.",
        )
        max_holes: Optional[int] = Field(
            default=8,
            ge=0,
            description="Maximum number of regions to zero out.",
        )
        num_holes_range: Annotated[Tuple[int, int], AfterValidator(check_1plus), AfterValidator(nondecreasing)] = (1, 1)

        min_height: Optional[ScalarType] = Field(
            default=None,
            ge=0,
            description="Minimum height of the hole.",
        )
        max_height: Optional[ScalarType] = Field(
            default=8,
            ge=0,
            description="Maximum height of the hole.",
        )
        hole_height_range: Tuple[ScalarType, ScalarType] = (8, 8)

        min_width: Optional[ScalarType] = Field(
            default=None,
            ge=0,
            description="Minimum width of the hole.",
        )
        max_width: Optional[ScalarType] = Field(
            default=8,
            ge=0,
            description="Maximum width of the hole.",
        )
        hole_width_range: Tuple[ScalarType, ScalarType] = (8, 8)

        fill_value: Union[ColorType, Literal["random"]] = Field(default=0, description="Value for dropped pixels.")
        mask_fill_value: Optional[ColorType] = Field(default=None, description="Fill value for dropped pixels in mask.")

        @staticmethod
        def update_range(
            min_value: Optional[NumericType],
            max_value: Optional[NumericType],
            default_range: Tuple[NumericType, NumericType],
        ) -> Tuple[NumericType, NumericType]:
            if max_value is not None:
                return (min_value or max_value, max_value)

            return default_range

        @staticmethod
        # Validation for hole dimensions ranges
        def validate_range(range_value: Tuple[ScalarType, ScalarType], range_name: str, minimum: float = 0) -> None:
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
        max_holes: Optional[int] = None,
        max_height: Optional[ScalarType] = None,
        max_width: Optional[ScalarType] = None,
        min_holes: Optional[int] = None,
        min_height: Optional[ScalarType] = None,
        min_width: Optional[ScalarType] = None,
        fill_value: Union[ColorType, Literal["random"]] = 0,
        mask_fill_value: Optional[ColorType] = None,
        num_holes_range: Tuple[int, int] = (1, 1),
        hole_height_range: Tuple[ScalarType, ScalarType] = (8, 8),
        hole_width_range: Tuple[ScalarType, ScalarType] = (8, 8),
        always_apply: Optional[bool] = None,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range

        self.fill_value = fill_value  # type: ignore[assignment]
        self.mask_fill_value = mask_fill_value

    def apply(
        self,
        img: np.ndarray,
        fill_value: Union[ColorType, Literal["random"]],
        holes: Iterable[Tuple[int, int, int, int]],
        **params: Any,
    ) -> np.ndarray:
        return cutout(img, holes, fill_value)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        mask_fill_value: ScalarType,
        holes: Iterable[Tuple[int, int, int, int]],
        **params: Any,
    ) -> np.ndarray:
        if mask_fill_value is None:
            return mask
        return cutout(mask, holes, mask_fill_value)

    @staticmethod
    def calculate_hole_dimensions(
        height: int,
        width: int,
        height_range: Tuple[ScalarType, ScalarType],
        width_range: Tuple[ScalarType, ScalarType],
    ) -> Tuple[int, int]:
        """Calculate random hole dimensions based on the provided ranges."""
        if isinstance(height_range[0], int):
            min_height = height_range[0]
            max_height = height_range[1]

            min_width = width_range[0]
            max_width = width_range[1]
            max_height = min(max_height, height)
            max_width = min(max_width, width)
            hole_height = random.randint(int(min_height), int(max_height))
            hole_width = random.randint(int(min_width), int(max_width))

        else:  # Assume float
            hole_height = int(height * random.uniform(height_range[0], height_range[1]))
            hole_width = int(width * random.uniform(width_range[0], width_range[1]))
        return hole_height, hole_width

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        num_holes = random.randint(self.num_holes_range[0], self.num_holes_range[1])

        for _ in range(num_holes):
            hole_height, hole_width = self.calculate_hole_dimensions(
                height,
                width,
                self.hole_height_range,
                self.hole_width_range,
            )

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def apply_to_keypoints(
        self,
        keypoints: Sequence[KeypointType],
        holes: Iterable[Tuple[int, int, int, int]],
        **params: Any,
    ) -> List[KeypointType]:
        return [keypoint for keypoint in keypoints if not any(keypoint_in_hole(keypoint, hole) for hole in holes)]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return (
            "num_holes_range",
            "hole_height_range",
            "hole_width_range",
            "fill_value",
            "mask_fill_value",
        )

    @property
    def targets(self) -> Dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "keypoints": self.apply_to_keypoints,
        }
