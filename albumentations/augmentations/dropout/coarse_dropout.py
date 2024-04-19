from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import Field, model_validator
from typing_extensions import Literal, Self

from albumentations import random_utils
from albumentations.core.pydantic import OnePlusIntNonDecreasingRangeType
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import ColorType, KeypointType, ScalarType, Targets

from .functional import cutout, keypoint_in_hole

__all__ = ["CoarseDropout"]


class CoarseDropout(DualTransform):
    """CoarseDropout of the rectangular regions in the image.

    Args:
        num_holes_range (Tuple[int, int]): A range specifying the minimum and maximum
            number of regions to zero out. The first value is the minimum number of holes,
            and the second is the maximum.
        hole_height_range (Tuple[ScalarType, ScalarType]): A range specifying the minimum
            and maximum height of the holes. The first value is the minimum height, and
            the second is the maximum height.
        hole_width_range (Tuple[ScalarType, ScalarType]): A range specifying the minimum
            and maximum width of the holes. The first value is the minimum width, and
            the second is the maximum width.
        fill_value (ColorType): Value for dropped pixels in the image.
        mask_fill_value (Optional[ColorType]): Fill value for dropped pixels in the mask.
            If `None`, the mask is not affected. Default: `None`.

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
            deprecated="Use num_holes_range instead.",
        )
        max_holes: Optional[int] = Field(
            default=8,
            ge=0,
            description="Maximum number of regions to zero out.",
            deprecated="Use num_holes_range instead.",
        )
        num_holes_range: OnePlusIntNonDecreasingRangeType = (1, 1)

        min_height: Optional[ScalarType] = Field(
            default=None,
            ge=0,
            description="Minimum height of the hole.",
            deprecated="Use hole_height_range instead.",
        )
        max_height: Optional[ScalarType] = Field(
            default=8,
            ge=0,
            description="Maximum height of the hole.",
            deprecated="Use hole_height_range instead.",
        )
        hole_height_range: Tuple[ScalarType, ScalarType] = (8, 8)

        min_width: Optional[ScalarType] = Field(
            default=None,
            ge=0,
            description="Minimum width of the hole.",
            deprecated="Use hole_width_range instead.",
        )
        max_width: Optional[ScalarType] = Field(
            default=8,
            ge=0,
            description="Maximum width of the hole.",
            deprecated="Use hole_width_range instead.",
        )
        hole_width_range: Tuple[ScalarType, ScalarType] = (8, 8)

        fill_value: Union[ColorType, Literal["random"]] = Field(default=0, description="Value for dropped pixels.")
        mask_fill_value: Optional[ColorType] = Field(default=None, description="Fill value for dropped pixels in mask.")

        @model_validator(mode="after")
        def check_holes_and_dimensions(self) -> Self:
            if self.max_holes is not None:
                if self.min_holes is None:
                    self.num_holes_range = (self.max_holes, self.max_holes)
                    self.min_holes = None
                else:
                    self.num_holes_range = (self.min_holes, self.max_holes)

                self.max_holes = None

            if self.max_height is not None:
                if self.min_height is None:
                    self.hole_height_range = (self.max_height, self.max_height)
                    self.min_height = None
                else:
                    self.hole_height_range = (self.min_height, self.max_height)

                self.max_height = None

            if self.max_width is not None:
                if self.min_width is None:
                    self.hole_width_range = (self.max_width, self.max_width)
                else:
                    self.hole_width_range = (self.min_width, self.max_width)

            if not 0 <= self.hole_width_range[0] <= self.hole_width_range[1]:
                raise ValueError(
                    "First value in hole_width_range should be less or equal than the second value. "
                    f"Got: {self.hole_width_range}",
                )

            if not 0 <= self.hole_height_range[0] <= self.hole_height_range[1]:
                raise ValueError(
                    "First value in hole_height_range should be less or equal than the second value. "
                    f"Got: {self.hole_height_range}",
                )

            if not 1 <= self.num_holes_range[0] <= self.num_holes_range[1]:
                raise ValueError(
                    "First value in hole_height_range should be less or equal than second and at least 1. "
                    f"Got: {self.num_holes_range}",
                )
            return self

    def __init__(
        self,
        max_holes: Optional[int] = 8,
        max_height: Optional[ScalarType] = 8,
        max_width: Optional[ScalarType] = 8,
        min_holes: Optional[int] = None,
        min_height: Optional[ScalarType] = None,
        min_width: Optional[ScalarType] = None,
        fill_value: Union[ColorType, Literal["random"]] = 0,
        mask_fill_value: Optional[ColorType] = None,
        num_holes_range: Tuple[int, int] = (1, 1),
        hole_height_range: Tuple[ScalarType, ScalarType] = (8, 8),
        hole_width_range: Tuple[ScalarType, ScalarType] = (8, 8),
        always_apply: bool = False,
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
        fill_value: Union[ColorType, Literal["random"]] = 0,
        holes: Iterable[Tuple[int, int, int, int]] = (),
        **params: Any,
    ) -> np.ndarray:
        return cutout(img, holes, fill_value)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        mask_fill_value: ScalarType = 0,
        holes: Iterable[Tuple[int, int, int, int]] = (),
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
            hole_height = random_utils.randint(int(height_range[0]), int(height_range[1] + 1))
            hole_width = random_utils.randint(int(width_range[0]), int(width_range[1] + 1))
        else:  # Assume float
            hole_height = int(height * random_utils.uniform(height_range[0], height_range[1]))
            hole_width = int(width * random_utils.uniform(width_range[0], width_range[1]))
        return hole_height, hole_width

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        num_holes = random_utils.randint(self.num_holes_range[0], self.num_holes_range[1] + 1)

        for _ in range(num_holes):
            hole_height, hole_width = self.calculate_hole_dimensions(
                height,
                width,
                self.hole_height_range,
                self.hole_width_range,
            )

            y1 = random_utils.randint(0, height - hole_height + 1)
            x1 = random_utils.randint(0, width - hole_width + 1)
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
        holes: Iterable[Tuple[int, int, int, int]] = (),
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
