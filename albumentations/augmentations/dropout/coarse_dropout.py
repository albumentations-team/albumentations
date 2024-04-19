import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import Field, model_validator
from typing_extensions import Self

from albumentations.core.pydantic import OnePlusIntNonDecreasingRangeType
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import ColorType, KeypointType, ScalarType, Targets

from .functional import cutout, keypoint_in_hole

__all__ = ["CoarseDropout"]


class CoarseDropout(DualTransform):
    """CoarseDropout of the rectangular regions in the image.

    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int, float): Maximum height of the hole.
        If float, it is calculated as a fraction of the image height.
        max_width (int, float): Maximum width of the hole.
        If float, it is calculated as a fraction of the image width.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int, float): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
            If float, it is calculated as a fraction of the image height.
        min_width (int, float): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
            If float, it is calculated as a fraction of the image width.

        fill_value (int, float, list of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, list of int, list of float): fill value for dropped pixels
            in mask. If `None` - mask is not affected. Default: `None`.

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

        hole_height_range: Tuple[ScalarType, ScalarType] = (8, 8)
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

        hole_width_range: Tuple[ScalarType, ScalarType] = (8, 8)
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

        fill_value: ColorType = Field(default=0, description="Value for dropped pixels.")
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
        fill_value: ColorType = 0,
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

        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value

    def apply(
        self,
        img: np.ndarray,
        fill_value: ScalarType = 0,
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

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _ in range(random.randint(self.num_holes_range[0], self.num_holes_range[1])):
            if all(isinstance(x, int) for x in self.hole_height_range + self.hole_width_range):
                hole_height = random.randint(int(self.hole_height_range[0]), int(self.hole_height_range[1]))
                hole_width = random.randint(int(self.hole_width_range[0]), int(self.hole_width_range[1]))
            elif all(isinstance(x, float) for x in self.hole_height_range + self.hole_width_range):
                hole_height = int(height * random.uniform(self.hole_height_range[0], self.hole_height_range[1]))
                hole_width = int(width * random.uniform(self.hole_width_range[0], self.hole_width_range[1]))
            else:
                msg = f"Min width, max width, min height and max height should all either be ints or floats. \
                    Got: {[ type(x) for x in self.hole_height_range + self.hole_width_range]} respectively"
                raise ValueError(msg)

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
