import random
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
from pydantic import Field, model_validator
from typing_extensions import Self

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

        num_holes_range (Tuple[int, int]): Range of regions to zero out.
        Where num_holes_range[0] is minimum number of holes and num_holes_range[1] is minimum number of holes.

        hole_width_range (Tuple[(int, float), (int, float)]): Width range of the hole.
            Hole width range, where `hole_width_range[0]` is the minimum width of the hole and
            `hole_width_range[1]` is the maximum width of the hole.
            If values is float, it is calculated as a fraction of the image width.

        hole_height_range (Tuple[(int, float), (int, float)]): Height range of the hole.
            Hole height range, where `hole_height_range[0]` is the minimum height of the hole and
            `hole_height_range[1]` is the maximum height of the hole.
            If values float, it is calculated as a fraction of the image height.

        fill_value (int, float, list of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, list of int, list of float): fill value for dropped pixels
            in mask. If `None` - mask is not affected. Default: `None`.

    Deprecated:
        min_holes, max_holes, min_height, max_height, min_width, max_width:
        These will be replaced by num_holes_range, hole_height_range, and hole_width_range.

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
        max_holes: int = Field(default=8, ge=0, description="Maximum number of regions to zero out.")
        max_height: ScalarType = Field(default=8, ge=0, description="Maximum height of the hole.")
        max_width: ScalarType = Field(default=8, ge=0, description="Maximum width of the hole.")
        min_holes: Optional[int] = Field(default=None, ge=0, description="Minimum number of regions to zero out.")
        min_height: Optional[ScalarType] = Field(default=None, ge=0, description="Minimum height of the hole.")
        min_width: Optional[ScalarType] = Field(default=None, ge=0, description="Minimum width of the hole.")
        num_holes_range: Optional[Tuple[int, int]] = Field(description="Range of regions to zero out.")
        hole_height_range: Optional[Tuple[ScalarType, ScalarType]] = Field(
            description="Height range of the hole.", validate_default=True
        )
        hole_width_range: Optional[Tuple[ScalarType, ScalarType]] = Field(description="Width range of the hole.")
        fill_value: ColorType = Field(default=0, description="Value for dropped pixels.")
        mask_fill_value: Optional[ColorType] = Field(default=None, description="Fill value for dropped pixels in mask.")

        @model_validator(mode="after")
        def check_holes_and_dimensions(self) -> Self:
            self.min_holes = self.min_holes if self.min_holes is not None else self.max_holes

            self.min_height = self.min_height if self.min_height is not None else self.max_height
            self.min_width = self.min_width if self.min_width is not None else self.max_width

            if not 0 < self.min_height <= self.max_height:
                raise ValueError(
                    f"Invalid combination of min_height and max_height. Got: {[self.min_height, self.max_height]}"
                )
            if not 0 < self.min_width <= self.max_width:
                raise ValueError(
                    f"Invalid combination of min_width and max_width. Got: {[self.min_width, self.max_width]}"
                )
            if not 0 < self.min_holes <= self.max_holes:
                raise ValueError(
                    f"Invalid combination of min_holes and max_holes. Got: {[self.min_holes, self.max_holes]}"
                )

            self.num_holes_range = self.num_holes_range if self.num_holes_range else (self.min_holes, self.max_holes)

            self.hole_width_range = self.hole_width_range if self.hole_width_range else (self.min_width, self.max_width)

            self.hole_height_range = (
                self.hole_height_range if self.hole_height_range else (self.min_height, self.max_height)
            )

            if 0 > self.num_holes_range[0] > self.num_holes_range[1]:
                raise ValueError(
                    "Invalid range for num_holes_range. "
                    "Range must satisfy this condition: num_holes_range[0] <= num_holes_range[1]"
                )
            if 0 > self.hole_width_range[0] > self.hole_width_range[1]:
                raise ValueError(
                    "Invalid range for hole_width_range. "
                    "Range must satisfy this condition: hole_width_range[0] <= hole_width_range[1]"
                )
            if 0 > self.hole_height_range[0] > self.hole_height_range[1]:
                raise ValueError(
                    "Invalid range for hole_height_range. "
                    "Range must satisfy this condition: hole_height_range[0] <= hole_height_range[1]"
                )
            return self

    def __init__(
        self,
        max_holes: int = 8,
        max_height: ScalarType = 8,
        max_width: ScalarType = 8,
        min_holes: Optional[int] = None,
        min_height: Optional[ScalarType] = None,
        min_width: Optional[ScalarType] = None,
        fill_value: ColorType = 0,
        mask_fill_value: Optional[ColorType] = None,
        always_apply: bool = False,
        p: float = 0.5,
        num_holes_range: Optional[Tuple[int, int]] = None,
        hole_width_range: Optional[Tuple[ScalarType, ScalarType]] = None,
        hole_height_range: Optional[Tuple[ScalarType, ScalarType]] = None,
    ):
        super().__init__(always_apply, p)
        self.min_holes = cast(int, min_holes)
        self.max_holes = max_holes
        self.min_height = cast(ScalarType, min_height)
        self.max_height = max_height
        self.min_width = cast(ScalarType, min_width)
        self.max_width = max_width
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        self.num_holes_range = num_holes_range
        self.hole_width_range = hole_width_range
        self.hole_height_range = hole_height_range

        warnings.warn("min_holes, max_holes will be deprecated; use num_holes_range instead.", DeprecationWarning)
        warnings.warn("min_height, max_height will be deprecated; use hole_height_range instead.", DeprecationWarning)
        warnings.warn("min_width, max_width will be deprecated; use hole_width_range instead.", DeprecationWarning)

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
        num_holes_range: Tuple[int, int] = (
            self.num_holes_range if self.num_holes_range else (self.min_holes, self.max_holes)
        )
        hole_width_range: Tuple[ScalarType, ScalarType] = (
            self.hole_width_range if self.hole_width_range else (self.min_width, self.max_width)
        )
        hole_height_range: Tuple[ScalarType, ScalarType] = (
            self.hole_height_range if self.hole_height_range else (self.min_height, self.max_height)
        )
        for _ in range(random.randint(*num_holes_range)):
            if all(isinstance(x, int) for x in hole_height_range + hole_width_range):
                hole_height = random.randint(cast(int, hole_height_range[0]), cast(int, hole_height_range[1]))
                hole_width = random.randint(cast(int, hole_width_range[0]), cast(int, hole_width_range[1]))
            elif all(isinstance(x, float) for x in hole_height_range + hole_width_range):
                hole_height = int(height * random.uniform(*hole_height_range))
                hole_width = int(width * random.uniform(*hole_width_range))
            else:
                msg = f"Hole width range and hole height range \
                    values should all either be ints or floats. \
                    Got: {[type(x) for x in hole_height_range + hole_width_range]} respectively"
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
        self, keypoints: Sequence[KeypointType], holes: Iterable[Tuple[int, int, int, int]] = (), **params: Any
    ) -> List[KeypointType]:
        return [keypoint for keypoint in keypoints if not any(keypoint_in_hole(keypoint, hole) for hole in holes)]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return (
            "max_holes",
            "max_height",
            "max_width",
            "min_holes",
            "min_height",
            "min_width",
            "fill_value",
            "mask_fill_value",
            "hole_width_range",
            "num_holes_range",
            "hole_height_range",
        )
