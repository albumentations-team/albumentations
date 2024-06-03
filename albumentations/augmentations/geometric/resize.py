import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import cv2
import numpy as np
from pydantic import Field, ValidationInfo, field_validator

from albumentations.core.pydantic import InterpolationType, ProbabilityType
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import (
    BoxInternalType,
    KeypointInternalType,
    ScaleFloatType,
    Targets,
)
from albumentations.core.utils import to_tuple

from . import functional as fgeometric

__all__ = ["RandomScale", "LongestMaxSize", "SmallestMaxSize", "Resize"]


class RandomScale(DualTransform):
    """Randomly resize the input. Output image size is different from the input image size.

    Args:
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        scale_limit: ScaleFloatType = Field(
            default=0.1,
            description="Scaling factor range. If a single float value => (1-scale_limit, 1 + scale_limit).",
        )
        interpolation: InterpolationType = cv2.INTER_LINEAR

        @field_validator("scale_limit")
        @classmethod
        def check_scale_limit(cls, v: ScaleFloatType) -> Tuple[float, float]:
            return to_tuple(v, bias=1.0)

    def __init__(
        self,
        scale_limit: ScaleFloatType = 0.1,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: Optional[bool] = None,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.scale_limit = cast(Tuple[float, float], scale_limit)
        self.interpolation = interpolation

    def get_params(self) -> Dict[str, float]:
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(
        self,
        img: np.ndarray,
        scale: float,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.scale(img, scale, interpolation)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        scale: float,
        **params: Any,
    ) -> KeypointInternalType:
        return fgeometric.keypoint_scale(keypoint, scale, scale)

    def get_transform_init_args(self) -> Dict[str, Any]:
        return {"interpolation": self.interpolation, "scale_limit": to_tuple(self.scale_limit, bias=-1.0)}


class MaxSizeInitSchema(BaseTransformInitSchema):
    max_size: Union[int, List[int]] = Field(
        default=1024,
        description="Maximum size of the smallest side of the image after the transformation.",
    )
    interpolation: InterpolationType = cv2.INTER_LINEAR
    p: ProbabilityType = 1

    @field_validator("max_size")
    @classmethod
    def check_scale_limit(cls, v: ScaleFloatType, info: ValidationInfo) -> Union[int, List[int]]:
        result = v if isinstance(v, (list, tuple)) else [v]
        for value in result:
            if not value >= 1:
                raise ValueError(f"{info.field_name} must be bigger or equal to 1.")

        return cast(Union[int, List[int]], result)


class LongestMaxSize(DualTransform):
    """Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int, list of int): maximum size of the image after the transformation. When using a list, max size
            will be randomly selected from the values in the list.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(MaxSizeInitSchema):
        pass

    def __init__(
        self,
        max_size: Union[int, Sequence[int]] = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: Optional[bool] = None,
        p: float = 1,
    ):
        super().__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(
        self,
        img: np.ndarray,
        max_size: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.longest_max_size(img, max_size=max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        max_size: int,
        **params: Any,
    ) -> KeypointInternalType:
        height = params["rows"]
        width = params["cols"]

        scale = max_size / max([height, width])
        return fgeometric.keypoint_scale(keypoint, scale, scale)

    def get_params(self) -> Dict[str, int]:
        return {"max_size": self.max_size if isinstance(self.max_size, int) else random.choice(self.max_size)}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("max_size", "interpolation")


class SmallestMaxSize(DualTransform):
    """Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int, list of int): maximum size of smallest side of the image after the transformation. When using a
            list, max size will be randomly selected from the values in the list.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)

    class InitSchema(MaxSizeInitSchema):
        pass

    def __init__(
        self,
        max_size: Union[int, Sequence[int]] = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: Optional[bool] = None,
        p: float = 1,
    ):
        super().__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(
        self,
        img: np.ndarray,
        max_size: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.smallest_max_size(img, max_size=max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return bbox

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        max_size: int,
        **params: Any,
    ) -> KeypointInternalType:
        height = params["rows"]
        width = params["cols"]

        scale = max_size / min([height, width])
        return fgeometric.keypoint_scale(keypoint, scale, scale)

    def get_params(self) -> Dict[str, int]:
        return {"max_size": self.max_size if isinstance(self.max_size, int) else random.choice(self.max_size)}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("max_size", "interpolation")


class Resize(DualTransform):
    """Resize the input to the given height and width.

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        height: int = Field(ge=1, description="Desired height of the output.")
        width: int = Field(ge=1, description="Desired width of the output.")
        interpolation: InterpolationType = cv2.INTER_LINEAR
        p: ProbabilityType = 1

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: Optional[bool] = None,
        p: float = 1,
    ):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img: np.ndarray, interpolation: int, **params: Any) -> np.ndarray:
        return fgeometric.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        return fgeometric.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("height", "width", "interpolation")
