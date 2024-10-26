from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union, cast

import cv2
import numpy as np
from pydantic import Field, ValidationInfo, field_validator

from albumentations.core.pydantic import InterpolationType
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import ScaleFloatType, ScaleIntType, Targets
from albumentations.core.utils import to_tuple

from . import functional as fgeometric

__all__ = ["RandomScale", "LongestMaxSize", "SmallestMaxSize", "Resize"]


class RandomScale(DualTransform):
    """Randomly resize the input. Output image size is different from the input image size.

    Args:
        scale_limit (float or tuple[float, float]): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The output image size is different from the input image size.
        - Scale factor is sampled independently per image side (width and height).
        - Bounding box coordinates are scaled accordingly.
        - Keypoint coordinates are scaled accordingly.

    Mathematical formulation:
        Let (W, H) be the original image dimensions and (W', H') be the output dimensions.
        The scale factor s is sampled from the range [1 + scale_limit[0], 1 + scale_limit[1]].
        Then, W' = W * s and H' = H * s.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.RandomScale(scale_limit=0.1, p=1.0)
        >>> result = transform(image=image)
        >>> scaled_image = result['image']
        # scaled_image will have dimensions in the range [90, 110] x [90, 110]
        # (assuming the scale_limit of 0.1 results in a scaling factor between 0.9 and 1.1)

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        scale_limit: ScaleFloatType
        interpolation: InterpolationType
        mask_interpolation: InterpolationType

        @field_validator("scale_limit")
        @classmethod
        def check_scale_limit(cls, v: ScaleFloatType) -> tuple[float, float]:
            return to_tuple(v, bias=1.0)

    def __init__(
        self,
        scale_limit: ScaleFloatType = (-0.1, 0.1),
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.scale_limit = cast(tuple[float, float], scale_limit)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def get_params(self) -> dict[str, float]:
        return {"scale": self.py_random.uniform(*self.scale_limit)}

    def apply(
        self,
        img: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.scale(img, scale, self.interpolation)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.scale(mask, scale, self.mask_interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.keypoints_scale(keypoints, scale, scale)

    def get_transform_init_args(self) -> dict[str, Any]:
        return {
            "interpolation": self.interpolation,
            "mask_interpolation": self.mask_interpolation,
            "scale_limit": to_tuple(self.scale_limit, bias=-1.0),
        }


class MaxSizeInitSchema(BaseTransformInitSchema):
    max_size: int | list[int]
    interpolation: InterpolationType
    mask_interpolation: InterpolationType | None = None

    @field_validator("max_size")
    @classmethod
    def check_scale_limit(cls, v: ScaleIntType, info: ValidationInfo) -> int | list[int]:
        result = v if isinstance(v, (list, tuple)) else [v]
        for value in result:
            if not value >= 1:
                raise ValueError(f"{info.field_name} must be bigger or equal to 1.")

        return cast(Union[int, list[int]], result)


class LongestMaxSize(DualTransform):
    """Rescale an image so that the longest side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int, Sequence[int]): Maximum size of the image after the transformation. When using a list or tuple,
            the max size will be randomly selected from the values provided.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - If the longest side of the image is already equal to max_size, the image will not be resized.
        - This transform will not crop the image. The resulting image may be smaller than max_size in both dimensions.
        - For non-square images, the shorter side will be scaled proportionally to maintain the aspect ratio.

    Example:
        >>> import albumentations as A
        >>> import cv2
        >>> transform = A.Compose([
        ...     A.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR),
        ...     A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT),
        ... ])
        >>> # Assume we have a 1500x800 image
        >>> transformed = transform(image=image)
        >>> transformed_image = transformed['image']
        >>> transformed_image.shape
        (1024, 546, 3)  # The longest side (1500) is scaled to 1024, and the other side is scaled proportionally

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(MaxSizeInitSchema):
        pass

    def __init__(
        self,
        max_size: int | Sequence[int] = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 1,
    ):
        super().__init__(p, always_apply)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.max_size = max_size

    def apply(
        self,
        img: np.ndarray,
        max_size: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.longest_max_size(img, max_size=max_size, interpolation=self.interpolation)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        max_size: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.longest_max_size(mask, max_size=max_size, interpolation=self.mask_interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        max_size: int,
        **params: Any,
    ) -> np.ndarray:
        image_shape = params["shape"][:2]

        scale = max_size / max(image_shape)
        return fgeometric.keypoints_scale(keypoints, scale, scale)

    def get_params(self) -> dict[str, int]:
        return {"max_size": self.max_size if isinstance(self.max_size, int) else self.py_random.choice(self.max_size)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "max_size", "interpolation", "mask_interpolation"


class SmallestMaxSize(DualTransform):
    """Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int, list of int): Maximum size of smallest side of the image after the transformation. When using a
            list, max size will be randomly selected from the values in the list.
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - If the smallest side of the image is already equal to max_size, the image will not be resized.
        - This transform will not crop the image. The resulting image may be larger than max_size in both dimensions.
        - For non-square images, the larger side will be scaled proportionally to maintain the aspect ratio.
        - Bounding boxes and keypoints are scaled accordingly.

    Mathematical Details:
        1. Let (W, H) be the original width and height of the image.
        2. The scaling factor s is calculated as:
           s = max_size / min(W, H)
        3. The new dimensions (W', H') are:
           W' = W * s
           H' = H * s
        4. The image is resized to (W', H') using the specified interpolation method.
        5. Bounding boxes and keypoints are scaled by the same factor s.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
        >>> transform = A.SmallestMaxSize(max_size=120, p=1.0)
        >>> result = transform(image=image)
        >>> resized_image = result['image']
        # resized_image will have shape (120, 180, 3), as the smallest side (100)
        # is scaled to 120, and the larger side is scaled proportionally
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)

    class InitSchema(MaxSizeInitSchema):
        pass

    def __init__(
        self,
        max_size: int | Sequence[int] = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 1,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.max_size = max_size

    def apply(
        self,
        img: np.ndarray,
        max_size: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.smallest_max_size(img, max_size=max_size, interpolation=self.interpolation)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        max_size: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.smallest_max_size(mask, max_size=max_size, interpolation=self.mask_interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        max_size: int,
        **params: Any,
    ) -> np.ndarray:
        image_shape = params["shape"][:2]

        scale = max_size / min(image_shape)
        return fgeometric.keypoints_scale(keypoints, scale, scale)

    def get_params(self) -> dict[str, int]:
        return {"max_size": self.max_size if isinstance(self.max_size, int) else self.py_random.choice(self.max_size)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "max_size", "interpolation", "mask_interpolation"


class Resize(DualTransform):
    """Resize the input to the given height and width.

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        height: int = Field(ge=1)
        width: int = Field(ge=1)
        interpolation: InterpolationType
        mask_interpolation: InterpolationType

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 1,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.resize(img, (self.height, self.width), interpolation=self.interpolation)

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.resize(mask, (self.height, self.width), interpolation=self.mask_interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        height, width = params["shape"][:2]
        scale_x = self.width / width
        scale_y = self.height / height
        return fgeometric.keypoints_scale(keypoints, scale_x, scale_y)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "height", "width", "interpolation", "mask_interpolation"
