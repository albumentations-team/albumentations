from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import cv2
import numpy as np
from albucore import batch_transform
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from albumentations.core.pydantic import InterpolationType
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import ALL_TARGETS, ScaleFloatType
from albumentations.core.utils import to_tuple

from . import functional as fgeometric

__all__ = ["LongestMaxSize", "RandomScale", "Resize", "SmallestMaxSize"]


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
        image, mask, bboxes, keypoints, volume, mask3d

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

    _targets = ALL_TARGETS

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
        p: float = 0.5,
    ):
        super().__init__(p=p)
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


class MaxSizeTransform(DualTransform):
    """Base class for transforms that resize based on maximum size constraints."""

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        max_size: int | list[int] | None
        max_size_hw: tuple[int | None, int | None] | None
        interpolation: InterpolationType
        mask_interpolation: InterpolationType

        @model_validator(mode="after")
        def validate_size_parameters(self) -> Self:
            if self.max_size is None and self.max_size_hw is None:
                raise ValueError("Either max_size or max_size_hw must be specified")
            if self.max_size is not None and self.max_size_hw is not None:
                raise ValueError("Only one of max_size or max_size_hw should be specified")
            return self

    def __init__(
        self,
        max_size: int | Sequence[int] | None = 1024,
        max_size_hw: tuple[int | None, int | None] | None = None,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        p: float = 1,
    ):
        super().__init__(p=p)
        self.max_size = max_size
        self.max_size_hw = max_size_hw
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    def apply(
        self,
        img: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        height, width = img.shape[:2]
        new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
        return fgeometric.resize(img, (new_height, new_width), interpolation=self.interpolation)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        height, width = mask.shape[:2]
        new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
        return fgeometric.resize(mask, (new_height, new_width), interpolation=self.mask_interpolation)

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

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(images, *args, **params)

    @batch_transform("spatial", has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(volume, *args, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(volumes, *args, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_mask3d(self, mask3d: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply_to_mask(mask3d, *args, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_masks3d(self, masks3d: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply_to_mask(masks3d, *args, **params)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "max_size", "max_size_hw", "interpolation", "mask_interpolation"


class LongestMaxSize(MaxSizeTransform):
    """Rescale an image so that the longest side is equal to max_size or sides meet max_size_hw constraints,
        keeping the aspect ratio.

    Args:
        max_size (int, Sequence[int], optional): Maximum size of the longest side after the transformation.
            When using a list or tuple, the max size will be randomly selected from the values provided. Default: 1024.
        max_size_hw (tuple[int | None, int | None], optional): Maximum (height, width) constraints. Supports:
            - (height, width): Both dimensions must fit within these bounds
            - (height, None): Only height is constrained, width scales proportionally
            - (None, width): Only width is constrained, height scales proportionally
            If specified, max_size must be None. Default: None.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - If the longest side of the image is already equal to max_size, the image will not be resized.
        - This transform will not crop the image. The resulting image may be smaller than specified in both dimensions.
        - For non-square images, both sides will be scaled proportionally to maintain the aspect ratio.
        - Bounding boxes and keypoints are scaled accordingly.

    Mathematical Details:
        Let (W, H) be the original width and height of the image.

        When using max_size:
            1. The scaling factor s is calculated as:
               s = max_size / max(W, H)
            2. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

        When using max_size_hw=(H_target, W_target):
            1. For both dimensions specified:
               s = min(H_target/H, W_target/W)
               This ensures both dimensions fit within the specified bounds.

            2. For height only (W_target=None):
               s = H_target/H
               Width will scale proportionally.

            3. For width only (H_target=None):
               s = W_target/W
               Height will scale proportionally.

            4. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

    Examples:
        >>> import albumentations as A
        >>> import cv2
        >>> # Using max_size
        >>> transform1 = A.LongestMaxSize(max_size=1024)
        >>> # Input image (1500, 800) -> Output (1024, 546)
        >>>
        >>> # Using max_size_hw with both dimensions
        >>> transform2 = A.LongestMaxSize(max_size_hw=(800, 1024))
        >>> # Input (1500, 800) -> Output (800, 427)
        >>> # Input (800, 1500) -> Output (546, 1024)
        >>>
        >>> # Using max_size_hw with only height
        >>> transform3 = A.LongestMaxSize(max_size_hw=(800, None))
        >>> # Input (1500, 800) -> Output (800, 427)
        >>>
        >>> # Common use case with padding
        >>> transform4 = A.Compose([
        ...     A.LongestMaxSize(max_size=1024),
        ...     A.PadIfNeeded(min_height=1024, min_width=1024),
        ... ])
    """

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        img_h, img_w = params["shape"][:2]

        if self.max_size is not None:
            if isinstance(self.max_size, (list, tuple)):
                max_size = self.py_random.choice(self.max_size)
            else:
                max_size = self.max_size
            scale = max_size / max(img_h, img_w)
        elif self.max_size_hw is not None:
            # We know max_size_hw is not None here due to model validator
            max_h, max_w = self.max_size_hw
            if max_h is not None and max_w is not None:
                # Scale based on longest side to maintain aspect ratio
                h_scale = max_h / img_h
                w_scale = max_w / img_w
                scale = min(h_scale, w_scale)
            elif max_h is not None:
                # Only height specified
                scale = max_h / img_h
            else:
                # Only width specified
                scale = max_w / img_w

        return {"scale": scale}


class SmallestMaxSize(MaxSizeTransform):
    """Rescale an image so that minimum side is equal to max_size or sides meet max_size_hw constraints,
    keeping the aspect ratio.

    Args:
        max_size (int, list of int, optional): Maximum size of smallest side of the image after the transformation.
            When using a list, max size will be randomly selected from the values in the list. Default: 1024.
        max_size_hw (tuple[int | None, int | None], optional): Maximum (height, width) constraints. Supports:
            - (height, width): Both dimensions must be at least these values
            - (height, None): Only height is constrained, width scales proportionally
            - (None, width): Only width is constrained, height scales proportionally
            If specified, max_size must be None. Default: None.
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - If the smallest side of the image is already equal to max_size, the image will not be resized.
        - This transform will not crop the image. The resulting image may be larger than specified in both dimensions.
        - For non-square images, both sides will be scaled proportionally to maintain the aspect ratio.
        - Bounding boxes and keypoints are scaled accordingly.

    Mathematical Details:
        Let (W, H) be the original width and height of the image.

        When using max_size:
            1. The scaling factor s is calculated as:
               s = max_size / min(W, H)
            2. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

        When using max_size_hw=(H_target, W_target):
            1. For both dimensions specified:
               s = max(H_target/H, W_target/W)
               This ensures both dimensions are at least as large as specified.

            2. For height only (W_target=None):
               s = H_target/H
               Width will scale proportionally.

            3. For width only (H_target=None):
               s = W_target/W
               Height will scale proportionally.

            4. The new dimensions (W', H') are:
               W' = W * s
               H' = H * s

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> # Using max_size
        >>> transform1 = A.SmallestMaxSize(max_size=120)
        >>> # Input image (100, 150) -> Output (120, 180)
        >>>
        >>> # Using max_size_hw with both dimensions
        >>> transform2 = A.SmallestMaxSize(max_size_hw=(100, 200))
        >>> # Input (80, 160) -> Output (100, 200)
        >>> # Input (160, 80) -> Output (400, 200)
        >>>
        >>> # Using max_size_hw with only height
        >>> transform3 = A.SmallestMaxSize(max_size_hw=(100, None))
        >>> # Input (80, 160) -> Output (100, 200)
    """

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        img_h, img_w = params["shape"][:2]

        if self.max_size is not None:
            if isinstance(self.max_size, (list, tuple)):
                max_size = self.py_random.choice(self.max_size)
            else:
                max_size = self.max_size
            scale = max_size / min(img_h, img_w)
        elif self.max_size_hw is not None:
            max_h, max_w = self.max_size_hw
            if max_h is not None and max_w is not None:
                # Scale based on smallest side to maintain aspect ratio
                h_scale = max_h / img_h
                w_scale = max_w / img_w
                scale = max(h_scale, w_scale)
            elif max_h is not None:
                # Only height specified
                scale = max_h / img_h
            else:
                # Only width specified
                scale = max_w / img_w

        return {"scale": scale}


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
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    """

    _targets = ALL_TARGETS

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
        p: float = 1,
    ):
        super().__init__(p=p)
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
