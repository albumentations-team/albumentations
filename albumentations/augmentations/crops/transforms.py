import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import cv2
import numpy as np
from pydantic import AfterValidator, Field, field_validator, model_validator
from typing_extensions import Annotated, Self

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.bbox_utils import union_of_bboxes
from albumentations.core.pydantic import (
    BorderModeType,
    InterpolationType,
    OnePlusIntRangeType,
    ProbabilityType,
    ZeroOneRangeType,
    check_0plus,
    check_01,
)
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import (
    NUM_MULTI_CHANNEL_DIMENSIONS,
    PAIR,
    BoxInternalType,
    KeypointInternalType,
    PercentType,
    PxType,
    ScalarType,
    ScaleFloatType,
    ScaleIntType,
    Targets,
)

from . import functional as fcrops

__all__ = [
    "RandomCrop",
    "CenterCrop",
    "Crop",
    "CropNonEmptyMaskIfExists",
    "RandomSizedCrop",
    "RandomResizedCrop",
    "RandomCropNearBBox",
    "RandomSizedBBoxSafeCrop",
    "CropAndPad",
    "RandomCropFromBorders",
    "BBoxSafeRandomCrop",
]


class CropInitSchema(BaseTransformInitSchema):
    height: Optional[int] = Field(description="Height of the crop", ge=1)
    width: Optional[int] = Field(description="Width of the crop", ge=1)
    p: ProbabilityType = 1


class RandomCrop(DualTransform):
    """Crop a random part of the input.

    Args:
        height: height of the crop.
        width: width of the crop.
        p: probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(CropInitSchema):
        pass

    def __init__(self, height: int, width: int, always_apply: Optional[bool] = None, p: float = 1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img: np.ndarray, h_start: int, w_start: int, **params: Any) -> np.ndarray:
        return fcrops.random_crop(img, self.height, self.width, h_start, w_start)

    def get_params(self) -> Dict[str, float]:
        return {"h_start": random.random(), "w_start": random.random()}

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return fcrops.bbox_random_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        return fcrops.keypoint_random_crop(keypoint, self.height, self.width, **params)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("height", "width")


class CenterCrop(DualTransform):
    """Crop the central part of the input.

    Args:
        height: height of the crop.
        width: width of the crop.
        p: probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(CropInitSchema):
        pass

    def __init__(self, height: int, width: int, always_apply: Optional[bool] = None, p: float = 1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return fcrops.center_crop(img, self.height, self.width)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return fcrops.bbox_center_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        return fcrops.keypoint_center_crop(keypoint, self.height, self.width, **params)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("height", "width")


class Crop(DualTransform):
    """Crop region from image.

    Args:
        x_min: Minimum upper left x coordinate.
        y_min: Minimum upper left y coordinate.
        x_max: Maximum lower right x coordinate.
        y_max: Maximum lower right y coordinate.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        x_min: Annotated[int, Field(ge=0, description="Minimum upper left x coordinate")]
        y_min: Annotated[int, Field(ge=0, description="Minimum upper left y coordinate")]
        x_max: Annotated[int, Field(gt=0, description="Maximum lower right x coordinate")]
        y_max: Annotated[int, Field(gt=0, description="Maximum lower right y coordinate")]
        p: ProbabilityType = 1

        @model_validator(mode="after")
        def validate_coordinates(self) -> Self:
            if not self.x_min < self.x_max:
                msg = "x_max must be greater than x_min"
                raise ValueError(msg)
            if not self.y_min < self.y_max:
                msg = "y_max must be greater than y_min"
                raise ValueError(msg)
            return self

    def __init__(
        self,
        x_min: int = 0,
        y_min: int = 0,
        x_max: int = 1024,
        y_max: int = 1024,
        always_apply: Optional[bool] = None,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return fcrops.crop(img, x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return fcrops.bbox_crop(bbox, x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        return fcrops.crop_keypoint_by_coords(keypoint, crop_coords=(self.x_min, self.y_min, self.x_max, self.y_max))

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return ("x_min", "y_min", "x_max", "y_max")


class CropNonEmptyMaskIfExists(DualTransform):
    """Crop area with mask if mask is non-empty, else make random crop.

    Args:
        height: vertical size of crop in pixels
        width: horizontal size of crop in pixels
        ignore_values (list of int): values to ignore in mask, `0` values are always ignored
            (e.g. if background value is 5 set `ignore_values=[5]` to ignore)
        ignore_channels (list of int): channels to ignore in mask
            (e.g. if background is a first channel set `ignore_channels=[0]` to ignore)
        p: probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(CropInitSchema):
        ignore_values: Optional[List[int]] = Field(
            default=None,
            description="Values to ignore in mask, `0` values are always ignored",
        )
        ignore_channels: Optional[List[int]] = Field(default=None, description="Channels to ignore in mask")

    def __init__(
        self,
        height: int,
        width: int,
        ignore_values: Optional[List[int]] = None,
        ignore_channels: Optional[List[int]] = None,
        always_apply: Optional[bool] = None,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)

        self.height = height
        self.width = width
        self.ignore_values = ignore_values
        self.ignore_channels = ignore_channels

    def apply(
        self,
        img: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop(img, x_min, y_min, x_max, y_max)

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> BoxInternalType:
        return fcrops.bbox_crop(
            bbox,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            rows=params["rows"],
            cols=params["cols"],
        )

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> KeypointInternalType:
        return fcrops.crop_keypoint_by_coords(keypoint, crop_coords=(x_min, y_min, x_max, y_max))

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_height, mask_width = mask.shape[:2]

        if self.ignore_values is not None:
            ignore_values_np = np.array(self.ignore_values)
            mask = np.where(np.isin(mask, ignore_values_np), 0, mask)

        if mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS and self.ignore_channels is not None:
            target_channels = np.array([ch for ch in range(mask.shape[-1]) if ch not in self.ignore_channels])
            mask = np.take(mask, target_channels, axis=-1)

        if self.height > mask_height or self.width > mask_width:
            raise ValueError(
                f"Crop size ({self.height},{self.width}) is larger than image ({mask_height},{mask_width})",
            )

        return mask

    def update_params(self, params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        super().update_params(params, **kwargs)
        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = kwargs["masks"]
            mask = self._preprocess_mask(np.copy(masks[0]))  # need copy as we perform in-place mod afterwards
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            msg = "Can not find mask for CropNonEmptyMaskIfExists"
            raise RuntimeError(msg)

        mask_height, mask_width = mask.shape[:2]

        if mask.any():
            mask = mask.sum(axis=-1) if mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, self.width - 1)
            y_min = y - random.randint(0, self.height - 1)
            x_min = np.clip(x_min, 0, mask_width - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
        else:
            x_min = random.randint(0, mask_width - self.width)
            y_min = random.randint(0, mask_height - self.height)

        x_max = x_min + self.width
        y_max = y_min + self.height

        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max})
        return params

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return ("height", "width", "ignore_values", "ignore_channels")


class BaseRandomSizedCropInitSchema(BaseTransformInitSchema):
    size: Tuple[int, int]

    @field_validator("size")
    @classmethod
    def check_size(cls, value: Tuple[int, int]) -> Tuple[int, int]:
        if any(x <= 0 for x in value):
            raise ValueError("All elements of 'size' must be positive integers.")
        return value


class _BaseRandomSizedCrop(DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    class InitSchema(BaseRandomSizedCropInitSchema):
        interpolation: InterpolationType = cv2.INTER_LINEAR
        size: Tuple[int, int]

    def __init__(
        self,
        size: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: Optional[bool] = None,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.size = size
        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_height: int,
        crop_width: int,
        h_start: int,
        w_start: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        crop = fcrops.random_crop(img, crop_height, crop_width, h_start, w_start)
        return fgeometric.resize(crop, self.size[0], self.size[1], interpolation)

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        crop_height: int,
        crop_width: int,
        h_start: int,
        w_start: int,
        rows: int,
        cols: int,
        **params: Any,
    ) -> BoxInternalType:
        return fcrops.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        crop_height: int,
        crop_width: int,
        h_start: int,
        w_start: int,
        rows: int,
        cols: int,
        **params: Any,
    ) -> KeypointInternalType:
        keypoint = fcrops.keypoint_random_crop(keypoint, crop_height, crop_width, h_start, w_start, rows, cols)
        scale_y = self.size[0] / crop_height
        scale_x = self.size[1] / crop_width
        return fgeometric.keypoint_scale(keypoint, scale_x, scale_y)


class RandomSizedCrop(_BaseRandomSizedCrop):
    """Crop a random portion of the input and rescale it to a specific size.

    Args:
        min_max_height ((int, int)): crop size limits.
        size ((int, int)): target size for the output image, i.e. (height, width) after crop and resize
        w2h_ratio (float): aspect ratio of crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        interpolation: InterpolationType = cv2.INTER_LINEAR
        p: ProbabilityType = 1
        min_max_height: OnePlusIntRangeType
        w2h_ratio: Annotated[float, Field(gt=0, description="Aspect ratio of crop.")]
        width: Optional[int] = Field(
            None,
            deprecated=(
                "Initializing with 'size' as an integer and a separate 'width' is deprecated. "
                "Please use a tuple (height, width) for the 'size' argument."
            ),
        )
        height: Optional[int] = Field(
            None,
            deprecated=(
                "Initializing with 'height' and 'width' is deprecated. "
                "Please use a tuple (height, width) for the 'size' argument."
            ),
        )
        size: Optional[ScaleIntType] = None

        @model_validator(mode="after")
        def process(self) -> Self:
            if isinstance(self.size, int):
                if isinstance(self.width, int):
                    self.size = (self.size, self.width)
                else:
                    msg = "If size is an integer, width as integer must be specified."
                    raise TypeError(msg)

            if self.size is None:
                if self.height is None or self.width is None:
                    message = "If 'size' is not provided, both 'height' and 'width' must be specified."
                    raise ValueError(message)
                self.size = (self.height, self.width)
            return self

    def __init__(
        self,
        min_max_height: Tuple[int, int],
        # NOTE @zetyquickly: when (width, height) are deprecated, make 'size' non optional
        size: Optional[ScaleIntType] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        w2h_ratio: float = 1.0,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: Optional[bool] = None,
        p: float = 1.0,
    ):
        super().__init__(size=cast(Tuple[int, int], size), interpolation=interpolation, always_apply=always_apply, p=p)
        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def get_params(self) -> Dict[str, Union[int, float]]:
        crop_height = random.randint(self.min_max_height[0], self.min_max_height[1])
        return {
            "h_start": random.random(),
            "w_start": random.random(),
            "crop_height": crop_height,
            "crop_width": int(crop_height * self.w2h_ratio),
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "min_max_height", "size", "w2h_ratio", "interpolation"


class RandomResizedCrop(_BaseRandomSizedCrop):
    """Torchvision's variant of crop a random part of the input and rescale it to some size.

    Args:
        size (int, int): target size for the output image, i.e. (height, width) after crop and resize
        scale ((float, float)): range of size of the origin size cropped
        ratio ((float, float)): range of aspect ratio of the origin aspect ratio cropped
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        scale: Annotated[Tuple[float, float], AfterValidator(check_01)] = (0.08, 1.0)
        ratio: Annotated[Tuple[float, float], AfterValidator(check_0plus)] = (0.75, 1.3333333333333333)
        width: Optional[int] = Field(
            None,
            deprecated="Initializing with 'height' and 'width' is deprecated. Use size instead.",
        )
        height: Optional[int] = Field(
            None,
            deprecated="Initializing with 'height' and 'width' is deprecated. Use size instead.",
        )
        size: Optional[ScaleIntType] = None
        p: ProbabilityType = 1
        interpolation: InterpolationType = cv2.INTER_LINEAR

        @model_validator(mode="after")
        def process(self) -> Self:
            if isinstance(self.size, int):
                if isinstance(self.width, int):
                    self.size = (self.size, self.width)
                else:
                    msg = "If size is an integer, width as integer must be specified."
                    raise TypeError(msg)

            if self.size is None:
                if self.height is None or self.width is None:
                    message = "If 'size' is not provided, both 'height' and 'width' must be specified."
                    raise ValueError(message)
                self.size = (self.height, self.width)

            return self

    def __init__(
        self,
        # NOTE @zetyquickly: when (width, height) are deprecated, make 'size' non optional
        size: Optional[ScaleIntType] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: Optional[bool] = None,
        p: float = 1.0,
    ):
        super().__init__(size=cast(Tuple[int, int], size), interpolation=interpolation, always_apply=always_apply, p=p)
        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        img = params["image"]
        img_height, img_width = img.shape[:2]
        area = img_height * img_width

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            width = int(round(math.sqrt(target_area * aspect_ratio)))
            height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < width <= img_width and 0 < height <= img_height:
                i = random.randint(0, img_height - height)
                j = random.randint(0, img_width - width)
                return {
                    "crop_height": height,
                    "crop_width": width,
                    "h_start": i * 1.0 / (img_height - height + 1e-10),
                    "w_start": j * 1.0 / (img_width - width + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img_width / img_height
        if in_ratio < min(self.ratio):
            width = img_width
            height = int(round(img_width / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            height = img_height
            width = int(round(height * max(self.ratio)))
        else:  # whole image
            width = img_width
            height = img_height
        i = (img_height - height) // 2
        j = (img_width - width) // 2
        return {
            "crop_height": height,
            "crop_width": width,
            "h_start": i * 1.0 / (img_height - height + 1e-10),
            "w_start": j * 1.0 / (img_width - width + 1e-10),
        }

    def get_params(self) -> Dict[str, Any]:
        return {}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "size", "scale", "ratio", "interpolation"


class RandomCropNearBBox(DualTransform):
    """Crop bbox from image with random shift by x,y coordinates

    Args:
        max_part_shift (float, (float, float)): Max shift in `height` and `width` dimensions relative
            to `cropping_bbox` dimension.
            If max_part_shift is a single float, the range will be (0, max_part_shift).
            Default (0, 0.3).
        cropping_bbox_key (str): Additional target key for cropping box. Default `cropping_bbox`.
        cropping_box_key (str): [Deprecated] Use `cropping_bbox_key` instead.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Examples:
        >>> aug = Compose([RandomCropNearBBox(max_part_shift=(0.1, 0.5), cropping_bbox_key='test_bbox')],
        >>>              bbox_params=BboxParams("pascal_voc"))
        >>> result = aug(image=image, bboxes=bboxes, test_bbox=[0, 5, 10, 20])

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        max_part_shift: ZeroOneRangeType = (0, 0.3)
        cropping_bbox_key: str = Field(default="cropping_bbox", description="Additional target key for cropping box.")
        p: ProbabilityType = 1

    def __init__(
        self,
        max_part_shift: ScaleFloatType = (0, 0.3),
        cropping_bbox_key: str = "cropping_bbox",
        cropping_box_key: Optional[str] = None,  # Deprecated
        always_apply: Optional[bool] = None,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        # Check for deprecated parameter and issue warning
        if cropping_box_key is not None:
            warn(
                "The parameter 'cropping_box_key' is deprecated and will be removed in future versions. "
                "Use 'cropping_bbox_key' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Ensure the new parameter is used even if the old one is passed
            cropping_bbox_key = cropping_box_key

        self.max_part_shift = cast(Tuple[float, float], max_part_shift)
        self.cropping_bbox_key = cropping_bbox_key

    def apply(
        self,
        img: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        return fcrops.clamping_crop(img, x_min, y_min, x_max, y_max)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, int]:
        bbox = params[self.cropping_bbox_key]
        image = params["image"]
        height, width = image.shape[:2]

        h_max_shift = round((bbox[3] - bbox[1]) * self.max_part_shift[0])
        w_max_shift = round((bbox[2] - bbox[0]) * self.max_part_shift[1])

        x_min = bbox[0] - random.randint(-w_max_shift, w_max_shift)
        x_max = bbox[2] + random.randint(-w_max_shift, w_max_shift)

        y_min = bbox[1] - random.randint(-h_max_shift, h_max_shift)
        y_max = bbox[3] + random.randint(-h_max_shift, h_max_shift)

        x_min = max(0, x_min)
        y_min = max(0, y_min)

        x_max = min(width, x_max)
        y_max = min(height, y_max)

        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return fcrops.bbox_crop(bbox, **params)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> KeypointInternalType:
        return fcrops.crop_keypoint_by_coords(keypoint, crop_coords=(x_min, y_min, x_max, y_max))

    @property
    def targets_as_params(self) -> List[str]:
        return ["image", self.cropping_bbox_key]

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("max_part_shift", "cropping_bbox_key")


class BBoxSafeRandomCrop(DualTransform):
    """Crop a random part of the input without loss of bboxes.

    Args:
        erosion_rate: erosion rate applied on input image height before crop.
        p: probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes
    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        erosion_rate: float = Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Erosion rate applied on input image height before crop.",
        )
        p: ProbabilityType = 1

    def __init__(self, erosion_rate: float = 0.0, always_apply: Optional[bool] = None, p: float = 1.0):
        super().__init__(always_apply, p)
        self.erosion_rate = erosion_rate

    def apply(
        self,
        img: np.ndarray,
        crop_height: int,
        crop_width: int,
        h_start: int,
        w_start: int,
        **params: Any,
    ) -> np.ndarray:
        return fcrops.random_crop(img, crop_height, crop_width, h_start, w_start)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
            }
        # get union of all bboxes
        x, y, x2, y2 = union_of_bboxes(
            width=img_w,
            height=img_h,
            bboxes=params["bboxes"],
            erosion_rate=self.erosion_rate,
        )
        # find bigger region
        bx, by = x * random.random(), y * random.random()
        bx2, by2 = x2 + (1 - x2) * random.random(), y2 + (1 - y2) * random.random()
        bw, bh = bx2 - bx, by2 - by
        crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w if bw >= 1.0 else int(img_w * bw)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        crop_height: int,
        crop_width: int,
        h_start: int,
        w_start: int,
        rows: int,
        cols: int,
        **params: Any,
    ) -> BoxInternalType:
        return fcrops.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    @property
    def targets_as_params(self) -> List[str]:
        return ["image", "bboxes"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("erosion_rate",)

    @property
    def targets(self) -> Dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
        }


class RandomSizedBBoxSafeCrop(BBoxSafeRandomCrop):
    """Crop a random part of the input and rescale it to some size without loss of bboxes.

    Args:
        height: height after crop and resize.
        width: width after crop and resize.
        erosion_rate: erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes
    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES)

    class InitSchema(CropInitSchema):
        erosion_rate: float = Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Erosion rate applied on input image height before crop.",
        )
        interpolation: InterpolationType = cv2.INTER_LINEAR

    def __init__(
        self,
        height: int,
        width: int,
        erosion_rate: float = 0.0,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: Optional[bool] = None,
        p: float = 1.0,
    ):
        super().__init__(erosion_rate, always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_height: int,
        crop_width: int,
        h_start: int,
        w_start: int,
        **params: Any,
    ) -> np.ndarray:
        crop = fcrops.random_crop(img, crop_height, crop_width, h_start, w_start)
        return fgeometric.resize(crop, self.height, self.width, self.interpolation)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return (*super().get_transform_init_args_names(), "height", "width", "interpolation")


class CropAndPad(DualTransform):
    """Crop and pad images by pixel amounts or fractions of image sizes.
    Cropping removes pixels at the sides (i.e., extracts a subimage from a given full image).
    Padding adds pixels to the sides (e.g., black pixels).
    This transformation will never crop images below a height or width of 1.

    Note:
        This transformation automatically resizes images back to their original size. To deactivate this, add the
        parameter `keep_size=False`.

    Args:
        px (int,
            Tuple[int, int],
            Tuple[int, int, int, int],
            Tuple[Union[int, Tuple[int, int], List[int]],
                  Union[int, Tuple[int, int], List[int]],
                  Union[int, Tuple[int, int], List[int]],
                  Union[int, Tuple[int, int], List[int]]]):
            The number of pixels to crop (negative values) or pad (positive values) on each side of the image.
                Either this or the parameter `percent` may be set, not both at the same time.

                * If `None`, then pixel-based cropping/padding will not be used.
                * If `int`, then that exact number of pixels will always be cropped/padded.
                * If a `tuple` of two `int`s with values `a` and `b`, then each side will be cropped/padded by a
                    random amount sampled uniformly per image and side from the interval `[a, b]`.
                    If `sample_independently` is set to `False`, only one value will be sampled per
                        image and used for all sides.
                * If a `tuple` of four entries, then the entries represent top, right, bottom, and left.
                    Each entry may be:
                    - A single `int` (always crop/pad by exactly that value).
                    - A `tuple` of two `int`s `a` and `b` (crop/pad by an amount within `[a, b]`).
                    - A `list` of `int`s (crop/pad by a random value that is contained in the `list`).

        percent (float,
                 Tuple[float, float],
                 Tuple[float, float, float, float],
                 Tuple[Union[float, Tuple[float, float], List[float]],
                       Union[float, Tuple[float, float], List[float]],
                       Union[float, Tuple[float, float], List[float]],
                       Union[float, Tuple[float, float], List[float]]]):
            The number of pixels to crop (negative values) or pad (positive values) on each side of the image given
                as a *fraction* of the image height/width. E.g. if this is set to `-0.1`, the transformation will
                always crop away `10%` of the image's height at both the top and the bottom (both `10%` each),
                as well as `10%` of the width at the right and left. Expected value range is `(-1.0, inf)`.
                Either this or the parameter `px` may be set, not both at the same time.

                * If `None`, then fraction-based cropping/padding will not be used.
                * If `float`, then that fraction will always be cropped/padded.
                * If a `tuple` of two `float`s with values `a` and `b`, then each side will be cropped/padded by a
                random fraction sampled uniformly per image and side from the interval `[a, b]`.
                If `sample_independently` is set to `False`, only one value will be sampled per image and used
                for all sides.
                * If a `tuple` of four entries, then the entries represent top, right, bottom, and left.
                    Each entry may be:
                    - A single `float` (always crop/pad by exactly that percent value).
                    - A `tuple` of two `float`s `a` and `b` (crop/pad by a fraction from `[a, b]`).
                    - A `list` of `float`s (crop/pad by a random value that is contained in the `list`).

        pad_mode (int): OpenCV border mode.
        pad_cval (Union[int, float, Tuple[Union[int, float], Union[int, float]], List[Union[int, float]]]):
            The constant value to use if the pad mode is `BORDER_CONSTANT`.
                * If `number`, then that value will be used.
                * If a `tuple` of two numbers and at least one of them is a `float`, then a random number
                    will be uniformly sampled per image from the continuous interval `[a, b]` and used as the value.
                    If both numbers are `int`s, the interval is discrete.
                * If a `list` of numbers, then a random value will be chosen from the elements of the `list` and
                    used as the value.

        pad_cval_mask (Union[int, float, Tuple[Union[int, float], Union[int, float]], List[Union[int, float]]]):
            Same as `pad_cval` but only for masks.

        keep_size (bool):
            After cropping and padding, the resulting image will usually have a different height/width compared to
            the original input image. If this parameter is set to `True`, then the cropped/padded image will be
            resized to the input image's size, i.e., the output shape is always identical to the input shape.

        sample_independently (bool):
            If `False` and the values for `px`/`percent` result in exactly one probability distribution for all
            image sides, only one single value will be sampled from that probability distribution and used for
            all sides. I.e., the crop/pad amount then is the same for all sides. If `True`, four values
            will be sampled independently, one per side.

        interpolation (int):
            OpenCV flag that is used to specify the interpolation algorithm for images. Should be one of:
            `cv2.INTER_NEAREST`, `cv2.INTER_LINEAR`, `cv2.INTER_CUBIC`, `cv2.INTER_AREA`, `cv2.INTER_LANCZOS4`.
            Default: `cv2.INTER_LINEAR`.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        unit8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        px: Optional[PxType] = Field(
            default=None,
            description="Number of pixels to crop (negative) or pad (positive).",
        )
        percent: Optional[PercentType] = Field(
            default=None,
            description="Fraction of image size to crop (negative) or pad (positive).",
        )
        pad_mode: BorderModeType = cv2.BORDER_CONSTANT
        pad_cval: Union[ScalarType, Tuple[ScalarType, ScalarType], List[ScalarType]] = Field(
            default=0,
            description="Padding value if pad_mode is BORDER_CONSTANT.",
        )
        pad_cval_mask: Union[ScalarType, Tuple[ScalarType, ScalarType], List[ScalarType]] = Field(
            default=0,
            description="Padding value for masks if pad_mode is BORDER_CONSTANT.",
        )
        keep_size: bool = Field(
            default=True,
            description="Whether to resize the image back to the original size after cropping and padding.",
        )
        sample_independently: bool = Field(
            default=True,
            description="Whether to sample the crop/pad size independently for each side.",
        )
        interpolation: InterpolationType = cv2.INTER_LINEAR
        p: ProbabilityType = 1

        @model_validator(mode="after")
        def check_px_percent(self) -> Self:
            if self.px is None and self.percent is None:
                msg = "Both px and percent parameters cannot be None simultaneously."
                raise ValueError(msg)
            if self.px is not None and self.percent is not None:
                msg = "Only px or percent may be set!"
                raise ValueError(msg)
            return self

    def __init__(
        self,
        px: Optional[Union[int, List[int]]] = None,
        percent: Optional[Union[float, List[float]]] = None,
        pad_mode: int = cv2.BORDER_CONSTANT,
        pad_cval: Union[ScalarType, Tuple[ScalarType, ScalarType], List[ScalarType]] = 0,
        pad_cval_mask: Union[ScalarType, Tuple[ScalarType, ScalarType], List[ScalarType]] = 0,
        keep_size: bool = True,
        sample_independently: bool = True,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: Optional[bool] = None,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)

        self.px = px
        self.percent = percent

        self.pad_mode = pad_mode
        self.pad_cval = pad_cval
        self.pad_cval_mask = pad_cval_mask

        self.keep_size = keep_size
        self.sample_independently = sample_independently

        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        crop_params: Sequence[int],
        pad_params: Sequence[int],
        pad_value: float,
        rows: int,
        cols: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop_and_pad(
            img,
            crop_params,
            pad_params,
            pad_value,
            rows,
            cols,
            interpolation,
            self.pad_mode,
            self.keep_size,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        crop_params: Sequence[int],
        pad_params: Sequence[int],
        pad_value_mask: float,
        rows: int,
        cols: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fcrops.crop_and_pad(
            mask,
            crop_params,
            pad_params,
            pad_value_mask,
            rows,
            cols,
            interpolation,
            self.pad_mode,
            self.keep_size,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        crop_params: Sequence[int],
        pad_params: Sequence[int],
        rows: int,
        cols: int,
        result_rows: int,
        result_cols: int,
        **params: Any,
    ) -> BoxInternalType:
        return fcrops.crop_and_pad_bbox(bbox, crop_params, pad_params, rows, cols, result_rows, result_cols)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        crop_params: Sequence[int],
        pad_params: Sequence[int],
        rows: int,
        cols: int,
        result_rows: int,
        result_cols: int,
        **params: Any,
    ) -> KeypointInternalType:
        return fcrops.crop_and_pad_keypoint(
            keypoint,
            crop_params,
            pad_params,
            rows,
            cols,
            result_rows,
            result_cols,
            self.keep_size,
        )

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    @staticmethod
    def __prevent_zero(val1: int, val2: int, max_val: int) -> Tuple[int, int]:
        regain = abs(max_val) + 1
        regain1 = regain // 2
        regain2 = regain // 2
        if regain1 + regain2 < regain:
            regain1 += 1

        if regain1 > val1:
            diff = regain1 - val1
            regain1 = val1
            regain2 += diff
        elif regain2 > val2:
            diff = regain2 - val2
            regain2 = val2
            regain1 += diff

        return val1 - regain1, val2 - regain2

    @staticmethod
    def _prevent_zero(crop_params: List[int], height: int, width: int) -> List[int]:
        top, right, bottom, left = crop_params

        remaining_height = height - (top + bottom)
        remaining_width = width - (left + right)

        if remaining_height < 1:
            top, bottom = CropAndPad.__prevent_zero(top, bottom, height)
        if remaining_width < 1:
            left, right = CropAndPad.__prevent_zero(left, right, width)

        return [max(top, 0), max(right, 0), max(bottom, 0), max(left, 0)]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        height, width = params["image"].shape[:2]

        if self.px is not None:
            new_params = self._get_px_params()
        else:
            percent_params = self._get_percent_params()
            new_params = [
                int(percent_params[0] * height),
                int(percent_params[1] * width),
                int(percent_params[2] * height),
                int(percent_params[3] * width),
            ]

        pad_params = [max(i, 0) for i in new_params]

        crop_params = self._prevent_zero([-min(i, 0) for i in new_params], height, width)

        top, right, bottom, left = crop_params
        crop_params = [left, top, width - right, height - bottom]
        result_rows = crop_params[3] - crop_params[1]
        result_cols = crop_params[2] - crop_params[0]
        if result_cols == width and result_rows == height:
            crop_params = []

        top, right, bottom, left = pad_params
        pad_params = [top, bottom, left, right]
        if any(pad_params):
            result_rows += top + bottom
            result_cols += left + right
        else:
            pad_params = []

        return {
            "crop_params": crop_params or None,
            "pad_params": pad_params or None,
            "pad_value": None if pad_params is None else self._get_pad_value(self.pad_cval),
            "pad_value_mask": None if pad_params is None else self._get_pad_value(self.pad_cval_mask),
            "result_rows": result_rows,
            "result_cols": result_cols,
        }

    def _get_px_params(self) -> List[int]:
        if self.px is None:
            msg = "px is not set"
            raise ValueError(msg)

        if isinstance(self.px, int):
            params = [self.px] * 4
        elif len(self.px) == PAIR:
            if self.sample_independently:
                params = [random.randrange(*self.px) for _ in range(4)]
            else:
                px = random.randrange(*self.px)
                params = [px] * 4
        elif isinstance(self.px[0], int):
            params = self.px
        elif len(self.px[0]) == PAIR:
            params = [random.randrange(*i) for i in self.px]
        else:
            params = [random.choice(i) for i in self.px]

        return params

    def _get_percent_params(self) -> List[float]:
        if self.percent is None:
            msg = "percent is not set"
            raise ValueError(msg)

        if isinstance(self.percent, float):
            params = [self.percent] * 4
        elif len(self.percent) == PAIR:
            if self.sample_independently:
                params = [random.uniform(*self.percent) for _ in range(4)]
            else:
                px = random.uniform(*self.percent)
                params = [px] * 4
        elif isinstance(self.percent[0], (int, float)):
            params = self.percent
        elif len(self.percent[0]) == PAIR:
            params = [random.uniform(*i) for i in self.percent]
        else:
            params = [random.choice(i) for i in self.percent]

        return params  # params = [top, right, bottom, left]

    @staticmethod
    def _get_pad_value(
        pad_value: Union[ScalarType, Tuple[ScalarType, ScalarType], List[ScalarType]],
    ) -> ScalarType:
        if isinstance(pad_value, (int, float)):
            return pad_value

        if len(pad_value) == PAIR:
            a, b = pad_value
            if isinstance(a, int) and isinstance(b, int):
                return random.randint(a, b)

            return random.uniform(a, b)

        return random.choice(pad_value)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return (
            "px",
            "percent",
            "pad_mode",
            "pad_cval",
            "pad_cval_mask",
            "keep_size",
            "sample_independently",
            "interpolation",
        )


class RandomCropFromBorders(DualTransform):
    """Randomly crops parts of the image from the borders without resizing at the end. The cropped regions are defined
    as fractions of the original image dimensions, specified for each side of the image (left, right, top, bottom).

    Args:
        crop_left (float): Fraction of the width to randomly crop from the left side. Must be in the range [0.0, 1.0].
                            Default is 0.1.
        crop_right (float): Fraction of the width to randomly crop from the right side. Must be in the range [0.0, 1.0].
                            Default is 0.1.
        crop_top (float): Fraction of the height to randomly crop from the top side. Must be in the range [0.0, 1.0].
                          Default is 0.1.
        crop_bottom (float): Fraction of the height to randomly crop from the bottom side.
                             Must be in the range [0.0, 1.0]. Default is 0.1.
        p (float): Probability of applying the transform. Default is 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        crop_left: float = Field(
            default=0.1,
            ge=0.0,
            le=1.0,
            description="Fraction of width to randomly crop from the left side.",
        )
        crop_right: float = Field(
            default=0.1,
            ge=0.0,
            le=1.0,
            description="Fraction of width to randomly crop from the right side.",
        )
        crop_top: float = Field(
            default=0.1,
            ge=0.0,
            le=1.0,
            description="Fraction of height to randomly crop from the top side.",
        )
        crop_bottom: float = Field(
            default=0.1,
            ge=0.0,
            le=1.0,
            description="Fraction of height to randomly crop from the bottom side.",
        )
        p: ProbabilityType = 1

        @model_validator(mode="after")
        def validate_crop_values(self) -> Self:
            if self.crop_left + self.crop_right > 1.0:
                msg = "The sum of crop_left and crop_right must be <= 1."
                raise ValueError(msg)
            if self.crop_top + self.crop_bottom > 1.0:
                msg = "The sum of crop_top and crop_bottom must be <= 1."
                raise ValueError(msg)
            return self

    def __init__(
        self,
        crop_left: float = 0.1,
        crop_right: float = 0.1,
        crop_top: float = 0.1,
        crop_bottom: float = 0.1,
        always_apply: Optional[bool] = None,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, int]:
        height, width = params["image"].shape[:2]

        x_min = random.randint(0, int(self.crop_left * width))
        x_max = random.randint(max(x_min + 1, int((1 - self.crop_right) * width)), width)

        y_min = random.randint(0, int(self.crop_top * height))
        y_max = random.randint(max(y_min + 1, int((1 - self.crop_bottom) * height)), height)

        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def apply(
        self,
        img: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        return fcrops.clamping_crop(img, x_min, y_min, x_max, y_max)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        return fcrops.clamping_crop(mask, x_min, y_min, x_max, y_max)

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> BoxInternalType:
        rows, cols = params["rows"], params["cols"]
        return fcrops.bbox_crop(bbox, x_min, y_min, x_max, y_max, rows, cols)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> KeypointInternalType:
        return fcrops.crop_keypoint_by_coords(keypoint, crop_coords=(x_min, y_min, x_max, y_max))

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "crop_left", "crop_right", "crop_top", "crop_bottom"
