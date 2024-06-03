import math
import random
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
from pydantic import Field
from typing_extensions import Literal

from albumentations.augmentations.crops import functional as fcrops
from albumentations.augmentations.functional import center
from albumentations.core.pydantic import BorderModeType, InterpolationType, SymmetricRangeType
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import (
    BoxInternalType,
    ColorType,
    KeypointInternalType,
    ScaleFloatType,
    Targets,
)

from . import functional as fgeometric

__all__ = ["Rotate", "RandomRotate90", "SafeRotate"]

SMALL_NUMBER = 1e-10


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def apply(self, img: np.ndarray, factor: float, **params: Any) -> np.ndarray:
        """Args:
        factor (int): number of times the input will be rotated by 90 degrees.

        """
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self) -> Dict[str, int]:
        # Random int in the range [0, 3]
        return {"factor": random.randint(0, 3)}

    def apply_to_bbox(self, bbox: BoxInternalType, factor: int, **params: Any) -> BoxInternalType:
        return fgeometric.bbox_rot90(bbox, factor, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, factor: int, **params: Any) -> BoxInternalType:
        return fgeometric.keypoint_rot90(keypoint, factor, **params)

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()


class RotateInitSchema(BaseTransformInitSchema):
    limit: SymmetricRangeType = (-90, 90)

    interpolation: InterpolationType = cv2.INTER_LINEAR
    border_mode: BorderModeType = cv2.BORDER_CONSTANT

    value: Optional[ColorType] = Field(default=None, description="Padding value if border_mode is cv2.BORDER_CONSTANT.")
    mask_value: Optional[ColorType] = Field(
        default=None,
        description="Padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.",
    )


class Rotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit: range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        crop_border (bool): If True would make a largest possible crop within rotated image
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(RotateInitSchema):
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box"
        crop_border: bool = Field(
            default=False,
            description="If True, makes a largest possible crop within the rotated image.",
        )

    def __init__(
        self,
        limit: ScaleFloatType = (-90, 90),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: Optional[ColorType] = None,
        mask_value: Optional[ColorType] = None,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        crop_border: bool = False,
        always_apply: Optional[bool] = None,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.limit = cast(Tuple[float, float], limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method
        self.crop_border = crop_border

    def apply(
        self,
        img: np.ndarray,
        angle: float,
        interpolation: int,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        img_out = fgeometric.rotate(img, angle, interpolation, self.border_mode, self.value)
        if self.crop_border:
            return fcrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_mask(
        self,
        mask: np.ndarray,
        angle: float,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        img_out = fgeometric.rotate(mask, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value)
        if self.crop_border:
            return fcrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        angle: float,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        cols: int,
        rows: int,
        **params: Any,
    ) -> np.ndarray:
        bbox_out = fgeometric.bbox_rotate(bbox, angle, self.rotate_method, rows, cols)
        if self.crop_border:
            return fcrops.bbox_crop(bbox_out, x_min, y_min, x_max, y_max, rows, cols)
        return bbox_out

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        angle: float,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        cols: int,
        rows: int,
        **params: Any,
    ) -> KeypointInternalType:
        keypoint_out = fgeometric.keypoint_rotate(keypoint, angle, rows, cols, **params)
        if self.crop_border:
            return fcrops.crop_keypoint_by_coords(keypoint_out, (x_min, y_min, x_max, y_max))
        return keypoint_out

    @staticmethod
    def _rotated_rect_with_max_area(height: int, width: int, angle: float) -> Dict[str, int]:
        """Given a rectangle of size wxh that has been rotated by 'angle' (in
        degrees), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.

        Reference:
            https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """
        angle = math.radians(angle)
        width_is_longer = width >= height
        side_long, side_short = (width, height) if width_is_longer else (height, width)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # it is sufficient to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < SMALL_NUMBER:
            # half constrained case: two crop corners touch the longer side,
            # the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a

        return {
            "x_min": max(0, int(width / 2 - wr / 2)),
            "x_max": min(width, int(width / 2 + wr / 2)),
            "y_min": max(0, int(height / 2 - hr / 2)),
            "y_max": min(height, int(height / 2 + hr / 2)),
        }

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out_params = {"angle": random.uniform(self.limit[0], self.limit[1])}
        if self.crop_border:
            height, width = params["image"].shape[:2]
            out_params.update(self._rotated_rect_with_max_area(height, width, out_params["angle"]))
        else:
            out_params.update({"x_min": -1, "x_max": -1, "y_min": -1, "y_max": -1})

        return out_params

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("limit", "interpolation", "border_mode", "value", "mask_value", "rotate_method", "crop_border")


class SafeRotate(DualTransform):
    """Rotate the input inside the input's frame by an angle selected randomly from the uniform distribution.

    The resulting image may have artifacts in it. After rotation, the image may have a different aspect ratio, and
    after resizing, it returns to its original shape with the original aspect ratio of the image. For these reason we
    may see some artifacts.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(RotateInitSchema):
        pass

    def __init__(
        self,
        limit: ScaleFloatType = (-90, 90),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: Optional[ColorType] = None,
        mask_value: Optional[ColorType] = None,
        always_apply: Optional[bool] = None,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.limit = cast(Tuple[float, float], limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img: np.ndarray, matrix: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.safe_rotate(img, matrix, self.interpolation, self.value, self.border_mode)

    def apply_to_mask(self, mask: np.ndarray, matrix: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.safe_rotate(mask, matrix, cv2.INTER_NEAREST, self.mask_value, self.border_mode)

    def apply_to_bbox(self, bbox: BoxInternalType, cols: int, rows: int, **params: Any) -> BoxInternalType:
        return fgeometric.bbox_safe_rotate(bbox, params["matrix"], cols, rows)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        angle: float,
        scale_x: float,
        scale_y: float,
        cols: int,
        rows: int,
        **params: Any,
    ) -> KeypointInternalType:
        return fgeometric.keypoint_safe_rotate(keypoint, params["matrix"], angle, scale_x, scale_y, cols, rows)

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        angle = random.uniform(self.limit[0], self.limit[1])

        image = params["image"]
        height, width = image.shape[:2]

        # https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
        image_center = center(width, height)

        # Rotation Matrix
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        new_w = math.ceil(height * abs_sin + width * abs_cos)
        new_h = math.ceil(height * abs_cos + width * abs_sin)

        scale_x = width / new_w
        scale_y = height / new_h

        # Shift the image to create padding
        rotation_mat[0, 2] += new_w / 2 - image_center[0]
        rotation_mat[1, 2] += new_h / 2 - image_center[1]

        # Rescale to original size
        scale_mat = np.diag(np.ones(3))
        scale_mat[0, 0] *= scale_x
        scale_mat[1, 1] *= scale_y
        _tmp = np.diag(np.ones(3))
        _tmp[:2] = rotation_mat
        _tmp = scale_mat @ _tmp
        rotation_mat = _tmp[:2]

        return {"matrix": rotation_mat, "angle": angle, "scale_x": scale_x, "scale_y": scale_y}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "limit", "interpolation", "border_mode", "value", "mask_value"
