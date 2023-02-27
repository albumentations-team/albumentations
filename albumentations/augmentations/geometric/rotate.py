import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ...core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    FillValueType,
    KeypointInternalType,
    to_tuple,
)
from ..crops import functional as FCrops
from . import functional as F

__all__ = ["Rotate", "RandomRotate90", "SafeRotate"]


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, factor=0, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        # Random int in the range [0, 3]
        return {"factor": random.randint(0, 3)}

    def apply_to_bbox(self, bbox, factor=0, **params):
        return F.bbox_rot90(bbox, factor, **params)

    def apply_to_keypoint(self, keypoint, factor=0, **params):
        return F.keypoint_rot90(keypoint, factor, **params)

    def get_transform_init_args_names(self):
        return ()


class Rotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.

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
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        crop_border (bool): If True would make a largest possible crop within rotated image
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        rotate_method="largest_box",
        crop_border=False,
        always_apply=False,
        p=0.5,
    ):
        super(Rotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method
        self.crop_border = crop_border

        if rotate_method not in ["largest_box", "ellipse"]:
            raise ValueError(f"Rotation method {self.rotate_method} is not valid.")

    def apply(
        self, img, angle=0, interpolation=cv2.INTER_LINEAR, x_min=None, x_max=None, y_min=None, y_max=None, **params
    ):
        img_out = F.rotate(img, angle, interpolation, self.border_mode, self.value)
        if self.crop_border:
            img_out = FCrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_mask(self, img, angle=0, x_min=None, x_max=None, y_min=None, y_max=None, **params):
        img_out = F.rotate(img, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value)
        if self.crop_border:
            img_out = FCrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_bbox(self, bbox, angle=0, x_min=None, x_max=None, y_min=None, y_max=None, cols=0, rows=0, **params):
        bbox_out = F.bbox_rotate(bbox, angle, self.rotate_method, rows, cols)
        if self.crop_border:
            bbox_out = FCrops.bbox_crop(bbox_out, x_min, y_min, x_max, y_max, rows, cols)
        return bbox_out

    def apply_to_keypoint(
        self, keypoint, angle=0, x_min=None, x_max=None, y_min=None, y_max=None, cols=0, rows=0, **params
    ):
        keypoint_out = F.keypoint_rotate(keypoint, angle, rows, cols, **params)
        if self.crop_border:
            keypoint_out = FCrops.crop_keypoint_by_coords(keypoint_out, (x_min, y_min, x_max, y_max))
        return keypoint_out

    @staticmethod
    def _rotated_rect_with_max_area(h, w, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        degrees), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.

        Code from: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """

        angle = math.radians(angle)
        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # it is sufficient to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            # the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

        return dict(
            x_min=max(0, int(w / 2 - wr / 2)),
            x_max=min(w, int(w / 2 + wr / 2)),
            y_min=max(0, int(h / 2 - hr / 2)),
            y_max=min(h, int(h / 2 + hr / 2)),
        )

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out_params = {"angle": random.uniform(self.limit[0], self.limit[1])}
        if self.crop_border:
            h, w = params["image"].shape[:2]
            out_params.update(self._rotated_rect_with_max_area(h, w, out_params["angle"]))
        return out_params

    def get_transform_init_args_names(self):
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

    def __init__(
        self,
        limit: Union[float, Tuple[float, float]] = 90,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: FillValueType = None,
        mask_value: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(SafeRotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img: np.ndarray, matrix: np.ndarray = np.array(None), **params) -> np.ndarray:
        return F.safe_rotate(img, matrix, self.interpolation, self.value, self.border_mode)

    def apply_to_mask(self, img: np.ndarray, matrix: np.ndarray = np.array(None), **params) -> np.ndarray:
        return F.safe_rotate(img, matrix, cv2.INTER_NEAREST, self.mask_value, self.border_mode)

    def apply_to_bbox(self, bbox: BoxInternalType, cols: int = 0, rows: int = 0, **params) -> BoxInternalType:
        return F.bbox_safe_rotate(bbox, params["matrix"], cols, rows)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        angle: float = 0,
        scale_x: float = 0,
        scale_y: float = 0,
        cols: int = 0,
        rows: int = 0,
        **params
    ) -> KeypointInternalType:
        return F.keypoint_safe_rotate(keypoint, params["matrix"], angle, scale_x, scale_y, cols, rows)

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        angle = random.uniform(self.limit[0], self.limit[1])

        image = params["image"]
        h, w = image.shape[:2]

        # https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
        image_center = (w / 2, h / 2)

        # Rotation Matrix
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        new_w = math.ceil(h * abs_sin + w * abs_cos)
        new_h = math.ceil(h * abs_cos + w * abs_sin)

        scale_x = w / new_w
        scale_y = h / new_h

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

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str, str]:
        return ("limit", "interpolation", "border_mode", "value", "mask_value")
