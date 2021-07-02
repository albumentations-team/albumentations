import cv2
import random
import numpy as np

from . import functional as F
from ...core.transforms_interface import DualTransform, to_tuple

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
        always_apply=False,
        p=0.5,
    ):
        super(Rotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.rotate(img, angle, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, angle=0, **params):
        return F.rotate(img, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def get_params(self):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}

    def apply_to_bbox(self, bbox, angle=0, **params):
        return F.bbox_rotate(bbox, angle, params["rows"], params["cols"])

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        return F.keypoint_rotate(keypoint, angle, **params)

    def get_transform_init_args_names(self):
        return ("limit", "interpolation", "border_mode", "value", "mask_value")


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
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(SafeRotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.safe_rotate(
            img=img, value=self.value, angle=angle, interpolation=interpolation, border_mode=self.border_mode
        )

    def apply_to_mask(self, img, angle=0, **params):
        return F.safe_rotate(
            img=img, value=self.mask_value, angle=angle, interpolation=cv2.INTER_NEAREST, border_mode=self.border_mode
        )

    def get_params(self):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}

    def apply_to_bbox(self, bbox, angle=0, **params):
        return F.bbox_safe_rotate(bbox=bbox, angle=angle, rows=params["rows"], cols=params["cols"])

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        return F.keypoint_safe_rotate(keypoint, angle=angle, rows=params["rows"], cols=params["cols"])

    def get_transform_init_args_names(self):
        return ("limit", "interpolation", "border_mode", "value", "mask_value")
