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

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method.
            cv2.BORDER_CONSTANT is the only supported flag at the moment
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
        return self.__process_image__(img, self.value, angle, interpolation, **params)

    def apply_to_mask(self, img, angle=0, **params):
        mask = self.__process_image__(img, self.mask_value, angle=angle, **params)
        return mask

    def get_params(self):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}

    def apply_to_bbox(self, bbox, angle=0, **params):
        old_rows = params["rows"]
        old_cols = params["cols"]

        # Rows and columns of the rotated image (not cropped)
        new_rows, new_cols = F.rotated_img_size(angle=angle, rows=old_rows, cols=old_cols)

        col_diff = np.ceil(abs(new_cols - old_cols) / 2)
        row_diff = np.ceil(abs(new_rows - old_rows) / 2)

        # Normalize shifts
        norm_col_shift = col_diff / new_cols
        norm_row_shift = row_diff / new_rows

        # shift bbox
        shifted_bbox = bbox
        shifted_bbox[0] += norm_col_shift
        shifted_bbox[2] += norm_col_shift
        shifted_bbox[1] += norm_row_shift
        shifted_bbox[3] += norm_row_shift

        rotated_bbox = F.bbox_rotate(shifted_bbox, angle, new_rows, new_cols)

        # Bounding boxes are scale invariant, so this does not need to be rescaled to the old size
        return rotated_bbox

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        old_rows = params["rows"]
        old_cols = params["cols"]

        # Rows and columns of the rotated image (not cropped)
        new_rows, new_cols = F.rotated_img_size(angle=angle, rows=old_rows, cols=old_cols)

        col_diff = int(np.ceil(abs(new_cols - old_cols) / 2))
        row_diff = int(np.ceil(abs(new_rows - old_rows) / 2))

        # Shift keypoint
        shifted_keypoint = keypoint
        shifted_keypoint[0] += col_diff
        shifted_keypoint[1] += row_diff

        # Rotate keypoint
        rotated_keypoint = F.keypoint_rotate(shifted_keypoint, angle, **params)

        # Scale the keypoint
        return F.keypoint_scale(rotated_keypoint, old_cols / new_cols, old_rows / new_rows)

    def get_transform_init_args_names(self):
        return ("limit", "interpolation", "border_mode", "value", "mask_value")

    def __process_image__(self, img, value, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        old_rows = params["rows"]
        old_cols = params["cols"]
        old_shape = img.shape

        # Rows and columns of the rotated image (not cropped)
        new_rows, new_cols = F.rotated_img_size(angle=angle, rows=old_rows, cols=old_cols)

        col_diff = int(np.ceil(abs(new_cols - old_cols)))
        row_diff = int(np.ceil(abs(new_rows - old_rows)))

        # Pad the original image to make it the expected size
        padded_img = cv2.copyMakeBorder(
            src=img,
            top=row_diff,
            bottom=row_diff,
            left=col_diff,
            right=col_diff,
            borderType=self.border_mode,
            value=value,
        )

        # Rotate image
        rotated_img = F.rotate(padded_img, angle, interpolation, self.border_mode, value)

        # Resize image back to the original size
        resized_img = F.resize(img=rotated_img, height=old_rows, width=old_cols, interpolation=interpolation)

        # Greyscale channel check
        if len(old_shape) == 3:
            # Assume the channel is the third index
            channel = old_shape[2]
            if channel == 1 and len(resized_img.shape) != len(old_shape):
                # Add the greyscale channel back
                resized_img = resized_img[:, :, np.newaxis]

        return resized_img
