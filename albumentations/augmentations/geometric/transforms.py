import math
import random
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import skimage.transform

from albumentations.core.bbox_utils import (
    denormalize_bbox,
    normalize_bbox,
    to_ndarray_bboxes,
    to_tuple_bboxes,
)

from ... import random_utils
from ...core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    ImageColorType,
    KeypointInternalType,
    ScaleFloatType,
    to_tuple,
)
from ..functional import bbox_from_mask
from . import functional as F

__all__ = [
    "ShiftScaleRotate",
    "ElasticTransform",
    "Perspective",
    "Affine",
    "PiecewiseAffine",
    "VerticalFlip",
    "HorizontalFlip",
    "Flip",
    "Transpose",
    "OpticalDistortion",
    "GridDistortion",
    "PadIfNeeded",
]


class ShiftScaleRotate(DualTransform):
    """Randomly apply affine transforms: translate, scale and rotate the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        shift_limit_x ((float, float) or float): shift factor range for width. If it is set then this value
            instead of shift_limit will be used for shifting width.  If shift_limit_x is a single float value,
            the range will be (-shift_limit_x, shift_limit_x). Absolute values for lower and upper bounds should lie in
            the range [0, 1]. Default: None.
        shift_limit_y ((float, float) or float): shift factor range for height. If it is set then this value
            instead of shift_limit will be used for shifting height.  If shift_limit_y is a single float value,
            the range will be (-shift_limit_y, shift_limit_y). Absolute values for lower and upper bounds should lie
            in the range [0, 1]. Default: None.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        reflec_annotation: If True and the border_mode is one of: cv2.BORDER_REFLECT, cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101, then annotations(keypoints are not supported yet) are also reflected or wrapped
            in the same way as the image.
            Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        shift_limit_x=None,
        shift_limit_y=None,
        rotate_method="largest_box",
        reflect_annotation=False,
        always_apply=False,
        p=0.5,
    ):
        super(ShiftScaleRotate, self).__init__(always_apply, p)
        self.shift_limit_x = to_tuple(shift_limit_x if shift_limit_x is not None else shift_limit)
        self.shift_limit_y = to_tuple(shift_limit_y if shift_limit_y is not None else shift_limit)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.rotate_limit = to_tuple(rotate_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method
        self.reflect_annotation = reflect_annotation

        if self.rotate_method not in ["largest_box", "ellipse"]:
            raise ValueError(f"Rotation method {self.rotate_method} is not valid.")

    def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def apply_to_keypoint(self, keypoint, angle=0, scale=0, dx=0, dy=0, rows=0, cols=0, **params):
        return F.keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols)

    def get_params(self):
        return {
            "angle": random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
            "scale": random.uniform(self.scale_limit[0], self.scale_limit[1]),
            "dx": random.uniform(self.shift_limit_x[0], self.shift_limit_x[1]),
            "dy": random.uniform(self.shift_limit_y[0], self.shift_limit_y[1]),
        }

    def apply_to_bboxes(self, bboxes, angle, scale, dx, dy, rows, cols, **params):
        if not self.reflect_annotation:
            bboxes = super().apply_to_bboxes(
                bboxes, angle=angle, scale=scale, dx=dx, dy=dy, rows=rows, cols=cols, **params
            )
        else:
            bboxes, labels = to_ndarray_bboxes(bboxes)
            bboxes = F.bboxes_shift_scale_rotate_reflect(
                bboxes, angle, scale, dx, dy, self.rotate_method, rows, cols, self.border_mode
            )
            bboxes = to_tuple_bboxes(bboxes, labels)
        return bboxes

    def apply_to_bbox(self, bbox, angle, scale, dx, dy, **params):
        return F.bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, self.rotate_method, **params)

    def get_transform_init_args(self):
        return {
            "shift_limit_x": self.shift_limit_x,
            "shift_limit_y": self.shift_limit_y,
            "scale_limit": to_tuple(self.scale_limit, bias=-1.0),
            "rotate_limit": self.rotate_limit,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
            "value": self.value,
            "mask_value": self.mask_value,
            "rotate_method": self.rotate_method,
        }


class ElasticTransform(DualTransform):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Args:
        alpha (float):
        sigma (float): Gaussian filter parameter.
        alpha_affine (float): The range will be (-alpha_affine, alpha_affine)
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
        approximate (boolean): Whether to smooth displacement map with fixed kernel size.
                               Enabling this option gives ~2X speedup on large images.
        same_dxdy (boolean): Whether to use same random generated shift for x and y.
                             Enabling this option gives ~2X speedup.

    Targets:
        image, mask, bbox

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        alpha=1,
        sigma=50,
        alpha_affine=50,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        approximate=False,
        same_dxdy=False,
        p=0.5,
    ):
        super(ElasticTransform, self).__init__(always_apply, p)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.approximate = approximate
        self.same_dxdy = same_dxdy

    def apply(self, img, random_state=None, interpolation=cv2.INTER_LINEAR, **params):
        return F.elastic_transform(
            img,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            interpolation,
            self.border_mode,
            self.value,
            np.random.RandomState(random_state),
            self.approximate,
            self.same_dxdy,
        )

    def apply_to_mask(self, img, random_state=None, **params):
        return F.elastic_transform(
            img,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
            np.random.RandomState(random_state),
            self.approximate,
            self.same_dxdy,
        )

    def apply_to_bbox(self, bbox, random_state=None, **params):
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = F.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = F.elastic_transform(
            mask,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
            np.random.RandomState(random_state),
            self.approximate,
        )
        bbox_returned = bbox_from_mask(mask)
        bbox_returned = F.normalize_bbox(bbox_returned, rows, cols)
        return bbox_returned

    def get_params(self):
        return {"random_state": random.randint(0, 10000)}

    def get_transform_init_args_names(self):
        return (
            "alpha",
            "sigma",
            "alpha_affine",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
            "approximate",
            "same_dxdy",
        )


class Perspective(DualTransform):
    """Perform a random four point perspective transform of the input.

    Args:
        scale (float or (float, float)): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners.
            If scale is a single float value, the range will be (0, scale). Default: (0.05, 0.1).
        keep_size (bool): Whether to resize image’s back to their original size after applying the perspective
            transform. If set to False, the resulting images may end up having different shapes
            and will always be a list, never an array. Default: True
        pad_mode (OpenCV flag): OpenCV border mode.
        pad_val (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
            Default: 0
        mask_pad_val (int, float, list of int, list of float): padding value for mask
            if border_mode is cv2.BORDER_CONSTANT. Default: 0
        fit_output (bool): If True, the image plane size and position will be adjusted to still capture
            the whole image after perspective transformation. (Followed by image resizing if keep_size is set to True.)
            Otherwise, parts of the transformed image may be outside of the image plane.
            This setting should not be set to True when using large scale values as it could lead to very large images.
            Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale=(0.05, 0.1),
        keep_size=True,
        pad_mode=cv2.BORDER_CONSTANT,
        pad_val=0,
        mask_pad_val=0,
        fit_output=False,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.scale = to_tuple(scale, 0)
        self.keep_size = keep_size
        self.pad_mode = pad_mode
        self.pad_val = pad_val
        self.mask_pad_val = mask_pad_val
        self.fit_output = fit_output
        self.interpolation = interpolation

    def apply(self, img, matrix=None, max_height=None, max_width=None, **params):
        return F.perspective(
            img, matrix, max_width, max_height, self.pad_val, self.pad_mode, self.keep_size, params["interpolation"]
        )

    def apply_to_bbox(self, bbox, matrix=None, max_height=None, max_width=None, **params):
        return F.perspective_bbox(bbox, params["rows"], params["cols"], matrix, max_width, max_height, self.keep_size)

    def apply_to_keypoint(self, keypoint, matrix=None, max_height=None, max_width=None, **params):
        return F.perspective_keypoint(
            keypoint, params["rows"], params["cols"], matrix, max_width, max_height, self.keep_size
        )

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]

        scale = random_utils.uniform(*self.scale)
        points = random_utils.normal(0, scale, [4, 2])
        points = np.mod(np.abs(points), 1)

        # top left -- no changes needed, just use jitter
        # top right
        points[1, 0] = 1.0 - points[1, 0]  # w = 1.0 - jitter
        # bottom right
        points[2] = 1.0 - points[2]  # w = 1.0 - jitt
        # bottom left
        points[3, 1] = 1.0 - points[3, 1]  # h = 1.0 - jitter

        points[:, 0] *= w
        points[:, 1] *= h

        # Obtain a consistent order of the points and unpack them individually.
        # Warning: don't just do (tl, tr, br, bl) = _order_points(...)
        # here, because the reordered points is used further below.
        points = self._order_points(points)
        tl, tr, br, bl = points

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        min_width = None
        max_width = None
        while min_width is None or min_width < 2:
            width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            max_width = int(max(width_top, width_bottom))
            min_width = int(min(width_top, width_bottom))
            if min_width < 2:
                step_size = (2 - min_width) / 2
                tl[0] -= step_size
                tr[0] += step_size
                bl[0] -= step_size
                br[0] += step_size

        # compute the height of the new image, which will be the maximum distance between the top-right
        # and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        min_height = None
        max_height = None
        while min_height is None or min_height < 2:
            height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = int(max(height_right, height_left))
            min_height = int(min(height_right, height_left))
            if min_height < 2:
                step_size = (2 - min_height) / 2
                tl[1] -= step_size
                tr[1] -= step_size
                bl[1] += step_size
                br[1] += step_size

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left order
        # do not use width-1 or height-1 here, as for e.g. width=3, height=2
        # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
        dst = np.array([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]], dtype=np.float32)

        # compute the perspective transform matrix and then apply it
        m = cv2.getPerspectiveTransform(points, dst)

        if self.fit_output:
            m, max_width, max_height = self._expand_transform(m, (h, w))

        return {"matrix": m, "max_height": max_height, "max_width": max_width, "interpolation": self.interpolation}

    @classmethod
    def _expand_transform(cls, matrix, shape):
        height, width = shape
        # do not use width-1 or height-1 here, as for e.g. width=3, height=2, max_height
        # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
        rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        dst = cv2.perspectiveTransform(np.array([rect]), matrix)[0]

        # get min x, y over transformed 4 points
        # then modify target points by subtracting these minima  => shift to (0, 0)
        dst -= dst.min(axis=0, keepdims=True)
        dst = np.around(dst, decimals=0)

        matrix_expanded = cv2.getPerspectiveTransform(rect, dst)
        max_width, max_height = dst.max(axis=0)
        return matrix_expanded, int(max_width), int(max_height)

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        pts = np.array(sorted(pts, key=lambda x: x[0]))
        left = pts[:2]  # points with smallest x coordinate - left points
        right = pts[2:]  # points with greatest x coordinate - right points

        if left[0][1] < left[1][1]:
            tl, bl = left
        else:
            bl, tl = left

        if right[0][1] < right[1][1]:
            tr, br = right
        else:
            br, tr = right

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def get_transform_init_args_names(self):
        return "scale", "keep_size", "pad_mode", "pad_val", "mask_pad_val", "fit_output", "interpolation"


class Affine(DualTransform):
    """Augmentation to apply affine transformations to images.
    This is mostly a wrapper around the corresponding classes and functions in OpenCV.

    Affine transformations involve:

        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a defined content, e.g.
    if the image is translated to the left, pixels are created on the right.
    A method has to be defined to deal with these pixel values.
    The parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameters `interpolation` and
    `mask_interpolation` deals with the method of interpolation used for this.

    Args:
        scale (number, tuple of number or dict): Scaling factor to use, where ``1.0`` denotes "no change" and
            ``0.5`` is zoomed out to ``50`` percent of the original size.
                * If a single number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                  That the same range will be used for both x- and y-axis. To keep the aspect ratio, set
                  ``keep_ratio=True``, then the same value will be used for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes. Note that when
                  the ``keep_ratio=True``, the x- and y-axis ranges should be the same.
        translate_percent (None, number, tuple of number or dict): Translation as a fraction of the image height/width
            (x-translation, y-translation), where ``0`` denotes "no change"
            and ``0.5`` denotes "half of the axis size".
                * If ``None`` then equivalent to ``0.0`` unless `translate_px` has a value other than ``None``.
                * If a single number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``.
                  That sampled fraction value will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        translate_px (None, int, tuple of int or dict): Translation in pixels.
                * If ``None`` then equivalent to ``0`` unless `translate_percent` has a value other than ``None``.
                * If a single int, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from
                  the discrete interval ``[a..b]``. That number will be used identically for both x- and y-axis.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        rotate (number or tuple of number): Rotation in degrees (**NOT** radians), i.e. expected value range is
            around ``[-360, 360]``. Rotation happens around the *center* of the image,
            not the top left corner as in some other frameworks.
                * If a number, then that value will be used for all images.
                * If a tuple ``(a, b)``, then a value will be uniformly sampled per image from the interval ``[a, b]``
                  and used as the rotation value.
        shear (number, tuple of number or dict): Shear in degrees (**NOT** radians), i.e. expected value range is
            around ``[-360, 360]``, with reasonable values being in the range of ``[-45, 45]``.
                * If a number, then that value will be used for all images as
                  the shear on the x-axis (no shear on the y-axis will be done).
                * If a tuple ``(a, b)``, then two value will be uniformly sampled per image
                  from the interval ``[a, b]`` and be used as the x- and y-shear value.
                * If a dictionary, then it is expected to have the keys ``x`` and/or ``y``.
                  Each of these keys can have the same values as described above.
                  Using a dictionary allows to set different values for the two axis and sampling will then happen
                  *independently* per axis, resulting in samples that differ between the axes.
        interpolation (int): OpenCV interpolation flag.
        mask_interpolation (int): OpenCV interpolation flag.
        cval (number or sequence of number): The constant value to use when filling in newly created pixels.
            (E.g. translating by 1px to the right will create a new 1px-wide column of pixels
            on the left of the image).
            The value is only used when `mode=constant`. The expected value range is ``[0, 255]`` for ``uint8`` images.
        cval_mask (number or tuple of number): Same as cval but only for masks.
        mode (int): OpenCV border flag.
        fit_output (bool): If True, the image plane size and position will be adjusted to tightly capture
            the whole image after affine transformation (`translate_percent` and `translate_px` are ignored).
            Otherwise (``False``),  parts of the transformed image may end up outside the image plane.
            Fitting the output shape can be useful to avoid corners of the image being outside the image plane
            after applying rotations. Default: False
        keep_ratio (bool): When True, the original aspect ratio will be kept when the random scale is applied.
                           Default: False.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        scale: Optional[Union[float, Sequence[float], dict]] = None,
        translate_percent: Optional[Union[float, Sequence[float], dict]] = None,
        translate_px: Optional[Union[int, Sequence[int], dict]] = None,
        rotate: Optional[Union[float, Sequence[float]]] = None,
        shear: Optional[Union[float, Sequence[float], dict]] = None,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        cval: Union[int, float, Sequence[int], Sequence[float]] = 0,
        cval_mask: Union[int, float, Sequence[int], Sequence[float]] = 0,
        mode: int = cv2.BORDER_CONSTANT,
        fit_output: bool = False,
        keep_ratio: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        params = [scale, translate_percent, translate_px, rotate, shear]
        if all([p is None for p in params]):
            scale = {"x": (0.9, 1.1), "y": (0.9, 1.1)}
            translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
            rotate = (-15, 15)
            shear = {"x": (-10, 10), "y": (-10, 10)}
        else:
            scale = scale if scale is not None else 1.0
            rotate = rotate if rotate is not None else 0.0
            shear = shear if shear is not None else 0.0

        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.cval = cval
        self.cval_mask = cval_mask
        self.mode = mode
        self.scale = self._handle_dict_arg(scale, "scale")
        self.translate_percent, self.translate_px = self._handle_translate_arg(translate_px, translate_percent)
        self.rotate = to_tuple(rotate, rotate)
        self.fit_output = fit_output
        self.shear = self._handle_dict_arg(shear, "shear")
        self.keep_ratio = keep_ratio

        if self.keep_ratio and self.scale["x"] != self.scale["y"]:
            raise ValueError(
                "When keep_ratio is True, the x and y scale range should be identical. got {}".format(self.scale)
            )

    def get_transform_init_args_names(self):
        return (
            "interpolation",
            "mask_interpolation",
            "cval",
            "mode",
            "scale",
            "translate_percent",
            "translate_px",
            "rotate",
            "fit_output",
            "shear",
            "cval_mask",
            "keep_ratio",
        )

    @staticmethod
    def _handle_dict_arg(val: Union[float, Sequence[float], dict], name: str, default: float = 1.0):
        if isinstance(val, dict):
            if "x" not in val and "y" not in val:
                raise ValueError(
                    f'Expected {name} dictionary to contain at least key "x" or ' 'key "y". Found neither of them.'
                )
            x = val.get("x", default)
            y = val.get("y", default)
            return {"x": to_tuple(x, x), "y": to_tuple(y, y)}
        return {"x": to_tuple(val, val), "y": to_tuple(val, val)}

    @classmethod
    def _handle_translate_arg(
        cls,
        translate_px: Optional[Union[float, Sequence[float], dict]],
        translate_percent: Optional[Union[float, Sequence[float], dict]],
    ):
        if translate_percent is None and translate_px is None:
            translate_px = 0

        if translate_percent is not None and translate_px is not None:
            raise ValueError(
                "Expected either translate_percent or translate_px to be " "provided, " "but neither of them was."
            )

        if translate_percent is not None:
            # translate by percent
            return cls._handle_dict_arg(translate_percent, "translate_percent", default=0.0), translate_px

        if translate_px is None:
            raise ValueError("translate_px is None.")
        # translate by pixels
        return translate_percent, cls._handle_dict_arg(translate_px, "translate_px")

    def apply(
        self,
        img: np.ndarray,
        matrix: skimage.transform.ProjectiveTransform = None,
        output_shape: Sequence[int] = (),
        **params
    ) -> np.ndarray:
        return F.warp_affine(
            img,
            matrix,
            interpolation=self.interpolation,
            cval=self.cval,
            mode=self.mode,
            output_shape=output_shape,
        )

    def apply_to_mask(
        self,
        img: np.ndarray,
        matrix: skimage.transform.ProjectiveTransform = None,
        output_shape: Sequence[int] = (),
        **params
    ) -> np.ndarray:
        return F.warp_affine(
            img,
            matrix,
            interpolation=self.mask_interpolation,
            cval=self.cval_mask,
            mode=self.mode,
            output_shape=output_shape,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        matrix: skimage.transform.ProjectiveTransform = None,
        rows: int = 0,
        cols: int = 0,
        output_shape: Sequence[int] = (),
        **params
    ) -> BoxInternalType:
        return F.bbox_affine(bbox, matrix, rows, cols, output_shape)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        matrix: skimage.transform.ProjectiveTransform = None,
        scale: dict = None,
        **params
    ) -> KeypointInternalType:
        assert scale is not None and matrix is not None
        return F.keypoint_affine(keypoint, matrix=matrix, scale=scale)

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params: dict) -> dict:
        h, w = params["image"].shape[:2]

        translate: Dict[str, Union[int, float]]
        if self.translate_px is not None:
            translate = {key: random.randint(*value) for key, value in self.translate_px.items()}
        elif self.translate_percent is not None:
            translate = {key: random.uniform(*value) for key, value in self.translate_percent.items()}
            translate["x"] = translate["x"] * w
            translate["y"] = translate["y"] * h
        else:
            translate = {"x": 0, "y": 0}

        # Look to issue https://github.com/albumentations-team/albumentations/issues/1079
        shear = {key: -random.uniform(*value) for key, value in self.shear.items()}
        scale = {key: random.uniform(*value) for key, value in self.scale.items()}
        if self.keep_ratio:
            scale["y"] = scale["x"]

        # Look to issue https://github.com/albumentations-team/albumentations/issues/1079
        rotate = -random.uniform(*self.rotate)

        # for images we use additional shifts of (0.5, 0.5) as otherwise
        # we get an ugly black border for 90deg rotations
        shift_x = w / 2 - 0.5
        shift_y = h / 2 - 0.5

        matrix_to_topleft = skimage.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        matrix_shear_y_rot = skimage.transform.AffineTransform(rotation=-np.pi / 2)
        matrix_shear_y = skimage.transform.AffineTransform(shear=np.deg2rad(shear["y"]))
        matrix_shear_y_rot_inv = skimage.transform.AffineTransform(rotation=np.pi / 2)
        matrix_transforms = skimage.transform.AffineTransform(
            scale=(scale["x"], scale["y"]),
            translation=(translate["x"], translate["y"]),
            rotation=np.deg2rad(rotate),
            shear=np.deg2rad(shear["x"]),
        )
        matrix_to_center = skimage.transform.SimilarityTransform(translation=[shift_x, shift_y])
        matrix = (
            matrix_to_topleft
            + matrix_shear_y_rot
            + matrix_shear_y
            + matrix_shear_y_rot_inv
            + matrix_transforms
            + matrix_to_center
        )
        if self.fit_output:
            matrix, output_shape = self._compute_affine_warp_output_shape(matrix, params["image"].shape)
        else:
            output_shape = params["image"].shape

        return {
            "rotate": rotate,
            "scale": scale,
            "matrix": matrix,
            "output_shape": output_shape,
        }

    @staticmethod
    def _compute_affine_warp_output_shape(
        matrix: skimage.transform.ProjectiveTransform, input_shape: Sequence[int]
    ) -> Tuple[skimage.transform.ProjectiveTransform, Sequence[int]]:
        height, width = input_shape[:2]

        if height == 0 or width == 0:
            return matrix, input_shape

        # determine shape of output image
        corners = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
        corners = matrix(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_height = maxr - minr + 1
        out_width = maxc - minc + 1
        if len(input_shape) == 3:
            output_shape = np.ceil((out_height, out_width, input_shape[2]))
        else:
            output_shape = np.ceil((out_height, out_width))
        output_shape_tuple = tuple([int(v) for v in output_shape.tolist()])
        # fit output image in new shape
        translation = (-minc, -minr)
        matrix_to_fit = skimage.transform.SimilarityTransform(translation=translation)
        matrix = matrix + matrix_to_fit
        return matrix, output_shape_tuple


class PiecewiseAffine(DualTransform):
    """Apply affine transformations that differ between local neighbourhoods.
    This augmentation places a regular grid of points on an image and randomly moves the neighbourhood of these point
    around via affine transformations. This leads to local distortions.

    This is mostly a wrapper around scikit-image's ``PiecewiseAffine``.
    See also ``Affine`` for a similar technique.

    Note:
        This augmenter is very slow. Try to use ``ElasticTransformation`` instead, which is at least 10x faster.

    Note:
        For coordinate-based inputs (keypoints, bounding boxes, polygons, ...),
        this augmenter still has to perform an image-based augmentation,
        which will make it significantly slower and not fully correct for such inputs than other transforms.

    Args:
        scale (float, tuple of float): Each point on the regular grid is moved around via a normal distribution.
            This scale factor is equivalent to the normal distribution's sigma.
            Note that the jitter (how far each point is moved in which direction) is multiplied by the height/width of
            the image if ``absolute_scale=False`` (default), so this scale can be the same for different sized images.
            Recommended values are in the range ``0.01`` to ``0.05`` (weak to strong augmentations).
                * If a single ``float``, then that value will always be used as the scale.
                * If a tuple ``(a, b)`` of ``float`` s, then a random value will
                  be uniformly sampled per image from the interval ``[a, b]``.
        nb_rows (int, tuple of int): Number of rows of points that the regular grid should have.
            Must be at least ``2``. For large images, you might want to pick a higher value than ``4``.
            You might have to then adjust scale to lower values.
                * If a single ``int``, then that value will always be used as the number of rows.
                * If a tuple ``(a, b)``, then a value from the discrete interval
                  ``[a..b]`` will be uniformly sampled per image.
        nb_cols (int, tuple of int): Number of columns. Analogous to `nb_rows`.
        interpolation (int): The order of interpolation. The order has to be in the range 0-5:
             - 0: Nearest-neighbor
             - 1: Bi-linear (default)
             - 2: Bi-quadratic
             - 3: Bi-cubic
             - 4: Bi-quartic
             - 5: Bi-quintic
        mask_interpolation (int): same as interpolation but for mask.
        cval (number): The constant value to use when filling in newly created pixels.
        cval_mask (number): Same as cval but only for masks.
        mode (str): {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
            Points outside the boundaries of the input are filled according
            to the given mode.  Modes match the behaviour of `numpy.pad`.
        absolute_scale (bool): Take `scale` as an absolute value rather than a relative value.
        keypoints_threshold (float): Used as threshold in conversion from distance maps to keypoints.
            The search for keypoints works by searching for the
            argmin (non-inverted) or argmax (inverted) in each channel. This
            parameters contains the maximum (non-inverted) or minimum (inverted) value to accept in order to view a hit
            as a keypoint. Use ``None`` to use no min/max. Default: 0.01

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        scale: ScaleFloatType = (0.03, 0.05),
        nb_rows: Union[int, Sequence[int]] = 4,
        nb_cols: Union[int, Sequence[int]] = 4,
        interpolation: int = 1,
        mask_interpolation: int = 0,
        cval: int = 0,
        cval_mask: int = 0,
        mode: str = "constant",
        absolute_scale: bool = False,
        always_apply: bool = False,
        keypoints_threshold: float = 0.01,
        p: float = 0.5,
    ):
        super(PiecewiseAffine, self).__init__(always_apply, p)

        self.scale = to_tuple(scale, scale)
        self.nb_rows = to_tuple(nb_rows, nb_rows)
        self.nb_cols = to_tuple(nb_cols, nb_cols)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.cval = cval
        self.cval_mask = cval_mask
        self.mode = mode
        self.absolute_scale = absolute_scale
        self.keypoints_threshold = keypoints_threshold

    def get_transform_init_args_names(self):
        return (
            "scale",
            "nb_rows",
            "nb_cols",
            "interpolation",
            "mask_interpolation",
            "cval",
            "cval_mask",
            "mode",
            "absolute_scale",
            "keypoints_threshold",
        )

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params) -> dict:
        h, w = params["image"].shape[:2]

        nb_rows = np.clip(random.randint(*self.nb_rows), 2, None)
        nb_cols = np.clip(random.randint(*self.nb_cols), 2, None)
        nb_cells = nb_cols * nb_rows
        scale = random.uniform(*self.scale)

        jitter: np.ndarray = random_utils.normal(0, scale, (nb_cells, 2))
        if not np.any(jitter > 0):
            return {"matrix": None}

        y = np.linspace(0, h, nb_rows)
        x = np.linspace(0, w, nb_cols)

        # (H, W) and (H, W) for H=rows, W=cols
        xx_src, yy_src = np.meshgrid(x, y)

        # (1, HW, 2) => (HW, 2) for H=rows, W=cols
        points_src = np.dstack([yy_src.flat, xx_src.flat])[0]

        if self.absolute_scale:
            jitter[:, 0] = jitter[:, 0] / h if h > 0 else 0.0
            jitter[:, 1] = jitter[:, 1] / w if w > 0 else 0.0

        jitter[:, 0] = jitter[:, 0] * h
        jitter[:, 1] = jitter[:, 1] * w

        points_dest = np.copy(points_src)
        points_dest[:, 0] = points_dest[:, 0] + jitter[:, 0]
        points_dest[:, 1] = points_dest[:, 1] + jitter[:, 1]

        # Restrict all destination points to be inside the image plane.
        # This is necessary, as otherwise keypoints could be augmented
        # outside of the image plane and these would be replaced by
        # (-1, -1), which would not conform with the behaviour of the other augmenters.
        points_dest[:, 0] = np.clip(points_dest[:, 0], 0, h - 1)
        points_dest[:, 1] = np.clip(points_dest[:, 1], 0, w - 1)

        matrix = skimage.transform.PiecewiseAffineTransform()
        matrix.estimate(points_src[:, ::-1], points_dest[:, ::-1])

        return {
            "matrix": matrix,
        }

    def apply(self, img: np.ndarray, matrix: skimage.transform.PiecewiseAffineTransform = None, **params) -> np.ndarray:
        return F.piecewise_affine(img, matrix, self.interpolation, self.mode, self.cval)

    def apply_to_mask(
        self, img: np.ndarray, matrix: skimage.transform.PiecewiseAffineTransform = None, **params
    ) -> np.ndarray:
        return F.piecewise_affine(img, matrix, self.mask_interpolation, self.mode, self.cval_mask)

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        rows: int = 0,
        cols: int = 0,
        matrix: skimage.transform.PiecewiseAffineTransform = None,
        **params
    ) -> BoxInternalType:
        return F.bbox_piecewise_affine(bbox, matrix, rows, cols, self.keypoints_threshold)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        rows: int = 0,
        cols: int = 0,
        matrix: skimage.transform.PiecewiseAffineTransform = None,
        **params
    ):
        return F.keypoint_piecewise_affine(keypoint, matrix, rows, cols, self.keypoints_threshold)


class PadIfNeeded(DualTransform):
    """Pad side of the image / max if side is less than desired number.

    Args:
        min_height (int): minimal result image height.
        min_width (int): minimal result image width.
        pad_height_divisor (int): if not None, ensures image height is dividable by value of this argument.
        pad_width_divisor (int): if not None, ensures image width is dividable by value of this argument.
        position (Union[str, PositionType]): Position of the image. should be PositionType.CENTER or
            PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT.
            or PositionType.RANDOM. Default: PositionType.CENTER.
        border_mode (OpenCV flag): OpenCV border mode.
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32
    """

    class PositionType(Enum):
        CENTER = "center"
        TOP_LEFT = "top_left"
        TOP_RIGHT = "top_right"
        BOTTOM_LEFT = "bottom_left"
        BOTTOM_RIGHT = "bottom_right"
        RANDOM = "random"

    def __init__(
        self,
        min_height: Optional[int] = 1024,
        min_width: Optional[int] = 1024,
        pad_height_divisor: Optional[int] = None,
        pad_width_divisor: Optional[int] = None,
        position: Union[PositionType, str] = PositionType.CENTER,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: Optional[ImageColorType] = None,
        mask_value: Optional[ImageColorType] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        if (min_height is None) == (pad_height_divisor is None):
            raise ValueError("Only one of 'min_height' and 'pad_height_divisor' parameters must be set")

        if (min_width is None) == (pad_width_divisor is None):
            raise ValueError("Only one of 'min_width' and 'pad_width_divisor' parameters must be set")

        super(PadIfNeeded, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.pad_width_divisor = pad_width_divisor
        self.pad_height_divisor = pad_height_divisor
        self.position = PadIfNeeded.PositionType(position)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if self.min_height is not None:
            if rows < self.min_height:
                h_pad_top = int((self.min_height - rows) / 2.0)
                h_pad_bottom = self.min_height - rows - h_pad_top
            else:
                h_pad_top = 0
                h_pad_bottom = 0
        else:
            pad_remained = rows % self.pad_height_divisor
            pad_rows = self.pad_height_divisor - pad_remained if pad_remained > 0 else 0

            h_pad_top = pad_rows // 2
            h_pad_bottom = pad_rows - h_pad_top

        if self.min_width is not None:
            if cols < self.min_width:
                w_pad_left = int((self.min_width - cols) / 2.0)
                w_pad_right = self.min_width - cols - w_pad_left
            else:
                w_pad_left = 0
                w_pad_right = 0
        else:
            pad_remainder = cols % self.pad_width_divisor
            pad_cols = self.pad_width_divisor - pad_remainder if pad_remainder > 0 else 0

            w_pad_left = pad_cols // 2
            w_pad_right = pad_cols - w_pad_left

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = self.__update_position_params(
            h_top=h_pad_top, h_bottom=h_pad_bottom, w_left=w_pad_left, w_right=w_pad_right
        )

        params.update(
            {
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
            }
        )
        return params

    def apply(
        self, img: np.ndarray, pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0, **params
    ) -> np.ndarray:
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(
        self, img: np.ndarray, pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0, **params
    ) -> np.ndarray:
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.mask_value,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        rows: int = 0,
        cols: int = 0,
        **params
    ) -> BoxInternalType:
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        **params
    ) -> KeypointInternalType:
        x, y, angle, scale = keypoint[:4]
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return (
            "min_height",
            "min_width",
            "pad_height_divisor",
            "pad_width_divisor",
            "border_mode",
            "value",
            "mask_value",
        )

    def __update_position_params(
        self, h_top: int, h_bottom: int, w_left: int, w_right: int
    ) -> Tuple[int, int, int, int]:
        if self.position == PadIfNeeded.PositionType.TOP_LEFT:
            h_bottom += h_top
            w_right += w_left
            h_top = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.TOP_RIGHT:
            h_bottom += h_top
            w_left += w_right
            h_top = 0
            w_right = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_LEFT:
            h_top += h_bottom
            w_right += w_left
            h_bottom = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_RIGHT:
            h_top += h_bottom
            w_left += w_right
            h_bottom = 0
            w_right = 0

        elif self.position == PadIfNeeded.PositionType.RANDOM:
            h_pad = h_top + h_bottom
            w_pad = w_left + w_right
            h_top = random.randint(0, h_pad)
            h_bottom = h_pad - h_top
            w_left = random.randint(0, w_pad)
            w_right = w_pad - w_left

        return h_top, h_bottom, w_left, w_right


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.vflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_vflip(bbox, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_vflip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return F.hflip_cv2(img)

        return F.hflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_hflip(bbox, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_hflip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class Flip(DualTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, d: int = 0, **params) -> np.ndarray:
        """Args:
        d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        return F.random_flip(img, d)

    def get_params(self):
        # Random int in the range [-1, 1]
        return {"d": random.randint(-1, 1)}

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_flip(bbox, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_flip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class Transpose(DualTransform):
    """Transpose the input by swapping rows and columns.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.transpose(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_transpose(bbox, 0, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_transpose(keypoint)

    def get_transform_init_args_names(self):
        return ()


class OpticalDistortion(DualTransform):
    """
    Args:
        distort_limit (float, (float, float)): If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: (-0.05, 0.05).
        shift_limit (float, (float, float))): If shift_limit is a single float, the range
            will be (-shift_limit, shift_limit). Default: (-0.05, 0.05).
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

    Targets:
        image, mask, bbox

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        distort_limit: ScaleFloatType = 0.05,
        shift_limit: ScaleFloatType = 0.05,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: Optional[ImageColorType] = None,
        mask_value: Optional[ImageColorType] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(OpticalDistortion, self).__init__(always_apply, p)
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(
        self, img: np.ndarray, k: int = 0, dx: int = 0, dy: int = 0, interpolation: int = cv2.INTER_LINEAR, **params
    ) -> np.ndarray:
        return F.optical_distortion(img, k, dx, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img: np.ndarray, k: int = 0, dx: int = 0, dy: int = 0, **params) -> np.ndarray:
        return F.optical_distortion(img, k, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def apply_to_bbox(self, bbox: BoxInternalType, k: int = 0, dx: int = 0, dy: int = 0, **params) -> BoxInternalType:
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = F.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = F.optical_distortion(mask, k, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)
        bbox_returned = bbox_from_mask(mask)
        bbox_returned = F.normalize_bbox(bbox_returned, rows, cols)
        return bbox_returned

    def get_params(self):
        return {
            "k": random.uniform(self.distort_limit[0], self.distort_limit[1]),
            "dx": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
            "dy": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
        }

    def get_transform_init_args_names(self):
        return (
            "distort_limit",
            "shift_limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
        )


class GridDistortion(DualTransform):
    """
    Args:
        num_steps (int): count of grid cells on each side.
        distort_limit (float, (float, float)): If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: (-0.03, 0.03).
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
        normalized (bool): if true, distortion will be normalized to do not go outside the image. Default: False
            See for more information: https://github.com/albumentations-team/albumentations/pull/722

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        num_steps: int = 5,
        distort_limit: ScaleFloatType = 0.3,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: Optional[ImageColorType] = None,
        mask_value: Optional[ImageColorType] = None,
        normalized: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(GridDistortion, self).__init__(always_apply, p)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.normalized = normalized

    def apply(
        self, img: np.ndarray, stepsx: Tuple = (), stepsy: Tuple = (), interpolation: int = cv2.INTER_LINEAR, **params
    ) -> np.ndarray:
        return F.grid_distortion(img, self.num_steps, stepsx, stepsy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img: np.ndarray, stepsx: Tuple = (), stepsy: Tuple = (), **params) -> np.ndarray:
        return F.grid_distortion(
            img, self.num_steps, stepsx, stepsy, cv2.INTER_NEAREST, self.border_mode, self.mask_value
        )

    def apply_to_bbox(self, bbox: BoxInternalType, stepsx: Tuple = (), stepsy: Tuple = (), **params) -> BoxInternalType:
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = F.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = F.grid_distortion(
            mask, self.num_steps, stepsx, stepsy, cv2.INTER_NEAREST, self.border_mode, self.mask_value
        )
        bbox_returned = bbox_from_mask(mask)
        bbox_returned = F.normalize_bbox(bbox_returned, rows, cols)
        return bbox_returned

    def _normalize(self, h, w, xsteps, ysteps):

        # compensate for smaller last steps in source image.
        x_step = w // self.num_steps
        last_x_step = min(w, ((self.num_steps + 1) * x_step)) - (self.num_steps * x_step)
        xsteps[-1] *= last_x_step / x_step

        y_step = h // self.num_steps
        last_y_step = min(h, ((self.num_steps + 1) * y_step)) - (self.num_steps * y_step)
        ysteps[-1] *= last_y_step / y_step

        # now normalize such that distortion never leaves image bounds.
        tx = w / math.floor(w / self.num_steps)
        ty = h / math.floor(h / self.num_steps)
        xsteps = np.array(xsteps) * (tx / np.sum(xsteps))
        ysteps = np.array(ysteps) * (ty / np.sum(ysteps))

        return {"stepsx": xsteps, "stepsy": ysteps}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]

        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for _ in range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for _ in range(self.num_steps + 1)]

        if self.normalized:
            return self._normalize(h, w, stepsx, stepsy)

        return {"stepsx": stepsx, "stepsy": stepsy}

    def get_transform_init_args_names(self):
        return "num_steps", "distort_limit", "interpolation", "border_mode", "value", "mask_value", "normalized"
