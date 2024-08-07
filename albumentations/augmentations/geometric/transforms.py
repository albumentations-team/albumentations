from __future__ import annotations

import math
import random
from enum import Enum
from typing import Any, Callable, Literal, Tuple, cast, List, Dict
from warnings import warn

import cv2
import numpy as np
import skimage.transform
from PIL import Image
from albucore.utils import get_num_channels
from pydantic import Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Annotated, Self

from albumentations import random_utils
from albumentations.augmentations.functional import bbox_from_mask, center, center_bbox
from albumentations.augmentations.utils import check_range
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import (
    BIG_INTEGER,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    TWO,
    BoxInternalType,
    ColorType,
    D4Type,
    KeypointInternalType,
    ScalarType,
    ScaleFloatType,
    ScaleIntType,
    SizeType,
    Targets,
    d4_group_elements,
)
from albumentations.core.utils import to_tuple

from . import functional as fgeometric


from albumentations.core.pydantic import (
    BorderModeType,
    InterpolationType,
    NonNegativeFloatRangeType,
    ProbabilityType,
    SymmetricRangeType,
)

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
    "D4",
]


class ElasticTransform(DualTransform):
    """Apply elastic deformation to images, masks, and bounding boxes as described in [Simard2003]_.

    This transformation introduces random elastic distortions to images, which can be useful for data augmentation
    in training convolutional neural networks. The transformation can be applied in an approximate or precise manner,
    with an option to use the same displacement field for both x and y directions to speed up the process.

    Args:
        alpha (float): Scaling factor for the random displacement fields.
        sigma (float): Standard deviation for Gaussian filter applied to the displacement fields.
        interpolation (int): Interpolation method to be used. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default is cv2.INTER_LINEAR.
        border_mode (int): Pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default is cv2.BORDER_REFLECT_101.
        value (int, float, list of int, list of float, optional): Padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float, list of int, list of float, optional): Padding value if border_mode is
            cv2.BORDER_CONSTANT, applied to masks.
        approximate (bool, optional): Whether to smooth displacement map with a fixed kernel size.
            Enabling this option gives ~2X speedup on large images. Default is False.
        same_dxdy (bool, optional): Whether to use the same random displacement for x and y directions.
            Enabling this option gives ~2X speedup. Default is False.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32

    Reference:
        Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural Networks applied to
        Visual Document Analysis", in Proc. of the International Conference on Document Analysis and Recognition, 2003.
        https://gist.github.com/ernestum/601cdf56d2b424757de5
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        alpha: Annotated[float, Field(description="Alpha parameter.", ge=0)]
        sigma: Annotated[float, Field(default=50, description="Sigma parameter for Gaussian filter.", ge=1)]
        alpha_affine: None = Field(
            description="Alpha affine parameter.",
            deprecated="Use Affine transform to get affine effects",
        )
        interpolation: InterpolationType = cv2.INTER_LINEAR
        border_mode: BorderModeType = cv2.BORDER_REFLECT_101
        value: int | float | list[int] | list[float] | None = Field(
            default=None,
            description="Padding value if border_mode is cv2.BORDER_CONSTANT.",
        )
        mask_value: float | list[int] | list[float] | None = Field(
            default=None,
            description="Padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.",
        )
        approximate: Annotated[bool, Field(default=False, description="Approximate displacement map smoothing.")]
        same_dxdy: Annotated[bool, Field(default=False, description="Use same shift for x and y.")]

    def __init__(
        self,
        alpha: float = 3,
        sigma: float = 50,
        alpha_affine: None = None,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ScalarType | list[ScalarType] | None = None,
        mask_value: ScalarType | list[ScalarType] | None = None,
        always_apply: bool | None = None,
        approximate: bool = False,
        same_dxdy: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.approximate = approximate
        self.same_dxdy = same_dxdy

    def apply(
        self,
        img: np.ndarray,
        random_seed: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.elastic_transform(
            img,
            self.alpha,
            self.sigma,
            interpolation,
            self.border_mode,
            self.value,
            np.random.RandomState(random_seed),
            self.approximate,
            self.same_dxdy,
        )

    def apply_to_mask(self, mask: np.ndarray, random_seed: int, **params: Any) -> np.ndarray:
        return fgeometric.elastic_transform(
            mask,
            self.alpha,
            self.sigma,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
            np.random.RandomState(random_seed),
            self.approximate,
            self.same_dxdy,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        random_seed: int,
        **params: Any,
    ) -> BoxInternalType:
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = fgeometric.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = fgeometric.elastic_transform(
            mask,
            self.alpha,
            self.sigma,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
            np.random.RandomState(random_seed),
            self.approximate,
        )
        bbox_returned = bbox_from_mask(mask)
        return cast(BoxInternalType, fgeometric.normalize_bbox(bbox_returned, rows, cols))

    def get_params(self) -> dict[str, int]:
        return {"random_seed": random_utils.get_random_seed()}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "alpha",
            "sigma",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
            "approximate",
            "same_dxdy",
        )

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
        }


class Perspective(DualTransform):
    """Perform a random four point perspective transform of the input.

    Args:
        scale: standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners.
            If scale is a single float value, the range will be (0, scale). Default: (0.05, 0.1).
        keep_size: Whether to resize image back to their original size after applying the perspective
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

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        scale: NonNegativeFloatRangeType = (0.05, 0.1)
        keep_size: Annotated[bool, Field(default=True, description="Keep size after transform.")]
        pad_mode: BorderModeType = cv2.BORDER_CONSTANT
        pad_val: ColorType | None = Field(
            default=0,
            description="Padding value if border_mode is cv2.BORDER_CONSTANT.",
        )
        mask_pad_val: ColorType | None = Field(
            default=0,
            description="Mask padding value if border_mode is cv2.BORDER_CONSTANT.",
        )
        fit_output: Annotated[bool, Field(default=False, description="Adjust image plane to capture whole image.")]
        interpolation: InterpolationType = cv2.INTER_LINEAR

    def __init__(
        self,
        scale: ScaleFloatType = (0.05, 0.1),
        keep_size: bool = True,
        pad_mode: int = cv2.BORDER_CONSTANT,
        pad_val: ColorType = 0,
        mask_pad_val: ColorType = 0,
        fit_output: bool = False,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)
        self.scale = cast(Tuple[float, float], scale)
        self.keep_size = keep_size
        self.pad_mode = pad_mode
        self.pad_val = pad_val
        self.mask_pad_val = mask_pad_val
        self.fit_output = fit_output
        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        matrix: np.ndarray,
        max_height: int,
        max_width: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.perspective(
            img,
            matrix,
            max_width,
            max_height,
            self.pad_val,
            self.pad_mode,
            self.keep_size,
            params["interpolation"],
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        matrix: np.ndarray,
        max_height: int,
        max_width: int,
        **params: Any,
    ) -> BoxInternalType:
        return fgeometric.perspective_bbox(
            bbox,
            params["rows"],
            params["cols"],
            matrix,
            max_width,
            max_height,
            self.keep_size,
        )

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        matrix: np.ndarray,
        max_height: int,
        max_width: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.perspective_keypoint(
            keypoint,
            params["rows"],
            params["cols"],
            matrix,
            max_width,
            max_height,
            self.keep_size,
        )

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]

        scale = random.uniform(*self.scale)
        points = random_utils.normal(0, scale, [4, 2])
        points = np.mod(np.abs(points), 0.32)

        # top left -- no changes needed, just use jitter
        # top right
        points[1, 0] = 1.0 - points[1, 0]  # w = 1.0 - jitter
        # bottom right
        points[2] = 1.0 - points[2]  # w = 1.0 - jitt
        # bottom left
        points[3, 1] = 1.0 - points[3, 1]  # h = 1.0 - jitter

        points[:, 0] *= width
        points[:, 1] *= height

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
        while min_width is None or min_width < TWO:
            width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            max_width = int(max(width_top, width_bottom))
            min_width = int(min(width_top, width_bottom))
            if min_width < TWO:
                step_size = (2 - min_width) / 2
                tl[0] -= step_size
                tr[0] += step_size
                bl[0] -= step_size
                br[0] += step_size

        # compute the height of the new image, which will be the maximum distance between the top-right
        # and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        min_height = None
        max_height = None
        while min_height is None or min_height < TWO:
            height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = int(max(height_right, height_left))
            min_height = int(min(height_right, height_left))
            if min_height < TWO:
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
            m, max_width, max_height = self._expand_transform(m, (height, width))

        return {"matrix": m, "max_height": max_height, "max_width": max_width, "interpolation": self.interpolation}

    @classmethod
    def _expand_transform(cls, matrix: np.ndarray, shape: SizeType) -> tuple[np.ndarray, int, int]:
        height, width = shape[:2]
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

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "scale", "keep_size", "pad_mode", "pad_val", "mask_pad_val", "fit_output", "interpolation"


class Affine(DualTransform):
    """Augmentation to apply affine transformations to images.

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
        rotate_method (Literal["largest_box", "ellipse"]): rotation method used for the bounding boxes.
            Should be one of "largest_box" or "ellipse"[1]. Default: "largest_box"
        balanced_scale (bool): When True, scaling factors are chosen to be either entirely below or above 1,
            ensuring balanced scaling. Default: False.

            This is important because without it, scaling tends to lean towards upscaling. For example, if we want
            the image to zoom in and out by 2x, we may pick an interval [0.5, 2]. Since the interval [0.5, 1] is
            three times smaller than [1, 2], values above 1 are picked three times more often if sampled directly
            from [0.5, 2]. With `balanced_scale`, the  function ensures that half the time, the scaling
            factor is picked from below 1 (zooming out), and the other half from above 1 (zooming in).
            This makes the zooming in and out process more balanced.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    Reference:
        [1] https://arxiv.org/abs/2109.13488

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        scale: ScaleFloatType | dict[str, Any] | None = Field(
            default=None,
            description="Scaling factor or dictionary for independent axis scaling.",
        )
        translate_percent: ScaleFloatType | dict[str, Any] | None = Field(
            default=None,
            description="Translation as a fraction of the image dimension.",
        )
        translate_px: ScaleIntType | dict[str, Any] | None = Field(
            default=None,
            description="Translation in pixels.",
        )
        rotate: ScaleFloatType | None = Field(default=None, description="Rotation angle in degrees.")
        shear: ScaleFloatType | dict[str, Any] | None = Field(
            default=None,
            description="Shear angle in degrees.",
        )
        interpolation: InterpolationType = cv2.INTER_LINEAR
        mask_interpolation: InterpolationType = cv2.INTER_NEAREST

        cval: ColorType = Field(default=0, description="Value used for constant padding.")
        cval_mask: ColorType = Field(default=0, description="Value used for mask constant padding.")
        mode: BorderModeType = cv2.BORDER_CONSTANT
        fit_output: Annotated[bool, Field(default=False, description="Adjust output to capture whole image.")]
        keep_ratio: Annotated[bool, Field(default=False, description="Maintain aspect ratio when scaling.")]
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box"
        balanced_scale: Annotated[bool, Field(default=False, description="Use balanced scaling.")]

    def __init__(
        self,
        scale: ScaleFloatType | dict[str, Any] | None = None,
        translate_percent: ScaleFloatType | dict[str, Any] | None = None,
        translate_px: ScaleIntType | dict[str, Any] | None = None,
        rotate: ScaleFloatType | None = None,
        shear: ScaleFloatType | dict[str, Any] | None = None,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        cval: ColorType = 0,
        cval_mask: ColorType = 0,
        mode: int = cv2.BORDER_CONSTANT,
        fit_output: bool = False,
        keep_ratio: bool = False,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        balanced_scale: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        params = [scale, translate_percent, translate_px, rotate, shear]
        if all(p is None for p in params):
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
        self.rotate_method = rotate_method
        self.balanced_scale = balanced_scale

        if self.keep_ratio and self.scale["x"] != self.scale["y"]:
            raise ValueError(f"When keep_ratio is True, the x and y scale range should be identical. got {self.scale}")

    def get_transform_init_args_names(self) -> tuple[str, ...]:
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
            "rotate_method",
            "balanced_scale",
        )

    @staticmethod
    def _handle_dict_arg(
        val: float | tuple[float, float] | dict[str, Any],
        name: str,
        default: float = 1.0,
    ) -> dict[str, Any]:
        if isinstance(val, dict):
            if "x" not in val and "y" not in val:
                raise ValueError(
                    f'Expected {name} dictionary to contain at least key "x" or key "y". Found neither of them.',
                )
            x = val.get("x", default)
            y = val.get("y", default)
            return {"x": to_tuple(x, x), "y": to_tuple(y, y)}
        return {"x": to_tuple(val, val), "y": to_tuple(val, val)}

    @classmethod
    def _handle_translate_arg(
        cls,
        translate_px: ScaleFloatType | dict[str, Any] | None,
        translate_percent: ScaleFloatType | dict[str, Any] | None,
    ) -> Any:
        if translate_percent is None and translate_px is None:
            translate_px = 0

        if translate_percent is not None and translate_px is not None:
            msg = "Expected either translate_percent or translate_px to be provided, but both were provided."
            raise ValueError(msg)

        if translate_percent is not None:
            # translate by percent
            return cls._handle_dict_arg(translate_percent, "translate_percent", default=0.0), translate_px

        if translate_px is None:
            msg = "translate_px is None."
            raise ValueError(msg)
        # translate by pixels
        return translate_percent, cls._handle_dict_arg(translate_px, "translate_px")

    def apply(
        self,
        img: np.ndarray,
        matrix: skimage.transform.ProjectiveTransform,
        output_shape: SizeType,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.warp_affine(
            img,
            matrix,
            interpolation=self.interpolation,
            cval=self.cval,
            mode=self.mode,
            output_shape=output_shape,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        matrix: skimage.transform.ProjectiveTransform,
        output_shape: SizeType,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.warp_affine(
            mask,
            matrix,
            interpolation=self.mask_interpolation,
            cval=self.cval_mask,
            mode=self.mode,
            output_shape=output_shape,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        bbox_matrix: skimage.transform.ProjectiveTransform,
        rows: int,
        cols: int,
        output_shape: SizeType,
        **params: Any,
    ) -> BoxInternalType:
        return fgeometric.bbox_affine(bbox, bbox_matrix, self.rotate_method, rows, cols, output_shape)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        matrix: skimage.transform.ProjectiveTransform,
        scale: dict[str, Any],
        **params: Any,
    ) -> KeypointInternalType:
        if scale is None:
            msg = "Expected scale to be provided, but got None."
            raise ValueError(msg)
        if matrix is None:
            msg = "Expected matrix to be provided, but got None."
            raise ValueError(msg)

        return fgeometric.keypoint_affine(keypoint, matrix=matrix, scale=scale)

    @staticmethod
    def get_scale(scale: dict[str, tuple[float, float]], keep_ratio: bool, balanced_scale: bool) -> dict[str, float]:
        result_scale = {}
        if balanced_scale:
            for key, value in scale.items():
                lower_interval = (value[0], 1.0) if value[0] < 1 else None
                upper_interval = (1.0, value[1]) if value[1] > 1 else None

                if lower_interval is not None and upper_interval is not None:
                    selected_interval = random.choice([lower_interval, upper_interval])
                elif lower_interval is not None:
                    selected_interval = lower_interval
                elif upper_interval is not None:
                    selected_interval = upper_interval
                else:
                    raise ValueError(f"Both lower_interval and upper_interval are None for key: {key}")

                result_scale[key] = random.uniform(*selected_interval)
        else:
            result_scale = {key: random.uniform(*value) for key, value in scale.items()}

        if keep_ratio:
            result_scale["y"] = result_scale["x"]

        return result_scale

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]

        translate: dict[str, int | float]
        if self.translate_px is not None:
            translate = {key: random.randint(*value) for key, value in self.translate_px.items()}
        elif self.translate_percent is not None:
            translate = {key: random.uniform(*value) for key, value in self.translate_percent.items()}
            translate["x"] = translate["x"] * width
            translate["y"] = translate["y"] * height
        else:
            translate = {"x": 0, "y": 0}

        shear = {key: -random.uniform(*value) for key, value in self.shear.items()}

        scale = self.get_scale(self.scale, self.keep_ratio, self.balanced_scale)
        rotate = -random.uniform(*self.rotate)

        shift_x, shift_y = center(width, height)
        shift_x_bbox, shift_y_bbox = center_bbox(width, height)

        # Image transformation matrix
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

        # Bounding box transformation matrix
        matrix_to_topleft_bbox = skimage.transform.SimilarityTransform(translation=[-shift_x_bbox, -shift_y_bbox])
        matrix_to_center_bbox = skimage.transform.SimilarityTransform(translation=[shift_x_bbox, shift_y_bbox])
        bbox_matrix = (
            matrix_to_topleft_bbox
            + matrix_shear_y_rot
            + matrix_shear_y
            + matrix_shear_y_rot_inv
            + matrix_transforms
            + matrix_to_center_bbox
        )

        if self.fit_output:
            matrix, output_shape = self._compute_affine_warp_output_shape(matrix, params["shape"])
        else:
            output_shape = params["shape"]

        return {
            "rotate": rotate,
            "scale": scale,
            "matrix": matrix,
            "bbox_matrix": bbox_matrix,
            "output_shape": output_shape,
        }

    @staticmethod
    def _compute_affine_warp_output_shape(
        matrix: skimage.transform.ProjectiveTransform,
        input_shape: SizeType,
    ) -> tuple[skimage.transform.ProjectiveTransform, SizeType]:
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

        if len(input_shape) == NUM_MULTI_CHANNEL_DIMENSIONS:
            output_shape = np.ceil((out_height, out_width, input_shape[2]))
        else:
            output_shape = np.ceil((out_height, out_width))

        output_shape_tuple = tuple(int(v) for v in output_shape.tolist())
        # fit output image in new shape
        translation = -minc, -minr
        matrix_to_fit = skimage.transform.SimilarityTransform(translation=translation)
        matrix += matrix_to_fit
        return matrix, output_shape_tuple


class ShiftScaleRotate(Affine):
    """Randomly apply affine transforms: translate, scale and rotate the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [-1, 1]. Default: (-0.0625, 0.0625).
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
            the range [-1, 1]. Default: None.
        shift_limit_y ((float, float) or float): shift factor range for height. If it is set then this value
            instead of shift_limit will be used for shifting height.  If shift_limit_y is a single float value,
            the range will be (-shift_limit_y, shift_limit_y). Absolute values for lower and upper bounds should lie
            in the range [-, 1]. Default: None.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        shift_limit: SymmetricRangeType = (-0.0625, 0.0625)
        scale_limit: SymmetricRangeType = (-0.1, 0.1)
        rotate_limit: SymmetricRangeType = (-45, 45)
        interpolation: InterpolationType = cv2.INTER_LINEAR
        border_mode: BorderModeType = cv2.BORDER_REFLECT_101
        value: ColorType = 0
        mask_value: ColorType = 0
        shift_limit_x: ScaleFloatType | None = Field(default=None)
        shift_limit_y: ScaleFloatType | None = Field(default=None)
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box"

        @model_validator(mode="after")
        def check_shift_limit(self) -> Self:
            bounds = -1, 1
            self.shift_limit_x = to_tuple(self.shift_limit_x if self.shift_limit_x is not None else self.shift_limit)
            check_range(self.shift_limit_x, *bounds, "shift_limit_x")
            self.shift_limit_y = to_tuple(self.shift_limit_y if self.shift_limit_y is not None else self.shift_limit)
            check_range(self.shift_limit_y, *bounds, "shift_limit_y")
            return self

        @field_validator("scale_limit")
        @classmethod
        def check_scale_limit(cls, value: ScaleFloatType, info: ValidationInfo) -> ScaleFloatType:
            bounds = 0, float("inf")
            result = to_tuple(value, bias=1.0)
            check_range(result, *bounds, str(info.field_name))
            return result

    def __init__(
        self,
        shift_limit: ScaleFloatType = (-0.0625, 0.0625),
        scale_limit: ScaleFloatType = (-0.1, 0.1),
        rotate_limit: ScaleFloatType = (-45, 45),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType = 0,
        mask_value: ColorType = 0,
        shift_limit_x: ScaleFloatType | None = None,
        shift_limit_y: ScaleFloatType | None = None,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(
            scale=scale_limit,
            translate_percent={"x": shift_limit_x, "y": shift_limit_y},
            rotate=rotate_limit,
            shear=(0, 0),
            interpolation=interpolation,
            mask_interpolation=cv2.INTER_NEAREST,
            cval=value,
            cval_mask=mask_value,
            mode=border_mode,
            fit_output=False,
            keep_ratio=False,
            rotate_method=rotate_method,
            always_apply=always_apply,
            p=p,
        )
        warn(
            "ShiftScaleRotate is deprecated. Please use Affine transform instead .",
            DeprecationWarning,
            stacklevel=2,
        )
        self.shift_limit_x = cast(Tuple[float, float], shift_limit_x)
        self.shift_limit_y = cast(Tuple[float, float], shift_limit_y)
        self.scale_limit = cast(Tuple[float, float], scale_limit)
        self.rotate_limit = cast(Tuple[int, int], rotate_limit)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def get_transform_init_args(self) -> dict[str, Any]:
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


class PiecewiseAffine(DualTransform):
    """Apply affine transformations that differ between local neighborhoods.
    This augmentation places a regular grid of points on an image and randomly moves the neighborhood of these point
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

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        scale: NonNegativeFloatRangeType = (0.03, 0.05)
        nb_rows: ScaleIntType = Field(default=4, description="Number of rows in the regular grid.")
        nb_cols: ScaleIntType = Field(default=4, description="Number of columns in the regular grid.")
        interpolation: InterpolationType = cv2.INTER_LINEAR
        mask_interpolation: InterpolationType = cv2.INTER_NEAREST
        cval: int = Field(default=0, description="Constant value used for newly created pixels.")
        cval_mask: int = Field(default=0, description="Constant value used for newly created mask pixels.")
        mode: Literal["constant", "edge", "symmetric", "reflect", "wrap"] = "constant"
        absolute_scale: bool = Field(
            default=False,
            description="Whether scale is an absolute value rather than relative.",
        )
        keypoints_threshold: float = Field(
            default=0.01,
            description="Threshold for conversion from distance maps to keypoints.",
        )

        @field_validator("nb_rows", "nb_cols")
        @classmethod
        def process_range(cls, value: ScaleFloatType, info: ValidationInfo) -> tuple[float, float]:
            bounds = 2, BIG_INTEGER
            result = to_tuple(value, value)
            check_range(result, *bounds, info.field_name)
            return result

    def __init__(
        self,
        scale: ScaleFloatType = (0.03, 0.05),
        nb_rows: ScaleIntType = 4,
        nb_cols: ScaleIntType = 4,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        cval: int = 0,
        cval_mask: int = 0,
        mode: Literal["constant", "edge", "symmetric", "reflect", "wrap"] = "constant",
        absolute_scale: bool = False,
        always_apply: bool | None = None,
        keypoints_threshold: float = 0.01,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)

        warn(
            "This augmenter is very slow. Try to use ``ElasticTransformation`` instead, which is at least 10x faster.",
            stacklevel=2,
        )

        self.scale = cast(Tuple[float, float], scale)
        self.nb_rows = cast(Tuple[int, int], nb_rows)
        self.nb_cols = cast(Tuple[int, int], nb_cols)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.cval = cval
        self.cval_mask = cval_mask
        self.mode = mode
        self.absolute_scale = absolute_scale
        self.keypoints_threshold = keypoints_threshold

    def get_transform_init_args_names(self) -> tuple[str, ...]:
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

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]

        nb_rows = np.clip(random.randint(*self.nb_rows), 2, None)
        nb_cols = np.clip(random.randint(*self.nb_cols), 2, None)
        nb_cells = nb_cols * nb_rows
        scale = random.uniform(*self.scale)

        jitter: np.ndarray = random_utils.normal(0, scale, (nb_cells, 2))
        if not np.any(jitter > 0):
            for _ in range(10):  # See: https://github.com/albumentations-team/albumentations/issues/1442
                jitter = random_utils.normal(0, scale, (nb_cells, 2))
                if np.any(jitter > 0):
                    break
            if not np.any(jitter > 0):
                return {"matrix": None}

        y = np.linspace(0, height, nb_rows)
        x = np.linspace(0, width, nb_cols)

        # (H, W) and (H, W) for H=rows, W=cols
        xx_src, yy_src = np.meshgrid(x, y)

        # (1, HW, 2) => (HW, 2) for H=rows, W=cols
        points_src = np.dstack([yy_src.flat, xx_src.flat])[0]

        if self.absolute_scale:
            jitter[:, 0] = jitter[:, 0] / height if height > 0 else 0.0
            jitter[:, 1] = jitter[:, 1] / width if width > 0 else 0.0

        jitter[:, 0] = jitter[:, 0] * height
        jitter[:, 1] = jitter[:, 1] * width

        points_dest = np.copy(points_src)
        points_dest[:, 0] = points_dest[:, 0] + jitter[:, 0]
        points_dest[:, 1] = points_dest[:, 1] + jitter[:, 1]

        # Restrict all destination points to be inside the image plane.
        # This is necessary, as otherwise keypoints could be augmented
        # outside of the image plane and these would be replaced by
        # (-1, -1), which would not conform with the behaviour of the other augmenters.
        points_dest[:, 0] = np.clip(points_dest[:, 0], 0, height - 1)
        points_dest[:, 1] = np.clip(points_dest[:, 1], 0, width - 1)

        matrix = skimage.transform.PiecewiseAffineTransform()
        matrix.estimate(points_src[:, ::-1], points_dest[:, ::-1])

        return {
            "matrix": matrix,
        }

    def apply(
        self,
        img: np.ndarray,
        matrix: skimage.transform.PiecewiseAffineTransform,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.piecewise_affine(img, matrix, self.interpolation, self.mode, self.cval)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        matrix: skimage.transform.PiecewiseAffineTransform,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.piecewise_affine(mask, matrix, self.mask_interpolation, self.mode, self.cval_mask)

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        rows: int,
        cols: int,
        matrix: skimage.transform.PiecewiseAffineTransform,
        **params: Any,
    ) -> BoxInternalType:
        return fgeometric.bbox_piecewise_affine(bbox, matrix, rows, cols, self.keypoints_threshold)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        rows: int,
        cols: int,
        matrix: skimage.transform.PiecewiseAffineTransform,
        **params: Any,
    ) -> KeypointInternalType:
        return fgeometric.keypoint_piecewise_affine(keypoint, matrix, rows, cols, self.keypoints_threshold)


class PadIfNeeded(DualTransform):
    """Pads the sides of an image if the image dimensions are less than the specified minimum dimensions.
    If the `pad_height_divisor` or `pad_width_divisor` is specified, the function additionally ensures
    that the image dimensions are divisible by these values.

    Args:
        min_height (int): Minimum desired height of the image. Ensures image height is at least this value.
        min_width (int): Minimum desired width of the image. Ensures image width is at least this value.
        pad_height_divisor (int, optional): If set, pads the image height to make it divisible by this value.
        pad_width_divisor (int, optional): If set, pads the image width to make it divisible by this value.
        position (Union[str, PositionType]): Position where the image is to be placed after padding.
            Can be one of 'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', or 'random'.
            Default is 'center'.
        border_mode (int): Specifies the border mode to use if padding is required.
            The default is `cv2.BORDER_REFLECT_101`. If `value` is provided and `border_mode` is set to a mode
            that does not use a constant value, it should be manually set to `cv2.BORDER_CONSTANT`.
        value (Union[int, float, list[int], list[float]], optional): Value to fill the border pixels if
            the border mode is `cv2.BORDER_CONSTANT`. Default is None.
        mask_value (Union[int, float, list[int], list[float]], optional): Similar to `value` but used for padding masks.
            Default is None.
        p (float): Probability of applying the transform. Default is 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    """

    class PositionType(Enum):
        """Enumerates the types of positions for placing an object within a container.

        This Enum class is utilized to define specific anchor positions that an object can
        assume relative to a container. It's particularly useful in image processing, UI layout,
        and graphic design to specify the alignment and positioning of elements.

        Attributes:
            CENTER (str): Specifies that the object should be placed at the center.
            TOP_LEFT (str): Specifies that the object should be placed at the top-left corner.
            TOP_RIGHT (str): Specifies that the object should be placed at the top-right corner.
            BOTTOM_LEFT (str): Specifies that the object should be placed at the bottom-left corner.
            BOTTOM_RIGHT (str): Specifies that the object should be placed at the bottom-right corner.
            RANDOM (str): Indicates that the object's position should be determined randomly.

        """

        CENTER = "center"
        TOP_LEFT = "top_left"
        TOP_RIGHT = "top_right"
        BOTTOM_LEFT = "bottom_left"
        BOTTOM_RIGHT = "bottom_right"
        RANDOM = "random"

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        min_height: int | None = Field(default=None, ge=1, description="Minimal result image height.")
        min_width: int | None = Field(default=None, ge=1, description="Minimal result image width.")
        pad_height_divisor: int | None = Field(
            default=None,
            ge=1,
            description="Ensures image height is divisible by this value.",
        )
        pad_width_divisor: int | None = Field(
            default=None,
            ge=1,
            description="Ensures image width is divisible by this value.",
        )
        position: str = Field(default="center", description="Position of the padded image.")
        border_mode: BorderModeType = cv2.BORDER_REFLECT_101
        value: ColorType | None = Field(default=None, description="Value for border if BORDER_CONSTANT is used.")
        mask_value: ColorType | None = Field(
            default=None,
            description="Value for mask border if BORDER_CONSTANT is used.",
        )
        p: ProbabilityType = 1.0

        @model_validator(mode="after")
        def validate_divisibility(self) -> Self:
            if (self.min_height is None) == (self.pad_height_divisor is None):
                msg = "Only one of 'min_height' and 'pad_height_divisor' parameters must be set"
                raise ValueError(msg)
            if (self.min_width is None) == (self.pad_width_divisor is None):
                msg = "Only one of 'min_width' and 'pad_width_divisor' parameters must be set"
                raise ValueError(msg)

            if self.value is not None and self.border_mode in {cv2.BORDER_REFLECT_101, cv2.BORDER_REFLECT101}:
                self.border_mode = cv2.BORDER_CONSTANT

            if self.border_mode == cv2.BORDER_CONSTANT and self.value is None:
                msg = "If 'border_mode' is set to 'BORDER_CONSTANT', 'value' must be provided."
                raise ValueError(msg)

            return self

    def __init__(
        self,
        min_height: int | None = 1024,
        min_width: int | None = 1024,
        pad_height_divisor: int | None = None,
        pad_width_divisor: int | None = None,
        position: PositionType | str = PositionType.CENTER,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType | None = None,
        mask_value: ColorType | None = None,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(p, always_apply)
        self.min_height = min_height
        self.min_width = min_width
        self.pad_width_divisor = pad_width_divisor
        self.pad_height_divisor = pad_height_divisor
        self.position = PadIfNeeded.PositionType(position)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        params = super().update_params(params, **kwargs)
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
            h_top=h_pad_top,
            h_bottom=h_pad_bottom,
            w_left=w_pad_left,
            w_right=w_pad_right,
        )

        params.update(
            {
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
            },
        )
        return params

    def apply(
        self,
        img: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.pad_with_params(
            mask,
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
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        rows: int,
        cols: int,
        **params: Any,
    ) -> BoxInternalType:
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return cast(BoxInternalType, normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right))

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> KeypointInternalType:
        x, y, angle, scale = keypoint[:4]
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "min_height",
            "min_width",
            "pad_height_divisor",
            "pad_width_divisor",
            "position",
            "border_mode",
            "value",
            "mask_value",
        )

    def __update_position_params(
        self,
        h_top: int,
        h_bottom: int,
        w_left: int,
        w_right: int,
    ) -> tuple[int, int, int, int]:
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

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.vflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return fgeometric.bbox_vflip(bbox, params["shape"][0], params["shape"][1])

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        return fgeometric.keypoint_vflip(keypoint, params["shape"][0], params["shape"][1])

    def get_transform_init_args_names(self) -> tuple[()]:
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

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if get_num_channels(img) > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return fgeometric.hflip_cv2(img)

        return fgeometric.hflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return fgeometric.bbox_hflip(bbox, params["shape"][0], params["shape"][1])

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        return fgeometric.keypoint_hflip(keypoint, params["shape"][0], params["shape"][1])

    def get_transform_init_args_names(self) -> tuple[()]:
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

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def apply(self, img: np.ndarray, d: int, **params: Any) -> np.ndarray:
        """Args:
        d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        return fgeometric.random_flip(img, d)

    def get_params(self) -> dict[str, int]:
        # Random int in the range [-1, 1]
        return {"d": random.randint(-1, 1)}

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return fgeometric.bbox_flip(bbox, params["d"], params["shape"][0], params["shape"][1])

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        return fgeometric.keypoint_flip(keypoint, params["d"], params["shape"][0], params["shape"][1])

    def get_transform_init_args_names(self) -> tuple[()]:
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

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.transpose(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return fgeometric.bbox_transpose(bbox, params["shape"][0], params["shape"][1])

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        return fgeometric.keypoint_transpose(keypoint, params["shape"][0], params["shape"][1])

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()


class OpticalDistortion(DualTransform):
    """Args:
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
        image, mask, bboxes

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        distort_limit: SymmetricRangeType = (-0.05, 0.05)
        shift_limit: SymmetricRangeType = (-0.05, 0.05)
        interpolation: InterpolationType = cv2.INTER_LINEAR
        border_mode: BorderModeType = cv2.BORDER_REFLECT_101
        value: ColorType | None = Field(
            default=None,
            description="Padding value if border_mode is cv2.BORDER_CONSTANT.",
        )
        mask_value: ColorType | None = Field(
            default=None,
            description="Padding value for mask if border_mode is cv2.BORDER_CONSTANT.",
        )

    def __init__(
        self,
        distort_limit: ScaleFloatType = (-0.05, 0.05),
        shift_limit: ScaleFloatType = (-0.05, 0.05),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType | None = None,
        mask_value: ColorType | None = None,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)
        self.shift_limit = cast(Tuple[float, float], shift_limit)
        self.distort_limit = cast(Tuple[float, float], distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(
        self,
        img: np.ndarray,
        k: int,
        dx: int,
        dy: int,
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.optical_distortion(img, k, dx, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, mask: np.ndarray, k: int, dx: int, dy: int, **params: Any) -> np.ndarray:
        return fgeometric.optical_distortion(mask, k, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        k: int,
        dx: int,
        dy: int,
        **params: Any,
    ) -> BoxInternalType:
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = fgeometric.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = fgeometric.optical_distortion(mask, k, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)
        bbox_returned = bbox_from_mask(mask)
        return cast(BoxInternalType, fgeometric.normalize_bbox(bbox_returned, rows, cols))

    def get_params(self) -> dict[str, Any]:
        return {
            "k": random.uniform(self.distort_limit[0], self.distort_limit[1]),
            "dx": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
            "dy": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "distort_limit",
            "shift_limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
        )

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
        }


class GridDistortion(DualTransform):
    """Applies grid distortion augmentation to images, masks, and bounding boxes. This technique involves dividing
    the image into a grid of cells and randomly displacing the intersection points of the grid,
    resulting in localized distortions.

    Args:
        num_steps (int): Number of grid cells on each side (minimum 1).
        distort_limit (float, (float, float)): Range of distortion limits. If a single float is provided,
            the range will be from (-distort_limit, distort_limit). Default: (-0.3, 0.3).
        interpolation (OpenCV flag): Interpolation algorithm used for image transformation. Options are:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): Pixel extrapolation method used when pixels outside the image are required.
            Options are: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101.
        value (int, float, list of ints, list of floats, optional): Value used for padding when
            border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float, list of ints, list of floats, optional): Padding value for masks when
            border_mode is cv2.BORDER_CONSTANT.
        normalized (bool): If True, ensures that distortion does not exceed image boundaries. Default: False.
            Reference: https://github.com/albumentations-team/albumentations/pull/722

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32

    Note:
        This transform is helpful in medical imagery, Optical Character Recognition, and other tasks where local
        distance may not be preserved.
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        num_steps: Annotated[int, Field(ge=1, description="Count of grid cells on each side.")]
        distort_limit: SymmetricRangeType = (-0.3, 0.3)
        interpolation: InterpolationType = cv2.INTER_LINEAR
        border_mode: BorderModeType = cv2.BORDER_REFLECT_101
        value: ColorType | None = Field(
            default=None,
            description="Padding value if border_mode is cv2.BORDER_CONSTANT.",
        )
        mask_value: ColorType | None = Field(
            default=None,
            description="Padding value for mask if border_mode is cv2.BORDER_CONSTANT.",
        )
        normalized: bool = Field(
            default=False,
            description="If true, distortion will be normalized to not go outside the image.",
        )

        @field_validator("distort_limit")
        @classmethod
        def check_limits(cls, v: tuple[float, float], info: ValidationInfo) -> tuple[float, float]:
            bounds = -1, 1
            result = to_tuple(v)
            check_range(result, *bounds, info.field_name)
            return result

    def __init__(
        self,
        num_steps: int = 5,
        distort_limit: ScaleFloatType = (-0.3, 0.3),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType | None = None,
        mask_value: ColorType | None = None,
        normalized: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)

        self.num_steps = num_steps
        self.distort_limit = cast(Tuple[float, float], distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.normalized = normalized

    def apply(
        self,
        img: np.ndarray,
        stepsx: tuple[()],
        stepsy: tuple[()],
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.grid_distortion(
            img,
            self.num_steps,
            stepsx,
            stepsy,
            interpolation,
            self.border_mode,
            self.value,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        stepsx: tuple[()],
        stepsy: tuple[()],
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.grid_distortion(
            mask,
            self.num_steps,
            stepsx,
            stepsy,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        stepsx: tuple[()],
        stepsy: tuple[()],
        **params: Any,
    ) -> BoxInternalType:
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = fgeometric.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = fgeometric.grid_distortion(
            mask,
            self.num_steps,
            stepsx,
            stepsy,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
        )
        bbox_returned = bbox_from_mask(mask)
        return cast(BoxInternalType, fgeometric.normalize_bbox(bbox_returned, rows, cols))

    def _normalize(self, h: int, w: int, xsteps: list[float], ysteps: list[float]) -> dict[str, Any]:
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

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]

        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for _ in range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for _ in range(self.num_steps + 1)]

        if self.normalized:
            return self._normalize(height, width, stepsx, stepsy)

        return {"stepsx": stepsx, "stepsy": stepsy}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "num_steps", "distort_limit", "interpolation", "border_mode", "value", "mask_value", "normalized"

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
        }


class D4(DualTransform):
    """Applies one of the eight possible D4 dihedral group transformations to a square-shaped input,
        maintaining the square shape. These transformations correspond to the symmetries of a square,
        including rotations and reflections.

    The D4 group transformations include:
    - 'e' (identity): No transformation is applied.
    - 'r90' (rotation by 90 degrees counterclockwise)
    - 'r180' (rotation by 180 degrees)
    - 'r270' (rotation by 270 degrees counterclockwise)
    - 'v' (reflection across the vertical midline)
    - 'hvt' (reflection across the anti-diagonal)
    - 'h' (reflection across the horizontal midline)
    - 't' (reflection across the main diagonal)

    Even if the probability (`p`) of applying the transform is set to 1, the identity transformation
    'e' may still occur, which means the input will remain unchanged in one out of eight cases.

    Args:
        p (float): Probability of applying the transform. Default is 1, meaning the
                   transform is applied every time it is called.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        This transform is particularly useful when augmenting data that does not have a clear orientation:
        - Top view satellite or drone imagery
        - Medical images

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        p: ProbabilityType = 1

    def __init__(
        self,
        always_apply: bool | None = None,
        p: float = 1,
    ):
        super().__init__(p, always_apply)

    def apply(self, img: np.ndarray, group_element: D4Type, **params: Any) -> np.ndarray:
        return fgeometric.d4(img, group_element)

    def apply_to_bbox(self, bbox: BoxInternalType, group_element: D4Type, **params: Any) -> BoxInternalType:
        return fgeometric.bbox_d4(bbox, group_element, params["shape"][0], params["shape"][1])

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        group_element: D4Type,
        **params: Any,
    ) -> KeypointInternalType:
        return fgeometric.keypoint_d4(keypoint, group_element, params["shape"][0], params["shape"][1])

    def get_params(self) -> dict[str, D4Type]:
        return {
            "group_element": random_utils.choice(d4_group_elements),
        }

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()

class GridElasticDeform(DualTransform):
    """ Grid-based Elastic deformation Albumentation implementation

    This class applies elastic transformations on a grid-based approach,
    where the granularity of the distortions can be controlled using the
    width and height of the overlaying distortion grid. Larger grid sizes
    result in finer, less severe distortions.

    Params:
        n_grid_width (int): Number of grid cells along the width
        n_grid_height (int): Number of grid cells along the height
        magnitude (int): Magnitude of distortion
        p (float): Probability of applying the transform
        always_apply (bool | None): If true, the transform will always be applied.

    Targets:
        image, mask

    Image types:
        uint8
    """

    def __init__(
        self,
        n_grid_width: int,
        n_grid_height: int,
        magnitude: int,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.n_grid_width = n_grid_width
        self.n_grid_height = n_grid_height
        self.magnitude = abs(magnitude)

    def calculate_dimensions(
        self,
        width_of_square,
        height_of_square,
        width_of_last_square,
        height_of_last_square,
    ):
        dimensions = []
        for vertical_tile in range(self.n_grid_width):
            for horizontal_tile in range(self.n_grid_height):
                x1 = horizontal_tile * width_of_square
                y1 = vertical_tile * height_of_square
                x2 = x1 + (
                    width_of_last_square
                    if horizontal_tile == self.n_grid_height - 1
                    else width_of_square
                )
                y2 = y1 + (
                    height_of_last_square
                    if vertical_tile == self.n_grid_width - 1
                    else height_of_square
                )
                dimensions.append([x1, y1, x2, y2])

        return dimensions

    def calculate_polygons(self, dimensions, horizontal_tiles, vertical_tiles):
        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        last_column = [
            (horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)
        ]
        last_row = range(
            (horizontal_tiles * vertical_tiles) - horizontal_tiles,
            horizontal_tiles * vertical_tiles,
        )

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append(
                    [i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles]
                )

        for a, b, c, d in polygon_indices:
            dx = random.randint(-self.magnitude, self.magnitude)
            dy = random.randint(-self.magnitude, self.magnitude)

            polygons[a][4] += dx
            polygons[a][5] += dy
            polygons[b][2] += dx
            polygons[b][3] += dy
            polygons[c][6] += dx
            polygons[c][7] += dy
            polygons[d][0] += dx
            polygons[d][1] += dy

        return polygons

    def generate_mesh(self, polygons, dimensions):
        return [[dimensions[i], polygons[i]] for i in range(len(dimensions))]

    def distort_image(self, image: np.ndarray, generated_mesh: List[List]):
        image = Image.fromarray(image)
        return np.array(
            image.transform(
                image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC
            )
        )

    def get_params_dependent_on_data(
        self, params: Dict[str, Any], data: dict[str, Any]
    ) -> Dict[str, Any]:

        img = data["image"]
        h, w = img.shape[:2]

        horizontal_tiles = self.n_grid_width
        vertical_tiles = self.n_grid_height

        width_of_square = int(w / horizontal_tiles)
        height_of_square = int(h / vertical_tiles)

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = self.calculate_dimensions(
            width_of_square,
            height_of_square,
            width_of_last_square,
            height_of_last_square,
        )
        polygons = self.calculate_polygons(dimensions, horizontal_tiles, vertical_tiles)
        generated_mesh = self.generate_mesh(polygons, dimensions)

        return {"generated_mesh": generated_mesh}

    def apply(self, img, generated_mesh, **params):
        return self.distort_image(img, generated_mesh)

    def apply_to_mask(self, mask, generated_mesh, **params):
        return self.distort_image(mask, generated_mesh)

    def get_transform_init_args_names(self):
        return ("n_grid_width", "n_grid_height", "magnitude")
