from __future__ import annotations

import random
from typing import Annotated, Any, Literal, cast
from warnings import warn

import cv2
import numpy as np
from albucore import hflip, vflip
from pydantic import AfterValidator, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from albumentations.augmentations.utils import check_range
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from albumentations.core.pydantic import (
    BorderModeType,
    InterpolationType,
    NonNegativeFloatRangeType,
    SymmetricRangeType,
    check_1plus,
)
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import (
    BIG_INTEGER,
    ColorType,
    D4Type,
    PositionType,
    ScalarType,
    ScaleFloatType,
    ScaleIntType,
    Targets,
    d4_group_elements,
)
from albumentations.core.utils import to_tuple

from . import functional as fgeometric

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
    "GridElasticDeform",
]


class BaseDistortion(DualTransform):
    """Base class for distortion-based transformations.

    This class provides a foundation for implementing various types of image distortions,
    such as optical distortions, grid distortions, and elastic transformations. It handles
    the common operations of applying distortions to images, masks, bounding boxes, and keypoints.

    Args:
        interpolation (int): Interpolation method to be used for image transformation.
            Should be one of the OpenCV interpolation types (e.g., cv2.INTER_LINEAR,
            cv2.INTER_CUBIC). Default: cv2.INTER_LINEAR
        border_mode (int): Border mode to be used for handling pixels outside the image boundaries.
            Should be one of the OpenCV border types (e.g., cv2.BORDER_REFLECT_101,
            cv2.BORDER_CONSTANT). Default: cv2.BORDER_REFLECT_101
        value (int, float, list of int, list of float, optional): Padding value if border_mode is
            cv2.BORDER_CONSTANT. Default: None
        mask_value (ColorType | None): Padding value for mask if
            border_mode is cv2.BORDER_CONSTANT. Default: None
        mask_interpolation (int): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - This is an abstract base class and should not be used directly.
        - Subclasses should implement the `get_params_dependent_on_data` method to generate
          the distortion maps (map_x and map_y).
        - The distortion is applied consistently across all targets (image, mask, bboxes, keypoints)
          to maintain coherence in the augmented data.

    Example of a subclass:
        class CustomDistortion(BaseDistortion):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Add custom parameters here

            def get_params_dependent_on_data(self, params, data):
                # Generate and return map_x and map_y based on the distortion logic
                return {"map_x": map_x, "map_y": map_y}

            def get_transform_init_args_names(self):
                return super().get_transform_init_args_names() + ("custom_param1", "custom_param2")
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        interpolation: InterpolationType
        mask_interpolation: InterpolationType
        border_mode: BorderModeType
        value: ColorType | None
        mask_value: ColorType | None

    def __init__(
        self,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType | None = None,
        mask_value: ColorType | None = None,
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.mask_interpolation = mask_interpolation

    def apply(self, img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.remap(img, map_x, map_y, self.interpolation, self.border_mode, self.value)

    def apply_to_mask(self, mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.remap(mask, map_x, map_y, self.mask_interpolation, self.border_mode, self.mask_value)

    def apply_to_bboxes(self, bboxes: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, **params: Any) -> np.ndarray:
        image_shape = params["shape"][:2]
        bboxes_denorm = denormalize_bboxes(bboxes, image_shape)
        bboxes_returned = fgeometric.remap_bboxes(bboxes_denorm, map_x, map_y, image_shape)
        return normalize_bboxes(bboxes_returned, image_shape)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.remap_keypoints(keypoints, map_x, map_y, params["shape"])

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "interpolation", "border_mode", "value", "mask_value", "mask_interpolation"


class ElasticTransform(BaseDistortion):
    """Apply elastic deformation to images, masks, bounding boxes, and keypoints.

    This transformation introduces random elastic distortions to the input data. It's particularly
    useful for data augmentation in training deep learning models, especially for tasks like
    image segmentation or object detection where you want to maintain the relative positions of
    features while introducing realistic deformations.

    The transform works by generating random displacement fields and applying them to the input.
    These fields are smoothed using a Gaussian filter to create more natural-looking distortions.

    Args:
        alpha (float): Scaling factor for the random displacement fields. Higher values result in
            more pronounced distortions. Default: 1.0
        sigma (float): Standard deviation of the Gaussian filter used to smooth the displacement
            fields. Higher values result in smoother, more global distortions. Default: 50.0
        interpolation (int): Interpolation method to be used for image transformation. Should be one
            of the OpenCV interpolation types. Default: cv2.INTER_LINEAR
        border_mode (int): Border mode to be used for handling pixels outside the image boundaries.
            Should be one of the OpenCV border types. Default: cv2.BORDER_REFLECT_101
        value (int, float, list of int, list of float, optional): Padding value if border_mode is
            cv2.BORDER_CONSTANT. Default: None
        mask_value (int, float, list of int, list of float, optional): Padding value for mask if
            border_mode is cv2.BORDER_CONSTANT. Default: None
        approximate (bool): Whether to use an approximate version of the elastic transform. If True,
            uses a fixed kernel size for Gaussian smoothing, which can be faster but potentially
            less accurate for large sigma values. Default: False
        same_dxdy (bool): Whether to use the same random displacement field for both x and y
            directions. Can speed up the transform at the cost of less diverse distortions. Default: False
        mask_interpolation (int): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The transform will maintain consistency across all targets (image, mask, bboxes, keypoints)
          by using the same displacement fields for all.
        - The 'approximate' parameter determines whether to use a precise or approximate method for
          generating displacement fields. The approximate method can be faster but may be less
          accurate for large sigma values.
        - Bounding boxes that end up outside the image after transformation will be removed.
        - Keypoints that end up outside the image after transformation will be removed.

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.ElasticTransform(alpha=1, sigma=50, p=0.5),
        ... ])
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>> transformed_image = transformed['image']
        >>> transformed_mask = transformed['mask']
        >>> transformed_bboxes = transformed['bboxes']
        >>> transformed_keypoints = transformed['keypoints']
    """

    class InitSchema(BaseDistortion.InitSchema):
        alpha: Annotated[float, Field(ge=0)]
        sigma: Annotated[float, Field(ge=1)]
        approximate: bool
        same_dxdy: bool

    def __init__(
        self,
        alpha: float = 1,
        sigma: float = 50,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ScalarType | list[ScalarType] | None = None,
        mask_value: ScalarType | list[ScalarType] | None = None,
        always_apply: bool | None = None,
        approximate: bool = False,
        same_dxdy: bool = False,
        mask_interpolation: int = cv2.INTER_NEAREST,
        p: float = 0.5,
    ):
        super().__init__(
            interpolation=interpolation,
            border_mode=border_mode,
            value=value,
            mask_value=mask_value,
            always_apply=always_apply,
            p=p,
            mask_interpolation=mask_interpolation,
        )
        self.alpha = alpha
        self.sigma = sigma
        self.approximate = approximate
        self.same_dxdy = same_dxdy

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]
        kernel_size = (0, 0) if self.approximate else (17, 17)

        # Generate displacement fields
        dx, dy = fgeometric.generate_displacement_fields(
            (height, width),
            self.alpha,
            self.sigma,
            same_dxdy=self.same_dxdy,
            kernel_size=kernel_size,
            random_generator=self.random_generator,
        )

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        return {"map_x": map_x, "map_y": map_y}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (*super().get_transform_init_args_names(), "alpha", "sigma", "approximate", "same_dxdy")


class Perspective(DualTransform):
    """Apply random four point perspective transformation to the input.

    Args:
        scale (float or tuple of float): Standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners.
            If scale is a single float value, the range will be (0, scale).
            Default: (0.05, 0.1).
        keep_size (bool): Whether to resize image back to its original size after applying the perspective transform.
            If set to False, the resulting images may end up having different shapes.
            Default: True.
        pad_mode (OpenCV flag): OpenCV border mode used for padding.
            Default: cv2.BORDER_CONSTANT.
        pad_val (int, float, list of int, list of float): Padding value if border_mode is cv2.BORDER_CONSTANT.
            Default: 0.
        mask_pad_val (int, float, list of int, list of float): Padding value for mask if border_mode is
            cv2.BORDER_CONSTANT. Default: 0.
        fit_output (bool): If True, the image plane size and position will be adjusted to still capture
            the whole image after perspective transformation. This is followed by image resizing if keep_size is set
            to True. If False, parts of the transformed image may be outside of the image plane.
            This setting should not be set to True when using large scale values as it could lead to very large images.
            Default: False.
        interpolation (int): Interpolation method to be used for image transformation. Should be one
            of the OpenCV interpolation types. Default: cv2.INTER_LINEAR
        mask_interpolation (int): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    Note:
        This transformation creates a perspective effect by randomly moving the four corners of the image.
        The amount of movement is controlled by the 'scale' parameter.

        When 'keep_size' is True, the output image will have the same size as the input image,
        which may cause some parts of the transformed image to be cut off or padded.

        When 'fit_output' is True, the transformation ensures that the entire transformed image is visible,
        which may result in a larger output image if keep_size is False.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.Perspective(scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5),
        ... ])
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        scale: NonNegativeFloatRangeType
        keep_size: bool
        pad_mode: BorderModeType
        pad_val: ColorType | None
        mask_pad_val: ColorType | None
        fit_output: bool
        interpolation: InterpolationType
        mask_interpolation: InterpolationType

    def __init__(
        self,
        scale: ScaleFloatType = (0.05, 0.1),
        keep_size: bool = True,
        pad_mode: int = cv2.BORDER_CONSTANT,
        pad_val: ColorType = 0,
        mask_pad_val: ColorType = 0,
        fit_output: bool = False,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply=always_apply)
        self.scale = cast(tuple[float, float], scale)
        self.keep_size = keep_size
        self.pad_mode = pad_mode
        self.pad_val = pad_val
        self.mask_pad_val = mask_pad_val
        self.fit_output = fit_output
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

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
            self.interpolation,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        matrix: np.ndarray,
        max_height: int,
        max_width: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.perspective(
            mask,
            matrix,
            max_width,
            max_height,
            self.pad_val,
            self.pad_mode,
            self.keep_size,
            self.mask_interpolation,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        matrix: np.ndarray,
        max_height: int,
        max_width: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.perspective_bboxes(
            bboxes,
            params["shape"],
            matrix,
            max_width,
            max_height,
            self.keep_size,
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        matrix: np.ndarray,
        max_height: int,
        max_width: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.perspective_keypoints(
            keypoints,
            params["shape"],
            matrix,
            max_width,
            max_height,
            self.keep_size,
        )

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"][:2]

        scale = self.py_random.uniform(*self.scale)

        points = fgeometric.generate_perspective_points(image_shape, scale, self.random_generator)
        points = fgeometric.order_points(points)

        matrix, max_width, max_height = fgeometric.compute_perspective_params(points, image_shape)

        if self.fit_output:
            matrix, max_width, max_height = fgeometric.expand_transform(matrix, image_shape)

        return {"matrix": matrix, "max_height": max_height, "max_width": max_width}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "scale",
            "keep_size",
            "pad_mode",
            "pad_val",
            "mask_pad_val",
            "fit_output",
            "interpolation",
            "mask_interpolation",
        )


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
        scale: ScaleFloatType | dict[str, Any] | None
        translate_percent: ScaleFloatType | dict[str, Any] | None
        translate_px: ScaleIntType | dict[str, Any] | None
        rotate: ScaleFloatType | None
        shear: ScaleFloatType | dict[str, Any] | None
        interpolation: InterpolationType
        mask_interpolation: InterpolationType

        cval: ColorType
        cval_mask: ColorType
        mode: BorderModeType
        fit_output: bool
        keep_ratio: bool
        rotate_method: Literal["largest_box", "ellipse"]
        balanced_scale: bool

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
        matrix: np.ndarray,
        output_shape: tuple[int, int],
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
        matrix: np.ndarray,
        output_shape: tuple[int, int],
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

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        bbox_matrix: np.ndarray,
        output_shape: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.bboxes_affine(
            bboxes,
            bbox_matrix,
            self.rotate_method,
            params["shape"][:2],
            self.mode,
            output_shape,
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        matrix: np.ndarray,
        scale: dict[str, Any],
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.keypoints_affine(keypoints, matrix, params["shape"], scale, self.mode)

    @staticmethod
    def get_scale(
        scale: dict[str, float | tuple[float, float]],
        keep_ratio: bool,
        balanced_scale: bool,
        random_state: random.Random,
    ) -> fgeometric.ScaleDict:
        result_scale = {}
        for key, value in scale.items():
            if isinstance(value, (int, float)):
                result_scale[key] = float(value)
            elif isinstance(value, tuple):
                if balanced_scale:
                    lower_interval = (value[0], 1.0) if value[0] < 1 else None
                    upper_interval = (1.0, value[1]) if value[1] > 1 else None

                    if lower_interval is not None and upper_interval is not None:
                        selected_interval = random_state.choice([lower_interval, upper_interval])
                    elif lower_interval is not None:
                        selected_interval = lower_interval
                    elif upper_interval is not None:
                        selected_interval = upper_interval
                    else:
                        result_scale[key] = 1.0
                        continue

                    result_scale[key] = random_state.uniform(*selected_interval)
                else:
                    result_scale[key] = random_state.uniform(*value)
            else:
                raise TypeError(
                    f"Invalid scale value for key {key}: {value}. Expected a float or a tuple of two floats.",
                )

        if keep_ratio:
            result_scale["y"] = result_scale["x"]

        return cast(fgeometric.ScaleDict, result_scale)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"][:2]

        translate = self._get_translate_params(image_shape)
        shear = self._get_shear_params()
        scale = self.get_scale(self.scale, self.keep_ratio, self.balanced_scale, self.py_random)
        rotate = self.py_random.uniform(*self.rotate)

        image_shift = fgeometric.center(image_shape)
        bbox_shift = fgeometric.center_bbox(image_shape)

        matrix = fgeometric.create_affine_transformation_matrix(translate, shear, scale, rotate, image_shift)
        bbox_matrix = fgeometric.create_affine_transformation_matrix(translate, shear, scale, rotate, bbox_shift)

        if self.fit_output:
            matrix, output_shape = fgeometric.compute_affine_warp_output_shape(matrix, image_shape)
            bbox_matrix, _ = fgeometric.compute_affine_warp_output_shape(bbox_matrix, image_shape)
        else:
            output_shape = image_shape

        return {
            "rotate": rotate,
            "scale": scale,
            "matrix": matrix,
            "bbox_matrix": bbox_matrix,
            "output_shape": output_shape,
        }

    def _get_translate_params(self, image_shape: tuple[int, int]) -> fgeometric.TranslateDict:
        height, width = image_shape[:2]
        if self.translate_px is not None:
            return cast(
                fgeometric.TranslateDict,
                {key: self.py_random.randint(*value) for key, value in self.translate_px.items()},
            )
        if self.translate_percent is not None:
            translate = {key: self.py_random.uniform(*value) for key, value in self.translate_percent.items()}
            return cast(fgeometric.TranslateDict, {"x": translate["x"] * width, "y": translate["y"] * height})
        return cast(fgeometric.TranslateDict, {"x": 0, "y": 0})

    def _get_shear_params(self) -> fgeometric.ShearDict:
        return cast(fgeometric.ShearDict, {key: -self.py_random.uniform(*value) for key, value in self.shear.items()})


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
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
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
        mask_interpolation: InterpolationType

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
        mask_interpolation: InterpolationType = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(
            scale=scale_limit,
            translate_percent={"x": shift_limit_x, "y": shift_limit_y},
            rotate=rotate_limit,
            shear=(0, 0),
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
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
            "ShiftScaleRotate is deprecated. Please use Affine transform instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.shift_limit_x = cast(tuple[float, float], shift_limit_x)
        self.shift_limit_y = cast(tuple[float, float], shift_limit_y)
        self.scale_limit = cast(tuple[float, float], scale_limit)
        self.rotate_limit = cast(tuple[int, int], rotate_limit)
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
            "mask_interpolation": self.mask_interpolation,
        }


class PiecewiseAffine(DualTransform):
    """Apply piecewise affine transformations to the input image.

    This augmentation places a regular grid of points on an image and randomly moves the neighborhood of these points
    around via affine transformations. This leads to local distortions in the image.

    Args:
        scale (tuple[float, float] | float): Standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners.
            If scale is a single float value, the range will be (0, scale).
            Recommended values are in the range (0.01, 0.05) for small distortions,
            and (0.05, 0.1) for larger distortions. Default: (0.03, 0.05).
        nb_rows (tuple[int, int] | int): Number of rows of points that the regular grid should have.
            Must be at least 2. For large images, you might want to pick a higher value than 4.
            If a single int, then that value will always be used as the number of rows.
            If a tuple (a, b), then a value from the discrete interval [a..b] will be uniformly sampled per image.
            Default: 4.
        nb_cols (tuple[int, int] | int): Number of columns of points that the regular grid should have.
            Must be at least 2. For large images, you might want to pick a higher value than 4.
            If a single int, then that value will always be used as the number of columns.
            If a tuple (a, b), then a value from the discrete interval [a..b] will be uniformly sampled per image.
            Default: 4.
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        cval (number): The constant value to use when filling in newly created pixels.
            Default: 0.
        cval_mask (number): The constant value to use when filling in newly created pixels in masks.
            Default: 0.
        mode (str): {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
            Points outside the boundaries of the input are filled according
            to the given mode. Default: 'constant'.
        absolute_scale (bool): If set to True, the value of the scale parameter will be treated as an absolute
            pixel value. If set to False, it will be treated as a fraction of the image height and width.
            Default: False.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    Note:
        - This augmentation is very slow. Consider using `ElasticTransform` instead, which is at least 10x faster.
        - The augmentation may not always produce visible effects, especially with small scale values.
        - For keypoints and bounding boxes, the transformation might move them outside the image boundaries.
          In such cases, the keypoints will be set to (-1, -1) and the bounding boxes will be removed.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.PiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=0.5),
        ... ])
        >>> transformed = transform(image=image)
        >>> transformed_image = transformed["image"]
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        scale: NonNegativeFloatRangeType
        nb_rows: ScaleIntType
        nb_cols: ScaleIntType
        interpolation: InterpolationType
        mask_interpolation: InterpolationType
        cval: int
        cval_mask: int
        mode: Literal["constant", "edge", "symmetric", "reflect", "wrap"]
        absolute_scale: bool
        keypoints_threshold: float = Field(deprecated="This parameter is not used anymore")

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
        nb_rows: ScaleIntType = (4, 4),
        nb_cols: ScaleIntType = (4, 4),
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
        super().__init__(p=p, always_apply=always_apply)

        warn(
            "This augmenter is very slow. Try to use ``ElasticTransformation`` instead, which is at least 10x faster.",
            stacklevel=2,
        )

        self.scale = cast(tuple[float, float], scale)
        self.nb_rows = cast(tuple[int, int], nb_rows)
        self.nb_cols = cast(tuple[int, int], nb_cols)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.cval = cval
        self.cval_mask = cval_mask
        self.mode = mode
        self.absolute_scale = absolute_scale

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
        )

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]

        nb_rows = np.clip(self.py_random.randint(*self.nb_rows), 2, None)
        nb_cols = np.clip(self.py_random.randint(*self.nb_cols), 2, None)
        scale = self.py_random.uniform(*self.scale)

        map_x, map_y = fgeometric.create_piecewise_affine_maps(
            image_shape=(height, width),
            grid=(nb_rows, nb_cols),
            scale=scale,
            absolute_scale=self.absolute_scale,
            random_generator=self.random_generator,
        )

        border_modes = {
            "constant": cv2.BORDER_CONSTANT,  # fill with constant value (cval)
            "edge": cv2.BORDER_REPLICATE,  # repeat edge pixels
            "reflect": cv2.BORDER_REFLECT,  # mirror with edge pixels duplicated
            "reflect_101": cv2.BORDER_REFLECT_101,  # mirror without duplicating edge pixels
            "wrap": cv2.BORDER_WRAP,  # tile the image
        }
        border_mode = border_modes.get(self.mode, cv2.BORDER_REFLECT_101)

        return {"map_x": map_x, "map_y": map_y, "border_mode": border_mode}

    def apply(
        self,
        img: np.ndarray,
        map_x: np.ndarray | None,
        map_y: np.ndarray | None,
        border_mode: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.remap(img, map_x, map_y, self.interpolation, border_mode, self.cval)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        map_x: np.ndarray | None,
        map_y: np.ndarray | None,
        border_mode: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.remap(mask, map_x, map_y, self.mask_interpolation, border_mode, self.cval_mask)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        map_x: np.ndarray | None,
        map_y: np.ndarray | None,
        border_mode: int,
        **params: Any,
    ) -> np.ndarray:
        image_shape = params["shape"][:2]
        denormalized_bboxes = denormalize_bboxes(bboxes, image_shape)
        return normalize_bboxes(
            fgeometric.bboxes_piecewise_affine(denormalized_bboxes, map_x, map_y, border_mode, image_shape),
            image_shape,
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        map_x: np.ndarray | None,
        map_y: np.ndarray | None,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.remap_keypoints(keypoints, map_x, map_y, params["shape"])


class PadIfNeeded(DualTransform):
    """Pads the sides of an image if the image dimensions are less than the specified minimum dimensions.
    If the `pad_height_divisor` or `pad_width_divisor` is specified, the function additionally ensures
    that the image dimensions are divisible by these values.

    Args:
        min_height (int | None): Minimum desired height of the image. Ensures image height is at least this value.
            If not specified, pad_height_divisor must be provided.
        min_width (int | None): Minimum desired width of the image. Ensures image width is at least this value.
            If not specified, pad_width_divisor must be provided.
        pad_height_divisor (int | None): If set, pads the image height to make it divisible by this value.
            If not specified, min_height must be provided.
        pad_width_divisor (int | None): If set, pads the image width to make it divisible by this value.
            If not specified, min_width must be provided.
        position (Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"]):
            Position where the image is to be placed after padding. Default is 'center'.
        border_mode (int): Specifies the border mode to use if padding is required.
            The default is `cv2.BORDER_REFLECT_101`.
        value (int, float, list[int], list[float] | None): Value to fill the border pixels if
            the border mode is `cv2.BORDER_CONSTANT`. Default is None.
        mask_value (int, float, list[int], list[float] | None): Similar to `value` but used for padding masks.
            Default is None.
        p (float): Probability of applying the transform. Default is 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - Either `min_height` or `pad_height_divisor` must be set, but not both.
        - Either `min_width` or `pad_width_divisor` must be set, but not both.
        - If `border_mode` is set to `cv2.BORDER_CONSTANT`, `value` must be provided.
        - The transform will maintain consistency across all targets (image, mask, bboxes, keypoints).
        - For bounding boxes, the coordinates will be adjusted to account for the padding.
        - For keypoints, their positions will be shifted according to the padding.

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=0),
        ... ])
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>> padded_image = transformed['image']
        >>> padded_mask = transformed['mask']
        >>> adjusted_bboxes = transformed['bboxes']
        >>> adjusted_keypoints = transformed['keypoints']
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        min_height: int | None = Field(ge=1)
        min_width: int | None = Field(ge=1)
        pad_height_divisor: int | None = Field(ge=1)
        pad_width_divisor: int | None = Field(ge=1)
        position: PositionType
        border_mode: BorderModeType
        value: ColorType | None
        mask_value: ColorType | None

        @model_validator(mode="after")
        def validate_divisibility(self) -> Self:
            if (self.min_height is None) == (self.pad_height_divisor is None):
                msg = "Only one of 'min_height' and 'pad_height_divisor' parameters must be set"
                raise ValueError(msg)
            if (self.min_width is None) == (self.pad_width_divisor is None):
                msg = "Only one of 'min_width' and 'pad_width_divisor' parameters must be set"
                raise ValueError(msg)

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
        position: PositionType = "center",
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType | None = None,
        mask_value: ColorType | None = None,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.min_height = min_height
        self.min_width = min_width
        self.pad_width_divisor = pad_width_divisor
        self.pad_height_divisor = pad_height_divisor
        self.position = position
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        params = super().update_params(params, **kwargs)
        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = fgeometric.get_padding_params(
            image_shape=params["shape"][:2],
            min_height=self.min_height,
            min_width=self.min_width,
            pad_height_divisor=self.pad_height_divisor,
            pad_width_divisor=self.pad_width_divisor,
        )

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = fgeometric.adjust_padding_by_position(
            h_top=h_pad_top,
            h_bottom=h_pad_bottom,
            w_left=w_pad_left,
            w_right=w_pad_right,
            position=self.position,
            py_random=self.py_random,
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

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        image_shape = params["shape"][:2]
        bboxes_np = denormalize_bboxes(bboxes, params["shape"])

        result = fgeometric.pad_bboxes(
            bboxes_np,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            self.border_mode,
            image_shape=image_shape,
        )

        rows, cols = params["shape"][:2]

        return normalize_bboxes(result, (rows + pad_top + pad_bottom, cols + pad_left + pad_right))

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.pad_keypoints(
            keypoints,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            self.border_mode,
            image_shape=params["shape"][:2],
        )

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


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - This transform flips the image upside down. The top of the image becomes the bottom and vice versa.
        - The dimensions of the image remain unchanged.
        - For multi-channel images (like RGB), each channel is flipped independently.
        - Bounding boxes are adjusted to match their new positions in the flipped image.
        - Keypoints are moved to their new positions in the flipped image.

    Mathematical Details:
        1. For an input image I of shape (H, W, C), the output O is:
           O[i, j, k] = I[H-1-i, j, k] for all i in [0, H-1], j in [0, W-1], k in [0, C-1]
        2. For bounding boxes with coordinates (x_min, y_min, x_max, y_max):
           new_bbox = (x_min, H-y_max, x_max, H-y_min)
        3. For keypoints with coordinates (x, y):
           new_keypoint = (x, H-y)
        where H is the height of the image.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.array([
        ...     [[1, 2, 3], [4, 5, 6]],
        ...     [[7, 8, 9], [10, 11, 12]]
        ... ])
        >>> transform = A.VerticalFlip(p=1.0)
        >>> result = transform(image=image)
        >>> flipped_image = result['image']
        >>> print(flipped_image)
        [[[ 7  8  9]
          [10 11 12]]
         [[ 1  2  3]
          [ 4  5  6]]]
        # The original image is flipped vertically, with rows reversed

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return vflip(img)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.bboxes_vflip(bboxes)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.keypoints_vflip(keypoints, params["rows"])

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
        return hflip(img)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.bboxes_hflip(bboxes)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.keypoints_hflip(keypoints, params["cols"])

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()


class Flip(DualTransform):
    """Deprecated. Consider using HorizontalFlip, VerticalFlip, RandomRotate90 or D4."""

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def __init__(self, always_apply: bool | None = None, p: float = 0.5):
        super().__init__(p=p, always_apply=always_apply)
        warn(
            "Flip is deprecated. Consider using HorizontalFlip, VerticalFlip, RandomRotate90 or D4.",
            DeprecationWarning,
            stacklevel=2,
        )

    def apply(self, img: np.ndarray, d: int, **params: Any) -> np.ndarray:
        """Args:
        d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        return fgeometric.random_flip(img, d)

    def get_params(self) -> dict[str, int]:
        # Random int in the range [-1, 1]
        return {"d": self.py_random.randint(-1, 1)}

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.bboxes_flip(bboxes, params["d"])

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.keypoints_flip(keypoints, params["d"], params["shape"])

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()


class Transpose(DualTransform):
    """Transpose the input by swapping its rows and columns.

    This transform flips the image over its main diagonal, effectively switching its width and height.
    It's equivalent to a 90-degree rotation followed by a horizontal flip.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The dimensions of the output will be swapped compared to the input. For example,
          an input image of shape (100, 200, 3) will result in an output of shape (200, 100, 3).
        - This transform is its own inverse. Applying it twice will return the original input.
        - For multi-channel images (like RGB), the channels are preserved in their original order.
        - Bounding boxes will have their coordinates adjusted to match the new image dimensions.
        - Keypoints will have their x and y coordinates swapped.

    Mathematical Details:
        1. For an input image I of shape (H, W, C), the output O is:
           O[i, j, k] = I[j, i, k] for all i in [0, W-1], j in [0, H-1], k in [0, C-1]
        2. For bounding boxes with coordinates (x_min, y_min, x_max, y_max):
           new_bbox = (y_min, x_min, y_max, x_max)
        3. For keypoints with coordinates (x, y):
           new_keypoint = (y, x)

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.array([
        ...     [[1, 2, 3], [4, 5, 6]],
        ...     [[7, 8, 9], [10, 11, 12]]
        ... ])
        >>> transform = A.Transpose(p=1.0)
        >>> result = transform(image=image)
        >>> transposed_image = result['image']
        >>> print(transposed_image)
        [[[ 1  2  3]
          [ 7  8  9]]
         [[ 4  5  6]
          [10 11 12]]]
        # The original 2x2x3 image is now 2x2x3, with rows and columns swapped

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.transpose(img)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.bboxes_transpose(bboxes)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.keypoints_transpose(keypoints)

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()


class OpticalDistortion(BaseDistortion):
    """Apply optical distortion to images, masks, bounding boxes, and keypoints.

    This transformation simulates lens distortion effects by warping the image using
    a camera matrix and distortion coefficients. It's particularly useful for
    augmenting data in computer vision tasks where camera lens effects are relevant.

    Args:
        distort_limit (float or tuple of float): Range of distortion coefficient.
            If distort_limit is a single float, the range will be (-distort_limit, distort_limit).
            Default: (-0.05, 0.05).
        shift_limit (float or tuple of float): Range of shifts for the image center.
            If shift_limit is a single float, the range will be (-shift_limit, shift_limit).
            Default: (-0.05, 0.05).
        interpolation (OpenCV flag): Interpolation method used for image transformation.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
            cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): Border mode used for handling pixels outside the image.
            Should be one of: cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101.
        value (int, float, list of int, list of float): Padding value if border_mode
            is cv2.BORDER_CONSTANT. Default: None.
        mask_value (int, float, list of int, list of float): Padding value for mask
            if border_mode is cv2.BORDER_CONSTANT. Default: None.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The distortion is applied using OpenCV's initUndistortRectifyMap and remap functions.
        - The distortion coefficient (k) is randomly sampled from the distort_limit range.
        - The image center is shifted by dx and dy, randomly sampled from the shift_limit range.
        - Bounding boxes and keypoints are transformed along with the image to maintain consistency.

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
        ... ])
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>> transformed_image = transformed['image']
        >>> transformed_mask = transformed['mask']
        >>> transformed_bboxes = transformed['bboxes']
        >>> transformed_keypoints = transformed['keypoints']
    """

    class InitSchema(BaseDistortion.InitSchema):
        distort_limit: SymmetricRangeType
        shift_limit: SymmetricRangeType

    def __init__(
        self,
        distort_limit: ScaleFloatType = (-0.05, 0.05),
        shift_limit: ScaleFloatType = (-0.05, 0.05),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: ColorType | None = None,
        mask_value: ColorType | None = None,
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(
            interpolation=interpolation,
            border_mode=border_mode,
            value=value,
            mask_value=mask_value,
            always_apply=always_apply,
            mask_interpolation=mask_interpolation,
            p=p,
        )
        self.shift_limit = cast(tuple[float, float], shift_limit)
        self.distort_limit = cast(tuple[float, float], distort_limit)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]

        fx = width
        fy = height

        k = self.py_random.uniform(*self.distort_limit)
        dx = round(self.py_random.uniform(*self.shift_limit))
        dy = round(self.py_random.uniform(*self.shift_limit))

        cx = width * 0.5 + dx
        cy = height * 0.5 + dy

        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
        map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)

        return {"map_x": map_x, "map_y": map_y}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (*super().get_transform_init_args_names(), "distort_limit", "shift_limit")


class GridDistortion(BaseDistortion):
    """Apply grid distortion to images, masks, bounding boxes, and keypoints.

    This transformation divides the image into a grid and randomly distorts each cell,
    creating localized warping effects. It's particularly useful for data augmentation
    in tasks like medical image analysis, OCR, and other domains where local geometric
    variations are meaningful.

    Args:
        num_steps (int): Number of grid cells on each side of the image. Higher values
            create more granular distortions. Must be at least 1. Default: 5.
        distort_limit (float or tuple[float, float]): Range of distortion. If a single float
            is provided, the range will be (-distort_limit, distort_limit). Higher values
            create stronger distortions. Should be in the range of -1 to 1.
            Default: (-0.3, 0.3).
        interpolation (int): OpenCV interpolation method used for image transformation.
            Options include cv2.INTER_LINEAR, cv2.INTER_CUBIC, etc. Default: cv2.INTER_LINEAR.
        border_mode (int): OpenCV border mode used for handling pixels outside the image.
            Options include cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT, etc.
            Default: cv2.BORDER_REFLECT_101.
        value (int, float, list of int, list of float, optional): Padding value if
            border_mode is cv2.BORDER_CONSTANT. Default: None.
        mask_value (int, float, list of int, list of float, optional): Padding value for
            mask if border_mode is cv2.BORDER_CONSTANT. Default: None.
        normalized (bool): If True, ensures that the distortion does not move pixels
            outside the image boundaries. This can result in less extreme distortions
            but guarantees that no information is lost. Default: True.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The same distortion is applied to all targets (image, mask, bboxes, keypoints)
          to maintain consistency.
        - When normalized=True, the distortion is adjusted to ensure all pixels remain
          within the image boundaries.

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ... ])
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>> transformed_image = transformed['image']
        >>> transformed_mask = transformed['mask']
        >>> transformed_bboxes = transformed['bboxes']
        >>> transformed_keypoints = transformed['keypoints']

    """

    class InitSchema(BaseDistortion.InitSchema):
        num_steps: Annotated[int, Field(ge=1)]
        distort_limit: SymmetricRangeType
        normalized: bool

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
        normalized: bool = True,
        mask_interpolation: int = cv2.INTER_NEAREST,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(
            interpolation=interpolation,
            border_mode=border_mode,
            value=value,
            mask_value=mask_value,
            always_apply=always_apply,
            mask_interpolation=mask_interpolation,
            p=p,
        )
        self.num_steps = num_steps
        self.distort_limit = cast(tuple[float, float], distort_limit)
        self.normalized = normalized

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"][:2]
        steps_x = [1 + self.py_random.uniform(*self.distort_limit) for _ in range(self.num_steps + 1)]
        steps_y = [1 + self.py_random.uniform(*self.distort_limit) for _ in range(self.num_steps + 1)]

        if self.normalized:
            normalized_params = fgeometric.normalize_grid_distortion_steps(
                image_shape,
                self.num_steps,
                steps_x,
                steps_y,
            )
            steps_x, steps_y = normalized_params["steps_x"], normalized_params["steps_y"]

        map_x, map_y = fgeometric.generate_grid(image_shape, steps_x, steps_y, self.num_steps)

        return {"map_x": map_x, "map_y": map_y}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (*super().get_transform_init_args_names(), "num_steps", "distort_limit", "normalized")


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
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - This transform is particularly useful for augmenting data that does not have a clear orientation,
          such as top-view satellite or drone imagery, or certain types of medical images.
        - The input image should be square-shaped for optimal results. Non-square inputs may lead to
          unexpected behavior or distortions.
        - When applied to bounding boxes or keypoints, their coordinates will be adjusted according
          to the selected transformation.
        - This transform preserves the aspect ratio and size of the input.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.D4(p=1.0),
        ... ])
        >>> transformed = transform(image=image)
        >>> transformed_image = transformed['image']
        # The resulting image will be one of the 8 possible D4 transformations of the input
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(
        self,
        always_apply: bool | None = None,
        p: float = 1,
    ):
        super().__init__(p=p, always_apply=always_apply)

    def apply(self, img: np.ndarray, group_element: D4Type, **params: Any) -> np.ndarray:
        return fgeometric.d4(img, group_element)

    def apply_to_bboxes(self, bboxes: np.ndarray, group_element: D4Type, **params: Any) -> np.ndarray:
        return fgeometric.bboxes_d4(bboxes, group_element)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        group_element: D4Type,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.keypoints_d4(keypoints, group_element, params["shape"])

    def get_params(self) -> dict[str, D4Type]:
        return {
            "group_element": self.random_generator.choice(d4_group_elements),
        }

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()


class GridElasticDeform(DualTransform):
    """Apply elastic deformations to images, masks, bounding boxes, and keypoints using a grid-based approach.

    This transformation overlays a grid on the input and applies random displacements to the grid points,
    resulting in local elastic distortions. The granularity and intensity of the distortions can be
    controlled using the dimensions of the overlaying distortion grid and the magnitude parameter.


    Args:
        num_grid_xy (tuple[int, int]): Number of grid cells along the width and height.
            Specified as (grid_width, grid_height). Each value must be greater than 1.
        magnitude (int): Maximum pixel-wise displacement for distortion. Must be greater than 0.
        interpolation (int): Interpolation method to be used for the image transformation.
            Default: cv2.INTER_LINEAR
        mask_interpolation (int): Interpolation method to be used for mask transformation.
            Default: cv2.INTER_NEAREST
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Example:
        >>> transform = GridElasticDeform(num_grid_xy=(4, 4), magnitude=10, p=1.0)
        >>> result = transform(image=image, mask=mask)
        >>> transformed_image, transformed_mask = result['image'], result['mask']

    Note:
        This transformation is particularly useful for data augmentation in medical imaging
        and other domains where elastic deformations can simulate realistic variations.
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        num_grid_xy: Annotated[tuple[int, int], AfterValidator(check_1plus)]
        magnitude: int = Field(gt=0)
        interpolation: InterpolationType
        mask_interpolation: InterpolationType

    def __init__(
        self,
        num_grid_xy: tuple[int, int],
        magnitude: int,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.num_grid_xy = num_grid_xy
        self.magnitude = magnitude
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    @staticmethod
    def generate_mesh(polygons: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
        return np.hstack((dimensions.reshape(-1, 4), polygons))

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        image_shape = params["shape"][:2]

        # Replace calculate_grid_dimensions with split_uniform_grid
        tiles = fgeometric.split_uniform_grid(image_shape, self.num_grid_xy, self.random_generator)

        # Convert tiles to the format expected by generate_distorted_grid_polygons
        dimensions = np.array(
            [
                [tile[1], tile[0], tile[3], tile[2]]  # Reorder to [x_min, y_min, x_max, y_max]
                for tile in tiles
            ],
        ).reshape(self.num_grid_xy[::-1] + (4,))  # Reshape to (grid_height, grid_width, 4)

        polygons = fgeometric.generate_distorted_grid_polygons(dimensions, self.magnitude, self.random_generator)

        generated_mesh = self.generate_mesh(polygons, dimensions)

        return {"generated_mesh": generated_mesh}

    def apply(self, img: np.ndarray, generated_mesh: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.distort_image(img, generated_mesh, self.interpolation)

    def apply_to_mask(self, mask: np.ndarray, generated_mesh: np.ndarray, **params: Any) -> np.ndarray:
        return fgeometric.distort_image(mask, generated_mesh, self.mask_interpolation)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        generated_mesh: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        bboxes_denorm = denormalize_bboxes(bboxes, params["shape"][:2])
        return normalize_bboxes(
            fgeometric.bbox_distort_image(
                bboxes_denorm,
                generated_mesh,
                params["shape"][:2],
            ),
            params["shape"][:2],
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        generated_mesh: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        return fgeometric.distort_image_keypoints(keypoints, generated_mesh, params["shape"][:2])

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "num_grid_xy", "magnitude", "interpolation", "mask_interpolation"
