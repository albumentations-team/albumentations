"""Geometric transformation classes for image augmentation.

This module provides a collection of transforms that modify the geometric properties
of images and associated data (masks, bounding boxes, keypoints). Includes implementations
for flipping, transposing, affine transformations, distortions, padding, and more complex
transformations like grid shuffling and thin plate splines.
"""

from __future__ import annotations

import random
from typing import Annotated, Any, Literal, cast
from warnings import warn

import cv2
import numpy as np
from albucore import batch_transform, is_grayscale_image, is_rgb_image
from pydantic import (
    AfterValidator,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from albumentations.augmentations.utils import check_range
from albumentations.core.bbox_utils import (
    BboxProcessor,
    denormalize_bboxes,
    normalize_bboxes,
)
from albumentations.core.pydantic import (
    NonNegativeFloatRangeType,
    OnePlusIntRangeType,
    SymmetricRangeType,
    check_range_bounds,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import ALL_TARGETS
from albumentations.core.utils import to_tuple

from . import functional as fgeometric

__all__ = [
    "Affine",
    "GridElasticDeform",
    "Morphological",
    "Perspective",
    "RandomGridShuffle",
    "ShiftScaleRotate",
]

NUM_PADS_XY = 2
NUM_PADS_ALL_SIDES = 4


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
        border_mode (OpenCV flag): OpenCV border mode used for padding.
            Default: cv2.BORDER_CONSTANT.
        fill (tuple[float, ...] | float): Padding value if border_mode is cv2.BORDER_CONSTANT.
            Default: 0.
        fill_mask (tuple[float, ...] | float): Padding value for mask if border_mode is
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
        image, mask, keypoints, bboxes, volume, mask3d

    Image types:
        uint8, float32

    Note:
        This transformation creates a perspective effect by randomly moving the four corners of the image.
        The amount of movement is controlled by the 'scale' parameter.

        When 'keep_size' is True, the output image will have the same size as the input image,
        which may cause some parts of the transformed image to be cut off or padded.

        When 'fit_output' is True, the transformation ensures that the entire transformed image is visible,
        which may result in a larger output image if keep_size is False.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with parameters as tuples when possible
        >>> transform = A.Compose([
        ...     A.Perspective(
        ...         scale=(0.05, 0.1),
        ...         keep_size=True,
        ...         fit_output=False,
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = transformed['image']      # Perspective-transformed image
        >>> transformed_mask = transformed['mask']        # Perspective-transformed mask
        >>> transformed_bboxes = transformed['bboxes']    # Perspective-transformed bounding boxes
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for transformed bboxes
        >>> transformed_keypoints = transformed['keypoints']  # Perspective-transformed keypoints
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for transformed keypoints

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        scale: NonNegativeFloatRangeType
        keep_size: bool
        fit_output: bool
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

    def __init__(
        self,
        scale: tuple[float, float] | float = (0.05, 0.1),
        keep_size: bool = True,
        fit_output: bool = False,
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_NEAREST,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.scale = cast("tuple[float, float]", scale)
        self.keep_size = keep_size
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask
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
        """Apply the perspective transform to an image.

        Args:
            img (np.ndarray): Image to be distorted.
            matrix (np.ndarray): Transformation matrix.
            max_height (int): Maximum height of the image.
            max_width (int): Maximum width of the image.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted image.

        """
        return fgeometric.perspective(
            img,
            matrix,
            max_width,
            max_height,
            self.fill,
            self.border_mode,
            self.keep_size,
            self.interpolation,
        )

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the perspective transform to a batch of images.

        Args:
            images (np.ndarray): Batch of images to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Batch of distorted images.

        """
        return self.apply(images, **params)

    @batch_transform("spatial", has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the perspective transform to a volume.

        Args:
            volume (np.ndarray): Volume to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted volume.

        """
        return self.apply(volume, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the perspective transform to a batch of volumes.

        Args:
            volumes (np.ndarray): Batch of volumes to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Batch of distorted volumes.

        """
        return self.apply(volumes, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_mask3d(self, mask3d: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the perspective transform to a 3D mask.

        Args:
            mask3d (np.ndarray): 3D mask to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted 3D mask.

        """
        return self.apply_to_mask(mask3d, **params)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        matrix: np.ndarray,
        max_height: int,
        max_width: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the perspective transform to a mask.

        Args:
            mask (np.ndarray): Mask to be distorted.
            matrix (np.ndarray): Transformation matrix.
            max_height (int): Maximum height of the mask.
            max_width (int): Maximum width of the mask.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted mask.

        """
        return fgeometric.perspective(
            mask,
            matrix,
            max_width,
            max_height,
            self.fill_mask,
            self.border_mode,
            self.keep_size,
            self.mask_interpolation,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        matrix_bbox: np.ndarray,
        max_height: int,
        max_width: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the perspective transform to a batch of bounding boxes.

        Args:
            bboxes (np.ndarray): Batch of bounding boxes to be distorted.
            matrix_bbox (np.ndarray): Transformation matrix.
            max_height (int): Maximum height of the bounding boxes.
            max_width (int): Maximum width of the bounding boxes.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Batch of distorted bounding boxes.

        """
        return fgeometric.perspective_bboxes(
            bboxes,
            params["shape"],
            matrix_bbox,
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
        """Apply the perspective transform to a batch of keypoints.

        Args:
            keypoints (np.ndarray): Batch of keypoints to be distorted.
            matrix (np.ndarray): Transformation matrix.
            max_height (int): Maximum height of the keypoints.
            max_width (int): Maximum width of the keypoints.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Batch of distorted keypoints.

        """
        return fgeometric.perspective_keypoints(
            keypoints,
            params["shape"],
            matrix,
            max_width,
            max_height,
            self.keep_size,
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): Parameters.
            data (dict[str, Any]): Data.

        Returns:
            dict[str, Any]: Parameters.

        """
        image_shape = params["shape"][:2]
        scale = self.py_random.uniform(*self.scale)

        points = fgeometric.generate_perspective_points(
            image_shape,
            scale,
            self.random_generator,
        )
        points = fgeometric.order_points(points)

        matrix, max_width, max_height = fgeometric.compute_perspective_params(
            points,
            image_shape,
        )

        if self.fit_output:
            matrix, max_width, max_height = fgeometric.expand_transform(
                matrix,
                image_shape,
            )

        return {
            "matrix": matrix,
            "max_height": max_height,
            "max_width": max_width,
            "matrix_bbox": matrix,
        }


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
    The parameters `fill` and `fill_mask` of this class deal with this.

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
        fill (tuple[float, ...] | float): The constant value to use when filling in newly created pixels.
            (E.g. translating by 1px to the right will create a new 1px-wide column of pixels
            on the left of the image).
            The value is only used when `mode=constant`. The expected value range is ``[0, 255]`` for ``uint8`` images.
        fill_mask (tuple[float, ...] | float): Same as fill but only for masks.
        border_mode (int): OpenCV border flag.
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
        image, mask, keypoints, bboxes, volume, mask3d

    Image types:
        uint8, float32

    References:
        Towards Rotation Invariance in Object Detection: https://arxiv.org/abs/2109.13488

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with different parameter types
        >>> transform = A.Compose([
        ...     A.Affine(
        ...         # Tuple for scale (will be used for both x and y)
        ...         scale=(0.8, 1.2),
        ...         # Dictionary with tuples for different x/y translations
        ...         translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)},
        ...         # Tuple for rotation range
        ...         rotate=(-30, 30),
        ...         # Dictionary with tuples for different x/y shearing
        ...         shear={"x": (-10, 10), "y": (-5, 5)},
        ...         # Interpolation methods
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         # Other parameters
        ...         fit_output=False,
        ...         keep_ratio=True,
        ...         rotate_method="largest_box",
        ...         balanced_scale=True,
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         fill=0,
        ...         fill_mask=0,
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = transformed['image']      # Image with affine transforms applied
        >>> transformed_mask = transformed['mask']        # Mask with affine transforms applied
        >>> transformed_bboxes = transformed['bboxes']    # Bounding boxes with affine transforms applied
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for transformed bboxes
        >>> transformed_keypoints = transformed['keypoints']  # Keypoints with affine transforms applied
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for transformed keypoints
        >>>
        >>> # Simpler example with only essential parameters
        >>> simple_transform = A.Compose([
        ...     A.Affine(
        ...         scale=1.1,  # Single scalar value for scale
        ...         rotate=15,  # Single scalar value for rotation (degrees)
        ...         translate_px=30,  # Single scalar value for translation (pixels)
        ...         p=1.0
        ...     ),
        ... ])
        >>> simple_result = simple_transform(image=image)
        >>> simple_transformed = simple_result['image']

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        scale: tuple[float, float] | float | dict[str, float | tuple[float, float]]
        translate_percent: tuple[float, float] | float | dict[str, float | tuple[float, float]] | None
        translate_px: tuple[float, float] | float | dict[str, float | tuple[float, float]] | None
        rotate: tuple[float, float] | float
        shear: tuple[float, float] | float | dict[str, float | tuple[float, float]]
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]

        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

        fit_output: bool
        keep_ratio: bool
        rotate_method: Literal["largest_box", "ellipse"]
        balanced_scale: bool

        @field_validator("shear", "scale")
        @classmethod
        def _process_shear(
            cls,
            value: tuple[float, float] | float | dict[str, float | tuple[float, float]],
            info: ValidationInfo,
        ) -> dict[str, tuple[float, float]]:
            return cls._handle_dict_arg(value, info.field_name)

        @field_validator("rotate")
        @classmethod
        def _process_rotate(
            cls,
            value: tuple[float, float] | float,
        ) -> tuple[float, float]:
            return to_tuple(value, value)

        @model_validator(mode="after")
        def _handle_translate(self) -> Self:
            if self.translate_percent is None and self.translate_px is None:
                self.translate_px = 0

            if self.translate_percent is not None and self.translate_px is not None:
                msg = "Expected either translate_percent or translate_px to be provided, but both were provided."
                raise ValueError(msg)

            if self.translate_percent is not None:
                self.translate_percent = self._handle_dict_arg(
                    self.translate_percent,
                    "translate_percent",
                    default=0.0,
                )  # type: ignore[assignment]

            if self.translate_px is not None:
                self.translate_px = self._handle_dict_arg(
                    self.translate_px,
                    "translate_px",
                    default=0,
                )  # type: ignore[assignment]

            return self

        @staticmethod
        def _handle_dict_arg(
            val: tuple[float, float]
            | dict[str, float | tuple[float, float]]
            | float
            | tuple[int, int]
            | dict[str, int | tuple[int, int]],
            name: str | None,
            default: float = 1.0,
        ) -> dict[str, tuple[float, float]]:
            if isinstance(val, float):
                return {"x": (val, val), "y": (val, val)}
            if isinstance(val, dict):
                if "x" not in val and "y" not in val:
                    raise ValueError(
                        f'Expected {name} dictionary to contain at least key "x" or key "y". Found neither of them.',
                    )
                x = val.get("x", default)
                y = val.get("y", default)
                return {"x": to_tuple(x, x), "y": to_tuple(y, y)}
            return {"x": to_tuple(val, val), "y": to_tuple(val, val)}

    def __init__(
        self,
        scale: tuple[float, float] | float | dict[str, float | tuple[float, float]] = (1.0, 1.0),
        translate_percent: tuple[float, float] | float | dict[str, float | tuple[float, float]] | None = None,
        translate_px: tuple[int, int] | int | dict[str, int | tuple[int, int]] | None = None,
        rotate: tuple[float, float] | float = 0.0,
        shear: tuple[float, float] | float | dict[str, float | tuple[float, float]] = (0.0, 0.0),
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_NEAREST,
        fit_output: bool = False,
        keep_ratio: bool = False,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        balanced_scale: bool = False,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.fill = fill
        self.fill_mask = fill_mask
        self.border_mode = border_mode
        self.scale = cast("dict[str, tuple[float, float]]", scale)
        self.translate_percent = cast("dict[str, tuple[float, float]]", translate_percent)
        self.translate_px = cast("dict[str, tuple[int, int]]", translate_px)
        self.rotate = cast("tuple[float, float]", rotate)
        self.fit_output = fit_output
        self.shear = cast("dict[str, tuple[float, float]]", shear)
        self.keep_ratio = keep_ratio
        self.rotate_method = rotate_method
        self.balanced_scale = balanced_scale

        if self.keep_ratio and self.scale["x"] != self.scale["y"]:
            raise ValueError(
                f"When keep_ratio is True, the x and y scale range should be identical. got {self.scale}",
            )

    def apply(
        self,
        img: np.ndarray,
        matrix: np.ndarray,
        output_shape: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the affine transform to an image.

        Args:
            img (np.ndarray): Image to be distorted.
            matrix (np.ndarray): Transformation matrix.
            output_shape (tuple[int, int]): Output shape.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted image.

        """
        return fgeometric.warp_affine(
            img,
            matrix,
            interpolation=self.interpolation,
            fill=self.fill,
            border_mode=self.border_mode,
            output_shape=output_shape,
        )

    def apply_to_mask(
        self,
        mask: np.ndarray,
        matrix: np.ndarray,
        output_shape: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the affine transform to a mask.

        Args:
            mask (np.ndarray): Mask to be distorted.
            matrix (np.ndarray): Transformation matrix.
            output_shape (tuple[int, int]): Output shape.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted mask.

        """
        return fgeometric.warp_affine(
            mask,
            matrix,
            interpolation=self.mask_interpolation,
            fill=self.fill_mask,
            border_mode=self.border_mode,
            output_shape=output_shape,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        bbox_matrix: np.ndarray,
        output_shape: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the affine transform to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to be distorted.
            bbox_matrix (np.ndarray): Transformation matrix.
            output_shape (tuple[int, int]): Output shape.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted bounding boxes.

        """
        return fgeometric.bboxes_affine(
            bboxes,
            bbox_matrix,
            self.rotate_method,
            params["shape"][:2],
            self.border_mode,
            output_shape,
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        matrix: np.ndarray,
        scale: dict[str, float],
        **params: Any,
    ) -> np.ndarray:
        """Apply the affine transform to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to be distorted.
            matrix (np.ndarray): Transformation matrix.
            scale (dict[str, float]): Scale.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted keypoints.

        """
        return fgeometric.keypoints_affine(
            keypoints,
            matrix,
            params["shape"],
            scale,
            self.border_mode,
        )

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the affine transform to a batch of images.

        Args:
            images (np.ndarray): Images to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted images.

        """
        return self.apply(images, **params)

    @batch_transform("spatial", has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the affine transform to a volume.

        Args:
            volume (np.ndarray): Volume to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted volume.

        """
        return self.apply(volume, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the affine transform to a batch of volumes.

        Args:
            volumes (np.ndarray): Volumes to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted volumes.

        """
        return self.apply(volumes, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_mask3d(self, mask3d: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the affine transform to a 3D mask.

        Args:
            mask3d (np.ndarray): 3D mask to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted 3D mask.

        """
        return self.apply_to_mask(mask3d, **params)

    @staticmethod
    def _get_scale(
        scale: dict[str, tuple[float, float]],
        keep_ratio: bool,
        balanced_scale: bool,
        random_state: random.Random,
    ) -> dict[str, float]:
        result_scale = {}
        for key, value in scale.items():
            if isinstance(value, (int, float)):
                result_scale[key] = float(value)
            elif isinstance(value, tuple):
                if balanced_scale:
                    lower_interval = (value[0], 1.0) if value[0] < 1 else None
                    upper_interval = (1.0, value[1]) if value[1] > 1 else None

                    if lower_interval is not None and upper_interval is not None:
                        selected_interval = random_state.choice(
                            [lower_interval, upper_interval],
                        )
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

        return result_scale

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): Parameters.
            data (dict[str, Any]): Data.

        Returns:
            dict[str, Any]: Parameters.

        """
        image_shape = params["shape"][:2]

        translate = self._get_translate_params(image_shape)
        shear = self._get_shear_params()
        scale = self._get_scale(
            self.scale,
            self.keep_ratio,
            self.balanced_scale,
            self.py_random,
        )
        rotate = self.py_random.uniform(*self.rotate)

        image_shift = fgeometric.center(image_shape)
        bbox_shift = fgeometric.center_bbox(image_shape)

        matrix = fgeometric.create_affine_transformation_matrix(
            translate,
            shear,
            scale,
            rotate,
            image_shift,
        )
        bbox_matrix = fgeometric.create_affine_transformation_matrix(
            translate,
            shear,
            scale,
            rotate,
            bbox_shift,
        )

        if self.fit_output:
            matrix, output_shape = fgeometric.compute_affine_warp_output_shape(
                matrix,
                image_shape,
            )
            bbox_matrix, _ = fgeometric.compute_affine_warp_output_shape(
                bbox_matrix,
                image_shape,
            )
        else:
            output_shape = image_shape

        return {
            "rotate": rotate,
            "scale": scale,
            "matrix": matrix,
            "bbox_matrix": bbox_matrix,
            "output_shape": output_shape,
        }

    def _get_translate_params(self, image_shape: tuple[int, int]) -> dict[str, int]:
        height, width = image_shape[:2]
        if self.translate_px is not None:
            return {
                "x": self.py_random.randint(int(self.translate_px["x"][0]), int(self.translate_px["x"][1])),
                "y": self.py_random.randint(int(self.translate_px["y"][0]), int(self.translate_px["y"][1])),
            }
        if self.translate_percent is not None:
            translate = {key: self.py_random.uniform(*value) for key, value in self.translate_percent.items()}
            return cast(
                "dict[str, int]",
                {"x": int(translate["x"] * width), "y": int(translate["y"] * height)},
            )
        return cast("dict[str, int]", {"x": 0, "y": 0})

    def _get_shear_params(self) -> dict[str, float]:
        return {
            "x": -self.py_random.uniform(*self.shear["x"]),
            "y": -self.py_random.uniform(*self.shear["y"]),
        }


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
            Default: cv2.BORDER_CONSTANT
        fill (tuple[float, ...] | float): padding value if border_mode is cv2.BORDER_CONSTANT.
        fill_mask (tuple[float, ...] | float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
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
        image, mask, keypoints, bboxes, volume, mask3d

    Image types:
        uint8, float32

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with parameters as tuples when possible
        >>> transform = A.Compose([
        ...     A.ShiftScaleRotate(
        ...         shift_limit=(-0.0625, 0.0625),
        ...         scale_limit=(-0.1, 0.1),
        ...         rotate_limit=(-45, 45),
        ...         interpolation=cv2.INTER_LINEAR,
        ...         border_mode=cv2.BORDER_CONSTANT,
        ...         rotate_method="largest_box",
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = transformed['image']      # Shifted, scaled and rotated image
        >>> transformed_mask = transformed['mask']        # Shifted, scaled and rotated mask
        >>> transformed_bboxes = transformed['bboxes']    # Shifted, scaled and rotated bounding boxes
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for transformed bboxes
        >>> transformed_keypoints = transformed['keypoints']  # Shifted, scaled and rotated keypoints
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for transformed keypoints

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        shift_limit: SymmetricRangeType
        scale_limit: SymmetricRangeType
        rotate_limit: SymmetricRangeType
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]

        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

        shift_limit_x: tuple[float, float] | float | None
        shift_limit_y: tuple[float, float] | float | None
        rotate_method: Literal["largest_box", "ellipse"]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]

        @model_validator(mode="after")
        def _check_shift_limit(self) -> Self:
            bounds = -1, 1
            self.shift_limit_x = to_tuple(
                self.shift_limit_x if self.shift_limit_x is not None else self.shift_limit,
            )
            check_range(self.shift_limit_x, *bounds, "shift_limit_x")
            self.shift_limit_y = to_tuple(
                self.shift_limit_y if self.shift_limit_y is not None else self.shift_limit,
            )
            check_range(self.shift_limit_y, *bounds, "shift_limit_y")

            return self

        @field_validator("scale_limit")
        @classmethod
        def _check_scale_limit(
            cls,
            value: tuple[float, float] | float,
            info: ValidationInfo,
        ) -> tuple[float, float]:
            bounds = 0, float("inf")
            result = to_tuple(value, bias=1.0)
            check_range(result, *bounds, str(info.field_name))
            return result

    def __init__(
        self,
        shift_limit: tuple[float, float] | float = (-0.0625, 0.0625),
        scale_limit: tuple[float, float] | float = (-0.1, 0.1),
        rotate_limit: tuple[float, float] | float = (-45, 45),
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        shift_limit_x: tuple[float, float] | float | None = None,
        shift_limit_y: tuple[float, float] | float | None = None,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_NEAREST,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 0.5,
    ):
        shift_limit_x = cast("tuple[float, float]", shift_limit_x)
        shift_limit_y = cast("tuple[float, float]", shift_limit_y)
        super().__init__(
            scale=scale_limit,
            translate_percent={"x": shift_limit_x, "y": shift_limit_y},
            rotate=rotate_limit,
            shear=(0, 0),
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            fill=fill,
            fill_mask=fill_mask,
            border_mode=border_mode,
            fit_output=False,
            keep_ratio=False,
            rotate_method=rotate_method,
            p=p,
        )
        warn(
            "ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.",
            UserWarning,
            stacklevel=2,
        )
        self.shift_limit_x = shift_limit_x
        self.shift_limit_y = shift_limit_y

        self.scale_limit = cast("tuple[float, float]", scale_limit)
        self.rotate_limit = cast("tuple[int, int]", rotate_limit)
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask

    def get_transform_init_args(self) -> dict[str, Any]:
        """Get the transform initialization arguments.

        Returns:
            dict[str, Any]: Transform initialization arguments.

        """
        return {
            "shift_limit_x": self.shift_limit_x,
            "shift_limit_y": self.shift_limit_y,
            "scale_limit": to_tuple(self.scale_limit, bias=-1.0),
            "rotate_limit": self.rotate_limit,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
            "fill": self.fill,
            "fill_mask": self.fill_mask,
            "rotate_method": self.rotate_method,
            "mask_interpolation": self.mask_interpolation,
        }


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
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Number of channels:
        1, 3

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with parameters as tuples when possible
        >>> transform = A.Compose([
        ...     A.GridElasticDeform(
        ...         num_grid_xy=(4, 4),
        ...         magnitude=10,
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         p=1.0
        ...     ),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = transformed['image']      # Elastically deformed image
        >>> transformed_mask = transformed['mask']        # Elastically deformed mask
        >>> transformed_bboxes = transformed['bboxes']    # Elastically deformed bounding boxes
        >>> transformed_bbox_labels = transformed['bbox_labels']  # Labels for transformed bboxes
        >>> transformed_keypoints = transformed['keypoints']  # Elastically deformed keypoints
        >>> transformed_keypoint_labels = transformed['keypoint_labels']  # Labels for transformed keypoints

    Note:
        This transformation is particularly useful for data augmentation in medical imaging
        and other domains where elastic deformations can simulate realistic variations.

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        num_grid_xy: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]
        magnitude: int = Field(gt=0)
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]

    def __init__(
        self,
        num_grid_xy: tuple[int, int],
        magnitude: int,
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_NEAREST,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.num_grid_xy = num_grid_xy
        self.magnitude = magnitude
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation

    @staticmethod
    def _generate_mesh(polygons: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
        return np.hstack((dimensions.reshape(-1, 4), polygons))

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): Parameters.
            data (dict[str, Any]): Data.

        Returns:
            dict[str, Any]: Parameters.

        """
        image_shape = params["shape"][:2]

        # Replace calculate_grid_dimensions with split_uniform_grid
        tiles = fgeometric.split_uniform_grid(
            image_shape,
            self.num_grid_xy,
            self.random_generator,
        )

        # Convert tiles to the format expected by generate_distorted_grid_polygons
        dimensions = np.array(
            [
                [
                    tile[1],
                    tile[0],
                    tile[3],
                    tile[2],
                ]  # Reorder to [x_min, y_min, x_max, y_max]
                for tile in tiles
            ],
        ).reshape(
            self.num_grid_xy[::-1] + (4,),
        )  # Reshape to (grid_height, grid_width, 4)

        polygons = fgeometric.generate_distorted_grid_polygons(
            dimensions,
            self.magnitude,
            self.random_generator,
        )

        generated_mesh = self._generate_mesh(polygons, dimensions)

        return {"generated_mesh": generated_mesh}

    def apply(
        self,
        img: np.ndarray,
        generated_mesh: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the GridElasticDeform transform to an image.

        Args:
            img (np.ndarray): Image to be transformed.
            generated_mesh (np.ndarray): Generated mesh.
            **params (Any): Additional parameters.

        """
        if not is_rgb_image(img) and not is_grayscale_image(img):
            raise ValueError("GridElasticDeform transform is only supported for RGB and grayscale images.")
        return fgeometric.distort_image(img, generated_mesh, self.interpolation)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        generated_mesh: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the GridElasticDeform transform to a mask.

        Args:
            mask (np.ndarray): Mask to be transformed.
            generated_mesh (np.ndarray): Generated mesh.
            **params (Any): Additional parameters.

        """
        return fgeometric.distort_image(mask, generated_mesh, self.mask_interpolation)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        generated_mesh: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the GridElasticDeform transform to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to be transformed.
            generated_mesh (np.ndarray): Generated mesh.
            **params (Any): Additional parameters.

        """
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
        """Apply the GridElasticDeform transform to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to be transformed.
            generated_mesh (np.ndarray): Generated mesh.
            **params (Any): Additional parameters.

        """
        return fgeometric.distort_image_keypoints(
            keypoints,
            generated_mesh,
            params["shape"][:2],
        )


class RandomGridShuffle(DualTransform):
    """Randomly shuffles the grid's cells on an image, mask, or keypoints,
    effectively rearranging patches within the image.
    This transformation divides the image into a grid and then permutes these grid cells based on a random mapping.

    Args:
        grid (tuple[int, int]): Size of the grid for splitting the image into cells. Each cell is shuffled randomly.
            For example, (3, 3) will divide the image into a 3x3 grid, resulting in 9 cells to be shuffled.
            Default: (3, 3)
        p (float): Probability that the transform will be applied. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image, mask, keypoints, bboxes, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - This transform maintains consistency across all targets. If applied to an image and its corresponding
          mask or keypoints, the same shuffling will be applied to all.
        - The number of cells in the grid should be at least 2 (i.e., grid should be at least (1, 2), (2, 1), or (2, 2))
          for the transform to have any effect.
        - Keypoints are moved along with their corresponding grid cell.
        - This transform could be useful when only micro features are important for the model, and memorizing
          the global structure could be harmful. For example:
          - Identifying the type of cell phone used to take a picture based on micro artifacts generated by
            phone post-processing algorithms, rather than the semantic features of the photo.
            See more at https://ieeexplore.ieee.org/abstract/document/8622031
          - Identifying stress, glucose, hydration levels based on skin images.

    Mathematical Formulation:
        1. The image is divided into a grid of size (m, n) as specified by the 'grid' parameter.
        2. A random permutation P of integers from 0 to (m*n - 1) is generated.
        3. Each cell in the grid is assigned a number from 0 to (m*n - 1) in row-major order.
        4. The cells are then rearranged according to the permutation P.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with grid as a tuple
        >>> transform = A.Compose([
        ...     A.RandomGridShuffle(grid=(3, 3), p=1.0),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Get the transformed data
        >>> transformed_image = transformed['image']     # Grid-shuffled image
        >>> transformed_mask = transformed['mask']       # Grid-shuffled mask
        >>> transformed_bboxes = transformed['bboxes']   # Grid-shuffled bounding boxes
        >>> transformed_keypoints = transformed['keypoints']  # Grid-shuffled keypoints
        >>>
        >>> # Visualization example with a simpler grid
        >>> simple_image = np.array([
        ...     [1, 1, 1, 2, 2, 2],
        ...     [1, 1, 1, 2, 2, 2],
        ...     [1, 1, 1, 2, 2, 2],
        ...     [3, 3, 3, 4, 4, 4],
        ...     [3, 3, 3, 4, 4, 4],
        ...     [3, 3, 3, 4, 4, 4]
        ... ])
        >>> simple_transform = A.RandomGridShuffle(grid=(2, 2), p=1.0)
        >>> simple_result = simple_transform(image=simple_image)
        >>> simple_transformed = simple_result['image']
        >>> # The result could look like:
        >>> # array([[4, 4, 4, 2, 2, 2],
        >>> #        [4, 4, 4, 2, 2, 2],
        >>> #        [4, 4, 4, 2, 2, 2],
        >>> #        [3, 3, 3, 1, 1, 1],
        >>> #        [3, 3, 3, 1, 1, 1],
        >>> #        [3, 3, 3, 1, 1, 1]])

    """

    class InitSchema(BaseTransformInitSchema):
        grid: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]

    _targets = ALL_TARGETS

    def __init__(
        self,
        grid: tuple[int, int] = (3, 3),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.grid = grid

    def apply(
        self,
        img: np.ndarray,
        tiles: np.ndarray,
        mapping: list[int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the RandomGridShuffle transform to an image.

        Args:
            img (np.ndarray): Image to be transformed.
            tiles (np.ndarray): Tiles to be transformed.
            mapping (list[int]): Mapping of the tiles.
            **params (Any): Additional parameters.

        """
        return fgeometric.swap_tiles_on_image(img, tiles, mapping)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        tiles: np.ndarray,
        mapping: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the RandomGridShuffle transform to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to be transformed.
            tiles (np.ndarray): Tiles to be transformed.
            mapping (np.ndarray): Mapping of the tiles.
            **params (Any): Additional parameters.

        """
        image_shape = params["shape"][:2]
        bboxes_denorm = denormalize_bboxes(bboxes, image_shape)
        processor = cast("BboxProcessor", self.get_processor("bboxes"))
        if processor is None:
            return bboxes
        bboxes_returned = fgeometric.bboxes_grid_shuffle(
            bboxes_denorm,
            tiles,
            mapping,
            image_shape,
            min_area=processor.params.min_area,
            min_visibility=processor.params.min_visibility,
        )
        return normalize_bboxes(bboxes_returned, image_shape)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        tiles: np.ndarray,
        mapping: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the RandomGridShuffle transform to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to be transformed.
            tiles (np.ndarray): Tiles to be transformed.
            mapping (np.ndarray): Mapping of the tiles.
            **params (Any): Additional parameters.

        """
        return fgeometric.swap_tiles_on_keypoints(keypoints, tiles, mapping)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the RandomGridShuffle transform to a batch of images.

        Args:
            images (np.ndarray): Images to be transformed.
            **params (Any): Additional parameters.

        """
        return self.apply(images, **params)

    @batch_transform("spatial", has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the RandomGridShuffle transform to a volume.

        Args:
            volume (np.ndarray): Volume to be transformed.
            **params (Any): Additional parameters.

        """
        return self.apply(volume, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the RandomGridShuffle transform to a batch of volumes.

        Args:
            volumes (np.ndarray): Volumes to be transformed.
            **params (Any): Additional parameters.

        """
        return self.apply(volumes, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_mask3d(self, mask3d: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the RandomGridShuffle transform to a 3D mask.

        Args:
            mask3d (np.ndarray): 3D mask to be transformed.
            **params (Any): Additional parameters.

        """
        return self.apply(mask3d, **params)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Get the parameters dependent on the data.

        Args:
            params (dict[str, Any]): Parameters.
            data (dict[str, Any]): Data.

        Returns:
            dict[str, np.ndarray]: Parameters.

        """
        image_shape = params["shape"][:2]

        original_tiles = fgeometric.split_uniform_grid(
            image_shape,
            self.grid,
            self.random_generator,
        )
        shape_groups = fgeometric.create_shape_groups(original_tiles)
        mapping = fgeometric.shuffle_tiles_within_shape_groups(
            shape_groups,
            self.random_generator,
        )

        return {"tiles": original_tiles, "mapping": mapping}


class Morphological(DualTransform):
    """Apply a morphological operation (dilation or erosion) to an image,
    with particular value for enhancing document scans.

    Morphological operations modify the structure of the image.
    Dilation expands the white (foreground) regions in a binary or grayscale image, while erosion shrinks them.
    These operations are beneficial in document processing, for example:
    - Dilation helps in closing up gaps within text or making thin lines thicker,
        enhancing legibility for OCR (Optical Character Recognition).
    - Erosion can remove small white noise and detach connected objects,
        making the structure of larger objects more pronounced.

    Args:
        scale (int or tuple/list of int): Specifies the size of the structuring element (kernel) used for the operation.
            - If an integer is provided, a square kernel of that size will be used.
            - If a tuple or list is provided, it should contain two integers representing the minimum
                and maximum sizes for the dilation kernel.
        operation (Literal["erosion", "dilation"]): The morphological operation to apply.
            Default is 'dilation'.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Targets:
        image, mask, keypoints, bboxes, volume, mask3d

    Image types:
        uint8, float32

    References:
        Nougat: https://github.com/facebookresearch/nougat

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a document-like binary image with text
        >>> image = np.ones((200, 500), dtype=np.uint8) * 255  # White background
        >>> # Add some "text" (black pixels)
        >>> cv2.putText(image, "Document Text", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        >>> # Add some "noise" (small black dots)
        >>> for _ in range(50):
        ...     x, y = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
        ...     cv2.circle(image, (x, y), 1, 0, -1)
        >>>
        >>> # Create a mask representing text regions
        >>> mask = np.zeros_like(image)
        >>> mask[image < 128] = 1  # Binary mask where text exists
        >>>
        >>> # Example 1: Apply dilation to thicken text and fill gaps
        >>> dilation_transform = A.Morphological(
        ...     scale=3,               # Size of the structuring element
        ...     operation="dilation",  # Expand white regions (or black if inverted)
        ...     p=1.0                  # Always apply
        ... )
        >>> result = dilation_transform(image=image, mask=mask)
        >>> dilated_image = result['image']    # Text is thicker, gaps are filled
        >>> dilated_mask = result['mask']      # Mask is expanded around text regions
        >>>
        >>> # Example 2: Apply erosion to thin text or remove noise
        >>> erosion_transform = A.Morphological(
        ...     scale=(2, 3),          # Random kernel size between 2 and 3
        ...     operation="erosion",   # Shrink white regions (or expand black if inverted)
        ...     p=1.0                  # Always apply
        ... )
        >>> result = erosion_transform(image=image, mask=mask)
        >>> eroded_image = result['image']     # Text is thinner, small noise may be removed
        >>> eroded_mask = result['mask']       # Mask is contracted around text regions
        >>>
        >>> # Note: For document processing, dilation often helps enhance readability for OCR
        >>> # while erosion can help remove noise or separate connected components

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        scale: OnePlusIntRangeType
        operation: Literal["erosion", "dilation"]

    def __init__(
        self,
        scale: tuple[int, int] | int = (2, 3),
        operation: Literal["erosion", "dilation"] = "dilation",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.scale = cast("tuple[int, int]", scale)
        self.operation = operation

    def apply(
        self,
        img: np.ndarray,
        kernel: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the Morphological transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the Morphological transform to.
            kernel (tuple[int, int]): The structuring element (kernel) used for the operation.
            **params (Any): Additional parameters for the transform.

        """
        return fgeometric.morphology(img, kernel, self.operation)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        kernel: tuple[int, int],
        **params: Any,
    ) -> np.ndarray:
        """Apply the Morphological transform to the input bounding boxes.

        Args:
            bboxes (np.ndarray): The input bounding boxes to apply the Morphological transform to.
            kernel (tuple[int, int]): The structuring element (kernel) used for the operation.
            **params (Any): Additional parameters for the transform.

        """
        image_shape = params["shape"]

        denormalized_boxes = denormalize_bboxes(bboxes, image_shape)

        result = fgeometric.bboxes_morphology(
            denormalized_boxes,
            kernel,
            self.operation,
            image_shape,
        )

        return normalize_bboxes(result, image_shape)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Morphological transform to the input keypoints.

        Args:
            keypoints (np.ndarray): The input keypoints to apply the Morphological transform to.
            **params (Any): Additional parameters for the transform.

        """
        return keypoints

    def get_params(self) -> dict[str, float]:
        """Generate parameters for the Morphological transform.

        Returns:
            dict[str, float]: The parameters of the transform.

        """
        return {
            "kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.scale),
        }
