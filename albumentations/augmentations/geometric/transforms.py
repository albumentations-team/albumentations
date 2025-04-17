"""Geometric transformation classes for image augmentation.

This module provides a collection of transforms that modify the geometric properties
of images and associated data (masks, bounding boxes, keypoints). Includes implementations
for flipping, transposing, affine transformations, distortions, padding, and more complex
transformations like grid shuffling and thin plate splines.
"""

from __future__ import annotations

import random
from numbers import Real
from typing import Annotated, Any, Literal, cast
from warnings import warn

import cv2
import numpy as np
from albucore import batch_transform, hflip, is_grayscale_image, is_rgb_image, vflip
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
    SymmetricRangeType,
    check_range_bounds,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import (
    ALL_TARGETS,
    BIG_INTEGER,
    d4_group_elements,
)
from albumentations.core.utils import to_tuple

from . import functional as fgeometric

__all__ = [
    "D4",
    "Affine",
    "ElasticTransform",
    "GridDistortion",
    "GridElasticDeform",
    "HorizontalFlip",
    "OpticalDistortion",
    "Pad",
    "PadIfNeeded",
    "Perspective",
    "PiecewiseAffine",
    "RandomGridShuffle",
    "ShiftScaleRotate",
    "SquareSymmetry",
    "ThinPlateSpline",
    "Transpose",
    "VerticalFlip",
]

NUM_PADS_XY = 2
NUM_PADS_ALL_SIDES = 4


class BaseDistortion(DualTransform):
    """Base class for distortion-based transformations.

    This class provides a foundation for implementing various types of image distortions,
    such as optical distortions, grid distortions, and elastic transformations. It handles
    the common operations of applying distortions to images, masks, bounding boxes, and keypoints.

    Args:
        interpolation (int): Interpolation method to be used for image transformation.
            Should be one of the OpenCV interpolation types (e.g., cv2.INTER_LINEAR,
            cv2.INTER_CUBIC).
        mask_interpolation (int): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
        keypoint_remapping_method (Literal["direct", "mask"]): Method to use for keypoint remapping.
            - "mask": Uses mask-based remapping. Faster, especially for many keypoints, but may be
              less accurate for large distortions. Recommended for large images or many keypoints.
            - "direct": Uses inverse mapping. More accurate for large distortions but slower.
            Default: "mask"
        p (float): Probability of applying the transform.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

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

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        interpolation: Literal[cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]
        keypoint_remapping_method: Literal["direct", "mask"]
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ],
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ],
        keypoint_remapping_method: Literal["direct", "mask"],
        p: float,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
    ):
        super().__init__(p=p)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.keypoint_remapping_method = keypoint_remapping_method
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask

    def apply(
        self,
        img: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the distortion to the input image.

        Args:
            img (np.ndarray): Input image to be distorted.
            map_x (np.ndarray): X-coordinate map of the distortion.
            map_y (np.ndarray): Y-coordinate map of the distortion.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted image.

        """
        return fgeometric.remap(
            img,
            map_x,
            map_y,
            self.interpolation,
            self.border_mode,
            self.fill,
        )

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the distortion to a batch of images.

        Args:
            images (np.ndarray): Batch of images to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Batch of distorted images.

        """
        return self.apply(images, **params)

    @batch_transform("spatial", has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the distortion to a volume.

        Args:
            volume (np.ndarray): Volume to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted volume.

        """
        return self.apply(volume, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the distortion to a batch of volumes.

        Args:
            volumes (np.ndarray): Batch of volumes to be distorted.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Batch of distorted volumes.

        """
        return self.apply(volumes, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_mask3d(self, mask3d: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the distortion to a 3D mask.

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
        map_x: np.ndarray,
        map_y: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the distortion to a mask.

        Args:
            mask (np.ndarray): Mask to be distorted.
            map_x (np.ndarray): X-coordinate map of the distortion.
            map_y (np.ndarray): Y-coordinate map of the distortion.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted mask.

        """
        return fgeometric.remap(
            mask,
            map_x,
            map_y,
            self.mask_interpolation,
            self.border_mode,
            self.fill_mask,
        )

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the distortion to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to be distorted.
            map_x (np.ndarray): X-coordinate map of the distortion.
            map_y (np.ndarray): Y-coordinate map of the distortion.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted bounding boxes.

        """
        image_shape = params["shape"][:2]
        bboxes_denorm = denormalize_bboxes(bboxes, image_shape)
        bboxes_returned = fgeometric.remap_bboxes(
            bboxes_denorm,
            map_x,
            map_y,
            image_shape,
        )
        return normalize_bboxes(bboxes_returned, image_shape)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        map_x: np.ndarray,
        map_y: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the distortion to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to be distorted.
            map_x (np.ndarray): X-coordinate map of the distortion.
            map_y (np.ndarray): Y-coordinate map of the distortion.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Distorted keypoints.

        """
        if self.keypoint_remapping_method == "direct":
            return fgeometric.remap_keypoints(keypoints, map_x, map_y, params["shape"])
        return fgeometric.remap_keypoints_via_mask(keypoints, map_x, map_y, params["shape"])


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
        approximate (bool): Whether to use an approximate version of the elastic transform. If True,
            uses a fixed kernel size for Gaussian smoothing, which can be faster but potentially
            less accurate for large sigma values. Default: False
        same_dxdy (bool): Whether to use the same random displacement field for both x and y
            directions. Can speed up the transform at the cost of less diverse distortions. Default: False
        mask_interpolation (int): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        noise_distribution (Literal["gaussian", "uniform"]): Distribution used to generate the displacement fields.
            "gaussian" generates fields using normal distribution (more natural deformations).
            "uniform" generates fields using uniform distribution (more mechanical deformations).
            Default: "gaussian".
        keypoint_remapping_method (Literal["direct", "mask"]): Method to use for keypoint remapping.
            - "mask": Uses mask-based remapping. Faster, especially for many keypoints, but may be
              less accurate for large distortions. Recommended for large images or many keypoints.
            - "direct": Uses inverse mapping. More accurate for large distortions but slower.
            Default: "mask"

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

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
        noise_distribution: Literal["gaussian", "uniform"]
        keypoint_remapping_method: Literal["direct", "mask"]

    def __init__(
        self,
        alpha: float = 1,
        sigma: float = 50,
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_LINEAR,
        approximate: bool = False,
        same_dxdy: bool = False,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_NEAREST,
        noise_distribution: Literal["gaussian", "uniform"] = "gaussian",
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
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
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            p=p,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
        )
        self.alpha = alpha
        self.sigma = sigma
        self.approximate = approximate
        self.same_dxdy = same_dxdy
        self.noise_distribution = noise_distribution

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate displacement fields for the elastic transform.

        Args:
            params (dict[str, Any]): Dictionary containing parameters for the transform.
            data (dict[str, Any]): Dictionary containing data for the transform.

        Returns:
            dict[str, Any]: Dictionary containing displacement fields for the elastic transform.

        """
        height, width = params["shape"][:2]
        kernel_size = (17, 17) if self.approximate else (0, 0)

        # Generate displacement fields
        dx, dy = fgeometric.generate_displacement_fields(
            (height, width),
            self.alpha,
            self.sigma,
            same_dxdy=self.same_dxdy,
            kernel_size=kernel_size,
            random_generator=self.random_generator,
            noise_distribution=self.noise_distribution,
        )

        # Vectorized map generation
        coords = np.stack(np.meshgrid(np.arange(width), np.arange(height)))
        maps = coords + np.stack([dx, dy])
        return {
            "map_x": maps[0].astype(np.float32),
            "map_y": maps[1].astype(np.float32),
        }


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

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Compose([
        ...     A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.5),
        ... ])
        >>> result = transform(image=image)
        >>> transformed_image = result['image']

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

        fill: tuple[float, ...] | float = 0
        fill_mask: tuple[float, ...] | float = 0

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


class PiecewiseAffine(BaseDistortion):
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
        absolute_scale (bool): If set to True, the value of the scale parameter will be treated as an absolute
            pixel value. If set to False, it will be treated as a fraction of the image height and width.
            Default: False.
        keypoint_remapping_method (Literal["direct", "mask"]): Method to use for keypoint remapping.
            - "mask": Uses mask-based remapping. Faster, especially for many keypoints, but may be
              less accurate for large distortions. Recommended for large images or many keypoints.
            - "direct": Uses inverse mapping. More accurate for large distortions but slower.
            Default: "mask"
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes, volume, mask3d

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

    class InitSchema(BaseDistortion.InitSchema):
        scale: NonNegativeFloatRangeType
        nb_rows: tuple[int, int] | int
        nb_cols: tuple[int, int] | int
        absolute_scale: bool

        @field_validator("nb_rows", "nb_cols")
        @classmethod
        def _process_range(
            cls,
            value: tuple[int, int] | int,
            info: ValidationInfo,
        ) -> tuple[int, int]:
            bounds = 2, BIG_INTEGER
            result = to_tuple(value, value)
            check_range(result, *bounds, info.field_name)
            return result

    def __init__(
        self,
        scale: tuple[float, float] | float = (0.03, 0.05),
        nb_rows: tuple[int, int] | int = (4, 4),
        nb_cols: tuple[int, int] | int = (4, 4),
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
        absolute_scale: bool = False,
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
        p: float = 0.5,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
    ):
        super().__init__(
            p=p,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
        )

        warn(
            "This augmenter is very slow. Try to use ``ElasticTransform`` instead, which is at least 10x faster.",
            stacklevel=2,
        )

        self.scale = cast("tuple[float, float]", scale)
        self.nb_rows = cast("tuple[int, int]", nb_rows)
        self.nb_cols = cast("tuple[int, int]", nb_cols)
        self.absolute_scale = absolute_scale

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

        nb_rows = np.clip(self.py_random.randint(*self.nb_rows), 2, None)
        nb_cols = np.clip(self.py_random.randint(*self.nb_cols), 2, None)
        scale = self.py_random.uniform(*self.scale)

        map_x, map_y = fgeometric.create_piecewise_affine_maps(
            image_shape=image_shape,
            grid=(nb_rows, nb_cols),
            scale=scale,
            absolute_scale=self.absolute_scale,
            random_generator=self.random_generator,
        )

        return {"map_x": map_x, "map_y": map_y}


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

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

    _targets = ALL_TARGETS

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the vertical flip to an image.

        Args:
            img (np.ndarray): Image to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped image.

        """
        return vflip(img)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the vertical flip to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped bounding boxes.

        """
        return fgeometric.bboxes_vflip(bboxes)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the vertical flip to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped keypoints.

        """
        return fgeometric.keypoints_vflip(keypoints, params["shape"][0])

    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the vertical flip to a batch of images.

        Args:
            images (np.ndarray): Images to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped images.

        """
        return fgeometric.volume_vflip(images)

    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the vertical flip to a volume.

        Args:
            volume (np.ndarray): Volume to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped volume.

        """
        return self.apply_to_images(volume, **params)

    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the vertical flip to a batch of volumes.

        Args:
            volumes (np.ndarray): Volumes to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped volumes.

        """
        return fgeometric.volumes_vflip(volumes)

    def apply_to_mask3d(self, mask3d: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the vertical flip to a 3D mask.

        Args:
            mask3d (np.ndarray): 3D mask to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped 3D mask.

        """
        return self.apply_to_images(mask3d, **params)

    def apply_to_masks3d(self, masks3d: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the vertical flip to a 3D mask.

        Args:
            masks3d (np.ndarray): 3D masks to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped 3D mask.

        """
        return self.apply_to_volumes(masks3d, **params)


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    """

    _targets = ALL_TARGETS

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the horizontal flip to an image.

        Args:
            img (np.ndarray): Image to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped image.

        """
        return hflip(img)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the horizontal flip to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped bounding boxes.

        """
        return fgeometric.bboxes_hflip(bboxes)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the horizontal flip to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped keypoints.

        """
        return fgeometric.keypoints_hflip(keypoints, params["shape"][1])

    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the horizontal flip to a batch of images.

        Args:
            images (np.ndarray): Images to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped images.

        """
        return fgeometric.volume_hflip(images)

    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the horizontal flip to a volume.

        Args:
            volume (np.ndarray): Volume to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped volume.

        """
        return self.apply_to_images(volume, **params)

    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the horizontal flip to a batch of volumes.

        Args:
            volumes (np.ndarray): Volumes to be flipped.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Flipped volumes.

        """
        return fgeometric.volumes_hflip(volumes)

    def apply_to_mask3d(self, mask3d: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the horizontal flip to a 3D mask.

        Args:
            mask3d (np.ndarray): 3D mask to be flipped.
            **params (Any): Additional parameters.

        """
        return self.apply_to_images(mask3d, **params)

    def apply_to_masks3d(self, masks3d: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the horizontal flip to a 3D mask.

        Args:
            masks3d (np.ndarray): 3D masks to be flipped.
            **params (Any): Additional parameters.

        """
        return self.apply_to_volumes(masks3d, **params)


class Transpose(DualTransform):
    """Transpose the input by swapping its rows and columns.

    This transform flips the image over its main diagonal, effectively switching its width and height.
    It's equivalent to a 90-degree rotation followed by a horizontal flip.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

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

    _targets = ALL_TARGETS

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the transpose to an image.

        Args:
            img (np.ndarray): Image to be transposed.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transposed image.

        """
        return fgeometric.transpose(img)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the transpose to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to be transposed.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transposed bounding boxes.

        """
        return fgeometric.bboxes_transpose(bboxes)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the transpose to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to be transposed.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transposed keypoints.

        """
        return fgeometric.keypoints_transpose(keypoints)


class OpticalDistortion(BaseDistortion):
    """Apply optical distortion to images, masks, bounding boxes, and keypoints.

    Supports two distortion models:
    1. Camera matrix model (original):
       Uses OpenCV's camera calibration model with k1=k2=k distortion coefficients

    2. Fisheye model:
       Direct radial distortion: r_dist = r * (1 + gamma * r²)

    Args:
        distort_limit (float | tuple[float, float]): Range of distortion coefficient.
            For camera model: recommended range (-0.05, 0.05)
            For fisheye model: recommended range (-0.3, 0.3)
            Default: (-0.05, 0.05)

        mode (Literal['camera', 'fisheye']): Distortion model to use:
            - 'camera': Original camera matrix model
            - 'fisheye': Fisheye lens model
            Default: 'camera'

        interpolation (OpenCV flag): Interpolation method used for image transformation.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
            cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.

        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.

        keypoint_remapping_method (Literal["direct", "mask"]): Method to use for keypoint remapping.
            - "mask": Uses mask-based remapping. Faster, especially for many keypoints, but may be
              less accurate for large distortions. Recommended for large images or many keypoints.
            - "direct": Uses inverse mapping. More accurate for large distortions but slower.
            Default: "mask"

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - The distortion is applied using OpenCV's initUndistortRectifyMap and remap functions.
        - The distortion coefficient (k) is randomly sampled from the distort_limit range.
        - Bounding boxes and keypoints are transformed along with the image to maintain consistency.
        - Fisheye model directly applies radial distortion

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.OpticalDistortion(distort_limit=0.1, p=1.0),
        ... ])
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>> transformed_image = transformed['image']
        >>> transformed_mask = transformed['mask']
        >>> transformed_bboxes = transformed['bboxes']
        >>> transformed_keypoints = transformed['keypoints']

    """

    class InitSchema(BaseDistortion.InitSchema):
        distort_limit: SymmetricRangeType
        mode: Literal["camera", "fisheye"]
        keypoint_remapping_method: Literal["direct", "mask"]

    def __init__(
        self,
        distort_limit: tuple[float, float] | float = (-0.05, 0.05),
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
        mode: Literal["camera", "fisheye"] = "camera",
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
        p: float = 0.5,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            p=p,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
        )
        self.distort_limit = cast("tuple[float, float]", distort_limit)
        self.mode = mode

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

        # Get distortion coefficient
        k = self.py_random.uniform(*self.distort_limit)

        # Get distortion maps based on mode
        if self.mode == "camera":
            map_x, map_y = fgeometric.get_camera_matrix_distortion_maps(
                image_shape,
                k,
            )
        else:  # fisheye
            map_x, map_y = fgeometric.get_fisheye_distortion_maps(
                image_shape,
                k,
            )

        return {"map_x": map_x, "map_y": map_y}


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
        normalized (bool): If True, ensures that the distortion does not move pixels
            outside the image boundaries. This can result in less extreme distortions
            but guarantees that no information is lost. Default: True.
        mask_interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        keypoint_remapping_method (Literal["direct", "mask"]): Method to use for keypoint remapping.
            - "mask": Uses mask-based remapping. Faster, especially for many keypoints, but may be
              less accurate for large distortions. Recommended for large images or many keypoints.
            - "direct": Uses inverse mapping. More accurate for large distortions but slower.
            Default: "mask"
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

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
        keypoint_remapping_method: Literal["direct", "mask"]

        @field_validator("distort_limit")
        @classmethod
        def _check_limits(
            cls,
            v: tuple[float, float],
            info: ValidationInfo,
        ) -> tuple[float, float]:
            bounds = -1, 1
            result = to_tuple(v)
            check_range(result, *bounds, info.field_name)
            return result

    def __init__(
        self,
        num_steps: int = 5,
        distort_limit: tuple[float, float] | float = (-0.3, 0.3),
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_LINEAR,
        normalized: bool = True,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_NEAREST,
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
        p: float = 0.5,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            p=p,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
        )
        self.num_steps = num_steps
        self.distort_limit = cast("tuple[float, float]", distort_limit)
        self.normalized = normalized

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
        steps_x = [1 + self.py_random.uniform(*self.distort_limit) for _ in range(self.num_steps + 1)]
        steps_y = [1 + self.py_random.uniform(*self.distort_limit) for _ in range(self.num_steps + 1)]

        if self.normalized:
            normalized_params = fgeometric.normalize_grid_distortion_steps(
                image_shape,
                self.num_steps,
                steps_x,
                steps_y,
            )
            steps_x, steps_y = (
                normalized_params["steps_x"],
                normalized_params["steps_y"],
            )

        map_x, map_y = fgeometric.generate_grid(
            image_shape,
            steps_x,
            steps_y,
            self.num_steps,
        )

        return {"map_x": map_x, "map_y": map_y}


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
        image, mask, bboxes, keypoints, volume, mask3d

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

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(
        self,
        p: float = 1,
    ):
        super().__init__(p=p)

    def apply(
        self,
        img: np.ndarray,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> np.ndarray:
        """Apply the D4 transform to an image.

        Args:
            img (np.ndarray): Image to be transformed.
            group_element (Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]): Group element to apply.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transformed image.

        """
        return fgeometric.d4(img, group_element)

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> np.ndarray:
        """Apply the D4 transform to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to be transformed.
            group_element (Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]): Group element to apply.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transformed bounding boxes.

        """
        return fgeometric.bboxes_d4(bboxes, group_element)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        group_element: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
        **params: Any,
    ) -> np.ndarray:
        """Apply the D4 transform to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to be transformed.
            group_element (Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]): Group element to apply.
            **params (Any): Additional parameters.

        """
        return fgeometric.keypoints_d4(keypoints, group_element, params["shape"])

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the D4 transform to a batch of images.

        Args:
            images (np.ndarray): Images to be transformed.
            **params (Any): Additional parameters.

        """
        return self.apply(images, **params)

    @batch_transform("spatial", has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the D4 transform to a volume.

        Args:
            volume (np.ndarray): Volume to be transformed.
            **params (Any): Additional parameters.

        """
        return self.apply(volume, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the D4 transform to a batch of volumes.

        Args:
            volumes (np.ndarray): Volumes to be transformed.
            **params (Any): Additional parameters.

        """
        return self.apply(volumes, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=False)
    def apply_to_mask3d(self, mask3d: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the D4 transform to a 3D mask.

        Args:
            mask3d (np.ndarray): 3D mask to be transformed.
            **params (Any): Additional parameters.

        """
        return self.apply(mask3d, **params)

    def get_params(self) -> dict[str, Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]]:
        """Get the parameters for the D4 transform.

        Returns:
            dict[str, Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]]: Parameters.

        """
        return {
            "group_element": self.random_generator.choice(d4_group_elements),
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

    Example:
        >>> transform = GridElasticDeform(num_grid_xy=(4, 4), magnitude=10, p=1.0)
        >>> result = transform(image=image, mask=mask)
        >>> transformed_image, transformed_mask = result['image'], result['mask']

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

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.array([
        ...     [1, 1, 1, 2, 2, 2],
        ...     [1, 1, 1, 2, 2, 2],
        ...     [1, 1, 1, 2, 2, 2],
        ...     [3, 3, 3, 4, 4, 4],
        ...     [3, 3, 3, 4, 4, 4],
        ...     [3, 3, 3, 4, 4, 4]
        ... ])
        >>> transform = A.RandomGridShuffle(grid=(2, 2), p=1.0)
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        # The resulting image might look like this (one possible outcome):
        # [[4, 4, 4, 2, 2, 2],
        #  [4, 4, 4, 2, 2, 2],
        #  [4, 4, 4, 2, 2, 2],
        #  [3, 3, 3, 1, 1, 1],
        #  [3, 3, 3, 1, 1, 1],
        #  [3, 3, 3, 1, 1, 1]]

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


class Pad(DualTransform):
    """Pad the sides of an image by specified number of pixels.

    Args:
        padding (int, tuple[int, int] or tuple[int, int, int, int]): Padding values. Can be:
            * int - pad all sides by this value
            * tuple[int, int] - (pad_x, pad_y) to pad left/right by pad_x and top/bottom by pad_y
            * tuple[int, int, int, int] - (left, top, right, bottom) specific padding per side
        fill (tuple[float, ...] | float): Padding value if border_mode is cv2.BORDER_CONSTANT
        fill_mask (tuple[float, ...] | float): Padding value for mask if border_mode is cv2.BORDER_CONSTANT
        border_mode (OpenCV flag): OpenCV border mode
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    References:
        PyTorch Pad: https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Pad.html

    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        padding: int | tuple[int, int] | tuple[int, int, int, int]
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
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.padding = padding
        self.fill = fill
        self.fill_mask = fill_mask
        self.border_mode = border_mode

    def apply(
        self,
        img: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Pad transform to an image.

        Args:
            img (np.ndarray): Image to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
        return fgeometric.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.fill,
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
        """Apply the Pad transform to a mask.

        Args:
            mask (np.ndarray): Mask to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
        return fgeometric.pad_with_params(
            mask,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.fill_mask,
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
        """Apply the Pad transform to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
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
        return normalize_bboxes(
            result,
            (rows + pad_top + pad_bottom, cols + pad_left + pad_right),
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Pad transform to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to be transformed.
            pad_top (int): Top padding.
            pad_bottom (int): Bottom padding.
            pad_left (int): Left padding.
            pad_right (int): Right padding.
            **params (Any): Additional parameters.

        """
        return fgeometric.pad_keypoints(
            keypoints,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            self.border_mode,
            image_shape=params["shape"][:2],
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
        if isinstance(self.padding, Real):
            pad_top = pad_bottom = pad_left = pad_right = self.padding
        elif isinstance(self.padding, (tuple, list)):
            if len(self.padding) == NUM_PADS_XY:
                pad_left = pad_right = self.padding[0]
                pad_top = pad_bottom = self.padding[1]
            elif len(self.padding) == NUM_PADS_ALL_SIDES:
                pad_left, pad_top, pad_right, pad_bottom = self.padding  # type: ignore[misc]
            else:
                raise TypeError(
                    "Padding must be a single number, a pair of numbers, or a quadruple of numbers",
                )
        else:
            raise TypeError(
                "Padding must be a single number, a pair of numbers, or a quadruple of numbers",
            )

        return {
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
        }


class PadIfNeeded(Pad):
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
            The default is `cv2.BORDER_CONSTANT`.
        fill (tuple[float, ...] | float | None): Value to fill the border pixels if the border mode
            is `cv2.BORDER_CONSTANT`. Default is None.
        fill_mask (tuple[float, ...] | float | None): Similar to `fill` but used for padding masks. Default is None.
        p (float): Probability of applying the transform. Default is 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - Either `min_height` or `pad_height_divisor` must be set, but not both.
        - Either `min_width` or `pad_width_divisor` must be set, but not both.
        - If `border_mode` is set to `cv2.BORDER_CONSTANT`, `value` must be provided.
        - The transform will maintain consistency across all targets (image, mask, bboxes, keypoints, volume).
        - For bounding boxes, the coordinates will be adjusted to account for the padding.
        - For keypoints, their positions will be shifted according to the padding.

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT, fill=0),
        ... ])
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>> padded_image = transformed['image']
        >>> padded_mask = transformed['mask']
        >>> adjusted_bboxes = transformed['bboxes']
        >>> adjusted_keypoints = transformed['keypoints']

    """

    class InitSchema(BaseTransformInitSchema):
        min_height: int | None = Field(ge=1)
        min_width: int | None = Field(ge=1)
        pad_height_divisor: int | None = Field(ge=1)
        pad_width_divisor: int | None = Field(ge=1)
        position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"]
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]

        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

        @model_validator(mode="after")
        def _validate_divisibility(self) -> Self:
            if (self.min_height is None) == (self.pad_height_divisor is None):
                msg = "Only one of 'min_height' and 'pad_height_divisor' parameters must be set"
                raise ValueError(msg)
            if (self.min_width is None) == (self.pad_width_divisor is None):
                msg = "Only one of 'min_width' and 'pad_width_divisor' parameters must be set"
                raise ValueError(msg)

            if self.border_mode == cv2.BORDER_CONSTANT and self.fill is None:
                msg = "If 'border_mode' is set to 'BORDER_CONSTANT', 'fill' must be provided."
                raise ValueError(msg)

            return self

    def __init__(
        self,
        min_height: int | None = 1024,
        min_width: int | None = 1024,
        pad_height_divisor: int | None = None,
        pad_width_divisor: int | None = None,
        position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = "center",
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        p: float = 1.0,
    ):
        # Initialize with dummy padding that will be calculated later
        super().__init__(
            padding=0,
            fill=fill,
            fill_mask=fill_mask,
            border_mode=border_mode,
            p=p,
        )
        self.min_height = min_height
        self.min_width = min_width
        self.pad_height_divisor = pad_height_divisor
        self.pad_width_divisor = pad_width_divisor
        self.position = position

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

        return {
            "pad_top": h_pad_top,
            "pad_bottom": h_pad_bottom,
            "pad_left": w_pad_left,
            "pad_right": w_pad_right,
        }


class ThinPlateSpline(BaseDistortion):
    r"""Apply Thin Plate Spline (TPS) transformation to create smooth, non-rigid deformations.

    Imagine the image printed on a thin metal plate that can be bent and warped smoothly:
    - Control points act like pins pushing or pulling the plate
    - The plate resists sharp bending, creating smooth deformations
    - The transformation maintains continuity (no tears or folds)
    - Areas between control points are interpolated naturally

    The transform works by:
    1. Creating a regular grid of control points (like pins in the plate)
    2. Randomly displacing these points (like pushing/pulling the pins)
    3. Computing a smooth interpolation (like the plate bending)
    4. Applying the resulting deformation to the image


    Args:
        scale_range (tuple[float, float]): Range for random displacement of control points.
            Values should be in [0.0, 1.0]:
            - 0.0: No displacement (identity transform)
            - 0.1: Subtle warping
            - 0.2-0.4: Moderate deformation (recommended range)
            - 0.5+: Strong warping
            Default: (0.2, 0.4)

        num_control_points (int): Number of control points per side.
            Creates a grid of num_control_points x num_control_points points.
            - 2: Minimal deformation (affine-like)
            - 3-4: Moderate flexibility (recommended)
            - 5+: More local deformation control
            Must be >= 2. Default: 4

        interpolation (int): OpenCV interpolation flag. Used for image sampling.
            See also: cv2.INTER_*
            Default: cv2.INTER_LINEAR

        mask_interpolation (int): OpenCV interpolation flag. Used for mask sampling.
            See also: cv2.INTER_*
            Default: cv2.INTER_NEAREST

        keypoint_remapping_method (Literal["direct", "mask"]): Method to use for keypoint remapping.
            - "mask": Uses mask-based remapping. Faster, especially for many keypoints, but may be
              less accurate for large distortions. Recommended for large images or many keypoints.
            - "direct": Uses inverse mapping. More accurate for large distortions but slower.
            Default: "mask"

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, mask, keypoints, bboxes, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - The transformation preserves smoothness and continuity
        - Stronger scale values may create more extreme deformations
        - Higher number of control points allows more local deformations
        - The same deformation is applied consistently to all targets

    Example:
        >>> import albumentations as A
        >>> # Basic usage
        >>> transform = A.ThinPlateSpline()
        >>>
        >>> # Subtle deformation
        >>> transform = A.ThinPlateSpline(
        ...     scale_range=(0.1, 0.2),
        ...     num_control_points=3
        ... )
        >>>
        >>> # Strong warping with fine control
        >>> transform = A.ThinPlateSpline(
        ...     scale_range=(0.3, 0.5),
        ...     num_control_points=5,
        ... )

    References:
        - "Principal Warps: Thin-Plate Splines and the Decomposition of Deformations"
          by F.L. Bookstein
          https://doi.org/10.1109/34.24792

        - Thin Plate Splines in Computer Vision:
          https://en.wikipedia.org/wiki/Thin_plate_spline

        - Similar implementation in Kornia:
          https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomThinPlateSpline

    See Also:
        - ElasticTransform: For different type of non-rigid deformation
        - GridDistortion: For grid-based warping
        - OpticalDistortion: For lens-like distortions

    """

    class InitSchema(BaseDistortion.InitSchema):
        scale_range: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        num_control_points: int = Field(ge=2)
        keypoint_remapping_method: Literal["direct", "mask"]

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.2, 0.4),
        num_control_points: int = 4,
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
        keypoint_remapping_method: Literal["direct", "mask"] = "mask",
        p: float = 0.5,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
    ):
        super().__init__(
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            keypoint_remapping_method=keypoint_remapping_method,
            p=p,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
        )
        self.scale_range = scale_range
        self.num_control_points = num_control_points

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
        height, width = params["shape"][:2]
        src_points = fgeometric.generate_control_points(self.num_control_points)

        # Add random displacement to destination points
        scale = self.py_random.uniform(*self.scale_range) / 10
        dst_points = src_points + self.random_generator.normal(
            0,
            scale,
            src_points.shape,
        )

        # Compute TPS weights
        weights, affine = fgeometric.compute_tps_weights(src_points, dst_points)

        # Create grid of points
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        points = np.stack([x.flatten(), y.flatten()], axis=1).astype(np.float32)

        # Transform points
        transformed = fgeometric.tps_transform(
            points / [width, height],
            src_points,
            weights,
            affine,
        )
        transformed *= [width, height]

        return {
            "map_x": transformed[:, 0].reshape(height, width).astype(np.float32),
            "map_y": transformed[:, 1].reshape(height, width).astype(np.float32),
        }


class SquareSymmetry(D4):
    """Applies one of the eight possible square symmetry transformations to a square-shaped input.
    This is an alias for D4 transform with a more intuitive name for those not familiar with group theory.

    The square symmetry transformations include:
    - Identity: No transformation is applied
    - 90° rotation: Rotate 90 degrees counterclockwise
    - 180° rotation: Rotate 180 degrees
    - 270° rotation: Rotate 270 degrees counterclockwise
    - Vertical flip: Mirror across vertical axis
    - Anti-diagonal flip: Mirror across anti-diagonal
    - Horizontal flip: Mirror across horizontal axis
    - Main diagonal flip: Mirror across main diagonal

    Args:
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

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
        ...     A.SquareSymmetry(p=1.0),
        ... ])
        >>> transformed = transform(image=image)
        >>> transformed_image = transformed['image']
        # The resulting image will be one of the 8 possible square symmetry transformations of the input

    """
