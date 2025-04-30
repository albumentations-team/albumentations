"""Geometric distortion transforms for image augmentation.

This module provides various geometric distortion transformations that modify the spatial arrangement
of pixels in images while preserving their intensity values. These transforms can create
non-rigid deformations that are useful for data augmentation, especially when training models
that need to be robust to geometric variations.

Available transforms:
- ElasticTransform: Creates random elastic deformations by displacing pixels along random vectors
- GridDistortion: Distorts the image by moving the nodes of a grid placed on the image
- OpticalDistortion: Simulates lens distortion effects (barrel/pincushion) using camera or fisheye models
- PiecewiseAffine: Divides the image into a grid and applies random affine transformations to each cell
- ThinPlateSpline: Applies smooth deformations based on the thin plate spline interpolation technique

All transforms inherit from BaseDistortion, which provides a common interface and functionality
for applying distortion maps to various target types (images, masks, bounding boxes, keypoints).
These transforms are particularly useful for:

- Data augmentation to increase training set diversity
- Simulating real-world distortion effects like camera lens aberrations
- Creating more challenging test cases for computer vision models
- Medical image analysis where anatomy might appear in different shapes

Each transform supports customization through various parameters controlling the strength,
type, and characteristics of the distortion, as well as interpolation methods for different
target types.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast
from warnings import warn

import cv2
import numpy as np
from albucore import batch_transform
from pydantic import (
    AfterValidator,
    Field,
    ValidationInfo,
    field_validator,
)

from albumentations.augmentations.utils import check_range
from albumentations.core.bbox_utils import (
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
)
from albumentations.core.utils import to_tuple

from . import functional as fgeometric

__all__ = [
    "ElasticTransform",
    "GridDistortion",
    "OpticalDistortion",
    "PiecewiseAffine",
    "ThinPlateSpline",
]


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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> class CustomDistortion(A.BaseDistortion):
        ...     def __init__(self, distort_limit=0.3, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         self.distort_limit = distort_limit
        ...
        ...     def get_params_dependent_on_data(self, params, data):
        ...         height, width = params["shape"][:2]
        ...         # Create distortion maps - a simple radial distortion in this example
        ...         map_x = np.zeros((height, width), dtype=np.float32)
        ...         map_y = np.zeros((height, width), dtype=np.float32)
        ...
        ...         # Calculate distortion center
        ...         center_x = width / 2
        ...         center_y = height / 2
        ...
        ...         # Generate distortion maps
        ...         for y in range(height):
        ...             for x in range(width):
        ...                 # Distance from center
        ...                 dx = (x - center_x) / width
        ...                 dy = (y - center_y) / height
        ...                 r = np.sqrt(dx * dx + dy * dy)
        ...
        ...                 # Apply radial distortion
        ...                 factor = 1 + self.distort_limit * r
        ...                 map_x[y, x] = x + dx * factor
        ...                 map_y[y, x] = y + dy * factor
        ...
        ...         return {"map_x": map_x, "map_y": map_y}
        >>>
        >>> # Prepare sample data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]
        >>>
        >>> # Define transform with the custom distortion
        >>> transform = A.Compose([
        ...     CustomDistortion(
        ...         distort_limit=0.2,
        ...         interpolation=cv2.INTER_LINEAR,
        ...         mask_interpolation=cv2.INTER_NEAREST,
        ...         keypoint_remapping_method="mask",
        ...         p=1.0
        ...     )
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
        >>> transformed_image = transformed['image']
        >>> transformed_mask = transformed['mask']
        >>> transformed_bboxes = transformed['bboxes']
        >>> transformed_keypoints = transformed['keypoints']

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

    Examples:
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

    Examples:
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

    Examples:
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

    Examples:
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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create sample data
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 1  # Square mask
        >>> bboxes = np.array([[10, 10, 40, 40]])  # Single box
        >>> bbox_labels = [1]
        >>> keypoints = np.array([[50, 50]])  # Single keypoint at center
        >>> keypoint_labels = [0]
        >>>
        >>> # Set up transform with Compose to handle all targets
        >>> transform = A.Compose([
        ...     A.ThinPlateSpline(scale_range=(0.2, 0.4), p=1.0)
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>>
        >>> # Apply to all targets
        >>> result = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>>
        >>> # Access transformed results
        >>> transformed_image = result['image']
        >>> transformed_mask = result['mask']
        >>> transformed_bboxes = result['bboxes']
        >>> transformed_bbox_labels = result['bbox_labels']
        >>> transformed_keypoints = result['keypoints']
        >>> transformed_keypoint_labels = result['keypoint_labels']

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
