"""Transforms for rotating images and associated data.

This module provides classes for rotating images, masks, bounding boxes, and keypoints.
Includes transforms for 90-degree rotations and arbitrary angle rotations with various
border handling options.
"""

from __future__ import annotations

import math
from typing import Any, cast

import cv2
import numpy as np
from typing_extensions import Literal

from albumentations.augmentations.crops import functional as fcrops
from albumentations.augmentations.geometric.transforms import Affine
from albumentations.core.pydantic import SymmetricRangeType
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import ALL_TARGETS

from . import functional as fgeometric

__all__ = ["RandomRotate90", "Rotate", "SafeRotate"]

SMALL_NUMBER = 1e-10


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Even with p=1.0, the transform has a 1/4 probability of being identity:
    - With probability p * 1/4: no rotation (0 degrees)
    - With probability p * 1/4: rotate 90 degrees
    - With probability p * 1/4: rotate 180 degrees
    - With probability p * 1/4: rotate 270 degrees

    For example:
    - With p=1.0: Each rotation angle (including 0°) has 0.25 probability
    - With p=0.8: Each rotation angle has 0.2 probability, and no transform has 0.2 probability
    - With p=0.5: Each rotation angle has 0.125 probability, and no transform has 0.5 probability

    Common applications:
    - Aerial/satellite imagery: Objects can appear in any orientation
    - Medical imaging: Scans/slides may not have a consistent orientation
    - Document analysis: Pages or symbols might be rotated
    - Microscopy: Cell orientation is often arbitrary
    - Game development: Sprites/textures that should work in multiple orientations

    Not recommended for:
    - Natural scene images where gravity matters (e.g., landscape photography)
    - Face detection/recognition tasks
    - Text recognition (unless text can appear rotated)
    - Tasks where object orientation is important for classification

    Note:
        If your domain has both 90-degree rotation AND flip symmetries
        (e.g., satellite imagery, microscopy), consider using `D4` transform instead.
        `D4` is more efficient and mathematically correct as it:
        - Samples uniformly from all 8 possible combinations of rotations and flips
        - Properly represents the dihedral group D4 symmetries
        - Avoids potential correlation between separate rotation and flip augmentations

    Args:
        p (float): probability of applying the transform. Default: 1.0.
            Note that even with p=1.0, there's still a 0.25 probability
            of getting a 0-degree rotation (identity transform).

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> # Create example data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]  # Class labels for bounding boxes
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]  # Labels for keypoints
        >>> # Define the transform
        >>> transform = A.Compose([
        ...     A.RandomRotate90(p=1.0),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>> # Apply the transform to all targets
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>> rotated_image = transformed["image"]
        >>> rotated_mask = transformed["mask"]
        >>> rotated_bboxes = transformed["bboxes"]
        >>> rotated_bbox_labels = transformed["bbox_labels"]
        >>> rotated_keypoints = transformed["keypoints"]
        >>> rotated_keypoint_labels = transformed["keypoint_labels"]

    """

    _targets = ALL_TARGETS

    def __init__(
        self,
        p: float = 1,
    ):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, factor: Literal[0, 1, 2, 3], **params: Any) -> np.ndarray:
        """Apply rotation to the input image.

        Args:
            img (np.ndarray): Image to rotate.
            factor (Literal[0, 1, 2, 3]): Number of times to rotate by 90 degrees.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Rotated image.

        """
        return fgeometric.rot90(img, factor)

    def get_params(self) -> dict[str, int]:
        """Get parameters for the transform.

        Returns:
            dict[str, int]: Dictionary with the rotation factor.

        """
        # Random int in the range [0, 3]
        return {"factor": self.py_random.randint(0, 3)}

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        factor: Literal[0, 1, 2, 3],
        **params: Any,
    ) -> np.ndarray:
        """Apply rotation to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to rotate.
            factor (Literal[0, 1, 2, 3]): Number of times to rotate by 90 degrees.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Rotated bounding boxes.

        """
        return fgeometric.bboxes_rot90(bboxes, factor)

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        factor: Literal[0, 1, 2, 3],
        **params: Any,
    ) -> np.ndarray:
        """Apply rotation to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to rotate.
            factor (Literal[0, 1, 2, 3]): Number of times to rotate by 90 degrees.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Rotated keypoints.

        """
        return fgeometric.keypoints_rot90(keypoints, factor, params["shape"])

    def apply_to_volume(self, volume: np.ndarray, factor: Literal[0, 1, 2, 3], **params: Any) -> np.ndarray:
        """Apply rotation to the input volume.

        Args:
            volume (np.ndarray): Volume to rotate.
            factor (Literal[0, 1, 2, 3]): Number of times to rotate by 90 degrees.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Rotated volume.

        """
        return fgeometric.volume_rot90(volume, factor)

    def apply_to_volumes(self, volumes: np.ndarray, factor: Literal[0, 1, 2, 3], **params: Any) -> np.ndarray:
        """Apply rotation to the input volumes.

        Args:
            volumes (np.ndarray): Volumes to rotate.
            factor (Literal[0, 1, 2, 3]): Number of times to rotate by 90 degrees.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Rotated volumes.

        """
        return fgeometric.volumes_rot90(volumes, factor)

    def apply_to_mask3d(self, mask3d: np.ndarray, factor: Literal[0, 1, 2, 3], **params: Any) -> np.ndarray:
        """Apply rotation to the input mask3d.

        Args:
            mask3d (np.ndarray): Mask3d to rotate.
            factor (Literal[0, 1, 2, 3]): Number of times to rotate by 90 degrees.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Rotated mask3d.

        """
        return fgeometric.volume_rot90(mask3d, factor)

    def apply_to_masks3d(self, masks3d: np.ndarray, factor: Literal[0, 1, 2, 3], **params: Any) -> np.ndarray:
        """Apply rotation to the input masks3d.

        Args:
            masks3d (np.ndarray): Masks3d to rotate.
            factor (Literal[0, 1, 2, 3]): Number of times to rotate by 90 degrees.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Rotated masks3d.

        """
        return fgeometric.volumes_rot90(masks3d, factor)


class RotateInitSchema(BaseTransformInitSchema):
    limit: SymmetricRangeType

    interpolation: Literal[cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    mask_interpolation: Literal[
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
    fill_mask: tuple[float, ...] | float | None


class Rotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.

    Args:
        limit (float | tuple[float, float]): Range from which a random angle is picked. If limit is a single float,
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): Flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_CONSTANT
        fill (tuple[float, ...] | float): Padding value if border_mode is cv2.BORDER_CONSTANT.
        fill_mask (tuple[float, ...] | float): Padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        rotate_method (Literal["largest_box", "ellipse"]): Method to rotate bounding boxes.
            Should be 'largest_box' or 'ellipse'. Default: 'largest_box'
        crop_border (bool): Whether to crop border after rotation. If True, the output image size might differ
            from the input. Default: False
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - The rotation angle is randomly selected for each execution within the range specified by 'limit'.
        - When 'crop_border' is False, the output image will have the same size as the input, potentially
          introducing black triangles in the corners.
        - When 'crop_border' is True, the output image is cropped to remove black triangles, which may result
          in a smaller image.
        - Bounding boxes are rotated and may change size or shape.
        - Keypoints are rotated around the center of the image.

    Mathematical Details:
        1. An angle θ is randomly sampled from the range specified by 'limit'.
        2. The image is rotated around its center by θ degrees.
        3. The rotation matrix R is:
           R = [cos(θ)  -sin(θ)]
               [sin(θ)   cos(θ)]
        4. Each point (x, y) in the image is transformed to (x', y') by:
           [x']   [cos(θ)  -sin(θ)] [x - cx]   [cx]
           [y'] = [sin(θ)   cos(θ)] [y - cy] + [cy]
           where (cx, cy) is the center of the image.
        5. If 'crop_border' is True, the image is cropped to the largest rectangle that fits inside the rotated image.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> # Create example data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]  # Class labels for bounding boxes
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]  # Labels for keypoints
        >>> # Define the transform
        >>> transform = A.Compose([
        ...     A.Rotate(limit=45, p=1.0),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>> # Apply the transform to all targets
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>> rotated_image = transformed["image"]
        >>> rotated_mask = transformed["mask"]
        >>> rotated_bboxes = transformed["bboxes"]
        >>> rotated_bbox_labels = transformed["bbox_labels"]
        >>> rotated_keypoints = transformed["keypoints"]
        >>> rotated_keypoint_labels = transformed["keypoint_labels"]

    """

    _targets = ALL_TARGETS

    class InitSchema(RotateInitSchema):
        rotate_method: Literal["largest_box", "ellipse"]
        crop_border: bool

        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        limit: tuple[float, float] | float = (-90, 90),
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_LINEAR,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        crop_border: bool = False,
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
        super().__init__(p=p)
        self.limit = cast("tuple[float, float]", limit)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask
        self.rotate_method = rotate_method
        self.crop_border = crop_border

    def apply(
        self,
        img: np.ndarray,
        matrix: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply affine transformation to the image.

        Args:
            img (np.ndarray): Image to transform.
            matrix (np.ndarray): Affine transformation matrix.
            x_min (int): Minimum x-coordinate for cropping (if crop_border is True).
            x_max (int): Maximum x-coordinate for cropping (if crop_border is True).
            y_min (int): Minimum y-coordinate for cropping (if crop_border is True).
            y_max (int): Maximum y-coordinate for cropping (if crop_border is True).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transformed image.

        """
        img_out = fgeometric.warp_affine(
            img,
            matrix,
            self.interpolation,
            self.fill,
            self.border_mode,
            params["shape"][:2],
        )
        if self.crop_border:
            return fcrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_mask(
        self,
        mask: np.ndarray,
        matrix: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply affine transformation to the mask.

        Args:
            mask (np.ndarray): Mask to transform.
            matrix (np.ndarray): Affine transformation matrix.
            x_min (int): Minimum x-coordinate for cropping (if crop_border is True).
            x_max (int): Maximum x-coordinate for cropping (if crop_border is True).
            y_min (int): Minimum y-coordinate for cropping (if crop_border is True).
            y_max (int): Maximum y-coordinate for cropping (if crop_border is True).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transformed mask.

        """
        img_out = fgeometric.warp_affine(
            mask,
            matrix,
            self.mask_interpolation,
            self.fill_mask,
            self.border_mode,
            params["shape"][:2],
        )
        if self.crop_border:
            return fcrops.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        bbox_matrix: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply affine transformation to bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes to transform.
            bbox_matrix (np.ndarray): Affine transformation matrix for bounding boxes.
            x_min (int): Minimum x-coordinate for cropping (if crop_border is True).
            x_max (int): Maximum x-coordinate for cropping (if crop_border is True).
            y_min (int): Minimum y-coordinate for cropping (if crop_border is True).
            y_max (int): Maximum y-coordinate for cropping (if crop_border is True).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transformed bounding boxes.

        """
        image_shape = params["shape"][:2]
        bboxes_out = fgeometric.bboxes_affine(
            bboxes,
            bbox_matrix,
            self.rotate_method,
            image_shape,
            self.border_mode,
            image_shape,
        )
        if self.crop_border:
            return fcrops.crop_bboxes_by_coords(
                bboxes_out,
                (x_min, y_min, x_max, y_max),
                image_shape,
            )
        return bboxes_out

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        matrix: np.ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply affine transformation to keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to transform.
            matrix (np.ndarray): Affine transformation matrix.
            x_min (int): Minimum x-coordinate for cropping (if crop_border is True).
            x_max (int): Maximum x-coordinate for cropping (if crop_border is True).
            y_min (int): Minimum y-coordinate for cropping (if crop_border is True).
            y_max (int): Maximum y-coordinate for cropping (if crop_border is True).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Transformed keypoints.

        """
        keypoints_out = fgeometric.keypoints_affine(
            keypoints,
            matrix,
            params["shape"][:2],
            scale={"x": 1, "y": 1},
            border_mode=self.border_mode,
        )
        if self.crop_border:
            return fcrops.crop_keypoints_by_coords(
                keypoints_out,
                (x_min, y_min, x_max, y_max),
            )
        return keypoints_out

    @staticmethod
    def _rotated_rect_with_max_area(
        height: int,
        width: int,
        angle: float,
    ) -> dict[str, int]:
        """Given a rectangle of size wxh that has been rotated by 'angle' (in
        degrees), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.

        References:
            Rotate image and crop out black borders: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

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
            wr, hr = (
                (width * cos_a - height * sin_a) / cos_2a,
                (height * cos_a - width * sin_a) / cos_2a,
            )

        return {
            "x_min": max(0, int(width / 2 - wr / 2)),
            "x_max": min(width, int(width / 2 + wr / 2)),
            "y_min": max(0, int(height / 2 - hr / 2)),
            "y_max": min(height, int(height / 2 + hr / 2)),
        }

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get parameters dependent on the data.

        Args:
            params (dict[str, Any]): Dictionary containing parameters.
            data (dict[str, Any]): Dictionary containing data.

        Returns:
            dict[str, Any]: Dictionary with parameters for transformation.

        """
        angle = self.py_random.uniform(*self.limit)

        if self.crop_border:
            height, width = params["shape"][:2]
            out_params = self._rotated_rect_with_max_area(height, width, angle)
        else:
            out_params = {"x_min": -1, "x_max": -1, "y_min": -1, "y_max": -1}

        center = fgeometric.center(params["shape"][:2])
        bbox_center = fgeometric.center_bbox(params["shape"][:2])

        translate: dict[str, int] = {"x": 0, "y": 0}
        shear: dict[str, float] = {"x": 0, "y": 0}
        scale: dict[str, float] = {"x": 1, "y": 1}
        rotate = angle

        matrix = fgeometric.create_affine_transformation_matrix(
            translate,
            shear,
            scale,
            rotate,
            center,
        )
        bbox_matrix = fgeometric.create_affine_transformation_matrix(
            translate,
            shear,
            scale,
            rotate,
            bbox_center,
        )
        out_params["matrix"] = matrix
        out_params["bbox_matrix"] = bbox_matrix

        return out_params


class SafeRotate(Affine):
    """Rotate the input inside the input's frame by an angle selected randomly from the uniform distribution.

    This transformation ensures that the entire rotated image fits within the original frame by scaling it
    down if necessary. The resulting image maintains its original dimensions but may contain artifacts due to the
    rotation and scaling process.

    Args:
        limit (float | tuple[float, float]): Range from which a random angle is picked. If limit is a single float,
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): Flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        fill (tuple[float, float] | float): Padding value if border_mode is cv2.BORDER_CONSTANT.
        fill_mask (tuple[float, float] | float): Padding value if border_mode is cv2.BORDER_CONSTANT applied
            for masks.
        rotate_method (Literal["largest_box", "ellipse"]): Method to rotate bounding boxes.
            Should be 'largest_box' or 'ellipse'. Default: 'largest_box'
        mask_interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm for mask.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_NEAREST.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - The rotation is performed around the center of the image.
        - After rotation, the image is scaled to fit within the original frame, which may cause some distortion.
        - The output image will always have the same dimensions as the input image.
        - Bounding boxes and keypoints are transformed along with the image.

    Mathematical Details:
        1. An angle θ is randomly sampled from the range specified by 'limit'.
        2. The image is rotated around its center by θ degrees.
        3. The rotation matrix R is:
           R = [cos(θ)  -sin(θ)]
               [sin(θ)   cos(θ)]
        4. The scaling factor s is calculated to ensure the rotated image fits within the original frame:
           s = min(width / (width * |cos(θ)| + height * |sin(θ)|),
                   height / (width * |sin(θ)| + height * |cos(θ)|))
        5. The combined transformation matrix T is:
           T = [s*cos(θ)  -s*sin(θ)  tx]
               [s*sin(θ)   s*cos(θ)  ty]
           where tx and ty are translation factors to keep the image centered.
        6. Each point (x, y) in the image is transformed to (x', y') by:
           [x']   [s*cos(θ)   s*sin(θ)] [x - cx]   [cx]
           [y'] = [-s*sin(θ)  s*cos(θ)] [y - cy] + [cy]
           where (cx, cy) is the center of the image.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> # Create example data
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> bboxes = np.array([[10, 10, 50, 50], [40, 40, 80, 80]], dtype=np.float32)
        >>> bbox_labels = [1, 2]  # Class labels for bounding boxes
        >>> keypoints = np.array([[20, 30], [60, 70]], dtype=np.float32)
        >>> keypoint_labels = [0, 1]  # Labels for keypoints
        >>> # Define the transform
        >>> transform = A.Compose([
        ...     A.SafeRotate(limit=45, p=1.0),
        ... ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_labels']),
        ...    keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels']))
        >>> # Apply the transform to all targets
        >>> transformed = transform(
        ...     image=image,
        ...     mask=mask,
        ...     bboxes=bboxes,
        ...     bbox_labels=bbox_labels,
        ...     keypoints=keypoints,
        ...     keypoint_labels=keypoint_labels
        ... )
        >>> rotated_image = transformed["image"]
        >>> rotated_mask = transformed["mask"]
        >>> rotated_bboxes = transformed["bboxes"]
        >>> rotated_bbox_labels = transformed["bbox_labels"]
        >>> rotated_keypoints = transformed["keypoints"]
        >>> rotated_keypoint_labels = transformed["keypoint_labels"]

    """

    _targets = ALL_TARGETS

    class InitSchema(RotateInitSchema):
        rotate_method: Literal["largest_box", "ellipse"]

    def __init__(
        self,
        limit: tuple[float, float] | float = (-90, 90),
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ] = cv2.INTER_LINEAR,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
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
        super().__init__(
            rotate=limit,
            interpolation=interpolation,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            rotate_method=rotate_method,
            fit_output=True,
            mask_interpolation=mask_interpolation,
            p=p,
        )
        self.limit = cast("tuple[float, float]", limit)

    def _create_safe_rotate_matrix(
        self,
        angle: float,
        center: tuple[float, float],
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, dict[str, float]]:
        height, width = image_shape[:2]
        rotation_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new image size
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        new_w = int(height * abs_sin + width * abs_cos)
        new_h = int(height * abs_cos + width * abs_sin)

        # Adjust the rotation matrix to take into account the new size
        rotation_mat[0, 2] += new_w / 2 - center[0]
        rotation_mat[1, 2] += new_h / 2 - center[1]

        # Calculate scaling factors
        scale_x = width / new_w
        scale_y = height / new_h

        # Create scaling matrix
        scale_mat = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

        # Combine rotation and scaling
        matrix = scale_mat @ np.vstack([rotation_mat, [0, 0, 1]])

        return matrix, {"x": scale_x, "y": scale_y}

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get parameters dependent on the data.

        Args:
            params (dict[str, Any]): Dictionary containing parameters.
            data (dict[str, Any]): Dictionary containing data.

        Returns:
            dict[str, Any]: Dictionary with parameters for transformation.

        """
        image_shape = params["shape"][:2]
        angle = self.py_random.uniform(*self.limit)

        # Calculate centers for image and bbox
        image_center = fgeometric.center(image_shape)
        bbox_center = fgeometric.center_bbox(image_shape)

        # Create matrices for image and bbox
        matrix, scale = self._create_safe_rotate_matrix(
            angle,
            image_center,
            image_shape,
        )
        bbox_matrix, _ = self._create_safe_rotate_matrix(
            angle,
            bbox_center,
            image_shape,
        )

        return {
            "rotate": angle,
            "scale": scale,
            "matrix": matrix,
            "bbox_matrix": bbox_matrix,
            "output_shape": image_shape,
        }
