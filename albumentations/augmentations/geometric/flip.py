"""Geometric transformations for flip and symmetry operations.

This module contains transforms that apply various flip and symmetry operations
to images and other target types. These transforms modify the geometric arrangement
of the input data while preserving the pixel values themselves.

Available transforms:
- VerticalFlip: Flips the input upside down (around the x-axis)
- HorizontalFlip: Flips the input left to right (around the y-axis)
- Transpose: Swaps rows and columns (flips around the main diagonal)
- D4: Applies one of eight possible square symmetry transformations (dihedral group D4)
- SquareSymmetry: Alias for D4 with a more intuitive name

These transforms are particularly useful for:
- Data augmentation to improve model generalization
- Addressing orientation biases in training data
- Working with data that doesn't have a natural orientation (e.g., satellite imagery)
- Exploiting symmetries in the problem domain

All transforms support various target types including images, masks, bounding boxes,
keypoints, volumes, and 3D masks, ensuring consistent transformation across
different data modalities.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from albucore import batch_transform, hflip, vflip

from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import (
    ALL_TARGETS,
    d4_group_elements,
)

from . import functional as fgeometric

__all__ = [
    "D4",
    "HorizontalFlip",
    "SquareSymmetry",
    "Transpose",
    "VerticalFlip",
]


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

    Examples:
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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Prepare sample data
        >>> image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        >>> mask = np.array([[1, 0], [0, 1]])
        >>> bboxes = np.array([[0.1, 0.5, 0.3, 0.9]])  # [x_min, y_min, x_max, y_max] format
        >>> keypoints = np.array([[0.1, 0.5], [0.9, 0.5]])  # [x, y] format
        >>>
        >>> # Create a transform with horizontal flip
        >>> transform = A.Compose([
        ...     A.HorizontalFlip(p=1.0)  # Always apply for this example
        ... ], bbox_params=A.BboxParams(format='yolo', label_fields=[]),
        ...    keypoint_params=A.KeypointParams(format='normalized'))
        >>>
        >>> # Apply the transform
        >>> transformed = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
        >>>
        >>> # Get the transformed data
        >>> flipped_image = transformed["image"]  # Image flipped horizontally
        >>> flipped_mask = transformed["mask"]    # Mask flipped horizontally
        >>> flipped_bboxes = transformed["bboxes"]  # BBox coordinates adjusted for horizontal flip
        >>> flipped_keypoints = transformed["keypoints"]  # Keypoint x-coordinates flipped

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

    Examples:
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

    Examples:
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

    Examples:
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
