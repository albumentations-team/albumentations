"""Functional implementations of geometric image transformations.

This module provides low-level functions for geometric operations such as rotation,
resizing, flipping, perspective transforms, and affine transformations on images,
bounding boxes and keypoints.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast
from warnings import warn

import cv2
import numpy as np
from albucore import (
    get_num_channels,
    hflip,
    maybe_process_in_chunks,
    preserve_channel_dim,
    vflip,
)

from albumentations.augmentations.utils import angle_2pi_range, handle_empty_array
from albumentations.core.bbox_utils import (
    bboxes_from_masks,
    bboxes_to_mask,
    denormalize_bboxes,
    mask_to_bboxes,
    masks_from_bboxes,
    normalize_bboxes,
)
from albumentations.core.type_definitions import (
    NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS,
    NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    REFLECT_BORDER_MODES,
)

PAIR = 2

ROT90_180_FACTOR = 2
ROT90_270_FACTOR = 3


@handle_empty_array("bboxes")
def bboxes_rot90(bboxes: np.ndarray, factor: int) -> np.ndarray:
    """Rotates bounding boxes by 90 degrees CCW (see np.rot90)

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)
        factor (int): Number of 90-degree rotations (1, 2, or 3)

    Returns:
        np.ndarray: Rotated bounding boxes

    """
    if factor == 0:
        return bboxes

    rotated_bboxes = bboxes.copy()
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    if factor == 1:
        rotated_bboxes[:, 0] = y_min
        rotated_bboxes[:, 1] = 1 - x_max
        rotated_bboxes[:, 2] = y_max
        rotated_bboxes[:, 3] = 1 - x_min
    elif factor == ROT90_180_FACTOR:
        rotated_bboxes[:, 0] = 1 - x_max
        rotated_bboxes[:, 1] = 1 - y_max
        rotated_bboxes[:, 2] = 1 - x_min
        rotated_bboxes[:, 3] = 1 - y_min
    elif factor == ROT90_270_FACTOR:
        rotated_bboxes[:, 0] = 1 - y_max
        rotated_bboxes[:, 1] = x_min
        rotated_bboxes[:, 2] = 1 - y_min
        rotated_bboxes[:, 3] = x_max

    return rotated_bboxes


@handle_empty_array("bboxes")
def bboxes_d4(
    bboxes: np.ndarray,
    group_member: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
) -> np.ndarray:
    """Applies a `D_4` symmetry group transformation to a bounding box.

    The function transforms a bounding box according to the specified group member from the `D_4` group.
    These transformations include rotations and reflections, specified to work on an image's bounding box given
    its dimensions.

    Args:
        bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).
        Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).
        group_member (Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]): A string identifier for the
            `D_4` group transformation to apply.

    Returns:
        BoxInternalType: The transformed bounding box.

    Raises:
        ValueError: If an invalid group member is specified.

    """
    transformations = {
        "e": lambda x: x,  # Identity transformation
        "r90": lambda x: bboxes_rot90(x, 1),  # Rotate 90 degrees
        "r180": lambda x: bboxes_rot90(x, 2),  # Rotate 180 degrees
        "r270": lambda x: bboxes_rot90(x, 3),  # Rotate 270 degrees
        "v": lambda x: bboxes_vflip(x),  # Vertical flip
        "hvt": lambda x: bboxes_transpose(
            bboxes_rot90(x, 2),
        ),  # Reflect over anti-diagonal
        "h": lambda x: bboxes_hflip(x),  # Horizontal flip
        "t": lambda x: bboxes_transpose(x),  # Transpose (reflect over main diagonal)
    }

    # Execute the appropriate transformation
    if group_member in transformations:
        return transformations[group_member](bboxes)

    raise ValueError(f"Invalid group member: {group_member}")


@handle_empty_array("keypoints")
@angle_2pi_range
def keypoints_rot90(
    keypoints: np.ndarray,
    factor: Literal[0, 1, 2, 3],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rotate keypoints by 90 degrees counter-clockwise (CCW) a specified number of times.

    Args:
        keypoints (np.ndarray): An array of keypoints with shape (N, 4+) in the format (x, y, angle, scale, ...).
        factor (int): The number of 90 degree CCW rotations to apply. Must be in the range [0, 3].
        image_shape (tuple[int, int]): The shape of the image (height, width).

    Returns:
        np.ndarray: The rotated keypoints with the same shape as the input.

    """
    if factor == 0:
        return keypoints

    height, width = image_shape[:2]
    rotated_keypoints = keypoints.copy().astype(np.float32)

    x, y, angle = keypoints[:, 0], keypoints[:, 1], keypoints[:, 3]

    if factor == 1:
        rotated_keypoints[:, 0] = y
        rotated_keypoints[:, 1] = width - 1 - x
        rotated_keypoints[:, 3] = angle - np.pi / 2
    elif factor == ROT90_180_FACTOR:
        rotated_keypoints[:, 0] = width - 1 - x
        rotated_keypoints[:, 1] = height - 1 - y
        rotated_keypoints[:, 3] = angle - np.pi
    elif factor == ROT90_270_FACTOR:
        rotated_keypoints[:, 0] = height - 1 - y
        rotated_keypoints[:, 1] = x
        rotated_keypoints[:, 3] = angle + np.pi / 2

    return rotated_keypoints


@handle_empty_array("keypoints")
def keypoints_d4(
    keypoints: np.ndarray,
    group_member: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"],
    image_shape: tuple[int, int],
    **params: Any,
) -> np.ndarray:
    """Applies a `D_4` symmetry group transformation to a keypoint.

    This function adjusts a keypoint's coordinates according to the specified `D_4` group transformation,
    which includes rotations and reflections suitable for image processing tasks. These transformations account
    for the dimensions of the image to ensure the keypoint remains within its boundaries.

    Args:
        keypoints (np.ndarray): An array of keypoints with shape (N, 4+) in the format (x, y, angle, scale, ...).
        group_member (Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]): A string identifier for
            the `D_4` group transformation to apply.
            Valid values are 'e', 'r90', 'r180', 'r270', 'v', 'hv', 'h', 't'.
        image_shape (tuple[int, int]): The shape of the image.
        params (Any): Not used.

    Returns:
        KeypointInternalType: The transformed keypoint.

    Raises:
        ValueError: If an invalid group member is specified, indicating that the specified transformation
            does not exist.

    """
    rows, cols = image_shape[:2]
    transformations = {
        "e": lambda x: x,  # Identity transformation
        "r90": lambda x: keypoints_rot90(x, 1, image_shape),  # Rotate 90 degrees
        "r180": lambda x: keypoints_rot90(x, 2, image_shape),  # Rotate 180 degrees
        "r270": lambda x: keypoints_rot90(x, 3, image_shape),  # Rotate 270 degrees
        "v": lambda x: keypoints_vflip(x, rows),  # Vertical flip
        "hvt": lambda x: keypoints_transpose(
            keypoints_rot90(x, 2, image_shape),
        ),  # Reflect over anti diagonal
        "h": lambda x: keypoints_hflip(x, cols),  # Horizontal flip
        "t": lambda x: keypoints_transpose(x),  # Transpose (reflect over main diagonal)
    }
    # Execute the appropriate transformation
    if group_member in transformations:
        return transformations[group_member](keypoints)

    raise ValueError(f"Invalid group member: {group_member}")


@preserve_channel_dim
def resize(
    img: np.ndarray,
    target_shape: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    """Resize an image to the specified dimensions.

    This function resizes an input image to the target shape using the specified
    interpolation method. If the image is already the target size, it is returned unchanged.

    Args:
        img (np.ndarray): Input image to resize.
        target_shape (tuple[int, int]): Target (height, width) dimensions.
        interpolation (int): Interpolation method to use (cv2 interpolation flag).
            Examples: cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, etc.

    Returns:
        np.ndarray: Resized image with shape target_shape + original channel dimensions.

    """
    if target_shape == img.shape[:2]:
        return img

    height, width = target_shape[:2]
    resize_fn = maybe_process_in_chunks(
        cv2.resize,
        dsize=(width, height),
        interpolation=interpolation,
    )
    return resize_fn(img)


@preserve_channel_dim
def scale(img: np.ndarray, scale: float, interpolation: int) -> np.ndarray:
    """Scale an image by a factor while preserving aspect ratio.

    This function scales both height and width dimensions of the image by the same factor.

    Args:
        img (np.ndarray): Input image to scale.
        scale (float): Scale factor. Values > 1 will enlarge the image, values < 1 will shrink it.
        interpolation (int): Interpolation method to use (cv2 interpolation flag).

    Returns:
        np.ndarray: Scaled image.

    """
    height, width = img.shape[:2]
    new_size = int(height * scale), int(width * scale)
    return resize(img, new_size, interpolation)


@handle_empty_array("keypoints")
def keypoints_scale(
    keypoints: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Scale keypoints by given factors.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+)
        scale_x (float): Scale factor for x coordinates
        scale_y (float): Scale factor for y coordinates

    Returns:
        np.ndarray: Scaled keypoints

    """
    # Extract x, y, z, angle, and scale
    x, y, z, angle, scale = (
        keypoints[:, 0],
        keypoints[:, 1],
        keypoints[:, 2],
        keypoints[:, 3],
        keypoints[:, 4],
    )

    # Scale x and y
    x_scaled = x * scale_x
    y_scaled = y * scale_y

    # Scale the keypoint scale by the maximum of scale_x and scale_y
    scale_scaled = scale * max(scale_x, scale_y)

    # Create the output array
    scaled_keypoints = np.column_stack([x_scaled, y_scaled, z, angle, scale_scaled])

    # If there are additional columns, preserve them
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        return np.column_stack(
            [scaled_keypoints, keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:]],
        )

    return scaled_keypoints


@preserve_channel_dim
def perspective(
    img: np.ndarray,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    border_val: float | list[float] | np.ndarray,
    border_mode: int,
    keep_size: bool,
    interpolation: int,
) -> np.ndarray:
    """Apply perspective transformation to an image.

    This function warps an image according to a perspective transformation matrix.
    It can either maintain the original dimensions or use the specified max dimensions.

    Args:
        img (np.ndarray): Input image to transform.
        matrix (np.ndarray): 3x3 perspective transformation matrix.
        max_width (int): Maximum width of the output image if keep_size is False.
        max_height (int): Maximum height of the output image if keep_size is False.
        border_val (float | list[float] | np.ndarray): Border value(s) to fill areas outside the transformed image.
        border_mode (int): OpenCV border mode (e.g., cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT).
        keep_size (bool): If True, maintain the original image dimensions.
        interpolation (int): Interpolation method for resampling (cv2 interpolation flag).

    Returns:
        np.ndarray: Perspective-transformed image.

    """
    if not keep_size:
        perspective_func = maybe_process_in_chunks(
            cv2.warpPerspective,
            M=matrix,
            dsize=(max_width, max_height),
            borderMode=border_mode,
            borderValue=border_val,
            flags=interpolation,
        )
    else:
        height, width = img.shape[:2]

        scale_x = width / max_width
        scale_y = height / max_height
        scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
        adjusted_matrix = np.dot(scale_matrix, matrix)

        perspective_func = maybe_process_in_chunks(
            cv2.warpPerspective,
            M=adjusted_matrix,
            dsize=(width, height),
            borderMode=border_mode,
            borderValue=border_val,
            flags=interpolation,
        )

    return perspective_func(img)


@handle_empty_array("bboxes")
def perspective_bboxes(
    bboxes: np.ndarray,
    image_shape: tuple[int, int],
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> np.ndarray:
    """Applies perspective transformation to bounding boxes.

    This function transforms bounding boxes using the given perspective transformation matrix.
    It handles bounding boxes with additional attributes beyond the standard coordinates.

    Args:
        bboxes (np.ndarray): An array of bounding boxes with shape (num_bboxes, 4+).
                             Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).
                             Additional columns beyond the first 4 are preserved unchanged.
        image_shape (tuple[int, int]): The shape of the image (height, width).
        matrix (np.ndarray): The perspective transformation matrix.
        max_width (int): The maximum width of the output image.
        max_height (int): The maximum height of the output image.
        keep_size (bool): If True, maintains the original image size after transformation.

    Returns:
        np.ndarray: An array of transformed bounding boxes with the same shape as input.
                    The first 4 columns contain the transformed coordinates, and any
                    additional columns are preserved from the input.

    Note:
        - This function modifies only the coordinate columns (first 4) of the input bounding boxes.
        - Any additional attributes (columns beyond the first 4) are kept unchanged.
        - The function handles denormalization and renormalization of coordinates internally.

    Example:
        >>> bboxes = np.array([[0.1, 0.1, 0.3, 0.3, 1], [0.5, 0.5, 0.8, 0.8, 2]])
        >>> image_shape = (100, 100)
        >>> matrix = np.array([[1.5, 0.2, -20], [-0.1, 1.3, -10], [0.002, 0.001, 1]])
        >>> transformed_bboxes = perspective_bboxes(bboxes, image_shape, matrix, 150, 150, False)

    """
    height, width = image_shape[:2]
    transformed_bboxes = bboxes.copy()
    denormalized_coords = denormalize_bboxes(bboxes[:, :4], image_shape)

    x_min, y_min, x_max, y_max = denormalized_coords.T
    points = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
    ).transpose(2, 0, 1)
    points_reshaped = points.reshape(-1, 1, 2)

    transformed_points = cv2.perspectiveTransform(
        points_reshaped.astype(np.float32),
        matrix,
    )
    transformed_points = transformed_points.reshape(-1, 4, 2)

    new_coords = np.array(
        [[np.min(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 0]), np.max(box[:, 1])] for box in transformed_points],
    )

    if keep_size:
        scale_x, scale_y = width / max_width, height / max_height
        new_coords[:, [0, 2]] *= scale_x
        new_coords[:, [1, 3]] *= scale_y
        output_shape = image_shape
    else:
        output_shape = (max_height, max_width)

    normalized_coords = normalize_bboxes(new_coords, output_shape)
    transformed_bboxes[:, :4] = normalized_coords

    return transformed_bboxes


def rotation2d_matrix_to_euler_angles(matrix: np.ndarray, y_up: bool) -> float:
    """Args:
    matrix (np.ndarray): Rotation matrix
    y_up (bool): is Y axis looks up or down

    """
    if y_up:
        return np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.arctan2(-matrix[1, 0], matrix[0, 0])


@handle_empty_array("keypoints")
@angle_2pi_range
def perspective_keypoints(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> np.ndarray:
    """Apply perspective transformation to keypoints.

    Args:
        keypoints (np.ndarray): Array of shape (N, 5+) in format [x, y, z, angle, scale, ...]
        image_shape (tuple[int, int]): Original image shape (height, width)
        matrix (np.ndarray): 3x3 perspective transformation matrix
        max_width (int): Maximum width after transformation
        max_height (int): Maximum height after transformation
        keep_size (bool): Whether to keep original size

    Returns:
        np.ndarray: Transformed keypoints array with same shape as input

    """
    keypoints = keypoints.copy().astype(np.float32)

    height, width = image_shape[:2]

    x, y, z, angle, scale = (
        keypoints[:, 0],
        keypoints[:, 1],
        keypoints[:, 2],
        keypoints[:, 3],
        keypoints[:, 4],
    )

    # Reshape keypoints for perspective transform
    keypoint_vector = np.column_stack((x, y)).astype(np.float32).reshape(-1, 1, 2)

    # Apply perspective transform
    transformed_points = cv2.perspectiveTransform(keypoint_vector, matrix).squeeze()

    # Unsqueeze if we have a single keypoint
    if transformed_points.ndim == 1:
        transformed_points = transformed_points[np.newaxis, :]

    x, y = transformed_points[:, 0], transformed_points[:, 1]

    # Update angles
    angle += rotation2d_matrix_to_euler_angles(matrix[:2, :2], y_up=True)

    # Calculate scale factors
    scale_x = np.sign(matrix[0, 0]) * np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sign(matrix[1, 1]) * np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    scale *= max(scale_x, scale_y)

    if keep_size:
        scale_x = width / max_width
        scale_y = height / max_height
        x *= scale_x
        y *= scale_y
        scale *= max(scale_x, scale_y)

    # Create the output array with unchanged z coordinate
    transformed_keypoints = np.column_stack([x, y, z, angle, scale])

    # If there are additional columns, preserve them
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        return np.column_stack(
            [
                transformed_keypoints,
                keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:],
            ],
        )

    return transformed_keypoints


def is_identity_matrix(matrix: np.ndarray) -> bool:
    """Check if the given matrix is an identity matrix.

    Args:
        matrix (np.ndarray): A 3x3 affine transformation matrix.

    Returns:
        bool: True if the matrix is an identity matrix, False otherwise.

    """
    return np.allclose(matrix, np.eye(3, dtype=matrix.dtype))


def warp_affine_with_value_extension(
    image: np.ndarray,
    matrix: np.ndarray,
    dsize: tuple[int, int],
    flags: int,
    border_mode: int,
    border_value: tuple[float, ...] | float,
) -> np.ndarray:
    """Warp affine with value extension.

    This function warps an image with a given affine transformation matrix.
    It also extends the value to a sequence of floats.

    Args:
        image (np.ndarray): The image to warp.
        matrix (np.ndarray): The affine transformation matrix.
        dsize (tuple[int, int]): The size of the output image.
        flags (int): The flags for the warp.
        border_mode (int): The border mode to use.
        border_value (tuple[float, ...] | float): The value to pad the image with.

    Returns:
        np.ndarray: The warped image.

    """
    num_channels = get_num_channels(image)
    extended_value = extend_value(border_value, num_channels)

    return cv2.warpAffine(
        image,
        matrix,
        dsize,
        flags=flags,
        borderMode=border_mode,
        borderValue=extended_value,
    )


@preserve_channel_dim
def warp_affine(
    image: np.ndarray,
    matrix: np.ndarray,
    interpolation: int,
    fill: tuple[float, ...] | float,
    border_mode: int,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Apply an affine transformation to an image.

    This function transforms an image using the specified affine transformation matrix.
    If the transformation matrix is an identity matrix, the original image is returned.

    Args:
        image (np.ndarray): Input image to transform.
        matrix (np.ndarray): 2x3 or 3x3 affine transformation matrix.
        interpolation (int): Interpolation method for resampling.
        fill (tuple[float, ...] | float): Border value(s) to fill areas outside the transformed image.
        border_mode (int): OpenCV border mode for handling pixels outside the image boundaries.
        output_shape (tuple[int, int]): Shape (height, width) of the output image.

    Returns:
        np.ndarray: Affine-transformed image with dimensions specified by output_shape.

    """
    if is_identity_matrix(matrix):
        return image

    height = int(np.round(output_shape[0]))
    width = int(np.round(output_shape[1]))

    cv2_matrix = matrix[:2, :]

    warp_fn = maybe_process_in_chunks(
        warp_affine_with_value_extension,
        matrix=cv2_matrix,
        dsize=(width, height),
        flags=interpolation,
        border_mode=border_mode,
        border_value=fill,
    )
    return warp_fn(image)


@handle_empty_array("keypoints")
@angle_2pi_range
def keypoints_affine(
    keypoints: np.ndarray,
    matrix: np.ndarray,
    image_shape: tuple[int, int],
    scale: dict[str, float],
    border_mode: int,
) -> np.ndarray:
    """Apply an affine transformation to keypoints.

    This function transforms keypoints using the given affine transformation matrix.
    It handles reflection padding if necessary, updates coordinates, angles, and scales.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 4+) where N is the number of keypoints.
                                Each keypoint is represented as [x, y, angle, scale, ...].
        matrix (np.ndarray): The 2x3 or 3x3 affine transformation matrix.
        image_shape (tuple[int, int]): Shape of the image (height, width).
        scale (dict[str, float]): Dictionary containing scale factors for x and y directions.
                                  Expected keys are 'x' and 'y'.
        border_mode (int): Border mode for handling keypoints near image edges.
                            Use cv2.BORDER_REFLECT_101, cv2.BORDER_REFLECT, etc.

    Returns:
        np.ndarray: Transformed keypoints array with the same shape as input.

    Notes:
        - The function applies reflection padding if the mode is in REFLECT_BORDER_MODES.
        - Coordinates (x, y) are transformed using the affine matrix.
        - Angles are adjusted based on the rotation component of the affine transformation.
        - Scales are multiplied by the maximum of x and y scale factors.
        - The @angle_2pi_range decorator ensures angles remain in the [0, 2π] range.

    Example:
        >>> keypoints = np.array([[100, 100, 0, 1]])
        >>> matrix = np.array([[1.5, 0, 10], [0, 1.2, 20]])
        >>> scale = {'x': 1.5, 'y': 1.2}
        >>> transformed_keypoints = keypoints_affine(keypoints, matrix, (480, 640), scale, cv2.BORDER_REFLECT_101)

    """
    keypoints = keypoints.copy().astype(np.float32)

    if is_identity_matrix(matrix):
        return keypoints

    if border_mode in REFLECT_BORDER_MODES:
        # Step 1: Compute affine transform padding
        pad_left, pad_right, pad_top, pad_bottom = calculate_affine_transform_padding(
            matrix,
            image_shape,
        )
        grid_dimensions = get_pad_grid_dimensions(
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            image_shape,
        )
        keypoints = generate_reflected_keypoints(
            keypoints,
            grid_dimensions,
            image_shape,
            center_in_origin=True,
        )

    # Extract x, y coordinates (z is preserved)
    xy = keypoints[:, :2]

    # Ensure matrix is 2x3
    if matrix.shape == (3, 3):
        matrix = matrix[:2]

    # Transform x, y coordinates
    xy_transformed = cv2.transform(xy.reshape(-1, 1, 2), matrix).squeeze()

    # Calculate angle adjustment
    angle_adjustment = rotation2d_matrix_to_euler_angles(matrix[:2, :2], y_up=False)

    # Update angles (now at index 3)
    keypoints[:, 3] = keypoints[:, 3] + angle_adjustment

    # Update scales (now at index 4)
    max_scale = max(scale["x"], scale["y"])
    keypoints[:, 4] *= max_scale

    # Update x, y coordinates and preserve z
    keypoints[:, :2] = xy_transformed

    return keypoints


@handle_empty_array("points")
def apply_affine_to_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply affine transformation to a set of points.

    This function handles potential division by zero by replacing zero values
    in the homogeneous coordinate with a small epsilon value.

    Args:
        points (np.ndarray): Array of points with shape (N, 2).
        matrix (np.ndarray): 3x3 affine transformation matrix.

    Returns:
        np.ndarray: Transformed points with shape (N, 2).

    """
    homogeneous_points = np.column_stack([points, np.ones(points.shape[0])])
    transformed_points = homogeneous_points @ matrix.T

    # Handle potential division by zero
    epsilon = np.finfo(transformed_points.dtype).eps
    transformed_points[:, 2] = np.where(
        np.abs(transformed_points[:, 2]) < epsilon,
        np.sign(transformed_points[:, 2]) * epsilon,
        transformed_points[:, 2],
    )

    return transformed_points[:, :2] / transformed_points[:, 2:]


def calculate_affine_transform_padding(
    matrix: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Calculate the necessary padding for an affine transformation to avoid empty spaces."""
    height, width = image_shape[:2]

    # Check for identity transform
    if is_identity_matrix(matrix):
        return (0, 0, 0, 0)

    # Original corners
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    # Transform corners
    transformed_corners = apply_affine_to_points(corners, matrix)

    # Ensure transformed_corners is 2D
    transformed_corners = transformed_corners.reshape(-1, 2)

    # Find box that includes both original and transformed corners
    all_corners = np.vstack((corners, transformed_corners))
    min_x, min_y = all_corners.min(axis=0)
    max_x, max_y = all_corners.max(axis=0)

    # Compute the inverse transform
    inverse_matrix = np.linalg.inv(matrix)

    # Apply inverse transform to all corners of the bounding box
    bbox_corners = np.array(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
    )
    inverse_corners = apply_affine_to_points(bbox_corners, inverse_matrix).reshape(
        -1,
        2,
    )

    min_x, min_y = inverse_corners.min(axis=0)
    max_x, max_y = inverse_corners.max(axis=0)

    pad_left = max(0, math.ceil(0 - min_x))
    pad_right = max(0, math.ceil(max_x - width))
    pad_top = max(0, math.ceil(0 - min_y))
    pad_bottom = max(0, math.ceil(max_y - height))

    return pad_left, pad_right, pad_top, pad_bottom


@handle_empty_array("bboxes")
def bboxes_affine_largest_box(bboxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an affine transformation to bounding boxes and return the largest enclosing boxes.

    This function transforms each corner of every bounding box using the given affine transformation
    matrix, then computes the new bounding boxes that fully enclose the transformed corners.

    Args:
        bboxes (np.ndarray): An array of bounding boxes with shape (N, 4+) where N is the number of
                             bounding boxes. Each row should contain [x_min, y_min, x_max, y_max]
                             followed by any additional attributes (e.g., class labels).
        matrix (np.ndarray): The 3x3 affine transformation matrix to apply.

    Returns:
        np.ndarray: An array of transformed bounding boxes with the same shape as the input.
                    Each row contains [new_x_min, new_y_min, new_x_max, new_y_max] followed by
                    any additional attributes from the input bounding boxes.

    Note:
        - This function assumes that the input bounding boxes are in the format [x_min, y_min, x_max, y_max].
        - The resulting bounding boxes are the smallest axis-aligned boxes that completely
          enclose the transformed original boxes. They may be larger than the minimal possible
          bounding box if the original box becomes rotated.
        - Any additional attributes beyond the first 4 coordinates are preserved unchanged.
        - This method is called "largest box" because it returns the largest axis-aligned box
          that encloses all corners of the transformed bounding box.

    Example:
        >>> bboxes = np.array([[10, 10, 20, 20, 1], [30, 30, 40, 40, 2]])  # Two boxes with class labels
        >>> matrix = np.array([[2, 0, 5], [0, 2, 5], [0, 0, 1]])  # Scale by 2 and translate by (5, 5)
        >>> transformed_bboxes = bboxes_affine_largest_box(bboxes, matrix)
        >>> print(transformed_bboxes)
        [[ 25.  25.  45.  45.   1.]
         [ 65.  65.  85.  85.   2.]]

    """
    # Extract corners of all bboxes
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    corners = (
        np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]).transpose(2, 0, 1).reshape(-1, 2)
    )

    # Transform all corners at once
    transformed_corners = apply_affine_to_points(corners, matrix).reshape(-1, 4, 2)

    # Compute new bounding boxes
    new_x_min = np.min(transformed_corners[:, :, 0], axis=1)
    new_x_max = np.max(transformed_corners[:, :, 0], axis=1)
    new_y_min = np.min(transformed_corners[:, :, 1], axis=1)
    new_y_max = np.max(transformed_corners[:, :, 1], axis=1)

    return np.column_stack([new_x_min, new_y_min, new_x_max, new_y_max, bboxes[:, 4:]])


@handle_empty_array("bboxes")
def bboxes_affine_ellipse(bboxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an affine transformation to bounding boxes using an ellipse approximation method.

    This function transforms bounding boxes by approximating each box with an ellipse,
    transforming points along the ellipse's circumference, and then computing the
    new bounding box that encloses the transformed ellipse.

    Args:
        bboxes (np.ndarray): An array of bounding boxes with shape (N, 4+) where N is the number of
                             bounding boxes. Each row should contain [x_min, y_min, x_max, y_max]
                             followed by any additional attributes (e.g., class labels).
        matrix (np.ndarray): The 3x3 affine transformation matrix to apply.

    Returns:
        np.ndarray: An array of transformed bounding boxes with the same shape as the input.
                    Each row contains [new_x_min, new_y_min, new_x_max, new_y_max] followed by
                    any additional attributes from the input bounding boxes.

    Note:
        - This function assumes that the input bounding boxes are in the format [x_min, y_min, x_max, y_max].
        - The ellipse approximation method can provide a tighter bounding box compared to the
          largest box method, especially for rotations.
        - 360 points are used to approximate each ellipse, which provides a good balance between
          accuracy and computational efficiency.
        - Any additional attributes beyond the first 4 coordinates are preserved unchanged.
        - This method may be more suitable for objects that are roughly elliptical in shape.

    """
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    bbox_width = (x_max - x_min) / 2
    bbox_height = (y_max - y_min) / 2
    center_x = x_min + bbox_width
    center_y = y_min + bbox_height

    angles = np.arange(0, 360, dtype=np.float32)
    cos_angles = np.cos(np.radians(angles))
    sin_angles = np.sin(np.radians(angles))

    # Generate points for all ellipses at once
    x = bbox_width[:, np.newaxis] * sin_angles + center_x[:, np.newaxis]
    y = bbox_height[:, np.newaxis] * cos_angles + center_y[:, np.newaxis]
    points = np.stack([x, y], axis=-1).reshape(-1, 2)

    # Transform all points at once using the helper function
    transformed_points = apply_affine_to_points(points, matrix)

    transformed_points = transformed_points.reshape(len(bboxes), -1, 2)

    # Compute new bounding boxes
    new_x_min = np.min(transformed_points[:, :, 0], axis=1)
    new_x_max = np.max(transformed_points[:, :, 0], axis=1)
    new_y_min = np.min(transformed_points[:, :, 1], axis=1)
    new_y_max = np.max(transformed_points[:, :, 1], axis=1)

    return np.column_stack([new_x_min, new_y_min, new_x_max, new_y_max, bboxes[:, 4:]])


@handle_empty_array("bboxes")
def bboxes_affine(
    bboxes: np.ndarray,
    matrix: np.ndarray,
    rotate_method: Literal["largest_box", "ellipse"],
    image_shape: tuple[int, int],
    border_mode: int,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Apply an affine transformation to bounding boxes.

    For reflection border modes (cv2.BORDER_REFLECT_101, cv2.BORDER_REFLECT), this function:
    1. Calculates necessary padding to avoid information loss
    2. Applies padding to the bounding boxes
    3. Adjusts the transformation matrix to account for padding
    4. Applies the affine transformation
    5. Validates the transformed bounding boxes

    For other border modes, it directly applies the affine transformation without padding.

    Args:
        bboxes (np.ndarray): Input bounding boxes
        matrix (np.ndarray): Affine transformation matrix
        rotate_method (str): Method for rotating bounding boxes ('largest_box' or 'ellipse')
        image_shape (Sequence[int]): Shape of the input image
        border_mode (int): OpenCV border mode
        output_shape (Sequence[int]): Shape of the output image

    Returns:
        np.ndarray: Transformed and normalized bounding boxes

    """
    if is_identity_matrix(matrix):
        return bboxes

    bboxes = denormalize_bboxes(bboxes, image_shape)

    if border_mode in REFLECT_BORDER_MODES:
        # Step 1: Compute affine transform padding
        pad_left, pad_right, pad_top, pad_bottom = calculate_affine_transform_padding(
            matrix,
            image_shape,
        )
        grid_dimensions = get_pad_grid_dimensions(
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            image_shape,
        )
        bboxes = generate_reflected_bboxes(
            bboxes,
            grid_dimensions,
            image_shape,
            center_in_origin=True,
        )

    # Apply affine transform
    if rotate_method == "largest_box":
        transformed_bboxes = bboxes_affine_largest_box(bboxes, matrix)
    elif rotate_method == "ellipse":
        transformed_bboxes = bboxes_affine_ellipse(bboxes, matrix)
    else:
        raise ValueError(f"Method {rotate_method} is not a valid rotation method.")

    # Validate and normalize bboxes
    validated_bboxes = validate_bboxes(transformed_bboxes, output_shape)

    return normalize_bboxes(validated_bboxes, output_shape)


def to_distance_maps(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    inverted: bool = False,
) -> np.ndarray:
    """Generate a ``(H,W,N)`` array of distance maps for ``N`` keypoints.
    The ``n``-th distance map contains at every location ``(y, x)`` the
    euclidean distance to the ``n``-th keypoint.
    This function can be used as a helper when augmenting keypoints with a
    method that only supports the augmentation of images.

    Args:
        keypoints (np.ndarray): A numpy array of shape (N, 2+) where N is the number of keypoints.
                   Each row represents a keypoint's (x, y) coordinates.
        image_shape (tuple[int, int]): Shape of the image (height, width)
        inverted (bool): If ``True``, inverted distance maps are returned where each
            distance value d is replaced by ``d/(d+1)``, i.e. the distance
            maps have values in the range ``(0.0, 1.0]`` with ``1.0`` denoting
            exactly the position of the respective keypoint.

    Returns:
        np.ndarray: A float32 array of shape (H, W, N) containing ``N`` distance maps for ``N``
            keypoints. Each location ``(y, x, n)`` in the array denotes the
            euclidean distance at ``(y, x)`` to the ``n``-th keypoint.
            If `inverted` is ``True``, the distance ``d`` is replaced
            by ``d/(d+1)``. The height and width of the array match the
            height and width in ``image_shape``.

    """
    height, width = image_shape[:2]
    if len(keypoints) == 0:
        return np.zeros((height, width, 0), dtype=np.float32)

    # Create coordinate grids
    yy, xx = np.mgrid[:height, :width]

    # Convert keypoints to numpy array
    keypoints_array = np.array(keypoints)

    # Compute distances for all keypoints at once
    distances = np.sqrt(
        (xx[..., np.newaxis] - keypoints_array[:, 0]) ** 2 + (yy[..., np.newaxis] - keypoints_array[:, 1]) ** 2,
    )

    if inverted:
        return (1 / (distances + 1)).astype(np.float32)
    return distances.astype(np.float32)


def validate_if_not_found_coords(
    if_not_found_coords: Sequence[int] | dict[str, Any] | None,
) -> tuple[bool, float, float]:
    """Validate and process `if_not_found_coords` parameter."""
    if if_not_found_coords is None:
        return True, -1, -1
    if isinstance(if_not_found_coords, (tuple, list)):
        if len(if_not_found_coords) != PAIR:
            msg = "Expected tuple/list 'if_not_found_coords' to contain exactly two entries."
            raise ValueError(msg)
        return False, if_not_found_coords[0], if_not_found_coords[1]
    if isinstance(if_not_found_coords, dict):
        return False, if_not_found_coords["x"], if_not_found_coords["y"]

    msg = "Expected if_not_found_coords to be None, tuple, list, or dict."
    raise ValueError(msg)


def find_keypoint(
    position: tuple[int, int],
    distance_map: np.ndarray,
    threshold: float | None,
    inverted: bool,
) -> tuple[float, float] | None:
    """Determine if a valid keypoint can be found at the given position."""
    y, x = position
    value = distance_map[y, x]
    if not inverted and threshold is not None and value >= threshold:
        return None
    if inverted and threshold is not None and value <= threshold:
        return None
    return float(x), float(y)


def from_distance_maps(
    distance_maps: np.ndarray,
    inverted: bool,
    if_not_found_coords: Sequence[int] | dict[str, Any] | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """Convert distance maps back to keypoints coordinates.

    This function is the inverse of `to_distance_maps`. It takes distance maps generated for a set of keypoints
    and reconstructs the original keypoint coordinates. The function supports both regular and inverted distance maps,
    and can handle cases where keypoints are not found or fall outside a specified threshold.

    Args:
        distance_maps (np.ndarray): A 3D numpy array of shape (height, width, nb_keypoints) containing
            distance maps for each keypoint. Each channel represents the distance map for one keypoint.
        inverted (bool): If True, treats the distance maps as inverted (where higher values indicate
            closer proximity to keypoints). If False, treats them as regular distance maps (where lower
            values indicate closer proximity).
        if_not_found_coords (Sequence[int] | dict[str, Any] | None, optional): Coordinates to use for
            keypoints that are not found or fall outside the threshold. Can be:
            - None: Drop keypoints that are not found.
            - Sequence of two integers: Use these as (x, y) coordinates for not found keypoints.
            - Dict with 'x' and 'y' keys: Use these values for not found keypoints.
            Defaults to None.
        threshold (float | None, optional): A threshold value to determine valid keypoints. For inverted
            maps, values >= threshold are considered valid. For regular maps, values <= threshold are
            considered valid. If None, all keypoints are considered valid. Defaults to None.

    Returns:
        np.ndarray: A 2D numpy array of shape (nb_keypoints, 2) containing the (x, y) coordinates
        of the reconstructed keypoints. If `drop_if_not_found` is True (derived from if_not_found_coords),
        the output may have fewer rows than input keypoints.

    Raises:
        ValueError: If the input `distance_maps` is not a 3D array.

    Notes:
        - The function uses vectorized operations for improved performance, especially with large numbers of keypoints.
        - When `threshold` is None, all keypoints are considered valid, and `if_not_found_coords` is not used.
        - The function assumes that the input distance maps are properly normalized and scaled according to the
          original image dimensions.

    Example:
        >>> distance_maps = np.random.rand(100, 100, 3)  # 3 keypoints
        >>> inverted = True
        >>> if_not_found_coords = [0, 0]
        >>> threshold = 0.5
        >>> keypoints = from_distance_maps(distance_maps, inverted, if_not_found_coords, threshold)
        >>> print(keypoints.shape)
        (3, 2)

    """
    if distance_maps.ndim != NUM_MULTI_CHANNEL_DIMENSIONS:
        msg = f"Expected three-dimensional input, got {distance_maps.ndim} dimensions and shape {distance_maps.shape}."
        raise ValueError(msg)
    height, width, nb_keypoints = distance_maps.shape

    drop_if_not_found, if_not_found_x, if_not_found_y = validate_if_not_found_coords(
        if_not_found_coords,
    )

    # Find the indices of max/min values for all keypoints at once
    if inverted:
        hitidx_flat = np.argmax(
            distance_maps.reshape(height * width, nb_keypoints),
            axis=0,
        )
    else:
        hitidx_flat = np.argmin(
            distance_maps.reshape(height * width, nb_keypoints),
            axis=0,
        )

    # Convert flat indices to 2D coordinates
    hitidx_y, hitidx_x = np.unravel_index(hitidx_flat, (height, width))

    # Create keypoints array
    keypoints = np.column_stack((hitidx_x, hitidx_y)).astype(float)

    if threshold is not None:
        # Check threshold condition
        if inverted:
            valid_mask = distance_maps[hitidx_y, hitidx_x, np.arange(nb_keypoints)] >= threshold
        else:
            valid_mask = distance_maps[hitidx_y, hitidx_x, np.arange(nb_keypoints)] <= threshold

        if not drop_if_not_found:
            # Replace invalid keypoints with if_not_found_coords
            keypoints[~valid_mask] = [if_not_found_x, if_not_found_y]
        else:
            # Keep only valid keypoints
            return keypoints[valid_mask]

    return keypoints


def d4(img: np.ndarray, group_member: Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]) -> np.ndarray:
    """Applies a `D_4` symmetry group transformation to an image array.

    This function manipulates an image using transformations such as rotations and flips,
    corresponding to the `D_4` dihedral group symmetry operations.
    Each transformation is identified by a unique group member code.

    Args:
        img (np.ndarray): The input image array to transform.
        group_member (Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]): A string identifier indicating
            the specific transformation to apply. Valid codes include:
            - 'e': Identity (no transformation).
            - 'r90': Rotate 90 degrees counterclockwise.
            - 'r180': Rotate 180 degrees.
            - 'r270': Rotate 270 degrees counterclockwise.
            - 'v': Vertical flip.
            - 'hvt': Transpose over second diagonal
            - 'h': Horizontal flip.
            - 't': Transpose (reflect over the main diagonal).

    Returns:
        np.ndarray: The transformed image array.

    Raises:
        ValueError: If an invalid group member is specified.

    """
    transformations = {
        "e": lambda x: x,  # Identity transformation
        "r90": lambda x: rot90(x, 1),  # Rotate 90 degrees
        "r180": lambda x: rot90(x, 2),  # Rotate 180 degrees
        "r270": lambda x: rot90(x, 3),  # Rotate 270 degrees
        "v": vflip,  # Vertical flip
        "hvt": lambda x: transpose(rot90(x, 2)),  # Reflect over anti-diagonal
        "h": hflip,  # Horizontal flip
        "t": transpose,  # Transpose (reflect over main diagonal)
    }

    # Execute the appropriate transformation
    if group_member in transformations:
        return transformations[group_member](img)

    raise ValueError(f"Invalid group member: {group_member}")


def transpose(img: np.ndarray) -> np.ndarray:
    """Transposes the first two dimensions of an array of any dimensionality.
    Retains the order of any additional dimensions.

    Args:
        img (np.ndarray): Input array.

    Returns:
        np.ndarray: Transposed array.

    """
    # Generate the new axes order
    new_axes = list(range(img.ndim))
    new_axes[0], new_axes[1] = 1, 0  # Swap the first two dimensions

    # Transpose the array using the new axes order
    return img.transpose(new_axes)


def rot90(img: np.ndarray, factor: Literal[0, 1, 2, 3]) -> np.ndarray:
    """Rotate an image 90 degrees counterclockwise.

    Args:
        img (np.ndarray): The input image to rotate.
        factor (Literal[0, 1, 2, 3]): The number of 90-degree rotations to apply.

    Returns:
        np.ndarray: The rotated image.

    """
    return np.rot90(img, factor)


@handle_empty_array("bboxes")
def bboxes_vflip(bboxes: np.ndarray) -> np.ndarray:
    """Flip bounding boxes vertically.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)

    Returns:
        np.ndarray: Vertically flipped bounding boxes

    """
    flipped_bboxes = bboxes.copy()
    flipped_bboxes[:, 1] = 1 - bboxes[:, 3]  # new y_min = 1 - y_max
    flipped_bboxes[:, 3] = 1 - bboxes[:, 1]  # new y_max = 1 - y_min

    return flipped_bboxes


@handle_empty_array("bboxes")
def bboxes_hflip(bboxes: np.ndarray) -> np.ndarray:
    """Flip bounding boxes horizontally.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)

    Returns:
        np.ndarray: Horizontally flipped bounding boxes

    """
    flipped_bboxes = bboxes.copy()
    flipped_bboxes[:, 0] = 1 - bboxes[:, 2]  # new x_min = 1 - x_max
    flipped_bboxes[:, 2] = 1 - bboxes[:, 0]  # new x_max = 1 - x_min

    return flipped_bboxes


@handle_empty_array("bboxes")
def bboxes_transpose(bboxes: np.ndarray) -> np.ndarray:
    """Transpose bounding boxes along the main diagonal.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)

    Returns:
        np.ndarray: Transposed bounding boxes

    """
    transposed_bboxes = bboxes.copy()
    transposed_bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]

    return transposed_bboxes


@handle_empty_array("keypoints")
@angle_2pi_range
def keypoints_vflip(keypoints: np.ndarray, rows: int) -> np.ndarray:
    """Flip keypoints vertically.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+)
        rows (int): Number of rows in the image

    Returns:
        np.ndarray: Vertically flipped keypoints

    """
    flipped_keypoints = keypoints.copy().astype(np.float32)

    # Flip y-coordinates
    flipped_keypoints[:, 1] = (rows - 1) - keypoints[:, 1]

    # Negate angles
    flipped_keypoints[:, 3] = -keypoints[:, 3]

    return flipped_keypoints


@handle_empty_array("keypoints")
@angle_2pi_range
def keypoints_hflip(keypoints: np.ndarray, cols: int) -> np.ndarray:
    """Flip keypoints horizontally.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+)
        cols (int): Number of columns in the image

    Returns:
        np.ndarray: Horizontally flipped keypoints

    """
    flipped_keypoints = keypoints.copy().astype(np.float32)

    # Flip x-coordinates
    flipped_keypoints[:, 0] = (cols - 1) - keypoints[:, 0]

    # Adjust angles
    flipped_keypoints[:, 3] = np.pi - keypoints[:, 3]

    return flipped_keypoints


@handle_empty_array("keypoints")
@angle_2pi_range
def keypoints_transpose(keypoints: np.ndarray) -> np.ndarray:
    """Transpose keypoints along the main diagonal.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+)

    Returns:
        np.ndarray: Transposed keypoints

    """
    transposed_keypoints = keypoints.copy()

    # Swap x and y coordinates
    transposed_keypoints[:, [0, 1]] = keypoints[:, [1, 0]]

    # Adjust angles to reflect the coordinate swap
    angles = keypoints[:, 3]
    transposed_keypoints[:, 3] = np.where(
        angles <= np.pi,
        np.pi / 2 - angles,
        3 * np.pi / 2 - angles,
    )

    return transposed_keypoints


@preserve_channel_dim
def pad(
    img: np.ndarray,
    min_height: int,
    min_width: int,
    border_mode: int,
    value: tuple[float, ...] | float | None,
) -> np.ndarray:
    """Pad an image to ensure minimum dimensions.

    This function adds padding to an image if its dimensions are smaller than
    the specified minimum dimensions. Padding is added evenly on all sides.

    Args:
        img (np.ndarray): Input image to pad.
        min_height (int): Minimum height of the output image.
        min_width (int): Minimum width of the output image.
        border_mode (int): OpenCV border mode for padding.
        value (tuple[float, ...] | float | None): Value(s) to fill the border pixels.

    Returns:
        np.ndarray: Padded image with dimensions at least (min_height, min_width).

    """
    height, width = img.shape[:2]

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    img = pad_with_params(
        img,
        h_pad_top,
        h_pad_bottom,
        w_pad_left,
        w_pad_right,
        border_mode,
        value,
    )

    if img.shape[:2] != (max(min_height, height), max(min_width, width)):
        raise RuntimeError(
            f"Invalid result shape. Got: {img.shape[:2]}. Expected: {(max(min_height, height), max(min_width, width))}",
        )

    return img


def extend_value(value: tuple[float, ...] | float, num_channels: int) -> Sequence[float]:
    """Extend value to a sequence of floats.

    This function extends a value to a sequence of floats.
    It is used to pad an image with a given value.

    Args:
        value (tuple[float, ...] | float): The value to extend.
        num_channels (int): The number of channels in the image.

    Returns:
        Sequence[float]: The extended value.

    """
    return [value] * num_channels if isinstance(value, float) else value


def copy_make_border_with_value_extension(
    img: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_mode: int,
    value: tuple[float, ...] | float,
) -> np.ndarray:
    """Copy and make border with value extension.

    This function copies and makes border with value extension.
    It is used to pad an image with a given value.

    Args:
        img (np.ndarray): The image to pad.
        top (int): The amount to pad the top of the image.
        bottom (int): The amount to pad the bottom of the image.
        left (int): The amount to pad the left of the image.
        right (int): The amount to pad the right of the image.
        border_mode (int): The border mode to use.
        value (tuple[float, ...] | float): The value to pad the image with.

    Returns:
        np.ndarray: The padded image.

    """
    # For 0-channel images, return empty array of correct padded size
    if img.size == 0:
        height, width = img.shape[:2]
        return np.zeros(
            (height + top + bottom, width + left + right, 0),
            dtype=img.dtype,
        )

    num_channels = get_num_channels(img)
    extended_value = extend_value(value, num_channels)

    return cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        borderType=border_mode,
        value=extended_value,
    )


@preserve_channel_dim
def pad_with_params(
    img: np.ndarray,
    h_pad_top: int,
    h_pad_bottom: int,
    w_pad_left: int,
    w_pad_right: int,
    border_mode: int,
    value: tuple[float, ...] | float | None,
) -> np.ndarray:
    """Pad an image with explicitly defined padding on each side.

    This function adds specified amounts of padding to each side of the image.

    Args:
        img (np.ndarray): Input image to pad.
        h_pad_top (int): Number of pixels to add at the top.
        h_pad_bottom (int): Number of pixels to add at the bottom.
        w_pad_left (int): Number of pixels to add on the left.
        w_pad_right (int): Number of pixels to add on the right.
        border_mode (int): OpenCV border mode for padding.
        value (tuple[float, ...] | float | None): Value(s) to fill the border pixels.

    Returns:
        np.ndarray: Padded image.

    """
    pad_fn = maybe_process_in_chunks(
        copy_make_border_with_value_extension,
        top=h_pad_top,
        bottom=h_pad_bottom,
        left=w_pad_left,
        right=w_pad_right,
        border_mode=border_mode,
        value=value,
    )

    return pad_fn(img)


def pad_images_with_params(
    images: np.ndarray,
    h_pad_top: int,
    h_pad_bottom: int,
    w_pad_left: int,
    w_pad_right: int,
    border_mode: int,
    value: tuple[float, ...] | float | None,
) -> np.ndarray:
    """Pad a batch of images with explicitly defined padding on each side.

    This function adds specified amounts of padding to each side of the image for each
    image in the batch.

    Args:
        images (np.ndarray): Input batch of images to pad.
        h_pad_top (int): Number of pixels to add at the top.
        h_pad_bottom (int): Number of pixels to add at the bottom.
        w_pad_left (int): Number of pixels to add on the left.
        w_pad_right (int): Number of pixels to add on the right.
        border_mode (int): OpenCV border mode for padding.
        value (tuple[float, ...] | float | None): Value(s) to fill the border pixels.

    Returns:
        np.ndarray: Padded batch of images.

    """
    no_channel_dim = images.ndim == 3
    if no_channel_dim:
        images = images[..., np.newaxis]

    cv2np_border_modes = {
        cv2.BORDER_CONSTANT: "constant",
        cv2.BORDER_REPLICATE: "edge",
        cv2.BORDER_REFLECT: "symmetric",
        cv2.BORDER_WRAP: "wrap",
        cv2.BORDER_REFLECT_101: "reflect",
        cv2.BORDER_REFLECT101: "reflect",
        cv2.BORDER_DEFAULT: "reflect",  # same as cv2.BORDER_REFLECT_101
    }
    mode = cv2np_border_modes[border_mode]

    pad_width = ((0, 0), (h_pad_top, h_pad_bottom), (w_pad_left, w_pad_right), (0, 0))
    if mode == "constant":
        constant_values = np.array(((0, 0), (value, value), (value, value), (0, 0)), dtype=object)
        kwargs = {"constant_values": constant_values}
    else:
        kwargs = {}

    images = np.pad(images, pad_width=pad_width, mode=mode, **kwargs)
    if no_channel_dim:
        images = images[..., 0]

    return images


@preserve_channel_dim
def remap(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    border_mode: int,
    value: tuple[float, ...] | float | None = None,
) -> np.ndarray:
    """Remap an image according to given coordinate maps.

    This function applies a generic geometrical transformation using
    mapping functions that specify the position of each pixel in the output image.

    Args:
        img (np.ndarray): Input image to transform.
        map_x (np.ndarray): Map of x-coordinates with same height and width as the input image.
        map_y (np.ndarray): Map of y-coordinates with same height and width as the input image.
        interpolation (int): Interpolation method for resampling.
        border_mode (int): OpenCV border mode for handling pixels outside the image boundaries.
        value (tuple[float, ...] | float | None, optional): Border value(s) if border_mode is BORDER_CONSTANT.

    Returns:
        np.ndarray: Remapped image with the same shape as the input image.

    """
    # Combine map_x and map_y into a single map array of type CV_32FC2
    map_xy = np.stack([map_x, map_y], axis=-1).astype(np.float32)

    # Create remap function with chunks processing
    remap_func = maybe_process_in_chunks(
        cv2.remap,
        map1=map_xy,
        map2=None,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )

    # Apply the remapping
    return remap_func(img)


def remap_keypoints_via_mask(
    keypoints: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Remap keypoints using mask and cv2.remap method."""
    height, width = image_shape[:2]

    # Handle empty keypoints array
    if len(keypoints) == 0:
        return np.zeros((0, 2 if keypoints.size == 0 else keypoints.shape[1]))

    # Create mask where each keypoint has unique index
    kp_mask = np.zeros((height, width), dtype=np.int16)
    for idx, kp in enumerate(keypoints, start=1):
        x, y = round(kp[0]), round(kp[1])
        if 0 <= x < width and 0 <= y < height:
            # Note: cv2.circle takes (x,y) coordinates
            cv2.circle(kp_mask, (x, y), 1, idx, -1)

    # Remap the mask
    transformed_kp_mask = cv2.remap(
        kp_mask,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_NEAREST,
    )

    # Extract transformed keypoints
    new_points = []
    for idx, kp in enumerate(keypoints, start=1):
        # Find points with this index
        points = np.where(transformed_kp_mask == idx)
        if len(points[0]) > 0:
            # Convert back to (x,y) coordinates
            new_points.append(np.concatenate([[points[1][0], points[0][0]], kp[2:]]))

    return np.array(new_points) if new_points else np.zeros((0, keypoints.shape[1]))


@handle_empty_array("keypoints")
def remap_keypoints(
    keypoints: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Transform keypoints using coordinate mapping functions.

    This function applies the inverse of the mapping defined by map_x and map_y
    to keypoint coordinates. The inverse mapping is necessary because the mapping
    functions define how pixels move from the source to the destination image,
    while keypoints need to be transformed from the destination back to the source.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 2+), where
            the first two columns are x and y coordinates.
        map_x (np.ndarray): Map of x-coordinates with shape equal to image_shape.
        map_y (np.ndarray): Map of y-coordinates with shape equal to image_shape.
        image_shape (tuple[int, int]): Shape (height, width) of the original image.

    Returns:
        np.ndarray: Transformed keypoints with the same shape as the input keypoints.
            Returns an empty array if input keypoints is empty.

    """
    height, width = image_shape[:2]

    # Extract x and y coordinates
    x, y = keypoints[:, 0], keypoints[:, 1]

    # Clip coordinates to image boundaries
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    # Convert to integer indices
    x_idx, y_idx = x.astype(int), y.astype(int)
    inv_map_x, inv_map_y = generate_inverse_distortion_map(map_x, map_y, image_shape[:2])
    # Apply the inverse mapping
    new_x = inv_map_x[y_idx, x_idx]
    new_y = inv_map_y[y_idx, x_idx]

    # Clip the new coordinates to ensure they're within the image bounds
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)

    # Create the transformed keypoints array
    return np.column_stack([new_x, new_y, keypoints[:, 2:]])


def generate_inverse_distortion_map(
    map_x: np.ndarray,
    map_y: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate inverse mapping for strong distortions."""
    h, w = shape

    # Initialize inverse maps
    inv_map_x = np.zeros((h, w), dtype=np.float32)
    inv_map_y = np.zeros((h, w), dtype=np.float32)

    # For each source point, record where it maps to
    for y in range(h):
        for x in range(w):
            # Get destination point
            dst_x = map_x[y, x]
            dst_y = map_y[y, x]

            # If destination is within bounds
            if 0 <= dst_x < w and 0 <= dst_y < h:
                # Get neighborhood coordinates
                dst_x_floor = int(np.floor(dst_x))
                dst_x_ceil = min(dst_x_floor + 1, w - 1)
                dst_y_floor = int(np.floor(dst_y))
                dst_y_ceil = min(dst_y_floor + 1, h - 1)

                # Fill neighborhood
                for ny in range(dst_y_floor, dst_y_ceil + 1):
                    for nx in range(dst_x_floor, dst_x_ceil + 1):
                        # Only update if empty or closer to pixel center
                        if inv_map_x[ny, nx] == 0 or (
                            abs(nx - dst_x) + abs(ny - dst_y)
                            < abs(nx - inv_map_x[ny, nx]) + abs(ny - inv_map_y[ny, nx])
                        ):
                            inv_map_x[ny, nx] = x
                            inv_map_y[ny, nx] = y

    return inv_map_x, inv_map_y


@handle_empty_array("bboxes")
def remap_bboxes(
    bboxes: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Remap bounding boxes using displacement maps."""
    # Convert bboxes to mask
    bbox_masks = bboxes_to_mask(bboxes, image_shape)

    # Ensure maps are float32
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    transformed_masks = remap(bbox_masks, map_x, map_y, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, value=0)

    # Convert masks back to bboxes
    return mask_to_bboxes(transformed_masks, bboxes)


def generate_displacement_fields(
    image_shape: tuple[int, int],
    alpha: float,
    sigma: float,
    same_dxdy: bool,
    kernel_size: tuple[int, int],
    random_generator: np.random.Generator,
    noise_distribution: Literal["gaussian", "uniform"],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate displacement fields for elastic transform.

    This function generates displacement fields for elastic transform based on the provided parameters.
    It generates noise either from a Gaussian or uniform distribution and normalizes it to the range [-1, 1].

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        alpha (float): The alpha parameter for the elastic transform.
        sigma (float): The sigma parameter for the elastic transform.
        same_dxdy (bool): Whether to use the same displacement field for both x and y directions.
        kernel_size (tuple[int, int]): The size of the kernel for the elastic transform.
        random_generator (np.random.Generator): The random number generator to use.
        noise_distribution (Literal["gaussian", "uniform"]): The distribution of the noise.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - fields: The displacement fields for the elastic transform.
            - output_shape: The output shape of the elastic warp.

    """
    # Pre-allocate memory and generate noise in one step
    if noise_distribution == "gaussian":
        # Generate and normalize in one step, directly as float32
        fields = random_generator.standard_normal(
            (1 if same_dxdy else 2, *image_shape[:2]),
            dtype=np.float32,
        )
        # Normalize inplace
        max_abs = np.abs(fields, out=np.empty_like(fields)).max()
        if max_abs > 1e-6:
            fields /= max_abs
    else:  # uniform is already normalized to [-1, 1]
        fields = random_generator.uniform(
            -1,
            1,
            size=(1 if same_dxdy else 2, *image_shape[:2]),
        ).astype(np.float32)

    # # Apply Gaussian blur if needed using fast OpenCV operations
    # When kernel_size is (0,0) cv2.GaussianBlur uses automatic kernel size. Kernel == (0,0) is NOT a noop.
    # Reshape to 2D array (combining first dimension with height)
    shape = fields.shape
    fields = fields.reshape(-1, shape[-1])

    # Apply blur to all fields at once
    cv2.GaussianBlur(
        fields,
        kernel_size,
        sigma,
        dst=fields,
        borderType=cv2.BORDER_REPLICATE,
    )

    # Restore original shape
    fields = fields.reshape(shape)

    # Scale by alpha inplace
    fields *= alpha

    # Return views of the array to avoid copies
    return (fields[0], fields[0]) if same_dxdy else (fields[0], fields[1])


@handle_empty_array("bboxes")
def pad_bboxes(
    bboxes: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    border_mode: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Pad bounding boxes by a given amount.

    This function pads bounding boxes by a given amount.
    It handles both reflection and padding.

    Args:
        bboxes (np.ndarray): The bounding boxes to pad.
        pad_top (int): The amount to pad the top of the bounding boxes.
        pad_bottom (int): The amount to pad the bottom of the bounding boxes.
        pad_left (int): The amount to pad the left of the bounding boxes.
        pad_right (int): The amount to pad the right of the bounding boxes.
        border_mode (int): The border mode to use.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: The padded bounding boxes.

    """
    if border_mode not in REFLECT_BORDER_MODES:
        shift_vector = np.array([pad_left, pad_top, pad_left, pad_top])
        return shift_bboxes(bboxes, shift_vector)

    grid_dimensions = get_pad_grid_dimensions(
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        image_shape,
    )

    bboxes = generate_reflected_bboxes(bboxes, grid_dimensions, image_shape)

    # Calculate the number of grid cells added on each side
    original_row, original_col = grid_dimensions["original_position"]

    image_height, image_width = image_shape[:2]

    # Subtract the offset based on the number of added grid cells
    left_shift = original_col * image_width - pad_left
    top_shift = original_row * image_height - pad_top

    shift_vector = np.array([-left_shift, -top_shift, -left_shift, -top_shift])

    bboxes = shift_bboxes(bboxes, shift_vector)

    new_height = pad_top + pad_bottom + image_height
    new_width = pad_left + pad_right + image_width

    return validate_bboxes(bboxes, (new_height, new_width))


def validate_bboxes(bboxes: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    """Validate bounding boxes and remove invalid ones.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (n, 4) where each row is [x_min, y_min, x_max, y_max].
        image_shape (tuple[int, int]): Shape of the image as (height, width).

    Returns:
        np.ndarray: Array of valid bounding boxes, potentially with fewer boxes than the input.

    Example:
        >>> bboxes = np.array([[10, 20, 30, 40], [-10, -10, 5, 5], [100, 100, 120, 120]])
        >>> valid_bboxes = validate_bboxes(bboxes, (100, 100))
        >>> print(valid_bboxes)
        [[10 20 30 40]]

    """
    rows, cols = image_shape[:2]

    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    valid_indices = (x_max > 0) & (y_max > 0) & (x_min < cols) & (y_min < rows)

    return bboxes[valid_indices]


def shift_bboxes(bboxes: np.ndarray, shift_vector: np.ndarray) -> np.ndarray:
    """Shift bounding boxes by a given vector.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (n, m) where n is the number of bboxes
                             and m >= 4. The first 4 columns are [x_min, y_min, x_max, y_max].
        shift_vector (np.ndarray): Vector to shift the bounding boxes by, with shape (4,) for
                                   [shift_x, shift_y, shift_x, shift_y].

    Returns:
        np.ndarray: Shifted bounding boxes with the same shape as input.

    """
    # Create a copy of the input array to avoid modifying it in-place
    shifted_bboxes = bboxes.copy()

    # Add the shift vector to the first 4 columns
    shifted_bboxes[:, :4] += shift_vector

    return shifted_bboxes


def get_pad_grid_dimensions(
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    image_shape: tuple[int, int],
) -> dict[str, tuple[int, int]]:
    """Calculate the dimensions of the grid needed for reflection padding and the position of the original image.

    Args:
        pad_top (int): Number of pixels to pad above the image.
        pad_bottom (int): Number of pixels to pad below the image.
        pad_left (int): Number of pixels to pad to the left of the image.
        pad_right (int): Number of pixels to pad to the right of the image.
        image_shape (tuple[int, int]): Shape of the original image as (height, width).

    Returns:
        dict[str, tuple[int, int]]: A dictionary containing:
            - 'grid_shape': A tuple (grid_rows, grid_cols) where:
                - grid_rows (int): Number of times the image needs to be repeated vertically.
                - grid_cols (int): Number of times the image needs to be repeated horizontally.
            - 'original_position': A tuple (original_row, original_col) where:
                - original_row (int): Row index of the original image in the grid.
                - original_col (int): Column index of the original image in the grid.

    """
    rows, cols = image_shape[:2]

    grid_rows = 1 + math.ceil(pad_top / rows) + math.ceil(pad_bottom / rows)
    grid_cols = 1 + math.ceil(pad_left / cols) + math.ceil(pad_right / cols)
    original_row = math.ceil(pad_top / rows)
    original_col = math.ceil(pad_left / cols)

    return {
        "grid_shape": (grid_rows, grid_cols),
        "original_position": (original_row, original_col),
    }


def generate_reflected_bboxes(
    bboxes: np.ndarray,
    grid_dims: dict[str, tuple[int, int]],
    image_shape: tuple[int, int],
    center_in_origin: bool = False,
) -> np.ndarray:
    """Generate reflected bounding boxes for the entire reflection grid.

    Args:
        bboxes (np.ndarray): Original bounding boxes.
        grid_dims (dict[str, tuple[int, int]]): Grid dimensions and original position.
        image_shape (tuple[int, int]): Shape of the original image as (height, width).
        center_in_origin (bool): If True, center the grid at the origin. Default is False.

    Returns:
        np.ndarray: Array of reflected and shifted bounding boxes for the entire grid.

    """
    rows, cols = image_shape[:2]
    grid_rows, grid_cols = grid_dims["grid_shape"]
    original_row, original_col = grid_dims["original_position"]

    # Prepare flipped versions of bboxes
    bboxes_hflipped = flip_bboxes(bboxes, flip_horizontal=True, image_shape=image_shape)
    bboxes_vflipped = flip_bboxes(bboxes, flip_vertical=True, image_shape=image_shape)
    bboxes_hvflipped = flip_bboxes(
        bboxes,
        flip_horizontal=True,
        flip_vertical=True,
        image_shape=image_shape,
    )

    # Shift all versions to the original position
    shift_vector = np.array(
        [
            original_col * cols,
            original_row * rows,
            original_col * cols,
            original_row * rows,
        ],
    )
    bboxes = shift_bboxes(bboxes, shift_vector)
    bboxes_hflipped = shift_bboxes(bboxes_hflipped, shift_vector)
    bboxes_vflipped = shift_bboxes(bboxes_vflipped, shift_vector)
    bboxes_hvflipped = shift_bboxes(bboxes_hvflipped, shift_vector)

    new_bboxes = []

    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            # Determine which version of bboxes to use based on grid position
            if (grid_row - original_row) % 2 == 0 and (grid_col - original_col) % 2 == 0:
                current_bboxes = bboxes
            elif (grid_row - original_row) % 2 == 0:
                current_bboxes = bboxes_hflipped
            elif (grid_col - original_col) % 2 == 0:
                current_bboxes = bboxes_vflipped
            else:
                current_bboxes = bboxes_hvflipped

            # Shift to the current grid cell
            cell_shift = np.array(
                [
                    (grid_col - original_col) * cols,
                    (grid_row - original_row) * rows,
                    (grid_col - original_col) * cols,
                    (grid_row - original_row) * rows,
                ],
            )
            shifted_bboxes = shift_bboxes(current_bboxes, cell_shift)

            new_bboxes.append(shifted_bboxes)

    result = np.vstack(new_bboxes)

    return shift_bboxes(result, -shift_vector) if center_in_origin else result


@handle_empty_array("bboxes")
def flip_bboxes(
    bboxes: np.ndarray,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    image_shape: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Flip bounding boxes horizontally and/or vertically.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (n, m) where each row is
            [x_min, y_min, x_max, y_max, ...].
        flip_horizontal (bool): Whether to flip horizontally.
        flip_vertical (bool): Whether to flip vertically.
        image_shape (tuple[int, int]): Shape of the image as (height, width).

    Returns:
        np.ndarray: Flipped bounding boxes.

    """
    rows, cols = image_shape[:2]
    flipped_bboxes = bboxes.copy()
    if flip_horizontal:
        flipped_bboxes[:, [0, 2]] = cols - flipped_bboxes[:, [2, 0]]
    if flip_vertical:
        flipped_bboxes[:, [1, 3]] = rows - flipped_bboxes[:, [3, 1]]
    return flipped_bboxes


@preserve_channel_dim
def distort_image(
    image: np.ndarray,
    generated_mesh: np.ndarray,
    interpolation: int,
) -> np.ndarray:
    """Apply perspective distortion to an image based on a generated mesh.

    This function applies a perspective transformation to each cell of the image defined by the
    generated mesh. The distortion is applied using OpenCV's perspective transformation and
    blending techniques.

    Args:
        image (np.ndarray): The input image to be distorted. Can be a 2D grayscale image or a
                            3D color image.
        generated_mesh (np.ndarray): A 2D array where each row represents a quadrilateral cell
                                    as [x1, y1, x2, y2, dst_x1, dst_y1, dst_x2, dst_y2, dst_x3, dst_y3, dst_x4, dst_y4].
                                    The first four values define the source rectangle, and the last eight values
                                    define the destination quadrilateral.
        interpolation (int): Interpolation method to be used in the perspective transformation.
                             Should be one of the OpenCV interpolation flags (e.g., cv2.INTER_LINEAR).

    Returns:
        np.ndarray: The distorted image with the same shape and dtype as the input image.

    Note:
        - The function preserves the channel dimension of the input image.
        - Each cell of the generated mesh is transformed independently and then blended into the output image.
        - The distortion is applied using perspective transformation, which allows for more complex
          distortions compared to affine transformations.

    Example:
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> mesh = np.array([[0, 0, 50, 50, 5, 5, 45, 5, 45, 45, 5, 45]])
        >>> distorted = distort_image(image, mesh, cv2.INTER_LINEAR)
        >>> distorted.shape
        (100, 100, 3)

    """
    distorted_image = np.zeros_like(image)

    for mesh in generated_mesh:
        # Extract source rectangle and destination quadrilateral
        x1, y1, x2, y2 = mesh[:4]  # Source rectangle
        dst_quad = mesh[4:].reshape(4, 2)  # Destination quadrilateral

        # Convert source rectangle to quadrilateral
        src_quad = np.array(
            [
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2],  # Bottom-left
            ],
            dtype=np.float32,
        )

        # Calculate Perspective transformation matrix
        perspective_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)

        # Apply Perspective transformation
        warped = cv2.warpPerspective(
            image,
            perspective_mat,
            (image.shape[1], image.shape[0]),
            flags=interpolation,
        )

        # Create mask for the transformed region
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_quad), 255)

        # Copy only the warped quadrilateral area to the output image
        distorted_image = cv2.copyTo(warped, mask, distorted_image)

    return distorted_image


@handle_empty_array("bboxes")
def bbox_distort_image(
    bboxes: np.ndarray,
    generated_mesh: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Distort bounding boxes based on a generated mesh.

    This function applies a perspective transformation to each bounding box based on the provided generated mesh.
    It ensures that the bounding boxes are clipped to the image boundaries after transformation.

    Args:
        bboxes (np.ndarray): The bounding boxes to distort.
        generated_mesh (np.ndarray): The generated mesh to distort the bounding boxes with.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: The distorted bounding boxes.

    """
    bboxes = bboxes.copy()
    masks = masks_from_bboxes(bboxes, image_shape)

    transformed_masks = cv2.merge(
        [distort_image(mask, generated_mesh, cv2.INTER_NEAREST) for mask in masks],
    )

    if transformed_masks.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        transformed_masks = transformed_masks.transpose(2, 0, 1)

    # Normalize the returned bboxes
    bboxes[:, :4] = bboxes_from_masks(transformed_masks)

    return bboxes


@handle_empty_array("keypoints")
def distort_image_keypoints(
    keypoints: np.ndarray,
    generated_mesh: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Distort keypoints based on a generated mesh.

    This function applies a perspective transformation to each keypoint based on the provided generated mesh.
    It ensures that the keypoints are clipped to the image boundaries after transformation.

    Args:
        keypoints (np.ndarray): The keypoints to distort.
        generated_mesh (np.ndarray): The generated mesh to distort the keypoints with.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: The distorted keypoints.

    """
    distorted_keypoints = keypoints.copy()
    height, width = image_shape[:2]

    for mesh in generated_mesh:
        x1, y1, x2, y2 = mesh[:4]  # Source rectangle
        dst_quad = mesh[4:].reshape(4, 2)  # Destination quadrilateral

        src_quad = np.array(
            [
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2],  # Bottom-left
            ],
            dtype=np.float32,
        )

        perspective_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)

        mask = (keypoints[:, 0] >= x1) & (keypoints[:, 0] < x2) & (keypoints[:, 1] >= y1) & (keypoints[:, 1] < y2)
        cell_keypoints = keypoints[mask]

        if len(cell_keypoints) > 0:
            # Convert to float32 before applying the transformation
            points_float32 = cell_keypoints[:, :2].astype(np.float32).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(
                points_float32,
                perspective_mat,
            ).reshape(-1, 2)

            # Update distorted keypoints
            distorted_keypoints[mask, :2] = transformed_points

    # Clip keypoints to image boundaries
    distorted_keypoints[:, 0] = np.clip(
        distorted_keypoints[:, 0],
        0,
        width - 1,
        out=distorted_keypoints[:, 0],
    )
    distorted_keypoints[:, 1] = np.clip(
        distorted_keypoints[:, 1],
        0,
        height - 1,
        out=distorted_keypoints[:, 1],
    )

    return distorted_keypoints


def generate_distorted_grid_polygons(
    dimensions: np.ndarray,
    magnitude: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate distorted grid polygons based on input dimensions and magnitude.

    This function creates a grid of polygons and applies random distortions to the internal vertices,
    while keeping the boundary vertices fixed. The distortion is applied consistently across shared
    vertices to avoid gaps or overlaps in the resulting grid.

    Args:
        dimensions (np.ndarray): A 3D array of shape (grid_height, grid_width, 4) where each element
                                 is [x_min, y_min, x_max, y_max] representing the dimensions of a grid cell.
        magnitude (int): Maximum pixel-wise displacement for distortion. The actual displacement
                         will be randomly chosen in the range [-magnitude, magnitude].
        random_generator (np.random.Generator): A random number generator.

    Returns:
        np.ndarray: A 2D array of shape (total_cells, 8) where each row represents a distorted polygon
                    as [x1, y1, x2, y1, x2, y2, x1, y2]. The total_cells is equal to grid_height * grid_width.

    Note:
        - Only internal grid points are distorted; boundary points remain fixed.
        - The function ensures consistent distortion across shared vertices of adjacent cells.
        - The distortion is applied to the following points of each internal cell:
            * Bottom-right of the cell above and to the left
            * Bottom-left of the cell above
            * Top-right of the cell to the left
            * Top-left of the current cell
        - Each square represents a cell, and the X marks indicate the coordinates where displacement occurs.
            +--+--+--+--+
            |  |  |  |  |
            +--X--X--X--+
            |  |  |  |  |
            +--X--X--X--+
            |  |  |  |  |
            +--X--X--X--+
            |  |  |  |  |
            +--+--+--+--+
        - For each X, the coordinates of the left, right, top, and bottom edges
          in the four adjacent cells are displaced.

    Example:
        >>> dimensions = np.array([[[0, 0, 50, 50], [50, 0, 100, 50]],
        ...                        [[0, 50, 50, 100], [50, 50, 100, 100]]])
        >>> distorted = generate_distorted_grid_polygons(dimensions, magnitude=10)
        >>> distorted.shape
        (4, 8)

    """
    grid_height, grid_width = dimensions.shape[:2]
    total_cells = grid_height * grid_width

    # Initialize polygons
    polygons = np.zeros((total_cells, 8), dtype=np.float32)
    polygons[:, 0:2] = dimensions.reshape(-1, 4)[:, [0, 1]]  # x1, y1
    polygons[:, 2:4] = dimensions.reshape(-1, 4)[:, [2, 1]]  # x2, y1
    polygons[:, 4:6] = dimensions.reshape(-1, 4)[:, [2, 3]]  # x2, y2
    polygons[:, 6:8] = dimensions.reshape(-1, 4)[:, [0, 3]]  # x1, y2

    # Generate displacements for internal grid points only
    internal_points_height, internal_points_width = grid_height - 1, grid_width - 1
    displacements = random_generator.integers(
        -magnitude,
        magnitude + 1,
        size=(internal_points_height, internal_points_width, 2),
    ).astype(np.float32)

    # Apply displacements to internal polygon vertices
    for i in range(1, grid_height):
        for j in range(1, grid_width):
            dx, dy = displacements[i - 1, j - 1]

            # Bottom-right of cell (i-1, j-1)
            polygons[(i - 1) * grid_width + (j - 1), 4:6] += [dx, dy]

            # Bottom-left of cell (i-1, j)
            polygons[(i - 1) * grid_width + j, 6:8] += [dx, dy]

            # Top-right of cell (i, j-1)
            polygons[i * grid_width + (j - 1), 2:4] += [dx, dy]

            # Top-left of cell (i, j)
            polygons[i * grid_width + j, 0:2] += [dx, dy]

    return polygons


@handle_empty_array("keypoints")
def pad_keypoints(
    keypoints: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    border_mode: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Pad keypoints by a given amount.

    This function pads keypoints by a given amount.
    It handles both reflection and padding.

    Args:
        keypoints (np.ndarray): The keypoints to pad.
        pad_top (int): The amount to pad the top of the keypoints.
        pad_bottom (int): The amount to pad the bottom of the keypoints.
        pad_left (int): The amount to pad the left of the keypoints.
        pad_right (int): The amount to pad the right of the keypoints.
        border_mode (int): The border mode to use.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: The padded keypoints.

    """
    if border_mode not in REFLECT_BORDER_MODES:
        shift_vector = np.array([pad_left, pad_top, 0])
        return shift_keypoints(keypoints, shift_vector)

    grid_dimensions = get_pad_grid_dimensions(
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        image_shape,
    )

    keypoints = generate_reflected_keypoints(keypoints, grid_dimensions, image_shape)

    rows, cols = image_shape[:2]

    # Calculate the number of grid cells added on each side
    original_row, original_col = grid_dimensions["original_position"]

    # Subtract the offset based on the number of added grid cells
    keypoints[:, 0] -= original_col * cols - pad_left  # x
    keypoints[:, 1] -= original_row * rows - pad_top  # y

    new_height = pad_top + pad_bottom + rows
    new_width = pad_left + pad_right + cols

    return validate_keypoints(keypoints, (new_height, new_width))


def validate_keypoints(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Validate keypoints and remove those that fall outside the image boundaries.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, M) where N is the number of keypoints
                                and M >= 2. The first two columns represent x and y coordinates.
        image_shape (tuple[int, int]): Shape of the image as (height, width).

    Returns:
        np.ndarray: Array of valid keypoints that fall within the image boundaries.

    Note:
        This function only checks the x and y coordinates (first two columns) of the keypoints.
        Any additional columns (e.g., angle, scale) are preserved for valid keypoints.

    """
    rows, cols = image_shape[:2]

    x, y = keypoints[:, 0], keypoints[:, 1]

    valid_indices = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)

    return keypoints[valid_indices]


def shift_keypoints(keypoints: np.ndarray, shift_vector: np.ndarray) -> np.ndarray:
    """Shift keypoints by a given shift vector.

    This function shifts the keypoints by a given shift vector.
    It only shifts the x, y and z coordinates of the keypoints.

    Args:
        keypoints (np.ndarray): The keypoints to shift.
        shift_vector (np.ndarray): The shift vector to apply to the keypoints.

    Returns:
        np.ndarray: The shifted keypoints.

    """
    shifted_keypoints = keypoints.copy()
    shifted_keypoints[:, :3] += shift_vector[:3]  # Only shift x, y and z
    return shifted_keypoints


def generate_reflected_keypoints(
    keypoints: np.ndarray,
    grid_dims: dict[str, tuple[int, int]],
    image_shape: tuple[int, int],
    center_in_origin: bool = False,
) -> np.ndarray:
    """Generate reflected keypoints for the entire reflection grid.

    This function creates a grid of keypoints by reflecting and shifting the original keypoints.
    It handles both centered and non-centered grids based on the `center_in_origin` parameter.

    Args:
        keypoints (np.ndarray): Original keypoints array of shape (N, 4+), where N is the number of keypoints,
                                and each keypoint is represented by at least 4 values (x, y, angle, scale, ...).
        grid_dims (dict[str, tuple[int, int]]): A dictionary containing grid dimensions and original position.
            It should have the following keys:
            - "grid_shape": tuple[int, int] representing (grid_rows, grid_cols)
            - "original_position": tuple[int, int] representing (original_row, original_col)
        image_shape (tuple[int, int]): Shape of the original image as (height, width).
        center_in_origin (bool, optional): If True, center the grid at the origin. Default is False.

    Returns:
        np.ndarray: Array of reflected and shifted keypoints for the entire grid. The shape is
                    (N * grid_rows * grid_cols, 4+), where N is the number of original keypoints.

    Note:
        - The function handles keypoint flipping and shifting to create a grid of reflected keypoints.
        - It preserves the angle and scale information of the keypoints during transformations.
        - The resulting grid can be either centered at the origin or positioned based on the original grid.

    """
    grid_rows, grid_cols = grid_dims["grid_shape"]
    original_row, original_col = grid_dims["original_position"]

    # Prepare flipped versions of keypoints
    keypoints_hflipped = flip_keypoints(
        keypoints,
        flip_horizontal=True,
        image_shape=image_shape,
    )
    keypoints_vflipped = flip_keypoints(
        keypoints,
        flip_vertical=True,
        image_shape=image_shape,
    )
    keypoints_hvflipped = flip_keypoints(
        keypoints,
        flip_horizontal=True,
        flip_vertical=True,
        image_shape=image_shape,
    )

    rows, cols = image_shape[:2]

    # Shift all versions to the original position
    shift_vector = np.array(
        [original_col * cols, original_row * rows, 0, 0, 0],
    )  # Only shift x and y
    keypoints = shift_keypoints(keypoints, shift_vector)
    keypoints_hflipped = shift_keypoints(keypoints_hflipped, shift_vector)
    keypoints_vflipped = shift_keypoints(keypoints_vflipped, shift_vector)
    keypoints_hvflipped = shift_keypoints(keypoints_hvflipped, shift_vector)

    new_keypoints = []

    for grid_row in range(grid_rows):
        for grid_col in range(grid_cols):
            # Determine which version of keypoints to use based on grid position
            if (grid_row - original_row) % 2 == 0 and (grid_col - original_col) % 2 == 0:
                current_keypoints = keypoints
            elif (grid_row - original_row) % 2 == 0:
                current_keypoints = keypoints_hflipped
            elif (grid_col - original_col) % 2 == 0:
                current_keypoints = keypoints_vflipped
            else:
                current_keypoints = keypoints_hvflipped

            # Shift to the current grid cell
            cell_shift = np.array(
                [
                    (grid_col - original_col) * cols,
                    (grid_row - original_row) * rows,
                    0,
                    0,
                    0,
                ],
            )
            shifted_keypoints = shift_keypoints(current_keypoints, cell_shift)

            new_keypoints.append(shifted_keypoints)

    result = np.vstack(new_keypoints)

    return shift_keypoints(result, -shift_vector) if center_in_origin else result


@handle_empty_array("keypoints")
def flip_keypoints(
    keypoints: np.ndarray,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    image_shape: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Flip keypoints horizontally or vertically.

    This function flips keypoints horizontally or vertically based on the provided parameters.
    It also flips the angle of the keypoints when flipping horizontally.

    Args:
        keypoints (np.ndarray): The keypoints to flip.
        flip_horizontal (bool): Whether to flip horizontally.
        flip_vertical (bool): Whether to flip vertically.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: The flipped keypoints.

    """
    rows, cols = image_shape[:2]
    flipped_keypoints = keypoints.copy()
    if flip_horizontal:
        flipped_keypoints[:, 0] = cols - flipped_keypoints[:, 0]
        flipped_keypoints[:, 3] = -flipped_keypoints[:, 3]  # Flip angle
    if flip_vertical:
        flipped_keypoints[:, 1] = rows - flipped_keypoints[:, 1]
        flipped_keypoints[:, 3] = -flipped_keypoints[:, 3]  # Flip angle
    return flipped_keypoints


def create_affine_transformation_matrix(
    translate: Mapping[str, float],
    shear: dict[str, float],
    scale: dict[str, float],
    rotate: float,
    shift: tuple[float, float],
) -> np.ndarray:
    """Create an affine transformation matrix combining translation, shear, scale, and rotation.

    Args:
        translate (dict[str, float]): Translation in x and y directions.
        shear (dict[str, float]): Shear in x and y directions (in degrees).
        scale (dict[str, float]): Scale factors for x and y directions.
        rotate (float): Rotation angle in degrees.
        shift (tuple[float, float]): Shift to apply before and after transformations.

    Returns:
        np.ndarray: The resulting 3x3 affine transformation matrix.

    """
    # Convert angles to radians
    rotate_rad = np.deg2rad(rotate % 360)

    shear_x_rad = np.deg2rad(shear["x"])
    shear_y_rad = np.deg2rad(shear["y"])

    # Create individual transformation matrices
    # 1. Shift to top-left
    m_shift_topleft = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]])

    # 2. Scale
    m_scale = np.array([[scale["x"], 0, 0], [0, scale["y"], 0], [0, 0, 1]])

    # 3. Rotation
    m_rotate = np.array(
        [
            [np.cos(rotate_rad), np.sin(rotate_rad), 0],
            [-np.sin(rotate_rad), np.cos(rotate_rad), 0],
            [0, 0, 1],
        ],
    )

    # 4. Shear
    m_shear = np.array(
        [[1, np.tan(shear_x_rad), 0], [np.tan(shear_y_rad), 1, 0], [0, 0, 1]],
    )

    # 5. Translation
    m_translate = np.array([[1, 0, translate["x"]], [0, 1, translate["y"]], [0, 0, 1]])

    # 6. Shift back to center
    m_shift_center = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])

    # Combine all transformations
    # The order is important: transformations are applied from right to left
    m = m_shift_center @ m_translate @ m_shear @ m_rotate @ m_scale @ m_shift_topleft

    # Ensure the last row is exactly [0, 0, 1]
    m[2] = [0, 0, 1]

    return m


def compute_transformed_image_bounds(
    matrix: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the bounds of an image after applying an affine transformation.

    Args:
        matrix (np.ndarray): The 3x3 affine transformation matrix.
        image_shape (Tuple[int, int]): The shape of the image as (height, width).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - min_coords: An array with the minimum x and y coordinates.
            - max_coords: An array with the maximum x and y coordinates.

    """
    height, width = image_shape[:2]

    # Define the corners of the image
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])

    # Transform the corners
    transformed_corners = corners @ matrix.T
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]

    # Calculate the bounding box of the transformed corners
    min_coords = np.floor(transformed_corners.min(axis=0)).astype(int)
    max_coords = np.ceil(transformed_corners.max(axis=0)).astype(int)

    return min_coords, max_coords


def compute_affine_warp_output_shape(
    matrix: np.ndarray,
    input_shape: tuple[int, ...],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Compute the output shape of an affine warp.

    This function computes the output shape of an affine warp based on the input matrix and input shape.
    It calculates the transformed image bounds and then determines the output shape based on the input shape.

    Args:
        matrix (np.ndarray): The 3x3 affine transformation matrix.
        input_shape (tuple[int, ...]): The shape of the input image as (height, width, ...).

    Returns:
        tuple[np.ndarray, tuple[int, int]]: A tuple containing:
            - matrix: The 3x3 affine transformation matrix.
            - output_shape: The output shape of the affine warp.

    """
    height, width = input_shape[:2]

    if height == 0 or width == 0:
        return matrix, cast("tuple[int, int]", input_shape[:2])

    min_coords, max_coords = compute_transformed_image_bounds(matrix, (height, width))
    minc, minr = min_coords
    maxc, maxr = max_coords

    out_height = maxr - minr + 1
    out_width = maxc - minc + 1

    if len(input_shape) == NUM_MULTI_CHANNEL_DIMENSIONS:
        output_shape = np.ceil((out_height, out_width, input_shape[2]))
    else:
        output_shape = np.ceil((out_height, out_width))

    output_shape_tuple = tuple(int(v) for v in output_shape.tolist())

    # fit output image in new shape
    translation = np.array([[1, 0, -minc], [0, 1, -minr], [0, 0, 1]])
    matrix = translation @ matrix

    return matrix, cast("tuple[int, int]", output_shape_tuple)


def center(image_shape: tuple[int, int]) -> tuple[float, float]:
    """Calculate the center coordinates if image. Used by images, masks and keypoints.

    Args:
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        tuple[float, float]: center_x, center_y

    """
    height, width = image_shape[:2]
    return width / 2 - 0.5, height / 2 - 0.5


def center_bbox(image_shape: tuple[int, int]) -> tuple[float, float]:
    """Calculate the center coordinates for of image for bounding boxes.

    Args:
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        tuple[float, float]: center_x, center_y

    """
    height, width = image_shape[:2]
    return width / 2, height / 2


def generate_grid(
    image_shape: tuple[int, int],
    steps_x: list[float],
    steps_y: list[float],
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a distorted grid for image transformation based on given step sizes.

    This function creates two 2D arrays (map_x and map_y) that represent a distorted version
    of the original image grid. These arrays can be used with OpenCV's remap function to
    apply grid distortion to an image.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        steps_x (list[float]): List of step sizes for the x-axis distortion. The length
            should be num_steps + 1. Each value represents the relative step size for
            a segment of the grid in the x direction.
        steps_y (list[float]): List of step sizes for the y-axis distortion. The length
            should be num_steps + 1. Each value represents the relative step size for
            a segment of the grid in the y direction.
        num_steps (int): The number of steps to divide each axis into. This determines
            the granularity of the distortion grid.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two 2D numpy arrays:
            - map_x: A 2D array of float32 values representing the x-coordinates
              of the distorted grid.
            - map_y: A 2D array of float32 values representing the y-coordinates
              of the distorted grid.

    Note:
        - The function generates a grid where each cell can be distorted independently.
        - The distortion is controlled by the steps_x and steps_y parameters, which
          determine how much each grid line is shifted.
        - The resulting map_x and map_y can be used directly with cv2.remap() to
          apply the distortion to an image.
        - The distortion is applied smoothly across each grid cell using linear
          interpolation.

    Example:
        >>> image_shape = (100, 100)
        >>> steps_x = [1.1, 0.9, 1.0, 1.2, 0.95, 1.05]
        >>> steps_y = [0.9, 1.1, 1.0, 1.1, 0.9, 1.0]
        >>> num_steps = 5
        >>> map_x, map_y = generate_grid(image_shape, steps_x, steps_y, num_steps)
        >>> distorted_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    """
    height, width = image_shape[:2]
    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0.0
    for idx, step in enumerate(steps_x):
        x = idx * x_step
        start = int(x)
        end = min(int(x) + x_step, width)
        cur = prev + x_step * step
        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0.0
    for idx, step in enumerate(steps_y):
        y = idx * y_step
        start = int(y)
        end = min(int(y) + y_step, height)
        cur = prev + y_step * step
        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    return np.meshgrid(xx, yy)


def normalize_grid_distortion_steps(
    image_shape: tuple[int, int],
    num_steps: int,
    x_steps: list[float],
    y_steps: list[float],
) -> dict[str, np.ndarray]:
    """Normalize the grid distortion steps.

    This function normalizes the grid distortion steps, ensuring that the distortion never leaves the image bounds.
    It compensates for smaller last steps in the source image and normalizes the steps such that the distortion
    never leaves the image bounds.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        num_steps (int): The number of steps to divide each axis into. This determines
            the granularity of the distortion grid.
        x_steps (list[float]): List of step sizes for the x-axis distortion. The length
            should be num_steps + 1. Each value represents the relative step size for
            a segment of the grid in the x direction.
        y_steps (list[float]): List of step sizes for the y-axis distortion. The length
            should be num_steps + 1. Each value represents the relative step size for
            a segment of the grid in the y direction.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the normalized step sizes for the x and y axes.

    """
    height, width = image_shape[:2]

    # compensate for smaller last steps in source image.
    x_step = width // num_steps
    last_x_step = min(width, ((num_steps + 1) * x_step)) - (num_steps * x_step)
    x_steps[-1] *= last_x_step / x_step

    y_step = height // num_steps
    last_y_step = min(height, ((num_steps + 1) * y_step)) - (num_steps * y_step)
    y_steps[-1] *= last_y_step / y_step

    # now normalize such that distortion never leaves image bounds.
    tx = width / math.floor(width / num_steps)
    ty = height / math.floor(height / num_steps)
    x_steps = np.array(x_steps) * (tx / np.sum(x_steps))
    y_steps = np.array(y_steps) * (ty / np.sum(y_steps))

    return {"steps_x": x_steps, "steps_y": y_steps}


def almost_equal_intervals(n: int, parts: int) -> np.ndarray:
    """Generates an array of nearly equal integer intervals that sum up to `n`.

    This function divides the number `n` into `parts` nearly equal parts. It ensures that
    the sum of all parts equals `n`, and the difference between any two parts is at most one.
    This is useful for distributing a total amount into nearly equal discrete parts.

    Args:
        n (int): The total value to be split.
        parts (int): The number of parts to split into.

    Returns:
        np.ndarray: An array of integers where each integer represents the size of a part.

    Example:
        >>> almost_equal_intervals(20, 3)
        array([7, 7, 6])  # Splits 20 into three parts: 7, 7, and 6
        >>> almost_equal_intervals(16, 4)
        array([4, 4, 4, 4])  # Splits 16 into four equal parts

    """
    part_size, remainder = divmod(n, parts)
    # Create an array with the base part size and adjust the first `remainder` parts by adding 1
    return np.array(
        [part_size + 1 if i < remainder else part_size for i in range(parts)],
    )


def generate_shuffled_splits(
    size: int,
    divisions: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate shuffled splits for a given dimension size and number of divisions.

    Args:
        size (int): Total size of the dimension (height or width).
        divisions (int): Number of divisions (rows or columns).
        random_generator (np.random.Generator | None): The random generator to use for shuffling the splits.
            If None, the splits are not shuffled.

    Returns:
        np.ndarray: Cumulative edges of the shuffled intervals.

    """
    intervals = almost_equal_intervals(size, divisions)
    random_generator.shuffle(intervals)
    return np.insert(np.cumsum(intervals), 0, 0)


def split_uniform_grid(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Splits an image shape into a uniform grid specified by the grid dimensions.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        grid (tuple[int, int]): The grid size as (rows, columns).
        random_generator (np.random.Generator): The random generator to use for shuffling the splits.
            If None, the splits are not shuffled.

    Returns:
        np.ndarray: An array containing the tiles' coordinates in the format (start_y, start_x, end_y, end_x).

    Note:
        The function uses `generate_shuffled_splits` to generate the splits for the height and width of the image.
        The splits are then used to calculate the coordinates of the tiles.

    """
    n_rows, n_cols = grid

    height_splits = generate_shuffled_splits(
        image_shape[0],
        grid[0],
        random_generator=random_generator,
    )
    width_splits = generate_shuffled_splits(
        image_shape[1],
        grid[1],
        random_generator=random_generator,
    )

    # Calculate tiles coordinates
    tiles = [
        (height_splits[i], width_splits[j], height_splits[i + 1], width_splits[j + 1])
        for i in range(n_rows)
        for j in range(n_cols)
    ]

    return np.array(tiles, dtype=np.int16)


def generate_perspective_points(
    image_shape: tuple[int, int],
    scale: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate perspective points for a given image shape and scale.

    This function generates perspective points for a given image shape and scale.
    It uses a normal distribution to generate the points, and then modulates them to be within the image bounds.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        scale (float): The scale of the perspective points.
        random_generator (np.random.Generator): The random generator to use for generating the points.

    Returns:
        np.ndarray: The perspective points.

    """
    height, width = image_shape[:2]
    points = random_generator.normal(0, scale, (4, 2))
    points = np.mod(np.abs(points), 0.32)

    # top left -- no changes needed, just use jitter
    # top right
    points[1, 0] = 1.0 - points[1, 0]  # w = 1.0 - jitter
    # bottom right
    points[2] = 1.0 - points[2]  # w = 1.0 - jitter
    # bottom left
    points[3, 1] = 1.0 - points[3, 1]  # h = 1.0 - jitter

    points[:, 0] *= width
    points[:, 1] *= height

    return points


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in a clockwise manner.

    This function orders the points in a clockwise manner, ensuring that the points are in the correct
    order for perspective transformation.

    Args:
        pts (np.ndarray): The points to order.

    Returns:
        np.ndarray: The ordered points.

    """
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


def compute_perspective_params(
    points: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, int, int]:
    """Compute perspective transformation parameters.

    This function computes the perspective transformation parameters for a given set of points.
    It adjusts the points to ensure that the transformed image retains its original dimensions.

    Args:
        points (np.ndarray): The points to compute the perspective transformation parameters for.
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        tuple[np.ndarray, int, int]: The perspective transformation parameters and the maximum
            dimensions of the transformed image.

    """
    height, width = image_shape
    top_left, top_right, bottom_right, bottom_left = points

    def adjust_dimension(
        dim1: np.ndarray,
        dim2: np.ndarray,
        min_size: int = 2,
    ) -> float:
        size = np.sqrt(np.sum((dim1 - dim2) ** 2))
        if size < min_size:
            step_size = (min_size - size) / 2
            dim1[dim1 > dim2] += step_size
            dim2[dim1 > dim2] -= step_size
            dim1[dim1 <= dim2] -= step_size
            dim2[dim1 <= dim2] += step_size
            size = min_size
        return size

    max_width = max(
        adjust_dimension(top_right, top_left),
        adjust_dimension(bottom_right, bottom_left),
    )
    max_height = max(
        adjust_dimension(bottom_right, top_right),
        adjust_dimension(bottom_left, top_left),
    )

    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, dst)

    return matrix, int(max_width), int(max_height)


def expand_transform(
    matrix: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, int, int]:
    """Expand a transformation matrix to include padding.

    This function expands a transformation matrix to include padding, ensuring that the transformed
    image retains its original dimensions. It first calculates the destination points of the transformed
    image, then adjusts the matrix to include padding, and finally returns the expanded matrix and the
    maximum dimensions of the transformed image.

    Args:
        matrix (np.ndarray): The transformation matrix to expand.
        shape (tuple[int, int]): The shape of the image.

    Returns:
        tuple[np.ndarray, int, int]: The expanded matrix and the maximum dimensions of the transformed image.

    """
    height, width = shape[:2]
    rect = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    dst = cv2.perspectiveTransform(np.array([rect]), matrix)[0]

    dst -= dst.min(axis=0, keepdims=True)
    dst = np.around(dst, decimals=0)

    matrix_expanded = cv2.getPerspectiveTransform(rect, dst)
    max_width, max_height = dst.max(axis=0)
    return matrix_expanded, int(max_width), int(max_height)


def create_piecewise_affine_maps(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    scale: float,
    absolute_scale: bool,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Create maps for piecewise affine transformation using OpenCV's remap function.

    This function creates maps for piecewise affine transformation using OpenCV's remap function.
    It generates the control points for the transformation, then uses the remap function to create
    the transformation maps.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        grid (tuple[int, int]): The grid size as (rows, columns).
        scale (float): The scale of the transformation.
        absolute_scale (bool): Whether to use absolute scale.
        random_generator (np.random.Generator): The random generator to use for generating the points.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: The transformation maps.

    """
    height, width = image_shape[:2]
    nb_rows, nb_cols = grid

    # Input validation
    if height <= 0 or width <= 0 or nb_rows <= 0 or nb_cols <= 0:
        raise ValueError("Dimensions must be positive")
    if scale <= 0:
        return None, None

    # Create source points grid
    y = np.linspace(0, height - 1, nb_rows, dtype=np.float32)
    x = np.linspace(0, width - 1, nb_cols, dtype=np.float32)
    xx_src, yy_src = np.meshgrid(x, y)

    # Initialize destination maps at full resolution
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    # Generate jitter for control points
    jitter_scale = scale / 3 if absolute_scale else scale * min(width, height) / 3

    jitter = random_generator.normal(0, jitter_scale, (nb_rows, nb_cols, 2)).astype(
        np.float32,
    )

    # Create control points with jitter
    control_points = np.zeros((nb_rows * nb_cols, 4), dtype=np.float32)
    for i in range(nb_rows):
        for j in range(nb_cols):
            idx = i * nb_cols + j
            # Source points
            control_points[idx, 0] = xx_src[i, j]
            control_points[idx, 1] = yy_src[i, j]
            # Destination points with jitter
            control_points[idx, 2] = np.clip(
                xx_src[i, j] + jitter[i, j, 1],
                0,
                width - 1,
            )
            control_points[idx, 3] = np.clip(
                yy_src[i, j] + jitter[i, j, 0],
                0,
                height - 1,
            )

    # Create full resolution maps
    for i in range(height):
        for j in range(width):
            # Find nearest control points and interpolate
            dx = j - control_points[:, 0]
            dy = i - control_points[:, 1]
            dist = dx * dx + dy * dy
            weights = 1 / (dist + 1e-8)
            weights = weights / np.sum(weights)

            map_x[i, j] = np.sum(weights * control_points[:, 2])
            map_y[i, j] = np.sum(weights * control_points[:, 3])

    # Ensure output is within bounds
    map_x = np.clip(map_x, 0, width - 1, out=map_x)
    map_y = np.clip(map_y, 0, height - 1, out=map_y)

    return map_x, map_y


@handle_empty_array("bboxes")
def bboxes_piecewise_affine(
    bboxes: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    border_mode: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Apply a piecewise affine transformation to bounding boxes.

    This function applies a piecewise affine transformation to the bounding boxes of an image.
    It first converts the bounding boxes to masks, then applies the transformation, and finally
    converts the transformed masks back to bounding boxes.

    Args:
        bboxes (np.ndarray): The bounding boxes to transform.
        map_x (np.ndarray): The x-coordinates of the transformation.
        map_y (np.ndarray): The y-coordinates of the transformation.
        border_mode (int): The border mode to use for the transformation.
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        np.ndarray: The transformed bounding boxes.

    """
    masks = masks_from_bboxes(bboxes, image_shape).transpose(1, 2, 0)

    map_xy = np.stack([map_x, map_y], axis=-1).astype(np.float32)

    # Call remap with the combined map and empty second map
    transformed_masks = cv2.remap(
        masks,
        map_xy,
        None,
        cv2.INTER_NEAREST,
        borderMode=border_mode,
        borderValue=0,
    )

    if transformed_masks.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        transformed_masks = transformed_masks.transpose(2, 0, 1)

    # Normalize the returned bboxes
    bboxes[:, :4] = bboxes_from_masks(transformed_masks)

    return bboxes


def get_dimension_padding(
    current_size: int,
    min_size: int | None,
    divisor: int | None,
) -> tuple[int, int]:
    """Calculate padding for a single dimension.

    Args:
        current_size (int): Current size of the dimension
        min_size (int | None): Minimum size requirement, if any
        divisor (int | None): Divisor for padding to make size divisible, if any

    Returns:
        tuple[int, int]: (pad_before, pad_after)

    """
    if min_size is not None:
        if current_size < min_size:
            pad_before = int((min_size - current_size) / 2.0)
            pad_after = min_size - current_size - pad_before
            return pad_before, pad_after
    elif divisor is not None:
        remainder = current_size % divisor
        if remainder > 0:
            total_pad = divisor - remainder
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            return pad_before, pad_after

    return 0, 0


def get_padding_params(
    image_shape: tuple[int, int],
    min_height: int | None,
    min_width: int | None,
    pad_height_divisor: int | None,
    pad_width_divisor: int | None,
) -> tuple[int, int, int, int]:
    """Calculate padding parameters based on target dimensions.

    Args:
        image_shape (tuple[int, int]): (height, width) of the image
        min_height (int | None): Minimum height requirement, if any
        min_width (int | None): Minimum width requirement, if any
        pad_height_divisor (int | None): Divisor for height padding, if any
        pad_width_divisor (int | None): Divisor for width padding, if any

    Returns:
        tuple[int, int, int, int]: (pad_top, pad_bottom, pad_left, pad_right)

    """
    rows, cols = image_shape[:2]

    h_pad_top, h_pad_bottom = get_dimension_padding(
        rows,
        min_height,
        pad_height_divisor,
    )
    w_pad_left, w_pad_right = get_dimension_padding(cols, min_width, pad_width_divisor)

    return h_pad_top, h_pad_bottom, w_pad_left, w_pad_right


def adjust_padding_by_position(
    h_top: int,
    h_bottom: int,
    w_left: int,
    w_right: int,
    position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"],
    py_random: np.random.RandomState,
) -> tuple[int, int, int, int]:
    """Adjust padding values based on desired position."""
    if position == "center":
        return h_top, h_bottom, w_left, w_right

    if position == "top_left":
        return 0, h_top + h_bottom, 0, w_left + w_right

    if position == "top_right":
        return 0, h_top + h_bottom, w_left + w_right, 0

    if position == "bottom_left":
        return h_top + h_bottom, 0, 0, w_left + w_right

    if position == "bottom_right":
        return h_top + h_bottom, 0, w_left + w_right, 0

    if position == "random":
        h_pad = h_top + h_bottom
        w_pad = w_left + w_right
        h_top = py_random.randint(0, h_pad)
        h_bottom = h_pad - h_top
        w_left = py_random.randint(0, w_pad)
        w_right = w_pad - w_left
        return h_top, h_bottom, w_left, w_right

    raise ValueError(f"Unknown position: {position}")


def swap_tiles_on_keypoints(
    keypoints: np.ndarray,
    tiles: np.ndarray,
    mapping: np.ndarray,
) -> np.ndarray:
    """Swap the positions of keypoints based on a tile mapping.

    This function takes a set of keypoints and repositions them according to a mapping of tile swaps.
    Keypoints are moved from their original tiles to new positions in the swapped tiles.

    Args:
        keypoints (np.ndarray): A 2D numpy array of shape (N, 2) where N is the number of keypoints.
                                Each row represents a keypoint's (x, y) coordinates.
        tiles (np.ndarray): A 2D numpy array of shape (M, 4) where M is the number of tiles.
                            Each row represents a tile's (start_y, start_x, end_y, end_x) coordinates.
        mapping (np.ndarray): A 1D numpy array of shape (M,) where M is the number of tiles.
                              Each element i contains the index of the tile that tile i should be swapped with.

    Returns:
        np.ndarray: A 2D numpy array of the same shape as the input keypoints, containing the new positions
                    of the keypoints after the tile swap.

    Raises:
        RuntimeWarning: If any keypoint is not found within any tile.

    Notes:
        - Keypoints that do not fall within any tile will remain unchanged.
        - The function assumes that the tiles do not overlap and cover the entire image space.

    """
    if not keypoints.size:
        return keypoints

    # Broadcast keypoints and tiles for vectorized comparison
    kp_x = keypoints[:, 0][:, np.newaxis]  # Shape: (num_keypoints, 1)
    kp_y = keypoints[:, 1][:, np.newaxis]  # Shape: (num_keypoints, 1)

    start_y, start_x, end_y, end_x = tiles.T  # Each shape: (num_tiles,)

    # Check if each keypoint is inside each tile
    in_tile = (kp_y >= start_y) & (kp_y < end_y) & (kp_x >= start_x) & (kp_x < end_x)

    # Find which tile each keypoint belongs to
    tile_indices = np.argmax(in_tile, axis=1)

    # Check if any keypoint is not in any tile
    not_in_any_tile = ~np.any(in_tile, axis=1)
    if np.any(not_in_any_tile):
        warn(
            "Some keypoints are not in any tile. They will be returned unchanged. This is unexpected and should be "
            "investigated.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Get the new tile indices
    new_tile_indices = np.array(mapping)[tile_indices]

    # Calculate the offsets
    old_start_x = tiles[tile_indices, 1]
    old_start_y = tiles[tile_indices, 0]
    new_start_x = tiles[new_tile_indices, 1]
    new_start_y = tiles[new_tile_indices, 0]

    # Apply the transformation
    new_keypoints = keypoints.copy()
    new_keypoints[:, 0] = (keypoints[:, 0] - old_start_x) + new_start_x
    new_keypoints[:, 1] = (keypoints[:, 1] - old_start_y) + new_start_y

    # Keep original coordinates for keypoints not in any tile
    new_keypoints[not_in_any_tile] = keypoints[not_in_any_tile]

    return new_keypoints


def swap_tiles_on_image(
    image: np.ndarray,
    tiles: np.ndarray,
    mapping: list[int] | None = None,
) -> np.ndarray:
    """Swap tiles on the image according to the new format.

    Args:
        image (np.ndarray): Input image.
        tiles (np.ndarray): Array of tiles with each tile as [start_y, start_x, end_y, end_x].
        mapping (list[int] | None): list of new tile indices.

    Returns:
        np.ndarray: Output image with tiles swapped according to the random shuffle.

    """
    # If no tiles are provided, return a copy of the original image
    if tiles.size == 0 or mapping is None:
        return image.copy()

    # Create a copy of the image to retain original for reference
    new_image = np.empty_like(image)
    for num, new_index in enumerate(mapping):
        start_y, start_x, end_y, end_x = tiles[new_index]
        start_y_orig, start_x_orig, end_y_orig, end_x_orig = tiles[num]
        # Assign the corresponding tile from the original image to the new image
        new_image[start_y:end_y, start_x:end_x] = image[
            start_y_orig:end_y_orig,
            start_x_orig:end_x_orig,
        ]

    return new_image


def is_valid_component(
    component_area: float,
    original_area: float,
    min_area: float | None,
    min_visibility: float | None,
) -> bool:
    """Validate if a component meets the minimum requirements."""
    visibility = component_area / original_area
    return (min_area is None or component_area >= min_area) and (min_visibility is None or visibility >= min_visibility)


@handle_empty_array("bboxes")
def bboxes_grid_shuffle(
    bboxes: np.ndarray,
    tiles: np.ndarray,
    mapping: list[int],
    image_shape: tuple[int, int],
    min_area: float,
    min_visibility: float,
) -> np.ndarray:
    """Shuffle bounding boxes according to grid mapping.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)
        tiles (np.ndarray): Array of grid tiles
        mapping (list[int]): Mapping of tile indices
        image_shape (tuple[int, int]): Shape of the image (height, width)
        min_area (float): Minimum area of a bounding box to keep
        min_visibility (float): Minimum visibility ratio of a bounding box to keep

    Returns:
        np.ndarray: Shuffled bounding boxes

    """
    # Convert bboxes to masks
    masks = masks_from_bboxes(bboxes, image_shape)

    # Apply grid shuffle to each mask and handle split components
    all_component_masks = []
    extra_bbox_data = []  # Store additional bbox data for each component

    for idx, mask in enumerate(masks):
        original_area = np.sum(mask)  # Get original mask area

        # Shuffle the mask
        shuffled_mask = swap_tiles_on_image(mask, tiles, mapping)

        # Find connected components
        num_components, components = cv2.connectedComponents(
            shuffled_mask.astype(np.uint8),
        )

        # For each component, create a separate binary mask
        for comp_idx in range(1, num_components):  # Skip background (0)
            component_mask = (components == comp_idx).astype(np.uint8)

            # Calculate area and visibility ratio
            component_area = np.sum(component_mask)
            # Check if component meets minimum requirements
            if is_valid_component(
                component_area,
                original_area,
                min_area,
                min_visibility,
            ):
                all_component_masks.append(component_mask)
                # Append additional bbox data for this component
                if bboxes.shape[1] > NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS:
                    extra_bbox_data.append(bboxes[idx, 4:])

    # Convert all component masks to bboxes
    if all_component_masks:
        all_component_masks = np.array(all_component_masks)
        shuffled_bboxes = bboxes_from_masks(all_component_masks)

        # Add back additional bbox data if present
        if extra_bbox_data:
            extra_bbox_data = np.array(extra_bbox_data)
            return np.column_stack([shuffled_bboxes, extra_bbox_data])
    else:
        # Handle case where no valid components were found
        return np.zeros((0, bboxes.shape[1]), dtype=bboxes.dtype)

    return shuffled_bboxes


def create_shape_groups(tiles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Groups tiles by their shape and stores the indices for each shape."""
    shape_groups = defaultdict(list)
    for index, (start_y, start_x, end_y, end_x) in enumerate(tiles):
        shape = (end_y - start_y, end_x - start_x)
        shape_groups[shape].append(index)
    return shape_groups


def shuffle_tiles_within_shape_groups(
    shape_groups: dict[tuple[int, int], list[int]],
    random_generator: np.random.Generator,
) -> list[int]:
    """Shuffles indices within each group of similar shapes and creates a list where each
    index points to the index of the tile it should be mapped to.

    Args:
        shape_groups (dict[tuple[int, int], list[int]]): Groups of tile indices categorized by shape.
        random_generator (np.random.Generator): The random generator to use for shuffling the indices.
            If None, a new random generator will be used.

    Returns:
        list[int]: A list where each index is mapped to the new index of the tile after shuffling.

    """
    # Initialize the output list with the same size as the total number of tiles, filled with -1
    num_tiles = sum(len(indices) for indices in shape_groups.values())
    mapping = [-1] * num_tiles

    # Prepare the random number generator

    for indices in shape_groups.values():
        shuffled_indices = indices.copy()
        random_generator.shuffle(shuffled_indices)

        for old, new in zip(indices, shuffled_indices):
            mapping[old] = new

    return mapping


def compute_pairwise_distances(
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """Compute pairwise distances between two sets of points.

    Args:
        points1 (np.ndarray): First set of points with shape (N, 2)
        points2 (np.ndarray): Second set of points with shape (M, 2)

    Returns:
        np.ndarray: Matrix of pairwise distances with shape (N, M)

    """
    points1 = np.ascontiguousarray(points1, dtype=np.float32)
    points2 = np.ascontiguousarray(points2, dtype=np.float32)

    # Compute squared terms
    p1_squared = cv2.multiply(points1, points1).sum(axis=1, keepdims=True)
    p2_squared = cv2.multiply(points2, points2).sum(axis=1)[None, :]

    # Compute dot product
    dot_product = cv2.gemm(points1, points2.T, 1, None, 0)

    return p1_squared + p2_squared - 2 * dot_product


def compute_tps_weights(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Thin Plate Spline weights.

    Args:
        src_points (np.ndarray): Source control points with shape (num_points, 2)
        dst_points (np.ndarray): Destination control points with shape (num_points, 2)

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (nonlinear_weights, affine_weights)
        - nonlinear_weights: TPS kernel weights for nonlinear deformation (num_points, 2)
        - affine_weights: Weights for affine transformation (3, 2)
            [constant term, x scale/shear, y scale/shear]

    Note:
        The TPS interpolation is decomposed into:
        1. Nonlinear part (controlled by kernel weights)
        2. Affine part (global scaling, rotation, translation)

    """
    num_points = src_points.shape[0]

    # Compute pairwise distances
    distances = compute_pairwise_distances(src_points, src_points)

    kernel_matrix = np.where(
        distances > 0,
        distances * distances * cv2.log(distances + 1e-6),
        0,
    ).astype(np.float32)

    # Build system matrix efficiently
    affine_terms = np.empty((num_points, 3), dtype=np.float32)
    affine_terms[:, 0] = 1
    affine_terms[:, 1:] = src_points

    # Construct system matrix
    system_matrix = np.zeros((num_points + 3, num_points + 3), dtype=np.float32)
    system_matrix[:num_points, :num_points] = kernel_matrix
    system_matrix[:num_points, num_points:] = affine_terms
    system_matrix[num_points:, :num_points] = affine_terms.T

    # Prepare target coordinates
    target = np.zeros((num_points + 3, 2), dtype=np.float32)
    target[:num_points] = dst_points

    weights = cv2.solve(system_matrix, target, flags=cv2.DECOMP_LU)[1]

    return weights[:num_points], weights[num_points:]


def tps_transform(
    target_points: np.ndarray,
    control_points: np.ndarray,
    nonlinear_weights: np.ndarray,
    affine_weights: np.ndarray,
) -> np.ndarray:
    """Apply TPS transformation with consistent types."""
    # Ensure float32 type for all inputs
    target_points = np.ascontiguousarray(target_points, dtype=np.float32)
    control_points = np.ascontiguousarray(control_points, dtype=np.float32)
    nonlinear_weights = np.ascontiguousarray(nonlinear_weights, dtype=np.float32)
    affine_weights = np.ascontiguousarray(affine_weights, dtype=np.float32)

    distances = compute_pairwise_distances(target_points, control_points)

    # Ensure kernel matrix is float32
    kernel_matrix = np.where(
        distances > 0,
        distances * cv2.log(distances + 1e-6),
        0,
    ).astype(np.float32)

    # Prepare affine terms
    num_points = len(target_points)
    affine_terms = np.empty((num_points, 3), dtype=np.float32)
    affine_terms[:, 0] = 1
    affine_terms[:, 1:] = target_points

    # Matrix multiplications with consistent float32 type
    nonlinear_part = cv2.gemm(kernel_matrix, nonlinear_weights, 1, None, 0)
    affine_part = cv2.gemm(affine_terms, affine_weights, 1, None, 0)

    return nonlinear_part + affine_part


def get_camera_matrix_distortion_maps(
    image_shape: tuple[int, int],
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate distortion maps using camera matrix model.

    Args:
        image_shape (tuple[int, int]): Image shape (height, width)
        k (float): Distortion coefficient

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (map_x, map_y) distortion maps

    """
    height, width = image_shape[:2]

    center_x, center_y = width / 2, height / 2

    camera_matrix = np.array(
        [[width, 0, center_x], [0, height, center_y], [0, 0, 1]],
        dtype=np.float32,
    )
    distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        None,
        (width, height),
        cv2.CV_32FC1,
    )


def get_fisheye_distortion_maps(
    image_shape: tuple[int, int],
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate distortion maps using fisheye model.

    Args:
        image_shape (tuple[int, int]): Image shape (height, width)
        k (float): Distortion coefficient

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (map_x, map_y) distortion maps

    """
    height, width = image_shape[:2]

    center_x, center_y = width / 2, height / 2
    # Create coordinate grid
    y, x = np.mgrid[:height, :width].astype(np.float32)

    x = x - center_x
    y = y - center_y

    # Calculate polar coordinates
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    # Normalize radius by the maximum possible radius to keep distortion in check
    max_radius = math.sqrt(max(center_x, width - center_x) ** 2 + max(center_y, height - center_y) ** 2)
    r_norm = r / max_radius

    # Apply fisheye distortion to normalized radius
    r_dist = r * (1 + k * r_norm * r_norm)

    # Convert back to cartesian coordinates
    map_x = r_dist * np.cos(theta) + center_x
    map_y = r_dist * np.sin(theta) + center_y

    return map_x, map_y


def generate_control_points(num_control_points: int) -> np.ndarray:
    """Generate control points for TPS transformation.

    Args:
        num_control_points (int): Number of control points per side

    Returns:
        np.ndarray: Control points with shape (N, 2)

    """
    if num_control_points == 2:
        # Generate 4 corners + center point similar to Kornia
        return np.array(
            [
                [0, 0],  # top-left
                [0, 1],  # bottom-left
                [1, 0],  # top-right
                [1, 1],  # bottom-right
                [0.5, 0.5],  # center
            ],
            dtype=np.float32,
        )

        # Generate regular grid
    x = np.linspace(0, 1, num_control_points)
    y = np.linspace(0, 1, num_control_points)
    return np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)


def volume_hflip(volume: np.ndarray) -> np.ndarray:
    """Perform horizontal flip on a volume (numpy array).

    Flips the volume along the width axis (axis=2). Handles inputs with
    shapes (D, H, W) or (D, H, W, C).

    Args:
        volume (np.ndarray): Input volume.

    Returns:
        np.ndarray: Horizontally flipped volume.

    """
    return np.flip(volume, axis=2)


def volume_vflip(volume: np.ndarray) -> np.ndarray:
    """Perform vertical flip on a volume (numpy array).

    Flips the volume along the height axis (axis=1). Handles inputs with
    shapes (D, H, W) or (D, H, W, C).

    Args:
        volume (np.ndarray): Input volume.

    Returns:
        np.ndarray: Vertically flipped volume.

    """
    return np.flip(volume, axis=1)


def volumes_hflip(volumes: np.ndarray) -> np.ndarray:
    """Perform horizontal flip on a batch of volumes (numpy array).

    Flips the volumes along the width axis (axis=3). Handles inputs with
    shapes (B, D, H, W) or (B, D, H, W, C).

    Args:
        volumes (np.ndarray): Input batch of volumes.

    Returns:
        np.ndarray: Horizontally flipped batch of volumes.

    """
    # Width axis is 3 for both (B, D, H, W) and (B, D, H, W, C)
    return np.flip(volumes, axis=3)


def volumes_vflip(volumes: np.ndarray) -> np.ndarray:
    """Perform vertical flip on a batch of volumes (numpy array).

    Flips the volumes along the height axis (axis=2). Handles inputs with
    shapes (B, D, H, W) or (B, D, H, W, C).

    Args:
        volumes (np.ndarray): Input batch of volumes.

    Returns:
        np.ndarray: Vertically flipped batch of volumes.

    """
    # Height axis is 2 for both (B, D, H, W) and (B, D, H, W, C)
    return np.flip(volumes, axis=2)


def volume_rot90(volume: np.ndarray, factor: Literal[0, 1, 2, 3]) -> np.ndarray:
    """Rotate a volume 90 degrees counter-clockwise multiple times.

    Rotates the volume in the height-width plane (axes 1 and 2).
    Handles inputs with shapes (D, H, W) or (D, H, W, C).

    Args:
        volume (np.ndarray): Input volume.
        factor (Literal[0, 1, 2, 3]): Number of 90-degree rotations.

    Returns:
        np.ndarray: Rotated volume.

    """
    # Axes 1 (height) and 2 (width) for rotation
    return np.rot90(volume, k=factor, axes=(1, 2))


def volumes_rot90(volumes: np.ndarray, factor: Literal[0, 1, 2, 3]) -> np.ndarray:
    """Rotate a batch of volumes 90 degrees counter-clockwise multiple times.

    Rotates the volumes in the height-width plane (axes 2 and 3).
    Handles inputs with shapes (B, D, H, W) or (B, D, H, W, C).

    Args:
        volumes (np.ndarray): Input batch of volumes.
        factor (Literal[0, 1, 2, 3]): Number of 90-degree rotations

    Returns:
        np.ndarray: Rotated batch of volumes.

    """
    # Axes 2 (height) and 3 (width) for rotation
    return np.rot90(volumes, k=factor, axes=(2, 3))


@preserve_channel_dim
def erode(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply erosion to an image.

    This function applies erosion to an image using the cv2.erode function.

    Args:
        img (np.ndarray): Input image as a numpy array.
        kernel (np.ndarray): Kernel as a numpy array.

    Returns:
        np.ndarray: The eroded image.

    """
    return cv2.erode(img, kernel, iterations=1)


@preserve_channel_dim
def dilate(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply dilation to an image.

    This function applies dilation to an image using the cv2.dilate function.

    Args:
        img (np.ndarray): Input image as a numpy array.
        kernel (np.ndarray): Kernel as a numpy array.

    Returns:
        np.ndarray: The dilated image.

    """
    return cv2.dilate(img, kernel, iterations=1)


def morphology(
    img: np.ndarray,
    kernel: np.ndarray,
    operation: Literal["dilation", "erosion"],
) -> np.ndarray:
    """Apply morphology to an image.

    This function applies morphology to an image using the cv2.morphologyEx function.

    Args:
        img (np.ndarray): Input image as a numpy array.
        kernel (np.ndarray): Kernel as a numpy array.
        operation (Literal["dilation", "erosion"]): The operation to apply.

    Returns:
        np.ndarray: The morphology applied to the image.

    """
    if operation == "dilation":
        return dilate(img, kernel)
    if operation == "erosion":
        return erode(img, kernel)

    raise ValueError(f"Unsupported operation: {operation}")


@handle_empty_array("bboxes")
def bboxes_morphology(
    bboxes: np.ndarray,
    kernel: np.ndarray,
    operation: Literal["dilation", "erosion"],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Apply morphology to bounding boxes.

    This function applies morphology to bounding boxes by first converting the bounding
    boxes to a mask and then applying the morphology to the mask.

    Args:
        bboxes (np.ndarray): Bounding boxes as a numpy array.
        kernel (np.ndarray): Kernel as a numpy array.
        operation (Literal["dilation", "erosion"]): The operation to apply.
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        np.ndarray: The morphology applied to the bounding boxes.

    """
    bboxes = bboxes.copy()
    masks = masks_from_bboxes(bboxes, image_shape)
    masks = morphology(masks, kernel, operation)
    bboxes[:, :4] = bboxes_from_masks(masks)
    return bboxes
