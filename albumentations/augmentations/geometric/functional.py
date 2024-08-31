from __future__ import annotations

import math
from typing import Any, Callable, Literal, Sequence, Tuple, TypedDict, cast

import cv2
import numpy as np
import skimage.transform
from albucore.utils import clipped, get_num_channels, maybe_process_in_chunks, preserve_channel_dim

from albumentations import random_utils
from albumentations.augmentations.functional import bbox_from_mask, center
from albumentations.augmentations.utils import angle_2pi_range, handle_empty_array
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from albumentations.core.types import (
    NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    REFLECT_BORDER_MODES,
    ColorType,
    D4Type,
    ScalarType,
)

__all__ = [
    "optical_distortion",
    "elastic_transform_approximate",
    "elastic_transform_precise",
    "grid_distortion",
    "pad",
    "pad_with_params",
    "rotate",
    "elastic_transform",
    "resize",
    "scale",
    "_func_max_size",
    "longest_max_size",
    "smallest_max_size",
    "perspective",
    "rotation2d_matrix_to_euler_angles",
    "is_identity_matrix",
    "warp_affine",
    "piecewise_affine",
    "to_distance_maps",
    "from_distance_maps",
    "hflip",
    "hflip_cv2",
    "transpose",
    "vflip",
    "d4",
    "bboxes_rotate",
    "keypoints_rotate",
    "bboxes_d4",
    "keypoints_d4",
    "bboxes_rot90",
    "keypoints_rot90",
    "bboxes_transpose",
    "keypoints_transpose",
    "bboxes_vflip",
    "keypoints_vflip",
    "bboxes_hflip",
    "keypoints_hflip",
]

PAIR = 2

ROT90_180_FACTOR = 2
ROT90_270_FACTOR = 3


@handle_empty_array
def bboxes_rot90(bboxes: np.ndarray, factor: int) -> np.ndarray:
    """Rotates bounding boxes by 90 degrees CCW (see np.rot90)

    Args:
        bboxes: A numpy array of bounding boxes with shape (num_bboxes, 4+).
                Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).
        factor: Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.

    Returns:
        np.ndarray: A numpy array of rotated bounding boxes with the same shape as input.

    Raises:
        ValueError: If factor is not in set {0, 1, 2, 3}.
    """
    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter factor must be in set {0, 1, 2, 3}")

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


@handle_empty_array
def bboxes_d4(
    bboxes: np.ndarray,
    group_member: D4Type,
) -> np.ndarray:
    """Applies a `D_4` symmetry group transformation to a bounding box.

    The function transforms a bounding box according to the specified group member from the `D_4` group.
    These transformations include rotations and reflections, specified to work on an image's bounding box given
    its dimensions.

    Parameters:
    -  bboxes: A numpy array of bounding boxes with shape (num_bboxes, 4+).
                Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).
    - group_member (D4Type): A string identifier for the `D_4` group transformation to apply.
        Valid values are 'e', 'r90', 'r180', 'r270', 'v', 'hvt', 'h', 't'.

    Returns:
    - BoxInternalType: The transformed bounding box.

    Raises:
    - ValueError: If an invalid group member is specified.

    Examples:
    - Applying a 90-degree rotation:
      `bbox_d4((10, 20, 110, 120), 'r90')`
      This would rotate the bounding box 90 degrees within a 100x100 image.
    """
    transformations = {
        "e": lambda x: x,  # Identity transformation
        "r90": lambda x: bboxes_rot90(x, 1),  # Rotate 90 degrees
        "r180": lambda x: bboxes_rot90(x, 2),  # Rotate 180 degrees
        "r270": lambda x: bboxes_rot90(x, 3),  # Rotate 270 degrees
        "v": lambda x: bboxes_vflip(x),  # Vertical flip
        "hvt": lambda x: bboxes_transpose(bboxes_rot90(x, 2)),  # Reflect over anti-diagonal
        "h": lambda x: bboxes_hflip(x),  # Horizontal flip
        "t": lambda x: bboxes_transpose(x),  # Transpose (reflect over main diagonal)
    }

    # Execute the appropriate transformation
    if group_member in transformations:
        return transformations[group_member](bboxes)

    raise ValueError(f"Invalid group member: {group_member}")


@handle_empty_array
@angle_2pi_range
def keypoints_rot90(
    keypoints: np.ndarray,
    factor: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rotate keypoints by 90 degrees counter-clockwise (CCW) a specified number of times.

    Args:
        keypoints (np.ndarray): An array of keypoints with shape (N, 4+) in the format (x, y, angle, scale, ...).
        factor (int): The number of 90 degree CCW rotations to apply. Must be in the range [0, 3].
        image_shape (tuple[int, int]): The shape of the image (height, width).

    Returns:
        np.ndarray: The rotated keypoints with the same shape as the input.

    Raises:
        ValueError: If the factor is not in the set {0, 1, 2, 3}.
    """
    if factor not in {0, 1, 2, 3}:
        raise ValueError("Parameter factor must be in set {0, 1, 2, 3}")

    if factor == 0:
        return keypoints

    height, width = image_shape[:2]
    rotated_keypoints = keypoints.copy().astype(np.float32)

    x, y, angle = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]

    if factor == 1:
        rotated_keypoints[:, 0] = y
        rotated_keypoints[:, 1] = width - 1 - x
        rotated_keypoints[:, 2] = angle - np.pi / 2
    elif factor == ROT90_180_FACTOR:
        rotated_keypoints[:, 0] = width - 1 - x
        rotated_keypoints[:, 1] = height - 1 - y
        rotated_keypoints[:, 2] = angle - np.pi
    elif factor == ROT90_270_FACTOR:
        rotated_keypoints[:, 0] = height - 1 - y
        rotated_keypoints[:, 1] = x
        rotated_keypoints[:, 2] = angle + np.pi / 2

    return rotated_keypoints


@handle_empty_array
def keypoints_d4(
    keypoints: np.ndarray,
    group_member: D4Type,
    image_shape: tuple[int, int],
    **params: Any,
) -> np.ndarray:
    """Applies a `D_4` symmetry group transformation to a keypoint.

    This function adjusts a keypoint's coordinates according to the specified `D_4` group transformation,
    which includes rotations and reflections suitable for image processing tasks. These transformations account
    for the dimensions of the image to ensure the keypoint remains within its boundaries.

    Parameters:
    - keypoints (np.ndarray): An array of keypoints with shape (N, 4+) in the format (x, y, angle, scale, ...).
    -group_member (D4Type): A string identifier for the `D_4` group transformation to apply.
        Valid values are 'e', 'r90', 'r180', 'r270', 'v', 'hv', 'h', 't'.
    - image_shape (tuple[int, int]): The shape of the image.
    - params (Any): Not used

    Returns:
    - KeypointInternalType: The transformed keypoint.

    Raises:
    - ValueError: If an invalid group member is specified, indicating that the specified transformation does not exist.

    Examples:
    - Rotating a keypoint by 90 degrees in a 100x100 image:
      `keypoint_d4((50, 30), 'r90', 100, 100)`
      This would move the keypoint from (50, 30) to (70, 50) assuming standard coordinate transformations.
    """
    rows, cols = image_shape[:2]
    transformations = {
        "e": lambda x: x,  # Identity transformation
        "r90": lambda x: keypoints_rot90(x, 1, image_shape),  # Rotate 90 degrees
        "r180": lambda x: keypoints_rot90(x, 2, image_shape),  # Rotate 180 degrees
        "r270": lambda x: keypoints_rot90(x, 3, image_shape),  # Rotate 270 degrees
        "v": lambda x: keypoints_vflip(x, rows),  # Vertical flip
        "hvt": lambda x: keypoints_transpose(keypoints_rot90(x, 2, image_shape)),  # Reflect over anti diagonal
        "h": lambda x: keypoints_hflip(x, cols),  # Horizontal flip
        "t": lambda x: keypoints_transpose(x),  # Transpose (reflect over main diagonal)
    }
    # Execute the appropriate transformation
    if group_member in transformations:
        return transformations[group_member](keypoints)

    raise ValueError(f"Invalid group member: {group_member}")


@preserve_channel_dim
def rotate(
    img: np.ndarray,
    angle: float,
    interpolation: int,
    border_mode: int,
    value: ColorType | None = None,
) -> np.ndarray:
    image_shape = img.shape[:2]

    image_center = center(image_shape)
    matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    height, width = image_shape

    warp_fn = maybe_process_in_chunks(
        warp_affine_with_value_extension,
        matrix=matrix,
        dsize=(width, height),
        flags=interpolation,
        border_mode=border_mode,
        border_value=value,
    )
    return warp_fn(img)


@handle_empty_array
def bboxes_rotate(
    bboxes: np.ndarray,
    angle: float,
    method: Literal["largest_box", "ellipse"],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rotates bounding boxes by angle degrees.

    Args:
        bboxes: A numpy array of bounding boxes with shape (num_bboxes, 4+).
                Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).
        angle: Angle of rotation in degrees.
        method: Rotation method used. Should be one of: "largest_box", "ellipse".
        image_shape: Image shape (height, width).

    Returns:
        np.ndarray: A numpy array of rotated bounding boxes with the same shape as input.

    Reference:
        https://arxiv.org/abs/2109.13488
    """
    bboxes = bboxes.copy()
    rows, cols = image_shape[:2]
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    scale = cols / float(rows)

    if method == "largest_box":
        x = np.column_stack([x_min, x_max, x_max, x_min]) - 0.5
        y = np.column_stack([y_min, y_min, y_max, y_max]) - 0.5
    elif method == "ellipse":
        w = (x_max - x_min) / 2
        h = (y_max - y_min) / 2
        data = np.arange(0, 360, dtype=np.float32)
        x = w[:, np.newaxis] * np.sin(np.radians(data)) + (w + x_min - 0.5)[:, np.newaxis]
        y = h[:, np.newaxis] * np.cos(np.radians(data)) + (h + y_min - 0.5)[:, np.newaxis]
    else:
        raise ValueError(f"Method {method} is not a valid rotation method.")

    angle_rad = np.deg2rad(angle)
    x_t = (np.cos(angle_rad) * x * scale + np.sin(angle_rad) * y) / scale
    y_t = -np.sin(angle_rad) * x * scale + np.cos(angle_rad) * y
    x_t = x_t + 0.5
    y_t = y_t + 0.5

    # Update the first 4 columns of the input array
    bboxes[:, 0] = np.min(x_t, axis=1)
    bboxes[:, 1] = np.min(y_t, axis=1)
    bboxes[:, 2] = np.max(x_t, axis=1)
    bboxes[:, 3] = np.max(y_t, axis=1)

    return bboxes


@handle_empty_array
@angle_2pi_range
def keypoints_rotate(
    keypoints: np.ndarray,
    angle: float,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rotate keypoints by a specified angle.

    Args:
        keypoints (np.ndarray): An array of keypoints with shape (N, 4+) in the format (x, y, angle, scale, ...).
        angle (float): The angle by which to rotate the keypoints, in degrees.
        image_shape (tuple[int, int]): The shape of the image the keypoints belong to (height, width).
        **params: Additional parameters.

    Returns:
        np.ndarray: The rotated keypoints with the same shape as the input.

    Note:
        The rotation is performed around the center of the image.
    """
    image_center = center(image_shape)
    matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Create a copy of the input keypoints to avoid modifying the original array
    rotated_keypoints = keypoints.copy().astype(np.float32)

    # Extract x and y coordinates
    xy = rotated_keypoints[:, :2]

    # Rotate x and y coordinates
    xy_rotated = cv2.transform(xy.reshape(-1, 1, 2), matrix).squeeze()

    # Update x and y coordinates
    rotated_keypoints[:, :2] = xy_rotated

    # Update angles
    rotated_keypoints[:, 2] += np.radians(angle)

    return rotated_keypoints


@preserve_channel_dim
def resize(img: np.ndarray, target_shape: tuple[int, int], interpolation: int) -> np.ndarray:
    if target_shape == img.shape[:2]:
        return img

    height, width = target_shape
    resize_fn = maybe_process_in_chunks(cv2.resize, dsize=(width, height), interpolation=interpolation)
    return resize_fn(img)


@preserve_channel_dim
def scale(img: np.ndarray, scale: float, interpolation: int) -> np.ndarray:
    height, width = img.shape[:2]
    new_size = int(height * scale), int(width * scale)
    return resize(img, new_size, interpolation)


@handle_empty_array
def keypoints_scale(keypoints: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    """Scales keypoints by scale_x and scale_y.

    Args:
        keypoints: A numpy array of keypoints with shape (N, 4+) in the format (x, y, angle, scale, ...).
        scale_x: Scale coefficient x-axis.
        scale_y: Scale coefficient y-axis.

    Returns:
        A numpy array of scaled keypoints with the same shape as input.
    """
    # Extract x, y, angle, and scale
    x, y, angle, scale = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], keypoints[:, 3]

    # Scale x and y
    x_scaled = x * scale_x
    y_scaled = y * scale_y

    # Scale the keypoint scale by the maximum of scale_x and scale_y
    scale_scaled = scale * max(scale_x, scale_y)

    # Create the output array
    scaled_keypoints = np.column_stack([x_scaled, y_scaled, angle, scale_scaled])

    # If there are additional columns, preserve them
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        scaled_keypoints = np.column_stack(
            [scaled_keypoints, keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:]],
        )

    return scaled_keypoints


def _func_max_size(img: np.ndarray, max_size: int, interpolation: int, func: Callable[..., Any]) -> np.ndarray:
    image_shape = img.shape[:2]

    scale = max_size / float(func(image_shape))

    if scale != 1.0:
        new_height, new_width = tuple(round(dim * scale) for dim in image_shape)
        return resize(img, (new_height, new_width), interpolation=interpolation)
    return img


@preserve_channel_dim
def longest_max_size(img: np.ndarray, max_size: int, interpolation: int) -> np.ndarray:
    return _func_max_size(img, max_size, interpolation, max)


@preserve_channel_dim
def smallest_max_size(img: np.ndarray, max_size: int, interpolation: int) -> np.ndarray:
    return _func_max_size(img, max_size, interpolation, min)


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
    image_shape = img.shape[:2]
    perspective_func = maybe_process_in_chunks(
        cv2.warpPerspective,
        M=matrix,
        dsize=(max_width, max_height),
        borderMode=border_mode,
        borderValue=border_val,
        flags=interpolation,
    )
    warped = perspective_func(img)

    if keep_size:
        return resize(warped, image_shape, interpolation=interpolation)

    return warped


@handle_empty_array
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

    # Create a copy of the input bboxes to avoid modifying the original array
    transformed_bboxes = bboxes.copy()

    # Denormalize bboxes
    denormalized_coords = denormalize_bboxes(bboxes[:, :4], image_shape)

    # Create points for each bbox
    x_min, y_min, x_max, y_max = denormalized_coords.T
    points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]).transpose(2, 0, 1)
    # Shape is: (num_bboxes, 4, 2)

    # Reshape points to (num_bboxes * 4, 2)
    points_reshaped = points.reshape(-1, 2)

    # Pad points_reshaped with two columns of zeros
    points_padded = np.pad(points_reshaped, ((0, 0), (0, 2)), mode="constant")

    # Apply perspective transformation to all points at once
    transformed_points = perspective_keypoints(points_padded, image_shape, matrix, max_width, max_height, keep_size)

    # Reshape back to (num_bboxes, 4, 2)
    transformed_points = transformed_points[:, :2].reshape(-1, 4, 2)
    # Get new bounding boxes
    new_coords = np.array(
        [[np.min(box[:, 0]), np.min(box[:, 1]), np.max(box[:, 0]), np.max(box[:, 1])] for box in transformed_points],
    )

    # Normalize the new bounding boxes
    output_shape = (height if keep_size else max_height, width if keep_size else max_width)
    normalized_coords = normalize_bboxes(new_coords, output_shape)

    # Update only the first 4 columns of the bboxes array
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


@handle_empty_array
@angle_2pi_range
def perspective_keypoints(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> np.ndarray:
    keypoints = keypoints.copy().astype(np.float32)
    x, y, angle, scale = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], keypoints[:, 3]

    # Reshape keypoints for perspective transform
    keypoint_vector = np.column_stack((x, y)).astype(np.float32).reshape(-1, 1, 2)

    # Apply perspective transform
    transformed_points = cv2.perspectiveTransform(keypoint_vector, matrix).squeeze()
    x, y = transformed_points[:, 0], transformed_points[:, 1]

    # Update angles
    angle += rotation2d_matrix_to_euler_angles(matrix[:2, :2], y_up=True)

    # Calculate scale factors
    scale_x = np.sign(matrix[0, 0]) * np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sign(matrix[1, 1]) * np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    scale *= max(scale_x, scale_y)

    if keep_size:
        width, height = image_shape[:2]
        scale_x = width / max_width
        scale_y = height / max_height
        x *= scale_x
        y *= scale_y
        scale *= max(scale_x, scale_y)

    # Create the output array
    transformed_keypoints = np.column_stack([x, y, angle, scale])

    # If there are additional columns, preserve them
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        transformed_keypoints = np.column_stack(
            [transformed_keypoints, keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:]],
        )

    return transformed_keypoints


def is_identity_matrix(matrix: skimage.transform.ProjectiveTransform) -> bool:
    return np.allclose(matrix.params, np.eye(3, dtype=np.float32))


def warp_affine_with_value_extension(
    image: np.ndarray,
    matrix: np.ndarray,
    dsize: Sequence[int],
    flags: int,
    border_mode: int,
    border_value: ColorType,
) -> np.ndarray:
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
    matrix: skimage.transform.ProjectiveTransform,
    interpolation: int,
    cval: ColorType,
    mode: int,
    output_shape: Sequence[int],
) -> np.ndarray:
    if is_identity_matrix(matrix):
        return image

    height = int(np.round(output_shape[0]))
    width = int(np.round(output_shape[1]))

    dsize = (width, height)

    warp_fn = maybe_process_in_chunks(
        warp_affine_with_value_extension,
        matrix=matrix.params[:2],
        dsize=dsize,
        flags=interpolation,
        border_mode=mode,
        border_value=cval,
    )
    return warp_fn(image)


@handle_empty_array
@angle_2pi_range
def keypoints_affine(
    keypoints: np.ndarray,
    matrix: skimage.transform.ProjectiveTransform,
    image_shape: tuple[int, int],
    scale: dict[str, Any],
    mode: int,
) -> np.ndarray:
    """Apply an affine transformation to keypoints.

    This function transforms keypoints using the given affine transformation matrix.
    It handles reflection padding if necessary, updates coordinates, angles, and scales.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 4+) where N is the number of keypoints.
                                Each keypoint is represented as [x, y, angle, scale, ...].
        matrix (skimage.transform.ProjectiveTransform): The affine transformation matrix.
        image_shape (tuple[int, int]): Shape of the image (height, width).
        scale (dict[str, Any]): Dictionary containing scale factors for x and y directions.
                                Expected keys are 'x' and 'y'.
        mode (int): Border mode for handling keypoints near image edges.
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
        >>> matrix = skimage.transform.ProjectiveTransform(...)
        >>> scale = {'x': 1.5, 'y': 1.2}
        >>> transformed_keypoints = keypoints_affine(keypoints, matrix, (480, 640), scale, cv2.BORDER_REFLECT_101)
    """
    keypoints = keypoints.copy().astype(np.float32)

    if is_identity_matrix(matrix):
        return keypoints

    if mode in REFLECT_BORDER_MODES:
        # Step 1: Compute affine transform padding
        pad_left, pad_right, pad_top, pad_bottom = calculate_affine_transform_padding(matrix, image_shape)
        grid_dimensions = get_pad_grid_dimensions(pad_top, pad_bottom, pad_left, pad_right, image_shape)
        keypoints = generate_reflected_keypoints(keypoints, grid_dimensions, image_shape, center_in_origin=True)

    # Extract x, y coordinates
    xy = keypoints[:, :2]

    # Transform x, y coordinates
    xy_transformed = cv2.transform(xy.reshape(-1, 1, 2), matrix.params[:2]).squeeze()

    # Calculate angle adjustment
    angle_adjustment = rotation2d_matrix_to_euler_angles(matrix.params[:2], y_up=False)

    # Update angles
    keypoints[:, 2] = keypoints[:, 2] + angle_adjustment

    # Update scales
    max_scale = max(scale["x"], scale["y"])

    keypoints[:, 3] *= max_scale

    # Update x, y coordinates
    keypoints[:, :2] = xy_transformed

    return keypoints


def calculate_affine_transform_padding(
    matrix: skimage.transform.ProjectiveTransform,
    image_shape: Sequence[int],
) -> tuple[int, int, int, int]:
    """Calculate the necessary padding for an affine transformation to avoid empty spaces."""
    height, width = image_shape[:2]

    # Check for identity transform
    if is_identity_matrix(matrix):
        return (0, 0, 0, 0)

    # Original corners
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    # Transform corners
    transformed_corners = matrix(corners)

    # Find box that includes both original and transformed corners
    all_corners = np.vstack((corners, transformed_corners))
    min_x, min_y = all_corners.min(axis=0)
    max_x, max_y = all_corners.max(axis=0)
    # Compute the inverse transform
    inverse_matrix = matrix.inverse

    # Apply inverse transform to all corners of the bounding box
    bbox_corners = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

    inverse_corners = inverse_matrix(bbox_corners)

    min_x, min_y = inverse_corners.min(axis=0)
    max_x, max_y = inverse_corners.max(axis=0)

    pad_left = max(0, math.ceil(0 - min_x))
    pad_right = max(0, math.ceil(max_x - width))
    pad_top = max(0, math.ceil(0 - min_y))
    pad_bottom = max(0, math.ceil(max_y - height))

    return pad_left, pad_right, pad_top, pad_bottom


@handle_empty_array
def bboxes_affine_largest_box(bboxes: np.ndarray, matrix: skimage.transform.ProjectiveTransform) -> np.ndarray:
    """Apply an affine transformation to bounding boxes and return the largest enclosing boxes.

    This function transforms each corner of every bounding box using the given affine transformation
    matrix, then computes the new bounding boxes that fully enclose the transformed corners.

    Args:
        bboxes (np.ndarray): An array of bounding boxes with shape (N, 4+) where N is the number of
                             bounding boxes. Each row should contain [x_min, y_min, x_max, y_max]
                             followed by any additional attributes (e.g., class labels).
        matrix (skimage.transform.ProjectiveTransform): The affine transformation matrix to apply.

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
        >>> matrix = skimage.transform.AffineTransform(scale=(2, 2), translation=(5, 5))
        >>> transformed_bboxes = bboxes_affine_largest_box(bboxes, matrix)
        >>> print(transformed_bboxes)
        [[ 25.  25.  45.  45.   1.]
         [ 65.  65.  85.  85.   2.]]
    """
    # Extract corners of all bboxes
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]).transpose(
        2,
        0,
        1,
    )  # Shape: (num_bboxes, 4, 2)

    # Transform all corners at once
    transformed_corners = skimage.transform.matrix_transform(corners.reshape(-1, 2), matrix.params)
    transformed_corners = transformed_corners.reshape(-1, 4, 2)

    # Compute new bounding boxes
    new_x_min = np.min(transformed_corners[:, :, 0], axis=1)
    new_x_max = np.max(transformed_corners[:, :, 0], axis=1)
    new_y_min = np.min(transformed_corners[:, :, 1], axis=1)
    new_y_max = np.max(transformed_corners[:, :, 1], axis=1)

    return np.column_stack([new_x_min, new_y_min, new_x_max, new_y_max, bboxes[:, 4:]])


@handle_empty_array
def bboxes_affine_ellipse(bboxes: np.ndarray, matrix: skimage.transform.ProjectiveTransform) -> np.ndarray:
    """Apply an affine transformation to bounding boxes using an ellipse approximation method.

    This function transforms bounding boxes by approximating each box with an ellipse,
    transforming points along the ellipse's circumference, and then computing the
    new bounding box that encloses the transformed ellipse.

    Args:
        bboxes (np.ndarray): An array of bounding boxes with shape (N, 4+) where N is the number of
                             bounding boxes. Each row should contain [x_min, y_min, x_max, y_max]
                             followed by any additional attributes (e.g., class labels).
        matrix (skimage.transform.ProjectiveTransform): The affine transformation matrix to apply.

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

    Example:
        >>> bboxes = np.array([[10, 10, 30, 20, 1], [40, 40, 60, 60, 2]])  # Two boxes with class labels
        >>> matrix = skimage.transform.AffineTransform(rotation=np.pi/4)  # 45-degree rotation
        >>> transformed_bboxes = bboxes_affine_ellipse(bboxes, matrix)
        >>> print(transformed_bboxes)
        [[ 5.86  5.86 34.14 24.14  1.  ]
         [30.   30.   70.   70.    2.  ]]
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

    # Transform all points at once
    transformed_points = skimage.transform.matrix_transform(points, matrix.params)
    transformed_points = transformed_points.reshape(len(bboxes), -1, 2)

    # Compute new bounding boxes
    new_x_min = np.min(transformed_points[:, :, 0], axis=1)
    new_x_max = np.max(transformed_points[:, :, 0], axis=1)
    new_y_min = np.min(transformed_points[:, :, 1], axis=1)
    new_y_max = np.max(transformed_points[:, :, 1], axis=1)

    return np.column_stack([new_x_min, new_y_min, new_x_max, new_y_max, bboxes[:, 4:]])


@handle_empty_array
def bboxes_affine(
    bboxes: np.ndarray,
    matrix: skimage.transform.ProjectiveTransform,
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
        matrix (skimage.transform.ProjectiveTransform): Affine transformation matrix
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
        pad_left, pad_right, pad_top, pad_bottom = calculate_affine_transform_padding(matrix, image_shape)
        grid_dimensions = get_pad_grid_dimensions(pad_top, pad_bottom, pad_left, pad_right, image_shape)
        bboxes = generate_reflected_bboxes(bboxes, grid_dimensions, image_shape, center_in_origin=True)

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


@clipped
def piecewise_affine(
    img: np.ndarray,
    matrix: skimage.transform.PiecewiseAffineTransform | None,
    interpolation: int,
    mode: str,
    cval: float,
) -> np.ndarray:
    if matrix is None:
        return img
    return skimage.transform.warp(
        img,
        matrix,
        order=interpolation,
        mode=mode,
        cval=cval,
        preserve_range=True,
        output_shape=img.shape,
    )


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
        keypoints: A numpy array of shape (N, 2+) where N is the number of keypoints.
                   Each row represents a keypoint's (x, y) coordinates.
        image_shape: tuple[int, int] shape of the image (height, width)
        inverted (bool): If ``True``, inverted distance maps are returned where each
            distance value d is replaced by ``d/(d+1)``, i.e. the distance
            maps have values in the range ``(0.0, 1.0]`` with ``1.0`` denoting
            exactly the position of the respective keypoint.

    Returns:
        np.ndarray: A ``float32`` array of shape (H, W, N) containing ``N`` distance maps for ``N``
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

    drop_if_not_found, if_not_found_x, if_not_found_y = validate_if_not_found_coords(if_not_found_coords)

    # Find the indices of max/min values for all keypoints at once
    if inverted:
        hitidx_flat = np.argmax(distance_maps.reshape(height * width, nb_keypoints), axis=0)
    else:
        hitidx_flat = np.argmin(distance_maps.reshape(height * width, nb_keypoints), axis=0)

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


@handle_empty_array
def keypoints_piecewise_affine(
    keypoints: np.ndarray,
    matrix: skimage.transform.PiecewiseAffineTransform | None,
    image_shape: tuple[int, int],
    keypoints_threshold: float,
) -> np.ndarray:
    if matrix is None:
        return keypoints

    a, s = keypoints[:, 2], keypoints[:, 3]

    # Create distance maps for all keypoints
    dist_maps = to_distance_maps(keypoints[:, :2], image_shape, True)

    # Apply piecewise affine transformation to all distance maps
    dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)

    # Convert distance maps back to keypoints
    transformed_xy = from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold)

    # Combine transformed x, y with original a, s
    transformed_keypoints = np.column_stack([transformed_xy, a, s])

    # If there are additional columns, preserve them
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        return np.column_stack(
            [transformed_keypoints, keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:]],
        )

    return transformed_keypoints


@handle_empty_array
def bboxes_piecewise_affine(
    bboxes: np.ndarray,
    matrix: skimage.transform.PiecewiseAffineTransform,
    image_shape: tuple[int, int],
    keypoints_threshold: float,
) -> np.ndarray:
    if matrix is None:
        return bboxes

    height, width = image_shape[:2]

    # Denormalize bboxes
    denorm_bboxes = denormalize_bboxes(bboxes, image_shape)

    # Create keypoints for all bboxes
    keypoints = np.array(
        [
            denorm_bboxes[:, [0, 1]],  # x_min, y_min
            denorm_bboxes[:, [2, 1]],  # x_max, y_min
            denorm_bboxes[:, [2, 3]],  # x_max, y_max
            denorm_bboxes[:, [0, 3]],  # x_min, y_max
        ],
    )
    keypoints = keypoints.transpose(1, 0, 2).reshape(-1, 2)

    # Create distance maps for all keypoints
    dist_maps = to_distance_maps(keypoints, image_shape, True)

    # Apply piecewise affine transformation to all distance maps
    dist_maps = piecewise_affine(dist_maps, matrix, 0, "constant", 0)

    # Convert distance maps back to keypoints
    transformed_keypoints = from_distance_maps(dist_maps, True, {"x": -1, "y": -1}, keypoints_threshold)

    # Reshape transformed keypoints back to (N, 4, 2) where N is the number of bboxes
    transformed_keypoints = transformed_keypoints.reshape(-1, 4, 2)

    # Filter out keypoints outside the image
    mask = (
        (transformed_keypoints[:, :, 0] >= 0)
        & (transformed_keypoints[:, :, 0] < width)
        & (transformed_keypoints[:, :, 1] >= 0)
        & (transformed_keypoints[:, :, 1] < height)
    )

    # Compute new bboxes
    new_bboxes = np.zeros_like(bboxes)
    for i in range(len(bboxes)):
        valid_points = transformed_keypoints[i][mask[i]]
        if len(valid_points) > 0:
            new_bboxes[i, 0] = valid_points[:, 0].min()
            new_bboxes[i, 1] = valid_points[:, 1].min()
            new_bboxes[i, 2] = valid_points[:, 0].max()
            new_bboxes[i, 3] = valid_points[:, 1].max()
        else:
            new_bboxes[i] = bboxes[i]  # Keep original bbox if all points are outside

    # Normalize the new bboxes
    return normalize_bboxes(new_bboxes, image_shape)


def vflip(img: np.ndarray) -> np.ndarray:
    return img[::-1, ...]


def hflip(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1, ...]


def hflip_cv2(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def d4(img: np.ndarray, group_member: D4Type) -> np.ndarray:
    """Applies a `D_4` symmetry group transformation to an image array.

    This function manipulates an image using transformations such as rotations and flips,
    corresponding to the `D_4` dihedral group symmetry operations.
    Each transformation is identified by a unique group member code.

    Parameters:
    - img (np.ndarray): The input image array to transform.
    - group_member (D4Type): A string identifier indicating the specific transformation to apply. Valid codes include:
      - 'e': Identity (no transformation).
      - 'r90': Rotate 90 degrees counterclockwise.
      - 'r180': Rotate 180 degrees.
      - 'r270': Rotate 270 degrees counterclockwise.
      - 'v': Vertical flip.
      - 'hvt': Transpose over second diagonal
      - 'h': Horizontal flip.
      - 't': Transpose (reflect over the main diagonal).

    Returns:
    - np.ndarray: The transformed image array.

    Raises:
    - ValueError: If an invalid group member is specified.

    Examples:
    - Rotating an image by 90 degrees:
      `transformed_image = d4(original_image, 'r90')`
    - Applying a horizontal flip to an image:
      `transformed_image = d4(original_image, 'h')`
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


@preserve_channel_dim
def random_flip(img: np.ndarray, code: int) -> np.ndarray:
    return cv2.flip(img, code)


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


def rot90(img: np.ndarray, factor: int) -> np.ndarray:
    return np.rot90(img, factor)


@handle_empty_array
def bboxes_vflip(bboxes: np.ndarray) -> np.ndarray:
    """Flip bounding boxes vertically around the x-axis.

    Args:
        bboxes: A numpy array of bounding boxes with shape (num_bboxes, 4+).
                Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).

    Returns:
        np.ndarray: A numpy array of vertically flipped bounding boxes with the same shape as input.
    """
    flipped_bboxes = bboxes.copy()
    flipped_bboxes[:, 1] = 1 - bboxes[:, 3]  # new y_min = 1 - y_max
    flipped_bboxes[:, 3] = 1 - bboxes[:, 1]  # new y_max = 1 - y_min

    return flipped_bboxes


@handle_empty_array
def bboxes_hflip(bboxes: np.ndarray) -> np.ndarray:
    """Flip bounding boxes horizontally around the y-axis.

    Args:
        bboxes: A numpy array of bounding boxes with shape (num_bboxes, 4+).
                Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).

    Returns:
        np.ndarray: A numpy array of horizontally flipped bounding boxes with the same shape as input.
    """
    flipped_bboxes = bboxes.copy()
    flipped_bboxes[:, 0] = 1 - bboxes[:, 2]  # new x_min = 1 - x_max
    flipped_bboxes[:, 2] = 1 - bboxes[:, 0]  # new x_max = 1 - x_min

    return flipped_bboxes


@handle_empty_array
def bboxes_flip(bboxes: np.ndarray, d: int) -> np.ndarray:
    """Flip a bounding box either vertically, horizontally or both depending on the value of `d`.

    Args:
        bboxes: A numpy array of bounding boxes with shape (num_bboxes, 4+).
                Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).
        d: dimension. 0 for vertical flip, 1 for horizontal, -1 for transpose

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        return bboxes_vflip(bboxes)
    if d == 1:
        return bboxes_hflip(bboxes)
    if d == -1:
        bboxes = bboxes_hflip(bboxes)
        return bboxes_vflip(bboxes)

    raise ValueError(f"Invalid d value {d}. Valid values are -1, 0 and 1")


@handle_empty_array
def bboxes_transpose(bboxes: np.ndarray) -> np.ndarray:
    """Transpose bounding boxes by swapping x and y coordinates.

    Args:
        bboxes: A numpy array of bounding boxes with shape (num_bboxes, 4+).
                Each row represents a bounding box (x_min, y_min, x_max, y_max, ...).

    Returns:
        np.ndarray: A numpy array of transposed bounding boxes with the same shape as input.
    """
    transposed_bboxes = bboxes.copy()
    transposed_bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]

    return transposed_bboxes


@handle_empty_array
@angle_2pi_range
def keypoints_vflip(keypoints: np.ndarray, rows: int) -> np.ndarray:
    """Flip keypoints vertically around the x-axis.

    Args:
        keypoints: A numpy array of shape (N, 4+) where each row represents a keypoint (x, y, angle, scale, ...).
        rows: Image height.

    Returns:
        np.ndarray: An array of flipped keypoints with the same shape as the input.
    """
    flipped_keypoints = keypoints.copy().astype(np.float32)

    # Flip y-coordinates
    flipped_keypoints[:, 1] = (rows - 1) - keypoints[:, 1]

    # Negate angles
    flipped_keypoints[:, 2] = -keypoints[:, 2]

    return flipped_keypoints


@handle_empty_array
@angle_2pi_range
def keypoints_hflip(keypoints: np.ndarray, cols: int) -> np.ndarray:
    """Flip keypoints horizontally around the y-axis.

    Args:
        keypoints: A numpy array of shape (N, 4+) where each row represents a keypoint (x, y, angle, scale, ...).
        cols: Image width.

    Returns:
        np.ndarray: An array of flipped keypoints with the same shape as the input.
    """
    flipped_keypoints = keypoints.copy().astype(np.float32)

    # Flip x-coordinates
    flipped_keypoints[:, 0] = (cols - 1) - keypoints[:, 0]

    # Adjust angles
    flipped_keypoints[:, 2] = np.pi - keypoints[:, 2]

    return flipped_keypoints


@handle_empty_array
@angle_2pi_range
def keypoints_flip(keypoints: np.ndarray, d: int, image_shape: tuple[int, int]) -> np.ndarray:
    """Flip a keypoint either vertically, horizontally or both depending on the value of `d`.

    Args:
        keypoints: A keypoints `(x, y, angle, scale)`.
        d: Number of flip. Must be -1, 0 or 1:
            * 0 - vertical flip,
            * 1 - horizontal flip,
            * -1 - vertical and horizontal flip.
        image_shape: A tuple of image shape `(height, width, channels)`.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    rows, cols = image_shape[:2]

    if d == 0:
        return keypoints_vflip(keypoints, rows)
    if d == 1:
        return keypoints_hflip(keypoints, cols)
    if d == -1:
        keypoints = keypoints_hflip(keypoints, cols)
        return keypoints_vflip(keypoints, rows)

    raise ValueError(f"Invalid d value {d}. Valid values are -1, 0 and 1")


@handle_empty_array
@angle_2pi_range
def keypoints_transpose(keypoints: np.ndarray) -> np.ndarray:
    """Transposes keypoints along the main diagonal.

    Args:
        keypoints: A numpy array of shape (N, 4+) where each row represents a keypoint (x, y, angle, scale, ...).

    Returns:
        np.ndarray: An array of transposed keypoints with the same shape as the input.
    """
    transposed_keypoints = keypoints.copy()

    # Swap x and y coordinates
    transposed_keypoints[:, [0, 1]] = keypoints[:, [1, 0]]

    # Adjust angles to reflect the coordinate swap
    angles = keypoints[:, 2]
    transposed_keypoints[:, 2] = np.where(angles <= np.pi, np.pi / 2 - angles, 3 * np.pi / 2 - angles)

    return transposed_keypoints


@preserve_channel_dim
def pad(
    img: np.ndarray,
    min_height: int,
    min_width: int,
    border_mode: int,
    value: ColorType | None,
) -> np.ndarray:
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

    img = pad_with_params(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value)

    if img.shape[:2] != (max(min_height, height), max(min_width, width)):
        raise RuntimeError(
            f"Invalid result shape. Got: {img.shape[:2]}. Expected: {(max(min_height, height), max(min_width, width))}",
        )

    return img


def extend_value(value: ColorType, num_channels: int) -> Sequence[ScalarType]:
    return [value] * num_channels if isinstance(value, (int, float)) else value


def copy_make_border_with_value_extension(
    img: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    border_mode: int,
    value: ColorType,
) -> np.ndarray:
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
    value: ColorType | None,
) -> np.ndarray:
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


@preserve_channel_dim
def optical_distortion(
    img: np.ndarray,
    k: int,
    dx: int,
    dy: int,
    interpolation: int,
    border_mode: int,
    value: ColorType | None = None,
) -> np.ndarray:
    """Barrel / pincushion distortion. Unconventional augment.

    Reference:
        |  https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
        |  https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
        |  https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
        |  http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/
    """
    height, width = img.shape[:2]

    fx = width
    fy = height

    cx = width * 0.5 + dx
    cy = height * 0.5 + dy

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)
    return cv2.remap(img, map1, map2, interpolation=interpolation, borderMode=border_mode, borderValue=value)


@preserve_channel_dim
def grid_distortion(
    img: np.ndarray,
    num_steps: int,
    xsteps: tuple[()],
    ysteps: tuple[()],
    interpolation: int,
    border_mode: int,
    value: ColorType | None = None,
) -> np.ndarray:
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        x = idx * x_step
        start = int(x)
        end = int(x) + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        y = idx * y_step
        start = int(y)
        end = int(y) + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    remap_fn = maybe_process_in_chunks(
        cv2.remap,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return remap_fn(img)


def elastic_transform_helper(
    img: np.ndarray,
    alpha: float,
    sigma: float,
    interpolation: int,
    border_mode: int,
    value: ColorType | None,
    random_state: np.random.RandomState | None,
    same_dxdy: bool,
    kernel_size: tuple[int, int],
) -> np.ndarray:
    height, width = img.shape[:2]

    dx = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
    cv2.GaussianBlur(dx, kernel_size, sigma, dst=dx)
    dx *= alpha

    dy = dx if same_dxdy else random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
    if not same_dxdy:
        cv2.GaussianBlur(dy, kernel_size, sigma, dst=dy)
        dy *= alpha

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    remap_fn = maybe_process_in_chunks(
        cv2.remap,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )
    return remap_fn(img)


def elastic_transform_precise(
    img: np.ndarray,
    alpha: float,
    sigma: float,
    interpolation: int,
    border_mode: int,
    value: ColorType | None,
    random_state: np.random.RandomState | None,
    same_dxdy: bool = False,
) -> np.ndarray:
    """Apply a precise elastic transformation to an image.

    This function applies an elastic deformation to the input image using a precise method.
    The transformation involves creating random displacement fields, smoothing them using Gaussian
    blur with adaptive kernel size, and then remapping the image according to the smoothed displacement fields.

    Args:
        img (np.ndarray): Input image.
        alpha (float): Scaling factor for the random displacement fields.
        sigma (float): Standard deviation for Gaussian blur applied to the displacement fields.
        interpolation (int): Interpolation method to be used (e.g., cv2.INTER_LINEAR).
        border_mode (int): Pixel extrapolation method (e.g., cv2.BORDER_CONSTANT).
        value (ColorType | None): Border value if border_mode is cv2.BORDER_CONSTANT.
        random_state (np.random.RandomState | None): Random state for reproducibility.
        same_dxdy (bool, optional): If True, use the same displacement field for both x and y directions.

    Returns:
        np.ndarray: Transformed image with precise elastic deformation applied.
    """
    return elastic_transform_helper(
        img,
        alpha,
        sigma,
        interpolation,
        border_mode,
        value,
        random_state,
        same_dxdy,
        kernel_size=(0, 0),
    )


def elastic_transform_approximate(
    img: np.ndarray,
    alpha: float,
    sigma: float,
    interpolation: int,
    border_mode: int,
    value: ColorType | None,
    random_state: np.random.RandomState | None,
    same_dxdy: bool = False,
) -> np.ndarray:
    """Apply an approximate elastic transformation to an image."""
    return elastic_transform_helper(
        img,
        alpha,
        sigma,
        interpolation,
        border_mode,
        value,
        random_state,
        same_dxdy,
        kernel_size=(17, 17),
    )


@preserve_channel_dim
def elastic_transform(
    img: np.ndarray,
    alpha: float,
    sigma: float,
    interpolation: int,
    border_mode: int,
    value: ColorType | None = None,
    random_state: np.random.RandomState | None = None,
    approximate: bool = False,
    same_dxdy: bool = False,
) -> np.ndarray:
    """Apply an elastic transformation to an image."""
    if approximate:
        return elastic_transform_approximate(
            img,
            alpha,
            sigma,
            interpolation,
            border_mode,
            value,
            random_state,
            same_dxdy,
        )
    return elastic_transform_precise(
        img,
        alpha,
        sigma,
        interpolation,
        border_mode,
        value,
        random_state,
        same_dxdy,
    )


@handle_empty_array
def pad_bboxes(
    bboxes: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    border_mode: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    if border_mode not in REFLECT_BORDER_MODES:
        shift_vector = np.array([pad_left, pad_top, pad_left, pad_top])
        return shift_bboxes(bboxes, shift_vector)

    grid_dimensions = get_pad_grid_dimensions(pad_top, pad_bottom, pad_left, pad_right, image_shape)

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

    return {"grid_shape": (grid_rows, grid_cols), "original_position": (original_row, original_col)}


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
    bboxes_hvflipped = flip_bboxes(bboxes, flip_horizontal=True, flip_vertical=True, image_shape=image_shape)

    # Shift all versions to the original position
    shift_vector = np.array([original_col * cols, original_row * rows, original_col * cols, original_row * rows])
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


@handle_empty_array
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
def distort_image(image: np.ndarray, generated_mesh: np.ndarray, interpolation: int) -> np.ndarray:
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
        warped = cv2.warpPerspective(image, perspective_mat, (image.shape[1], image.shape[0]), flags=interpolation)

        # Create mask for the transformed region
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_quad), 255)

        # Copy only the warped quadrilateral area to the output image
        distorted_image = cv2.copyTo(warped, mask, distorted_image)

    return distorted_image


def calculate_grid_dimensions(
    image_shape: tuple[int, int],
    num_grid_xy: tuple[int, int],
) -> np.ndarray:
    """Calculate the dimensions of a grid overlay on an image using vectorized operations.

    This function divides an image into a grid and calculates the dimensions
    (x_min, y_min, x_max, y_max) for each cell in the grid without using loops.

    Args:
        image_shape (tuple[int, int]): The shape of the image (height, width).
        num_grid_xy (tuple[int, int]): The number of grid cells in (x, y) directions.

    Returns:
        np.ndarray: A 3D array of shape (grid_height, grid_width, 4) where each element
                    is [x_min, y_min, x_max, y_max] for a grid cell.

    Example:
        >>> image_shape = (100, 150)
        >>> num_grid_xy = (3, 2)
        >>> dimensions = calculate_grid_dimensions(image_shape, num_grid_xy)
        >>> print(dimensions.shape)
        (2, 3, 4)
        >>> print(dimensions[0, 0])  # First cell
        [  0   0  50  50]
    """
    num_grid_yx = np.array(num_grid_xy[::-1])  # Reverse to match image_shape order
    image_shape = np.array(image_shape)

    square_shape = image_shape // num_grid_yx
    last_square_shape = image_shape - (square_shape * (num_grid_yx - 1))

    grid_width, grid_height = num_grid_xy

    # Create meshgrid for row and column indices
    col_indices, row_indices = np.meshgrid(np.arange(grid_width), np.arange(grid_height))

    # Calculate x_min and y_min
    x_min = col_indices * square_shape[1]
    y_min = row_indices * square_shape[0]

    # Calculate x_max and y_max
    x_max = np.where(col_indices == grid_width - 1, x_min + last_square_shape[1], x_min + square_shape[1])
    y_max = np.where(row_indices == grid_height - 1, y_min + last_square_shape[0], y_min + square_shape[0])

    # Stack the dimensions
    return np.stack([x_min, y_min, x_max, y_max], axis=-1).astype(np.int16)


def generate_distorted_grid_polygons(
    dimensions: np.ndarray,
    magnitude: int,
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
    displacements = random_utils.randint(
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


@handle_empty_array
def pad_keypoints(
    keypoints: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    border_mode: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    if border_mode not in {cv2.BORDER_REFLECT_101, cv2.BORDER_REFLECT101}:
        shift_vector = np.array([pad_left, pad_top])  # Only shift x and y
        return shift_keypoints(keypoints, shift_vector)

    grid_dimensions = get_pad_grid_dimensions(pad_top, pad_bottom, pad_left, pad_right, image_shape)

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


def validate_keypoints(keypoints: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
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
    shifted_keypoints = keypoints.copy()
    shifted_keypoints[:, :2] += shift_vector[:2]  # Only shift x and y
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
    keypoints_hflipped = flip_keypoints(keypoints, flip_horizontal=True, image_shape=image_shape)
    keypoints_vflipped = flip_keypoints(keypoints, flip_vertical=True, image_shape=image_shape)
    keypoints_hvflipped = flip_keypoints(keypoints, flip_horizontal=True, flip_vertical=True, image_shape=image_shape)

    rows, cols = image_shape[:2]

    # Shift all versions to the original position
    shift_vector = np.array([original_col * cols, original_row * rows, 0, 0])  # Only shift x and y
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
            cell_shift = np.array([(grid_col - original_col) * cols, (grid_row - original_row) * rows, 0, 0])
            shifted_keypoints = shift_keypoints(current_keypoints, cell_shift)

            new_keypoints.append(shifted_keypoints)

    result = np.vstack(new_keypoints)

    return shift_keypoints(result, -shift_vector) if center_in_origin else result


@handle_empty_array
def flip_keypoints(
    keypoints: np.ndarray,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    image_shape: tuple[int, int] = (0, 0),
) -> np.ndarray:
    rows, cols = image_shape[:2]
    flipped_keypoints = keypoints.copy()
    if flip_horizontal:
        flipped_keypoints[:, 0] = cols - flipped_keypoints[:, 0]
        flipped_keypoints[:, 2] = -flipped_keypoints[:, 2]  # Flip angle
    if flip_vertical:
        flipped_keypoints[:, 1] = rows - flipped_keypoints[:, 1]
        flipped_keypoints[:, 2] = -flipped_keypoints[:, 2]  # Flip angle
    return flipped_keypoints


class TranslateDict(TypedDict):
    x: float
    y: float


class ShearDict(TypedDict):
    x: float
    y: float


class ScaleDict(TypedDict):
    x: float
    y: float


def create_affine_transformation_matrix(
    translate: TranslateDict,
    shear: ShearDict,
    scale: ScaleDict,
    rotate: float,
    shift: tuple[float, float],
) -> skimage.transform.ProjectiveTransform:
    """Create an affine transformation matrix combining translation, shear, scale, and rotation.

    This function creates a complex affine transformation by combining multiple transformations
    in a specific order. The transformations are applied as follows:
    1. Shift to top-left: Moves the center of transformation to (0, 0)
    2. Apply main transformations: scale, rotation, shear, and translation
    3. Shift back to center: Moves the center of transformation back to its original position

    The order of these transformations is crucial as matrix multiplications are not commutative.

    Args:
        translate (TranslateDict): Translation in x and y directions.
                                   Keys: 'x', 'y'. Values: translation amounts in pixels.
        shear (ShearDict): Shear in x and y directions.
                           Keys: 'x', 'y'. Values: shear angles in degrees.
        scale (ScaleDict): Scale factors for x and y directions.
                           Keys: 'x', 'y'. Values: scale factors (1.0 means no scaling).
        rotate (float): Rotation angle in degrees. Positive values rotate counter-clockwise.
        shift (tuple[float, float]): Shift to apply before and after transformations.
                                     Typically the image center (width/2, height/2).

    Returns:
        skimage.transform.ProjectiveTransform: The resulting affine transformation matrix.

    Note:
        - All angle inputs (rotate, shear) are in degrees and are converted to radians internally.
        - The order of transformations in the AffineTransform is: scale, rotation, shear, translation.
        - The resulting transformation can be applied to coordinates using the __call__ method.
    """
    # Step 1: Create matrix to shift to top-left
    # This moves the center of transformation to (0, 0)
    matrix_to_topleft = skimage.transform.SimilarityTransform(translation=[shift[0], shift[1]])

    # Step 2: Create matrix for main transformations
    # This includes scaling, translation, rotation, and x-shear
    matrix_transforms = skimage.transform.AffineTransform(
        scale=(scale["x"], scale["y"]),
        rotation=np.deg2rad(rotate),
        shear=(np.deg2rad(shear["x"]), np.deg2rad(shear["y"])),  # Both x and y shear
        translation=(translate["x"], translate["y"]),
    )

    # Step 3: Create matrix to shift back to center
    # This is the inverse of the top-left shift
    matrix_to_center = matrix_to_topleft.inverse

    # Combine all transformations
    # The order is important: transformations are applied from right to left
    return (
        matrix_to_center  # 3. Shift back to original center
        + matrix_transforms  # 2. Apply main transformations
        + matrix_to_topleft  # 1. Shift to top-left
    )


def compute_transformed_image_bounds(
    matrix: skimage.transform.ProjectiveTransform,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the bounds of an image after applying an affine transformation.

    Args:
        matrix (skimage.transform.ProjectiveTransform): The affine transformation matrix.
        image_shape (tuple[int, int]): The shape of the image as (height, width).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - min_coords: An array with the minimum x and y coordinates.
            - max_coords: An array with the maximum x and y coordinates.
    """
    height, width = image_shape[:2]

    # Define the corners of the image
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    # Transform the corners
    transformed_corners = matrix(corners)

    # Calculate the bounding box of the transformed corners
    min_coords = np.floor(transformed_corners.min(axis=0)).astype(int)
    max_coords = np.ceil(transformed_corners.max(axis=0)).astype(int)

    return min_coords, max_coords


def compute_affine_warp_output_shape(
    matrix: skimage.transform.ProjectiveTransform,
    input_shape: tuple[int, ...],
) -> tuple[skimage.transform.ProjectiveTransform, tuple[int, int]]:
    height, width = input_shape[:2]

    if height == 0 or width == 0:
        return matrix, cast(Tuple[int, int], input_shape[:2])

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
    translation = -minc, -minr
    matrix_to_fit = skimage.transform.SimilarityTransform(translation=translation)
    matrix += matrix_to_fit
    return matrix, cast(Tuple[int, int], output_shape_tuple)


@handle_empty_array
def bboxes_optical_distortion(
    bboxes: np.ndarray,
    k: float,
    dx: int,
    dy: int,
    border_mode: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape[:2]

    # Denormalize bboxes
    bboxes_denorm = denormalize_bboxes(bboxes[:, :4], image_shape)

    # Create masks for each bbox
    masks = np.zeros((len(bboxes), height, width), dtype=np.uint8)
    for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes_denorm.astype(int)):
        masks[i, y_min:y_max, x_min:x_max] = 1

    # Apply optical distortion to all masks
    distorted_masks = np.array(
        [optical_distortion(mask, k, dx, dy, cv2.INTER_NEAREST, border_mode, -1) for mask in masks],
    )

    # Get bboxes from distorted masks
    distorted_bboxes = np.array([bbox_from_mask(mask) for mask in distorted_masks])

    # Normalize the distorted bboxes
    normalized_bboxes = normalize_bboxes(distorted_bboxes, image_shape)

    # Update the first 4 columns of the input array
    bboxes[:, :4] = normalized_bboxes

    return bboxes


@handle_empty_array
def bbox_elastic_transform(
    bboxes: np.ndarray,
    alpha: float,
    sigma: float,
    interpolation: int,
    border_mode: int,
    approximate: bool,
    same_dxdy: bool,
    random_seed: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    bboxes = bboxes.copy()
    bboxes_denorm = denormalize_bboxes(bboxes, image_shape)
    # Create a mask for each bbox
    masks = np.zeros((len(bboxes), *image_shape), dtype=np.uint8)
    for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes_denorm[:, :4].astype(int)):
        masks[i, y_min:y_max, x_min:x_max] = 1

    # Apply elastic transform to all masks
    rng = np.random.RandomState(random_seed)
    transformed_masks = np.stack(
        [
            elastic_transform(mask, alpha, sigma, interpolation, border_mode, -1, rng, approximate, same_dxdy)
            for mask in masks
        ],
    )

    # Get bboxes from transformed masks
    bboxes_returned = np.array([bbox_from_mask(mask) for mask in transformed_masks])

    # Normalize the returned bboxes
    bboxes[:, :4] = normalize_bboxes(bboxes_returned, image_shape)

    return bboxes


@handle_empty_array
def bboxes_grid_distortion(
    bboxes: np.ndarray,
    stepsx: tuple[float, ...],
    stepsy: tuple[float, ...],
    num_steps: int,
    border_mode: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    bboxes_denorm = denormalize_bboxes(bboxes, image_shape)

    # Create a mask for each bbox
    masks = np.zeros((len(bboxes), *image_shape[:2]), dtype=np.uint8)

    for i, bbox in enumerate(bboxes_denorm):
        x_min, y_min, x_max, y_max = bbox[:4].astype(int)
        masks[i, y_min:y_max, x_min:x_max] = 1

    # Apply grid distortion to all masks
    transformed_masks = np.stack(
        [
            grid_distortion(
                mask,
                num_steps,
                stepsx,
                stepsy,
                cv2.INTER_NEAREST,
                border_mode,
                -1,
            )
            for mask in masks
        ],
    )

    # Get bboxes from transformed masks
    bboxes_returned = np.array([bbox_from_mask(mask) for mask in transformed_masks])

    # Normalize the returned bboxes
    return normalize_bboxes(bboxes_returned, image_shape)
