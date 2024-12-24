import random
from typing import Literal

import numpy as np

from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.types import NUM_VOLUME_DIMENSIONS, ColorType


def adjust_padding_by_position3d(
    paddings: list[tuple[int, int]],  # [(front, back), (top, bottom), (left, right)]
    position: Literal["center", "random"],
    py_random: random.Random,
) -> tuple[int, int, int, int, int, int]:
    """Adjust padding values based on desired position for 3D data.

    Args:
        paddings: List of tuples containing padding pairs for each dimension [(d_pad), (h_pad), (w_pad)]
        position: Position of the image after padding. Either 'center' or 'random'
        py_random: Random number generator

    Returns:
        tuple[int, int, int, int, int, int]: Final padding values (d_front, d_back, h_top, h_bottom, w_left, w_right)
    """
    if position == "center":
        return (
            paddings[0][0],  # d_front
            paddings[0][1],  # d_back
            paddings[1][0],  # h_top
            paddings[1][1],  # h_bottom
            paddings[2][0],  # w_left
            paddings[2][1],  # w_right
        )

    # For random position, redistribute padding for each dimension
    d_pad = sum(paddings[0])
    h_pad = sum(paddings[1])
    w_pad = sum(paddings[2])

    return (
        py_random.randint(0, d_pad),  # d_front
        d_pad - py_random.randint(0, d_pad),  # d_back
        py_random.randint(0, h_pad),  # h_top
        h_pad - py_random.randint(0, h_pad),  # h_bottom
        py_random.randint(0, w_pad),  # w_left
        w_pad - py_random.randint(0, w_pad),  # w_right
    )


def pad_3d_with_params(
    volume: np.ndarray,
    padding: tuple[int, int, int, int, int, int],
    value: ColorType,
) -> np.ndarray:
    """Pad 3D volume with given parameters.

    Args:
        volume: Input volume with shape (depth, height, width) or (depth, height, width, channels)
        padding: Padding values in format:
            (depth_front, depth_back, height_top, height_bottom, width_left, width_right)
            where:
            - depth_front/back: padding at start/end of depth axis (z)
            - height_top/bottom: padding at start/end of height axis (y)
            - width_left/right: padding at start/end of width axis (x)
        value: Value to fill the padding

    Returns:
        Padded volume with same number of dimensions as input

    Note:
        The padding order matches the volume dimensions (depth, height, width).
        For each dimension, the first value is padding at the start (smaller indices),
        and the second value is padding at the end (larger indices).
    """
    depth_front, depth_back, height_top, height_bottom, width_left, width_right = padding

    # Skip if no padding is needed
    if all(p == 0 for p in padding):
        return volume

    # Handle both 3D and 4D arrays
    pad_width = [
        (depth_front, depth_back),  # depth (z) padding
        (height_top, height_bottom),  # height (y) padding
        (width_left, width_right),  # width (x) padding
    ]

    # Add channel padding if 4D array
    if volume.ndim == NUM_VOLUME_DIMENSIONS:
        pad_width.append((0, 0))  # no padding for channels

    return np.pad(
        volume,
        pad_width=pad_width,
        mode="constant",
        constant_values=value,
    )


def crop3d(
    volume: np.ndarray,
    crop_coords: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    """Crop 3D volume using coordinates.

    Args:
        volume: Input volume with shape (z, y, x) or (z, y, x, channels)
        crop_coords: Tuple of (z_min, z_max, y_min, y_max, x_min, x_max) coordinates for cropping

    Returns:
        Cropped volume with same number of dimensions as input
    """
    z_min, z_max, y_min, y_max, x_min, x_max = crop_coords

    return volume[z_min:z_max, y_min:y_max, x_min:x_max]


def cutout3d(volume: np.ndarray, holes: np.ndarray, fill_value: ColorType) -> np.ndarray:
    """Cut out holes in 3D volume and fill them with a given value."""
    volume = volume.copy()
    for z1, y1, x1, z2, y2, x2 in holes:
        volume[z1:z2, y1:y2, x1:x2] = fill_value
    return volume


def transform_cube(cube: np.ndarray, index: int) -> np.ndarray:
    """Transform cube by index (0-47)

    Args:
        cube: Input array with shape (D, H, W) or (D, H, W, C)
        index: Integer from 0 to 47 specifying which transformation to apply
    Returns:
        Transformed cube with same shape as input
    """
    if not (0 <= index < 48):
        raise ValueError("Index must be between 0 and 47")

    # First determine if we need reflection (indices 24-47)
    needs_reflection = index >= 24
    working_cube = cube[:, :, ::-1].copy() if needs_reflection else cube.copy()
    rotation_index = index % 24

    # Map rotation_index (0-23) to specific rotations
    if rotation_index < 4:
        # First 4: rotate around axis 0
        return np.rot90(working_cube, rotation_index, axes=(1, 2))

    if rotation_index < 8:
        # Next 4: flip 180° about axis 1, then rotate around axis 0
        temp = np.rot90(working_cube, 2, axes=(0, 2))
        return np.rot90(temp, rotation_index - 4, axes=(1, 2))

    if rotation_index < 16:
        # Next 8: split between 90° and 270° about axis 1, then rotate around axis 2
        if rotation_index < 12:
            temp = np.rot90(working_cube, axes=(0, 2))
            return np.rot90(temp, rotation_index - 8, axes=(0, 1))
        temp = np.rot90(working_cube, -1, axes=(0, 2))
        return np.rot90(temp, rotation_index - 12, axes=(0, 1))

    # Final 8: split between rotations about axis 2, then rotate around axis 1
    if rotation_index < 20:
        temp = np.rot90(working_cube, axes=(0, 1))
        return np.rot90(temp, rotation_index - 16, axes=(0, 2))
    temp = np.rot90(working_cube, -1, axes=(0, 1))
    return np.rot90(temp, rotation_index - 20, axes=(0, 2))


@handle_empty_array("keypoints")
def filter_keypoints_in_holes3d(keypoints: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Filter out keypoints that are inside any of the 3D holes.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 3+).
                               The first three columns are x, y, z coordinates.
        holes (np.ndarray): Array of holes with shape (num_holes, 6).
                           Each hole is represented as [z1, y1, x1, z2, y2, x2].

    Returns:
        np.ndarray: Array of keypoints that are not inside any hole.
    """
    if holes.size == 0:
        return keypoints

    # Broadcast keypoints and holes for vectorized comparison
    # Convert keypoints from XYZ to ZYX for comparison with holes
    kp_z = keypoints[:, 2][:, np.newaxis]  # Shape: (num_keypoints, 1)
    kp_y = keypoints[:, 1][:, np.newaxis]  # Shape: (num_keypoints, 1)
    kp_x = keypoints[:, 0][:, np.newaxis]  # Shape: (num_keypoints, 1)

    # Extract hole coordinates (in ZYX order)
    hole_z1 = holes[:, 0]  # Shape: (num_holes,)
    hole_y1 = holes[:, 1]
    hole_x1 = holes[:, 2]
    hole_z2 = holes[:, 3]
    hole_y2 = holes[:, 4]
    hole_x2 = holes[:, 5]

    # Check if each keypoint is inside each hole
    inside_hole = (
        (kp_z >= hole_z1)
        & (kp_z < hole_z2)
        & (kp_y >= hole_y1)
        & (kp_y < hole_y2)
        & (kp_x >= hole_x1)
        & (kp_x < hole_x2)
    )

    # A keypoint is valid if it's not inside any hole
    valid_keypoints = ~np.any(inside_hole, axis=1)

    # Return filtered keypoints with same dtype as input
    result = keypoints[valid_keypoints]
    if len(result) == 0:
        # Ensure empty result has correct shape and dtype
        return np.array([], dtype=keypoints.dtype).reshape(0, keypoints.shape[1])
    return result


def keypoints_rot90(
    keypoints: np.ndarray,
    k: int,
    axes: tuple[int, int],
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    if k == 0 or len(keypoints) == 0:
        return keypoints

    # Normalize factor to range [0, 3]
    k = ((k % 4) + 4) % 4

    result = keypoints.copy()

    # Get dimensions for the rotation axes
    dims = [volume_shape[ax] for ax in axes]

    # Get coordinates to rotate
    coords1 = result[:, axes[0]].copy()
    coords2 = result[:, axes[1]].copy()

    # Apply rotation based on factor (counterclockwise)
    if k == 1:  # 90 degrees CCW
        result[:, axes[0]] = (dims[1] - 1) - coords2
        result[:, axes[1]] = coords1
    elif k == 2:  # 180 degrees
        result[:, axes[0]] = (dims[0] - 1) - coords1
        result[:, axes[1]] = (dims[1] - 1) - coords2
    elif k == 3:  # 270 degrees CCW
        result[:, axes[0]] = coords2
        result[:, axes[1]] = (dims[0] - 1) - coords1

    return result


@handle_empty_array("keypoints")
def transform_cube_keypoints(
    keypoints: np.ndarray,
    index: int,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    if not (0 <= index < 48):
        raise ValueError("Index must be between 0 and 47")

    # Create working copy preserving all columns
    working_points = keypoints.copy()

    # Convert only XYZ coordinates to HWD, keeping other columns unchanged
    xyz = working_points[:, :3]  # Get first 3 columns (XYZ)
    xyz = xyz[:, [2, 1, 0]]  # XYZ -> HWD
    working_points[:, :3] = xyz  # Put back transformed coordinates

    current_shape = volume_shape

    # Handle reflection first (indices 24-47)
    if index >= 24:
        working_points[:, 2] = current_shape[2] - 1 - working_points[:, 2]  # Reflect W axis

    rotation_index = index % 24

    # Apply the same rotation logic as transform_cube
    if rotation_index < 4:
        # First 4: rotate around axis 0
        result = keypoints_rot90(working_points, k=rotation_index, axes=(1, 2), volume_shape=current_shape)
    elif rotation_index < 8:
        # Next 4: flip 180° about axis 1, then rotate around axis 0
        temp = keypoints_rot90(working_points, k=2, axes=(0, 2), volume_shape=current_shape)
        result = keypoints_rot90(temp, k=rotation_index - 4, axes=(1, 2), volume_shape=volume_shape)
    elif rotation_index < 16:
        if rotation_index < 12:
            temp = keypoints_rot90(working_points, k=1, axes=(0, 2), volume_shape=current_shape)
            temp_shape = (current_shape[2], current_shape[1], current_shape[0])
            result = keypoints_rot90(temp, k=rotation_index - 8, axes=(0, 1), volume_shape=temp_shape)
        else:
            temp = keypoints_rot90(working_points, k=3, axes=(0, 2), volume_shape=current_shape)
            temp_shape = (current_shape[2], current_shape[1], current_shape[0])
            result = keypoints_rot90(temp, k=rotation_index - 12, axes=(0, 1), volume_shape=temp_shape)
    elif rotation_index < 20:
        temp = keypoints_rot90(working_points, k=1, axes=(0, 1), volume_shape=current_shape)
        temp_shape = (current_shape[1], current_shape[0], current_shape[2])
        result = keypoints_rot90(temp, k=rotation_index - 16, axes=(0, 2), volume_shape=temp_shape)
    else:
        temp = keypoints_rot90(working_points, k=3, axes=(0, 1), volume_shape=current_shape)
        temp_shape = (current_shape[1], current_shape[0], current_shape[2])
        result = keypoints_rot90(temp, k=rotation_index - 20, axes=(0, 2), volume_shape=temp_shape)

    # Convert back from HWD to XYZ coordinates for first 3 columns only
    xyz = result[:, :3]
    xyz = xyz[:, [2, 1, 0]]  # HWD -> XYZ
    result[:, :3] = xyz

    return result
