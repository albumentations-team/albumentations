"""Functional implementations of dropout operations for image augmentation.

This module provides low-level functions for various dropout techniques used in image
augmentation, including channel dropout, grid dropout, mask dropout, and coarse dropout.
These functions create and apply dropout patterns to images, masks, bounding boxes, and
keypoints, with support for different filling methods and hole generation strategies.
"""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
from albucore import (
    MAX_VALUES_BY_DTYPE,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    get_num_channels,
    is_grayscale_image,
    preserve_channel_dim,
    uint8_io,
)

from albumentations.augmentations.geometric.functional import split_uniform_grid
from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.type_definitions import MONO_CHANNEL_DIMENSIONS

__all__ = [
    "calculate_grid_dimensions",
    "channel_dropout",
    "cutout",
    "filter_bboxes_by_holes",
    "filter_keypoints_in_holes",
    "generate_grid_holes",
    "generate_random_fill",
]


@preserve_channel_dim
def channel_dropout(
    img: np.ndarray,
    channels_to_drop: int | tuple[int, ...] | np.ndarray,
    fill: tuple[float, ...] | float = 0,
) -> np.ndarray:
    """Drop channels from an image.

    This function drops channels from an image.

    Args:
        img (np.ndarray): Input image.
        channels_to_drop (int | tuple[int, ...] | np.ndarray): Channels to drop.
        fill (tuple[float, ...] | float): Value to fill the dropped channels with.

    Returns:
        np.ndarray: Image with channels dropped.

    """
    if is_grayscale_image(img):
        msg = "Only one channel. ChannelDropout is not defined."
        raise NotImplementedError(msg)

    img = img.copy()
    img[..., channels_to_drop] = fill
    return img


def generate_random_fill(
    dtype: np.dtype,
    shape: tuple[int, ...],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate a random fill array based on the given dtype and target shape.

    This function creates a numpy array filled with random values. The range and type of these values
    depend on the input dtype. For integer dtypes, it generates random integers. For floating-point
    dtypes, it generates random floats.

    Args:
        dtype (np.dtype): The data type of the array to be generated.
        shape (tuple[int, ...]): The shape of the array to be generated.
        random_generator (np.random.Generator): The random generator to use for generating values.
            If None, the default numpy random generator is used.

    Returns:
        np.ndarray: A numpy array of the specified shape and dtype, filled with random values.

    Raises:
        ValueError: If the input dtype is neither integer nor floating-point.

    Examples:
        >>> import numpy as np
        >>> random_state = np.random.RandomState(42)
        >>> result = generate_random_fill(np.dtype('uint8'), (2, 2), random_state)
        >>> print(result)
        [[172 251]
         [ 80 141]]

    """
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    if np.issubdtype(dtype, np.integer):
        return random_generator.integers(0, max_value + 1, size=shape, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        return random_generator.uniform(0, max_value, size=shape).astype(dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


@uint8_io
def apply_inpainting(img: np.ndarray, holes: np.ndarray, method: Literal["inpaint_telea", "inpaint_ns"]) -> np.ndarray:
    """Apply OpenCV inpainting to fill the holes in the image.

    Args:
        img (np.ndarray): Input image (grayscale or BGR)
        holes (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        method (Literal["inpaint_telea", "inpaint_ns"]): Inpainting method to use

    Returns:
        np.ndarray: Inpainted image

    Raises:
        NotImplementedError: If image has more than 3 channels

    """
    num_channels = get_num_channels(img)
    # Create inpainting mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for x_min, y_min, x_max, y_max in holes:
        mask[y_min:y_max, x_min:x_max] = 255

    inpaint_method = cv2.INPAINT_TELEA if method == "inpaint_telea" else cv2.INPAINT_NS

    # Handle grayscale images by converting to 3 channels and back
    if num_channels == 1:
        if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            img = img.squeeze()
        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        result = cv2.inpaint(img_3ch, mask, 3, inpaint_method)
        return (
            cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)[..., None]
            if num_channels == NUM_MULTI_CHANNEL_DIMENSIONS
            else cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        )

    return cv2.inpaint(img, mask, 3, inpaint_method)


def fill_holes_with_value(img: np.ndarray, holes: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Fill holes with a constant value.

    Args:
        img (np.ndarray): Input image
        holes (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        fill (np.ndarray): Value to fill the holes with

    """
    for x_min, y_min, x_max, y_max in holes:
        img[y_min:y_max, x_min:x_max] = fill
    return img


def fill_volume_holes_with_value(volume: np.ndarray, holes: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Fill holes in a volume with a constant value.

    Args:
        volume (np.ndarray): Input volume
        holes (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        fill (np.ndarray): Value to fill the holes with

    """
    for x_min, y_min, x_max, y_max in holes:
        volume[:, y_min:y_max, x_min:x_max] = fill
    return volume


def fill_volumes_holes_with_value(volumes: np.ndarray, holes: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Fill holes in a batch of volumes with a constant value.

    Args:
        volumes (np.ndarray): Input batch of volumes
        holes (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        fill (np.ndarray): Value to fill the holes with

    """
    for x_min, y_min, x_max, y_max in holes:
        volumes[:, :, y_min:y_max, x_min:x_max] = fill
    return volumes


def fill_holes_with_random(
    img: np.ndarray,
    holes: np.ndarray,
    random_generator: np.random.Generator,
    uniform: bool,
) -> np.ndarray:
    """Fill holes with random values.

    Args:
        img (np.ndarray): Input image
        holes (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        random_generator (np.random.Generator): Random number generator
        uniform (bool): If True, use same random value for entire hole

    """
    for x_min, y_min, x_max, y_max in holes:
        shape = (1,) if uniform else (y_max - y_min, x_max - x_min)
        if img.ndim != MONO_CHANNEL_DIMENSIONS:
            shape = (1, img.shape[2]) if uniform else (*shape, img.shape[2])

        random_fill = generate_random_fill(img.dtype, shape, random_generator)
        img[y_min:y_max, x_min:x_max] = random_fill
    return img


def fill_volume_holes_with_random(
    volume: np.ndarray,
    holes: np.ndarray,
    random_generator: np.random.Generator,
    uniform: bool,
) -> np.ndarray:
    """Fill holes in a volume with random values.

    Args:
        volume (np.ndarray): Input volume of shape (D, H, W, C) or (D, H, W)
        holes (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        random_generator (np.random.Generator): Random number generator
        uniform (bool): If True, use same random value for entire hole in each image.

    """
    for x_min, y_min, x_max, y_max in holes:
        shape = (volume.shape[0], 1, 1) if uniform else (volume.shape[0], y_max - y_min, x_max - x_min)
        if volume.ndim != 3:
            shape = (volume.shape[0], 1, 1, volume.shape[3]) if uniform else (*shape, volume.shape[3])

        random_fill = generate_random_fill(volume.dtype, shape, random_generator)
        volume[:, y_min:y_max, x_min:x_max] = random_fill
    return volume


def fill_volumes_holes_with_random(
    volumes: np.ndarray,
    holes: np.ndarray,
    random_generator: np.random.Generator,
    uniform: bool,
) -> np.ndarray:
    """Fill holes in a batch of volumes with random values.

    Args:
        volumes (np.ndarray): Input volume of shape (N, D, H, W, C) or (N, D, H, W)
        holes (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        random_generator (np.random.Generator): Random number generator
        uniform (bool): If True, use same random value for entire hole for each image

    """
    for x_min, y_min, x_max, y_max in holes:
        shape = (
            (volumes.shape[0], volumes.shape[1], 1, 1)
            if uniform
            else (volumes.shape[0], volumes.shape[1], y_max - y_min, x_max - x_min)
        )
        if volumes.ndim != 4:
            shape = (
                (volumes.shape[0], volumes.shape[1], 1, 1, volumes.shape[4]) if uniform else (*shape, volumes.shape[4])
            )
        random_fill = generate_random_fill(volumes.dtype, shape, random_generator)
        volumes[:, :, y_min:y_max, x_min:x_max] = random_fill
    return volumes


def cutout(
    img: np.ndarray,
    holes: np.ndarray,
    fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply cutout augmentation to the image by cutting out holes and filling them.

    Args:
        img (np.ndarray): The image to augment
        holes (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        fill (tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]):
            Value to fill holes with. Can be:
            - number (int/float): Will be broadcast to all channels
            - sequence (tuple/list/ndarray): Must match number of channels
            - "random": Different random values for each pixel
            - "random_uniform": Same random value for entire hole
            - "inpaint_telea"/"inpaint_ns": OpenCV inpainting methods
        random_generator (np.random.Generator): Random number generator for random fills

    Raises:
        ValueError: If fill length doesn't match number of channels

    """
    img = img.copy()

    # Handle inpainting methods
    if isinstance(fill, str):
        if fill in {"inpaint_telea", "inpaint_ns"}:
            return apply_inpainting(img, holes, cast("Literal['inpaint_telea', 'inpaint_ns']", fill))
        if fill == "random":
            return fill_holes_with_random(img, holes, random_generator, uniform=False)
        if fill == "random_uniform":
            return fill_holes_with_random(img, holes, random_generator, uniform=True)
        raise ValueError(f"Unsupported string fill: {fill}")

    # Convert numeric fill values to numpy array
    if isinstance(fill, (int, float)):
        fill_array = np.array(fill, dtype=img.dtype)
        return fill_holes_with_value(img, holes, fill_array)

    # Handle sequence fill values
    fill_array = np.array(fill, dtype=img.dtype)

    # For multi-channel images, verify fill matches number of channels
    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        fill_array = fill_array.ravel()
        if fill_array.size != img.shape[2]:
            raise ValueError(
                f"Fill value must have same number of channels as image. "
                f"Got {fill_array.size}, expected {img.shape[2]}",
            )

    return fill_holes_with_value(img, holes, fill_array)


def cutout_on_volume(
    volume: np.ndarray,
    holes: np.ndarray,
    fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply cutout augmentation to a volume of shape (D, H, W) or (D, H, W, C) by cutting out holes and filling them.

    Args:
        volume (np.ndarray): The volume to augment
        holes (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        fill (tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]):
            Value to fill holes with. Can be:
            - number (int/float): Will be broadcast to all channels
            - sequence (tuple/list/ndarray): Must match number of channels
            - "random": Different random values for each pixel
            - "random_uniform": Same random value for entire hole, different values across images
            - "inpaint_telea"/"inpaint_ns": OpenCV inpainting methods
        random_generator (np.random.Generator): Random number generator for random fills

    Raises:
        ValueError: If fill length doesn't match number of channels

    """
    volume = volume.copy()

    # Handle inpainting methods
    if isinstance(fill, str):
        if fill in {"inpaint_telea", "inpaint_ns"}:
            processed_images = [
                apply_inpainting(img, holes, cast("Literal['inpaint_telea', 'inpaint_ns']", fill)) for img in volume
            ]
            result = np.array(processed_images)
            # Reshape to original volume shape: (D, H, W, C) or (D, H, W)
            return result.reshape(volume.shape)
        if fill == "random":
            return fill_volume_holes_with_random(volume, holes, random_generator, uniform=False)
        if fill == "random_uniform":
            return fill_volume_holes_with_random(volume, holes, random_generator, uniform=True)
        raise ValueError(f"Unsupported string fill: {fill}")

    # Convert numeric fill values to numpy array
    if isinstance(fill, (int, float)):
        fill_array = np.array(fill, dtype=volume.dtype)
        return fill_volume_holes_with_value(volume, holes, fill_array)

    # Handle sequence fill values
    fill_array = np.array(fill, dtype=volume.dtype)

    # For multi-channel images, verify fill matches number of channels
    if volume.ndim == 4:
        fill_array = fill_array.ravel()
        if fill_array.size != volume.shape[3]:
            raise ValueError(
                f"Fill value must have same number of channels as image. "
                f"Got {fill_array.size}, expected {volume.shape[3]}",
            )

    return fill_volume_holes_with_value(volume, holes, fill_array)


def cutout_on_volumes(
    volumes: np.ndarray,
    holes: np.ndarray,
    fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply cutout augmentation to a batch of volumes of shape (N, D, H, W) or (N, D, H, W, C)

    Args:
        volumes (np.ndarray): The image to augment
        holes (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        fill (tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]):
            Value to fill holes with. Can be:
            - number (int/float): Will be broadcast to all channels
            - sequence (tuple/list/ndarray): Must match number of channels
            - "random": Different random values for each pixel
            - "random_uniform": Same random value for entire hole, different values across images
            - "inpaint_telea"/"inpaint_ns": OpenCV inpainting methods
        random_generator (np.random.Generator): Random number generator for random fills

    Raises:
        ValueError: If fill length doesn't match number of channels

    """
    volumes = volumes.copy()

    # Handle inpainting methods
    if isinstance(fill, str):
        if fill in {"inpaint_telea", "inpaint_ns"}:
            processed_images = [
                apply_inpainting(img, holes, cast("Literal['inpaint_telea', 'inpaint_ns']", fill))
                for volume in volumes
                for img in volume
            ]
            result = np.array(processed_images)
            # Reshape to original batch of volumes shape: (N, D, H, W, C) or (N, D, H, W)
            return result.reshape(volumes.shape)
        if fill == "random":
            return fill_volumes_holes_with_random(volumes, holes, random_generator, uniform=False)
        if fill == "random_uniform":
            return fill_volumes_holes_with_random(volumes, holes, random_generator, uniform=True)
        raise ValueError(f"Unsupported string fill: {fill}")

    # Convert numeric fill values to numpy array
    if isinstance(fill, (int, float)):
        fill_array = np.array(fill, dtype=volumes.dtype)
        return fill_volumes_holes_with_value(volumes, holes, fill_array)

    # Handle sequence fill values
    fill_array = np.array(fill, dtype=volumes.dtype)

    # For multi-channel images, verify fill matches number of channels
    if volumes.ndim == 5:
        fill_array = fill_array.ravel()
        if fill_array.size != volumes.shape[4]:
            raise ValueError(
                f"Fill value must have same number of channels as image. "
                f"Got {fill_array.size}, expected {volumes.shape[4]}",
            )

    return fill_volumes_holes_with_value(volumes, holes, fill_array)


@handle_empty_array("keypoints")
def filter_keypoints_in_holes(keypoints: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Filter out keypoints that are inside any of the holes.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+).
                                The first two columns are x and y coordinates.
        holes (np.ndarray): Array of holes with shape (num_holes, 4).
                            Each hole is represented as [x1, y1, x2, y2].

    Returns:
        np.ndarray: Array of keypoints that are not inside any hole.

    """
    # Broadcast keypoints and holes for vectorized comparison
    kp_x = keypoints[:, 0][:, np.newaxis]  # Shape: (num_keypoints, 1)
    kp_y = keypoints[:, 1][:, np.newaxis]  # Shape: (num_keypoints, 1)

    hole_x1 = holes[:, 0]  # Shape: (num_holes,)
    hole_y1 = holes[:, 1]  # Shape: (num_holes,)
    hole_x2 = holes[:, 2]  # Shape: (num_holes,)
    hole_y2 = holes[:, 3]  # Shape: (num_holes,)

    # Check if each keypoint is inside each hole
    inside_hole = (kp_x >= hole_x1) & (kp_x < hole_x2) & (kp_y >= hole_y1) & (kp_y < hole_y2)

    # A keypoint is valid if it's not inside any hole
    valid_keypoints = ~np.any(inside_hole, axis=1)

    return keypoints[valid_keypoints]


@handle_empty_array("bboxes")
def resize_boxes_to_visible_area(
    boxes: np.ndarray,
    hole_mask: np.ndarray,
) -> np.ndarray:
    """Resize boxes to their largest visible rectangular regions."""
    # Extract box coordinates
    x1 = boxes[:, 0].astype(int)
    y1 = boxes[:, 1].astype(int)
    x2 = boxes[:, 2].astype(int)
    y2 = boxes[:, 3].astype(int)

    # Process each box individually to avoid array shape issues
    new_boxes: list[np.ndarray] = []

    regions = [hole_mask[y1[i] : y2[i], x1[i] : x2[i]] for i in range(len(boxes))]
    visible_areas = [1 - region for region in regions]

    for i, (visible, box) in enumerate(zip(visible_areas, boxes)):
        if not visible.any():
            continue

        # Find visible coordinates
        y_visible = visible.any(axis=1)
        x_visible = visible.any(axis=0)

        y_coords = np.nonzero(y_visible)[0]
        x_coords = np.nonzero(x_visible)[0]

        # Update only the coordinate part of the box
        new_box = box.copy()
        new_box[0] = x1[i] + x_coords[0]  # x_min
        new_box[1] = y1[i] + y_coords[0]  # y_min
        new_box[2] = x1[i] + x_coords[-1] + 1  # x_max
        new_box[3] = y1[i] + y_coords[-1] + 1  # y_max

        new_boxes.append(new_box)

        # Return empty array with correct shape if all boxes were removed

    return np.array(new_boxes) if new_boxes else np.zeros((0, boxes.shape[1]), dtype=boxes.dtype)


def filter_bboxes_by_holes(
    bboxes: np.ndarray,
    holes: np.ndarray,
    image_shape: tuple[int, int],
    min_area: float,
    min_visibility: float,
) -> np.ndarray:
    """Filter bounding boxes by holes.

    This function filters bounding boxes by holes.

    Args:
        bboxes (np.ndarray): Array of bounding boxes.
        holes (np.ndarray): Array of holes.
        image_shape (tuple[int, int]): Shape of the image.
        min_area (float): Minimum area of a bounding box.
        min_visibility (float): Minimum visibility of a bounding box.

    Returns:
        np.ndarray: Filtered bounding boxes.

    """
    if len(bboxes) == 0 or len(holes) == 0:
        return bboxes

    # Create hole mask
    hole_mask = np.zeros(image_shape, dtype=np.uint8)
    for hole in holes:
        x_min, y_min, x_max, y_max = hole.astype(int)
        hole_mask[y_min:y_max, x_min:x_max] = 1

    # Filter boxes by area and visibility
    bboxes_int = bboxes.astype(int)
    box_areas = (bboxes_int[:, 2] - bboxes_int[:, 0]) * (bboxes_int[:, 3] - bboxes_int[:, 1])
    intersection_areas = np.array([np.sum(hole_mask[y:y2, x:x2]) for x, y, x2, y2 in bboxes_int[:, :4]])
    remaining_areas = box_areas - intersection_areas
    visibility_ratios = remaining_areas / box_areas
    mask = (remaining_areas >= min_area) & (visibility_ratios >= min_visibility) & (remaining_areas > 0)

    valid_boxes = bboxes[mask]
    if len(valid_boxes) == 0:
        return np.empty((0, bboxes.shape[1]))

    # Try to resize valid boxes
    return resize_boxes_to_visible_area(valid_boxes, hole_mask)


def calculate_grid_dimensions(
    image_shape: tuple[int, int],
    unit_size_range: tuple[int, int] | None,
    holes_number_xy: tuple[int, int] | None,
    random_generator: np.random.Generator,
) -> tuple[int, int]:
    """Calculate the dimensions of grid units for GridDropout.

    This function determines the size of grid units based on the input parameters.
    It supports three modes of operation:
    1. Using a range of unit sizes
    2. Using a specified number of holes in x and y directions
    3. Falling back to a default calculation

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        unit_size_range (tuple[int, int] | None, optional): A range of possible unit sizes.
            If provided, a random size within this range will be chosen for both height and width.
        holes_number_xy (tuple[int, int] | None, optional): The number of holes in the x and y directions.
            If provided, the grid dimensions will be calculated to fit this number of holes.
        random_generator (np.random.Generator): The random generator to use for generating random values.

    Returns:
        tuple[int, int]: The calculated grid unit dimensions as (unit_height, unit_width).

    Raises:
        ValueError: If the upper limit of unit_size_range is greater than the shortest image edge.

    Notes:
        - If both unit_size_range and holes_number_xy are None, the function falls back to a default calculation,
          where the grid unit size is set to max(2, image_dimension // 10) for both height and width.
        - The function prioritizes unit_size_range over holes_number_xy if both are provided.
        - When using holes_number_xy, the actual number of holes may be slightly different due to integer division.

    Examples:
        >>> image_shape = (100, 200)
        >>> calculate_grid_dimensions(image_shape, unit_size_range=(10, 20))
        (15, 15)  # Random value between 10 and 20

        >>> calculate_grid_dimensions(image_shape, holes_number_xy=(5, 10))
        (20, 20)  # 100 // 5 and 200 // 10

        >>> calculate_grid_dimensions(image_shape)
        (10, 20)  # Default calculation: max(2, dimension // 10)

    """
    height, width = image_shape[:2]

    if unit_size_range is not None:
        if unit_size_range[1] > min(image_shape[:2]):
            raise ValueError("Grid size limits must be within the shortest image edge.")
        unit_size = random_generator.integers(*unit_size_range)
        return unit_size, unit_size

    if holes_number_xy:
        holes_number_x, holes_number_y = holes_number_xy
        unit_width = width // holes_number_x
        unit_height = height // holes_number_y
        return unit_height, unit_width

    # Default fallback
    unit_width = max(2, width // 10)
    unit_height = max(2, height // 10)
    return unit_height, unit_width


def generate_grid_holes(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    ratio: float,
    random_offset: bool,
    shift_xy: tuple[int, int],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate a list of holes for GridDropout using a uniform grid.

    This function creates a grid of holes for use in the GridDropout augmentation technique.
    It allows for customization of the grid size, hole size ratio, and positioning of holes.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        grid (tuple[int, int]): The grid size as (rows, columns). This determines the number of cells
            in the grid, where each cell may contain a hole.
        ratio (float): The ratio of the hole size to the grid cell size. Should be between 0 and 1.
            A ratio of 1 means the hole will fill the entire grid cell.
        random_offset (bool): If True, applies random offsets to each hole within its grid cell.
            If False, uses the global shift specified by shift_xy.
        shift_xy (tuple[int, int]): The global shift to apply to all holes as (shift_x, shift_y).
            Only used when random_offset is False.
        random_generator (np.random.Generator): The random generator for generating random offsets
            and shuffling. If None, a new Generator will be created.

    Returns:
        np.ndarray: An array of hole coordinates, where each hole is represented as
            [x1, y1, x2, y2]. The shape of the array is (n_holes, 4), where n_holes
            is determined by the grid size.

    Notes:
        - The function first creates a uniform grid based on the image shape and specified grid size.
        - Hole sizes are calculated based on the provided ratio and grid cell sizes.
        - If random_offset is True, each hole is randomly positioned within its grid cell.
        - If random_offset is False, all holes are shifted by the global shift_xy value.
        - The function ensures that all holes remain within the image boundaries.

    Examples:
        >>> image_shape = (100, 100)
        >>> grid = (5, 5)
        >>> ratio = 0.5
        >>> random_offset = True
        >>> random_state = np.random.RandomState(42)
        >>> shift_xy = (0, 0)
        >>> holes = generate_grid_holes(image_shape, grid, ratio, random_offset, random_state, shift_xy)
        >>> print(holes.shape)
        (25, 4)
        >>> print(holes[0])  # Example output: [x1, y1, x2, y2] of the first hole
        [ 1 21 11 31]

    """
    height, width = image_shape[:2]

    # Generate the uniform grid
    cells = split_uniform_grid(image_shape, grid, random_generator)

    # Calculate hole sizes based on the ratio
    cell_heights = cells[:, 2] - cells[:, 0]
    cell_widths = cells[:, 3] - cells[:, 1]
    hole_heights = np.clip(cell_heights * ratio, 1, cell_heights - 1).astype(int)
    hole_widths = np.clip(cell_widths * ratio, 1, cell_widths - 1).astype(int)

    # Calculate maximum possible offsets
    max_offset_y = cell_heights - hole_heights
    max_offset_x = cell_widths - hole_widths

    if random_offset:
        # Generate random offsets for each hole
        offset_y = random_generator.integers(0, max_offset_y + 1)
        offset_x = random_generator.integers(0, max_offset_x + 1)
    else:
        # Use global shift
        offset_y = np.full_like(max_offset_y, shift_xy[1])
        offset_x = np.full_like(max_offset_x, shift_xy[0])

    # Calculate hole coordinates
    x_min = np.clip(cells[:, 1] + offset_x, 0, width - hole_widths)
    y_min = np.clip(cells[:, 0] + offset_y, 0, height - hole_heights)
    x_max = np.minimum(x_min + hole_widths, width)
    y_max = np.minimum(y_min + hole_heights, height)

    return np.column_stack((x_min, y_min, x_max, y_max))


@handle_empty_array("bboxes")
def mask_dropout_bboxes(
    bboxes: np.ndarray,
    dropout_mask: np.ndarray,
    image_shape: tuple[int, int],
    min_area: float,
    min_visibility: float,
) -> np.ndarray:
    """Filter and resize bounding boxes based on dropout mask.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+)
        dropout_mask (np.ndarray): Binary mask indicating dropped areas
        image_shape (tuple[int, int]): Shape of the image (height, width)
        min_area (float): Minimum area of a bounding box to keep
        min_visibility (float): Minimum visibility ratio of a bounding box to keep

    Returns:
        np.ndarray: Filtered and resized bounding boxes

    """
    height, width = image_shape

    # Ensure dropout_mask is 2D
    if dropout_mask.ndim > 2:
        if dropout_mask.shape[0] == 1:  # Shape is (1, H, W)
            dropout_mask = dropout_mask.squeeze(0)
        elif dropout_mask.shape[-1] <= 4:  # Shape is (H, W, C)
            dropout_mask = np.any(dropout_mask, axis=-1)
        else:  # Shape is (C, H, W)
            dropout_mask = np.any(dropout_mask, axis=0)

    # Create binary masks for each bounding box
    y, x = np.ogrid[:height, :width]
    box_masks = (
        (x[None, :] >= bboxes[:, 0, None, None])
        & (x[None, :] <= bboxes[:, 2, None, None])
        & (y[None, :] >= bboxes[:, 1, None, None])
        & (y[None, :] <= bboxes[:, 3, None, None])
    )

    # Calculate the area of each bounding box
    box_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    # Calculate the visible area of each box (non-intersecting area with dropout mask)
    visible_areas = np.sum(box_masks & ~dropout_mask, axis=(1, 2))

    # Calculate visibility ratio (visible area / total box area)
    visibility_ratio = visible_areas / box_areas

    # Create a boolean mask for boxes to keep
    keep_mask = (visible_areas >= min_area) & (visibility_ratio >= min_visibility)

    return bboxes[keep_mask]


@handle_empty_array("keypoints")
def mask_dropout_keypoints(
    keypoints: np.ndarray,
    dropout_mask: np.ndarray,
) -> np.ndarray:
    """Filter keypoints based on dropout mask.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (num_keypoints, 2+)
        dropout_mask (np.ndarray): Binary mask indicating dropped areas

    Returns:
        np.ndarray: Filtered keypoints

    """
    # Ensure dropout_mask is 2D
    if dropout_mask.ndim > 2:
        if dropout_mask.shape[0] == 1:  # Shape is (1, H, W)
            dropout_mask = dropout_mask.squeeze(0)
        elif dropout_mask.shape[-1] <= 4:  # Shape is (H, W, C)
            dropout_mask = np.any(dropout_mask, axis=-1)
        else:  # Shape is (C, H, W)
            dropout_mask = np.any(dropout_mask, axis=0)

    # Get coordinates as integers
    coords = keypoints[:, :2].astype(int)

    # Filter out keypoints that are outside the mask dimensions
    valid_mask = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < dropout_mask.shape[1])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < dropout_mask.shape[0])
    )

    # For valid keypoints, check if they fall on non-dropped pixels
    if np.any(valid_mask):
        valid_coords = coords[valid_mask]
        valid_mask[valid_mask] = ~dropout_mask[valid_coords[:, 1], valid_coords[:, 0]]

    return keypoints[valid_mask]


def label(mask: np.ndarray, return_num: bool = False, connectivity: int = 2) -> np.ndarray | tuple[np.ndarray, int]:
    """Label connected regions of an integer array.

    This function uses OpenCV's connectedComponents under the hood but mimics
    the behavior of scikit-image's label function.

    Args:
        mask (np.ndarray): The array to label. Must be of integer type.
        return_num (bool): If True, return the number of labels (default: False).
        connectivity (int): Maximum number of orthogonal hops to consider a pixel/voxel
                            as a neighbor. Accepted values are 1 or 2. Default is 2.

    Returns:
        np.ndarray | tuple[np.ndarray, int]: Labeled array, where all connected regions are
        assigned the same integer value. If return_num is True, it also returns the number of labels.

    """
    # Create a copy of the original mask
    labeled = np.zeros_like(mask, dtype=np.int32)

    # Get unique non-zero values from the original mask
    unique_values = np.unique(mask[mask != 0])

    # Label each unique value separately
    next_label = 1
    for value in unique_values:
        binary_mask = (mask == value).astype(np.uint8)

        # Set connectivity for OpenCV (4 or 8)
        cv2_connectivity = 4 if connectivity == 1 else 8

        # Use OpenCV's connectedComponents
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=cv2_connectivity)

        # Assign new labels
        for i in range(1, num_labels):
            labeled[labels == i] = next_label
            next_label += 1

    num_labels = next_label - 1

    return (labeled, num_labels) if return_num else labeled


def get_holes_from_boxes(
    target_boxes: np.ndarray,
    num_holes_per_box: int,
    hole_height_range: tuple[float, float],
    hole_width_range: tuple[float, float],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate holes based on bounding boxes."""
    num_boxes = len(target_boxes)

    # Get box dimensions (N, )
    box_widths = target_boxes[:, 2] - target_boxes[:, 0]
    box_heights = target_boxes[:, 3] - target_boxes[:, 1]

    # Sample hole dimensions (N, num_holes)
    hole_heights = (
        random_generator.uniform(
            hole_height_range[0],
            hole_height_range[1],
            size=(num_boxes, num_holes_per_box),
        )
        * box_heights[:, None]
    ).astype(np.int32)

    hole_widths = (
        random_generator.uniform(
            hole_width_range[0],
            hole_width_range[1],
            size=(num_boxes, num_holes_per_box),
        )
        * box_widths[:, None]
    ).astype(np.int32)

    # Sample positions (N, num_holes)
    x_offsets = random_generator.uniform(0, 1, size=(num_boxes, num_holes_per_box)) * (
        box_widths[:, None] - hole_widths
    )
    y_offsets = random_generator.uniform(0, 1, size=(num_boxes, num_holes_per_box)) * (
        box_heights[:, None] - hole_heights
    )

    # Calculate final coordinates (N, num_holes)
    x_min = target_boxes[:, 0, None] + x_offsets
    y_min = target_boxes[:, 1, None] + y_offsets
    x_max = x_min + hole_widths
    y_max = y_min + hole_heights

    return np.stack([x_min, y_min, x_max, y_max], axis=-1).astype(np.int32).reshape(-1, 4)


def sample_points_from_components(
    mask: np.ndarray,
    num_points: int,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample points from connected components in a mask.

    Args:
        mask (np.ndarray): Binary mask
        num_points (int): Number of points to sample
        random_generator (np.random.Generator): Random number generator

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Tuple of (x_coordinates, y_coordinates) or None if no valid components

    """
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    if num_labels == 1:  # Only background
        return None

    centers = []
    obj_sizes = []
    for label in range(1, num_labels):  # Skip background (0)
        points = np.argwhere(labels == label)  # Returns (y, x) coordinates
        if len(points) == 0:
            continue

        # Calculate object size once per component
        obj_size = np.sqrt(len(points))

        # Randomly sample points from the component, allowing repeats
        indices = random_generator.choice(len(points), size=num_points, replace=True)
        sampled_points = points[indices]
        # Convert from (y, x) to (x, y)
        centers.extend(sampled_points[:, ::-1])
        # Add corresponding object size for each point
        obj_sizes.extend([obj_size] * num_points)

    return (np.array(centers), np.array(obj_sizes)) if centers else None


def get_holes_from_mask(
    mask: np.ndarray,
    num_holes_per_obj: int,
    mask_indices: list[int],
    hole_height_range: tuple[float, float],
    hole_width_range: tuple[float, float],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate holes based on segmentation mask."""
    # Create binary mask for target indices
    binary_mask = np.isin(mask, np.array(mask_indices))
    if not np.any(binary_mask):  # If no target objects found
        return np.array([], dtype=np.int32).reshape((0, 4))

    result = sample_points_from_components(binary_mask, num_holes_per_obj, random_generator)
    if result is None:
        return np.array([], dtype=np.int32).reshape((0, 4))

    centers, obj_sizes = result
    num_centers = len(centers)
    height, width = mask.shape[:2]

    # Sample hole dimensions (N,) using per-component object sizes
    hole_heights = (
        random_generator.uniform(
            hole_height_range[0],
            hole_height_range[1],
            size=num_centers,
        )
        * obj_sizes
    )
    hole_widths = (
        random_generator.uniform(
            hole_width_range[0],
            hole_width_range[1],
            size=num_centers,
        )
        * obj_sizes
    )

    # Calculate hole coordinates around centers
    half_heights = hole_heights // 2
    half_widths = hole_widths // 2

    holes = np.column_stack(
        [
            centers[:, 0] - half_widths,  # x_min
            centers[:, 1] - half_heights,  # y_min
            centers[:, 0] + half_widths,  # x_max
            centers[:, 1] + half_heights,  # y_max
        ],
    ).astype(np.int32)

    # Clip holes to image boundaries
    holes[:, 0] = np.clip(holes[:, 0], 0, width - 1)  # x_min
    holes[:, 1] = np.clip(holes[:, 1], 0, height - 1)  # y_min
    holes[:, 2] = np.clip(holes[:, 2], 0, width)  # x_max
    holes[:, 3] = np.clip(holes[:, 3], 0, height)  # y_max

    # Filter out holes that became too small after clipping
    valid_holes = (holes[:, 2] - holes[:, 0] > 0) & (holes[:, 3] - holes[:, 1] > 0)
    return holes[valid_holes]
