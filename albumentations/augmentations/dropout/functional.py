from __future__ import annotations

import cv2
import numpy as np
from albucore import MAX_VALUES_BY_DTYPE, is_grayscale_image, preserve_channel_dim
from typing_extensions import Literal

from albumentations.augmentations.geometric.functional import split_uniform_grid
from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.types import MONO_CHANNEL_DIMENSIONS, ColorType

__all__ = [
    "cutout",
    "channel_dropout",
    "filter_keypoints_in_holes",
    "generate_random_fill",
    "filter_bboxes_by_holes",
    "calculate_grid_dimensions",
    "generate_grid_holes",
]


@preserve_channel_dim
def channel_dropout(
    img: np.ndarray,
    channels_to_drop: int | tuple[int, ...] | np.ndarray,
    fill_value: ColorType = 0,
) -> np.ndarray:
    if is_grayscale_image(img):
        msg = "Only one channel. ChannelDropout is not defined."
        raise NotImplementedError(msg)

    img = img.copy()
    img[..., channels_to_drop] = fill_value
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


def cutout(
    img: np.ndarray,
    holes: np.ndarray,
    fill_value: ColorType | Literal["random"],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply cutout augmentation to the image by cutting out holes and filling them
    with either a given value or random noise.

    Args:
        img (np.ndarray): The image to augment. Can be a 2D (grayscale) or 3D (color) array.
        holes (np.ndarray): An array of holes with shape (num_holes, 4).
            Each hole is represented as [x1, y1, x2, y2].
        fill_value (ColorType | Literal["random"]): The fill value to use for the holes.
            Can be a single integer, a tuple or list of numbers for multichannel images,
            or the string "random" to fill with random noise.
        random_generator (np.random.Generator): The random generator to use for generating
            random fill values. If None, a new random generator will be used. Defaults to None.

    Returns:
        np.ndarray: The augmented image with cutout holes applied.

    Raises:
        ValueError: If the fill_value is not of the expected type.

    Note:
        - The function creates a copy of the input image before applying the cutout.
        - For multichannel images, the fill_value should match the number of channels.
        - When using "random" fill, the random values are generated to match the image's dtype and shape.

    Example:
        >>> import numpy as np
        >>> img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        >>> holes = np.array([[20, 20, 40, 40], [60, 60, 80, 80]])
        >>> result = cutout(img, holes, fill_value=0)
        >>> print(result.shape)
        (100, 100, 3)
    """
    img = img.copy()

    if isinstance(fill_value, (int, float, tuple, list)):
        fill_value = np.array(fill_value, dtype=img.dtype)

    for x_min, y_min, x_max, y_max in holes:
        if isinstance(fill_value, str) and fill_value == "random":
            shape = (
                (y_max - y_min, x_max - x_min)
                if img.ndim == MONO_CHANNEL_DIMENSIONS
                else (y_max - y_min, x_max - x_min, img.shape[2])
            )
            random_fill = generate_random_fill(img.dtype, shape, random_generator)
            img[y_min:y_max, x_min:x_max] = random_fill
        else:
            img[y_min:y_max, x_min:x_max] = fill_value

    return img


@handle_empty_array
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


def filter_bboxes_by_holes(
    bboxes: np.ndarray,
    holes: np.ndarray,
    image_shape: tuple[int, int],
    min_area: float,
    min_visibility: float,
) -> np.ndarray:
    """Filter bounding boxes based on their remaining visible area and visibility ratio after intersection with holes.

    Args:
        bboxes (np.ndarray): Array of bounding boxes, each represented as [x_min, y_min, x_max, y_max].
        holes (np.ndarray): Array of holes, each represented as [x_min, y_min, x_max, y_max].
        image_shape (tuple[int, int]): Shape of the image (height, width).
        min_area (int): Minimum remaining visible area to keep the bounding box.
        min_visibility (float): Minimum visibility ratio to keep the bounding box.
            Calculated as 1 - (intersection_area / bbox_area).

    Returns:
        np.ndarray: Filtered array of bounding boxes.
    """
    if len(bboxes) == 0 or len(holes) == 0:
        return bboxes

    # Create a blank mask for holes
    hole_mask = np.zeros(image_shape, dtype=np.uint8)

    # Fill in the holes on the mask
    for hole in holes:
        x_min, y_min, x_max, y_max = hole.astype(int)
        hole_mask[y_min:y_max, x_min:x_max] = 1

    # Vectorized calculation
    bboxes_int = bboxes.astype(int)
    x_min, y_min, x_max, y_max = bboxes_int[:, 0], bboxes_int[:, 1], bboxes_int[:, 2], bboxes_int[:, 3]

    # Calculate box areas
    box_areas = (x_max - x_min) * (y_max - y_min)

    # Create a mask of the same shape as bboxes
    mask = np.zeros(len(bboxes), dtype=bool)

    for i in range(len(bboxes)):
        intersection_area = np.sum(hole_mask[y_min[i] : y_max[i], x_min[i] : x_max[i]])
        remaining_area = box_areas[i] - intersection_area
        visibility_ratio = 1 - (intersection_area / box_areas[i])
        mask[i] = (remaining_area >= min_area) and (visibility_ratio >= min_visibility)

    return bboxes[mask]


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


@handle_empty_array
def mask_dropout_bboxes(
    bboxes: np.ndarray,
    dropout_mask: np.ndarray,
    image_shape: tuple[int, int],
    min_area: float,
    min_visibility: float,
) -> np.ndarray:
    """Filter out bounding boxes based on their intersection with the dropout mask.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (N, 4+) in format [x_min, y_min, x_max, y_max, ...].
        dropout_mask (np.ndarray): Boolean mask of shape (height, width) where True values indicate dropped out regions.
        image_shape (Tuple[int, int]): The shape of the original image as (height, width).
        min_area (float): Minimum area of the bounding box to be kept.
        min_visibility (float): Minimum visibility ratio of the bounding box to be kept.

    Returns:
        np.ndarray: Filtered array of bounding boxes.
    """
    height, width = image_shape

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
    visible_areas = np.sum(box_masks & ~dropout_mask.squeeze(), axis=(1, 2))

    # Calculate visibility ratio (visible area / total box area)
    visibility_ratio = visible_areas / box_areas

    # Create a boolean mask for boxes to keep
    keep_mask = (visible_areas >= min_area) & (visibility_ratio >= min_visibility)

    return bboxes[keep_mask]


@handle_empty_array
def mask_dropout_keypoints(keypoints: np.ndarray, dropout_mask: np.ndarray) -> np.ndarray:
    keep_indices = np.array([not dropout_mask[int(kp[1]), int(kp[0])] for kp in keypoints])
    return keypoints[keep_indices]


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
