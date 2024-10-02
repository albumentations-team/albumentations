from __future__ import annotations

import numpy as np
from albucore import MAX_VALUES_BY_DTYPE, is_grayscale_image, preserve_channel_dim
from typing_extensions import Literal

from albumentations import random_utils
from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.types import MONO_CHANNEL_DIMENSIONS, ColorType

__all__ = ["cutout", "channel_dropout", "filter_keypoints_in_holes", "generate_random_fill", "filter_bboxes_by_holes"]


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
    random_state: np.random.RandomState | None,
) -> np.ndarray:
    """Generate a random fill array based on the given dtype and target shape.

    This function creates a numpy array filled with random values. The range and type of these values
    depend on the input dtype. For integer dtypes, it generates random integers. For floating-point
    dtypes, it generates random floats.

    Args:
        dtype (np.dtype): The data type of the array to be generated.
        shape (tuple[int, ...]): The shape of the array to be generated.
        random_state (np.random.RandomState | None): The random state to use for generating values.
            If None, the default numpy random state is used.

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
        return random_utils.randint(0, max_value + 1, size=shape, dtype=dtype, random_state=random_state)
    if np.issubdtype(dtype, np.floating):
        return random_utils.uniform(0, max_value, size=shape, random_state=random_state).astype(dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


def cutout(
    img: np.ndarray,
    holes: np.ndarray,
    fill_value: ColorType | Literal["random"],
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """Apply cutout augmentation to the image by cutting out holes and filling them
    with either a given value or random noise.

    Args:
        img (np.ndarray): The image to augment. Can be a 2D (grayscale) or 3D (color) array.
        holes (np.ndarray): An array of holes with shape (num_holes, 4).
            Each hole is represented as [x1, y1, x2, y2].
        fill_value (Union[ColorType, Literal["random"]], optional): The fill value to use for the holes.
            Can be a single integer, a tuple or list of numbers for multichannel images,
            or the string "random" to fill with random noise.
        random_state (np.random.RandomState | None, optional): The random state to use for generating
            random fill values. If None, a new random state will be used. Defaults to None.

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
            random_fill = generate_random_fill(img.dtype, shape, random_state)
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
