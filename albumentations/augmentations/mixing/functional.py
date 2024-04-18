import random
from typing import Sequence, Tuple
from warnings import warn

import numpy as np

from albumentations.augmentations.functional import add_weighted
from albumentations.augmentations.utils import clipped, preserve_shape
from albumentations.core.types import MONO_SHAPE_LENGTH, MULTI_SHAPE_LENGTH


@clipped
@preserve_shape
def mix_arrays(array1: np.ndarray, array2: np.ndarray, mix_coef: float) -> np.ndarray:
    warn(
        "The function 'mix_arrays' is deprecated."
        "Please use 'add_weighted(array1, mix_coef, array_2, 1 - mix_coef)' instead",
        DeprecationWarning,
        stacklevel=2,
    )

    return add_weighted(array1, mix_coef, array2, 1 - mix_coef)


def generate_random_coordinates(
    small_images: Sequence[np.ndarray],
    canvas_shape: Tuple[int, int],
) -> Sequence[Tuple[int, int, int, int]]:
    """Generate random coordinates for placing small images on a larger canvas.

    Parameters:
        small_images (list of np.ndarray): List of small image arrays.
        canvas_shape (tuple): Shape of the canvas (height, width).

    Returns:
        list of tuples: Each tuple contains (x_min, y_min, x_max, y_max) coordinates for embedding an image.
    """
    coordinates = []
    canvas_height, canvas_width = canvas_shape

    for img in small_images:
        img_height, img_width = img.shape[:2]

        # Generate random start positions where the entire image will fit on the canvas
        x_min = random.randint(0, canvas_width - img_width)
        y_min = random.randint(0, canvas_height - img_height)

        # Calculate the end positions based on the image size
        x_max = x_min + img_width
        y_max = y_min + img_height

        # Append the bounding box coordinates
        coordinates.append((x_min, y_min, x_max, y_max))

    return coordinates


def blend_images_with_mask(
    original_image: np.ndarray,
    target_image: np.ndarray,
    mask: np.ndarray,
    start_x: int,
    start_y: int,
) -> np.ndarray:
    """Blend a smaller target image onto a larger original image using a binary mask, with the target image placed
    at a specified position defined by start_x and start_y in the original image.

    Parameters:
        original_image (np.ndarray): The original image array (height x width x channels).
        target_image (np.ndarray): The smaller target image array (height x width x channels).
        mask (np.ndarray): The binary mask array for the target image (height x width), where 1 indicates
                           the pixel of the target image to use.
        start_x (int): The x-coordinate of the top-left corner where the target image is placed in the original.
        start_y (int): The y-coordinate of the top-left corner where the target image is placed in the original.

    Returns:
        np.ndarray: The blended image.
    """
    # Validate dimensions
    if (
        start_x < 0
        or start_y < 0
        or (start_x + target_image.shape[1] > original_image.shape[1])
        or (start_y + target_image.shape[0] > original_image.shape[0])
    ):
        msg = "Target image and mask must fit within the dimensions of the original image"
        raise ValueError(msg)

    # Ensure mask is binary and expand to three dimensions if necessary
    mask = np.clip(mask, 0, 1)
    if len(mask.shape) == MONO_SHAPE_LENGTH and target_image.shape[2] == MULTI_SHAPE_LENGTH:
        mask = mask[:, :, np.newaxis]

    mask = np.repeat(mask, target_image.shape[2], axis=2)

    # Create a copy of the original to modify
    blended_image = original_image.copy()

    # Define the region in the original image that will receive the target image
    end_x = start_x + target_image.shape[1]
    end_y = start_y + target_image.shape[0]

    # Apply the mask and add the target image to the specified region
    blended_image[start_y:end_y, start_x:end_x] = (
        blended_image[start_y:end_y, start_x:end_x] * (1 - mask) + target_image * mask
    )

    return blended_image
