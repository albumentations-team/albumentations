from __future__ import annotations

import numpy as np
from albucore import get_num_channels

import albumentations.augmentations.geometric.functional as fgeometric

__all__ = ["copy_and_paste_blend", "create_2x2_mosaic_image"]


def copy_and_paste_blend(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    overlay_mask: np.ndarray,
    offset: tuple[int, int],
) -> np.ndarray:
    y_offset, x_offset = offset

    blended_image = base_image.copy()
    mask_indices = np.where(overlay_mask > 0)
    blended_image[mask_indices[0] + y_offset, mask_indices[1] + x_offset] = overlay_image[
        mask_indices[0],
        mask_indices[1],
    ]
    return blended_image


def resize_and_pad_for_2x2_mosaic(
    four_images: list[np.ndarray],
    target_h: int,
    target_w: int,
    num_channels: int,
    fill: tuple[float, ...] | float,
    interpolation: int,
    keep_aspect_ratio: bool,
) -> list[np.ndarray]:
    """Resize and pad (if necessary) images for a 2x2 mosaic.

    This function resizes 4 images so that their shape match the desired target shape
    (which is half the shape of the uncropped 2x2 mosaic image).
    It will pad the necessary areas of each image if resizing preserves image aspect ratio.
    The 1st image will be resized (and padded) to be placed in the top-left quadrant, the 2nd image
    in the top-right quadrant, the 3rd image in the bottom-left quadrant, and the 4th image in the
    bottom-right quadrant of the uncropped 2x2 mosaic image.

    Args:
        four_images (list[np.ndarray]): List of four images with shape (H, W) or (H, W, C).
        target_h (int): The height of each resized image.
        target_w (int): The width of each resized image.
        num_channels (int): The number of channels of each (resized) image.
        fill (tuple[float, ...] | float): The value used to fill the padding zone of each image
                                          (in case aspect ratio preserving resizing).
        interpolation (int): Interpolation mode for handling resizing.
                            Use cv2.INTER_NEAREST, cv2.INTER_LINEAR, etc.
        keep_aspect_ratio (bool): Whether to resize while preserving the aspect ratio.

    Returns:
        list[np.ndarray]: List of four resized (and padded if necessary) images with shape
                          (`target_h`, `target_w`) or (`target_h`, `target_w`, C).

    Notes: In case of `keep_aspect_ratio` set to `True`:
        - The processing is expected to be longer.
        - The first image will be padded (if needed) along its top & left borders.
        - The second image will be padded (if needed) along its top & right borders.
        - The third image will be padded (if needed) along its bottom & left borders.
        - The fourth image will be padded (if needed) along its bottom & right borders.

    Example:
        >>> images = [np.full((100, 100, 3), fill_value=(255.0, 255.0, 255.0), dtype=np.uint8),
        ...           np.full((200, 100, 3), fill_value=(255.0, 0.0, 0.0), dtype=np.uint8),
        ...           np.full((50, 100, 3), fill_value=(0.0, 255.0, 0.0), dtype=np.uint8),
        ...           np.full((50, 50, 3), fill_value=(0.0, 0.0, 255.0), dtype=np.uint8)]
        >>> resized_images = resize_and_pad_for_2x2_mosaic(
        ...     four_images=images,
        ...     target_h=100,
        ...     target_w=100,
        ...     num_channels=3,
        ...     fill=(0.0, 0.0, 0.0),
        ...     interpolation=cv2.INTER_NEAREST,
        ...     keep_aspect_ratio=True
        ... )
    """
    quadrant_shape = (target_h, target_w, num_channels) if num_channels > 1 else (target_h, target_w)
    fill_value = fgeometric.extend_value(fill, num_channels)

    # Resize images to match mosaic quadrant size
    if not keep_aspect_ratio:
        # Direct resize (distorts aspect ratio)
        resized_images = [fgeometric.resize(img, (target_h, target_w), interpolation) for img in four_images]

    else:
        # Preserve aspect ratio: scale to fit within (target_h, target_w), then pad
        resized_images = []
        for idx, img in enumerate(four_images):
            h, w = img.shape[:2]
            scale = min(target_h / h, target_w / w)
            resized_img = fgeometric.scale(img, scale, interpolation)
            resized_h, resized_w = resized_img.shape[:2]

            # Create a blank canvas and place the resized image in the good position
            canvas = np.full(
                quadrant_shape,
                fill_value,
                dtype=img.dtype,
            )  # We checked that all images have same type before
            if idx == 0:  # Top-left quadrant -> resized image should be aligned on bottom-right corner
                top = max(0, target_h - resized_h)
                left = max(0, target_w - resized_w)
            elif idx == 1:  # Top-right quadrant -> resized image should be aligned on bottom-left corner
                top = max(0, target_h - resized_h)
                left = 0
            elif idx == 2:  # Bottom-left quadrant -> resized image should be aligned on top-right corner
                top = 0
                left = max(0, target_w - resized_w)
            else:  # Bottom-right quadrant -> resized image should be aligned on top-left corner
                top = 0
                left = 0
            canvas[top : top + resized_h, left : left + resized_w] = resized_img

            resized_images.append(canvas)

    return resized_images


def create_2x2_mosaic_image(
    four_images: list[np.ndarray],
    center_pt: tuple[int, int],
    mosaic_size: tuple[int, int],
    keep_aspect_ratio: bool,
    interpolation: int,
    fill: tuple[float, ...] | float,
) -> np.ndarray:
    """Creates a 2x2 mosaic image.

    This function first creates a mosaic made from 4 images with double shape `mosaic_size`,
    before cropping an area centered on `center_pt` with the desired  `mosaic_size`.
    It resizes and pads as necessary the 4 images, and places them in the corresponding quadrant of
    the uncropped mosaic in this order: top-left, top-right, bottom-left, and bottom-right.

    Args:
        four_images (list[np.ndarray]): List of four images with shape (H, W) or (H, W, C).
        center_pt (tuple[int, int]): The (x,y) position of the center point around which
                                     to crop the final mosaic.
        mosaic_size (tuple[int, int]): The (height, width) of the final cropped mosaic
                                       (and also the one of each quadrant).
        keep_aspect_ratio (bool): Whether to resize while preserving the aspect ratio.
        interpolation (int): Interpolation mode for handling resizing.
                            Use cv2.INTER_NEAREST, cv2.INTER_LINEAR, etc.
        fill (tuple[float, ...] | float): The value used to fill the padding zone of each image
                                          (in case aspect ratio preserving resizing).

    Returns:
        np.ndarray: The mosaic image with shape (`mosaic_size[0]`, `mosaic_size[1]`)
                    or (`mosaic_size[0]`, `mosaic_size[1]`, C).

    Notes: In case of `keep_aspect_ratio` set to `True`, the processing is expected to be longer.

    Example:
        >>> images = [np.full((100, 100, 3), fill_value=(255.0, 255.0, 255.0), dtype=np.uint8),
        ...           np.full((200, 100, 3), fill_value=(255.0, 0.0, 0.0), dtype=np.uint8),
        ...           np.full((50, 100, 3), fill_value=(0.0, 255.0, 0.0), dtype=np.uint8),
        ...           np.full((50, 50, 3), fill_value=(0.0, 0.0, 255.0), dtype=np.uint8)]
        >>> mosaic = create_2x2_mosaic_image(
        ...     four_images=images,
        ...     center_pt=(100, 100),
        ...     mosaic_size=(100, 100),
        ...     keep_aspect_ratio=True,
        ...     interpolation=cv2.INTER_NEAREST,
        ...     fill=(0.0, 0.0, 0.0)
        ... )
    """
    target_h, target_w = mosaic_size
    num_channels = get_num_channels(four_images[0])
    mosaic_shape = (2 * target_h, 2 * target_w, num_channels) if num_channels > 1 else (2 * target_h, 2 * target_w)

    # Resize and pad each image for the 2x2 mosaic
    resized_images = resize_and_pad_for_2x2_mosaic(
        four_images,
        target_h=target_h,
        target_w=target_w,
        num_channels=num_channels,
        fill=fill,
        interpolation=interpolation,
        keep_aspect_ratio=keep_aspect_ratio,
    )

    # Create 2x2 mosaic
    mosaic = np.zeros(mosaic_shape, dtype=resized_images[0].dtype)
    # Place images in quadrants
    mosaic[:target_h, :target_w] = resized_images[0]  # Top-left quadrant
    mosaic[:target_h, target_w:] = resized_images[1]  # Top-right quadrant
    mosaic[target_h:, :target_w] = resized_images[2]  # Bottom-left quadrant
    mosaic[target_h:, target_w:] = resized_images[3]  # Bottom-right quadrant

    # Extract the final (target_h, target_w) crop centered at center_pt
    x1, y1 = max(0, center_pt[0] - target_w // 2), max(0, center_pt[1] - target_h // 2)
    x2, y2 = x1 + target_w, y1 + target_h
    # Ensure the crop is within bounds
    x1, x2 = min(x1, target_w), min(x2, 2 * target_w)
    y1, y2 = min(y1, target_h), min(y2, 2 * target_h)

    return mosaic[y1:y2, x1:x2]
