"""Functional implementations for image mixing operations.

This module provides utility functions for blending and combining images,
such as copy-and-paste operations with masking.
"""

from __future__ import annotations

import numpy as np
from albucore import get_num_channels

import albumentations.augmentations.crops.functional as fcrops
import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes

__all__ = [
    "copy_and_paste_blend",
    "create_2x2_mosaic_image",
    "get_2x2_mosaic_bboxes",
    "get_2x2_mosaic_center_crop",
    "get_2x2_mosaic_keypoints",
    "get_2x2_mosaic_quadrant_offset",
    "get_2x2_mosaic_quadrant_pad_after_resize",
    "resize_and_pad_for_2x2_mosaic",
]


def copy_and_paste_blend(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    overlay_mask: np.ndarray,
    offset: tuple[int, int],
) -> np.ndarray:
    """Blend images by copying pixels from an overlay image to a base image using a mask.

    This function copies pixels from the overlay image to the base image only where
    the mask has non-zero values. The overlay is placed at the specified offset
    from the top-left corner of the base image.

    Args:
        base_image (np.ndarray): The destination image that will be modified.
        overlay_image (np.ndarray): The source image containing pixels to copy.
        overlay_mask (np.ndarray): Binary mask indicating which pixels to copy from the overlay.
            Pixels are copied where mask > 0.
        offset (tuple[int, int]): The (y, x) offset specifying where to place the
            top-left corner of the overlay relative to the base image.

    Returns:
        np.ndarray: The blended image with the overlay applied to the base image.

    """
    y_offset, x_offset = offset

    blended_image = base_image.copy()
    mask_indices = np.where(overlay_mask > 0)
    blended_image[mask_indices[0] + y_offset, mask_indices[1] + x_offset] = overlay_image[
        mask_indices[0],
        mask_indices[1],
    ]
    return blended_image


def get_2x2_mosaic_quadrant_pad_after_resize(
    target_shape: tuple[int, int],
    resized_shape: tuple[int, int],
    quadrant_idx: int,
) -> tuple[int, int]:
    """Gets the padding offset in pixels to apply when padding a quadrant of a 2x2 mosaic image
    that has been resized while preserving the aspect ratio.

    Args:
        target_shape (tuple[int, int]): The (height, width) of the quadrant after padding.
        resized_shape (tuple[int, int]): The current (height, width) of the quadrant (resized).
        quadrant_idx (int): The quadran index (0 for top-left, 1 for top-right, 2 for bottom-left, and
                            3 for bottom-right).

    Returns:
        tuple[int, int]: The (x, y) padding offset to apply to coordinates (bboxes, keypoints...).
    """
    target_h, target_w = target_shape
    resized_h, resized_w = resized_shape

    # Bottom-right quadrant (idx 3) -> resized image should be aligned on top-left corner, no effect
    x_pad_offset, y_pad_offset = 0, 0
    if quadrant_idx == 0:  # Top-left quadrant -> resized image should be aligned on bottom-right corner
        y_pad_offset = max(0, target_h - resized_h)
        x_pad_offset = max(0, target_w - resized_w)
    elif quadrant_idx == 1:  # Top-right quadrant -> resized image should be aligned on bottom-left corner
        y_pad_offset = max(0, target_h - resized_h)
    elif quadrant_idx == 2:  # Bottom-left quadrant -> resized image should be aligned on top-right corner
        x_pad_offset = max(0, target_w - resized_w)

    return (x_pad_offset, y_pad_offset)


def get_2x2_mosaic_quadrant_offset(
    target_shape: tuple[int, int],
    quadrant_idx: int,
) -> tuple[int, int]:
    """Gets the offset in pixels to apply when placing the quadrant in a 2x2 mosaic image
    of shape 2 * target_shape.

    Args:
        target_shape (tuple[int, int]): The (height, width) of the quadrant.
        quadrant_idx (int): The quadran index (0 for top-left, 1 for top-right, 2 for bottom-left, and
                            3 for bottom-right).

    Returns:
        tuple[int, int]: The (x, y) offset to apply to coordinates (bboxes, keypoints...).
    """
    target_h, target_w = target_shape

    # The mosaic has shape (2 * target_h, 2 * target_w), so coordinates need to be translated
    # left/down by target_h and/or target_w depending on the quadrant.
    x_offset, y_offset = 0, 0  # No effect for top-left quadrant (idx 0).
    if quadrant_idx == 1:  # Top-right quadrant
        x_offset = target_w
    elif quadrant_idx == 2:  # Bottom-left quadrant
        y_offset = target_h
    elif quadrant_idx == 3:  # Bottom-right quadrant
        x_offset = target_w
        y_offset = target_h

    return (x_offset, y_offset)


def get_2x2_mosaic_center_crop(
    center_pt: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Gets the cropping area (shape `target_shape`) around a center point.

    Args:
        center_pt (tuple[int, int]): The center point coordinates (x, y).
        target_shape (tuple[int, int]): The (height, width) of the cropping area.

    Returns:
        tuple[int, int, int, int]: The cropping area coordinates
                                   (x_min, x_min + width, y_min, y_min + height).

    Notes: When cropping, take the area from [y_min:y_min + height, x_min:x_min + width].
    """
    target_h, target_w = target_shape

    # Extract the final (target_h, target_w) crop centered at center_pt.
    # Make sure it does not become negative. It cannot go beyond (2 * target_h - 1, 2 * target_w - 1).
    # We already ensured center_pt[0] is in [0, target_w - 1].
    # We already ensured center_pt[1] is in [0, target_h - 1].
    x1, y1 = max(0, center_pt[0] - target_w // 2), max(0, center_pt[1] - target_h // 2)
    x2, y2 = x1 + target_w, y1 + target_h

    return (x1, y1, x2, y2)


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

            x_pad_offset, y_pad_offset = get_2x2_mosaic_quadrant_pad_after_resize(
                target_shape=(target_h, target_w),
                resized_shape=(resized_h, resized_w),
                quadrant_idx=idx,
            )

            canvas[y_pad_offset : y_pad_offset + resized_h, x_pad_offset : x_pad_offset + resized_w] = resized_img

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
        np.ndarray: The final cropped mosaic image with shape (`mosaic_size[0]`, `mosaic_size[1]`)
                    or (`mosaic_size[0]`, `mosaic_size[1]`, C).

    Notes: In case of `keep_aspect_ratio` set to `True`, the processing is expected to be longer.

    Example:
        >>> images = [np.full((100, 100, 3), fill_value=(255.0, 255.0, 255.0), dtype=np.uint8),
        ...           np.full((100, 200, 3), fill_value=(255.0, 0.0, 0.0), dtype=np.uint8),
        ...           np.full((100, 50, 3), fill_value=(0.0, 255.0, 0.0), dtype=np.uint8),
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

    x1, y1, x2, y2 = get_2x2_mosaic_center_crop(
        center_pt=center_pt,
        target_shape=(target_h, target_w),
    )

    return mosaic[y1:y2, x1:x2]


def get_2x2_mosaic_bboxes(
    all_bboxes: list[np.ndarray | None],
    all_img_shapes: list[tuple[int, int]],
    center_pt: tuple[int, int],
    mosaic_size: tuple[int, int],
    keep_aspect_ratio: bool,
) -> np.ndarray:
    """Gets all bounding boxes from a 2x2 cropped mosaic image.

    This function applies the necessary transformations to the 4 images' bounding boxes when creating
    a mosaic image through the function `A.augmentations.mixing.functional.create_2x2_mosaic_image()`.

    Args:
        all_bboxes (list[np.ndarray | None]): List of all bounding boxes (shape (N, 4+)) from the 4 images.
        all_img_shapes (list[tuple[int, int]]): List of all 4 images (height, width).
        center_pt (tuple[int, int]): The (x,y) position of the center point around which
                                     the final mosaic has been cropped.
        mosaic_size (tuple[int, int]): The (height, width) of the final cropped mosaic
                                       (and also the one of each quadrant).
        keep_aspect_ratio (bool): Whether resizing preserved the aspect ratio when creating the mosaic.

    Returns:
        np.ndarray: The bounding boxes of the final cropped mosaic image (shape (M, 4+), M <= N).

    Notes:
        Bounding boxes that fall completely outside the crop area will be removed.
        Bounding boxes that partially overlap with the crop area will be adjusted to fit within it.

    Example:
        >>> bboxes = [np.array([[0, 0, 0.2, 0.2], [0.9, 0.9, 1.0, 1.0], [0.25, 0.25, 0.75, 0.75]])] * 4
        >>> mosaic_bboxes = get_2x2_mosaic_bboxes(
        ...     all_bboxes=bboxes,
        ...     all_img_shapes=[(100, 100), (100, 200), (100, 50), (50, 50)],
        ...     center_pt=(100, 100),
        ...     mosaic_size=(100, 100),
        ...     keep_aspect_ratio=True
        ... )
    """
    target_h, target_w = mosaic_size

    mosaic_bboxes = []
    for idx, bboxes in enumerate(all_bboxes):
        if bboxes is None:
            continue

        # Bounding box coordinates are scale invariant.
        # Only the padding operation has an effect.
        # Furthermore, the fourth image has been aligned on top-left corner
        # when padding, so bbox coordinates are not changed for this one.
        if keep_aspect_ratio:
            h, w = all_img_shapes[idx]
            scale = min(target_h / h, target_w / w)
            resized_h, resized_w = int(h * scale), int(w * scale)

            x_pad_offset, y_pad_offset = get_2x2_mosaic_quadrant_pad_after_resize(
                target_shape=(target_h, target_w),
                resized_shape=(resized_h, resized_w),
                quadrant_idx=idx,
            )

            denorm_bboxes = denormalize_bboxes(bboxes.copy().astype(np.float32), (resized_h, resized_w))
            denorm_bboxes[:, [0, 2]] += x_pad_offset
            denorm_bboxes[:, [1, 3]] += y_pad_offset
        else:
            denorm_bboxes = denormalize_bboxes(bboxes.copy().astype(np.float32), (target_h, target_w))

        # Handle effect of placing each image in its corresponding quadrant.
        x_offset, y_offset = get_2x2_mosaic_quadrant_offset(
            target_shape=(target_h, target_w),
            quadrant_idx=idx,
        )

        denorm_bboxes[:, [0, 2]] += x_offset
        denorm_bboxes[:, [1, 3]] += y_offset

        mosaic_bboxes.append(denorm_bboxes)

    mosaic_bboxes_np = np.concatenate(mosaic_bboxes, axis=0)
    if mosaic_bboxes_np.size == 0:
        return mosaic_bboxes_np

    # Handle center crop effect
    crop_coords = get_2x2_mosaic_center_crop(
        center_pt=center_pt,
        target_shape=(target_h, target_w),
    )
    crop_mosaic_bboxes = fcrops.crop_bboxes_by_coords(
        mosaic_bboxes_np,
        crop_coords=crop_coords,
        image_shape=(2 * target_h, 2 * target_w),
        normalized_input=False,
    )

    norm_mosaic_bboxes = normalize_bboxes(crop_mosaic_bboxes, (target_h, target_w))

    # Filter bboxes falling completely outside or partially overlapping
    norm_mosaic_bboxes[:, :4] = np.clip(norm_mosaic_bboxes[:, :4], 0.0, 1.0)
    keep_mosaic_bboxes = [bbox for bbox in norm_mosaic_bboxes if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 0]

    return np.array(keep_mosaic_bboxes)


def get_2x2_mosaic_keypoints(
    all_keypoints: list[np.ndarray | None],
    all_img_shapes: list[tuple[int, int]],
    center_pt: tuple[int, int],
    mosaic_size: tuple[int, int],
) -> np.ndarray:
    """Gets all keypoints from a 2x2 cropped mosaic image.

    This function applies the necessary transformations to the 4 images' keypoints when creating
    a mosaic image through the function `A.augmentations.mixing.functional.create_2x2_mosaic_image()`.

    Args:
        all_keypoints (list[np.ndarray | None]): List of all keypoints (shape (N, 4+)) from the 4 images.
        all_img_shapes (list[tuple[int, int]]): List of all 4 images (height, width).
        center_pt (tuple[int, int]): The (x,y) position of the center point around which
                                     the final mosaic has been cropped.
        mosaic_size (tuple[int, int]): The (height, width) of the final cropped mosaic
                                       (and also the one of each quadrant).

    Returns:
        np.ndarray: The keypoints of the final cropped mosaic image (shape (M, 4+), M <= N).

    Notes:
        Keypoints that fall completely outside the crop area will be removed.
        Keypoints scale cannot be strictly preserved when images have been resized without preserving
            the aspect ratio, as they become ellipses instead of circles.

    Example:
        >>> keypoints = [
        ...     np.array([[0, 0, 45, 2.0], [99, 99, 45, 2.0], [80, 80, 45, 2.0]]),
        ...     np.array([[0, 80, 45, 2.0], [199, 0, 45, 2.0], [40, 80, 45, 2.0]]),
        ...     np.array([[0, 99, 45, 2.0], [49, 20, 45, 2.0], [40, 20, 45, 2.0]]),
        ...     np.array([[15, 15, 45, 2.0], [49, 49, 45, 2.0]])
        ... ]
        >>> mosaic_keypoints = get_2x2_mosaic_keypoints(
        ...     all_keypoints=keypoints,
        ...     all_img_shapes=[(100, 100), (100, 200), (100, 50), (50, 50)],
        ...     center_pt=(100, 100),
        ...     mosaic_size=(100, 100)
        ... )
    """
    target_h, target_w = mosaic_size

    mosaic_keypoints = []
    for idx, kps in enumerate(all_keypoints):
        if kps is None:
            continue

        # Handle effect of resize and pad each image.
        # For a resizing without keeping aspect ratio, it is not really accurate,
        # but anyway in that case the keypoint is not a circle anymore, but an ellipse.
        h, w = all_img_shapes[idx]
        scale = min(target_h / h, target_w / w)
        resized_h, resized_w = int(h * scale), int(w * scale)

        x_pad_offset, y_pad_offset = get_2x2_mosaic_quadrant_pad_after_resize(
            target_shape=(target_h, target_w),
            resized_shape=(resized_h, resized_w),
            quadrant_idx=idx,
        )

        kps[:, [0, 1, 2, 4]] *= scale
        keypoints = fcrops.crop_and_pad_keypoints(
            kps,
            pad_params=(y_pad_offset, y_pad_offset + resized_h, x_pad_offset, x_pad_offset + resized_w),
        )

        # Handle effect of placing each image in its corresponding quadrant.
        x_offset, y_offset = get_2x2_mosaic_quadrant_offset(
            target_shape=(target_h, target_w),
            quadrant_idx=idx,
        )

        keypoints[:, 0] += x_offset
        keypoints[:, 1] += y_offset

        mosaic_keypoints.append(keypoints)

    mosaic_keypoints_np = np.concatenate(mosaic_keypoints, axis=0)
    if mosaic_keypoints_np.size == 0:
        return mosaic_keypoints_np

    # Handle center crop effect
    crop_coords = get_2x2_mosaic_center_crop(
        center_pt=center_pt,
        target_shape=(target_h, target_w),
    )

    crop_mosaic_keypoints = fcrops.crop_and_pad_keypoints(
        mosaic_keypoints_np,
        crop_params=crop_coords,
    )

    # Filter keypoints falling completely outside or disappearing
    keep_mosaic_keypoints = [
        keypoints
        for keypoints in crop_mosaic_keypoints
        if keypoints[0] >= 0
        and keypoints[0] < target_w
        and keypoints[1] >= 0
        and keypoints[1] < target_h
        and keypoints[4] > 0
    ]
    return np.array(keep_mosaic_keypoints)
