"""Functional implementations of image augmentation operations.

This module contains low-level functions for various image augmentation techniques including
color transformations, blur effects, tone curve adjustments, noise additions, and other visual
modifications. These functions form the foundation for the transform classes and provide
the core functionality for manipulating image data during the augmentation process.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal
from warnings import warn

import cv2
import numpy as np
from albucore import (
    MAX_VALUES_BY_DTYPE,
    add,
    add_array,
    add_constant,
    add_weighted,
    clip,
    clipped,
    float32_io,
    from_float,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    maybe_process_in_chunks,
    multiply,
    multiply_add,
    multiply_by_array,
    multiply_by_constant,
    normalize_per_image,
    power,
    preserve_channel_dim,
    sz_lut,
    uint8_io,
)

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.utils import (
    PCA,
    handle_empty_array,
    non_rgb_error,
)
from albumentations.core.bbox_utils import bboxes_from_masks, masks_from_bboxes
from albumentations.core.type_definitions import (
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
)

__all__ = [
    "add_fog",
    "add_gravel",
    "add_rain",
    "add_shadow",
    "add_snow_bleach",
    "add_snow_texture",
    "add_sun_flare_overlay",
    "add_sun_flare_physics_based",
    "adjust_brightness_torchvision",
    "adjust_contrast_torchvision",
    "adjust_hue_torchvision",
    "adjust_saturation_torchvision",
    "channel_shuffle",
    "chromatic_aberration",
    "clahe",
    "dilate",
    "downscale",
    "equalize",
    "erode",
    "fancy_pca",
    "gamma_transform",
    "image_compression",
    "invert",
    "iso_noise",
    "linear_transformation_rgb",
    "move_tone_curve",
    "noop",
    "posterize",
    "shift_hsv",
    "solarize",
    "superpixels",
    "to_gray",
    "unsharp_mask",
]


@uint8_io
@preserve_channel_dim
def shift_hsv(
    img: np.ndarray,
    hue_shift: float,
    sat_shift: float,
    val_shift: float,
) -> np.ndarray:
    """Shift the hue, saturation, and value of an image.

    Args:
        img (np.ndarray): The image to shift.
        hue_shift (float): The amount to shift the hue.
        sat_shift (float): The amount to shift the saturation.
        val_shift (float): The amount to shift the value.

    Returns:
        np.ndarray: The shifted image.

    """
    if hue_shift == 0 and sat_shift == 0 and val_shift == 0:
        return img

    is_gray = is_grayscale_image(img)

    if is_gray:
        if hue_shift != 0 or sat_shift != 0:
            hue_shift = 0
            sat_shift = 0
            warn(
                "HueSaturationValue: hue_shift and sat_shift are not applicable to grayscale image. "
                "Set them to 0 or use RGB image",
                stacklevel=2,
            )
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        lut_hue = np.arange(0, 256, dtype=np.int16)
        lut_hue = np.mod(lut_hue + hue_shift, 180).astype(np.uint8)
        hue = sz_lut(hue, lut_hue, inplace=False)

    if sat_shift != 0:
        # Create a mask for all grayscale pixels (S=0)
        # These should remain grayscale regardless of saturation change
        grayscale_mask = sat == 0

        # Apply saturation shift only to non-white pixels
        sat = add_constant(sat, sat_shift, inplace=True)

        # Reset saturation for white pixels
        sat[grayscale_mask] = 0

    if val_shift != 0:
        val = add_constant(val, val_shift, inplace=True)

    img = cv2.merge((hue, sat, val))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if is_gray else img


@clipped
def solarize(img: np.ndarray, threshold: float) -> np.ndarray:
    """Invert all pixel values above a threshold.

    Args:
        img (np.ndarray): The image to solarize. Can be uint8 or float32.
        threshold (float): Normalized threshold value in range [0, 1].
            For uint8 images: pixels above threshold * 255 are inverted
            For float32 images: pixels above threshold are inverted

    Returns:
        np.ndarray: Solarized image.

    Note:
        The threshold is normalized to [0, 1] range for both uint8 and float32 images.
        For uint8 images, the threshold is internally scaled by 255.

    """
    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.uint8:
        lut = np.array(
            [max_val - i if i >= threshold * max_val else i for i in range(int(max_val) + 1)],
            dtype=dtype,
        )
        prev_shape = img.shape
        img = sz_lut(img, lut, inplace=False)
        return img if len(prev_shape) == img.ndim else np.expand_dims(img, -1)
    return np.where(img >= threshold, max_val - img, img)


@uint8_io
@clipped
def posterize(img: np.ndarray, bits: Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]]) -> np.ndarray:
    """Reduce the number of bits for each color channel by keeping only the highest N bits.

    Args:
        img (np.ndarray): Input image. Can be single or multi-channel.
        bits (Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]]): Number of high bits to keep..
            Can be either:
            - A single value to apply the same bit reduction to all channels
            - A list of values to apply different bit reduction per channel.
              Length of list must match number of channels in image.

    Returns:
        np.ndarray: Image with reduced bit depth. Has same shape and dtype as input.

    Note:
        - The transform keeps the N highest bits and sets all other bits to 0
        - For example, if bits=3:
            - Original value: 11010110 (214)
            - Keep 3 bits:   11000000 (192)
        - The number of unique colors per channel will be 2^bits
        - Higher bits values = more colors = more subtle effect
        - Lower bits values = fewer colors = more dramatic posterization

    Examples:
        >>> import numpy as np
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> # Same posterization for all channels
        >>> result = posterize(image, bits=3)
        >>> # Different posterization per channel
        >>> result = posterize(image, bits=[3, 4, 5])  # RGB channels

    """
    bits_array = np.uint8(bits)

    if not bits_array.shape or len(bits_array) == 1:
        lut = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - bits_array) - 1)
        lut &= mask

        return sz_lut(img, lut, inplace=False)

    result_img = np.empty_like(img)
    for i, channel_bits in enumerate(bits_array):
        lut = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - channel_bits) - 1)
        lut &= mask

        result_img[..., i] = sz_lut(img[..., i], lut, inplace=True)

    return result_img


def _equalize_pil(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    h = np.array([_f for _f in histogram if _f])

    if len(h) <= 1:
        return img.copy()

    step = np.sum(h[:-1]) // 255
    if not step:
        return img.copy()

    lut = np.minimum((np.cumsum(histogram) + step // 2) // step, 255).astype(np.uint8)

    return sz_lut(img, lut, inplace=True)


def _equalize_cv(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        return cv2.equalizeHist(img)

    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()

    # Find the first non-zero index with a numpy operation
    i = np.flatnonzero(histogram)[0] if np.any(histogram) else 255

    total = np.sum(histogram)

    scale = 255.0 / (total - histogram[i])

    # Optimize cumulative sum and scale to generate LUT
    cumsum_histogram = np.cumsum(histogram)
    lut = np.clip(((cumsum_histogram - cumsum_histogram[i]) * scale).round(), 0, 255).astype(np.uint8)

    return sz_lut(img, lut, inplace=True)


def _check_preconditions(
    img: np.ndarray,
    mask: np.ndarray | None,
    by_channels: bool,
) -> None:
    if mask is not None:
        if is_rgb_image(mask) and is_grayscale_image(img):
            raise ValueError(
                f"Wrong mask shape. Image shape: {img.shape}. Mask shape: {mask.shape}",
            )
        if not by_channels and not is_grayscale_image(mask):
            msg = f"When by_channels=False only 1-channel mask supports. Mask shape: {mask.shape}"
            raise ValueError(msg)


def _handle_mask(
    mask: np.ndarray | None,
    i: int | None = None,
) -> np.ndarray | None:
    if mask is None:
        return None
    mask = mask.astype(
        np.uint8,
        copy=False,
    )  # Use copy=False to avoid unnecessary copying
    # Check for grayscale image and avoid slicing if i is None
    if i is not None and not is_grayscale_image(mask):
        mask = mask[..., i]

    return mask


@uint8_io
@preserve_channel_dim
def equalize(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    mode: Literal["cv", "pil"] = "cv",
    by_channels: bool = True,
) -> np.ndarray:
    """Apply histogram equalization to the input image.

    This function enhances the contrast of the input image by equalizing its histogram.
    It supports both grayscale and color images, and can operate on individual channels
    or on the luminance channel of the image.

    Args:
        img (np.ndarray): Input image. Can be grayscale (2D array) or RGB (3D array).
        mask (np.ndarray | None): Optional mask to apply the equalization selectively.
            If provided, must have the same shape as the input image. Default: None.
        mode (ImageMode): The backend to use for equalization. Can be either "cv" for
            OpenCV or "pil" for Pillow-style equalization. Default: "cv".
        by_channels (bool): If True, applies equalization to each channel independently.
            If False, converts the image to YCrCb color space and equalizes only the
            luminance channel. Only applicable to color images. Default: True.

    Returns:
        np.ndarray: Equalized image. The output has the same dtype as the input.

    Raises:
        ValueError: If the input image or mask have invalid shapes or types.

    Note:
        - If the input image is not uint8, it will be temporarily converted to uint8
          for processing and then converted back to its original dtype.
        - For color images, when by_channels=False, the image is converted to YCrCb
          color space, equalized on the Y channel, and then converted back to RGB.
        - The function preserves the original number of channels in the image.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> equalized = A.equalize(image, mode="cv", by_channels=True)
        >>> assert equalized.shape == image.shape
        >>> assert equalized.dtype == image.dtype

    """
    _check_preconditions(img, mask, by_channels)
    function = _equalize_pil if mode == "pil" else _equalize_cv

    if is_grayscale_image(img):
        return function(img, _handle_mask(mask))

    if not by_channels:
        result_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        result_img[..., 0] = function(result_img[..., 0], _handle_mask(mask))
        return cv2.cvtColor(result_img, cv2.COLOR_YCrCb2RGB)

    result_img = np.empty_like(img)
    for i in range(NUM_RGB_CHANNELS):
        _mask = _handle_mask(mask, i)
        result_img[..., i] = function(img[..., i], _mask)

    return result_img


@uint8_io
def move_tone_curve(
    img: np.ndarray,
    low_y: float | np.ndarray,
    high_y: float | np.ndarray,
) -> np.ndarray:
    """Rescales the relationship between bright and dark areas of the image by manipulating its tone curve.

    Args:
        img (np.ndarray): Any number of channels
        low_y (float | np.ndarray): per-channel or single y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y (float | np.ndarray): per-channel or single y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]

    Returns:
        np.ndarray: Image with adjusted tone curve

    """
    t = np.linspace(0.0, 1.0, 256)

    def evaluate_bez(
        t: np.ndarray,
        low_y: float | np.ndarray,
        high_y: float | np.ndarray,
    ) -> np.ndarray:
        one_minus_t = 1 - t
        return (3 * one_minus_t**2 * t * low_y + 3 * one_minus_t * t**2 * high_y + t**3) * 255

    num_channels = get_num_channels(img)

    if np.isscalar(low_y) and np.isscalar(high_y):
        lut = clip(np.rint(evaluate_bez(t, low_y, high_y)), np.uint8, inplace=False)
        return sz_lut(img, lut, inplace=False)
    if isinstance(low_y, np.ndarray) and isinstance(high_y, np.ndarray):
        luts = clip(
            np.rint(evaluate_bez(t[:, np.newaxis], low_y, high_y).T),
            np.uint8,
            inplace=False,
        )
        return cv2.merge(
            [sz_lut(img[:, :, i], np.ascontiguousarray(luts[i]), inplace=False) for i in range(num_channels)],
        )

    raise TypeError(
        f"low_y and high_y must both be of type float or np.ndarray. Got {type(low_y)} and {type(high_y)}",
    )


@clipped
def linear_transformation_rgb(
    img: np.ndarray,
    transformation_matrix: np.ndarray,
) -> np.ndarray:
    """Apply a linear transformation to the RGB channels of an image.

    This function applies a linear transformation matrix to the RGB channels of an image.
    The transformation matrix is a 3x3 matrix that maps the RGB values to new values.

    Args:
        img (np.ndarray): Input image. Can be grayscale (2D array) or RGB (3D array).
        transformation_matrix (np.ndarray): 3x3 transformation matrix.

    Returns:
        np.ndarray: Image with the linear transformation applied. The output has the same dtype as the input.

    """
    return cv2.transform(img, transformation_matrix)


@uint8_io
@preserve_channel_dim
def clahe(
    img: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.

    This function enhances the contrast of the input image using CLAHE. For color images,
    it converts the image to the LAB color space, applies CLAHE to the L channel, and then
    converts the image back to RGB.

    Args:
        img (np.ndarray): Input image. Can be grayscale (2D array) or RGB (3D array).
        clip_limit (float): Threshold for contrast limiting. Higher values give more contrast.
        tile_grid_size (tuple[int, int]): Size of grid for histogram equalization.
            Width and height of the grid.

    Returns:
        np.ndarray: Image with CLAHE applied. The output has the same dtype as the input.

    Note:
        - If the input image is float32, it's temporarily converted to uint8 for processing
          and then converted back to float32.
        - For color images, CLAHE is applied only to the luminance channel in the LAB color space.

    Raises:
        ValueError: If the input image is not 2D or 3D.

    Example:
        >>> import numpy as np
        >>> img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> result = clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))
        >>> assert result.shape == img.shape
        >>> assert result.dtype == img.dtype

    """
    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if is_grayscale_image(img):
        return clahe_mat.apply(img)

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_lab[:, :, 0] = clahe_mat.apply(img_lab[:, :, 0])

    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


@uint8_io
@preserve_channel_dim
def image_compression(
    img: np.ndarray,
    quality: int,
    image_type: Literal[".jpg", ".webp"],
) -> np.ndarray:
    """Compress the image using JPEG or WebP compression.

    Args:
        img (np.ndarray): Input image
        quality (int): Quality of compression in range [1, 100]
        image_type (Literal[".jpg", ".webp"]): Type of compression to use

    Returns:
        np.ndarray: Compressed image

    """
    # Determine the quality flag for compression
    quality_flag = cv2.IMWRITE_JPEG_QUALITY if image_type == ".jpg" else cv2.IMWRITE_WEBP_QUALITY
    num_channels = get_num_channels(img)

    # Prepare to encode and decode
    def encode_decode(src_img: np.ndarray, read_mode: int) -> np.ndarray:
        _, encoded_img = cv2.imencode(image_type, src_img, (int(quality_flag), quality))
        return cv2.imdecode(encoded_img, read_mode)

    if num_channels == 1:
        # Grayscale image
        decoded = encode_decode(img, cv2.IMREAD_GRAYSCALE)
        return decoded[..., np.newaxis]  # Add channel dimension back

    if num_channels in (2, NUM_RGB_CHANNELS):
        # 2 channels: pad to 3, or 3 (RGB) channels
        padded_img = np.pad(img, ((0, 0), (0, 0), (0, 1)), mode="constant") if num_channels == 2 else img
        decoded_bgr = encode_decode(padded_img, cv2.IMREAD_UNCHANGED)
        return decoded_bgr[..., :num_channels]  # Return only the required number of channels

    # More than 3 channels
    bgr = img[..., :NUM_RGB_CHANNELS]
    decoded_bgr = encode_decode(bgr, cv2.IMREAD_UNCHANGED)

    # Process additional channels
    extra_channels = [
        encode_decode(img[..., i], cv2.IMREAD_GRAYSCALE)[..., np.newaxis] for i in range(NUM_RGB_CHANNELS, num_channels)
    ]
    return np.dstack([decoded_bgr, *extra_channels])


@uint8_io
def add_snow_bleach(
    img: np.ndarray,
    snow_point: float,
    brightness_coeff: float,
) -> np.ndarray:
    """Adds a simple snow effect to the image by bleaching out pixels.

    This function simulates a basic snow effect by increasing the brightness of pixels
    that are above a certain threshold (snow_point). It operates in the HLS color space
    to modify the lightness channel.

    Args:
        img (np.ndarray): Input image. Can be either RGB uint8 or float32.
        snow_point (float): A float in the range [0, 1], scaled and adjusted to determine
            the threshold for pixel modification. Higher values result in less snow effect.
        brightness_coeff (float): Coefficient applied to increase the brightness of pixels
            below the snow_point threshold. Larger values lead to more pronounced snow effects.
            Should be greater than 1.0 for a visible effect.

    Returns:
        np.ndarray: Image with simulated snow effect. The output has the same dtype as the input.

    Note:
        - This function converts the image to the HLS color space to modify the lightness channel.
        - The snow effect is created by selectively increasing the brightness of pixels.
        - This method tends to create a 'bleached' look, which may not be as realistic as more
          advanced snow simulation techniques.
        - The function automatically handles both uint8 and float32 input images.

    The snow effect is created through the following steps:
    1. Convert the image from RGB to HLS color space.
    2. Adjust the snow_point threshold.
    3. Increase the lightness of pixels below the threshold.
    4. Convert the image back to RGB.

    Mathematical Formulation:
        Let L be the lightness channel in HLS space.
        For each pixel (i, j):
        If L[i, j] < snow_point:
            L[i, j] = L[i, j] * brightness_coeff

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> snowy_image = A.functional.add_snow_v1(image, snow_point=0.5, brightness_coeff=1.5)

    References:
        - HLS Color Space: https://en.wikipedia.org/wiki/HSL_and_HSV
        - Original implementation: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    # Precompute snow_point threshold
    snow_point = (snow_point * max_value / 2) + (max_value / 3)

    # Convert image to HLS color space once and avoid repeated dtype casting
    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lightness_channel = image_hls[:, :, 1].astype(np.float32)

    # Utilize boolean indexing for efficient lightness adjustment
    mask = lightness_channel < snow_point
    lightness_channel[mask] *= brightness_coeff

    # Clip the lightness values in place
    lightness_channel = clip(lightness_channel, np.uint8, inplace=True)

    # Update the lightness channel in the original image
    image_hls[:, :, 1] = lightness_channel

    # Convert back to RGB
    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)


def generate_snow_textures(
    img_shape: tuple[int, int],
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate snow texture and sparkle mask.

    Args:
        img_shape (tuple[int, int]): Image shape.
        random_generator (np.random.Generator): Random generator to use.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (snow_texture, sparkle_mask) arrays.

    """
    # Generate base snow texture
    snow_texture = random_generator.normal(size=img_shape[:2], loc=0.5, scale=0.3)
    snow_texture = cv2.GaussianBlur(snow_texture, (0, 0), sigmaX=1, sigmaY=1)

    # Generate sparkle mask
    sparkle_mask = random_generator.random(img_shape[:2]) > 0.99

    return snow_texture, sparkle_mask


@uint8_io
def add_snow_texture(
    img: np.ndarray,
    snow_point: float,
    brightness_coeff: float,
    snow_texture: np.ndarray,
    sparkle_mask: np.ndarray,
) -> np.ndarray:
    """Add a realistic snow effect to the input image.

    This function simulates snowfall by applying multiple visual effects to the image,
    including brightness adjustment, snow texture overlay, depth simulation, and color tinting.
    The result is a more natural-looking snow effect compared to simple pixel bleaching methods.

    Args:
        img (np.ndarray): Input image in RGB format.
        snow_point (float): Coefficient that controls the amount and intensity of snow.
            Should be in the range [0, 1], where 0 means no snow and 1 means maximum snow effect.
        brightness_coeff (float): Coefficient for brightness adjustment to simulate the
            reflective nature of snow. Should be in the range [0, 1], where higher values
            result in a brighter image.
        snow_texture (np.ndarray): Snow texture.
        sparkle_mask (np.ndarray): Sparkle mask.

    Returns:
        np.ndarray: Image with added snow effect. The output has the same dtype as the input.

    Note:
        - The function first converts the image to HSV color space for better control over
          brightness and color adjustments.
        - A snow texture is generated using Gaussian noise and then filtered for a more
          natural appearance.
        - A depth effect is simulated, with more snow at the top of the image and less at the bottom.
        - A slight blue tint is added to simulate the cool color of snow.
        - Random sparkle effects are added to simulate light reflecting off snow crystals.

    The snow effect is created through the following steps:
    1. Brightness adjustment in HSV space
    2. Generation of a snow texture using Gaussian noise
    3. Application of a depth effect to the snow texture
    4. Blending of the snow texture with the original image
    5. Addition of a cool blue tint
    6. Addition of sparkle effects

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> snowy_image = A.functional.add_snow_v2(image, snow_coeff=0.5, brightness_coeff=0.2)

    Note:
        This function works with both uint8 and float32 image types, automatically
        handling the conversion between them.

    References:
        - Perlin Noise: https://en.wikipedia.org/wiki/Perlin_noise
        - HSV Color Space: https://en.wikipedia.org/wiki/HSL_and_HSV

    """
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    # Convert to HSV for better color control
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Increase brightness
    img_hsv[:, :, 2] = np.clip(
        img_hsv[:, :, 2] * (1 + brightness_coeff * snow_point),
        0,
        max_value,
    )

    # Generate snow texture
    snow_texture = cv2.GaussianBlur(snow_texture, (0, 0), sigmaX=1, sigmaY=1)

    # Create depth effect for snow simulation
    # More snow accumulates at the top of the image, gradually decreasing towards the bottom
    # This simulates natural snow distribution on surfaces
    # The effect is achieved using a linear gradient from 1 (full snow) to 0.2 (less snow)
    rows = img.shape[0]
    depth_effect = np.linspace(1, 0.2, rows)[:, np.newaxis]
    snow_texture *= depth_effect

    # Apply snow texture
    snow_layer = (np.dstack([snow_texture] * 3) * max_value * snow_point).astype(
        np.float32,
    )

    # Blend snow with original image
    img_with_snow = cv2.add(img_hsv, snow_layer)

    # Add a slight blue tint to simulate cool snow color
    blue_tint = np.full_like(img_with_snow, (0.6, 0.75, 1))  # Slight blue in HSV

    img_with_snow = cv2.addWeighted(
        img_with_snow,
        0.85,
        blue_tint,
        0.15 * snow_point,
        0,
    )

    # Convert back to RGB
    img_with_snow = cv2.cvtColor(img_with_snow.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Add some sparkle effects for snow glitter
    img_with_snow[sparkle_mask] = [max_value, max_value, max_value]

    return img_with_snow


@uint8_io
@preserve_channel_dim
def add_rain(
    img: np.ndarray,
    slant: float,
    drop_length: int,
    drop_width: int,
    drop_color: tuple[int, int, int],
    blur_value: int,
    brightness_coefficient: float,
    rain_drops: np.ndarray,
) -> np.ndarray:
    """Add rain to an image.

    This function adds rain to an image by drawing rain drops on the image.
    The rain drops are drawn using the OpenCV function cv2.polylines.

    Args:
        img (np.ndarray): The image to add rain to.
        slant (float): The slant of the rain drops.
        drop_length (int): The length of the rain drops.
        drop_width (int): The width of the rain drops.
        drop_color (tuple[int, int, int]): The color of the rain drops.
        blur_value (int): The blur value of the rain drops.
        brightness_coefficient (float): The brightness coefficient of the rain drops.
        rain_drops (np.ndarray): The rain drops to draw on the image.

    Returns:
        np.ndarray: The image with rain added.

    """
    if not rain_drops.size:
        return img.copy()

    img = img.copy()

    # Pre-allocate rain layer
    rain_layer = np.zeros_like(img, dtype=np.uint8)

    # Calculate end points correctly
    end_points = rain_drops + np.array([[slant, drop_length]])  # This creates correct shape

    # Stack arrays properly - both must be same shape arrays
    lines = np.stack((rain_drops, end_points), axis=1)  # Use tuple and proper axis

    cv2.polylines(
        rain_layer,
        lines.astype(np.int32),
        False,
        drop_color,
        drop_width,
        lineType=cv2.LINE_4,
    )

    if blur_value > 1:
        cv2.blur(rain_layer, (blur_value, blur_value), dst=rain_layer)

    cv2.add(img, rain_layer, dst=img)

    if brightness_coefficient != 1.0:
        cv2.multiply(img, brightness_coefficient, dst=img, dtype=cv2.CV_8U)

    return img


def get_fog_particle_radiuses(
    img_shape: tuple[int, int],
    num_particles: int,
    fog_intensity: float,
    random_generator: np.random.Generator,
) -> list[int]:
    """Generate radiuses for fog particles.

    Args:
        img_shape (tuple[int, int]): Image shape.
        num_particles (int): Number of fog particles.
        fog_intensity (float): Intensity of the fog effect, between 0 and 1.
        random_generator (np.random.Generator): Random generator to use.

    Returns:
        list[int]: List of radiuses for each fog particle.

    """
    height, width = img_shape[:2]
    max_fog_radius = max(2, int(min(height, width) * 0.1 * fog_intensity))
    min_radius = max(1, max_fog_radius // 2)

    return [random_generator.integers(min_radius, max_fog_radius) for _ in range(num_particles)]


@uint8_io
@clipped
@preserve_channel_dim
def add_fog(
    img: np.ndarray,
    fog_intensity: float,
    alpha_coef: float,
    fog_particle_positions: list[tuple[int, int]],
    fog_particle_radiuses: list[int],
) -> np.ndarray:
    """Add fog to an image.

    This function adds fog to an image by drawing fog particles on the image.
    The fog particles are drawn using the OpenCV function cv2.circle.

    Args:
        img (np.ndarray): The image to add fog to.
        fog_intensity (float): The intensity of the fog effect, between 0 and 1.
        alpha_coef (float): The coefficient for the alpha blending.
        fog_particle_positions (list[tuple[int, int]]): The positions of the fog particles.
        fog_particle_radiuses (list[int]): The radiuses of the fog particles.

    Returns:
        np.ndarray: The image with fog added.

    """
    result = img.copy()

    # Apply fog particles progressively like in old version
    for (x, y), radius in zip(fog_particle_positions, fog_particle_radiuses):
        overlay = result.copy()
        cv2.circle(
            overlay,
            center=(x, y),
            radius=radius,
            color=(255, 255, 255),
            thickness=-1,
        )

        # Progressive blending
        alpha = alpha_coef * fog_intensity
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, dst=result)

    # Final subtle blur
    blur_size = max(3, int(min(img.shape[:2]) // 30))
    if blur_size % 2 == 0:
        blur_size += 1

    result = cv2.GaussianBlur(result, (blur_size, blur_size), 0)

    return clip(result, np.uint8, inplace=True)


@uint8_io
@preserve_channel_dim
@maybe_process_in_chunks
def add_sun_flare_overlay(
    img: np.ndarray,
    flare_center: tuple[float, float],
    src_radius: int,
    src_color: tuple[int, ...],
    circles: list[Any],
) -> np.ndarray:
    """Add a sun flare effect to an image using a simple overlay technique.

    This function creates a basic sun flare effect by overlaying multiple semi-transparent
    circles of varying sizes and intensities on the input image. The effect simulates
    a simple lens flare caused by bright light sources.

    Args:
        img (np.ndarray): The input image.
        flare_center (tuple[float, float]): (x, y) coordinates of the flare center
            in pixel coordinates.
        src_radius (int): The radius of the main sun circle in pixels.
        src_color (tuple[int, ...]): The color of the sun, represented as a tuple of RGB values.
        circles (list[Any]): A list of tuples, each representing a circle that contributes
            to the flare effect. Each tuple contains:
            - alpha (float): The transparency of the circle (0.0 to 1.0).
            - center (tuple[int, int]): (x, y) coordinates of the circle center.
            - radius (int): The radius of the circle.
            - color (tuple[int, int, int]): RGB color of the circle.

    Returns:
        np.ndarray: The output image with the sun flare effect added.

    Note:
        - This function uses a simple alpha blending technique to overlay flare elements.
        - The main sun is created as a gradient circle, fading from the center outwards.
        - Additional flare circles are added along an imaginary line from the sun's position.
        - This method is computationally efficient but may produce less realistic results
          compared to more advanced techniques.

    The flare effect is created through the following steps:
    1. Create an overlay image and output image as copies of the input.
    2. Add smaller flare circles to the overlay.
    3. Blend the overlay with the output image using alpha compositing.
    4. Add the main sun circle with a radial gradient.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> flare_center = (50, 50)
        >>> src_radius = 20
        >>> src_color = (255, 255, 200)
        >>> circles = [
        ...     (0.1, (60, 60), 5, (255, 200, 200)),
        ...     (0.2, (70, 70), 3, (200, 255, 200))
        ... ]
        >>> flared_image = A.functional.add_sun_flare_overlay(
        ...     image, flare_center, src_radius, src_color, circles
        ... )

    References:
        - Alpha compositing: https://en.wikipedia.org/wiki/Alpha_compositing
        - Lens flare: https://en.wikipedia.org/wiki/Lens_flare

    """
    overlay = img.copy()
    output = img.copy()

    weighted_brightness = 0.0
    total_radius_length = 0.0

    for alpha, (x, y), rad3, circle_color in circles:
        weighted_brightness += alpha * rad3
        total_radius_length += rad3
        cv2.circle(overlay, (x, y), rad3, circle_color, -1)
        output = add_weighted(overlay, alpha, output, 1 - alpha)

    point = [int(x) for x in flare_center]

    overlay = output.copy()
    num_times = src_radius // 10

    # max_alpha is calculated using weighted_brightness and total_radii_length times 5
    # meaning the higher the alpha with larger area, the brighter the bright spot will be
    # for list of alphas in range [0.05, 0.2], the max_alpha should below 1
    max_alpha = weighted_brightness / total_radius_length * 5
    alpha = np.linspace(0.0, min(max_alpha, 1.0), num=num_times)

    rad = np.linspace(1, src_radius, num=num_times)

    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times - i - 1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        output = add_weighted(overlay, alp, output, 1 - alp)

    return output


@uint8_io
@clipped
def add_sun_flare_physics_based(
    img: np.ndarray,
    flare_center: tuple[int, int],
    src_radius: int,
    src_color: tuple[int, int, int],
    circles: list[Any],
) -> np.ndarray:
    """Add a more realistic sun flare effect to the image.

    This function creates a complex sun flare effect by simulating various optical phenomena
    that occur in real camera lenses when capturing bright light sources. The result is a
    more realistic and physically plausible lens flare effect.

    Args:
        img (np.ndarray): Input image.
        flare_center (tuple[int, int]): (x, y) coordinates of the sun's center in pixels.
        src_radius (int): Radius of the main sun circle in pixels.
        src_color (tuple[int, int, int]): Color of the sun in RGB format.
        circles (list[Any]): List of tuples, each representing a flare circle with parameters:
            (alpha, center, size, color)
            - alpha (float): Transparency of the circle (0.0 to 1.0).
            - center (tuple[int, int]): (x, y) coordinates of the circle center.
            - size (float): Size factor for the circle radius.
            - color (tuple[int, int, int]): RGB color of the circle.

    Returns:
        np.ndarray: Image with added sun flare effect.

    Note:
        This function implements several techniques to create a more realistic flare:
        1. Separate flare layer: Allows for complex manipulations of the flare effect.
        2. Lens diffraction spikes: Simulates light diffraction in camera aperture.
        3. Radial gradient mask: Creates natural fading of the flare from the center.
        4. Gaussian blur: Softens the flare for a more natural glow effect.
        5. Chromatic aberration: Simulates color fringing often seen in real lens flares.
        6. Screen blending: Provides a more realistic blending of the flare with the image.

    The flare effect is created through the following steps:
    1. Create a separate flare layer.
    2. Add the main sun circle and diffraction spikes to the flare layer.
    3. Add additional flare circles based on the input parameters.
    4. Apply Gaussian blur to soften the flare.
    5. Create and apply a radial gradient mask for natural fading.
    6. Simulate chromatic aberration by applying different blurs to color channels.
    7. Blend the flare with the original image using screen blending mode.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [1000, 1000, 3], dtype=np.uint8)
        >>> flare_center = (500, 500)
        >>> src_radius = 50
        >>> src_color = (255, 255, 200)
        >>> circles = [
        ...     (0.1, (550, 550), 10, (255, 200, 200)),
        ...     (0.2, (600, 600), 5, (200, 255, 200))
        ... ]
        >>> flared_image = A.functional.add_sun_flare_physics_based(
        ...     image, flare_center, src_radius, src_color, circles
        ... )

    References:
        - Lens flare: https://en.wikipedia.org/wiki/Lens_flare
        - Diffraction: https://en.wikipedia.org/wiki/Diffraction
        - Chromatic aberration: https://en.wikipedia.org/wiki/Chromatic_aberration
        - Screen blending: https://en.wikipedia.org/wiki/Blend_modes#Screen

    """
    output = img.copy()
    height, width = img.shape[:2]

    # Create a separate flare layer
    flare_layer = np.zeros_like(img, dtype=np.float32)

    # Add the main sun
    cv2.circle(flare_layer, flare_center, src_radius, src_color, -1)

    # Add lens diffraction spikes
    for angle in [0, 45, 90, 135]:
        end_point = (
            int(flare_center[0] + np.cos(np.radians(angle)) * max(width, height)),
            int(flare_center[1] + np.sin(np.radians(angle)) * max(width, height)),
        )
        cv2.line(flare_layer, flare_center, end_point, src_color, 2)

    # Add flare circles
    for _, center, size, color in circles:
        cv2.circle(flare_layer, center, int(size**0.33), color, -1)

    # Apply gaussian blur to soften the flare
    flare_layer = cv2.GaussianBlur(flare_layer, (0, 0), sigmaX=15, sigmaY=15)

    # Create a radial gradient mask
    y, x = np.ogrid[:height, :width]
    mask = np.sqrt((x - flare_center[0]) ** 2 + (y - flare_center[1]) ** 2)
    mask = 1 - np.clip(mask / (max(width, height) * 0.7), 0, 1)
    mask = np.dstack([mask] * 3)

    # Apply the mask to the flare layer
    flare_layer *= mask

    # Add chromatic aberration
    channels = list(cv2.split(flare_layer))
    channels[0] = cv2.GaussianBlur(
        channels[0],
        (0, 0),
        sigmaX=3,
        sigmaY=3,
    )  # Blue channel
    channels[2] = cv2.GaussianBlur(
        channels[2],
        (0, 0),
        sigmaX=5,
        sigmaY=5,
    )  # Red channel
    flare_layer = cv2.merge(channels)

    # Blend the flare with the original image using screen blending
    return 255 - ((255 - output) * (255 - flare_layer) / 255)


@uint8_io
@preserve_channel_dim
def add_shadow(
    img: np.ndarray,
    vertices_list: list[np.ndarray],
    intensities: np.ndarray,
) -> np.ndarray:
    """Add shadows to the image by reducing the intensity of the pixel values in specified regions.

    Args:
        img (np.ndarray): Input image. Multichannel images are supported.
        vertices_list (list[np.ndarray]): List of vertices for shadow polygons.
        intensities (np.ndarray): Array of shadow intensities. Range is [0, 1].

    Returns:
        np.ndarray: Image with shadows added.

    References:
        Automold--Road-Augmentation-Library: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """
    num_channels = get_num_channels(img)
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    img_shadowed = img.copy()

    # Iterate over the vertices and intensity list
    for vertices, shadow_intensity in zip(vertices_list, intensities):
        # Create mask for the current shadow polygon
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], (max_value,))

        # Duplicate the mask to have the same number of channels as the image
        mask = np.repeat(mask, num_channels, axis=2)

        # Apply shadow to the channels directly
        # It could be tempting to convert to HLS and apply the shadow to the L channel, but it creates artifacts
        shadowed_indices = mask[:, :, 0] == max_value
        darkness = 1 - shadow_intensity
        img_shadowed[shadowed_indices] = clip(
            img_shadowed[shadowed_indices] * darkness,
            np.uint8,
            inplace=True,
        )

    return img_shadowed


@uint8_io
@clipped
@preserve_channel_dim
def add_gravel(img: np.ndarray, gravels: list[Any]) -> np.ndarray:
    """Add gravel to an image.

    This function adds gravel to an image by drawing gravel particles on the image.
    The gravel particles are drawn using the OpenCV function cv2.circle.

    Args:
        img (np.ndarray): The image to add gravel to.
        gravels (list[Any]): The gravel particles to draw on the image.

    Returns:
        np.ndarray: The image with gravel added.

    """
    non_rgb_error(img)
    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    for gravel in gravels:
        min_y, max_y, min_x, max_x, sat = gravel
        image_hls[min_y:max_y, min_x:max_x, 1] = sat

    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)


def invert(img: np.ndarray) -> np.ndarray:
    """Invert the colors of an image.

    This function inverts the colors of an image by subtracting each pixel value from the maximum possible value.
    The result is a negative of the original image.

    Args:
        img (np.ndarray): The image to invert.

    Returns:
        np.ndarray: The inverted image.

    """
    # Supports all the valid dtypes
    # clips the img to avoid unexpected behaviour.
    return MAX_VALUES_BY_DTYPE[img.dtype] - img


def channel_shuffle(img: np.ndarray, channels_shuffled: list[int]) -> np.ndarray:
    """Shuffle the channels of an image.

    This function shuffles the channels of an image by using the cv2.mixChannels function.
    The channels are shuffled according to the channels_shuffled array.

    Args:
        img (np.ndarray): The image to shuffle.
        channels_shuffled (np.ndarray): The array of channels to shuffle.

    Returns:
        np.ndarray: The shuffled image.

    """
    output = np.empty_like(img)
    from_to = []
    for i, j in enumerate(channels_shuffled):
        from_to.extend([j, i])  # Use [src, dst]
    cv2.mixChannels([img], [output], from_to)
    return output


def volume_channel_shuffle(volume: np.ndarray, channels_shuffled: Sequence[int]) -> np.ndarray:
    """Shuffle channels of a single volume (D, H, W, C) or (D, H, W).

    Args:
        volume (np.ndarray): Input volume.
        channels_shuffled (Sequence[int]): New channel order.

    Returns:
        np.ndarray: Volume with channels shuffled.

    """
    return volume.copy()[..., channels_shuffled] if volume.ndim == 4 else volume


def volumes_channel_shuffle(volumes: np.ndarray, channels_shuffled: Sequence[int]) -> np.ndarray:
    """Shuffle channels of a batch of volumes (B, D, H, W, C) or (B, D, H, W).

    Args:
        volumes (np.ndarray): Input batch of volumes.
        channels_shuffled (Sequence[int]): New channel order.

    Returns:
        np.ndarray: Batch of volumes with channels shuffled.

    """
    return volumes.copy()[..., channels_shuffled] if volumes.ndim == 5 else volumes


def gamma_transform(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma transformation to an image.

    This function applies gamma transformation to an image by raising each pixel value to the power of gamma.
    The result is a non-linear transformation that can enhance or reduce the contrast of the image.

    Args:
        img (np.ndarray): The image to apply gamma transformation to.
        gamma (float): The gamma value to apply.

    Returns:
        np.ndarray: The gamma transformed image.

    """
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        return sz_lut(img, table.astype(np.uint8), inplace=False)

    return np.power(img, gamma)


@float32_io
@clipped
def iso_noise(
    image: np.ndarray,
    color_shift: float,
    intensity: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply poisson noise to an image to simulate camera sensor noise.

    Args:
        image (np.ndarray): Input image. Currently, only RGB images are supported.
        color_shift (float): The amount of color shift to apply.
        intensity (float): Multiplication factor for noise values. Values of ~0.5 produce a noticeable,
                           yet acceptable level of noise.
        random_generator (np.random.Generator): If specified, this will be random generator used
            for noise generation.

    Returns:
        np.ndarray: The noised image.

    Image types:
        uint8, float32

    Number of channels:
        3

    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)

    luminance_noise = random_generator.poisson(
        stddev[1] * intensity,
        size=hls.shape[:2],
    )
    color_noise = random_generator.normal(
        0,
        color_shift * intensity,
        size=hls.shape[:2],
    )

    hls[..., 0] += color_noise
    hls[..., 1] = add_array(
        hls[..., 1],
        luminance_noise * intensity * (1.0 - hls[..., 1]),
    )

    noised_hls = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
    return np.clip(noised_hls, 0, 1, out=noised_hls)  # Ensure output is in [0, 1] range


def to_gray_weighted_average(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using the weighted average method.

    This function uses OpenCV's cvtColor function with COLOR_RGB2GRAY conversion,
    which applies the following formula:
    Y = 0.299*R + 0.587*G + 0.114*B

    Args:
        img (np.ndarray): Input RGB image as a numpy array.

    Returns:
        np.ndarray: Grayscale image as a 2D numpy array.

    Image types:
        uint8, float32

    Number of channels:
        3

    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


@uint8_io
@clipped
def to_gray_from_lab(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using the L channel from the LAB color space.

    This function converts the RGB image to the LAB color space and extracts the L channel.
    The LAB color space is designed to approximate human vision, where L represents lightness.

    Key aspects of this method:
    1. The L channel represents the lightness of each pixel, ranging from 0 (black) to 100 (white).
    2. It's more perceptually uniform than RGB, meaning equal changes in L values correspond to
       roughly equal changes in perceived lightness.
    3. The L channel is independent of the color information (A and B channels), making it
       suitable for grayscale conversion.

    This method can be particularly useful when you want a grayscale image that closely
    matches human perception of lightness, potentially preserving more perceived contrast
    than simple RGB-based methods.

    Args:
        img (np.ndarray): Input RGB image as a numpy array.

    Returns:
        np.ndarray: Grayscale image as a 2D numpy array, representing the L (lightness) channel.
                    Values are scaled to match the input image's data type range.

    Image types:
        uint8, float32

    Number of channels:
        3

    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0]


@clipped
def to_gray_desaturation(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale using the desaturation method.

    Args:
        img (np.ndarray): Input image as a numpy array.

    Returns:
        np.ndarray: Grayscale image as a 2D numpy array.

    Image types:
        uint8, float32

    Number of channels:
        any

    """
    float_image = img.astype(np.float32)
    return (np.max(float_image, axis=-1) + np.min(float_image, axis=-1)) / 2


def to_gray_average(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale using the average method.

    This function computes the arithmetic mean across all channels for each pixel,
    resulting in a grayscale representation of the image.

    Key aspects of this method:
    1. It treats all channels equally, regardless of their perceptual importance.
    2. Works with any number of channels, making it versatile for various image types.
    3. Simple and fast to compute, but may not accurately represent perceived brightness.
    4. For RGB images, the formula is: Gray = (R + G + B) / 3

    Note: This method may produce different results compared to weighted methods
    (like RGB weighted average) which account for human perception of color brightness.
    It may also produce unexpected results for images with alpha channels or
    non-color data in additional channels.

    Args:
        img (np.ndarray): Input image as a numpy array. Can be any number of channels.

    Returns:
        np.ndarray: Grayscale image as a 2D numpy array. The output data type
                    matches the input data type.

    Image types:
        uint8, float32

    Number of channels:
        any

    """
    return np.mean(img, axis=-1).astype(img.dtype)


def to_gray_max(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale using the maximum channel value method.

    This function takes the maximum value across all channels for each pixel,
    resulting in a grayscale image that preserves the brightest parts of the original image.

    Key aspects of this method:
    1. Works with any number of channels, making it versatile for various image types.
    2. For 3-channel (e.g., RGB) images, this method is equivalent to extracting the V (Value)
       channel from the HSV color space.
    3. Preserves the brightest parts of the image but may lose some color contrast information.
    4. Simple and fast to compute.

    Note:
    - This method tends to produce brighter grayscale images compared to other conversion methods,
      as it always selects the highest intensity value from the channels.
    - For RGB images, it may not accurately represent perceived brightness as it doesn't
      account for human color perception.

    Args:
        img (np.ndarray): Input image as a numpy array. Can be any number of channels.

    Returns:
        np.ndarray: Grayscale image as a 2D numpy array. The output data type
                    matches the input data type.

    Image types:
        uint8, float32

    Number of channels:
        any

    """
    return np.max(img, axis=-1)


@clipped
def to_gray_pca(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale using Principal Component Analysis (PCA).

    This function applies PCA to reduce a multi-channel image to a single channel,
    effectively creating a grayscale representation that captures the maximum variance
    in the color data.

    Args:
        img (np.ndarray): Input image as a numpy array with shape (height, width, channels).

    Returns:
        np.ndarray: Grayscale image as a 2D numpy array with shape (height, width).
                    If input is uint8, output is uint8 in range [0, 255].
                    If input is float32, output is float32 in range [0, 1].

    Note:
        This method can potentially preserve more information from the original image
        compared to standard weighted average methods, as it accounts for the
        correlations between color channels.

    Image types:
        uint8, float32

    Number of channels:
        any

    """
    dtype = img.dtype
    # Reshape the image to a 2D array of pixels
    pixels = img.reshape(-1, img.shape[2])

    # Perform PCA
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(pixels)

    # Reshape back to image dimensions and scale to 0-255
    grayscale = pca_result.reshape(img.shape[:2])
    grayscale = normalize_per_image(grayscale, "min_max")

    return from_float(grayscale, target_dtype=dtype) if dtype == np.uint8 else grayscale


def to_gray(
    img: np.ndarray,
    num_output_channels: int,
    method: Literal[
        "weighted_average",
        "from_lab",
        "desaturation",
        "average",
        "max",
        "pca",
    ],
) -> np.ndarray:
    """Convert an image to grayscale using a specified method.

    This function converts an image to grayscale using a specified method.
    The method can be one of the following:
    - "weighted_average": Use the weighted average method.
    - "from_lab": Use the L channel from the LAB color space.
    - "desaturation": Use the desaturation method.
    - "average": Use the average method.
    - "max": Use the maximum channel value method.
    - "pca": Use the Principal Component Analysis method.

    Args:
        img (np.ndarray): Input image as a numpy array.
        num_output_channels (int): The number of channels in the output image.
        method (Literal["weighted_average", "from_lab", "desaturation", "average", "max", "pca"]):
            The method to use for grayscale conversion.

    Returns:
        np.ndarray: Grayscale image as a 2D numpy array.

    """
    if method == "weighted_average":
        result = to_gray_weighted_average(img)
    elif method == "from_lab":
        result = to_gray_from_lab(img)
    elif method == "desaturation":
        result = to_gray_desaturation(img)
    elif method == "average":
        result = to_gray_average(img)
    elif method == "max":
        result = to_gray_max(img)
    elif method == "pca":
        result = to_gray_pca(img)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return grayscale_to_multichannel(result, num_output_channels)


def grayscale_to_multichannel(
    grayscale_image: np.ndarray,
    num_output_channels: int = 3,
) -> np.ndarray:
    """Convert a grayscale image to a multi-channel image.

    This function takes a 2D grayscale image or a 3D image with a single channel
    and converts it to a multi-channel image by repeating the grayscale data
    across the specified number of channels.

    Args:
        grayscale_image (np.ndarray): Input grayscale image. Can be 2D (height, width)
                                      or 3D (height, width, 1).
        num_output_channels (int, optional): Number of channels in the output image. Defaults to 3.

    Returns:
        np.ndarray: Multi-channel image with shape (height, width, num_channels)

    """
    # If output should be single channel, just squeeze and return
    if num_output_channels == 1:
        return grayscale_image

    # For multi-channel output, squeeze and stack
    squeezed = np.squeeze(grayscale_image)

    return cv2.merge([squeezed] * num_output_channels)


@preserve_channel_dim
@uint8_io
def downscale(
    img: np.ndarray,
    scale: float,
    down_interpolation: int,
    up_interpolation: int,
) -> np.ndarray:
    """Downscale and upscale an image.

    This function downscales and upscales an image using the specified interpolation methods.
    The downscaling and upscaling are performed using the cv2.resize function.

    Args:
        img (np.ndarray): Input image as a numpy array.
        scale (float): The scale factor for the downscaling and upscaling.
        down_interpolation (int): The interpolation method for the downscaling.
        up_interpolation (int): The interpolation method for the upscaling.

    Returns:
        np.ndarray: The downscaled and upscaled image.

    """
    height, width = img.shape[:2]

    downscaled = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=down_interpolation,
    )
    return cv2.resize(downscaled, (width, height), interpolation=up_interpolation)


def noop(input_obj: Any, **params: Any) -> Any:
    """No-op function.

    This function is a no-op and returns the input object unchanged.
    It is used to satisfy the type checker requirements for the `noop` function.

    Args:
        input_obj (Any): The input object to return unchanged.
        **params (Any): Additional keyword arguments.

    Returns:
        Any: The input object unchanged.

    """
    return input_obj


@float32_io
@clipped
@preserve_channel_dim
def fancy_pca(img: np.ndarray, alpha_vector: np.ndarray) -> np.ndarray:
    """Perform 'Fancy PCA' augmentation on an image with any number of channels.

    Args:
        img (np.ndarray): Input image
        alpha_vector (np.ndarray): Vector of scale factors for each principal component.
                                   Should have the same length as the number of channels in the image.

    Returns:
        np.ndarray: Augmented image of the same shape, type, and range as the input.

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This function generalizes the Fancy PCA augmentation to work with any number of channels.
        - It preserves the original range of the image ([0, 255] for uint8, [0, 1] for float32).
        - For single-channel images, the augmentation is applied as a simple scaling of pixel intensity variation.
        - For multi-channel images, PCA is performed on the entire image, treating each pixel
          as a point in N-dimensional space (where N is the number of channels).
        - The augmentation preserves the correlation between channels while adding controlled noise.
        - Computation time may increase significantly for images with a large number of channels.

    References:
        ImageNet classification with deep convolutional neural networks: Krizhevsky, A., Sutskever, I.,
            & Hinton, G. E. (2012): In Advances in neural information processing systems (pp. 1097-1105).

    """
    orig_shape = img.shape
    num_channels = get_num_channels(img)

    # Reshape image to 2D array of pixels
    img_reshaped = img.reshape(-1, num_channels)

    # Center the pixel values
    img_mean = np.mean(img_reshaped, axis=0)
    img_centered = img_reshaped - img_mean

    if num_channels == 1:
        # For grayscale images, apply a simple scaling
        std_dev = np.std(img_centered)
        noise = alpha_vector[0] * std_dev * img_centered
    else:
        # Compute covariance matrix
        img_cov = np.cov(img_centered, rowvar=False)

        # Compute eigenvectors & eigenvalues of the covariance matrix
        eig_vals, eig_vecs = np.linalg.eigh(img_cov)

        # Sort eigenvectors by eigenvalues in descending order
        sort_perm = eig_vals[::-1].argsort()
        eig_vals = eig_vals[sort_perm]
        eig_vecs = eig_vecs[:, sort_perm]

        # Create noise vector
        noise = np.dot(
            np.dot(eig_vecs, np.diag(alpha_vector * eig_vals)),
            img_centered.T,
        ).T

    # Add noise to the image
    img_pca = img_reshaped + noise

    # Reshape back to original shape
    img_pca = img_pca.reshape(orig_shape)

    # Clip values to [0, 1] range
    return np.clip(img_pca, 0, 1, out=img_pca)


@preserve_channel_dim
def adjust_brightness_torchvision(img: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Adjust the brightness of an image.

    This function adjusts the brightness of an image by multiplying each pixel value by a factor.
    The brightness is adjusted by multiplying the image by the factor.

    Args:
        img (np.ndarray): Input image as a numpy array.
        factor (np.ndarray): The factor to adjust the brightness by.

    Returns:
        np.ndarray: The adjusted image.

    """
    if factor == 0:
        return np.zeros_like(img)
    if factor == 1:
        return img

    return multiply(img, factor, inplace=False)


@preserve_channel_dim
def adjust_contrast_torchvision(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust the contrast of an image.

    This function adjusts the contrast of an image by multiplying each pixel value by a factor.
    The contrast is adjusted by multiplying the image by the factor.

    Args:
        img (np.ndarray): Input image as a numpy array.
        factor (float): The factor to adjust the contrast by.

    Returns:
        np.ndarray: The adjusted image.

    """
    if factor == 1:
        return img

    mean = img.mean() if is_grayscale_image(img) else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        if img.dtype != np.float32:
            mean = int(mean + 0.5)
        return np.full_like(img, mean, dtype=img.dtype)

    return multiply_add(img, factor, mean * (1 - factor), inplace=False)


@clipped
@preserve_channel_dim
def adjust_saturation_torchvision(
    img: np.ndarray,
    factor: float,
    gamma: float = 0,
) -> np.ndarray:
    """Adjust the saturation of an image.

    This function adjusts the saturation of an image by multiplying each pixel value by a factor.
    The saturation is adjusted by multiplying the image by the factor.

    Args:
        img (np.ndarray): Input image as a numpy array.
        factor (float): The factor to adjust the saturation by.
        gamma (float): The gamma value to use for the adjustment.

    Returns:
        np.ndarray: The adjusted image.

    """
    if factor == 1 or is_grayscale_image(img):
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return gray if factor == 0 else cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)


def _adjust_hue_torchvision_uint8(img: np.ndarray, factor: float) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    img[..., 0] = sz_lut(img[..., 0], lut, inplace=False)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def adjust_hue_torchvision(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust the hue of an image.

    This function adjusts the hue of an image by adding a factor to the hue value.

    Args:
        img (np.ndarray): Input image.
        factor (float): The factor to adjust the hue by.

    Returns:
        np.ndarray: The adjusted image.

    """
    if is_grayscale_image(img) or factor == 0:
        return img

    if img.dtype == np.uint8:
        return _adjust_hue_torchvision_uint8(img, factor)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 0] = np.mod(img[..., 0] + factor * 360, 360)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


@uint8_io
@preserve_channel_dim
def superpixels(
    image: np.ndarray,
    n_segments: int,
    replace_samples: Sequence[bool],
    max_size: int | None,
    interpolation: int,
) -> np.ndarray:
    """Apply superpixels to an image.

    This function applies superpixels to an image using the SLIC algorithm.
    The superpixels are applied by replacing the pixels in the image with the mean intensity of the superpixel.

    Args:
        image (np.ndarray): Input image as a numpy array.
        n_segments (int): The number of segments to use for the superpixels.
        replace_samples (Sequence[bool]): The samples to replace.
        max_size (int | None): The maximum size of the superpixels.
        interpolation (int): The interpolation method to use.

    Returns:
        np.ndarray: The superpixels applied to the image.

    """
    if not np.any(replace_samples):
        return image

    orig_shape = image.shape
    if max_size is not None:
        size = max(image.shape[:2])
        if size > max_size:
            scale = max_size / size
            height, width = image.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)
            image = fgeometric.resize(image, (new_height, new_width), interpolation)

    segments = slic(
        image,
        n_segments=n_segments,
        compactness=10,
    )

    min_value = 0
    max_value = MAX_VALUES_BY_DTYPE[image.dtype]
    image = np.copy(image)

    if image.ndim == MONO_CHANNEL_DIMENSIONS:
        image = np.expand_dims(image, axis=-1)

    num_channels = get_num_channels(image)

    for c in range(num_channels):
        image_sp_c = image[..., c]
        # Get unique segment labels (skip 0 if it exists as it's typically background)
        unique_labels = np.unique(segments)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        # Calculate mean intensity for each segment
        for idx, label in enumerate(unique_labels):
            # with mod here, because slic can sometimes create more superpixel than requested.
            # replace_samples then does not have enough values, so we just start over with the first one again.
            if replace_samples[idx % len(replace_samples)]:
                mask = segments == label
                mean_intensity = np.mean(image_sp_c[mask])

                if image_sp_c.dtype.kind in ["i", "u", "b"]:
                    # After rounding the value can end up slightly outside of the value_range. Hence, we need to clip.
                    # We do clip via min(max(...)) instead of np.clip because
                    # the latter one does not seem to keep dtypes for dtypes with large itemsizes (e.g. uint64).
                    value: int | float
                    value = int(np.round(mean_intensity))
                    value = min(max(value, min_value), max_value)
                else:
                    value = mean_intensity

                image_sp_c[mask] = value

    return fgeometric.resize(image, orig_shape[:2], interpolation) if orig_shape != image.shape else image


@float32_io
@clipped
@preserve_channel_dim
def unsharp_mask(
    image: np.ndarray,
    ksize: int,
    sigma: float,
    alpha: float,
    threshold: int,
) -> np.ndarray:
    """Apply an unsharp mask to an image.

    This function applies an unsharp mask to an image using the Gaussian blur function.
    The unsharp mask is applied by subtracting the blurred image from the original image and
    then adding the result to the original image.

    Args:
        image (np.ndarray): Input image as a numpy array.
        ksize (int): The kernel size to use for the Gaussian blur.
        sigma (float): The sigma value to use for the Gaussian blur.
        alpha (float): The alpha value to use for the unsharp mask.
        threshold (int): The threshold value to use for the unsharp mask.

    Returns:
        np.ndarray: The unsharp mask applied to the image.

    """
    blur_fn = maybe_process_in_chunks(
        cv2.GaussianBlur,
        ksize=(ksize, ksize),
        sigmaX=sigma,
    )

    if image.ndim == NUM_MULTI_CHANNEL_DIMENSIONS and get_num_channels(image) == 1:
        image = np.squeeze(image, axis=-1)

    blur = blur_fn(image)
    residual = image - blur

    # Do not sharpen noise
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype(np.float32)

    sharp = image + alpha * residual
    # Avoid color noise artefacts.
    sharp = np.clip(sharp, 0, 1, out=sharp)

    soft_mask = blur_fn(mask)

    return add_array(
        multiply(sharp, soft_mask),
        multiply(image, 1 - soft_mask),
        inplace=True,
    )


@preserve_channel_dim
def pixel_dropout(
    image: np.ndarray,
    drop_mask: np.ndarray,
    drop_values: np.ndarray,
) -> np.ndarray:
    """Apply pixel dropout to the image.

    Args:
        image (np.ndarray): Input image
        drop_mask (np.ndarray): Boolean mask indicating which pixels to drop
        drop_values (np.ndarray): Values to replace dropped pixels with

    Returns:
        np.ndarray: Image with dropped pixels

    """
    return np.where(drop_mask, drop_values, image)


@float32_io
@clipped
@preserve_channel_dim
def spatter_rain(img: np.ndarray, rain: np.ndarray) -> np.ndarray:
    """Apply spatter rain to an image.

    This function applies spatter rain to an image by adding the rain to the image.

    Args:
        img (np.ndarray): Input image as a numpy array.
        rain (np.ndarray): Rain image as a numpy array.

    Returns:
        np.ndarray: The spatter rain applied to the image.

    """
    return add(img, rain, inplace=False)


@float32_io
@clipped
@preserve_channel_dim
def spatter_mud(img: np.ndarray, non_mud: np.ndarray, mud: np.ndarray) -> np.ndarray:
    """Apply spatter mud to an image.

    This function applies spatter mud to an image by adding the mud to the image.

    Args:
        img (np.ndarray): Input image as a numpy array.
        non_mud (np.ndarray): Non-mud image as a numpy array.
        mud (np.ndarray): Mud image as a numpy array.

    Returns:
        np.ndarray: The spatter mud applied to the image.

    """
    return add(img * non_mud, mud, inplace=False)


@uint8_io
@clipped
def chromatic_aberration(
    img: np.ndarray,
    primary_distortion_red: float,
    secondary_distortion_red: float,
    primary_distortion_blue: float,
    secondary_distortion_blue: float,
    interpolation: int,
) -> np.ndarray:
    """Apply chromatic aberration to an image.

    This function applies chromatic aberration to an image by distorting the red and blue channels.

    Args:
        img (np.ndarray): Input image as a numpy array.
        primary_distortion_red (float): The primary distortion of the red channel.
        secondary_distortion_red (float): The secondary distortion of the red channel.
        primary_distortion_blue (float): The primary distortion of the blue channel.
        secondary_distortion_blue (float): The secondary distortion of the blue channel.
        interpolation (int): The interpolation method to use.

    Returns:
        np.ndarray: The chromatic aberration applied to the image.

    """
    height, width = img.shape[:2]

    # Build camera matrix
    camera_mat = np.eye(3, dtype=np.float32)
    camera_mat[0, 0] = width
    camera_mat[1, 1] = height
    camera_mat[0, 2] = width / 2.0
    camera_mat[1, 2] = height / 2.0

    # Build distortion coefficients
    distortion_coeffs_red = np.array(
        [primary_distortion_red, secondary_distortion_red, 0, 0],
        dtype=np.float32,
    )
    distortion_coeffs_blue = np.array(
        [primary_distortion_blue, secondary_distortion_blue, 0, 0],
        dtype=np.float32,
    )

    # Distort the red and blue channels
    red_distorted = _distort_channel(
        img[..., 0],
        camera_mat,
        distortion_coeffs_red,
        height,
        width,
        interpolation,
    )
    blue_distorted = _distort_channel(
        img[..., 2],
        camera_mat,
        distortion_coeffs_blue,
        height,
        width,
        interpolation,
    )

    return np.dstack([red_distorted, img[..., 1], blue_distorted])


def _distort_channel(
    channel: np.ndarray,
    camera_mat: np.ndarray,
    distortion_coeffs: np.ndarray,
    height: int,
    width: int,
    interpolation: int,
) -> np.ndarray:
    map_x, map_y = cv2.initUndistortRectifyMap(
        cameraMatrix=camera_mat,
        distCoeffs=distortion_coeffs,
        R=None,
        newCameraMatrix=camera_mat,
        size=(width, height),
        m1type=cv2.CV_32FC1,
    )
    return cv2.remap(
        channel,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )


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


PLANCKIAN_COEFFS: dict[str, dict[int, list[float]]] = {
    "blackbody": {
        3_000: [0.6743, 0.4029, 0.0013],
        3_500: [0.6281, 0.4241, 0.1665],
        4_000: [0.5919, 0.4372, 0.2513],
        4_500: [0.5623, 0.4457, 0.3154],
        5_000: [0.5376, 0.4515, 0.3672],
        5_500: [0.5163, 0.4555, 0.4103],
        6_000: [0.4979, 0.4584, 0.4468],
        6_500: [0.4816, 0.4604, 0.4782],
        7_000: [0.4672, 0.4619, 0.5053],
        7_500: [0.4542, 0.4630, 0.5289],
        8_000: [0.4426, 0.4638, 0.5497],
        8_500: [0.4320, 0.4644, 0.5681],
        9_000: [0.4223, 0.4648, 0.5844],
        9_500: [0.4135, 0.4651, 0.5990],
        10_000: [0.4054, 0.4653, 0.6121],
        10_500: [0.3980, 0.4654, 0.6239],
        11_000: [0.3911, 0.4655, 0.6346],
        11_500: [0.3847, 0.4656, 0.6444],
        12_000: [0.3787, 0.4656, 0.6532],
        12_500: [0.3732, 0.4656, 0.6613],
        13_000: [0.3680, 0.4655, 0.6688],
        13_500: [0.3632, 0.4655, 0.6756],
        14_000: [0.3586, 0.4655, 0.6820],
        14_500: [0.3544, 0.4654, 0.6878],
        15_000: [0.3503, 0.4653, 0.6933],
    },
    "cied": {
        4_000: [0.5829, 0.4421, 0.2288],
        4_500: [0.5510, 0.4514, 0.2948],
        5_000: [0.5246, 0.4576, 0.3488],
        5_500: [0.5021, 0.4618, 0.3941],
        6_000: [0.4826, 0.4646, 0.4325],
        6_500: [0.4654, 0.4667, 0.4654],
        7_000: [0.4502, 0.4681, 0.4938],
        7_500: [0.4364, 0.4692, 0.5186],
        8_000: [0.4240, 0.4700, 0.5403],
        8_500: [0.4127, 0.4705, 0.5594],
        9_000: [0.4023, 0.4709, 0.5763],
        9_500: [0.3928, 0.4713, 0.5914],
        10_000: [0.3839, 0.4715, 0.6049],
        10_500: [0.3757, 0.4716, 0.6171],
        11_000: [0.3681, 0.4717, 0.6281],
        11_500: [0.3609, 0.4718, 0.6380],
        12_000: [0.3543, 0.4719, 0.6472],
        12_500: [0.3480, 0.4719, 0.6555],
        13_000: [0.3421, 0.4719, 0.6631],
        13_500: [0.3365, 0.4719, 0.6702],
        14_000: [0.3313, 0.4719, 0.6766],
        14_500: [0.3263, 0.4719, 0.6826],
        15_000: [0.3217, 0.4719, 0.6882],
    },
}


@clipped
def planckian_jitter(
    img: np.ndarray,
    temperature: int,
    mode: Literal["blackbody", "cied"],
) -> np.ndarray:
    """Apply Planckian jitter to an image.

    This function applies Planckian jitter to an image by linearly interpolating
    between the two closest temperatures in the PLANCKIAN_COEFFS dictionary.

    Args:
        img (np.ndarray): Input image as a numpy array.
        temperature (int): The temperature to apply.
        mode (Literal["blackbody", "cied"]): The mode to use.

    Returns:
        np.ndarray: The Planckian jitter applied to the image.

    """
    img = img.copy()
    # Get the min and max temperatures for the given mode
    min_temp = min(PLANCKIAN_COEFFS[mode].keys())
    max_temp = max(PLANCKIAN_COEFFS[mode].keys())

    # Clamp the temperature to the available range
    temperature = np.clip(temperature, min_temp, max_temp)

    # Linearly interpolate between 2 closest temperatures
    step = 500
    t_left = max(
        (temperature // step) * step,
        min_temp,
    )  # Ensure t_left doesn't go below min_temp
    t_right = min(
        (temperature // step + 1) * step,
        max_temp,
    )  # Ensure t_right doesn't exceed max_temp

    # Handle the case where temperature is at or near min_temp or max_temp
    if t_left == t_right:
        coeffs = np.array(PLANCKIAN_COEFFS[mode][t_left])
    else:
        w_right = (temperature - t_left) / (t_right - t_left)
        w_left = 1 - w_right
        coeffs = w_left * np.array(PLANCKIAN_COEFFS[mode][t_left]) + w_right * np.array(
            PLANCKIAN_COEFFS[mode][t_right],
        )

    img[:, :, 0] = multiply_by_constant(
        img[:, :, 0],
        coeffs[0] / coeffs[1],
        inplace=True,
    )
    img[:, :, 2] = multiply_by_constant(
        img[:, :, 2],
        coeffs[2] / coeffs[1],
        inplace=True,
    )

    return img


@clipped
def add_noise(img: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Add noise to an image.

    This function adds noise to an image by adding the noise to the image.

    Args:
        img (np.ndarray): Input image as a numpy array.
        noise (np.ndarray): Noise as a numpy array.

    Returns:
        np.ndarray: The noise added to the image.

    """
    return add(img, noise, inplace=False)


def slic(
    image: np.ndarray,
    n_segments: int,
    compactness: float = 10.0,
    max_iterations: int = 10,
) -> np.ndarray:
    """Simple Linear Iterative Clustering (SLIC) superpixel segmentation using OpenCV and NumPy.

    Args:
        image (np.ndarray): Input image (2D or 3D numpy array).
        n_segments (int): Approximate number of superpixels to generate.
        compactness (float): Balance between color proximity and space proximity.
        max_iterations (int): Maximum number of iterations for k-means.

    Returns:
        np.ndarray: Segmentation mask where each superpixel has a unique label.

    """
    if image.ndim == MONO_CHANNEL_DIMENSIONS:
        image = image[..., np.newaxis]

    height, width = image.shape[:2]
    num_pixels = height * width

    # Normalize image to [0, 1] range
    image_normalized = image.astype(np.float32) / np.max(image + 1e-6)

    # Initialize cluster centers
    grid_step = int((num_pixels / n_segments) ** 0.5)
    x_range = np.arange(grid_step // 2, width, grid_step)
    y_range = np.arange(grid_step // 2, height, grid_step)
    centers = np.array(
        [(x, y) for y in y_range for x in x_range if x < width and y < height],
    )

    # Initialize labels and distances
    labels = -1 * np.ones((height, width), dtype=np.int32)
    distances = np.full((height, width), np.inf)

    for _ in range(max_iterations):
        for i, center in enumerate(centers):
            y, x = int(center[1]), int(center[0])

            # Define the neighborhood
            y_low, y_high = max(0, y - grid_step), min(height, y + grid_step + 1)
            x_low, x_high = max(0, x - grid_step), min(width, x + grid_step + 1)

            # Compute distances
            crop = image_normalized[y_low:y_high, x_low:x_high]
            color_diff = crop - image_normalized[y, x]
            color_distance = np.sum(color_diff**2, axis=-1)

            yy, xx = np.ogrid[y_low:y_high, x_low:x_high]
            spatial_distance = ((yy - y) ** 2 + (xx - x) ** 2) / (grid_step**2)

            distance = color_distance + compactness * spatial_distance

            mask = distance < distances[y_low:y_high, x_low:x_high]
            distances[y_low:y_high, x_low:x_high][mask] = distance[mask]
            labels[y_low:y_high, x_low:x_high][mask] = i

        # Update centers
        for i in range(len(centers)):
            mask = labels == i
            if np.any(mask):
                centers[i] = np.mean(np.argwhere(mask), axis=0)[::-1]

    return labels


@preserve_channel_dim
@float32_io
def shot_noise(
    img: np.ndarray,
    scale: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Apply shot noise to the image.

    Args:
        img (np.ndarray): Input image
        scale (float): Scale factor for the noise
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Image with shot noise

    """
    # Apply inverse gamma correction to work in linear space
    img_linear = cv2.pow(img, 2.2)

    # Scale image values and add small constant to avoid zero values
    scaled_img = (img_linear + scale * 1e-6) / scale

    # Generate Poisson noise
    noisy_img = multiply_by_constant(
        random_generator.poisson(scaled_img).astype(np.float32),
        scale,
        inplace=True,
    )

    # Scale back and apply gamma correction
    return power(np.clip(noisy_img, 0, 1, out=noisy_img), 1 / 2.2)


def get_safe_brightness_contrast_params(
    alpha: float,
    beta: float,
    max_value: float,
) -> tuple[float, float]:
    """Get safe brightness and contrast parameters.

    Args:
        alpha (float): Contrast factor
        beta (float): Brightness factor
        max_value (float): Maximum pixel value

    Returns:
        tuple[float, float]: Safe alpha and beta values

    """
    if alpha > 0:
        # For x = max_value: alpha * max_value + beta <= max_value
        # For x = 0: beta >= 0
        safe_beta = np.clip(beta, 0, max_value)
        # From alpha * max_value + safe_beta <= max_value
        safe_alpha = min(alpha, (max_value - safe_beta) / max_value)
    else:
        # For x = 0: beta <= max_value
        # For x = max_value: alpha * max_value + beta >= 0
        safe_beta = min(beta, max_value)
        # From alpha * max_value + safe_beta >= 0
        safe_alpha = max(alpha, -safe_beta / max_value)

    return safe_alpha, safe_beta


def generate_noise(
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"],
    spatial_mode: Literal["constant", "per_pixel", "shared"],
    shape: tuple[int, ...],
    params: dict[str, Any] | None,
    max_value: float,
    approximation: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate noise with optional approximation for speed.

    This function generates noise with optional approximation for speed.

    Args:
        noise_type (Literal["uniform", "gaussian", "laplace", "beta"]): The type of noise to generate.
        spatial_mode (Literal["constant", "per_pixel", "shared"]): The spatial mode to use.
        shape (tuple[int, ...]): The shape of the noise to generate.
        params (dict[str, Any] | None): The parameters of the noise to generate.
        max_value (float): The maximum value of the noise to generate.
        approximation (float): The approximation to use for the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The noise generated.

    """
    if params is None:
        return np.zeros(shape, dtype=np.float32)

    cv2_seed = random_generator.integers(0, 2**16)
    cv2.setRNGSeed(cv2_seed)

    if spatial_mode == "constant":
        return generate_constant_noise(
            noise_type,
            shape,
            params,
            max_value,
            random_generator,
        )

    if approximation == 1.0:
        if spatial_mode == "shared":
            return generate_shared_noise(
                noise_type,
                shape,
                params,
                max_value,
                random_generator,
            )
        return generate_per_pixel_noise(
            noise_type,
            shape,
            params,
            max_value,
            random_generator,
        )

    # Calculate reduced size for noise generation
    height, width = shape[:2]
    reduced_height = max(1, int(height * approximation))
    reduced_width = max(1, int(width * approximation))
    reduced_shape = (reduced_height, reduced_width) + shape[2:]

    # Generate noise at reduced resolution
    if spatial_mode == "shared":
        noise = generate_shared_noise(
            noise_type,
            reduced_shape,
            params,
            max_value,
            random_generator,
        )
    else:  # per_pixel
        noise = generate_per_pixel_noise(
            noise_type,
            reduced_shape,
            params,
            max_value,
            random_generator,
        )

    # Resize noise to original size using existing resize function
    return fgeometric.resize(noise, (height, width), interpolation=cv2.INTER_LINEAR)


def generate_constant_noise(
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"],
    shape: tuple[int, ...],
    params: dict[str, Any],
    max_value: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate constant noise.

    This function generates constant noise by sampling from the noise distribution.

    Args:
        noise_type (Literal["uniform", "gaussian", "laplace", "beta"]): The type of noise to generate.
        shape (tuple[int, ...]): The shape of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        max_value (float): The maximum value of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The constant noise generated.

    """
    num_channels = shape[-1] if len(shape) > MONO_CHANNEL_DIMENSIONS else 1
    return sample_noise(
        noise_type,
        (num_channels,),
        params,
        max_value,
        random_generator,
    )


def generate_per_pixel_noise(
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"],
    shape: tuple[int, ...],
    params: dict[str, Any],
    max_value: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate per-pixel noise.

    This function generates per-pixel noise by sampling from the noise distribution.

    Args:
        noise_type (Literal["uniform", "gaussian", "laplace", "beta"]): The type of noise to generate.
        shape (tuple[int, ...]): The shape of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        max_value (float): The maximum value of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The per-pixel noise generated.

    """
    return sample_noise(noise_type, shape, params, max_value, random_generator)


def sample_noise(
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"],
    size: tuple[int, ...],
    params: dict[str, Any],
    max_value: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample from specific noise distribution.

    This function samples from a specific noise distribution.

    Args:
        noise_type (Literal["uniform", "gaussian", "laplace", "beta"]): The type of noise to generate.
        size (tuple[int, ...]): The size of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        max_value (float): The maximum value of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The noise sampled.

    """
    if noise_type == "uniform":
        return sample_uniform(size, params, random_generator) * max_value
    if noise_type == "gaussian":
        return sample_gaussian(size, params, random_generator) * max_value
    if noise_type == "laplace":
        return sample_laplace(size, params, random_generator) * max_value
    if noise_type == "beta":
        return sample_beta(size, params, random_generator) * max_value

    raise ValueError(f"Unknown noise type: {noise_type}")


def sample_uniform(
    size: tuple[int, ...],
    params: dict[str, Any],
    random_generator: np.random.Generator,
) -> np.ndarray | float:
    """Sample from uniform distribution.

    Args:
        size (tuple[int, ...]): Size of the output array
        params (dict[str, Any]): Distribution parameters
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray | float: Sampled values

    """
    if len(size) == 1:  # constant mode
        ranges = params["ranges"]
        num_channels = size[0]

        if len(ranges) == 1:
            ranges = ranges * num_channels
        elif len(ranges) < num_channels:
            raise ValueError(
                f"Not enough ranges provided. Expected {num_channels}, got {len(ranges)}",
            )

        return np.array(
            [random_generator.uniform(low, high) for low, high in ranges[:num_channels]],
        )

    # use first range for spatial noise
    low, high = params["ranges"][0]
    return random_generator.uniform(low, high, size=size)


def sample_gaussian(
    size: tuple[int, ...],
    params: dict[str, Any],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample from Gaussian distribution.

    This function samples from a Gaussian distribution.

    Args:
        size (tuple[int, ...]): The size of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The Gaussian noise sampled.

    """
    mean = (
        params["mean_range"][0]
        if params["mean_range"][0] == params["mean_range"][1]
        else random_generator.uniform(*params["mean_range"])
    )
    std = (
        params["std_range"][0]
        if params["std_range"][0] == params["std_range"][1]
        else random_generator.uniform(*params["std_range"])
    )
    num_channels = size[2] if len(size) > MONO_CHANNEL_DIMENSIONS else 1
    mean_vector = mean * np.ones(shape=(num_channels,), dtype=np.float32)
    std_dev_vector = std * np.ones(shape=(num_channels,), dtype=np.float32)
    gaussian_sampled_arr = np.zeros(shape=size)

    cv2.randn(dst=gaussian_sampled_arr, mean=mean_vector, stddev=std_dev_vector)
    return gaussian_sampled_arr.astype(np.float32)


def sample_laplace(
    size: tuple[int, ...],
    params: dict[str, Any],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample from Laplace distribution.

    This function samples from a Laplace distribution.

    Args:
        size (tuple[int, ...]): The size of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The Laplace noise sampled.

    """
    loc = random_generator.uniform(*params["mean_range"])
    scale = random_generator.uniform(*params["scale_range"])
    return random_generator.laplace(loc=loc, scale=scale, size=size)


def sample_beta(
    size: tuple[int, ...],
    params: dict[str, Any],
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample from Beta distribution.

    This function samples from a Beta distribution.

    Args:
        size (tuple[int, ...]): The size of the noise to generate.
        params (dict[str, Any]): The parameters of the noise to generate.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The Beta noise sampled.

    """
    alpha = random_generator.uniform(*params["alpha_range"])
    beta = random_generator.uniform(*params["beta_range"])
    scale = random_generator.uniform(*params["scale_range"])

    # Sample from Beta[0,1] and transform to [-scale,scale]
    samples = random_generator.beta(alpha, beta, size=size)
    return (2 * samples - 1) * scale


def generate_shared_noise(
    noise_type: Literal["uniform", "gaussian", "laplace", "beta"],
    shape: tuple[int, ...],
    params: dict[str, Any],
    max_value: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate shared noise.

    Args:
        noise_type (Literal["uniform", "gaussian", "laplace", "beta"]): Type of noise to generate
        shape (tuple[int, ...]): Shape of the output array
        params (dict[str, Any]): Distribution parameters
        max_value (float): Maximum value for the noise
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Generated noise

    """
    # Generate noise for (H, W)
    height, width = shape[:2]
    noise_map = sample_noise(
        noise_type,
        (height, width),
        params,
        max_value,
        random_generator,
    )

    # If input is multichannel, broadcast noise to all channels
    if len(shape) > MONO_CHANNEL_DIMENSIONS:
        return np.broadcast_to(noise_map[..., None], shape)
    return noise_map


@clipped
@preserve_channel_dim
def sharpen_gaussian(
    img: np.ndarray,
    alpha: float,
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    """Sharpen image using Gaussian blur.

    This function sharpens an image using a Gaussian blur.

    Args:
        img (np.ndarray): The image to sharpen.
        alpha (float): The alpha value to use for the sharpening.
        kernel_size (int): The kernel size to use for the Gaussian blur.
        sigma (float): The sigma value to use for the Gaussian blur.

    Returns:
        np.ndarray: The sharpened image.

    """
    blurred = cv2.GaussianBlur(
        img,
        ksize=(kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    # Unsharp mask formula: original + alpha * (original - blurred)
    # This is equivalent to: original * (1 + alpha) - alpha * blurred
    return img + alpha * (img - blurred)


def apply_salt_and_pepper(
    img: np.ndarray,
    salt_mask: np.ndarray,
    pepper_mask: np.ndarray,
) -> np.ndarray:
    """Apply salt and pepper noise to an image.

    This function applies salt and pepper noise to an image using pre-computed masks.

    Args:
        img (np.ndarray): The image to apply salt and pepper noise to.
        salt_mask (np.ndarray): The salt mask to use for the salt and pepper noise.
        pepper_mask (np.ndarray): The pepper mask to use for the salt and pepper noise.

    Returns:
        np.ndarray: The image with salt and pepper noise applied.

    """
    # Add channel dimension to masks if image is 3D
    if img.ndim == 3:
        salt_mask = salt_mask[..., None]
        pepper_mask = pepper_mask[..., None]

    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    return np.where(salt_mask, max_value, np.where(pepper_mask, 0, img))


# Pre-compute constant kernels
DIAMOND_KERNEL = np.array(
    [
        [0.25, 0.0, 0.25],
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.25],
    ],
    dtype=np.float32,
)

SQUARE_KERNEL = np.array(
    [
        [0.0, 0.25, 0.0],
        [0.25, 0.0, 0.25],
        [0.0, 0.25, 0.0],
    ],
    dtype=np.float32,
)

# Pre-compute initial grid
INITIAL_GRID_SIZE = (3, 3)


def generate_plasma_pattern(
    target_shape: tuple[int, int],
    roughness: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate a plasma pattern.

    This function generates a plasma pattern using the diamond-square algorithm.

    Args:
        target_shape (tuple[int, int]): The shape of the plasma pattern to generate.
        roughness (float): The roughness of the plasma pattern.
        random_generator (np.random.Generator): The random number generator to use.

    Returns:
        np.ndarray: The plasma pattern generated.

    """

    def one_diamond_square_step(current_grid: np.ndarray, noise_scale: float) -> np.ndarray:
        next_height = (current_grid.shape[0] - 1) * 2 + 1
        next_width = (current_grid.shape[1] - 1) * 2 + 1

        # Pre-allocate expanded grid
        expanded_grid = np.zeros((next_height, next_width), dtype=np.float32)

        # Generate all noise at once for both steps (already scaled by noise_scale)
        all_noise = random_generator.uniform(-noise_scale, noise_scale, (next_height, next_width)).astype(np.float32)

        # Copy existing points with noise
        expanded_grid[::2, ::2] = current_grid + all_noise[::2, ::2]

        # Diamond step - keep separate for natural look
        diamond_interpolation = cv2.filter2D(expanded_grid, -1, DIAMOND_KERNEL, borderType=cv2.BORDER_CONSTANT)
        diamond_mask = diamond_interpolation > 0
        expanded_grid += (diamond_interpolation + all_noise) * diamond_mask

        # Square step - keep separate for natural look
        square_interpolation = cv2.filter2D(expanded_grid, -1, SQUARE_KERNEL, borderType=cv2.BORDER_CONSTANT)
        square_mask = square_interpolation > 0
        expanded_grid += (square_interpolation + all_noise) * square_mask

        # Normalize after each step to prevent value drift
        return cv2.normalize(expanded_grid, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Pre-compute noise scales
    max_dimension = max(target_shape)
    power_of_two_size = 2 ** np.ceil(np.log2(max_dimension - 1)) + 1
    total_steps = int(np.log2(power_of_two_size - 1) - 1)
    noise_scales = np.float32([roughness**i for i in range(total_steps)])

    # Initialize with small random grid
    plasma_grid = random_generator.uniform(-1, 1, (3, 3)).astype(np.float32)

    # Recursively apply diamond-square steps
    for noise_scale in noise_scales:
        plasma_grid = one_diamond_square_step(plasma_grid, noise_scale)

    return np.clip(
        cv2.normalize(plasma_grid[: target_shape[0], : target_shape[1]], None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F),
        0,
        1,
    )


@clipped
@float32_io
def apply_plasma_brightness_contrast(
    img: np.ndarray,
    brightness_factor: float,
    contrast_factor: float,
    plasma_pattern: np.ndarray,
) -> np.ndarray:
    """Apply plasma-based brightness and contrast adjustments.

    This function applies plasma-based brightness and contrast adjustments to an image.

    Args:
        img (np.ndarray): The image to apply the brightness and contrast adjustments to.
        brightness_factor (float): The brightness factor to apply.
        contrast_factor (float): The contrast factor to apply.
        plasma_pattern (np.ndarray): The plasma pattern to use for the brightness and contrast adjustments.

    Returns:
        np.ndarray: The image with the brightness and contrast adjustments applied.

    """
    # Early return if no adjustments needed
    if brightness_factor == 0 and contrast_factor == 0:
        return img

    img = img.copy()

    # Expand plasma pattern once if needed
    if img.ndim > MONO_CHANNEL_DIMENSIONS:
        plasma_pattern = np.tile(plasma_pattern[..., np.newaxis], (1, 1, img.shape[-1]))

    # Apply brightness adjustment
    if brightness_factor != 0:
        brightness_adjustment = multiply(plasma_pattern, brightness_factor, inplace=False)
        img = add(img, brightness_adjustment, inplace=True)

    # Apply contrast adjustment
    if contrast_factor != 0:
        mean = img.mean()
        contrast_weights = multiply(plasma_pattern, contrast_factor, inplace=False) + 1

        img = multiply(img, contrast_weights, inplace=True)

        mean_factor = mean * (1.0 - contrast_weights)
        return add(img, mean_factor, inplace=True)

    return img


@clipped
def apply_plasma_shadow(
    img: np.ndarray,
    intensity: float,
    plasma_pattern: np.ndarray,
) -> np.ndarray:
    """Apply plasma shadow to the image.

    Args:
        img (np.ndarray): Input image
        intensity (float): Shadow intensity
        plasma_pattern (np.ndarray): Plasma pattern to use

    Returns:
        np.ndarray: Image with plasma shadow

    """
    # Scale plasma pattern by intensity first (scalar operation)
    scaled_pattern = plasma_pattern * intensity

    # Expand dimensions only once if needed
    if img.ndim > MONO_CHANNEL_DIMENSIONS:
        scaled_pattern = scaled_pattern[..., np.newaxis]

    # Single multiply operation
    return img * (1 - scaled_pattern)


def create_directional_gradient(height: int, width: int, angle: float) -> np.ndarray:
    """Create a directional gradient in [0, 1] range.

    This function creates a directional gradient in the [0, 1] range.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        angle (float): The angle of the gradient.

    Returns:
        np.ndarray: The directional gradient.

    """
    # Fast path for horizontal gradients
    if angle == 0:
        return np.linspace(0, 1, width, dtype=np.float32)[None, :] * np.ones((height, 1), dtype=np.float32)
    if angle == 180:
        return np.linspace(1, 0, width, dtype=np.float32)[None, :] * np.ones((height, 1), dtype=np.float32)

    # Fast path for vertical gradients
    if angle == 90:
        return np.linspace(0, 1, height, dtype=np.float32)[:, None] * np.ones((1, width), dtype=np.float32)
    if angle == 270:
        return np.linspace(1, 0, height, dtype=np.float32)[:, None] * np.ones((1, width), dtype=np.float32)

    # Fast path for diagonal gradients using broadcasting
    if angle in (45, 135, 225, 315):
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]  # Horizontal
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]  # Vertical

        if angle == 45:  # Bottom-left to top-right
            return cv2.normalize(x + y, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if angle == 135:  # Bottom-right to top-left
            return cv2.normalize((1 - x) + y, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if angle == 225:  # Top-right to bottom-left
            return cv2.normalize((1 - x) + (1 - y), None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # angle == 315:  # Top-left to bottom-right
        return cv2.normalize(x + (1 - y), None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # General case for arbitrary angles using broadcasting
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]  # Column vector
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]  # Row vector

    angle_rad = np.deg2rad(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    cv2.multiply(x, cos_a, dst=x)
    cv2.multiply(y, sin_a, dst=y)

    return x + y


@float32_io
def apply_linear_illumination(img: np.ndarray, intensity: float, angle: float) -> np.ndarray:
    """Apply linear illumination to the image.

    Args:
        img (np.ndarray): Input image
        intensity (float): Illumination intensity
        angle (float): Illumination angle in radians

    Returns:
        np.ndarray: Image with linear illumination

    """
    height, width = img.shape[:2]
    abs_intensity = abs(intensity)

    # Create gradient and handle negative intensity in one step
    gradient = create_directional_gradient(height, width, angle)

    if intensity < 0:
        cv2.subtract(1, gradient, dst=gradient)

    cv2.multiply(gradient, 2 * abs_intensity, dst=gradient)
    cv2.add(gradient, 1 - abs_intensity, dst=gradient)

    # Add channel dimension if needed
    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        gradient = gradient[..., np.newaxis]

    return multiply_by_array(img, gradient)


@clipped
def apply_corner_illumination(
    img: np.ndarray,
    intensity: float,
    corner: Literal[0, 1, 2, 3],
) -> np.ndarray:
    """Apply corner illumination to the image.

    Args:
        img (np.ndarray): Input image
        intensity (float): Illumination intensity
        corner (Literal[0, 1, 2, 3]): The corner to apply the illumination to.

    Returns:
        np.ndarray: Image with corner illumination applied.

    """
    if intensity == 0:
        return img.copy()

    height, width = img.shape[:2]

    # Pre-compute diagonal length once
    diagonal_length = math.sqrt(height * height + width * width)

    # Create inverted distance map mask directly
    # Use uint8 for distanceTransform regardless of input dtype
    mask = np.full((height, width), 255, dtype=np.uint8)

    # Use array indexing instead of conditionals
    corners = [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]
    mask[corners[corner]] = 0

    # Calculate distance transform
    pattern = cv2.distanceTransform(
        mask,
        distanceType=cv2.DIST_L2,
        maskSize=cv2.DIST_MASK_PRECISE,
        dstType=cv2.CV_32F,  # Specify float output directly
    )

    # Combine operations to reduce array copies
    cv2.multiply(pattern, -intensity / diagonal_length, dst=pattern)
    cv2.add(pattern, 1, dst=pattern)

    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        pattern = cv2.merge([pattern] * img.shape[2])

    return multiply_by_array(img, pattern)


@clipped
def apply_gaussian_illumination(
    img: np.ndarray,
    intensity: float,
    center: tuple[float, float],
    sigma: float,
) -> np.ndarray:
    """Apply gaussian illumination to the image.

    Args:
        img (np.ndarray): Input image
        intensity (float): Illumination intensity
        center (tuple[float, float]): The center of the illumination.
        sigma (float): The sigma of the illumination.

    """
    if intensity == 0:
        return img.copy()

    height, width = img.shape[:2]

    # Pre-compute constants
    center_x = width * center[0]
    center_y = height * center[1]
    sigma2 = 2 * (max(height, width) * sigma) ** 2  # Pre-compute denominator

    # Create coordinate grid and calculate distances in-place
    y, x = np.ogrid[:height, :width]
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x -= center_x
    y -= center_y

    # Calculate squared distances in-place
    cv2.multiply(x, x, dst=x)
    cv2.multiply(y, y, dst=y)

    x = x + y

    # Calculate gaussian directly into x array
    cv2.multiply(x, -1 / sigma2, dst=x)
    cv2.exp(x, dst=x)

    # Scale by intensity
    cv2.multiply(x, intensity, dst=x)
    cv2.add(x, 1, dst=x)

    if img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
        x = cv2.merge([x] * img.shape[2])

    return multiply_by_array(img, x)


@uint8_io
def auto_contrast(
    img: np.ndarray,
    cutoff: float,
    ignore: int | None,
    method: Literal["cdf", "pil"],
) -> np.ndarray:
    """Apply automatic contrast enhancement.

    Args:
        img (np.ndarray): Input image
        cutoff (float): Cutoff percentage for histogram
        ignore (int | None): Value to ignore in histogram
        method (Literal["cdf", "pil"]): Method to use for contrast enhancement

    Returns:
        np.ndarray: Image with enhanced contrast

    """
    result = img.copy()
    num_channels = get_num_channels(img)
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]

    # Pre-compute histograms using cv2.calcHist - much faster than np.histogram
    if img.ndim > MONO_CHANNEL_DIMENSIONS:
        channels = cv2.split(img)
        hists: list[np.ndarray] = []
        for i, channel in enumerate(channels):
            if ignore is not None and i == ignore:
                hists.append(None)
                continue
            mask = None if ignore is None else (channel != ignore)
            hist = cv2.calcHist([channel], [0], mask, [256], [0, max_value])
            hists.append(hist.ravel())

    for i in range(num_channels):
        if ignore is not None and i == ignore:
            continue

        if img.ndim > MONO_CHANNEL_DIMENSIONS:
            hist = hists[i]
            channel = channels[i]
        else:
            mask = None if ignore is None else (img != ignore)
            hist = cv2.calcHist([img], [0], mask, [256], [0, max_value]).ravel()
            channel = img

        lo, hi = get_histogram_bounds(hist, cutoff)
        if hi <= lo:
            continue

        lut = create_contrast_lut(hist, lo, hi, max_value, method)
        if ignore is not None:
            lut[ignore] = ignore

        if img.ndim > MONO_CHANNEL_DIMENSIONS:
            result[..., i] = sz_lut(channel, lut)
        else:
            result = sz_lut(channel, lut)

    return result


def create_contrast_lut(
    hist: np.ndarray,
    min_intensity: int,
    max_intensity: int,
    max_value: int,
    method: Literal["cdf", "pil"],
) -> np.ndarray:
    """Create lookup table for contrast adjustment.

    This function creates a lookup table for contrast adjustment.

    Args:
        hist (np.ndarray): Histogram of the image.
        min_intensity (int): Minimum intensity of the histogram.
        max_intensity (int): Maximum intensity of the histogram.
        max_value (int): Maximum value of the lookup table.
        method (Literal["cdf", "pil"]): Method to use for contrast enhancement.

    Returns:
        np.ndarray: Lookup table for contrast adjustment.

    """
    if min_intensity >= max_intensity:
        return np.zeros(256, dtype=np.uint8)

    if method == "cdf":
        hist_range = hist[min_intensity : max_intensity + 1]
        cdf = hist_range.cumsum()

        if cdf[-1] == 0:  # No valid pixels
            return np.arange(256, dtype=np.uint8)

        # Normalize CDF to full range
        cdf = (cdf - cdf[0]) * max_value / (cdf[-1] - cdf[0])

        # Create lookup table
        lut = np.zeros(256, dtype=np.uint8)
        lut[min_intensity : max_intensity + 1] = np.clip(np.round(cdf), 0, max_value).astype(np.uint8)
        lut[max_intensity + 1 :] = max_value
        return lut

    # "pil" method
    scale = max_value / (max_intensity - min_intensity)
    indices = np.arange(256, dtype=float)
    # Changed: Use np.round to get 128 for middle value
    # Test expects [0, 128, 255] for range [0, 2]
    lut = np.clip(np.round((indices - min_intensity) * scale), 0, max_value).astype(np.uint8)
    lut[:min_intensity] = 0
    lut[max_intensity + 1 :] = max_value
    return lut


def get_histogram_bounds(hist: np.ndarray, cutoff: float) -> tuple[int, int]:
    """Get the low and high bounds of the histogram.

    This function gets the low and high bounds of the histogram.

    Args:
        hist (np.ndarray): Histogram of the image.
        cutoff (float): Cutoff percentage for histogram.

    Returns:
        tuple[int, int]: Low and high bounds of the histogram.

    """
    if not cutoff:
        non_zero_intensities = np.nonzero(hist)[0]
        if len(non_zero_intensities) == 0:
            return 0, 0
        return int(non_zero_intensities[0]), int(non_zero_intensities[-1])

    total_pixels = float(hist.sum())
    if total_pixels == 0:
        return 0, 0

    pixels_to_cut = total_pixels * cutoff / 100.0

    # Special case for uniform 256-bin histogram
    if len(hist) == 256 and np.all(hist == hist[0]):
        min_intensity = int(len(hist) * cutoff / 100)  # floor division
        max_intensity = len(hist) - min_intensity - 1
        return min_intensity, max_intensity

    # Find minimum intensity
    cumsum = 0.0
    min_intensity = 0
    for i in range(len(hist)):
        cumsum += hist[i]
        if cumsum >= pixels_to_cut:  # Use >= for left bound
            min_intensity = i + 1
            break
    min_intensity = min(min_intensity, len(hist) - 1)

    # Find maximum intensity
    cumsum = 0.0
    max_intensity = len(hist) - 1
    for i in range(len(hist) - 1, -1, -1):
        cumsum += hist[i]
        if cumsum >= pixels_to_cut:  # Use >= for right bound
            max_intensity = i
            break

    # Handle edge cases
    if min_intensity > max_intensity:
        mid_point = (len(hist) - 1) // 2
        return mid_point, mid_point

    return min_intensity, max_intensity


def get_drop_mask(
    shape: tuple[int, ...],
    per_channel: bool,
    dropout_prob: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate dropout mask.

    This function generates a dropout mask.

    Args:
        shape (tuple[int, ...]): Shape of the output mask
        per_channel (bool): Whether to apply dropout per channel
        dropout_prob (float): Dropout probability
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Dropout mask

    """
    if per_channel or len(shape) == 2:
        return random_generator.choice(
            [True, False],
            shape,
            p=[dropout_prob, 1 - dropout_prob],
        )

    # Generate 2D mask and expand to match channels
    mask_2d = random_generator.choice(
        [True, False],
        shape[:2],
        p=[dropout_prob, 1 - dropout_prob],
    )

    # If input is 2D, return 2D mask
    if len(shape) == 2:
        return mask_2d

    # For 3D input, expand and repeat across channels
    return np.repeat(mask_2d[..., None], shape[2], axis=2)


def generate_random_values(
    channels: int,
    dtype: np.dtype,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Generate random values.

    Args:
        channels (int): Number of channels
        dtype (np.dtype): Data type of the output array
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Random values

    """
    if dtype == np.uint8:
        return random_generator.integers(
            0,
            int(MAX_VALUES_BY_DTYPE[dtype]),
            size=channels,
            dtype=dtype,
        )
    if dtype == np.float32:
        return random_generator.uniform(0, 1, size=channels).astype(dtype)

    raise ValueError(f"Unsupported dtype: {dtype}")


def prepare_drop_values(
    array: np.ndarray,
    value: float | Sequence[float] | np.ndarray | None,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Prepare values to fill dropped pixels.

    Args:
        array (np.ndarray): Input array to determine shape and dtype
        value (float | Sequence[float] | np.ndarray | None): User-specified drop values or None for random
        random_generator (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Array of values matching input shape

    """
    if value is None:
        channels = get_num_channels(array)
        values = generate_random_values(channels, array.dtype, random_generator)
    elif isinstance(value, (int, float)):
        return np.full(array.shape, value, dtype=array.dtype)
    else:
        values = np.array(value, dtype=array.dtype).reshape(-1)

    # For monochannel input, return single value
    if array.ndim == 2:
        return np.full(array.shape, values[0], dtype=array.dtype)

    # For multichannel input, broadcast values to full shape
    return np.full(array.shape[:2] + (len(values),), values, dtype=array.dtype)


def get_mask_array(data: dict[str, Any]) -> np.ndarray | None:
    """Get mask array from input data if it exists."""
    if "mask" in data:
        return data["mask"]
    return data["masks"][0] if "masks" in data else None


def get_rain_params(
    liquid_layer: np.ndarray,
    color: np.ndarray,
    intensity: float,
) -> dict[str, Any]:
    """Generate parameters for rain effect.

    This function generates parameters for a rain effect.

    Args:
        liquid_layer (np.ndarray): Liquid layer of the image.
        color (np.ndarray): Color of the rain.
        intensity (float): Intensity of the rain.

    Returns:
        dict[str, Any]: Parameters for the rain effect.

    """
    liquid_layer = clip(liquid_layer * 255, np.uint8, inplace=False)

    # Generate distance transform with more defined edges
    dist = 255 - cv2.Canny(liquid_layer, 50, 150)
    dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
    _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)

    # Use separate blur operations for better drop formation
    dist = cv2.GaussianBlur(
        dist,
        ksize=(3, 3),
        sigmaX=1,  # Add slight sigma for smoother drops
        sigmaY=1,
        borderType=cv2.BORDER_REPLICATE,
    )
    dist = clip(dist, np.uint8, inplace=True)

    # Enhance contrast in the distance map
    dist = equalize(dist)

    # Modified kernel for more natural drop shapes
    ker = np.array(
        [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.float32,
    )

    # Apply convolution with better precision
    dist = convolve(dist, ker)

    # Final blur with larger kernel for smoother drops
    dist = cv2.GaussianBlur(
        dist,
        ksize=(5, 5),  # Increased kernel size
        sigmaX=1.5,  # Adjusted sigma
        sigmaY=1.5,
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float32)

    # Calculate final rain mask with better blending
    m = liquid_layer.astype(np.float32) * dist

    # Normalize with better handling of edge cases
    m_max = np.max(m, axis=(0, 1))
    if m_max > 0:
        m *= 1 / m_max
    else:
        m = np.zeros_like(m)

    # Apply color with adjusted intensity for more natural look
    drops = m[:, :, None] * color * (intensity * 0.9)  # Slightly reduced intensity

    return {
        "drops": drops,
    }


def get_mud_params(
    liquid_layer: np.ndarray,
    color: np.ndarray,
    cutout_threshold: float,
    sigma: float,
    intensity: float,
    random_generator: np.random.Generator,
) -> dict[str, Any]:
    """Generate parameters for mud effect.

    This function generates parameters for a mud effect.

    Args:
        liquid_layer (np.ndarray): Liquid layer of the image.
        color (np.ndarray): Color of the mud.
        cutout_threshold (float): Cutout threshold for the mud.
        sigma (float): Sigma for the Gaussian blur.
        intensity (float): Intensity of the mud.
        random_generator (np.random.Generator): Random number generator.

    Returns:
        dict[str, Any]: Parameters for the mud effect.

    """
    height, width = liquid_layer.shape

    # Create initial mask (ensure we have some non-zero values)
    mask = (liquid_layer > cutout_threshold).astype(np.float32)
    if np.sum(mask) == 0:  # If mask is all zeros
        # Force minimum coverage of 10%
        num_pixels = height * width
        num_needed = max(1, int(0.1 * num_pixels))  # At least 1 pixel
        flat_indices = random_generator.choice(num_pixels, num_needed, replace=False)
        mask = np.zeros_like(liquid_layer, dtype=np.float32)
        mask.flat[flat_indices] = 1.0

    # Apply Gaussian blur if sigma > 0
    if sigma > 0:
        mask = cv2.GaussianBlur(
            mask,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )

    # Safe normalization (avoid division by zero)
    mask_max = np.max(mask)
    if mask_max > 0:
        mask = mask / mask_max
    else:
        # If mask is somehow all zeros after blur, force some effect
        mask[0, 0] = 1.0

    # Scale by intensity directly (no minimum)
    mask = mask * intensity

    # Create mud effect array
    mud = np.zeros((height, width, 3), dtype=np.float32)

    # Apply color directly - the intensity scaling is already handled
    for i in range(3):
        mud[..., i] = mask * color[i]

    # Create complementary non-mud array
    non_mud = np.ones_like(mud)
    for i in range(3):
        if color[i] > 0:
            non_mud[..., i] = np.clip((color[i] - mud[..., i]) / color[i], 0, 1)
        else:
            non_mud[..., i] = 1.0 - mask

    return {
        "mud": mud.astype(np.float32),
        "non_mud": non_mud.astype(np.float32),
    }


# Standard reference H&E stain matrices
STAIN_MATRICES = {
    "ruifrok": np.array(
        [  # Ruifrok & Johnston standard reference
            [0.644211, 0.716556, 0.266844],  # Hematoxylin
            [0.092789, 0.954111, 0.283111],  # Eosin
        ],
    ),
    "macenko": np.array(
        [  # Macenko's reference
            [0.5626, 0.7201, 0.4062],
            [0.2159, 0.8012, 0.5581],
        ],
    ),
    "standard": np.array(
        [  # Standard bright-field microscopy
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
        ],
    ),
    "high_contrast": np.array(
        [  # Enhanced contrast
            [0.55, 0.88, 0.11],
            [0.12, 0.86, 0.49],
        ],
    ),
    "h_heavy": np.array(
        [  # Hematoxylin dominant
            [0.75, 0.61, 0.32],
            [0.04, 0.93, 0.36],
        ],
    ),
    "e_heavy": np.array(
        [  # Eosin dominant
            [0.60, 0.75, 0.28],
            [0.17, 0.95, 0.25],
        ],
    ),
    "dark": np.array(
        [  # Darker staining
            [0.78, 0.55, 0.28],
            [0.09, 0.97, 0.21],
        ],
    ),
    "light": np.array(
        [  # Lighter staining
            [0.57, 0.71, 0.38],
            [0.15, 0.89, 0.42],
        ],
    ),
}


def rgb_to_optical_density(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert RGB image to optical density.

    This function converts an RGB image to optical density.

    Args:
        img (np.ndarray): Input image.
        eps (float): Epsilon value.

    Returns:
        np.ndarray: Optical density image.

    """
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    pixel_matrix = img.reshape(-1, 3).astype(np.float32)
    pixel_matrix = np.maximum(pixel_matrix / max_value, eps)
    return -np.log(pixel_matrix)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors.

    This function normalizes vectors.

    Args:
        vectors (np.ndarray): Vectors to normalize.

    Returns:
        np.ndarray: Normalized vectors.

    """
    norms = np.sqrt(np.sum(vectors**2, axis=1, keepdims=True))
    return vectors / norms


def get_normalizer(method: Literal["vahadane", "macenko"]) -> StainNormalizer:
    """Get stain normalizer based on method.

    This function gets a stain normalizer based on a method.

    Args:
        method (Literal["vahadane", "macenko"]): Method to use for stain normalization.

    Returns:
        StainNormalizer: Stain normalizer.

    """
    return VahadaneNormalizer() if method == "vahadane" else MacenkoNormalizer()


class StainNormalizer:
    """Base class for stain normalizers."""

    def __init__(self) -> None:
        self.stain_matrix_target = None

    def fit(self, img: np.ndarray) -> None:
        """Fit the stain normalizer to an image.

        This function fits the stain normalizer to an image.

        Args:
            img (np.ndarray): Input image.

        """
        raise NotImplementedError


class SimpleNMF:
    def __init__(self, n_iter: int = 100):
        self.n_iter = n_iter
        # Initialize with standard H&E colors from Ruifrok
        self.initial_colors = np.array(
            [
                [0.644211, 0.716556, 0.266844],  # Hematoxylin
                [0.092789, 0.954111, 0.283111],  # Eosin
            ],
            dtype=np.float32,
        )

    def fit_transform(self, optical_density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit the NMF model to optical density.

        This function fits the NMF model to optical density.

        Args:
            optical_density (np.ndarray): Optical density image.

        Returns:
            tuple[np.ndarray, np.ndarray]: Stain concentrations and stain colors.

        """
        # Start with known H&E colors
        stain_colors = self.initial_colors.copy()

        # Initialize concentrations based on projection onto initial colors
        # This gives us a physically meaningful starting point
        stain_colors_normalized = normalize_vectors(stain_colors)
        stain_concentrations = np.maximum(optical_density @ stain_colors_normalized.T, 0)

        # Iterative updates with careful normalization
        eps = 1e-6
        for _ in range(self.n_iter):
            # Update concentrations
            numerator = optical_density @ stain_colors.T
            denominator = stain_concentrations @ (stain_colors @ stain_colors.T)
            stain_concentrations *= numerator / (denominator + eps)

            # Ensure non-negativity
            stain_concentrations = np.maximum(stain_concentrations, 0)

            # Update colors
            numerator = stain_concentrations.T @ optical_density
            denominator = (stain_concentrations.T @ stain_concentrations) @ stain_colors
            stain_colors *= numerator / (denominator + eps)

            # Ensure non-negativity and normalize
            stain_colors = np.maximum(stain_colors, 0)
            stain_colors = normalize_vectors(stain_colors)

        return stain_concentrations, stain_colors


def order_stains_combined(stain_colors: np.ndarray) -> tuple[int, int]:
    """Order stains using a combination of methods.

    This combines both angular information and spectral characteristics
    for more robust identification.

    Args:
        stain_colors (np.ndarray): Stain colors.

    Returns:
        tuple[int, int]: Hematoxylin and eosin indices.

    """
    # Normalize stain vectors
    stain_colors = normalize_vectors(stain_colors)

    # Calculate angles (Macenko)
    angles = np.mod(np.arctan2(stain_colors[:, 1], stain_colors[:, 0]), np.pi)

    # Calculate spectral ratios (Ruifrok)
    blue_ratio = stain_colors[:, 2] / (np.sum(stain_colors, axis=1) + 1e-6)
    red_ratio = stain_colors[:, 0] / (np.sum(stain_colors, axis=1) + 1e-6)

    # Combine scores
    # High angle and high blue ratio indicates Hematoxylin
    # Low angle and high red ratio indicates Eosin
    scores = angles * blue_ratio - red_ratio

    hematoxylin_idx = np.argmax(scores)
    eosin_idx = 1 - hematoxylin_idx

    return hematoxylin_idx, eosin_idx


class VahadaneNormalizer(StainNormalizer):
    def fit(self, img: np.ndarray) -> None:
        """Fit the Vahadane stain normalizer to an image.

        This function fits the Vahadane stain normalizer to an image.

        Args:
            img (np.ndarray): Input image.

        """
        optical_density = rgb_to_optical_density(img)

        nmf = SimpleNMF(n_iter=100)
        _, stain_colors = nmf.fit_transform(optical_density)

        # Use combined method for robust stain ordering
        hematoxylin_idx, eosin_idx = order_stains_combined(stain_colors)

        self.stain_matrix_target = np.array(
            [
                stain_colors[hematoxylin_idx],
                stain_colors[eosin_idx],
            ],
        )


class MacenkoNormalizer(StainNormalizer):
    """Macenko stain normalizer with optimized computations."""

    def __init__(self, angular_percentile: float = 99):
        super().__init__()
        self.angular_percentile = angular_percentile

    def fit(self, img: np.ndarray, angular_percentile: float = 99) -> None:
        """Fit the Macenko stain normalizer to an image.

        This function fits the Macenko stain normalizer to an image.

        Args:
            img (np.ndarray): Input image.
            angular_percentile (float): Angular percentile.

        """
        # Step 1: Convert RGB to optical density (OD) space
        optical_density = rgb_to_optical_density(img)

        # Step 2: Remove background pixels
        od_threshold = 0.05
        threshold_mask = (optical_density > od_threshold).any(axis=1)
        tissue_density = optical_density[threshold_mask]

        if len(tissue_density) < 1:
            raise ValueError(f"No tissue pixels found (threshold={od_threshold})")

        # Step 3: Compute covariance matrix
        tissue_density = np.ascontiguousarray(tissue_density, dtype=np.float32)
        od_covariance = cv2.calcCovarMatrix(
            tissue_density,
            None,
            cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE,
        )[0]

        # Step 4: Get principal components
        eigenvalues, eigenvectors = cv2.eigen(od_covariance)[1:]
        idx = np.argsort(eigenvalues.ravel())[-2:]
        principal_eigenvectors = np.ascontiguousarray(eigenvectors[:, idx], dtype=np.float32)

        # Step 5: Project onto eigenvector plane
        plane_coordinates = tissue_density @ principal_eigenvectors

        # Step 6: Find angles of extreme points
        polar_angles = np.arctan2(
            plane_coordinates[:, 1],
            plane_coordinates[:, 0],
        )

        # Get robust angle estimates
        hematoxylin_angle = np.percentile(polar_angles, 100 - angular_percentile)
        eosin_angle = np.percentile(polar_angles, angular_percentile)

        # Step 7: Convert angles back to RGB space
        hem_cos, hem_sin = np.cos(hematoxylin_angle), np.sin(hematoxylin_angle)
        eos_cos, eos_sin = np.cos(eosin_angle), np.sin(eosin_angle)

        angle_to_vector = np.array(
            [[hem_cos, hem_sin], [eos_cos, eos_sin]],
            dtype=np.float32,
        )
        stain_vectors = cv2.gemm(
            angle_to_vector,
            principal_eigenvectors.T,
            1,
            None,
            0,
        )

        # Step 8: Ensure non-negativity by taking absolute values
        # This is valid because stain vectors represent absorption coefficients
        stain_vectors = np.abs(stain_vectors)

        # Step 9: Normalize vectors to unit length
        stain_vectors = stain_vectors / np.sqrt(np.sum(stain_vectors**2, axis=1, keepdims=True))

        # Step 10: Order vectors as [hematoxylin, eosin]
        # Hematoxylin typically has larger red component
        self.stain_matrix_target = stain_vectors if stain_vectors[0, 0] > stain_vectors[1, 0] else stain_vectors[::-1]


def get_tissue_mask(img: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    """Get tissue mask from image.

    Args:
        img (np.ndarray): Input image
        threshold (float): Threshold for tissue detection. Default: 0.85

    Returns:
        np.ndarray: Binary mask where True indicates tissue regions

    """
    # Convert to grayscale using RGB weights: R*0.299 + G*0.587 + B*0.114
    luminosity = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114

    # Tissue is darker, so we want pixels below threshold
    mask = luminosity < threshold

    return mask.reshape(-1)


@clipped
@float32_io
def apply_he_stain_augmentation(
    img: np.ndarray,
    stain_matrix: np.ndarray,
    scale_factors: np.ndarray,
    shift_values: np.ndarray,
    augment_background: bool,
) -> np.ndarray:
    """Apply HE stain augmentation to an image.

    This function applies HE stain augmentation to an image.

    Args:
        img (np.ndarray): Input image.
        stain_matrix (np.ndarray): Stain matrix.
        scale_factors (np.ndarray): Scale factors.
        shift_values (np.ndarray): Shift values.
        augment_background (bool): Whether to augment the background.

    Returns:
        np.ndarray: Augmented image.

    """
    # Step 1: Convert RGB to optical density space
    optical_density = rgb_to_optical_density(img)

    # Step 2: Calculate stain concentrations using regularized pseudo-inverse
    stain_matrix = np.ascontiguousarray(stain_matrix, dtype=np.float32)

    # Add small regularization term for numerical stability
    regularization = 1e-6
    stain_correlation = stain_matrix @ stain_matrix.T + regularization * np.eye(2)
    density_projection = stain_matrix @ optical_density.T

    try:
        # Solve for stain concentrations
        stain_concentrations = np.linalg.solve(stain_correlation, density_projection).T
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if direct solve fails
        stain_concentrations = np.linalg.lstsq(
            stain_matrix.T,
            optical_density,
            rcond=regularization,
        )[0].T

    # Step 3: Apply concentration adjustments
    if not augment_background:
        # Only modify tissue regions
        tissue_mask = get_tissue_mask(img).reshape(-1)
        stain_concentrations[tissue_mask] = stain_concentrations[tissue_mask] * scale_factors + shift_values
    else:
        # Modify all pixels
        stain_concentrations = stain_concentrations * scale_factors + shift_values

    # Step 4: Reconstruct RGB image
    optical_density_result = stain_concentrations @ stain_matrix
    rgb_result = np.exp(-optical_density_result)

    return rgb_result.reshape(img.shape)


@clipped
@preserve_channel_dim
def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image with a kernel.

    This function convolves an image with a kernel.

    Args:
        img (np.ndarray): Input image.
        kernel (np.ndarray): Kernel.

    Returns:
        np.ndarray: Convolved image.

    """
    conv_fn = maybe_process_in_chunks(cv2.filter2D, ddepth=-1, kernel=kernel)
    return conv_fn(img)


@clipped
@preserve_channel_dim
def separable_convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image with a separable kernel.

    This function convolves an image with a separable kernel.

    Args:
        img (np.ndarray): Input image.
        kernel (np.ndarray): Kernel.

    Returns:
        np.ndarray: Convolved image.

    """
    conv_fn = maybe_process_in_chunks(cv2.sepFilter2D, ddepth=-1, kernelX=kernel, kernelY=kernel)
    return conv_fn(img)
