from __future__ import annotations

from collections import defaultdict
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
    multiply_by_constant,
    normalize_per_image,
    preserve_channel_dim,
    sz_lut,
    to_float,
    uint8_io,
)

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.utils import (
    PCA,
    handle_empty_array,
    non_rgb_error,
)
from albumentations.core.bbox_utils import bboxes_from_masks, masks_from_bboxes
from albumentations.core.types import (
    EIGHT,
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
    ColorType,
    ImageMode,
    SpatterMode,
)

__all__ = [
    "add_fog",
    "add_rain",
    "add_shadow",
    "add_gravel",
    "add_snow_bleach",
    "add_snow_texture",
    "add_sun_flare_overlay",
    "add_sun_flare_physics_based",
    "adjust_brightness_torchvision",
    "adjust_contrast_torchvision",
    "adjust_hue_torchvision",
    "adjust_saturation_torchvision",
    "brightness_contrast_adjust",
    "channel_shuffle",
    "clahe",
    "convolve",
    "downscale",
    "equalize",
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
    "swap_tiles_on_image",
    "to_gray",
    "unsharp_mask",
    "chromatic_aberration",
    "erode",
    "dilate",
    "generate_approx_gaussian_noise",
]


@uint8_io
@preserve_channel_dim
def shift_hsv(img: np.ndarray, hue_shift: float, sat_shift: float, val_shift: float) -> np.ndarray:
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
        sat = add_constant(sat, sat_shift)

    if val_shift != 0:
        val = add_constant(val, val_shift)

    img = cv2.merge((hue, sat, val))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if is_gray else img


@clipped
def solarize(img: np.ndarray, threshold: int) -> np.ndarray:
    """Invert all pixel values above a threshold.

    Args:
        img: The image to solarize.
        threshold: All pixels above this grayscale level are inverted.

    Returns:
        Solarized image.

    """
    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.uint8:
        lut = [(i if i < threshold else max_val - i) for i in range(int(max_val) + 1)]

        prev_shape = img.shape
        img = sz_lut(img, np.array(lut, dtype=dtype), inplace=False)

        return np.expand_dims(img, -1) if len(prev_shape) != len(img.shape) else img

    cond = img >= threshold
    img[cond] = max_val - img[cond]
    return img


@uint8_io
@clipped
def posterize(img: np.ndarray, bits: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]) -> np.ndarray:
    """Reduce the number of bits for each color channel.

    Args:
        img: image to posterize.
        bits: number of high bits. Must be in range [0, 8]

    Returns:
        Image with reduced color channels.

    """
    bits_array = np.uint8(bits)

    if not bits_array.shape or len(bits_array) == 1:
        if bits_array == 0:
            return np.zeros_like(img)
        if bits_array == EIGHT:
            return img

        lut = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - bits_array) - 1)
        lut &= mask

        return sz_lut(img, lut, inplace=False)

    result_img = np.empty_like(img)
    for i, channel_bits in enumerate(bits_array):
        if channel_bits == 0:
            result_img[..., i] = np.zeros_like(img[..., i])
        elif channel_bits == EIGHT:
            result_img[..., i] = img[..., i].copy()
        else:
            lut = np.arange(0, 256, dtype=np.uint8)
            mask = ~np.uint8(2 ** (8 - channel_bits) - 1)
            lut &= mask

            result_img[..., i] = sz_lut(img[..., i], lut, inplace=True)

    return result_img


def _equalize_pil(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    h = [_f for _f in histogram if _f]

    if len(h) <= 1:
        return img.copy()

    step = np.sum(h[:-1]) // 255
    if not step:
        return img.copy()

    lut = np.empty(256, dtype=np.uint8)
    n = step // 2
    for i in range(256):
        lut[i] = min(n // step, 255)
        n += histogram[i]

    return sz_lut(img, np.array(lut), inplace=True)


def _equalize_cv(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        return cv2.equalizeHist(img)

    histogram = cv2.calcHist([img], [0], mask, [256], (0, 256)).ravel()
    i = 0
    for val in histogram:
        if val > 0:
            break
        i += 1
    i = min(i, 255)

    total = np.sum(histogram)
    if histogram[i] == total:
        return np.full_like(img, i)

    scale = 255.0 / (total - histogram[i])
    _sum = 0

    lut = np.zeros(256, dtype=np.uint8)

    for idx in range(i + 1, len(histogram)):
        _sum += histogram[idx]
        lut[idx] = clip(round(_sum * scale), np.uint8)

    return sz_lut(img, lut, inplace=True)


def _check_preconditions(img: np.ndarray, mask: np.ndarray | None, by_channels: bool) -> None:
    if mask is not None:
        if is_rgb_image(mask) and is_grayscale_image(img):
            raise ValueError(f"Wrong mask shape. Image shape: {img.shape}. Mask shape: {mask.shape}")
        if not by_channels and not is_grayscale_image(mask):
            msg = f"When by_channels=False only 1-channel mask supports. Mask shape: {mask.shape}"
            raise ValueError(msg)


def _handle_mask(
    mask: np.ndarray | None,
    i: int | None = None,
) -> np.ndarray | None:
    if mask is None:
        return None
    mask = mask.astype(np.uint8)
    if is_grayscale_image(mask) or i is None:
        return mask

    return mask[..., i]


@uint8_io
@preserve_channel_dim
def equalize(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    mode: ImageMode = "cv",
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
        img: np.ndarray. Any number of channels
        low_y: per-channel or single y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y: per-channel or single y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]

    """
    t = np.linspace(0.0, 1.0, 256)

    def evaluate_bez(t: np.ndarray, low_y: float | np.ndarray, high_y: float | np.ndarray) -> np.ndarray:
        one_minus_t = 1 - t
        return (3 * one_minus_t**2 * t * low_y + 3 * one_minus_t * t**2 * high_y + t**3) * 255

    num_channels = get_num_channels(img)

    if np.isscalar(low_y) and np.isscalar(high_y):
        lut = clip(np.rint(evaluate_bez(t, low_y, high_y)), np.uint8)
        return sz_lut(img, lut, inplace=False)
    if isinstance(low_y, np.ndarray) and isinstance(high_y, np.ndarray):
        luts = clip(np.rint(evaluate_bez(t[:, np.newaxis], low_y, high_y).T), np.uint8)
        return cv2.merge(
            [sz_lut(img[:, :, i], np.ascontiguousarray(luts[i]), inplace=False) for i in range(num_channels)],
        )

    raise TypeError(
        f"low_y and high_y must both be of type float or np.ndarray. Got {type(low_y)} and {type(high_y)}",
    )


@clipped
def linear_transformation_rgb(img: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(img, transformation_matrix)


@uint8_io
@preserve_channel_dim
def clahe(img: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]) -> np.ndarray:
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
    img = img.copy()
    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if is_grayscale_image(img):
        return clahe_mat.apply(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    img[:, :, 0] = clahe_mat.apply(img[:, :, 0])

    return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)


@clipped
@preserve_channel_dim
def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    conv_fn = maybe_process_in_chunks(cv2.filter2D, ddepth=-1, kernel=kernel)
    return conv_fn(img)


@uint8_io
@preserve_channel_dim
def image_compression(img: np.ndarray, quality: int, image_type: Literal[".jpg", ".webp"]) -> np.ndarray:
    if image_type == ".jpg":
        quality_flag = cv2.IMWRITE_JPEG_QUALITY
    elif image_type == ".webp":
        quality_flag = cv2.IMWRITE_WEBP_QUALITY
    else:
        raise NotImplementedError("Only '.jpg' and '.webp' compression transforms are implemented. ")

    _, encoded_img = cv2.imencode(image_type, img, (int(quality_flag), quality))
    return cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)


@uint8_io
def add_snow_bleach(img: np.ndarray, snow_point: float, brightness_coeff: float) -> np.ndarray:
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

    snow_point *= max_value / 2
    snow_point += max_value / 3

    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    image_hls = np.array(image_hls, dtype=np.float32)

    image_hls[:, :, 1][image_hls[:, :, 1] < snow_point] *= brightness_coeff

    image_hls[:, :, 1] = clip(image_hls[:, :, 1], np.uint8)

    image_hls = np.array(image_hls, dtype=np.uint8)

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
    sparkle_mask = random_generator.random(img_shape[:2]) > 0.99  # noqa: PLR2004

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
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * (1 + brightness_coeff * snow_point), 0, max_value)

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
    snow_layer = (np.dstack([snow_texture] * 3) * max_value * snow_point).astype(np.float32)

    # Blend snow with original image
    img_with_snow = cv2.add(img_hsv, snow_layer)

    # Add a slight blue tint to simulate cool snow color
    blue_tint = np.full_like(img_with_snow, (0.6, 0.75, 1))  # Slight blue in HSV

    img_with_snow = cv2.addWeighted(img_with_snow, 0.85, blue_tint, 0.15 * snow_point, 0)

    # Convert back to RGB
    img_with_snow = cv2.cvtColor(img_with_snow.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Add some sparkle effects for snow glitter
    img_with_snow[sparkle_mask] = [max_value, max_value, max_value]

    return img_with_snow


@uint8_io
@preserve_channel_dim
def add_rain(
    img: np.ndarray,
    slant: int,
    drop_length: int,
    drop_width: int,
    drop_color: tuple[int, int, int],
    blur_value: int,
    brightness_coefficient: float,
    rain_drops: list[tuple[int, int]],
) -> np.ndarray:
    """Adds rain drops to the image.

    Args:
        img (np.ndarray): Input image.
        slant (int): The angle of the rain drops.
        drop_length (int): The length of each rain drop.
        drop_width (int): The width of each rain drop.
        drop_color (tuple[int, int, int]): The color of the rain drops in RGB format.
        blur_value (int): The size of the kernel used to blur the image. Rainy views are blurry.
        brightness_coefficient (float): Coefficient to adjust the brightness of the image. Rainy days are usually shady.
        rain_drops (list[tuple[int, int]]): A list of tuples where each tuple represents the (x, y)
            coordinates of the starting point of a rain drop.

    Returns:
        np.ndarray: Image with rain effect added.

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    for rain_drop_x0, rain_drop_y0 in rain_drops:
        rain_drop_x1 = rain_drop_x0 + slant
        rain_drop_y1 = rain_drop_y0 + drop_length

        cv2.line(
            img,
            (rain_drop_x0, rain_drop_y0),
            (rain_drop_x1, rain_drop_y1),
            drop_color,
            drop_width,
        )

    img = cv2.blur(img, (blur_value, blur_value))  # rainy view are blurry
    image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    image_hsv[:, :, 2] *= brightness_coefficient

    return cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


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
    """Add fog to the input image.

    Args:
        img (np.ndarray): Input image.
        fog_intensity (float): Intensity of the fog effect, between 0 and 1.
        alpha_coef (float): Base alpha (transparency) value for fog particles.
        fog_particle_positions (list[tuple[int, int]]): List of (x, y) coordinates for fog particles.
        fog_particle_radiuses (list[int]): List of radiuses for each fog particle.

    Returns:
        np.ndarray: Image with added fog effect.
    """
    height, width = img.shape[:2]
    num_channels = get_num_channels(img)

    fog_layer = np.zeros((height, width, num_channels), dtype=np.uint8)
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    for (x, y), radius in zip(fog_particle_positions, fog_particle_radiuses):
        color = max_value if num_channels == 1 else (max_value,) * num_channels
        cv2.circle(
            fog_layer,
            center=(x, y),
            radius=radius,
            color=color,
            thickness=-1,
        )

    # Apply gaussian blur to the fog layer
    fog_layer = cv2.GaussianBlur(fog_layer, (25, 25), 0)

    # Blend the fog layer with the original image
    alpha = np.mean(fog_layer, axis=2, keepdims=True) / max_value * alpha_coef * fog_intensity

    result = img * (1 - alpha) + fog_layer * alpha

    return clip(result, np.uint8)


@uint8_io
@preserve_channel_dim
def add_sun_flare_overlay(
    img: np.ndarray,
    flare_center: tuple[float, float],
    src_radius: int,
    src_color: ColorType,
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
        src_color (ColorType): The color of the sun, represented as a tuple of RGB values.
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

    for alpha, (x, y), rad3, (r_color, g_color, b_color) in circles:
        cv2.circle(overlay, (x, y), rad3, (r_color, g_color, b_color), -1)
        output = add_weighted(overlay, alpha, output, 1 - alpha)

    point = [int(x) for x in flare_center]

    overlay = output.copy()
    num_times = src_radius // 10
    alpha = np.linspace(0.0, 1, num=num_times)
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
    channels[0] = cv2.GaussianBlur(channels[0], (0, 0), sigmaX=3, sigmaY=3)  # Blue channel
    channels[2] = cv2.GaussianBlur(channels[2], (0, 0), sigmaX=5, sigmaY=5)  # Red channel
    flare_layer = cv2.merge(channels)

    # Blend the flare with the original image using screen blending
    return 255 - ((255 - output) * (255 - flare_layer) / 255)


@uint8_io
@preserve_channel_dim
def add_shadow(img: np.ndarray, vertices_list: list[np.ndarray], intensities: np.ndarray) -> np.ndarray:
    """Add shadows to the image by reducing the intensity of the pixel values in specified regions.

    Args:
        img (np.ndarray): Input image. Multichannel images are supported.
        vertices_list (list[np.ndarray]): List of vertices for shadow polygons.
        intensities (np.ndarray): Array of shadow intensities. Range is [0, 1].

    Returns:
        np.ndarray: Image with shadows added.

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
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
        img_shadowed[shadowed_indices] = clip(
            img_shadowed[shadowed_indices] * shadow_intensity,
            np.uint8,
        )

    return img_shadowed


@uint8_io
@clipped
@preserve_channel_dim
def add_gravel(img: np.ndarray, gravels: list[Any]) -> np.ndarray:
    non_rgb_error(img)
    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    for gravel in gravels:
        min_y, max_y, min_x, max_x, sat = gravel
        image_hls[min_y:max_y, min_x:max_x, 1] = sat

    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)


def invert(img: np.ndarray) -> np.ndarray:
    # Supports all the valid dtypes
    # clips the img to avoid unexpected behaviour.
    return MAX_VALUES_BY_DTYPE[img.dtype] - img


def channel_shuffle(img: np.ndarray, channels_shuffled: np.ndarray) -> np.ndarray:
    return img[..., channels_shuffled]


def gamma_transform(img: np.ndarray, gamma: float) -> np.ndarray:
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        return sz_lut(img, table.astype(np.uint8), inplace=False)

    return np.power(img, gamma)


def brightness_contrast_adjust(
    img: np.ndarray,
    alpha: float = 1,
    beta: float = 0,
    beta_by_max: bool = False,
) -> np.ndarray:
    if beta_by_max:
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        value = beta * max_value
    else:
        value = beta * np.mean(img)

    return multiply_add(img, alpha, value, inplace=False)


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

    luminance_noise = random_generator.poisson(stddev[1] * intensity, size=hls.shape[:2])
    color_noise = random_generator.normal(0, color_shift * intensity, size=hls.shape[:2])

    hls[..., 0] += color_noise
    hls[..., 1] = add_array(hls[..., 1], luminance_noise * intensity * (1.0 - hls[..., 1]))

    noised_hls = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
    return np.clip(noised_hls, 0, 1)  # Ensure output is in [0, 1] range


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
    method: Literal["weighted_average", "from_lab", "desaturation", "average", "max", "pca"],
) -> np.ndarray:
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


def grayscale_to_multichannel(grayscale_image: np.ndarray, num_output_channels: int = 3) -> np.ndarray:
    """Convert a grayscale image to a multi-channel image.

    This function takes a 2D grayscale image or a 3D image with a single channel
    and converts it to a multi-channel image by repeating the grayscale data
    across the specified number of channels.

    Args:
        grayscale_image (np.ndarray): Input grayscale image. Can be 2D (height, width)
                                      or 3D (height, width, 1).
        num_output_channels (int, optional): Number of channels in the output image. Defaults to 3.

    Returns:
        np.ndarray: Multi-channel image with shape (height, width, num_channels).

    Note:
        If the input is already a multi-channel image with the desired number of channels,
        it will be returned unchanged.
    """
    grayscale_image = grayscale_image.copy().squeeze()
    return np.stack([grayscale_image] * num_output_channels, axis=-1)


@preserve_channel_dim
def downscale(
    img: np.ndarray,
    scale: float,
    down_interpolation: int = cv2.INTER_AREA,
    up_interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    height, width = img.shape[:2]

    need_cast = (
        up_interpolation != cv2.INTER_NEAREST or down_interpolation != cv2.INTER_NEAREST
    ) and img.dtype == np.uint8

    if need_cast:
        img = to_float(img)

    downscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=down_interpolation)
    upscaled = cv2.resize(downscaled, (width, height), interpolation=up_interpolation)

    return from_float(upscaled, target_dtype=np.uint8) if need_cast else upscaled


def noop(input_obj: Any, **params: Any) -> Any:
    return input_obj


def swap_tiles_on_image(image: np.ndarray, tiles: np.ndarray, mapping: list[int] | None = None) -> np.ndarray:
    """Swap tiles on the image according to the new format.

    Args:
        image: Input image.
        tiles: Array of tiles with each tile as [start_y, start_x, end_y, end_x].
        mapping: list of new tile indices.

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
        new_image[start_y:end_y, start_x:end_x] = image[start_y_orig:end_y_orig, start_x_orig:end_x_orig]

    return new_image


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

    Reference:
        Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
        ImageNet classification with deep convolutional neural networks.
        In Advances in neural information processing systems (pp. 1097-1105).
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
        noise = np.dot(np.dot(eig_vecs, np.diag(alpha_vector * eig_vals)), img_centered.T).T

    # Add noise to the image
    img_pca = img_reshaped + noise

    # Reshape back to original shape
    img_pca = img_pca.reshape(orig_shape)

    # Clip values to [0, 1] range
    return np.clip(img_pca, 0, 1)


@preserve_channel_dim
def adjust_brightness_torchvision(img: np.ndarray, factor: np.ndarray) -> np:
    if factor == 0:
        return np.zeros_like(img)
    if factor == 1:
        return img

    return multiply(img, factor, inplace=False)


@preserve_channel_dim
def adjust_contrast_torchvision(img: np.ndarray, factor: float) -> np.ndarray:
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
def adjust_saturation_torchvision(img: np.ndarray, factor: float, gamma: float = 0) -> np.ndarray:
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
    sigma: float = 0.0,
    alpha: float = 0.2,
    threshold: int = 10,
) -> np.ndarray:
    blur_fn = maybe_process_in_chunks(cv2.GaussianBlur, ksize=(ksize, ksize), sigmaX=sigma)

    if image.ndim == NUM_MULTI_CHANNEL_DIMENSIONS and get_num_channels(image) == 1:
        image = np.squeeze(image, axis=-1)

    blur = blur_fn(image)
    residual = image - blur

    # Do not sharpen noise
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype(np.float32)

    sharp = image + alpha * residual
    # Avoid color noise artefacts.
    sharp = np.clip(sharp, 0, 1)

    soft_mask = blur_fn(mask)

    return add(multiply(sharp, soft_mask), multiply(image, 1 - soft_mask))


@preserve_channel_dim
def pixel_dropout(image: np.ndarray, drop_mask: np.ndarray, drop_value: float | Sequence[float]) -> np.ndarray:
    if isinstance(drop_value, (int, float)) and drop_value == 0:
        drop_values = np.zeros_like(image)
    else:
        drop_values = np.full_like(image, drop_value)
    return np.where(drop_mask, drop_values, image)


@float32_io
@clipped
@preserve_channel_dim
def spatter(
    img: np.ndarray,
    non_mud: np.ndarray | None,
    mud: np.ndarray | None,
    rain: np.ndarray | None,
    mode: SpatterMode,
) -> np.ndarray:
    if mode == "rain":
        if rain is None:
            msg = "Rain spatter requires rain mask"
            raise ValueError(msg)

        return img + rain
    if mode == "mud":
        if mud is None:
            msg = "Mud spatter requires mud mask"
            raise ValueError(msg)
        if non_mud is None:
            msg = "Mud spatter requires non_mud mask"
            raise ValueError(msg)

        return img * non_mud + mud

    raise ValueError(f"Unsupported spatter mode: {mode}")


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
    height, width = img.shape[:2]

    # Build camera matrix
    camera_mat = np.eye(3, dtype=np.float32)
    camera_mat[0, 0] = width
    camera_mat[1, 1] = height
    camera_mat[0, 2] = width / 2.0
    camera_mat[1, 2] = height / 2.0

    # Build distortion coefficients
    distortion_coeffs_red = np.array([primary_distortion_red, secondary_distortion_red, 0, 0], dtype=np.float32)
    distortion_coeffs_blue = np.array([primary_distortion_blue, secondary_distortion_blue, 0, 0], dtype=np.float32)

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
    return cv2.erode(img, kernel, iterations=1)


@preserve_channel_dim
def dilate(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.dilate(img, kernel, iterations=1)


def morphology(img: np.ndarray, kernel: np.ndarray, operation: Literal["dilation", "erosion"]) -> np.ndarray:
    if operation == "dilation":
        return dilate(img, kernel)
    if operation == "erosion":
        return erode(img, kernel)

    raise ValueError(f"Unsupported operation: {operation}")


@handle_empty_array
def bboxes_morphology(
    bboxes: np.ndarray,
    kernel: np.ndarray,
    operation: Literal["dilation", "erosion"],
    image_shape: tuple[int, int],
) -> np.ndarray:
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
def planckian_jitter(img: np.ndarray, temperature: int, mode: Literal["blackbody", "cied"]) -> np.ndarray:
    img = img.copy()
    # Get the min and max temperatures for the given mode
    min_temp = min(PLANCKIAN_COEFFS[mode].keys())
    max_temp = max(PLANCKIAN_COEFFS[mode].keys())

    # Clamp the temperature to the available range
    temperature = np.clip(temperature, min_temp, max_temp)

    # Linearly interpolate between 2 closest temperatures
    step = 500
    t_left = max((temperature // step) * step, min_temp)  # Ensure t_left doesn't go below min_temp
    t_right = min((temperature // step + 1) * step, max_temp)  # Ensure t_right doesn't exceed max_temp

    # Handle the case where temperature is at or near min_temp or max_temp
    if t_left == t_right:
        coeffs = np.array(PLANCKIAN_COEFFS[mode][t_left])
    else:
        w_right = (temperature - t_left) / (t_right - t_left)
        w_left = 1 - w_right
        coeffs = w_left * np.array(PLANCKIAN_COEFFS[mode][t_left]) + w_right * np.array(PLANCKIAN_COEFFS[mode][t_right])

    img[:, :, 0] = multiply_by_constant(img[:, :, 0], coeffs[0] / coeffs[1], inplace=True)
    img[:, :, 2] = multiply_by_constant(img[:, :, 2], coeffs[2] / coeffs[1], inplace=True)

    return img


def generate_approx_gaussian_noise(
    shape: tuple[int, ...],
    mean: float,
    sigma: float,
    scale: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    # Determine the low-resolution shape
    downscaled_height = int(shape[0] * scale)
    downsaled_width = int(shape[1] * scale)

    if len(shape) == NUM_MULTI_CHANNEL_DIMENSIONS:
        low_res_noise = random_generator.normal(mean, sigma, (downscaled_height, downsaled_width, shape[-1]))
    else:
        low_res_noise = random_generator.normal(mean, sigma, (downscaled_height, downsaled_width))

    # Upsample the noise to the original shape using OpenCV
    result = cv2.resize(low_res_noise, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return result.reshape(shape)


@clipped
def add_noise(img: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return add_array(img, noise)


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


def slic(image: np.ndarray, n_segments: int, compactness: float = 10.0, max_iterations: int = 10) -> np.ndarray:
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
    image_normalized = image.astype(np.float32) / np.max(image)

    # Initialize cluster centers
    grid_step = int((num_pixels / n_segments) ** 0.5)
    x_range = np.arange(grid_step // 2, width, grid_step)
    y_range = np.arange(grid_step // 2, height, grid_step)
    centers = np.array([(x, y) for y in y_range for x in x_range if x < width and y < height])

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
