from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence
from warnings import warn

import cv2
import numpy as np
import skimage
from albucore.functions import (
    add,
    add_array,
    add_weighted,
    from_float,
    multiply,
    multiply_add,
    normalize_per_image,
    to_float,
)
from albucore.utils import (
    MAX_VALUES_BY_DTYPE,
    clip,
    clipped,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    maybe_process_in_chunks,
    preserve_channel_dim,
)
from typing_extensions import Literal

from albumentations import random_utils
from albumentations.augmentations.utils import (
    PCA,
    non_rgb_error,
)
from albumentations.core.types import (
    EIGHT,
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
    ColorType,
    ImageMode,
    PlanckianJitterMode,
    SpatterMode,
)

__all__ = [
    "add_fog",
    "add_rain",
    "add_shadow",
    "add_gravel",
    "add_snow",
    "add_sun_flare",
    "adjust_brightness_torchvision",
    "adjust_contrast_torchvision",
    "adjust_hue_torchvision",
    "adjust_saturation_torchvision",
    "brightness_contrast_adjust",
    "center",
    "center_bbox",
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
    "split_uniform_grid",
    "chromatic_aberration",
    "erode",
    "dilate",
    "generate_approx_gaussian_noise",
]


def _shift_hsv_uint8(
    img: np.ndarray,
    hue_shift: np.ndarray,
    sat_shift: np.ndarray,
    val_shift: np.ndarray,
) -> np.ndarray:
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        lut_hue = np.arange(0, 256, dtype=np.int16)
        lut_hue = np.mod(lut_hue + hue_shift, 180).astype(dtype)
        hue = cv2.LUT(hue, lut_hue)

    sat = add(sat, sat_shift)
    val = add(val, val_shift)

    img = cv2.merge((hue, sat, val)).astype(dtype)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def _shift_hsv_non_uint8(
    img: np.ndarray,
    hue_shift: np.ndarray,
    sat_shift: np.ndarray,
    val_shift: np.ndarray,
) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        hue = cv2.add(hue, hue_shift)
        hue = np.mod(hue, 360)  # OpenCV fails with negative values

    sat = add(sat, sat_shift)
    val = add(val, val_shift)

    img = cv2.merge((hue, sat, val))
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


@preserve_channel_dim
def shift_hsv(img: np.ndarray, hue_shift: np.ndarray, sat_shift: np.ndarray, val_shift: np.ndarray) -> np.ndarray:
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

    if img.dtype == np.uint8:
        img = _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift)
    else:
        img = _shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift)

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if is_gray else img


def solarize(img: np.ndarray, threshold: int = 128) -> np.ndarray:
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
        img = cv2.LUT(img, np.array(lut, dtype=dtype))

        if len(prev_shape) != len(img.shape):
            img = np.expand_dims(img, -1)
        return img

    result_img = img.copy()
    cond = img >= threshold
    result_img[cond] = max_val - result_img[cond]
    return result_img


@preserve_channel_dim
def posterize(img: np.ndarray, bits: int) -> np.ndarray:
    """Reduce the number of bits for each color channel.

    Args:
        img: image to posterize.
        bits: number of high bits. Must be in range [0, 8]

    Returns:
        Image with reduced color channels.

    """
    bits_array = np.uint8(bits)

    if img.dtype != np.uint8:
        msg = "Image must have uint8 channel type"
        raise TypeError(msg)
    if np.any((bits_array < 0) | (bits_array > EIGHT)):
        msg = "bits must be in range [0, 8]"
        raise ValueError(msg)

    if not bits_array.shape or len(bits_array) == 1:
        if bits_array == 0:
            return np.zeros_like(img)
        if bits_array == EIGHT:
            return img.copy()

        lut = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - bits_array) - 1)
        lut &= mask

        return cv2.LUT(img, lut)

    if not is_rgb_image(img):
        msg = "If bits is iterable image must be RGB"
        raise TypeError(msg)

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

            result_img[..., i] = cv2.LUT(img[..., i], lut)

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

    return cv2.LUT(img, np.array(lut))


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

    return cv2.LUT(img, lut)


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
    original_dtype = img.dtype

    if original_dtype != np.uint8:
        img = from_float(img, dtype=np.uint8)

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

    return to_float(result_img, max_value=255) if original_dtype == np.float32 else result_img


@preserve_channel_dim
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
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.uint8)
        needs_float = True

    t = np.linspace(0.0, 1.0, 256)

    def evaluate_bez(t: np.ndarray, low_y: float | np.ndarray, high_y: float | np.ndarray) -> np.ndarray:
        one_minus_t = 1 - t
        return (3 * one_minus_t**2 * t * low_y + 3 * one_minus_t * t**2 * high_y + t**3) * 255

    num_channels = get_num_channels(img)

    if np.isscalar(low_y) and np.isscalar(high_y):
        lut = clip(np.rint(evaluate_bez(t, low_y, high_y)), np.uint8)
        output = cv2.LUT(img, lut)
    elif isinstance(low_y, np.ndarray) and isinstance(high_y, np.ndarray):
        luts = clip(np.rint(evaluate_bez(t[:, np.newaxis], low_y, high_y).T), np.uint8)
        output = cv2.merge([cv2.LUT(img[:, :, i], luts[i]) for i in range(num_channels)])
    else:
        raise TypeError(
            f"low_y and high_y must both be of type float or np.ndarray. Got {type(low_y)} and {type(high_y)}",
        )

    return to_float(output, max_value=255) if needs_float else output


@clipped
def linear_transformation_rgb(img: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(img, transformation_matrix)


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
    original_dtype = img.dtype

    if img.dtype == np.float32:
        img = from_float(img, dtype=np.uint8)

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if is_grayscale_image(img):
        result = clahe_mat.apply(img)
        return to_float(result, max_value=255) if original_dtype == np.float32 else result

    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    img[:, :, 0] = clahe_mat.apply(img[:, :, 0])

    result = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return to_float(result, max_value=255) if original_dtype == np.float32 else result


@preserve_channel_dim
def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    conv_fn = maybe_process_in_chunks(cv2.filter2D, ddepth=-1, kernel=kernel)
    return conv_fn(img)


@preserve_channel_dim
def image_compression(img: np.ndarray, quality: int, image_type: Literal[".jpg", ".webp"]) -> np.ndarray:
    if image_type == ".jpg":
        quality_flag = cv2.IMWRITE_JPEG_QUALITY
    elif image_type == ".webp":
        quality_flag = cv2.IMWRITE_WEBP_QUALITY
    else:
        raise NotImplementedError("Only '.jpg' and '.webp' compression transforms are implemented. ")

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        warn(
            "Image compression augmentation "
            "is most effective with uint8 inputs, "
            f"{input_dtype} is used as input.",
            UserWarning,
            stacklevel=2,
        )
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(f"Unexpected dtype {input_dtype} for image augmentation")

    _, encoded_img = cv2.imencode(image_type, img, (int(quality_flag), quality))
    img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

    return to_float(img, max_value=255) if needs_float else img


@preserve_channel_dim
def add_snow(img: np.ndarray, snow_point: float, brightness_coeff: float) -> np.ndarray:
    """Bleaches out pixels, imitating snow.

    Args:
        img (np.ndarray): Input image.
        snow_point (float): A float in the range [0, 1], scaled and adjusted to determine
            the threshold for pixel modification.
        brightness_coeff (float): Coefficient applied to increase the brightness of pixels below the snow_point
            threshold. Larger values lead to more pronounced snow effects.

    Returns:
        np.ndarray: Image with simulated snow effect.

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """
    non_rgb_error(img)

    input_dtype = img.dtype
    needs_float = False

    snow_point *= 127.5  # = 255 / 2
    snow_point += 85  # = 255 / 3

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True

    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    image_hls = np.array(image_hls, dtype=np.float32)

    image_hls[:, :, 1][image_hls[:, :, 1] < snow_point] *= brightness_coeff

    image_hls[:, :, 1] = clip(image_hls[:, :, 1], np.uint8)

    image_hls = np.array(image_hls, dtype=np.uint8)

    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return to_float(image_rgb, max_value=255) if needs_float else image_rgb


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
    non_rgb_error(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True

    image = img.copy()

    for rain_drop_x0, rain_drop_y0 in rain_drops:
        rain_drop_x1 = rain_drop_x0 + slant
        rain_drop_y1 = rain_drop_y0 + drop_length

        cv2.line(
            image,
            (rain_drop_x0, rain_drop_y0),
            (rain_drop_x1, rain_drop_y1),
            drop_color,
            drop_width,
        )

    image = cv2.blur(image, (blur_value, blur_value))  # rainy view are blurry
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    image_hsv[:, :, 2] *= brightness_coefficient

    image_rgb = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return to_float(image_rgb, max_value=255) if needs_float else image_rgb


@preserve_channel_dim
def add_fog(img: np.ndarray, fog_coef: float, alpha_coef: float, haze_list: list[tuple[int, int]]) -> np.ndarray:
    """Add fog to an image using the provided coefficients and haze points.

    Args:
        img (np.ndarray): The input image, expected to be a numpy array.
        fog_coef (float): The fog coefficient, used to determine the intensity of the fog.
        alpha_coef (float): The alpha coefficient, used to determine the transparency of the fog.
        haze_list (list[tuple[int, int]]): A list of tuples, where each tuple represents the x and y
            coordinates of a haze point.

    Returns:
        np.ndarray: The output image with added fog, as a numpy array.

    Raises:
        ValueError: If the input image's dtype is not uint8 or float32.

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    non_rgb_error(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(f"Unexpected dtype {input_dtype} for RandomFog augmentation")

    width = img.shape[1]

    hw = max(int(width // 3 * fog_coef), 10)

    for haze_points in haze_list:
        x, y = haze_points
        overlay = img.copy()
        output = img.copy()
        alpha = alpha_coef * fog_coef
        rad = hw // 2
        point = (x + hw // 2, y + hw // 2)
        cv2.circle(overlay, point, int(rad), (255, 255, 255), -1)
        output = add_weighted(overlay, alpha, output, 1 - alpha)

        img = output.copy()

    image_rgb = cv2.blur(img, (hw // 10, hw // 10))

    return to_float(image_rgb, max_value=255) if needs_float else image_rgb


@preserve_channel_dim
def add_sun_flare(
    img: np.ndarray,
    flare_center: tuple[float, float],
    src_radius: int,
    src_color: ColorType,
    circles: list[Any],
) -> np.ndarray:
    """Add a sun flare effect to an image.

    Args:
        img (np.ndarray): The input image.
        flare_center (tuple[float, float]): (x, y) coordinates of the flare center
        src_radius (int): The radius of the source of the flare.
        src_color (ColorType): The color of the flare, represented as a tuple of RGB values.
        circles (list[Any]): A list of tuples, each representing a circle that contributes to the flare effect.
            Each tuple contains the alpha value, the center coordinates, the radius, and the color of the circle.

    Returns:
        np.ndarray: The output image with the sun flare effect added.

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    non_rgb_error(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True

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

    return to_float(output, max_value=255) if needs_float else output


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
    input_dtype = img.dtype
    needs_float = False
    num_channels = get_num_channels(img)
    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True

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

    if needs_float:
        return to_float(img_shadowed, max_value=max_value)

    return img_shadowed


@preserve_channel_dim
def add_gravel(img: np.ndarray, gravels: list[Any]) -> np.ndarray:
    """Add gravel to the image.

    Args:
        img (numpy.ndarray): image to add gravel to
        gravels (list): list of gravel parameters. (float, float, float, float):
            (top-left x, top-left y, bottom-right x, bottom right y)

    Returns:
        numpy.ndarray:

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    non_rgb_error(img)
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(f"Unexpected dtype {input_dtype} for AddGravel augmentation")

    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    for gravel in gravels:
        y1, y2, x1, x2, sat = gravel
        image_hls[x1:x2, y1:y2, 1] = sat

    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return to_float(image_rgb, max_value=255) if needs_float else image_rgb


def invert(img: np.ndarray) -> np.ndarray:
    # Supports all the valid dtypes
    # clips the img to avoid unexpected behaviour.
    return MAX_VALUES_BY_DTYPE[img.dtype] - img


def channel_shuffle(img: np.ndarray, channels_shuffled: np.ndarray) -> np.ndarray:
    return img[..., channels_shuffled]


@preserve_channel_dim
def gamma_transform(img: np.ndarray, gamma: float) -> np.ndarray:
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        return cv2.LUT(img, table.astype(np.uint8))
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

    return multiply_add(img, alpha, value)


@clipped
def iso_noise(
    image: np.ndarray,
    color_shift: float = 0.05,
    intensity: float = 0.5,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """Apply poisson noise to an image to simulate camera sensor noise.

    Args:
        image (np.ndarray): Input image. Currently, only RGB images are supported.
        color_shift (float): The amount of color shift to apply. Default is 0.05.
        intensity (float): Multiplication factor for noise values. Values of ~0.5 produce a noticeable,
                           yet acceptable level of noise. Default is 0.5.
        random_state (np.random.RandomState | None): If specified, this will be random state used
            for noise generation.

    Returns:
        np.ndarray: The noised image.

    Image types:
        uint8, float32

    Number of channels:
        3
    """
    input_dtype = image.dtype
    factor = 1

    if input_dtype == np.uint8:
        image = to_float(image)
        factor = MAX_VALUES_BY_DTYPE[input_dtype]

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)

    luminance_noise = random_utils.poisson(stddev[1] * intensity * 255, size=hls.shape[:2], random_state=random_state)
    color_noise = random_utils.normal(0, color_shift * 360 * intensity, size=hls.shape[:2], random_state=random_state)

    hue = hls[..., 0]
    hue += color_noise
    hue %= 360

    luminance = hls[..., 1]
    luminance += (luminance_noise / 255) * (1.0 - luminance)

    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) * factor


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
    dtype = img.dtype
    img_uint8 = from_float(img, dtype=np.uint8) if dtype == np.float32 else img
    result = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)[..., 0]

    return to_float(result) if dtype == np.float32 else result


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

    return from_float(grayscale, dtype=np.uint8) if dtype == np.uint8 else grayscale


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

    Raises:
        ValueError: If the input is not a 2D grayscale image or 3D with shape (height, width, 1).

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

    return from_float(upscaled, dtype=np.uint8) if need_cast else upscaled


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


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Create bounding box from binary mask (fast version)

    Args:
        mask (numpy.ndarray): binary mask.

    Returns:
        tuple: A bounding box tuple `(x_min, y_min, x_max, y_max)`.

    """
    rows = np.any(mask, axis=1)
    if not rows.any():
        return -1, -1, -1, -1
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max + 1, y_max + 1


def mask_from_bbox(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Create binary mask from bounding box

    Args:
        img: input image
        bbox: A bounding box tuple `(x_min, y_min, x_max, y_max)`

    Returns:
        mask: binary mask

    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    x_min, y_min, x_max, y_max = bbox
    mask[y_min:y_max, x_min:x_max] = 1
    return mask


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
        any

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
    orig_dtype = img.dtype
    num_channels = get_num_channels(img)

    # Convert to float32 and scale to [0, 1] if necessary
    if orig_dtype == np.uint8:
        img = to_float(img)

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
    img_pca = np.clip(img_pca, 0, 1)

    return from_float(img_pca, dtype=orig_dtype) if orig_dtype == np.uint8 else img_pca


@preserve_channel_dim
def adjust_brightness_torchvision(img: np.ndarray, factor: np.ndarray) -> np:
    if factor == 0:
        return np.zeros_like(img)
    if factor == 1:
        return img

    return multiply(img, factor)


@preserve_channel_dim
def adjust_contrast_torchvision(img: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1:
        return img

    mean = img.mean() if is_grayscale_image(img) else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        if img.dtype != np.float32:
            mean = int(mean + 0.5)
        return np.full_like(img, mean, dtype=img.dtype)

    return multiply_add(img, factor, mean * (1 - factor))


@preserve_channel_dim
def adjust_saturation_torchvision(img: np.ndarray, factor: float, gamma: float = 0) -> np.ndarray:
    if factor == 1:
        return img

    if is_grayscale_image(img):
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray

    result = cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)
    if img.dtype == np.uint8:
        return result

    return clip(result, img.dtype)


def _adjust_hue_torchvision_uint8(img: np.ndarray, factor: float) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    img[..., 0] = cv2.LUT(img[..., 0], lut)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def adjust_hue_torchvision(img: np.ndarray, factor: float) -> np.ndarray:
    if is_grayscale_image(img):
        return img

    if factor == 0:
        return img

    if img.dtype == np.uint8:
        return _adjust_hue_torchvision_uint8(img, factor)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 0] = np.mod(img[..., 0] + factor * 360, 360)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


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
            resize_fn = maybe_process_in_chunks(cv2.resize, dsize=(new_width, new_height), interpolation=interpolation)
            image = resize_fn(image)

    segments = skimage.segmentation.slic(
        image,
        n_segments=n_segments,
        compactness=10,
        channel_axis=-1 if image.ndim > MONO_CHANNEL_DIMENSIONS else None,
    )

    min_value = 0
    max_value = MAX_VALUES_BY_DTYPE[image.dtype]
    image = np.copy(image)
    if image.ndim == MONO_CHANNEL_DIMENSIONS:
        image = image.reshape(*image.shape, 1)
    nb_channels = image.shape[2]
    for c in range(nb_channels):
        # segments+1 here because otherwise regionprops always misses the last label
        regions = skimage.measure.regionprops(segments + 1, intensity_image=image[..., c])
        for ridx, region in enumerate(regions):
            # with mod here, because slic can sometimes create more superpixel than requested.
            # replace_samples then does not have enough values, so we just start over with the first one again.
            if replace_samples[ridx % len(replace_samples)]:
                mean_intensity = region.mean_intensity
                image_sp_c = image[..., c]

                if image_sp_c.dtype.kind in ["i", "u", "b"]:
                    # After rounding the value can end up slightly outside of the value_range. Hence, we need to clip.
                    # We do clip via min(max(...)) instead of np.clip because
                    # the latter one does not seem to keep dtypes for dtypes with large itemsizes (e.g. uint64).
                    value: int | float
                    value = int(np.round(mean_intensity))
                    value = min(max(value, min_value), max_value)
                else:
                    value = mean_intensity

                image_sp_c[segments == ridx] = value

    if orig_shape != image.shape:
        resize_fn = maybe_process_in_chunks(
            cv2.resize,
            dsize=(orig_shape[1], orig_shape[0]),
            interpolation=interpolation,
        )
        return resize_fn(image)

    return image


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

    input_dtype = image.dtype

    if input_dtype == np.uint8:
        image = to_float(image)

    blur = blur_fn(image)
    residual = image - blur

    # Do not sharpen noise
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype("float32")

    sharp = image + alpha * residual
    # Avoid color noise artefacts.
    sharp = np.clip(sharp, 0, 1)

    soft_mask = blur_fn(mask)
    output = add(multiply(sharp, soft_mask), multiply(image, 1 - soft_mask))

    return from_float(output, dtype=input_dtype) if input_dtype == np.uint8 else output


@preserve_channel_dim
def pixel_dropout(image: np.ndarray, drop_mask: np.ndarray, drop_value: float | Sequence[float]) -> np.ndarray:
    if isinstance(drop_value, (int, float)) and drop_value == 0:
        drop_values = np.zeros_like(image)
    else:
        drop_values = np.full_like(image, drop_value)
    return np.where(drop_mask, drop_values, image)


@clipped
@preserve_channel_dim
def spatter(
    img: np.ndarray,
    non_mud: np.ndarray | None,
    mud: np.ndarray | None,
    rain: np.ndarray | None,
    mode: SpatterMode,
) -> np.ndarray:
    non_rgb_error(img)

    dtype = img.dtype

    img = to_float(img)

    if mode == "rain":
        if rain is None:
            msg = "Rain spatter requires rain mask"
            raise ValueError(msg)

        img += rain
    elif mode == "mud":
        if mud is None:
            msg = "Mud spatter requires mud mask"
            raise ValueError(msg)
        if non_mud is None:
            msg = "Mud spatter requires non_mud mask"
            raise ValueError(msg)

        img = img * non_mud + mud
    else:
        raise ValueError("Unsupported spatter mode: " + str(mode))

    return from_float(img, dtype=dtype)


def almost_equal_intervals(n: int, parts: int) -> np.ndarray:
    """Generates an array of nearly equal integer intervals that sum up to `n`.

    This function divides the number `n` into `parts` nearly equal parts. It ensures that
    the sum of all parts equals `n`, and the difference between any two parts is at most one.
    This is useful for distributing a total amount into nearly equal discrete parts.

    Args:
        n (int): The total value to be split.
        parts (int): The number of parts to split into.

    Returns:
        np.ndarray: An array of integers where each integer represents the size of a part.

    Example:
        >>> almost_equal_intervals(20, 3)
        array([7, 7, 6])  # Splits 20 into three parts: 7, 7, and 6
        >>> almost_equal_intervals(16, 4)
        array([4, 4, 4, 4])  # Splits 16 into four equal parts
    """
    part_size, remainder = divmod(n, parts)
    # Create an array with the base part size and adjust the first `remainder` parts by adding 1
    return np.array([part_size + 1 if i < remainder else part_size for i in range(parts)])


def generate_shuffled_splits(
    size: int,
    divisions: int,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """Generate shuffled splits for a given dimension size and number of divisions.

    Args:
        size (int): Total size of the dimension (height or width).
        divisions (int): Number of divisions (rows or columns).
        random_state (Optional[np.random.RandomState]): Seed for the random number generator for reproducibility.

    Returns:
        np.ndarray: Cumulative edges of the shuffled intervals.
    """
    intervals = almost_equal_intervals(size, divisions)
    intervals = random_utils.shuffle(intervals, random_state=random_state)
    return np.insert(np.cumsum(intervals), 0, 0)


def split_uniform_grid(
    image_shape: tuple[int, int],
    grid: tuple[int, int],
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """Splits an image shape into a uniform grid specified by the grid dimensions.

    Args:
        image_shape (tuple[int, int]): The shape of the image as (height, width).
        grid (tuple[int, int]): The grid size as (rows, columns).
        random_state (Optional[np.random.RandomState]): The random state to use for shuffling the splits.
            If None, the splits are not shuffled.

    Returns:
        np.ndarray: An array containing the tiles' coordinates in the format (start_y, start_x, end_y, end_x).

    Note:
        The function uses `generate_shuffled_splits` to generate the splits for the height and width of the image.
        The splits are then used to calculate the coordinates of the tiles.
    """
    n_rows, n_cols = grid

    height_splits = generate_shuffled_splits(image_shape[0], grid[0], random_state)
    width_splits = generate_shuffled_splits(image_shape[1], grid[1], random_state)

    # Calculate tiles coordinates
    tiles = [
        (height_splits[i], width_splits[j], height_splits[i + 1], width_splits[j + 1])
        for i in range(n_rows)
        for j in range(n_cols)
    ]

    return np.array(tiles)


def create_shape_groups(tiles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Groups tiles by their shape and stores the indices for each shape."""
    shape_groups = defaultdict(list)
    for index, (start_y, start_x, end_y, end_x) in enumerate(tiles):
        shape = (end_y - start_y, end_x - start_x)
        shape_groups[shape].append(index)
    return shape_groups


def shuffle_tiles_within_shape_groups(
    shape_groups: dict[tuple[int, int], list[int]],
    random_state: np.random.RandomState | None = None,
) -> list[int]:
    """Shuffles indices within each group of similar shapes and creates a list where each
    index points to the index of the tile it should be mapped to.

    Args:
        shape_groups (dict[tuple[int, int], list[int]]): Groups of tile indices categorized by shape.
        random_state (Optional[np.random.RandomState]): Seed for the random number generator for reproducibility.

    Returns:
        list[int]: A list where each index is mapped to the new index of the tile after shuffling.
    """
    # Initialize the output list with the same size as the total number of tiles, filled with -1
    num_tiles = sum(len(indices) for indices in shape_groups.values())
    mapping = [-1] * num_tiles

    # Prepare the random number generator

    for indices in shape_groups.values():
        shuffled_indices = random_utils.shuffle(indices.copy(), random_state=random_state)
        for old, new in zip(indices, shuffled_indices):
            mapping[old] = new

    return mapping


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


def morphology(img: np.ndarray, kernel: np.ndarray, operation: str) -> np.ndarray:
    if operation == "dilation":
        return dilate(img, kernel)
    if operation == "erosion":
        return erode(img, kernel)

    raise ValueError(f"Unsupported operation: {operation}")


def center(image_shape: tuple[int, int]) -> tuple[float, float]:
    """Calculate the center coordinates if image. Used by images, masks and keypoints.

    Args:
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        tuple[float, float]: The center coordinates.
    """
    height, width = image_shape[:2]
    return width / 2 - 0.5, height / 2 - 0.5


def center_bbox(image_shape: tuple[int, int]) -> tuple[float, float]:
    """Calculate the center coordinates for of image for bounding boxes.

    Args:
        image_shape (tuple[int, int]): The shape of the image.

    Returns:
        tuple[float, float]: The center coordinates.
    """
    height, width = image_shape[:2]
    return width / 2, height / 2


PLANCKIAN_COEFFS = {
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
def planckian_jitter(img: np.ndarray, temperature: int, mode: PlanckianJitterMode = "blackbody") -> np.ndarray:
    img = img.copy()
    # Linearly interpolate between 2 closest temperatures
    step = 500
    t_left = (temperature // step) * step
    t_right = (temperature // step + 1) * step

    w_left = (t_right - temperature) / step
    w_right = (temperature - t_left) / step

    coeffs = w_left * np.array(PLANCKIAN_COEFFS[mode][t_left]) + w_right * np.array(PLANCKIAN_COEFFS[mode][t_right])

    image = to_float(img) if img.dtype == np.uint8 else img

    image[:, :, 0] = image[:, :, 0] * (coeffs[0] / coeffs[1])
    image[:, :, 2] = image[:, :, 2] * (coeffs[2] / coeffs[1])
    image[image > 1] = 1

    return from_float(image, dtype=img.dtype) if img.dtype == np.uint8 else image


def generate_approx_gaussian_noise(
    shape: tuple[int, ...],
    mean: float = 0,
    sigma: float = 1,
    scale: float = 0.25,
) -> np.ndarray:
    # Determine the low-resolution shape
    downscaled_height = int(shape[0] * scale)
    downsaled_width = int(shape[1] * scale)

    if len(shape) == NUM_MULTI_CHANNEL_DIMENSIONS:
        low_res_noise = random_utils.normal(mean, sigma, (downscaled_height, downsaled_width, shape[-1]))
    else:
        low_res_noise = random_utils.normal(mean, sigma, (downscaled_height, downsaled_width))

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
