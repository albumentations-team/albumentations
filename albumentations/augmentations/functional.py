from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import cv2
import numpy as np
import skimage
from albucore.functions import add, add_array, add_weighted, multiply, multiply_add
from albucore.utils import (
    MAX_VALUES_BY_DTYPE,
    clip,
    clipped,
    contiguous,
    is_grayscale_image,
    is_rgb_image,
    maybe_process_in_chunks,
    preserve_channel_dim,
)
from typing_extensions import Literal

from albumentations import random_utils
from albumentations.augmentations.utils import (
    non_rgb_warning,
)
from albumentations.core.types import (
    EIGHT,
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    ColorType,
    ImageMode,
    NumericType,
    PlanckianJitterMode,
    SizeType,
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
    "channel_shuffle",
    "clahe",
    "convolve",
    "downscale",
    "equalize",
    "fancy_pca",
    "from_float",
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
    "to_float",
    "to_gray",
    "gray_to_rgb",
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

    if is_gray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


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


def _equalize_pil(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
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


def _equalize_cv(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
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


def _check_preconditions(img: np.ndarray, mask: Optional[np.ndarray], by_channels: bool) -> None:
    if img.dtype != np.uint8:
        msg = "Image must have uint8 channel type"
        raise TypeError(msg)

    if mask is not None:
        if is_rgb_image(mask) and is_grayscale_image(img):
            raise ValueError(f"Wrong mask shape. Image shape: {img.shape}. Mask shape: {mask.shape}")
        if not by_channels and not is_grayscale_image(mask):
            msg = f"When by_channels=False only 1-channel mask supports. Mask shape: {mask.shape}"
            raise ValueError(msg)


def _handle_mask(
    mask: Optional[np.ndarray],
    img: np.ndarray,
    by_channels: bool,
    i: Optional[int] = None,
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask = mask.astype(np.uint8)
    if is_grayscale_image(mask) or i is None:
        return mask

    return mask[..., i]


@preserve_channel_dim
def equalize(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    mode: ImageMode = "cv",
    by_channels: bool = True,
) -> np.ndarray:
    _check_preconditions(img, mask, by_channels)

    function = _equalize_pil if mode == "pil" else _equalize_cv

    if is_grayscale_image(img):
        return function(img, _handle_mask(mask, img, by_channels))

    if not by_channels:
        result_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        result_img[..., 0] = function(result_img[..., 0], _handle_mask(mask, img, by_channels))
        return cv2.cvtColor(result_img, cv2.COLOR_YCrCb2RGB)

    result_img = np.empty_like(img)
    for i in range(3):
        _mask = _handle_mask(mask, img, by_channels, i)
        result_img[..., i] = function(img[..., i], _mask)

    return result_img


@preserve_channel_dim
def move_tone_curve(img: np.ndarray, low_y: float, high_y: float) -> np.ndarray:
    """Rescales the relationship between bright and dark areas of the image by manipulating its tone curve.

    Args:
        img: RGB or grayscale image.
        low_y: y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y: y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]

    """
    input_dtype = img.dtype

    if not 0 <= low_y <= 1:
        msg = "low_shift must be in range [0, 1]"
        raise ValueError(msg)
    if not 0 <= high_y <= 1:
        msg = "high_shift must be in range [0, 1]"
        raise ValueError(msg)

    if input_dtype != np.uint8:
        raise ValueError(f"Unsupported image type {input_dtype}")

    t = np.linspace(0.0, 1.0, 256)

    # Defines response of a four-point Bezier curve
    def evaluate_bez(t: np.ndarray) -> np.ndarray:
        return 3 * (1 - t) ** 2 * t * low_y + 3 * (1 - t) * t**2 * high_y + t**3

    evaluate_bez = np.vectorize(evaluate_bez)
    remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)

    lut_fn = maybe_process_in_chunks(cv2.LUT, lut=remapping)
    return lut_fn(img)


@clipped
def linear_transformation_rgb(img: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    return cv2.transform(img, transformation_matrix)


@preserve_channel_dim
def clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    if img.dtype != np.uint8:
        msg = "clahe supports only uint8 inputs"
        raise TypeError(msg)

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=[int(x) for x in tile_grid_size])

    if is_grayscale_image(img):
        return clahe_mat.apply(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
    return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)


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
        NotImplementedError("Only '.jpg' and '.webp' compression transforms are implemented. ")

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

    if needs_float:
        img = to_float(img, max_value=255)
    return img


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
    non_rgb_warning(img)

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

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


@preserve_channel_dim
def add_rain(
    img: np.ndarray,
    slant: int,
    drop_length: int,
    drop_width: int,
    drop_color: Tuple[int, int, int],
    blur_value: int,
    brightness_coefficient: float,
    rain_drops: List[Tuple[int, int]],
) -> np.ndarray:
    """Adds rain drops to the image.

    Args:
        img (np.ndarray): Input image.
        slant (int): The angle of the rain drops.
        drop_length (int): The length of each rain drop.
        drop_width (int): The width of each rain drop.
        drop_color (Tuple[int, int, int]): The color of the rain drops in RGB format.
        blur_value (int): The size of the kernel used to blur the image. Rainy views are blurry.
        brightness_coefficient (float): Coefficient to adjust the brightness of the image. Rainy days are usually shady.
        rain_drops (List[Tuple[int, int]]): A list of tuples where each tuple represents the (x, y)
            coordinates of the starting point of a rain drop.

    Returns:
        np.ndarray: Image with rain effect added.

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    non_rgb_warning(img)

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

    if needs_float:
        return to_float(image_rgb, max_value=255)

    return image_rgb


@preserve_channel_dim
def add_fog(img: np.ndarray, fog_coef: float, alpha_coef: float, haze_list: List[Tuple[int, int]]) -> np.ndarray:
    """Add fog to an image using the provided coefficients and haze points.

    Args:
        img (np.ndarray): The input image, expected to be a numpy array.
        fog_coef (float): The fog coefficient, used to determine the intensity of the fog.
        alpha_coef (float): The alpha coefficient, used to determine the transparency of the fog.
        haze_list (List[Tuple[int, int]]): A list of tuples, where each tuple represents the x and y
            coordinates of a haze point.

    Returns:
        np.ndarray: The output image with added fog, as a numpy array.

    Raises:
        ValueError: If the input image's dtype is not uint8 or float32.

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    non_rgb_warning(img)

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

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


@preserve_channel_dim
def add_sun_flare(
    img: np.ndarray,
    flare_center_x: float,
    flare_center_y: float,
    src_radius: int,
    src_color: ColorType,
    circles: List[Any],
) -> np.ndarray:
    """Add a sun flare effect to an image.

    Args:
        img (np.ndarray): The input image.
        flare_center_x (float): The x-coordinate of the flare center.
        flare_center_y (float): The y-coordinate of the flare center.
        src_radius (int): The radius of the source of the flare.
        src_color (ColorType): The color of the flare, represented as a tuple of RGB values.
        circles (List[Any]): A list of tuples, each representing a circle that contributes to the flare effect.
            Each tuple contains the alpha value, the center coordinates, the radius, and the color of the circle.

    Returns:
        np.ndarray: The output image with the sun flare effect added.

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    non_rgb_warning(img)

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

    point = (int(flare_center_x), int(flare_center_y))

    overlay = output.copy()
    num_times = src_radius // 10
    alpha = np.linspace(0.0, 1, num=num_times)
    rad = np.linspace(1, src_radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times - i - 1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        output = add_weighted(overlay, alp, output, 1 - alp)

    if needs_float:
        return to_float(output, max_value=255)

    return output


@contiguous
@preserve_channel_dim
def add_shadow(img: np.ndarray, vertices_list: List[np.ndarray]) -> np.ndarray:
    """Add shadows to the image by reducing the intensity of the RGB values in specified regions.

    Args:
        img (np.ndarray): Input image.
        vertices_list (List[np.ndarray]): List of vertices for shadow polygons.

    Returns:
        np.ndarray: Image with shadows added.

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """
    non_rgb_warning(img)
    input_dtype = img.dtype
    needs_float = False

    max_value = MAX_VALUES_BY_DTYPE[np.uint8]

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True

    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, vertices_list, (max_value, max_value, max_value))

    # Apply shadow to the RGB channels directly
    # It could be tempting to convert to HLS and apply the shadow to the L channel, but it creates artifacts
    shadow_intensity = 0.5  # Adjust this value to control the shadow intensity
    img_shadowed = img.copy()
    shadowed_indices = mask[:, :, 0] == max_value
    img_shadowed[shadowed_indices] = clip(img_shadowed[shadowed_indices] * shadow_intensity, np.uint8)

    if needs_float:
        return to_float(img_shadowed, max_value=max_value)

    return img_shadowed


@contiguous
@preserve_channel_dim
def add_gravel(img: np.ndarray, gravels: List[Any]) -> np.ndarray:
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
    non_rgb_warning(img)
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

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


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
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Apply poisson noise to an image to simulate camera sensor noise.

    Args:
        image (np.ndarray): Input image. Currently, only RGB, uint8 images are supported.
        color_shift (float): The amount of color shift to apply. Default is 0.05.
        intensity (float): Multiplication factor for noise values. Values of ~0.5 produce a noticeable,
                           yet acceptable level of noise. Default is 0.5.
        random_state (Optional[int]): If specified, this will set the random seed for the noise generation,
                                      ensuring consistent results for the same input and seed.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The noised image.

    Raises:
        TypeError: If the input image's dtype is not uint8 or if the image is not RGB.
    """
    if image.dtype != np.uint8:
        msg = "Image must have uint8 channel type"
        raise TypeError(msg)
    if not is_rgb_image(image):
        msg = "Image must be RGB"
        raise TypeError(msg)

    one_over_255 = float(1.0 / 255.0)
    image = multiply(image, one_over_255).astype(np.float32)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)

    luminance_noise = random_utils.poisson(stddev[1] * intensity * 255, size=hls.shape[:2], random_state=random_state)
    color_noise = random_utils.normal(0, color_shift * 360 * intensity, size=hls.shape[:2], random_state=random_state)

    hue = hls[..., 0]
    hue += color_noise
    hue %= 360

    luminance = hls[..., 1]
    luminance += (luminance_noise / 255) * (1.0 - luminance)

    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) * 255
    return image.astype(np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


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
    if need_cast:
        return from_float(np.clip(upscaled, 0, 1), dtype=np.dtype("uint8"))
    return upscaled


def to_float(img: np.ndarray, max_value: Optional[float] = None) -> np.ndarray:
    if max_value is None:
        if img.dtype not in MAX_VALUES_BY_DTYPE:
            raise RuntimeError(f"Unsupported dtype {img.dtype}. Specify 'max_value' manually.")
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]

    return (img / max_value).astype(np.float32)


def from_float(img: np.ndarray, dtype: np.dtype, max_value: Optional[float] = None) -> np.ndarray:
    if max_value is None:
        if dtype not in MAX_VALUES_BY_DTYPE:
            msg = (
                f"Can't infer the maximum value for dtype {dtype}. "
                "You need to specify the maximum value manually by passing the max_value argument."
            )
            raise RuntimeError(msg)
        max_value = MAX_VALUES_BY_DTYPE[dtype]
    return (img * max_value).astype(dtype)


def noop(input_obj: Any, **params: Any) -> Any:
    return input_obj


def swap_tiles_on_image(image: np.ndarray, tiles: np.ndarray, mapping: Optional[List[int]] = None) -> np.ndarray:
    """Swap tiles on the image according to the new format.

    Args:
        image: Input image.
        tiles: Array of tiles with each tile as [start_y, start_x, end_y, end_x].
        mapping: List of new tile indices.

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


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
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


def mask_from_bbox(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
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


def fancy_pca(img: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Perform 'Fancy PCA' augmentation


    Args:
        img: numpy array with (h, w, rgb) shape, as ints between 0-255
        alpha: how much to perturb/scale the eigen vectors and values
                the paper used std=0.1

    Returns:
        numpy image-like array as uint8 range(0, 255)

    Reference:
        http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    """
    if not is_rgb_image(img) or img.dtype != np.uint8:
        msg = "Image must be RGB image in uint8 format."
        raise TypeError(msg)

    orig_img = img.astype(float).copy()

    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # > get [p1, p2, p3]
    m1 = np.column_stack(eig_vecs)

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    # > alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.array(m1) @ np.array(m2)

    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx] * 255

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # > orig_img /= 255.0
    return clip(orig_img, np.uint8)


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
    max_size: Optional[int],
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
                    value: Union[int, float]
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
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError(f"Unexpected dtype {input_dtype} for UnsharpMask augmentation")

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

    return from_float(output, dtype=input_dtype)


@preserve_channel_dim
def pixel_dropout(image: np.ndarray, drop_mask: np.ndarray, drop_value: Union[float, Sequence[float]]) -> np.ndarray:
    if isinstance(drop_value, (int, float)) and drop_value == 0:
        drop_values = np.zeros_like(image)
    else:
        drop_values = np.full_like(image, drop_value)
    return np.where(drop_mask, drop_values, image)


@clipped
@preserve_channel_dim
def spatter(
    img: np.ndarray,
    non_mud: Optional[np.ndarray],
    mud: Optional[np.ndarray],
    rain: Optional[np.ndarray],
    mode: SpatterMode,
) -> np.ndarray:
    non_rgb_warning(img)

    coef = MAX_VALUES_BY_DTYPE[img.dtype]
    img = img.astype(np.float32) * (1 / coef)

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

    return img * 255


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
    random_state: Optional[np.random.RandomState] = None,
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
    image_shape: Tuple[int, int],
    grid: Tuple[int, int],
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Splits an image shape into a uniform grid specified by the grid dimensions.

    Args:
        image_shape (Tuple[int, int]): The shape of the image as (height, width).
        grid (Tuple[int, int]): The grid size as (rows, columns).
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


def create_shape_groups(tiles: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """Groups tiles by their shape and stores the indices for each shape."""
    shape_groups = defaultdict(list)
    for index, (start_y, start_x, end_y, end_x) in enumerate(tiles):
        shape = (end_y - start_y, end_x - start_x)
        shape_groups[shape].append(index)
    return shape_groups


def shuffle_tiles_within_shape_groups(
    shape_groups: Dict[Tuple[int, int], List[int]],
    random_state: Optional[np.random.RandomState] = None,
) -> List[int]:
    """Shuffles indices within each group of similar shapes and creates a list where each
    index points to the index of the tile it should be mapped to.

    Args:
        shape_groups (Dict[Tuple[int, int], List[int]]): Groups of tile indices categorized by shape.
        random_state (Optional[np.random.RandomState]): Seed for the random number generator for reproducibility.

    Returns:
        List[int]: A list where each index is mapped to the new index of the tile after shuffling.
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
    non_rgb_warning(img)

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


def center(width: NumericType, height: NumericType) -> Tuple[float, float]:
    """Calculate the center coordinates of a rectangle.

    Args:
        width (NumericType): The width of the rectangle.
        height (NumericType): The height of the rectangle.

    Returns:
        Tuple[float, float]: The center coordinates of the rectangle.
    """
    return width / 2 - 0.5, height / 2 - 0.5


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

    image = img / 255.0 if img.dtype == np.uint8 else img

    image[:, :, 0] = image[:, :, 0] * (coeffs[0] / coeffs[1])
    image[:, :, 2] = image[:, :, 2] * (coeffs[2] / coeffs[1])
    image[image > 1] = 1

    if img.dtype == np.uint8:
        return image * 255.0

    return image


def generate_approx_gaussian_noise(
    shape: SizeType,
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
