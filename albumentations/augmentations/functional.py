from __future__ import division

from typing import Optional, Sequence, Union
from warnings import warn

import cv2
import numpy as np
import skimage

from albumentations import random_utils
from albumentations.augmentations.utils import (
    MAX_VALUES_BY_DTYPE,
    _maybe_process_in_chunks,
    clip,
    clipped,
    ensure_contiguous,
    is_grayscale_image,
    is_rgb_image,
    non_rgb_warning,
    preserve_channel_dim,
    preserve_shape,
)

__all__ = [
    "add_fog",
    "add_rain",
    "add_shadow",
    "add_snow",
    "add_sun_flare",
    "add_weighted",
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
    "gauss_noise",
    "image_compression",
    "invert",
    "iso_noise",
    "linear_transformation_rgb",
    "move_tone_curve",
    "multiply",
    "noop",
    "normalize",
    "paste",
    "posterize",
    "shift_hsv",
    "shift_rgb",
    "solarize",
    "superpixels",
    "swap_tiles_on_image",
    "to_float",
    "to_gray",
    "unsharp_mask",
]


def normalize_cv2(img, mean, denominator):
    if mean.shape and len(mean) != 4 and mean.shape != img.shape:
        mean = np.array(mean.tolist() + [0] * (4 - len(mean)), dtype=np.float64)
    if not denominator.shape:
        denominator = np.array([denominator.tolist()] * 4, dtype=np.float64)
    elif len(denominator) != 4 and denominator.shape != img.shape:
        denominator = np.array(denominator.tolist() + [1] * (4 - len(denominator)), dtype=np.float64)

    img = np.ascontiguousarray(img.astype("float32"))
    cv2.subtract(img, mean.astype(np.float64), img)
    cv2.multiply(img, denominator.astype(np.float64), img)
    return img


def normalize_numpy(img, mean, denominator):
    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        return normalize_cv2(img, mean, denominator)
    return normalize_numpy(img, mean, denominator)


def _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        lut_hue = np.arange(0, 256, dtype=np.int16)
        lut_hue = np.mod(lut_hue + hue_shift, 180).astype(dtype)
        hue = cv2.LUT(hue, lut_hue)

    if sat_shift != 0:
        lut_sat = np.arange(0, 256, dtype=np.int16)
        lut_sat = np.clip(lut_sat + sat_shift, 0, 255).astype(dtype)
        sat = cv2.LUT(sat, lut_sat)

    if val_shift != 0:
        lut_val = np.arange(0, 256, dtype=np.int16)
        lut_val = np.clip(lut_val + val_shift, 0, 255).astype(dtype)
        val = cv2.LUT(val, lut_val)

    img = cv2.merge((hue, sat, val)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def _shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        hue = cv2.add(hue, hue_shift)
        hue = np.mod(hue, 360)  # OpenCV fails with negative values

    if sat_shift != 0:
        sat = clip(cv2.add(sat, sat_shift), dtype, 1.0)

    if val_shift != 0:
        val = clip(cv2.add(val, val_shift), dtype, 1.0)

    img = cv2.merge((hue, sat, val))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


@preserve_shape
def shift_hsv(img, hue_shift, sat_shift, val_shift):
    if hue_shift == 0 and sat_shift == 0 and val_shift == 0:
        return img

    is_gray = is_grayscale_image(img)
    if is_gray:
        if hue_shift != 0 or sat_shift != 0:
            hue_shift = 0
            sat_shift = 0
            warn(
                "HueSaturationValue: hue_shift and sat_shift are not applicable to grayscale image. "
                "Set them to 0 or use RGB image"
            )
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.dtype == np.uint8:
        img = _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift)
    else:
        img = _shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift)

    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


def solarize(img, threshold=128):
    """Invert all pixel values above a threshold.

    Args:
        img (numpy.ndarray): The image to solarize.
        threshold (int): All pixels above this greyscale level are inverted.

    Returns:
        numpy.ndarray: Solarized image.

    """
    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.dtype("uint8"):
        lut = [(i if i < threshold else max_val - i) for i in range(max_val + 1)]

        prev_shape = img.shape
        img = cv2.LUT(img, np.array(lut, dtype=dtype))

        if len(prev_shape) != len(img.shape):
            img = np.expand_dims(img, -1)
        return img

    result_img = img.copy()
    cond = img >= threshold
    result_img[cond] = max_val - result_img[cond]
    return result_img


BLEND_METHODS = ("GAUSSIAN", "NORMAL_CLONE", "MIXED_CLONE", "MONOCHROME_TRANSFER")
CV2_BLEND_METHOD_MAP = {
    "NORMAL_CLONE": cv2.NORMAL_CLONE,
    "MIXED_CLONE": cv2.MIXED_CLONE,
    "MONOCHROME_TRANSFER": cv2.MONOCHROME_TRANSFER,
}


def paste(
    base_img: np.ndarray,
    obj_img: np.ndarray,
    obj_mask: np.ndarray,
    obj_top: int,
    obj_left: int,
    sigma: float = 1,
    blend_method: str = "GAUSSIAN",
):
    """
    Args:
        base_img (numpy.ndarray): An image where another object will be pasted onto it.
        obj_img (numpy.ndarray): An image to be pasted onto the base_img.
        obj_mask (numpy.ndarray): An mask cover the pasted object. The height and with should be the same
                                  as the obj_img's ones.
        obj_top (int): The y position of the top of the obj_img in the base_img coordinate.
        obj_left (int): The x position of the left of the obj_img in the base_img cooridate.
        sigma (float): Standard deviation for Gaussian kernel used in the "GAUSSIAN" blending method.
        blend_method (str): keyword to select a blending method. Should be one of:
                            ["GAUSSIAN", "NORMAL_CLONE", "MIXED_CLONE", "MONOCHROME_TRANSFER"].
                            Lower cases will be capitalized. Default is "GAUSSIAN".
    Returns:
        numpy.ndarray: Blended image.

    Reference:
        Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, Barret Zoph:
        "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
        Patrick Perez, Michel Gangnet, Andrew Blake: "Poisson Image Editing"
    """
    blend_method = blend_method.upper()

    if blend_method not in BLEND_METHODS:
        raise ValueError(f"blend_method should be one of {BLEND_METHODS}, but got {blend_method}")

    h, w = base_img.shape[:2]
    obj_h, obj_w = obj_img.shape[:2]

    if h < obj_top + obj_h:
        raise ValueError(
            f"obj_img height(={obj_h}) + obj_top(={obj_top}) should be less than or equal to base_img height(={h})"
            + f", got {obj_h + obj_top} > {h}"
        )

    if w < obj_left + obj_w:
        raise ValueError(
            f"obj_img width(={obj_w}) + obj_left(={obj_left}) should be less than or equal to base_img width(={w})"
            + f", got {obj_w + obj_left} > {w}"
        )

    if blend_method in CV2_BLEND_METHOD_MAP:
        # poisson blending
        obj_mask = np.where(obj_mask > 0, 255, 0).astype(np.uint8)
        # calc object center position
        obj_cent = (int(obj_left + obj_w / 2), int(obj_top + obj_h / 2))
        blend_img = cv2.seamlessClone(obj_img, base_img, obj_mask, p=obj_cent, flags=CV2_BLEND_METHOD_MAP[blend_method])
    else:
        # gaussian blending
        alpha = np.where(obj_mask > 0, 1.0, 0.0).astype(np.float32)
        alpha = skimage.filters.gaussian(alpha, sigma=sigma, preserve_range=True)
        alpha = alpha[:, :, None]

        idx_row = slice(obj_top, obj_top + obj_h)
        idx_col = slice(obj_left, obj_left + obj_w)

        blend_img = base_img.copy()
        img_dtype = base_img.dtype

        blend_img[idx_row, idx_col] = obj_img * alpha + blend_img[idx_row, idx_col] * (1 - alpha)
        blend_img = blend_img.astype(img_dtype)
    return blend_img


@preserve_shape
def posterize(img, bits):
    """Reduce the number of bits for each color channel.

    Args:
        img (numpy.ndarray): image to posterize.
        bits (int): number of high bits. Must be in range [0, 8]

    Returns:
        numpy.ndarray: Image with reduced color channels.

    """
    bits = np.uint8(bits)

    if img.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")
    if np.any((bits < 0) | (bits > 8)):
        raise ValueError("bits must be in range [0, 8]")

    if not bits.shape or len(bits) == 1:
        if bits == 0:
            return np.zeros_like(img)
        if bits == 8:
            return img.copy()

        lut = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - bits) - 1)
        lut &= mask

        return cv2.LUT(img, lut)

    if not is_rgb_image(img):
        raise TypeError("If bits is iterable image must be RGB")

    result_img = np.empty_like(img)
    for i, channel_bits in enumerate(bits):
        if channel_bits == 0:
            result_img[..., i] = np.zeros_like(img[..., i])
        elif channel_bits == 8:
            result_img[..., i] = img[..., i].copy()
        else:
            lut = np.arange(0, 256, dtype=np.uint8)
            mask = ~np.uint8(2 ** (8 - channel_bits) - 1)
            lut &= mask

            result_img[..., i] = cv2.LUT(img[..., i], lut)

    return result_img


def _equalize_pil(img, mask=None):
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


def _equalize_cv(img, mask=None):
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
    i += 1
    for i in range(i, len(histogram)):
        _sum += histogram[i]
        lut[i] = clip(round(_sum * scale), np.dtype("uint8"), 255)

    return cv2.LUT(img, lut)


@preserve_channel_dim
def equalize(img, mask=None, mode="cv", by_channels=True):
    """Equalize the image histogram.

    Args:
        img (numpy.ndarray): RGB or grayscale image.
        mask (numpy.ndarray): An optional mask.  If given, only the pixels selected by
            the mask are included in the analysis. Maybe 1 channel or 3 channel array.
        mode (str): {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
        by_channels (bool): If True, use equalization by channels separately,
            else convert image to YCbCr representation and use equalization by `Y` channel.

    Returns:
        numpy.ndarray: Equalized image.

    """
    if img.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")

    modes = ["cv", "pil"]

    if mode not in modes:
        raise ValueError("Unsupported equalization mode. Supports: {}. " "Got: {}".format(modes, mode))
    if mask is not None:
        if is_rgb_image(mask) and is_grayscale_image(img):
            raise ValueError("Wrong mask shape. Image shape: {}. " "Mask shape: {}".format(img.shape, mask.shape))
        if not by_channels and not is_grayscale_image(mask):
            raise ValueError(
                "When by_channels=False only 1-channel mask supports. " "Mask shape: {}".format(mask.shape)
            )

    if mode == "pil":
        function = _equalize_pil
    else:
        function = _equalize_cv

    if mask is not None:
        mask = mask.astype(np.uint8)

    if is_grayscale_image(img):
        return function(img, mask)

    if not by_channels:
        result_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        result_img[..., 0] = function(result_img[..., 0], mask)
        return cv2.cvtColor(result_img, cv2.COLOR_YCrCb2RGB)

    result_img = np.empty_like(img)
    for i in range(3):
        if mask is None:
            _mask = None
        elif is_grayscale_image(mask):
            _mask = mask
        else:
            _mask = mask[..., i]

        result_img[..., i] = function(img[..., i], _mask)

    return result_img


@preserve_shape
def move_tone_curve(img, low_y, high_y):
    """Rescales the relationship between bright and dark areas of the image by manipulating its tone curve.

    Args:
        img (numpy.ndarray): RGB or grayscale image.
        low_y (float): y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y (float): y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]
    """
    input_dtype = img.dtype

    if low_y < 0 or low_y > 1:
        raise ValueError("low_shift must be in range [0, 1]")
    if high_y < 0 or high_y > 1:
        raise ValueError("high_shift must be in range [0, 1]")

    if input_dtype != np.uint8:
        raise ValueError("Unsupported image type {}".format(input_dtype))

    t = np.linspace(0.0, 1.0, 256)

    # Defines responze of a four-point bezier curve
    def evaluate_bez(t):
        return 3 * (1 - t) ** 2 * t * low_y + 3 * (1 - t) * t**2 * high_y + t**3

    evaluate_bez = np.vectorize(evaluate_bez)
    remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)

    lut_fn = _maybe_process_in_chunks(cv2.LUT, lut=remapping)
    img = lut_fn(img)
    return img


@clipped
def _shift_rgb_non_uint8(img, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        return img + r_shift

    result_img = np.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[..., i] = img[..., i] + shift

    return result_img


def _shift_image_uint8(img, value):
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]

    lut = np.arange(0, max_value + 1).astype("float32")
    lut += value

    lut = np.clip(lut, 0, max_value).astype(img.dtype)
    return cv2.LUT(img, lut)


@preserve_shape
def _shift_rgb_uint8(img, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        h, w, c = img.shape
        img = img.reshape([h, w * c])

        return _shift_image_uint8(img, r_shift)

    result_img = np.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[..., i] = _shift_image_uint8(img[..., i], shift)

    return result_img


def shift_rgb(img, r_shift, g_shift, b_shift):
    if img.dtype == np.uint8:
        return _shift_rgb_uint8(img, r_shift, g_shift, b_shift)

    return _shift_rgb_non_uint8(img, r_shift, g_shift, b_shift)


@clipped
def linear_transformation_rgb(img, transformation_matrix):
    result_img = cv2.transform(img, transformation_matrix)

    return result_img


@preserve_channel_dim
def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img


@preserve_shape
def convolve(img, kernel):
    conv_fn = _maybe_process_in_chunks(cv2.filter2D, ddepth=-1, kernel=kernel)
    return conv_fn(img)


@preserve_shape
def image_compression(img, quality, image_type):
    if image_type in [".jpeg", ".jpg"]:
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
            "{} is used as input.".format(input_dtype),
            UserWarning,
        )
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for image augmentation".format(input_dtype))

    _, encoded_img = cv2.imencode(image_type, img, (int(quality_flag), quality))
    img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

    if needs_float:
        img = to_float(img, max_value=255)
    return img


@preserve_shape
def add_snow(img, snow_point, brightness_coeff):
    """Bleaches out pixels, imitation snow.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray): Image.
        snow_point: Number of show points.
        brightness_coeff: Brightness coefficient.

    Returns:
        numpy.ndarray: Image.

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    snow_point *= 127.5  # = 255 / 2
    snow_point += 85  # = 255 / 3

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomSnow augmentation".format(input_dtype))

    image_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    image_HLS = np.array(image_HLS, dtype=np.float32)

    image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] *= brightness_coeff

    image_HLS[:, :, 1] = clip(image_HLS[:, :, 1], np.uint8, 255)

    image_HLS = np.array(image_HLS, dtype=np.uint8)

    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)

    if needs_float:
        image_RGB = to_float(image_RGB, max_value=255)

    return image_RGB


@preserve_shape
def add_rain(
    img,
    slant,
    drop_length,
    drop_width,
    drop_color,
    blur_value,
    brightness_coefficient,
    rain_drops,
):
    """

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray): Image.
        slant (int):
        drop_length:
        drop_width:
        drop_color:
        blur_value (int): Rainy view are blurry.
        brightness_coefficient (float): Rainy days are usually shady.
        rain_drops:

    Returns:
        numpy.ndarray: Image.

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomRain augmentation".format(input_dtype))

    image = img.copy()

    for (rain_drop_x0, rain_drop_y0) in rain_drops:
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
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


@preserve_shape
def add_fog(img, fog_coef, alpha_coef, haze_list):
    """Add fog to the image.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray): Image.
        fog_coef (float): Fog coefficient.
        alpha_coef (float): Alpha coefficient.
        haze_list (list):

    Returns:
        numpy.ndarray: Image.

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomFog augmentation".format(input_dtype))

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
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        img = output.copy()

    image_rgb = cv2.blur(img, (hw // 10, hw // 10))

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


@preserve_shape
def add_sun_flare(img, flare_center_x, flare_center_y, src_radius, src_color, circles):
    """Add sun flare.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray):
        flare_center_x (float):
        flare_center_y (float):
        src_radius:
        src_color (int, int, int):
        circles (list):

    Returns:
        numpy.ndarray:

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomSunFlareaugmentation".format(input_dtype))

    overlay = img.copy()
    output = img.copy()

    for (alpha, (x, y), rad3, (r_color, g_color, b_color)) in circles:
        cv2.circle(overlay, (x, y), rad3, (r_color, g_color, b_color), -1)

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    point = (int(flare_center_x), int(flare_center_y))

    overlay = output.copy()
    num_times = src_radius // 10
    alpha = np.linspace(0.0, 1, num=num_times)
    rad = np.linspace(1, src_radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times - i - 1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        cv2.addWeighted(overlay, alp, output, 1 - alp, 0, output)

    image_rgb = output

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


@ensure_contiguous
@preserve_shape
def add_shadow(img, vertices_list):
    """Add shadows to the image.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (numpy.ndarray):
        vertices_list (list):

    Returns:
        numpy.ndarray:

    """
    non_rgb_warning(img)
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype("uint8"))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for RandomShadow augmentation".format(input_dtype))

    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(img)

    # adding all shadow polygons on empty mask, single 255 denotes only red channel
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255)

    # if red channel is hot, image's "Lightness" channel's brightness is lowered
    red_max_value_ind = mask[:, :, 0] == 255
    image_hls[:, :, 1][red_max_value_ind] = image_hls[:, :, 1][red_max_value_ind] * 0.5

    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


def invert(img: np.ndarray) -> np.ndarray:
    # Supports all the valid dtypes
    # clips the img to avoid unexpected behaviour.
    return MAX_VALUES_BY_DTYPE[img.dtype] - img


def channel_shuffle(img, channels_shuffled):
    img = img[..., channels_shuffled]
    return img


@preserve_shape
def gamma_transform(img, gamma):
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    else:
        img = np.power(img, gamma)

    return img


@clipped
def gauss_noise(image, gauss):
    image = image.astype("float32")
    return image + gauss


@clipped
def _brightness_contrast_adjust_non_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = img.dtype
    img = img.astype("float32")

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img


@preserve_shape
def _brightness_contrast_adjust_uint(img, alpha=1, beta=0, beta_by_max=False):
    dtype = np.dtype("uint8")

    max_value = MAX_VALUES_BY_DTYPE[dtype]

    lut = np.arange(0, max_value + 1).astype("float32")

    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += (alpha * beta) * np.mean(img)

    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img


def brightness_contrast_adjust(img, alpha=1, beta=0, beta_by_max=False):
    if img.dtype == np.uint8:
        return _brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)

    return _brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)


@clipped
def iso_noise(image, color_shift=0.05, intensity=0.5, random_state=None, **kwargs):
    """
    Apply poisson noise to image to simulate camera sensor noise.

    Args:
        image (numpy.ndarray): Input image, currently, only RGB, uint8 images are supported.
        color_shift (float):
        intensity (float): Multiplication factor for noise values. Values of ~0.5 are produce noticeable,
                   yet acceptable level of noise.
        random_state:
        **kwargs:

    Returns:
        numpy.ndarray: Noised image

    """
    if image.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")
    if not is_rgb_image(image):
        raise TypeError("Image must be RGB")

    one_over_255 = float(1.0 / 255.0)
    image = np.multiply(image, one_over_255, dtype=np.float32)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)

    luminance_noise = random_utils.poisson(stddev[1] * intensity * 255, size=hls.shape[:2], random_state=random_state)
    color_noise = random_utils.normal(0, color_shift * 360 * intensity, size=hls.shape[:2], random_state=random_state)

    hue = hls[..., 0]
    hue += color_noise
    hue[hue < 0] += 360
    hue[hue > 360] -= 360

    luminance = hls[..., 1]
    luminance += (luminance_noise / 255) * (1.0 - luminance)

    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) * 255
    return image.astype(np.uint8)


def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


@preserve_shape
def downscale(img, scale, down_interpolation=cv2.INTER_AREA, up_interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]

    need_cast = (
        up_interpolation != cv2.INTER_NEAREST or down_interpolation != cv2.INTER_NEAREST
    ) and img.dtype == np.uint8
    if need_cast:
        img = to_float(img)
    downscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=down_interpolation)
    upscaled = cv2.resize(downscaled, (w, h), interpolation=up_interpolation)
    if need_cast:
        upscaled = from_float(np.clip(upscaled, 0, 1), dtype=np.dtype("uint8"))
    return upscaled


def to_float(img, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(img.dtype)
            )
    return img.astype("float32") / max_value


def from_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(dtype)
            )
    return (img * max_value).astype(dtype)


def noop(input_obj, **params):  # skipcq: PYL-W0613
    return input_obj


def swap_tiles_on_image(image, tiles):
    """
    Swap tiles on image.

    Args:
        image (np.ndarray): Input image.
        tiles (np.ndarray): array of tuples(
            current_left_up_corner_row, current_left_up_corner_col,
            old_left_up_corner_row, old_left_up_corner_col,
            height_tile, width_tile)

    Returns:
        np.ndarray: Output image.

    """
    new_image = image.copy()

    for tile in tiles:
        new_image[tile[0] : tile[0] + tile[4], tile[1] : tile[1] + tile[5]] = image[
            tile[2] : tile[2] + tile[4], tile[3] : tile[3] + tile[5]
        ]

    return new_image


@clipped
def _multiply_uint8(img, multiplier):
    img = img.astype(np.float32)
    return np.multiply(img, multiplier)


@preserve_shape
def _multiply_uint8_optimized(img, multiplier):
    if is_grayscale_image(img) or len(multiplier) == 1:
        multiplier = multiplier[0]
        lut = np.arange(0, 256, dtype=np.float32)
        lut *= multiplier
        lut = clip(lut, np.uint8, MAX_VALUES_BY_DTYPE[img.dtype])
        func = _maybe_process_in_chunks(cv2.LUT, lut=lut)
        return func(img)

    channels = img.shape[-1]
    lut = [np.arange(0, 256, dtype=np.float32)] * channels
    lut = np.stack(lut, axis=-1)

    lut *= multiplier
    lut = clip(lut, np.uint8, MAX_VALUES_BY_DTYPE[img.dtype])

    images = []
    for i in range(channels):
        func = _maybe_process_in_chunks(cv2.LUT, lut=lut[:, i])
        images.append(func(img[:, :, i]))
    return np.stack(images, axis=-1)


@clipped
def _multiply_non_uint8(img, multiplier):
    return img * multiplier


def multiply(img, multiplier):
    """
    Args:
        img (numpy.ndarray): Image.
        multiplier (numpy.ndarray): Multiplier coefficient.

    Returns:
        numpy.ndarray: Image multiplied by `multiplier` coefficient.

    """
    if img.dtype == np.uint8:
        if len(multiplier.shape) == 1:
            return _multiply_uint8_optimized(img, multiplier)

        return _multiply_uint8(img, multiplier)

    return _multiply_non_uint8(img, multiplier)


def bbox_from_mask(mask):
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


def mask_from_bbox(img, bbox):
    """Create binary mask from bounding box

    Args:
        img (numpy.ndarray): input image
        bbox: A bounding box tuple `(x_min, y_min, x_max, y_max)`

    Returns:
        mask (numpy.ndarray): binary mask

    """

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    x_min, y_min, x_max, y_max = bbox
    mask[y_min:y_max, x_min:x_max] = 1
    return mask


def fancy_pca(img, alpha=0.1):
    """Perform 'Fancy PCA' augmentation from:
    http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    Args:
        img (numpy.ndarray): numpy array with (h, w, rgb) shape, as ints between 0-255
        alpha (float): how much to perturb/scale the eigen vecs and vals
                the paper used std=0.1

    Returns:
        numpy.ndarray: numpy image-like array as uint8 range(0, 255)

    """
    if not is_rgb_image(img) or img.dtype != np.uint8:
        raise TypeError("Image must be RGB image in uint8 format.")

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

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    # alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx] * 255

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    return orig_img


def _adjust_brightness_torchvision_uint8(img, factor):
    lut = np.arange(0, 256) * factor
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


@preserve_shape
def adjust_brightness_torchvision(img, factor):
    if factor == 0:
        return np.zeros_like(img)
    elif factor == 1:
        return img

    if img.dtype == np.uint8:
        return _adjust_brightness_torchvision_uint8(img, factor)

    return clip(img * factor, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def _adjust_contrast_torchvision_uint8(img, factor, mean):
    lut = np.arange(0, 256) * factor
    lut = lut + mean * (1 - factor)
    lut = clip(lut, img.dtype, 255)

    return cv2.LUT(img, lut)


@preserve_shape
def adjust_contrast_torchvision(img, factor):
    if factor == 1:
        return img

    if is_grayscale_image(img):
        mean = img.mean()
    else:
        mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        if img.dtype != np.float32:
            mean = int(mean + 0.5)
        return np.full_like(img, mean, dtype=img.dtype)

    if img.dtype == np.uint8:
        return _adjust_contrast_torchvision_uint8(img, factor, mean)

    return clip(
        img.astype(np.float32) * factor + mean * (1 - factor),
        img.dtype,
        MAX_VALUES_BY_DTYPE[img.dtype],
    )


@preserve_shape
def adjust_saturation_torchvision(img, factor, gamma=0):
    if factor == 1:
        return img

    if is_grayscale_image(img):
        gray = img
        return gray
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray

    result = cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)
    if img.dtype == np.uint8:
        return result

    # OpenCV does not clip values for float dtype
    return clip(result, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def _adjust_hue_torchvision_uint8(img, factor):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    img[..., 0] = cv2.LUT(img[..., 0], lut)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def adjust_hue_torchvision(img, factor):
    if is_grayscale_image(img):
        return img

    if factor == 0:
        return img

    if img.dtype == np.uint8:
        return _adjust_hue_torchvision_uint8(img, factor)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 0] = np.mod(img[..., 0] + factor * 360, 360)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


@preserve_shape
def superpixels(
    image: np.ndarray, n_segments: int, replace_samples: Sequence[bool], max_size: Optional[int], interpolation: int
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
            resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(new_width, new_height), interpolation=interpolation)
            image = resize_fn(image)

    from skimage.segmentation import slic

    segments = skimage.segmentation.slic(image, n_segments=n_segments, compactness=10)

    min_value = 0
    max_value = MAX_VALUES_BY_DTYPE[image.dtype]
    image = np.copy(image)
    if image.ndim == 2:
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
        resize_fn = _maybe_process_in_chunks(
            cv2.resize, dsize=(orig_shape[1], orig_shape[0]), interpolation=interpolation
        )
        image = resize_fn(image)

    return image


@clipped
def add_weighted(img1, alpha, img2, beta):
    return img1.astype(float) * alpha + img2.astype(float) * beta


@clipped
@preserve_shape
def unsharp_mask(image: np.ndarray, ksize: int, sigma: float = 0.0, alpha: float = 0.2, threshold: int = 10):
    blur_fn = _maybe_process_in_chunks(cv2.GaussianBlur, ksize=(ksize, ksize), sigmaX=sigma)

    input_dtype = image.dtype
    if input_dtype == np.uint8:
        image = to_float(image)
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError("Unexpected dtype {} for UnsharpMask augmentation".format(input_dtype))

    blur = blur_fn(image)
    residual = image - blur

    # Do not sharpen noise
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype("float32")

    sharp = image + alpha * residual
    # Avoid color noise artefacts.
    sharp = np.clip(sharp, 0, 1)

    soft_mask = blur_fn(mask)
    output = soft_mask * sharp + (1 - soft_mask) * image
    return from_float(output, dtype=input_dtype)


@preserve_shape
def pixel_dropout(image: np.ndarray, drop_mask: np.ndarray, drop_value: Union[float, Sequence[float]]) -> np.ndarray:
    if isinstance(drop_value, (int, float)) and drop_value == 0:
        drop_values = np.zeros_like(image)
    else:
        drop_values = np.full_like(image, drop_value)  # type: ignore
    return np.where(drop_mask, drop_values, image)


@clipped
@preserve_shape
def spatter(
    img: np.ndarray,
    non_mud: Optional[np.ndarray],
    mud: Optional[np.ndarray],
    rain: Optional[np.ndarray],
    mode: str,
) -> np.ndarray:
    non_rgb_warning(img)

    coef = MAX_VALUES_BY_DTYPE[img.dtype]
    img = img.astype(np.float32) * (1 / coef)

    if mode == "rain":
        assert rain is not None
        img = img + rain
    elif mode == "mud":
        assert non_mud is not None and mud is not None
        img = img * non_mud + mud
    else:
        raise ValueError("Unsupported spatter mode: " + str(mode))

    return img * 255
