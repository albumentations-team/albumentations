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
    from_float,
    is_grayscale_image,
    is_rgb_image,
    non_rgb_warning,
    preserve_shape,
    to_float,
)

__all__ = [
    "add_fog",
    "add_rain",
    "add_shadow",
    "add_snow",
    "add_sun_flare",
    "add_weighted",
    "convolve",
    "downscale",
    "gauss_noise",
    "image_compression",
    "iso_noise",
    "move_tone_curve",
    "multiply",
    "noop",
    "superpixels",
    "swap_tiles_on_image",
    "unsharp_mask",
]


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


@clipped
def gauss_noise(image, gauss):
    image = image.astype("float32")
    return image + gauss


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
