from __future__ import division

import math
from functools import wraps
from warnings import warn

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox

MAX_VALUES_BY_DTYPE = {
    np.dtype('uint8'): 255,
    np.dtype('uint16'): 65535,
    np.dtype('uint32'): 4294967295,
    np.dtype('float32'): 1.0,
}


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


def preserve_shape(func):
    """Preserve shape of the image."""
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


def preserve_channel_dim(func):
    """Preserve dummy channel dim."""
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
        return result

    return wrapped_function


def is_rgb_image(image):
    return len(image.shape) == 3 and image.shape[-1] == 3


def is_grayscale_image(image):
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)


def is_multispectral_image(image):
    return len(image.shape) == 3 and image.shape[-1] not in [1, 3]


def non_rgb_warning(image):
    if not is_rgb_image(image):
        message = 'This transformation expects 3-channel images'
        if is_grayscale_image(image):
            message += '\nYou can convert your grayscale image to RGB using cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))'
        if is_multispectral_image(image):  # Any image with a number of channels other than 1 and 3
            message += '\nThis transformation cannot be applied to multi-spectral images'

        raise ValueError(message)


def vflip(img):
    return np.ascontiguousarray(img[::-1, ...])


def hflip(img):
    return np.ascontiguousarray(img[:, ::-1, ...])


def hflip_cv2(img):
    return cv2.flip(img, 1)


@preserve_shape
def random_flip(img, code):
    return cv2.flip(img, code)


def transpose(img):
    return img.transpose(1, 0, 2) if len(img.shape) > 2 else img.transpose(1, 0)


def rot90(img, factor):
    img = np.rot90(img, factor)
    return np.ascontiguousarray(img)


def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def cutout(img, holes):
    # Make a copy of the input image since we don't want to modify it directly
    img = img.copy()

    for x1, y1, x2, y2 in holes:
        img[y1: y2, x1: x2] = 0
    return img


@preserve_channel_dim
def rotate(img, angle, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101):
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (width, height), flags=interpolation, borderMode=border_mode)
    return img


@preserve_channel_dim
def scale(img, scale, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    return img


@preserve_channel_dim
def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    img = cv2.resize(img, (width, height), interpolation=interpolation)
    return img


@preserve_channel_dim
def shift_scale_rotate(img, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height
    img = cv2.warpAffine(img, matrix, (width, height), flags=interpolation, borderMode=border_mode)
    return img


def bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, interpolation, rows, cols, **params):
    center = (0.5, 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    x = np.array([bbox[0], bbox[2], bbox[2], bbox[0]])
    y = np.array([bbox[1], bbox[1], bbox[3], bbox[3]])
    ones = np.ones(shape=(len(x)))
    points_ones = np.vstack([x, y, ones]).transpose()
    tr_points = matrix.dot(points_ones.T).T
    return [min(tr_points[:, 0]), min(tr_points[:, 1]), max(tr_points[:, 0]), max(tr_points[:, 1])]


def keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols, **params):
    height, width = rows, cols
    center = (width / 2, height / 2)
    x, y, a, s = keypoint
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height
    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    return [x, y, a + math.radians(angle), s * scale]


def crop(img, x_min, y_min, x_max, y_max):
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            'We should have x_min < x_max and y_min < y_max. But we got'
            ' (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})'.format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            'Values for crop should be non negative and equal or smaller than image sizes'
            '(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}'
            'height = {height}, width = {width})'.format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                height=height,
                width=width
            )
        )

    return img[y_min:y_max, x_min:x_max]


def get_center_crop_coords(height, width, crop_height, crop_width):
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def center_crop(img, crop_height, crop_width):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            'Requested crop size ({crop_height}, {crop_width}) is '
            'larger than the image size ({height}, {width})'.format(
                crop_height=crop_height,
                crop_width=crop_width,
                height=height,
                width=width,
            )
        )
    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    img = img[y1:y2, x1:x2]
    return img


def get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(img, crop_height, crop_width, h_start, w_start):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            'Requested crop size ({crop_height}, {crop_width}) is '
            'larger than the image size ({height}, {width})'.format(
                crop_height=crop_height,
                crop_width=crop_width,
                height=height,
                width=width,
            )
        )
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img


def clamping_crop(img, x_min, y_min, x_max, y_max):
    h, w = img.shape[:2]
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if y_max >= h:
        y_max = h - 1
    if x_max >= w:
        x_max = w - 1
    return img[int(y_min):int(y_max), int(x_min):int(x_max)]


def shift_hsv(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if dtype == np.uint8:
        img = img.astype(np.int32)
    hue, sat, val = cv2.split(img)
    hue = cv2.add(hue, hue_shift)
    hue = np.where(hue < 0, hue + 180, hue)
    hue = np.where(hue > 180, hue - 180, hue)
    hue = hue.astype(dtype)
    sat = clip(cv2.add(sat, sat_shift), dtype, 255 if dtype == np.uint8 else 1.0)
    val = clip(cv2.add(val, val_shift), dtype, 255 if dtype == np.uint8 else 1.0)
    img = cv2.merge((hue, sat, val)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


@clipped
def shift_rgb(img, r_shift, g_shift, b_shift):
    if img.dtype == np.uint8:
        img = img.astype('int32')
        r_shift, g_shift, b_shift = np.int32(r_shift), np.int32(g_shift), np.int32(b_shift)
    else:
        # Make a copy of the input image since we don't want to modify it directly
        img = img.copy()
    img[..., 0] += r_shift
    img[..., 1] += g_shift
    img[..., 2] += b_shift
    return img


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError('clahe supports only uint8 inputs')

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2:
        img = clahe.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img


@preserve_channel_dim
def pad(img, min_height, min_width, border_mode=cv2.BORDER_REFLECT_101, value=[0, 0, 0]):
    height, width = img.shape[:2]

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    img = pad_with_params(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value)

    assert img.shape[0] == max(min_height, height)
    assert img.shape[1] == max(min_width, width)

    return img


@preserve_channel_dim
def pad_with_params(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode=cv2.BORDER_REFLECT_101,
                    value=[0, 0, 0]):
    img = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value=value)
    return img


@preserve_shape
def blur(img, ksize):
    return cv2.blur(img, (ksize, ksize))


@preserve_shape
def gaussian_blur(img, ksize):
    # When sigma=0, it is computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)


def _func_max_size(img, max_size, interpolation, func):
    height, width = img.shape[:2]

    scale = max_size / float(func(width, height))

    if scale != 1.0:
        out_size = tuple(py3round(dim * scale) for dim in (width, height))
        img = cv2.resize(img, out_size, interpolation=interpolation)
    return img


@preserve_channel_dim
def longest_max_size(img, max_size, interpolation):
    return _func_max_size(img, max_size, interpolation, max)


@preserve_channel_dim
def smallest_max_size(img, max_size, interpolation):
    return _func_max_size(img, max_size, interpolation, min)


@preserve_shape
def median_blur(img, ksize):
    if img.dtype == np.float32 and ksize not in {3, 5}:
        raise ValueError(
            'Invalid ksize value {}. For a float32 image the only valid ksize values are 3 and 5'.format(ksize))
    return cv2.medianBlur(img, ksize)


@preserve_shape
def motion_blur(img, kernel):
    return cv2.filter2D(img, -1, kernel / np.sum(kernel))


@preserve_shape
def jpeg_compression(img, quality):
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        warn('Jpeg compression augmentation '
             'is most effective with uint8 inputs, '
             '{} is used as input.'.format(input_dtype),
             UserWarning)
        img = from_float(img, dtype=np.dtype('uint8'))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError('Unexpected dtype {} for Jpeg augmentation'.format(input_dtype))

    _, encoded_img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, quality))
    img = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

    if needs_float:
        img = to_float(img, max_value=255)
    return img


@preserve_shape
def add_snow(img, snow_point, brightness_coeff):
    """Bleaches out pixels, mitation snow.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img:
        snow_point:
        brightness_coeff:

    Returns:

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    snow_point *= 127.5  # = 255 / 2
    snow_point += 85  # = 255 / 3

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype('uint8'))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError('Unexpected dtype {} for RandomSnow augmentation'.format(input_dtype))

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
def add_rain(img, slant, drop_length, drop_width, drop_color, blur_value, brightness_coefficient, rain_drops):
    """

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (np.uint8):
        slant (int):
        drop_length:
        drop_width:
        drop_color:
        blur_value (int): rainy view are blurry
        brightness_coefficient (float): rainy days are usually shady
        rain_drops:

    Returns:

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype('uint8'))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError('Unexpected dtype {} for RandomSnow augmentation'.format(input_dtype))

    image = img.copy()

    for (rain_drop_x0, rain_drop_y0) in rain_drops:
        rain_drop_x1 = rain_drop_x0 + slant
        rain_drop_y1 = rain_drop_y0 + drop_length

        cv2.line(image, (rain_drop_x0, rain_drop_y0), (rain_drop_x1, rain_drop_y1), drop_color, drop_width)

    image = cv2.blur(image, (blur_value, blur_value))  # rainy view are blurry
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)
    image_hls[:, :, 1] *= brightness_coefficient

    image_rgb = cv2.cvtColor(image_hls.astype(np.uint8), cv2.COLOR_HLS2RGB)

    if needs_float:
        image_rgb = to_float(image_rgb, max_value=255)

    return image_rgb


@preserve_shape
def add_fog(img, fog_coef, alpha_coef, haze_list):
    """Add fog to the image.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (np.array):
        fog_coef (float):
        alpha_coef (float):
        haze_list (list):
    Returns:

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype('uint8'))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError('Unexpected dtype {} for RandomFog augmentation'.format(input_dtype))

    height, width = img.shape[:2]

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
        img (np.array):
        flare_center_x (float):
        flare_center_y (float):
        src_radius:
        src_color (int, int, int):
        circles (list):

    Returns:

    """
    non_rgb_warning(img)

    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype('uint8'))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError('Unexpected dtype {} for RandomSunFlareaugmentation'.format(input_dtype))

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


@preserve_shape
def add_shadow(img, vertices_list):
    """Add shadows to the image.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (np.array):
        vertices_list (list):

    Returns:

    """
    non_rgb_warning(img)
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == np.float32:
        img = from_float(img, dtype=np.dtype('uint8'))
        needs_float = True
    elif input_dtype not in (np.uint8, np.float32):
        raise ValueError('Unexpected dtype {} for RandomSnow augmentation'.format(input_dtype))

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


@preserve_shape
def optical_distortion(img, k=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101):
    """Barrel / pincushion distortion. Unconventional augment.

    Reference:
        |  https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
        |  https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
        |  https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
        |  http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/
    """
    height, width = img.shape[:2]

    fx = width
    fy = width

    cx = width * 0.5 + dx
    cy = height * 0.5 + dy

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)
    img = cv2.remap(img, map1, map2, interpolation=interpolation, borderMode=border_mode)
    return img


@preserve_shape
def grid_distortion(img, num_steps=10, xsteps=[], ysteps=[], interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101):
    """
    Reference:
        http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    """
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    img = cv2.remap(img, map_x, map_y, interpolation=interpolation, borderMode=border_mode)
    return img


@preserve_shape
def elastic_transform(image, alpha, sigma, alpha_affine, interpolation=cv2.INTER_LINEAR,
                      border_mode=cv2.BORDER_REFLECT_101, random_state=None, approximate=False):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(1234)

    height, width = image.shape[:2]

    # Random affine
    center_square = np.float32((height, width)) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, matrix, (width, height), flags=interpolation, borderMode=border_mode)

    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = (random_state.rand(height, width).astype(np.float32) * 2 - 1)
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha

        dy = (random_state.rand(height, width).astype(np.float32) * 2 - 1)
        cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
        dy *= alpha
    else:
        dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)
        dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    mapx = np.float32(x + dx)
    mapy = np.float32(y + dy)

    return cv2.remap(image, mapx, mapy, interpolation, borderMode=border_mode)


@preserve_shape
def elastic_transform_approx(image, alpha, sigma, alpha_affine, interpolation=cv2.INTER_LINEAR,
                             border_mode=cv2.BORDER_REFLECT_101, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications for speed).
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(1234)

    height, width = image.shape[:2]

    # Random affine
    center_square = np.float32((height, width)) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, matrix, (width, height), flags=interpolation, borderMode=border_mode)

    dx = (random_state.rand(height, width).astype(np.float32) * 2 - 1)
    cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
    dx *= alpha

    dy = (random_state.rand(height, width).astype(np.float32) * 2 - 1)
    cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
    dy *= alpha

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    mapx = np.float32(x + dx)
    mapy = np.float32(y + dy)

    return cv2.remap(image, mapx, mapy, interpolation, borderMode=border_mode)


def invert(img):
    return 255 - img


def channel_shuffle(img, channels_shuffled):
    img = img[..., channels_shuffled]
    return img


@preserve_shape
def gamma_transform(img, gamma):
    if img.dtype == np.uint8:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
    else:
        img = np.power(img, gamma)

    return img


@clipped
def gauss_noise(image, gauss):
    image = image.astype('float32')
    return image + gauss


@clipped
def brightness_contrast_adjust(img, alpha=1, beta=0):
    img = img.astype('float32')
    if alpha != 1:
        img *= alpha
    if beta != 0:
        img += beta * np.mean(img)
    return img


def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def to_float(img, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        except KeyError:
            raise RuntimeError(
                'Can\'t infer the maximum value for dtype {}. You need to specify the maximum value manually by '
                'passing the max_value argument'.format(img.dtype)
            )
    return img.astype('float32') / max_value


def from_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                'Can\'t infer the maximum value for dtype {}. You need to specify the maximum value manually by '
                'passing the max_value argument'.format(dtype)
            )
    return (img * max_value).astype(dtype)


def bbox_vflip(bbox, rows, cols):
    """Flip a bounding box vertically around the x-axis."""
    x_min, y_min, x_max, y_max = bbox
    return [x_min, 1 - y_max, x_max, 1 - y_min]


def bbox_hflip(bbox, rows, cols):
    """Flip a bounding box horizontally around the y-axis."""
    x_min, y_min, x_max, y_max = bbox
    return [1 - x_max, y_min, 1 - x_min, y_max]


def bbox_flip(bbox, d, rows, cols):
    """Flip a bounding box either vertically, horizontally or both depending on the value of `d`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        bbox = bbox_vflip(bbox, rows, cols)
    elif d == 1:
        bbox = bbox_hflip(bbox, rows, cols)
    elif d == -1:
        bbox = bbox_hflip(bbox, rows, cols)
        bbox = bbox_vflip(bbox, rows, cols)
    else:
        raise ValueError('Invalid d value {}. Valid values are -1, 0 and 1'.format(d))
    return bbox


def crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols):
    """Crop a bounding box using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.
    """
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox
    x1, y1, x2, y2 = crop_coords
    cropped_bbox = [x_min - x1, y_min - y1, x_max - x1, y_max - y1]
    return normalize_bbox(cropped_bbox, crop_height, crop_width)


def bbox_crop(bbox, x_min, y_min, x_max, y_max, rows, cols):
    crop_coords = [x_min, y_min, x_max, y_max]
    crop_height = y_max - y_min
    crop_width = x_max - x_min
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def bbox_center_crop(bbox, crop_height, crop_width, rows, cols):
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols):
    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def bbox_rot90(bbox, factor, rows, cols):
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)

    Args:
        bbox (tuple): A tuple (x_min, y_min, x_max, y_max).
        factor (int): Number of CCW rotations. Must be in range [0;3] See np.rot90.
        rows (int): Image rows.
        cols (int): Image cols.
    """
    if factor < 0 or factor > 3:
        raise ValueError('Parameter n must be in range [0;3]')
    x_min, y_min, x_max, y_max = bbox
    if factor == 1:
        bbox = [y_min, 1 - x_max, y_max, 1 - x_min]
    if factor == 2:
        bbox = [1 - x_max, 1 - y_max, 1 - x_min, 1 - y_min]
    if factor == 3:
        bbox = [1 - y_max, x_min, 1 - y_min, x_max]
    return bbox


def bbox_rotate(bbox, angle, rows, cols, interpolation):
    """Rotates a bounding box by angle degrees

    Args:
        bbox (tuple): A tuple (x_min, y_min, x_max, y_max).
        angle (int): Angle of rotation in degrees
        rows (int): Image rows.
        cols (int): Image cols.
        interpolation (int): interpolation method.

        return a tuple (x_min, y_min, x_max, y_max)
    """
    scale = cols / float(rows)
    x = np.array([bbox[0], bbox[2], bbox[2], bbox[0]])
    y = np.array([bbox[1], bbox[1], bbox[3], bbox[3]])
    x = x - 0.5
    y = y - 0.5
    angle = np.deg2rad(angle)
    x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
    y_t = (-np.sin(angle) * x * scale + np.cos(angle) * y)
    x_t = x_t + 0.5
    y_t = y_t + 0.5
    return [min(x_t), min(y_t), max(x_t), max(y_t)]


def bbox_transpose(bbox, axis, rows, cols):
    """Transposes a bounding box along given axis.

    Args:
        bbox (tuple): A tuple (x_min, y_min, x_max, y_max).
        axis (int): 0 - main axis, 1 - secondary axis.
        rows (int): Image rows.
        cols (int): Image cols.
    """
    x_min, y_min, x_max, y_max = bbox
    if axis != 0 and axis != 1:
        raise ValueError('Axis must be either 0 or 1.')
    if axis == 0:
        bbox = [y_min, x_min, y_max, x_max]
    if axis == 1:
        bbox = [1 - y_max, 1 - x_max, 1 - y_min, 1 - x_min]
    return bbox


def keypoint_vflip(kp, rows, cols):
    """Flip a keypoint vertically around the x-axis."""
    x, y, angle, scale = kp
    c = math.cos(angle)
    s = math.sin(angle)
    angle = math.atan2(-s, c)
    return [x, (rows - 1) - y, angle, scale]


def keypoint_hflip(kp, rows, cols):
    """Flip a keypoint horizontally around the y-axis."""
    x, y, angle, scale = kp
    c = math.cos(angle)
    s = math.sin(angle)
    angle = math.atan2(s, -c)
    return [(cols - 1) - x, y, angle, scale]


def keypoint_flip(bbox, d, rows, cols):
    """Flip a keypoint either vertically, horizontally or both depending on the value of `d`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        bbox = keypoint_vflip(bbox, rows, cols)
    elif d == 1:
        bbox = keypoint_hflip(bbox, rows, cols)
    elif d == -1:
        bbox = keypoint_hflip(bbox, rows, cols)
        bbox = keypoint_vflip(bbox, rows, cols)
    else:
        raise ValueError('Invalid d value {}. Valid values are -1, 0 and 1'.format(d))
    return bbox


def keypoint_rot90(keypoint, factor, rows, cols, **params):
    """Rotates a keypoint by 90 degrees CCW (see np.rot90)

    Args:
        keypoint (tuple): A tuple (x, y, angle, scale).
        factor (int): Number of CCW rotations. Must be in range [0;3] See np.rot90.
        rows (int): Image rows.
        cols (int): Image cols.
    """
    if factor < 0 or factor > 3:
        raise ValueError('Parameter n must be in range [0;3]')
    x, y, angle, scale = keypoint
    if factor == 1:
        keypoint = [y, (cols - 1) - x, angle - math.pi / 2, scale]
    if factor == 2:
        keypoint = [(cols - 1) - x, (rows - 1) - y, angle - math.pi, scale]
    if factor == 3:
        keypoint = [(rows - 1) - y, x, angle + math.pi / 2, scale]
    return keypoint


def keypoint_rotate(keypoint, angle, rows, cols, **params):
    matrix = cv2.getRotationMatrix2D(((cols - 1) * 0.5, (rows - 1) * 0.5), angle, 1.0)
    x, y, a, s = keypoint
    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    return [x, y, a + math.radians(angle), s]


def keypoint_scale(keypoint, scale_x, scale_y, **params):
    """Scales a keypoint by scale_x and scale_y."""
    x, y, a, s = keypoint
    return [x * scale_x, y * scale_y, a, s * max(scale_x, scale_y)]


def crop_keypoint_by_coords(keypoint, crop_coords, crop_height, crop_width, rows, cols):
    """Crop a keypoint using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.
    """
    x, y, a, s = keypoint
    x1, y1, x2, y2 = crop_coords
    cropped_keypoint = [x - x1, y - y1, a, s]
    return cropped_keypoint


def keypoint_random_crop(keypoint, crop_height, crop_width, h_start, w_start, rows, cols):
    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    return crop_keypoint_by_coords(keypoint, crop_coords, crop_height, crop_width, rows, cols)


def keypoint_center_crop(bbox, crop_height, crop_width, rows, cols):
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_keypoint_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def py3round(number):
    """Unified rounding in all python versions."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))

    return int(round(number))
