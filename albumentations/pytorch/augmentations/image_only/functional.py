import torch
import numpy as np
import kornia as K

from functools import wraps


MAX_VALUES_BY_DTYPE = {torch.uint8: 255, torch.float32: 1.0, torch.float64: 1.0}


def grayscale_to_rgb(image):
    assert image.size(0) <= 3, "Supports only rgb and grayscale images."

    if image.size(0) == 3:
        return image

    return torch.cat([image] * 3)


def rgb_image(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        img = grayscale_to_rgb(img)
        result = func(img, *args, **kwargs)
        return result

    return wrapped_function


def is_rgb_image(image):
    return image.size(0) == 3


def preserve_shape(func):
    """
    Preserve shape of the image

    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.view(*shape)
        return result

    return wrapped_function


def on_float_image(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        if img.dtype != torch.uint8:
            return func(img, *args, **kwargs)

        tmp = img.to(torch.float32) / 255.0
        result = func(tmp, *args, **kwargs)
        result = torch.clamp(torch.round(result * 255.0), 0, 255).to(img.dtype)
        return result

    return wrapped_function


def clip(img, dtype, maxval):
    return torch.clamp(img, 0, maxval).type(dtype)


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


def round_opencv(img):
    int_part = img.to(torch.int32).float()
    fract_part = img - int_part

    cond = (fract_part != 0.5) & (fract_part != -0.5)
    cond |= (int_part % 2) != 0

    result = torch.empty_like(img)
    tmp = img[cond]
    result[cond] = tmp + torch.where(tmp >= 0, torch.full_like(tmp, 0.5), torch.full_like(tmp, -0.5))
    result[~cond] = int_part[~cond]

    return result.to(torch.int32)


def from_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(dtype)
            )
    return (img * max_value).to(dtype)


def to_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(img.dtype)
            )
    return img.type(dtype) / max_value


def cutout(img, holes, fill_value=0):
    # Make a copy of the input image since we don't want to modify it directly
    img = img.clone()
    for x1, y1, x2, y2 in holes:
        img[:, y1:y2, x1:x2] = fill_value
    return img


def _rbg_to_hls_float(img):
    img = K.rgb_to_hls(img)
    img[0] *= 360.0 / (2.0 * np.pi)
    return img


@clipped
def _rbg_to_hls_uint8(img):
    img = K.rgb_to_hls(img.float() * (1.0 / 255.0))
    img[0] *= 180.0 / (2.0 * np.pi)
    img[1:] *= 255.0
    return round_opencv(img)


def rgb_to_hls(img):
    # TODO add tests after fix nans in kornia.rgb_to_hls
    if img.dtype in [torch.float32, torch.float64]:
        return _rbg_to_hls_float(img)
    elif img.dtype == torch.uint8:
        return _rbg_to_hls_uint8(img)

    raise ValueError("rbg_to_hls support only uint8, float32 and float64 dtypes. Got: {}".format(img.dtype))


@clipped
def _hls_to_rgb_uint8(img):
    img = img.float()
    img[0] *= 2.0 * np.pi / 180.0
    img[1:] /= 255.0

    img = K.hls_to_rgb(img)
    img *= 255.0
    return round_opencv(img)


def _hls_to_rgb_float(img):
    img = img.clone()
    img[0] *= 2.0 * np.pi / 360.0
    return K.hls_to_rgb(img)


def hls_to_rgb(img):
    if img.dtype in [torch.float32, torch.float64]:
        return _hls_to_rgb_float(img)
    elif img.dtype == torch.uint8:
        return _hls_to_rgb_uint8(img)

    raise ValueError("hls_to_rgb support only uint8, float32 and float64 dtypes. Got: {}".format(img.dtype))


def _rbg_to_hsv_float(img):
    img = K.rgb_to_hsv(img)
    img[0] *= 360.0 / (2.0 * np.pi)
    return img


@clipped
def _rbg_to_hsv_uint8(img):
    img = K.rgb_to_hsv(img.float() * (1.0 / 255.0))
    img[0] *= 180.0 / (2.0 * np.pi)
    img[1:] *= 255.0

    img = round_opencv(img)
    img[0] %= 180.0
    return img


def rgb_to_hsv(img):
    if img.dtype in [torch.float32, torch.float64]:
        return _rbg_to_hsv_float(img)
    elif img.dtype == torch.uint8:
        return _rbg_to_hsv_uint8(img)

    raise ValueError("rbg_to_hls support only uint8, float32 and float64 dtypes. Got: {}".format(img.dtype))


@clipped
def _hsv_to_rgb_uint8(img):
    img = img.float()
    img[0] *= 2.0 * np.pi / 180.0
    img[1:] /= 255.0

    img = K.hsv_to_rgb(img)
    img *= 255.0
    return round_opencv(img)


def _hsv_to_rgb_float(img):
    img = img.clone()
    img[0] *= 2.0 * np.pi / 360.0
    return K.hsv_to_rgb(img)


def hsv_to_rgb(img):
    if img.dtype in [torch.float32, torch.float64]:
        return _hsv_to_rgb_float(img)
    elif img.dtype == torch.uint8:
        return _hsv_to_rgb_uint8(img)

    raise ValueError("hls_to_rgb support only uint8, float32 and float64 dtypes. Got: {}".format(img.dtype))


def add_snow(img, snow_point, brightness_coeff):
    """Bleaches out pixels, imitation snow.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (torch.Tensor): Image.
        snow_point: Number of show points.
        brightness_coeff: Brightness coefficient.

    Returns:
        numpy.ndarray: Image.

    """
    input_dtype = img.dtype
    needs_float = False

    snow_point *= 127.5  # = 255 / 2
    snow_point += 85  # = 255 / 3

    if input_dtype == torch.float32:
        img = from_float(img, torch.uint8)
        needs_float = True
    elif input_dtype not in (torch.uint8, torch.float32):
        raise ValueError("Unexpected dtype {} for RandomSnow augmentation".format(input_dtype))

    image_HLS = rgb_to_hls(img)
    image_HLS = image_HLS.float()

    image_HLS[1][image_HLS[1] < snow_point] *= brightness_coeff
    image_HLS[1] = clip(image_HLS[1], torch.uint8, 255)

    image_HLS = image_HLS.to(torch.uint8)
    image_RGB = hls_to_rgb(image_HLS)

    image_HLS.numpy().transpose([1, 2, 0]).tofile("torch_hls_")
    image_RGB.numpy().transpose([1, 2, 0]).tofile("torch_hls_A")
    if needs_float:
        image_RGB = to_float(image_RGB, torch.float32)

    return image_RGB


def normalize(img, mean, std):
    if mean.shape:
        mean = mean[..., :, None, None]
    if std.shape:
        std = std[..., :, None, None]

    denominator = torch.reciprocal(std)

    img = img.float()
    img -= mean
    img *= denominator
    return img


@preserve_shape
@on_float_image
def blur(image, ksize):
    image = image.view(1, *image.shape)
    image = K.box_blur(image, ksize)
    return image


@rgb_image
def shift_hsv(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype

    img = rgb_to_hsv(img)
    hue, sat, val = img

    hue = hue + hue_shift
    hue = torch.where(hue < 0, hue + 180, hue)
    hue = torch.where(hue > 180, hue - 180, hue)
    hue = hue.to(dtype)

    sat = clip(sat + sat_shift, dtype, 255 if dtype == torch.uint8 else 1.0)
    val = clip(val + val_shift, dtype, 255 if dtype == torch.uint8 else 1.0)

    img = torch.stack((hue, sat, val)).to(dtype)
    img = hsv_to_rgb(img)

    return img


def solarize(img, threshold=128):
    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]

    result_img = img.clone()
    cond = img >= threshold
    result_img[cond] = max_val - result_img[cond]
    return result_img


@clipped
def shift_rgb(img, r_shift, g_shift, b_shift):
    img = img.float()

    if r_shift == g_shift == b_shift:
        return img + r_shift

    result_img = torch.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[i] = img[i] + shift

    return result_img


@clipped
def brightness_contrast_adjust(img, alpha=1, beta=0, beta_by_max=False):
    dtype = img.dtype
    img = img.float()

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * torch.mean(img)
    return img


@clipped
@preserve_shape
def motion_blur(img, kernel):
    kernel = torch.from_numpy(kernel.reshape([1, *kernel.shape])).float()

    dtype = img.dtype
    img = img.view(1, *img.shape)
    img = K.filter2D(img.float(), kernel.float())

    if dtype == torch.uint8:
        img = round_opencv(img)

    return img


@clipped
@preserve_shape
def median_blur(img, ksize):
    dtype = img.dtype
    img = img.view(1, *img.shape)
    img = K.median_blur(img.float(), ksize)

    if dtype == torch.uint8:
        img = round_opencv(img)

    return img


@preserve_shape
def gaussian_blur(img, ksize):
    ksize = np.array(ksize).astype(float)
    sigma = 0.3 * ((ksize - 1.0) * 0.5 - 1.0) + 0.8

    dtype = img.dtype
    img = img.view(1, *img.shape)
    img = K.gaussian_blur2d(img.float(), tuple(int(i) for i in ksize), sigma=tuple(sigma))

    if dtype == torch.uint8:
        img = round_opencv(img)

    return img.to(dtype)


@clipped
@rgb_image
def iso_noise(image, color_shift=0.05, intensity=0.5, random_state=None, **_):
    # TODO add tests after fix nans in kornia.rgb_to_hls
    assert image.dtype == torch.uint8, "Image must have uint8 channel type"

    if random_state is None:
        random_state = np.random.RandomState(42)

    image = image.float() / 255.0
    hls = rgb_to_hls(image)
    std = torch.std(hls[1])

    # TODO use pytorch random geterator
    luminance_noise = torch.from_numpy(random_state.poisson(std * intensity * 255.0, size=hls.shape[1:]))
    color_noise = torch.from_numpy(random_state.normal(0, color_shift * 360.0 * intensity, size=hls.shape[1:]))

    luminance_noise = luminance_noise.to(image.device)
    color_noise = color_noise.to(image.device)

    hue = hls[0]
    hue += color_noise
    hue[hue < 0] += 360.0
    hue[hue > 360] -= 360.0

    luminance = hls[1]
    luminance += (luminance_noise / 255.0) * (1.0 - luminance)

    image = hls_to_rgb(hls) * 255.0
    return image


@preserve_shape
def channel_dropout(img, channels_to_drop, fill_value=0):
    if img.size(0) == 1:
        raise NotImplementedError("Only one channel. ChannelDropout is not defined.")

    img = img.clone()
    img[channels_to_drop] = fill_value
    return img


def invert(img):
    max_val = MAX_VALUES_BY_DTYPE[img.dtype]
    return max_val - img


@clipped
def gamma_transform(img, gamma, eps):
    dtype = img.dtype
    img = img.float()

    if dtype == torch.uint8:
        invGamma = 1.0 / (gamma + eps)
        img = img / 255.0
        img = torch.pow(img, invGamma) * 255.0
    else:
        img = torch.pow(img, gamma)

    return img.to(dtype)


def channel_shuffle(img, channels_shuffled):
    img = img[channels_shuffled]
    return img


def to_gray(img):
    dtype = img.dtype

    if len(img.shape) < 3 and img.shape[0] != 3:
        raise ValueError("Image size must have a shape of (3, H, W). Got {}".format(img.shape))

    r, g, b = torch.chunk(img, chunks=3, dim=-3)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return torch.cat([gray] * 3, 0).to(dtype)
