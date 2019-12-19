import torch
import numpy as np
import kornia as K

import torch.nn.functional as TorchFunctional

from albumentations.pytorch.augmentations.utils import (
    MAX_VALUES_BY_DTYPE,
    preserve_shape,
    clipped,
    on_float_image,
    clip,
    rgb_image,
)


GAUS_KERNELS = {}  # Cache


def from_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(dtype)
            )
    return (img * max_value).to(dtype, non_blocking=True)


def to_float(img, dtype=torch.float32, max_value=None):
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


def __rgb_to_hls(image):
    if len(image.shape) < 3 or image.shape[0] != 3:
        raise ValueError("Input size must have a shape of (3, H, W). Got {}".format(image.shape))
    maxc = image.max(-3)[0]
    minc = image.min(-3)[0]

    imax = image.max(-3)[1]

    sum_max_min = maxc + minc

    luminance = sum_max_min / 2.0  # luminance

    deltac = maxc - minc

    s = torch.where(luminance < 0.5, deltac / sum_max_min, deltac / (2.0 - sum_max_min))  # saturation

    r, g, b = image / deltac

    hi = torch.empty_like(deltac)
    hi = torch.where(imax == 0, (g - b) % 6, hi)
    hi = torch.where(imax == 1, (b - r) + 2, hi)
    hi = torch.where(imax == 2, (r - g) + 4, hi)

    hi *= 60.0

    return torch.stack([hi, luminance, s], dim=-3)


def _rbg_to_hls_float(img):
    return __rgb_to_hls(img)


@clipped
def _rbg_to_hls_uint8(img):
    img = __rgb_to_hls(img.float() * (1.0 / 255.0))
    img[0] *= 180.0 / 360.0
    img[1:] *= 255.0
    return torch.round(img)


def rgb_to_hls(img):
    if img.dtype in [torch.float32, torch.float64]:
        return _rbg_to_hls_float(img)
    elif img.dtype == torch.uint8:
        return _rbg_to_hls_uint8(img)

    raise ValueError("rbg_to_hls support only uint8, float32 and float64 dtypes. Got: {}".format(img.dtype))


def __hls_to_rgb(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (3, H, W). Got {}".format(image.shape))

    h, l, s = image

    h_div = h / 30
    kr = h_div % 12

    h_div += 4
    kb = h_div % 12

    h_div += 4
    kg = h_div % 12

    a = s * torch.min(l, 1.0 - l)

    torch.min(kr - 3.0, 9.0 - kr, out=kr)
    torch.min(kg - 3.0, 9.0 - kg, out=kg)
    torch.min(kb - 3.0, 9.0 - kb, out=kb)

    torch.clamp(kr, -1.0, 1.0, out=kr)
    torch.clamp(kg, -1.0, 1.0, out=kg)
    torch.clamp(kb, -1.0, 1.0, out=kb)

    kr *= a
    kg *= a
    kb *= a

    torch.sub(l, kr, out=kr)
    torch.sub(l, kg, out=kg)
    torch.sub(l, kb, out=kb)

    return torch.stack([kr, kg, kb], dim=-3)


@clipped
def _hls_to_rgb_uint8(img):
    img = img.float()
    img[0] *= 360 / 180.0
    img[1:] /= 255.0

    img = __hls_to_rgb(img)
    img *= 255.0
    return torch.round(img)


def _hls_to_rgb_float(img):
    return __hls_to_rgb(img)


def hls_to_rgb(img):
    if img.dtype in [torch.float32, torch.float64]:
        return _hls_to_rgb_float(img)
    elif img.dtype == torch.uint8:
        return _hls_to_rgb_uint8(img)

    raise ValueError("hls_to_rgb support only uint8, float32 and float64 dtypes. Got: {}".format(img.dtype))


def __rgb_to_hsv(image):
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (3, H, W). Got {}".format(image.shape))

    r, g, b = image

    maxc = image.max(-3)[0]
    minc = image.min(-3)[0]

    v = maxc  # brightness

    deltac: torch.Tensor = maxc - minc

    s = deltac / v
    s = torch.where(torch.isnan(s), torch.zeros_like(s), s)

    # avoid division by zero
    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)

    maxg = g == maxc
    maxr = r == maxc

    r, g, b = (torch.stack([maxc] * 3) - image) / deltac

    h = 4.0 + g - r
    h = torch.where(maxg, 2.0 + r - b, h)
    h = torch.where(maxr, b - g, h)
    h = torch.where(minc == maxc, torch.zeros_like(h), h)

    h *= 60.0
    h %= 360.0

    return torch.stack([h, s, v], dim=-3)


def _rbg_to_hsv_float(img):
    return __rgb_to_hsv(img)


@clipped
def _rbg_to_hsv_uint8(img):
    img = __rgb_to_hsv(img.float() / 255.0)

    img[0] *= 180.0 / 360
    img[1:] *= 255.0

    img = torch.round(img)
    img[0] %= 180.0
    return img


def rgb_to_hsv(img):
    if img.dtype in [torch.float32, torch.float64]:
        return _rbg_to_hsv_float(img)
    elif img.dtype == torch.uint8:
        return _rbg_to_hsv_uint8(img)

    raise ValueError("rbg_to_hls support only uint8, float32 and float64 dtypes. Got: {}".format(img.dtype))


def __hsv_to_rgb(image):
    if len(image.shape) < 3 or image.shape[0] != 3:
        raise ValueError("Input size must have a shape of (3, H, W). Got {}".format(image.shape))

    h, s, v = image.clone()
    h = h / 60.0

    hi = torch.floor(h) % 6
    h = (h % 6) - hi

    sv = s * v
    svh = sv * h

    p: torch.Tensor = v - sv
    q: torch.Tensor = v - svh
    t: torch.Tensor = v - sv + svh

    out: torch.Tensor = torch.stack([hi, hi, hi])

    out = torch.where(out == 0, torch.stack((v, t, p), dim=-3), out)
    out = torch.where(out == 1, torch.stack((q, v, p), dim=-3), out)
    out = torch.where(out == 2, torch.stack((p, v, t), dim=-3), out)
    out = torch.where(out == 3, torch.stack((p, q, v), dim=-3), out)
    out = torch.where(out == 4, torch.stack((t, p, v), dim=-3), out)
    out = torch.where(out == 5, torch.stack((v, p, q), dim=-3), out)

    return out


@clipped
def _hsv_to_rgb_uint8(img):
    img = img.float()
    img[0] *= 2.0
    img[1:] /= 255.0

    img = __hsv_to_rgb(img)
    img *= 255.0
    return torch.round(img)


def _hsv_to_rgb_float(img):
    return __hsv_to_rgb(img)


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

    if input_dtype == torch.uint8:
        snow_point *= 127.5  # = 255 / 2
        snow_point += 85  # = 255 / 3
    else:
        snow_point *= 0.5
        snow_point += 1.0 / 3.0

    image_HLS = rgb_to_hls(img)
    image_HLS = image_HLS.float()

    image_HLS[1][image_HLS[1] < snow_point] *= brightness_coeff
    image_HLS[1] = clip(image_HLS[1], input_dtype, MAX_VALUES_BY_DTYPE[input_dtype])

    image_HLS = image_HLS.to(input_dtype, non_blocking=True)
    image_RGB = hls_to_rgb(image_HLS)

    return image_RGB


def normalize(img, mean, std):
    if mean.shape:
        mean = mean[..., :, None, None]
    if std.shape:
        std = std[..., :, None, None]

    denominator = torch.reciprocal(std)

    img = img.float()
    img -= mean.to(img.device, non_blocking=True)
    img *= denominator.to(img.device, non_blocking=True)
    return img


@preserve_shape
@on_float_image
def blur(image, ksize):
    image = image.view(1, *image.shape)

    scale = 1.0 / (ksize[0] * ksize[1])
    kernel = torch.full(ksize, scale, dtype=torch.float32, device=image.device)
    kernel = kernel.view(1, *ksize)

    image = K.filter2D(image, kernel)
    return image


@rgb_image
def shift_hsv(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype

    img = rgb_to_hsv(img)
    hue, sat, val = img

    hue = hue + hue_shift
    hue = torch.where(hue < 0, hue + 180, hue)
    hue = torch.where(hue > 180, hue - 180, hue)
    hue = hue.to(dtype, non_blocking=True)

    sat = clip(sat + sat_shift, dtype, 255 if dtype == torch.uint8 else 1.0)
    val = clip(val + val_shift, dtype, 255 if dtype == torch.uint8 else 1.0)

    img = torch.stack((hue, sat, val)).to(dtype, non_blocking=True)
    img = hsv_to_rgb(img)

    return img


def solarize(img, threshold=128):
    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]
    return torch.where(img >= threshold, max_val - img, img)


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
    img = K.filter2D(img.float(), kernel)

    if dtype == torch.uint8:
        img = torch.round(img)

    return img


@clipped
@preserve_shape
def median_blur(img, ksize):
    dtype = img.dtype
    img = img.view(1, *img.shape)

    wrange = ksize[0] * ksize[1]
    kernel = torch.diag(torch.zeros(wrange, dtype=dtype, device=img.device))
    kernel = kernel.view(wrange, 1, ksize[0], ksize[1])

    padding = tuple([(k - 1) // 2 for k in ksize])

    b, c, h, w = img.shape
    # map the local window to single vector
    features = torch.nn.functional.conv2d(img.view(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    img = torch.median(features, dim=2)[0]

    if dtype == torch.uint8:
        img = torch.round(img)

    return img


def _gaussian_kernel1d(window_size, sigma):
    x = np.arange(window_size) - window_size // 2
    if window_size % 2 == 0:
        x += 0.5
    gauss = np.exp(-(x ** 2) / (2.0 * (sigma ** 2)))
    return gauss / gauss.sum()


def _gaussian_kernel2d(kernel_size, sigma):
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x = np.expand_dims(_gaussian_kernel1d(ksize_x, sigma_x), -1)
    kernel_y = np.expand_dims(_gaussian_kernel1d(ksize_y, sigma_y), -1)
    kernel_2d = np.matmul(kernel_x, kernel_y.T)
    return np.expand_dims(kernel_2d, 0)


@preserve_shape
def gaussian_blur(img, ksize):
    global GAUS_KERNELS

    dtype = img.dtype
    img = img.view(1, *img.shape).float()

    ksize = tuple(ksize)
    if ksize in GAUS_KERNELS:
        kernel = GAUS_KERNELS[ksize]
    else:
        _ks = ksize
        ksize = np.array(ksize).astype(float)
        sigma = 0.3 * ((ksize - 1.0) * 0.5 - 1.0) + 0.8
        kernel = (
            torch.from_numpy(_gaussian_kernel2d(ksize, sigma))
            .to(img.device, non_blocking=True)
            .to(dtype, non_blocking=True)
        )
        GAUS_KERNELS[_ks] = kernel

    img = K.filter2D(img, kernel)

    if dtype == torch.uint8:
        img = torch.clamp(torch.round(img), 0, 255)

    return img.to(dtype, non_blocking=True)


@rgb_image
def iso_noise(image, color_shift=0.05, intensity=0.5, **_):
    # TODO add tests
    dtype = image.dtype
    if dtype == torch.uint8:
        image = image.float() / 255.0

    hls = rgb_to_hls(image)
    std = torch.std(hls[1]).cpu()

    # TODO use pytorch random generator
    luminance_noise = torch.full(hls.shape[1:], std * intensity * 255.0, dtype=dtype, device=image.device)
    luminance_noise = torch.poisson(luminance_noise)
    color_noise = torch.normal(
        0, color_shift * 360.0 * intensity, size=hls.shape[1:], dtype=dtype, device=image.device
    )

    hue = hls[0]
    hue += color_noise
    hue %= 360

    luminance = hls[1]
    luminance += (luminance_noise / 255.0) * (1.0 - luminance)

    image = hls_to_rgb(hls)
    if dtype == torch.uint8:
        image = clip(image, dtype, MAX_VALUES_BY_DTYPE[dtype])

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

    return img.to(dtype, non_blocking=True)


def channel_shuffle(img, channels_shuffled):
    img = img[channels_shuffled]
    return img


def to_gray(img):
    dtype = img.dtype

    if len(img.shape) < 3 and img.shape[0] != 3:
        raise ValueError("Image size must have a shape of (3, H, W). Got {}".format(img.shape))

    r, g, b = torch.chunk(img, chunks=3, dim=-3)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return torch.cat([gray] * 3, 0).to(dtype, non_blocking=True)


@clipped
@preserve_shape
def downscale(img, scale, interpolation="nearest"):
    dtype = img.dtype
    h, w = img.shape[1:]
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    img = img.view(1, *img.shape).float()

    downscaled = TorchFunctional.interpolate(img, (new_h, new_w), mode=interpolation)
    if dtype == torch.uint8 and interpolation == "nearest":
        downscaled = torch.clamp(downscaled, 0, 255).int().float()

    upscaled = TorchFunctional.interpolate(downscaled, (h, w), mode=interpolation)

    return upscaled
