import torch

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


def clip(img, dtype, maxval=None):
    return torch.clamp(img, 0, maxval if maxval is not None else MAX_VALUES_BY_DTYPE[dtype]).type(dtype)


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


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
