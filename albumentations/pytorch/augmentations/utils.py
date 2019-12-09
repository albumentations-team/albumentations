import cv2
import torch

from functools import wraps


MAX_VALUES_BY_DTYPE = {torch.uint8: 255, torch.float32: 1.0, torch.float64: 1.0}
OPENCV_TO_TORCH_INTERPOLATION = {
    cv2.INTER_LINEAR: "bilinear",
    cv2.INTER_NEAREST: "nearest",
    cv2.INTER_AREA: "area",
    cv2.INTER_CUBIC: "cicubic",
}


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
        img = func(img, *args, **kwargs)
        img = img.view(*shape)
        return img

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


def on_4d_image(dtype=None):
    def callable(func):
        @wraps(func)
        def wrapped_function(img, *args, **kwargs):
            old_dtype = img.dtype
            if dtype is not None:
                img = img.to(dtype)
            img = img.view(1, *img.shape)
            result = func(img, *args, **kwargs)
            result = result.view(*result.shape[1:])
            result = result.to(old_dtype)
            return result

        return wrapped_function

    return callable


def get_interpolation_mode(mode):
    if not isinstance(mode, str):
        return OPENCV_TO_TORCH_INTERPOLATION[mode]

    return mode
