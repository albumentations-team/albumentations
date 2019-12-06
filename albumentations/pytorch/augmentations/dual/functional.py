import torch
import numpy as np
import kornia as K
import torch.nn.functional as FTorch

from ..utils import on_4d_image, get_interpolation_mode


def copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode="constant", value=0):
    """
    Args:
        border_mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``

    """
    h, w = img.shape[-2:]
    pads = np.array([h_pad_top, h_pad_bottom, w_pad_left, w_pad_right])

    if border_mode not in ["constant", "replicate"] and (pads[:2] > h).any() or (pads[2:] > w).any():
        while (pads != 0).any():
            top, bot, left, right = [min(a, b) for a, b in zip(pads, [h - 1, h - 1, w - 1, w - 1])]
            img = copyMakeBorder(img, top, bot, left, right, border_mode, value)
            pads -= np.array([top, bot, left, right])
            h, w = img.shape[-2:]
        return img

    if border_mode != "constant":
        value = 0

    dtype = img.dtype
    img = img.view(1, *img.shape)
    img = FTorch.pad(img.float(), [w_pad_left, w_pad_right, h_pad_top, h_pad_bottom], mode=border_mode, value=value)
    return img.to(dtype).view(*img.shape[1:])


def pad_with_params(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode="reflect", value=None):
    return copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value)


def crop(img, x_min, y_min, x_max, y_max):
    height, width = img.shape[-2:]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            "(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            "height = {height}, width = {width})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, height=height, width=width
            )
        )

    return img[..., y_min:y_max, x_min:x_max]


def vflip(img):
    return torch.flip(img, [-2]).contiguous()


def hflip(img):
    return torch.flip(img, [-1]).contiguous()


def random_flip(img, code):
    if code == 0:
        code = [-2]
    elif code == 1:
        code = [-1]
    else:
        code = [-2, -1]

    return torch.flip(img, code).contiguous()


def transpose(img):
    return img.permute(0, 2, 1)


@on_4d_image(torch.float32)
def resize(img, height, width, interpolation="nearest"):
    return FTorch.interpolate(img.float(), [height, width], mode=get_interpolation_mode(interpolation))


def _func_max_size(img, max_size, interpolation, func):
    height, width = img.shape[-2:]

    scale = max_size / float(func(width, height))

    if scale != 1.0:
        new_height, new_width = tuple(round(dim * scale) for dim in (height, width))
        img = resize(img, height=new_height, width=new_width, interpolation=interpolation)
    return img


def longest_max_size(img, max_size, interpolation):
    return _func_max_size(img, max_size, interpolation, max)


def smallest_max_size(img, max_size, interpolation):
    return _func_max_size(img, max_size, interpolation, min)


def rot90(img, factor):
    return torch.rot90(img, factor, [-2, -1]).contiguous()


@on_4d_image(torch.float32)
def rotate(img, angle):
    height, width = img.shape[-2:]
    return K.rotate(img, torch.tensor(angle), torch.tensor([width / 2, height / 2]))


def scale(img, scale, interpolation="linear"):
    height, width = img.shape[-2:]
    new_height, new_width = int(height * scale), int(width * scale)
    return resize(img, new_height, new_width, interpolation)
