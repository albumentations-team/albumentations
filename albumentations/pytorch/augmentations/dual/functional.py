import torch
import kornia as K
import torch.nn.functional as FTorch


def copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode="constant", value=0):
    """
    Args:
        border_mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``

    """
    if border_mode != "constant":
        value = 0

    dtype = img.dtype
    img = img.view(1, *img.shape)
    img = FTorch.pad(img.float(), [w_pad_left, w_pad_right, h_pad_top, h_pad_bottom], mode=border_mode, value=value)
    return img.to(dtype).view(*img.shape[1:])


def pad_with_params(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode="reflect", value=None):
    return copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value)
