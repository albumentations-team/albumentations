import cv2
import torch
import numpy as np
import kornia as K
import torch.nn.functional as FTorch

import albumentations.augmentations.functional as AF

from ..utils import on_4d_image, get_interpolation_mode, get_border_mode


def copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode="constant", value=0):
    """
    Args:
        border_mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``

    """
    h, w = img.shape[-2:]
    pads = np.array([h_pad_top, h_pad_bottom, w_pad_left, w_pad_right])

    if border_mode not in ["constant", "replicate"] and (pads[:2] >= h).any() or (pads[2:] >= w).any():
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
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1).reshape(1, 2, 3).astype(np.float32)
    matrix = torch.from_numpy(matrix).to(img.device, non_blocking=True)

    return K.warp_affine(img, matrix, (height, width))


def scale(img, scale, interpolation="nearest"):
    height, width = img.shape[-2:]
    new_height, new_width = int(height * scale), int(width * scale)
    return resize(img, new_height, new_width, interpolation)


@on_4d_image(torch.float32)
def shift_scale_rotate(img, angle, scale, dx, dy, interpolation="nearest", border_mode="reflect"):
    # TODO add interpolation and border mode when kornia will add it
    # TODO add test when will be added interpolation and border mode
    interpolation = get_interpolation_mode(interpolation)

    height, width = img.shape[-2:]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    matrix = torch.from_numpy(matrix.astype(np.float32)).unsqueeze(0)

    return K.warp_affine(img, matrix.to(img.device, non_blocking=True), (width, height), interpolation, border_mode)


def get_center_crop_coords(height, width, crop_height, crop_width):
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def center_crop(img, crop_height, crop_width):
    height, width = img.shape[-2:]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    img = img[..., y1:y2, x1:x2]
    return img


def random_crop(img, crop_height, crop_width, h_start, w_start):
    height, width = img.shape[1:]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = AF.get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    return img[..., y1:y2, x1:x2]


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
    return img[..., int(y_min) : int(y_max), int(x_min) : int(x_max)]


@on_4d_image(torch.float32)
def optical_distortion(
    img, k=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None
):
    interpolation = get_interpolation_mode(interpolation)
    border_mode = get_border_mode(border_mode)

    height, width = img.shape[-2:]

    fx = width
    fy = width

    cx = width * 0.5 + dx
    cy = height * 0.5 + dy

    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)
    map_xy = np.stack([map1, map2], axis=-1)
    map_xy = map_xy.reshape((1,) + map_xy.shape)
    map_xy = torch.from_numpy(map_xy).to(img.device, non_blocking=True)

    img = FTorch.grid_sample(img, map_xy, mode=interpolation, padding_mode=border_mode)
    return img


@on_4d_image(torch.float32)
def grid_distortion(
    img, num_steps=10, xsteps=(), ysteps=(), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101
):
    interpolation = get_interpolation_mode(interpolation)
    border_mode = get_border_mode(border_mode)

    height, width = img.shape[-2:]

    x_step = width // num_steps
    xx = torch.zeros(width, dtype=torch.float32, device=img.device)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = torch.linspace(prev, cur, end - start, dtype=img.dtype, device=img.device)
        prev = cur

    y_step = height // num_steps
    yy = torch.zeros(height, dtype=torch.float32, device=img.device)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = torch.linspace(prev, cur, end - start, dtype=torch.float32, device=img.device)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    map_xy = np.stack([map_x, map_y], axis=-1)
    map_xy = map_xy.reshape((1,) + map_xy.shape)
    map_xy = torch.from_numpy(map_xy).to(img.device, non_blocking=True)

    return FTorch.grid_sample(img, map_xy, mode=interpolation, padding_mode=border_mode)
