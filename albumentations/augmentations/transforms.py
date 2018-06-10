import cv2
import random

import numpy as np

from ..core.transforms_interface import to_tuple, DualTransform, ImageOnlyTransform
from . import functional as F


__all__ = ['VerticalFlip', 'HorizontalFlip', 'Flip', 'Transpose', 'RandomRotate90',
       'Rotate', 'ShiftScaleRotate', 'CenterCrop', 'Distort1', 'Distort2',
       'ElasticTransform', 'ElasticTransform', 'HueSaturationValue',
       'RGBShift', 'RandomBrightness', 'RandomContrast', 'Blur', 'MotionBlur',
       'MedianBlur', 'GaussNoise', 'CLAHE', 'ChannelShuffle', 'InvertImg',
       'ToGray']


class VerticalFlip(DualTransform):
    def apply(self, img, **params):
        return F.vflip(img)


class HorizontalFlip(DualTransform):
    def apply(self, img, **params):
        return F.hflip(img)


class Flip(DualTransform):
    def apply(self, img, d=0):
        return F.random_flip(img, d)

    def get_params(self):
        return {'d': random.randint(-1, 1)}


class Transpose(DualTransform):
    def apply(self, img, **params):
        return F.transpose(img)


class RandomRotate90(DualTransform):
    def apply(self, img, factor=0):
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        return {'factor': random.randint(0, 4)}


class Rotate(DualTransform):
    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=.5):
        super().__init__(p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, angle=0):
        return F.rotate(img, angle, self.interpolation, self.border_mode)

    def get_params(self):
        return {'angle': random.uniform(self.limit[0], self.limit[1])}


class ShiftScaleRotate(DualTransform):
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5):
        super().__init__(p)
        self.shift_limit = to_tuple(shift_limit)
        self.scale_limit = to_tuple(scale_limit)
        self.rotate_limit = to_tuple(rotate_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, angle=0, scale=0, dx=0, dy=0):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, self.interpolation, self.border_mode)

    def get_params(self):
        return {'angle': random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
                'scale': random.uniform(1 + self.scale_limit[0], 1 + self.scale_limit[1]),
                'dx': round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
                'dy': round(random.uniform(self.shift_limit[0], self.shift_limit[1]))}


class CenterCrop(DualTransform):
    def __init__(self, height, width, p=0.5):
        super().__init__(p)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        return F.center_crop(img, self.height, self.width)


class Distort1(DualTransform):
    def __init__(self, distort_limit=0.05, shift_limit=0.05, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5):
        super().__init__(p)
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.shift_limit = to_tuple(shift_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, k=0, dx=0, dy=0):
        return F.distort1(img, k, dx, dy, self.interpolation, self.border_mode)

    def get_params(self):
        return {'k': random.uniform(self.distort_limit[0], self.distort_limit[1]),
                'dx': round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
                'dy': round(random.uniform(self.shift_limit[0], self.shift_limit[1]))}


class Distort2(DualTransform):
    def __init__(self, num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5):
        super().__init__(p)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.p = p
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, stepsx=[], stepsy=[]):
        return F.distort2(img, self.num_steps, stepsx, stepsy, self.interpolation, self.border_mode)

    def get_params(self):
        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        return {
            'stepsx': stepsx,
            'stepsy': stepsy
        }


class ElasticTransform(DualTransform):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5):
        super().__init__(p)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, random_state=None):
        return F.elastic_transform_fast(img, self.alpha, self.sigma, self.alpha_affine, self.interpolation,
                                        self.border_mode, np.random.RandomState(random_state))

    def get_params(self):
        return {'random_state': np.random.randint(0, 10000)}


class HueSaturationValue(ImageOnlyTransform):
    def __init__(self, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5):
        # def __init__(self, hue_shift_limit=100, sat_shift_limit=50, val_shift_limit=10, p=0.5, targets=('image')):
        super().__init__(p)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0):
        assert image.dtype == np.uint8 or self.hue_shift_limit < 1.
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return {'hue_shift': np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
                'sat_shift': np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
                'val_shift': np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])}


class RGBShift(ImageOnlyTransform):
    def __init__(self, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5):
        super().__init__(p)
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def apply(self, image, r_shift=0, g_shift=0, b_shift=0):
        return F.shift_rgb(image, r_shift, g_shift, b_shift)

    def get_params(self):
        return {'r_shift': np.random.uniform(self.r_shift_limit[0], self.r_shift_limit[1]),
                'g_shift': np.random.uniform(self.g_shift_limit[0], self.g_shift_limit[1]),
                'b_shift': np.random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])}


class RandomBrightness(ImageOnlyTransform):
    def __init__(self, limit=0.2, p=0.5):
        super().__init__(p)
        self.limit = to_tuple(limit)

    def apply(self, img, alpha=0.2):
        return F.random_brightness(img, alpha)

    def get_params(self):
        return {"alpha": 1.0 + np.random.uniform(self.limit[0], self.limit[1])}


class RandomContrast(ImageOnlyTransform):
    def __init__(self, limit=.2, p=.5):
        super().__init__(p)
        self.limit = to_tuple(limit)

    def apply(self, img, alpha=0.2):
        return F.random_contrast(img, alpha)

    def get_params(self):
        return {"alpha": 1.0 + np.random.uniform(self.limit[0], self.limit[1])}


class Blur(ImageOnlyTransform):
    def __init__(self, blur_limit=7, p=.5):
        super().__init__(p)
        self.blur_limit = to_tuple(blur_limit, 3)

    def apply(self, image, ksize=3):
        return F.blur(image, ksize)

    def get_params(self):
        return {
            'ksize': np.random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        }


class MotionBlur(Blur):
    def apply(self, img, ksize=9):
        return F.motion_blur(img, ksize=ksize)


class MedianBlur(Blur):
    def apply(self, image, ksize=3):
        return F.median_blur(image, ksize)


class GaussNoise(ImageOnlyTransform):
    def __init__(self, var_limit=(10, 50), p=.5):
        super().__init__(p)
        self.var_limit = to_tuple(var_limit)

    def apply(self, img, var=30):
        return F.gauss_noise(img, var=var)

    def get_params(self):
        return {
            'var': np.random.randint(self.var_limit[0], self.var_limit[1])
        }


class CLAHE(ImageOnlyTransform):
    def __init__(self, clipLimit=4.0, tileGridSize=(8, 8), p=0.5):
        super().__init__(p)
        self.clipLimit = to_tuple(clipLimit, 1)
        self.tileGridSize = tileGridSize

    def apply(self, img, clipLimit=2):
        return F.clahe(img, clipLimit, self.tileGridSize)

    def get_params(self):
        return {"clipLimit": np.random.uniform(self.clipLimit[0], self.clipLimit[1])}


class ChannelShuffle(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.channel_shuffle(img)


class InvertImg(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.invert(img)


class ToGray(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.to_gray(img)


