import albumentations as A
from . import functional as F
import torch

import random
import numpy as np


__all__ = [
    "NormalizeTorch",
    "CoarseDropoutTorch",
    "RandomSnowTorch",
    "BlurTorch",
    "HueSaturationValueTorch",
    "SolarizeTorch",
    "RGBShiftTorch",
    "RandomBrightnessContrastTorch",
    "RandomBrightnessTorch",
    "RandomContrastTorch",
    "MotionBlurTorch",
    "MedianBlurTorch",
    "GaussianBlurTorch",
    "ISONoiseTorch",
    "ChannelDropoutTorch",
    "InvertImgTorch",
    "RandomGammaTorch",
    "ChannelShuffleTorch",
    "ToGrayTorch",
]


class NormalizeTorch(A.Normalize):
    def __init__(
        self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0
    ):
        super(NormalizeTorch, self).__init__(mean, std, max_pixel_value, always_apply, p)
        self.mean = torch.tensor(self.mean) * self.max_pixel_value
        self.std = torch.tensor(self.std) * self.max_pixel_value

    def apply(self, image, **params):
        return F.normalize(image.type(torch.float32), self.mean, self.std)


class CoarseDropoutTorch(A.CoarseDropout):
    def apply(self, image, holes=(), **params):
        return F.cutout(image, holes, self.fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[-2:]

        holes = []
        for _n in range(random.randint(self.min_holes, self.max_holes)):
            hole_height = random.randint(self.min_height, self.max_height)
            hole_width = random.randint(self.min_width, self.max_width)

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}


class RandomSnowTorch(A.RandomSnow):
    def apply(self, image, snow_point=0.1, **params):
        return F.add_snow(image, snow_point, self.brightness_coeff)


class BlurTorch(A.Blur):
    def apply(self, image, ksize=3, **params):
        ksize = A.to_tuple(ksize, ksize)
        return F.blur(image, ksize)


class HueSaturationValueTorch(A.HueSaturationValue):
    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0, **params):
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)


class SolarizeTorch(A.Solarize):
    def apply(self, image, threshold=0, **params):
        return F.solarize(image, threshold)


class RGBShiftTorch(A.RGBShift):
    def apply(self, image, r_shift=0, g_shift=0, b_shift=0, **params):
        return F.shift_rgb(image, r_shift, g_shift, b_shift)


class RandomBrightnessContrastTorch(A.RandomBrightnessContrast):
    def apply(self, img, alpha=1.0, beta=0.0, **params):
        return F.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)


class RandomBrightnessTorch(A.RandomBrightness):
    def apply(self, img, alpha=1.0, beta=0.0, **params):
        return F.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)


class RandomContrastTorch(A.RandomContrast):
    def apply(self, img, alpha=1.0, beta=0.0, **params):
        return F.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)


class MotionBlurTorch(A.MotionBlur):
    def apply(self, img, kernel=None, **params):
        return F.motion_blur(img, kernel)


class MedianBlurTorch(A.MedianBlur):
    def apply(self, image, ksize=3, **params):
        ksize = A.to_tuple(ksize, ksize)
        return F.median_blur(image, ksize)


class GaussianBlurTorch(A.GaussianBlur):
    def apply(self, image, ksize=3, **params):
        ksize = A.to_tuple(ksize, ksize)
        return F.gaussian_blur(image, ksize)


class ISONoiseTorch(A.ISONoise):
    def apply(self, img, color_shift=0.05, intensity=1.0, random_state=None, **params):
        return F.iso_noise(img, color_shift, intensity, np.random.RandomState(random_state))


class ChannelDropoutTorch(A.ChannelDropout):
    def apply(self, img, channels_to_drop=(0,), **params):
        return F.channel_dropout(img, channels_to_drop, self.fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]

        num_channels = img.size(0)

        if num_channels == 1:
            raise NotImplementedError("Images has one channel. ChannelDropout is not defined.")

        if self.max_channels >= num_channels:
            raise ValueError("Can not drop all channels in ChannelDropout.")

        num_drop_channels = random.randint(self.min_channels, self.max_channels)

        channels_to_drop = random.sample(range(num_channels), k=num_drop_channels)

        return {"channels_to_drop": channels_to_drop}


class InvertImgTorch(A.InvertImg):
    def apply(self, img, **params):
        return F.invert(img)


class RandomGammaTorch(A.RandomGamma):
    def apply(self, img, gamma=1, **params):
        return F.gamma_transform(img, gamma=gamma, eps=self.eps)


class ChannelShuffleTorch(A.ChannelShuffle):
    def apply(self, img, channels_shuffled=(0, 1, 2), **params):
        return F.channel_shuffle(img, channels_shuffled)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        ch_arr = list(range(img.size(0)))
        random.shuffle(ch_arr)
        return {"channels_shuffled": ch_arr}


class ToGrayTorch(A.ToGray):
    def apply(self, img, **params):
        return F.to_gray(img)
