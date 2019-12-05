import albumentations as A
from . import functional as F
import torch

import random


__all__ = ["NormalizeTorch", "CoarseDropoutTorch", "RandomSnowTorch", "BlurTorch", "HueSaturationValueTorch"]


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
