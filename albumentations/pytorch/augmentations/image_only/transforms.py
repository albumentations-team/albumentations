import torch

import albumentations as A

from . import functional as F
from ...transforms import BasicTransformTorch

__all__ = ["NormalizeTorch", "ISONoiseTorch"]


class NormalizeTorch(BasicTransformTorch, A.Normalize):
    def __init__(
        self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0
    ):
        super(NormalizeTorch, self).__init__(mean, std, max_pixel_value, always_apply, p)
        self.mean = torch.tensor(self.mean) * self.max_pixel_value
        self.std = torch.tensor(self.std) * self.max_pixel_value

    def apply(self, image, **params):
        return F.normalize(image.to(torch.float32), self.mean, self.std)


class ISONoiseTorch(BasicTransformTorch, A.ISONoise):
    def apply(self, img, color_shift=0.05, intensity=1.0, random_state=None, **params):
        return F.iso_noise(img, color_shift, intensity)
