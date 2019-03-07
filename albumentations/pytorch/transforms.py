from __future__ import absolute_import

import numpy as np
import torch
from torchvision.transforms import functional as F

from ..core.transforms_interface import BasicTransform


__all__ = ['ToTensor']


def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def mask_to_tensor(mask, num_classes, sigmoid):
    # todo
    if num_classes > 1:
        if not sigmoid:
            # softmax
            long_mask = np.zeros((mask.shape[:2]), dtype=np.int64)
            if len(mask.shape) == 3:
                for c in range(mask.shape[2]):
                    long_mask[mask[..., c] > 0] = c
            else:
                long_mask[mask > 127] = 1
                long_mask[mask == 0] = 0
            mask = long_mask
        else:
            mask = np.moveaxis(mask / (255. if mask.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    else:
        mask = np.expand_dims(mask / (255. if mask.dtype == np.uint8 else 1), 0).astype(np.float32)
    return torch.from_numpy(mask)


class ToTensor(BasicTransform):
    """Convert image and mask to `torch.Tensor` and divide by 255 if image or mask are `uint8` type.
    WARNING! Please use this with care and look into sources before usage.

    Args:
        num_classes (int): only for segmentation
        sigmoid (bool, optional): only for segmentation, transform mask to LongTensor or not.
        normalize (dict, optional): dict with keys [mean, std] to pass it into torchvision.normalize

    """

    def __init__(self, num_classes=1, sigmoid=True, normalize=None):
        super(ToTensor, self).__init__(always_apply=True, p=1.)
        self.num_classes = num_classes
        self.sigmoid = sigmoid
        self.normalize = normalize

    def __call__(self, force_apply=True, **kwargs):
        kwargs.update({'image': img_to_tensor(kwargs['image'], self.normalize)})
        if 'mask' in kwargs.keys():
            kwargs.update({'mask': mask_to_tensor(kwargs['mask'], self.num_classes, sigmoid=self.sigmoid)})
        return kwargs

    @property
    def targets(self):
        raise NotImplementedError
