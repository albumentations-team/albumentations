from __future__ import absolute_import

import torch

from ..core.transforms_interface import BasicTransform


__all__ = ['ToTensor']


class ToTensor(BasicTransform):
    """Convert image and mask to `torch.Tensor`.

    """
    def __init__(self):
        super(ToTensor, self).__init__(always_apply=True)

    @property
    def targets(self):
        return {
            'image': self.apply,
            'mask': self.apply_to_mask
        }

    def apply(self, img, **params):
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return {}

    def get_params_dependent_on_targets(self, params):
        return {}
