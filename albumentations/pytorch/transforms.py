from __future__ import absolute_import

import warnings

import numpy as np
import torch
from torchvision.transforms import functional as F

from ..core.transforms_interface import BasicTransform


__all__ = ["ToTensor", "ToTensorV2"]


def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(im / (255.0 if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def mask_to_tensor(mask, num_classes, sigmoid):
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
            mask = np.moveaxis(mask / (255.0 if mask.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    else:
        mask = np.expand_dims(mask / (255.0 if mask.dtype == np.uint8 else 1), 0).astype(np.float32)
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
        super(ToTensor, self).__init__(always_apply=True, p=1.0)
        self.num_classes = num_classes
        self.sigmoid = sigmoid
        self.normalize = normalize
        warnings.warn(
            "ToTensor is deprecated and will be replaced by ToTensorV2 " "in albumentations 0.5.0", DeprecationWarning
        )

    def __call__(self, *args, force_apply=True, **kwargs):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        kwargs.update({"image": img_to_tensor(kwargs["image"], self.normalize)})
        if "mask" in kwargs.keys():
            kwargs.update({"mask": mask_to_tensor(kwargs["mask"], self.num_classes, sigmoid=self.sigmoid)})

        for k, _v in kwargs.items():
            if self._additional_targets.get(k) == "image":
                kwargs.update({k: img_to_tensor(kwargs[k], self.normalize)})
            if self._additional_targets.get(k) == "mask":
                kwargs.update({k: mask_to_tensor(kwargs[k], self.num_classes, sigmoid=self.sigmoid)})
        return kwargs

    @property
    def targets(self):
        raise NotImplementedError

    def get_transform_init_args_names(self):
        return "num_classes", "sigmoid", "normalize"


class ToTensorV2(BasicTransform):
    """Convert image and mask to `torch.Tensor`.

    Args:
        transpose_mask (bool): if True and an input mask has three dimensions, this transform will transpose dimensions
        so the shape `[height, width, num_channels]` becomes `[num_channels, height, width]`. The latter format is a
        standard format for PyTorch Tensors. Default: False.
    """

    def __init__(self, transpose_mask=False, always_apply=True, p=1.0):
        super(ToTensorV2, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}
