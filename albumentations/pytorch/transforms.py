from __future__ import absolute_import

import warnings

import numpy as np
import torch
from albumentations.augmentations.functional import is_grayscale_image
from torchvision.transforms import functional as F

from ..core.transforms_interface import BasicTransform


__all__ = ["ToTensor", "ToTensorV2", "FromTensorV2", "BasicTransformTorch"]


class BasicTransformTorch(BasicTransform):
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value

        if isinstance(kwargs["image"], np.ndarray):
            params.update({"cols": kwargs["image"].shape[1], "rows": kwargs["image"].shape[0]})
        else:
            params.update({"cols": kwargs["image"].shape[2], "rows": kwargs["image"].shape[1]})
        return params


def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(im / (255.0 if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
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

    def __call__(self, force_apply=True, **kwargs):
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
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, always_apply=True, device="cpu", p=1.0):
        super(ToTensorV2, self).__init__(always_apply=always_apply, p=p)
        self.device = device

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):
        if is_grayscale_image(img):
            img = img.reshape((1,) + img.shape[:2])

        return torch.from_numpy(img.transpose(2, 0, 1)).to(self.device)

    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask).to(self.device)

    def get_transform_init_args_names(self):
        return ("device",)

    def get_params_dependent_on_targets(self, params):
        return {}


class FromTensorV2(BasicTransform):
    """Convert image nad mask from `torch.Tensor` to `numpy.ndarray`"""

    def __init__(self, always_apply=True, p=1.0):
        super(FromTensorV2, self).__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):
        return img.detach().cpu().numpy().transpose(1, 2, 0).sequzee()

    def apply_to_mask(self, mask, **params):
        return mask.detach().cpu().numpy()

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}
