from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.transforms_interface import BasicTransform

__all__ = ["ToTensorV2"]


class ToTensorV2(BasicTransform):
    """Converts images/masks to PyTorch Tensors, inheriting from BasicTransform. Supports images in numpy `HWC` format
    and converts them to PyTorch `CHW` format. If the image is in `HW` format, it will be converted to PyTorch `HW`.

    Attributes:
        transpose_mask (bool): If True, transposes 3D input mask dimensions from `[height, width, num_channels]` to
            `[num_channels, height, width]`.
        always_apply (bool): Indicates if this transformation should be always applied. Default: True.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    def __init__(self, transpose_mask: bool = False, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self) -> Dict[str, Any]:
        return {"image": self.apply, "mask": self.apply_to_mask, "masks": self.apply_to_masks}

    def apply(self, img: np.ndarray, **params: Any) -> torch.Tensor:
        if len(img.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> torch.Tensor:
        if self.transpose_mask and mask.ndim == 3:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)

    def apply_to_masks(self, masks: List[np.ndarray], **params: Any) -> List[torch.Tensor]:
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params: Any) -> Dict[str, Any]:
        return {}
