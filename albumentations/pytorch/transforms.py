from __future__ import annotations

from typing import Any

import numpy as np
import torch

from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.types import (
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_VOLUME_DIMENSIONS,
    Targets,
)

__all__ = ["ToTensor3D", "ToTensorV2"]


class ToTensorV2(BasicTransform):
    """Converts images/masks to PyTorch Tensors, inheriting from BasicTransform.
    For images:
        - If input is in `HWC` format, converts to PyTorch `CHW` format
        - If input is in `HW` format, converts to PyTorch `1HW` format (adds channel dimension)

    Attributes:
        transpose_mask (bool): If True, transposes 3D input mask dimensions from `[height, width, num_channels]` to
            `[num_channels, height, width]`.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    _targets = (Targets.IMAGE, Targets.MASK)

    def __init__(self, transpose_mask: bool = False, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        self.transpose_mask = transpose_mask

    @property
    def targets(self) -> dict[str, Any]:
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
        }

    def apply(self, img: np.ndarray, **params: Any) -> torch.Tensor:
        if img.ndim not in {MONO_CHANNEL_DIMENSIONS, NUM_MULTI_CHANNEL_DIMENSIONS}:
            msg = "Albumentations only supports images in HW or HWC format"
            raise ValueError(msg)

        if img.ndim == MONO_CHANNEL_DIMENSIONS:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> torch.Tensor:
        if self.transpose_mask and mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)

    def apply_to_masks(self, masks: np.ndarray, **params: Any) -> torch.Tensor:
        """Convert numpy array of masks to torch tensor.

        Args:
            masks: numpy array of shape (N, H, W) or (N, H, W, C)
            params: Additional parameters

        Returns:
            torch.Tensor: If transpose_mask is True and input is (N, H, W, C),
                         returns tensor of shape (N, C, H, W).
                         Otherwise returns tensor with same shape as input.
        """
        if self.transpose_mask and masks.ndim == NUM_VOLUME_DIMENSIONS:  # (N, H, W, C)
            masks = np.transpose(masks, (0, 3, 1, 2))  # -> (N, C, H, W)
        return torch.from_numpy(masks)

    def apply_to_images(self, images: np.ndarray, **params: Any) -> torch.Tensor:
        """Convert batch of images from (N, H, W, C) to (N, C, H, W)."""
        if images.ndim != NUM_VOLUME_DIMENSIONS:  # N,H,W,C
            raise ValueError(f"Expected 4D array (N,H,W,C), got {images.ndim}D array")
        return torch.from_numpy(images.transpose(0, 3, 1, 2))  # -> (N,C,H,W)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("transpose_mask",)


class ToTensor3D(BasicTransform):
    """Convert 3D volumes and masks to PyTorch tensors.

    This transform is designed for 3D medical imaging data. It handles the conversion
    of numpy arrays to PyTorch tensors and performs necessary channel transpositions.

    For volumes:
        - Input:  (D, H, W, C) - depth, height, width, channels
        - Output: (C, D, H, W) - channels, depth, height, width

    For masks:
        - If transpose_mask=False:
            - Input:  (D, H, W) or (D, H, W, C)
            - Output: Same shape as input
        - If transpose_mask=True and mask has channels:
            - Input:  (D, H, W, C)
            - Output: (C, D, H, W)

    Args:
        transpose_mask (bool): If True and masks have channels, transposes masks from
            (D, H, W, C) to (C, D, H, W). Default: False
        p (float): Probability of applying the transform. Default: 1.0
    """

    _targets = (Targets.IMAGE, Targets.MASK)

    def __init__(self, transpose_mask: bool = False, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        self.transpose_mask = transpose_mask

    @property
    def targets(self) -> dict[str, Any]:
        return {
            "images": self.apply_to_images,
            "masks": self.apply_to_masks,
        }

    def apply_to_images(self, images: np.ndarray, **params: Any) -> torch.Tensor:
        """Convert 3D volume from (D,H,W,C) to (C,D,H,W)."""
        if images.ndim != NUM_VOLUME_DIMENSIONS:  # D,H,W,C
            raise ValueError(f"Expected 4D array (D,H,W,C), got {images.ndim}D array")
        return torch.from_numpy(images.transpose(3, 0, 1, 2))

    def apply_to_masks(self, masks: np.ndarray, **params: Any) -> torch.Tensor:
        """Convert 3D mask to tensor.

        If transpose_mask is True and mask has channels (D,H,W,C),
        converts to (C,D,H,W). Otherwise keeps the original shape.
        """
        if self.transpose_mask and masks.ndim == NUM_VOLUME_DIMENSIONS:  # D,H,W,C
            masks = masks.transpose(3, 0, 1, 2)
        return torch.from_numpy(masks)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("transpose_mask",)
