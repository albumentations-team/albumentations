"""Module containing PyTorch-specific transforms for Albumentations.

This module provides transforms that convert NumPy arrays to PyTorch tensors in
the appropriate format. It handles both 2D image data and 3D volumetric data,
ensuring that the tensor dimensions are correctly arranged according to PyTorch's
expected format (channels first). These transforms are typically used as the final
step in an augmentation pipeline before feeding data to a PyTorch model.
"""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import torch

from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.type_definitions import (
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

    def __init__(self, transpose_mask: bool = False, p: float = 1.0):
        super().__init__(p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self) -> dict[str, Any]:
        """Define mapping of target name to target function.

        Returns:
            dict[str, Any]: Dictionary mapping target names to corresponding transform functions.

        """
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
        }

    def apply(self, img: np.ndarray, **params: Any) -> torch.Tensor:
        """Convert a 2D image array to a PyTorch tensor.

        Converts image from HWC or HW format to CHW format, handling both
        single-channel and multi-channel images.

        Args:
            img (np.ndarray): Image as a numpy array of shape (H,W) or (H,W,C)
            **params (Any): Additional parameters

        Returns:
            torch.Tensor: PyTorch tensor in CHW format

        Raises:
            ValueError: If image dimensions are neither HW nor HWC

        """
        if img.ndim not in {MONO_CHANNEL_DIMENSIONS, NUM_MULTI_CHANNEL_DIMENSIONS}:
            msg = "Albumentations only supports images in HW or HWC format"
            raise ValueError(msg)

        if img.ndim == MONO_CHANNEL_DIMENSIONS:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> torch.Tensor:
        """Convert a mask array to a PyTorch tensor.

        If transpose_mask is True and mask has 3 dimensions (H,W,C),
        converts mask to channels-first format (C,H,W).

        Args:
            mask (np.ndarray): Mask as a numpy array
            **params (Any): Additional parameters

        Returns:
            torch.Tensor: PyTorch tensor of mask

        """
        if self.transpose_mask and mask.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
            mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask)

    @overload
    def apply_to_masks(self, masks: list[np.ndarray], **params: Any) -> list[torch.Tensor]: ...

    @overload
    def apply_to_masks(self, masks: np.ndarray, **params: Any) -> torch.Tensor: ...

    def apply_to_masks(self, masks: np.ndarray | list[np.ndarray], **params: Any) -> torch.Tensor | list[torch.Tensor]:
        """Convert numpy array or list of numpy array masks to torch tensor(s).

        Args:
            masks (np.ndarray | list[np.ndarray]): Numpy array of shape (N, H, W) or (N, H, W, C),
                or a list of numpy arrays with shape (H, W) or (H, W, C).
            **params (Any): Additional parameters.

        Returns:
            torch.Tensor | list[torch.Tensor]: If transpose_mask is True and input is (N, H, W, C),
                returns tensor of shape (N, C, H, W). If transpose_mask is True and input is (H, W, C), r
                eturns a list of tensors with shape (C, H, W). Otherwise, returns tensors with the same shape as input.

        """
        if isinstance(masks, list):
            return [self.apply_to_mask(mask, **params) for mask in masks]

        if self.transpose_mask and masks.ndim == NUM_VOLUME_DIMENSIONS:  # (N, H, W, C)
            masks = np.transpose(masks, (0, 3, 1, 2))  # -> (N, C, H, W)
        return torch.from_numpy(masks)

    def apply_to_images(self, images: np.ndarray, **params: Any) -> torch.Tensor:
        """Convert batch of images from (N, H, W, C) to (N, C, H, W)."""
        if images.ndim != NUM_VOLUME_DIMENSIONS:  # N,H,W,C
            raise ValueError(f"Expected 4D array (N,H,W,C), got {images.ndim}D array")
        return torch.from_numpy(images.transpose(0, 3, 1, 2))  # -> (N,C,H,W)


class ToTensor3D(BasicTransform):
    """Convert 3D volumes and masks to PyTorch tensors.

    This transform is designed for 3D medical imaging data. It converts numpy arrays
    to PyTorch tensors and ensures consistent channel positioning.

    For all inputs (volumes and masks):
        - Input:  (D, H, W, C) or (D, H, W) - depth, height, width, [channels]
        - Output: (C, D, H, W) - channels first format for PyTorch
                 For single-channel input, adds C=1 dimension

    Note:
        This transform always moves channels to first position as this is
        the standard PyTorch format. For masks that need to stay in DHWC format,
        use a different transform or handle the transposition after this transform.

    Args:
        p (float): Probability of applying the transform. Default: 1.0

    """

    _targets = (Targets.IMAGE, Targets.MASK)

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    @property
    def targets(self) -> dict[str, Any]:
        """Define mapping of target name to target function.

        Returns:
            dict[str, Any]: Dictionary mapping target names to corresponding transform functions

        """
        return {
            "volume": self.apply_to_volume,
            "mask3d": self.apply_to_mask3d,
        }

    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> torch.Tensor:
        """Convert 3D volume to channels-first tensor."""
        if volume.ndim == NUM_VOLUME_DIMENSIONS:  # D,H,W,C
            return torch.from_numpy(volume.transpose(3, 0, 1, 2))
        if volume.ndim == NUM_VOLUME_DIMENSIONS - 1:  # D,H,W
            return torch.from_numpy(volume[np.newaxis, ...])
        raise ValueError(f"Expected 3D or 4D array (D,H,W) or (D,H,W,C), got {volume.ndim}D array")

    def apply_to_mask3d(self, mask3d: np.ndarray, **params: Any) -> torch.Tensor:
        """Convert 3D mask to channels-first tensor."""
        return self.apply_to_volume(mask3d, **params)
