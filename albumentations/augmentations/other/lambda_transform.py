"""Lambda transform module for creating custom user-defined transformations.

This module provides a flexible transform class that allows users to define their own
custom transformation functions for different targets (image, mask, keypoints, bboxes).
It's particularly useful for implementing custom logic that isn't available in the
standard transforms.

The Lambda transform accepts different callable functions for each target type and
applies them when the transform is executed. This allows for maximum flexibility
while maintaining compatibility with the Albumentations pipeline structure.

Key features:
- Apply different custom functions to different target types
- Compatible with all Albumentations pipeline features
- Support for all image types and formats
- Ability to handle any number of channels
- Warning system for lambda expressions and multiprocessing compatibility

Note that using actual lambda expressions (rather than named functions) can cause
issues with multiprocessing, as lambdas cannot be properly pickled.
"""

from __future__ import annotations

import warnings
from types import LambdaType
from typing import Any, Callable

import numpy as np

from albumentations.augmentations.pixel import functional as fpixel
from albumentations.core.transforms_interface import NoOp
from albumentations.core.utils import format_args

__all__ = ["Lambda"]


class Lambda(NoOp):
    """A flexible transformation class for using user-defined transformation functions per targets.
    Function signature must include **kwargs to accept optional arguments like interpolation method, image size, etc:

    Args:
        image (Callable[..., Any] | None): Image transformation function.
        mask (Callable[..., Any] | None): Mask transformation function.
        keypoints (Callable[..., Any] | None): Keypoints transformation function.
        bboxes (Callable[..., Any] | None): BBoxes transformation function.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Number of channels:
        Any

    """

    def __init__(
        self,
        image: Callable[..., Any] | None = None,
        mask: Callable[..., Any] | None = None,
        keypoints: Callable[..., Any] | None = None,
        bboxes: Callable[..., Any] | None = None,
        name: str | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.name = name
        self.custom_apply_fns = dict.fromkeys(("image", "mask", "keypoints", "bboxes"), fpixel.noop)
        for target_name, custom_apply_fn in {
            "image": image,
            "mask": mask,
            "keypoints": keypoints,
            "bboxes": bboxes,
        }.items():
            if custom_apply_fn is not None:
                if isinstance(custom_apply_fn, LambdaType) and custom_apply_fn.__name__ == "<lambda>":
                    warnings.warn(
                        "Using lambda is incompatible with multiprocessing. "
                        "Consider using regular functions or partial().",
                        stacklevel=2,
                    )

                self.custom_apply_fns[target_name] = custom_apply_fn

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the Lambda transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the Lambda transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied Lambda transform.

        """
        fn = self.custom_apply_fns["image"]
        return fn(img, **params)

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the Lambda transform to the input mask.

        Args:
            mask (np.ndarray): The input mask to apply the Lambda transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The mask with the applied Lambda transform.

        """
        fn = self.custom_apply_fns["mask"]
        return fn(mask, **params)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the Lambda transform to the input bounding boxes.

        Args:
            bboxes (np.ndarray): The input bounding boxes to apply the Lambda transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The bounding boxes with the applied Lambda transform.

        """
        fn = self.custom_apply_fns["bboxes"]
        return fn(bboxes, **params)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the Lambda transform to the input keypoints.

        Args:
            keypoints (np.ndarray): The input keypoints to apply the Lambda transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The keypoints with the applied Lambda transform.

        """
        fn = self.custom_apply_fns["keypoints"]
        return fn(keypoints, **params)

    @classmethod
    def is_serializable(cls) -> bool:
        """Check if the Lambda transform is serializable.

        Returns:
            bool: True if the transform is serializable, False otherwise.

        """
        return False

    def to_dict_private(self) -> dict[str, Any]:
        """Convert the Lambda transform to a dictionary.

        Returns:
            dict[str, Any]: The dictionary representation of the transform.

        """
        if self.name is None:
            msg = (
                "To make a Lambda transform serializable you should provide the `name` argument, "
                "e.g. `Lambda(name='my_transform', image=<some func>, ...)`."
            )
            raise ValueError(msg)
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}

    def __repr__(self) -> str:
        """Return the string representation of the Lambda transform.

        Returns:
            str: The string representation of the Lambda transform.

        """
        state = {"name": self.name}
        state.update(self.custom_apply_fns.items())  # type: ignore[arg-type]
        state.update(self.get_base_init_args())
        return f"{self.__class__.__name__}({format_args(state)})"
