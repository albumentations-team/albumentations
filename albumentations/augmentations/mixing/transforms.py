import random
import warnings
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from albumentations.augmentations.utils import read_rgb_image
from albumentations.core.transforms_interface import DualTransform
from albumentations.random_utils import beta

from .functional import mix_arrays


class MixUp(DualTransform):
    """Perform the MixUp augmentation, combining pairs of images and labels in a convex manner to create new samples.

    MixUp augmentation blends an image with another (referred to as the 'reference image') and does the same
    with their respective labels, based on a mixing parameter lambda sampled from a Beta distribution.
    This approach helps in regularizing neural network models, often leading to improved performance by
    encouraging the model to learn more robust features.

    The original MixUp paper can be found here: https://arxiv.org/abs/1710.09412

    Args:
    ----
        reference_images (Sequence[Any]): A sequence containing the reference images for mixing.
        reference_labels (Sequence[np.ndarray]): A sequence containing the labels of the reference images.
        read_fn (Callable[[Any], np.ndarray]): Function to read and load the image, given an element from
                                               reference_images. Default is read_rgb_image.
        alpha (float): The parameter for the Beta distribution used to mix images and labels. Must be >= 0.
                       Higher values make the distribution closer to uniform, resulting in more balanced mixes.
        always_apply (bool): Whether the transform should always be applied.
        p (float): Probability that the transform will be applied.

    Raises:
    ------
        ValueError: If alpha is negative or if the number of reference images does not match the number of labels.

    Note:
    ----
        If no reference images are provided, this transform will act as a no-op.

    """

    def __init__(
        self,
        reference_images: Sequence[Any] = [],
        reference_labels: Sequence[np.ndarray] = [],
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        alpha: float = 0.4,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        if alpha < 0:
            msg = "Alpha must be >= 0."
            raise ValueError(msg)
        if len(reference_images) != len(reference_labels):
            msg = "The number of reference images must match the number of reference labels."
            raise ValueError(msg)
        if not reference_images:
            warnings.warn("No reference images provided for MixUp. This transform will act as a no-op.")

        self.alpha = alpha
        self.reference_images = reference_images
        self.reference_labels = reference_labels
        self.read_fn = read_fn

    def apply(self, img: np.ndarray, mix_img: np.ndarray, lam: float, **params: Any) -> np.ndarray:
        return mix_arrays(img, mix_img, lam) if mix_img is not None else img

    def apply_to_class_label(self, label: np.ndarray, mix_label: np.ndarray, lam: float, **params: Any) -> np.ndarray:
        return lam * label + (1 - lam) * mix_label if mix_label is not None else label

    def apply_to_mask(self, mask: np.ndarray, mix_mask: np.ndarray, lam: float, **params: Any) -> np.ndarray:
        return mix_arrays(mask, mix_mask, lam) if mix_mask is not None else mask

    def get_params(self) -> Dict[str, Any]:
        if not self.reference_images:  # Check if reference images list is empty
            return {"mix_img": None, "mix_label": None, "lam": 1.0}  # Return no-op parameters

        mix_idx = random.randint(0, len(self.reference_images) - 1)
        lam = beta(self.alpha, self.alpha)

        mix_img = self.read_fn(self.reference_images[mix_idx])
        mix_label = self.reference_labels[mix_idx]

        return {"mix_img": mix_img, "mix_label": mix_label, "lam": lam}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "reference_images", "reference_labels", "alpha"
