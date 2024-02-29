import random
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple, TypedDict, Union

import numpy as np
from typing_extensions import NotRequired

from albumentations.core.transforms_interface import DualTransform
from albumentations.random_utils import beta

from .functional import mix_arrays

__all__ = ["MixUp"]


class ReferenceImage(TypedDict):
    image: Union[str, Path]
    mask: NotRequired[np.ndarray]
    class_label: NotRequired[np.ndarray]


class MixUp(DualTransform):
    """MixUp data augmentation for images, masks, and class labels.

    This transformation performs the MixUp augmentation by blending an input image, mask, and class label
    with another set from a predefined reference dataset. The blending is controlled by a parameter lambda,
    sampled from a Beta distribution, dictating the proportion of the mix between the original and reference data.

    The MixUp augmentation is known for improving model generalization by encouraging linear behavior between
    classes and smoothing the decision boundaries. It can be applied not only to the images but also to the
    segmentation masks and class labels, providing a comprehensive data augmentation strategy.

    Reference:
        Zhang, H., Cisse, M., Dauphin, Y.N., and Lopez-Paz, D., mixup: Beyond Empirical Risk Minimization,
        ICLR 2018. https://arxiv.org/abs/1710.09412

    Args:
    ----
        reference_data Sequence[ReferenceImage]: A sequence of dictionaries containing the reference
            images, masks, and class labels for mixing. Each dictionary should have 'image', but 'mask', and
            'class_label' are optional. Defaults to an empty list, resulting in no operation if not provided.
        read_fn Callable[[Any], Dict[str, np.ndarray]]: A function to load and process the data
            from the reference_data dictionaries. It should accept one argument (one of the dictionaries) and
            return a processed dictionary containing the keys 'image' and optionally 'mask', and 'class_label'
            with their corresponding np.ndarray values. Class label should be one-hot encoded.
            Defaults to a lambda function that returns its input.
        alpha (float, optional): The alpha parameter of the Beta distribution used to sample the lambda value.
            Must be greater than or equal to 0. Higher values make the distribution closer to uniform,
            resulting in more balanced mixing. Defaults to 0.4.
        always_apply (bool, optional): Whether the transform should always be applied, regardless of the
            probability `p`. Defaults to False.
        p (float, optional): Probability that the transform will be applied. Defaults to 0.5.

    Raises:
    ------
        ValueError: If the alpha parameter is negative.

    Notes:
    -----
        - If no reference data is provided, this transform will issue a warning and act as a no-op.
        - The lambda value used for mixing is sampled once per batch and applied globally across all images
          and labels within that batch.
        - The class labels are expected to be numpy arrays, enabling the mixing process when applicable.

    """

    def __init__(
        self,
        reference_data: Sequence[ReferenceImage] = [],
        read_fn: Callable[[Any], np.ndarray] = lambda x: x,
        alpha: float = 0.4,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        if alpha < 0:
            msg = "Alpha must be >= 0."
            raise ValueError(msg)

        if not reference_data:
            warnings.warn("No reference data provided for MixUp. This transform will act as a no-op.")

        self.alpha = alpha
        self.reference_data = reference_data
        self.read_fn = read_fn

    def apply(self, img: np.ndarray, mix_data: ReferenceImage, lam: float, **params: Any) -> np.ndarray:
        mix_img = mix_data.get("image")
        return mix_arrays(img, mix_img, lam) if mix_img is not None else img

    def apply_to_mask(self, mask: np.ndarray, mix_data: ReferenceImage, lam: float, **params: Any) -> np.ndarray:
        mix_mask = mix_data.get("mask")
        return mix_arrays(mask, mix_mask, lam) if mix_mask is not None else mask

    def apply_to_class_label(
        self, label: np.ndarray, mix_data: ReferenceImage, lam: float, **params: Any
    ) -> np.ndarray:
        mix_label = mix_data.get("class_label")
        if mix_label is not None and label is not None:
            return lam * label + (1 - lam) * mix_label
        return label

    def get_params(self) -> Dict[str, Any]:
        if not self.reference_data:
            return {"mix_data": None, "lam": 1.0}

        mix_idx = random.randint(0, len(self.reference_data) - 1)
        lam = beta(self.alpha, self.alpha)

        mix_data = self.read_fn(self.reference_data[mix_idx])

        return {"mix_data": mix_data, "lam": lam}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "reference_data", "alpha"
