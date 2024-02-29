import random
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
from typing_extensions import NotRequired

from albumentations.core.transforms_interface import DualTransform
from albumentations.random_utils import beta

from .functional import mix_arrays

__all__ = ["MixUp"]


class ReferenceImage(TypedDict):
    image: Union[str, Path]
    mask: NotRequired[np.ndarray]
    global_label: NotRequired[np.ndarray]


class MixUp(DualTransform):
    """Performs MixUp data augmentation, blending images, masks, and class labels with reference data.

    MixUp augmentation linearly combines an input (image, mask, and class label) with another set from a predefined
    reference dataset. The mixing degree is controlled by a parameter λ (lambda), sampled from a Beta distribution.
    This method is known for improving model generalization by promoting linear behavior between classes and
    smoothing decision boundaries.

    Reference:
        Zhang, H., Cisse, M., Dauphin, Y.N., and Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization.
        In International Conference on Learning Representations. https://arxiv.org/abs/1710.09412

    Args:
    ----
        reference_data (Optional[Union[Generator[ReferenceImage, None, None], Sequence[ReferenceImage]]]):
            A sequence or generator of dictionaries containing the reference data for mixing. Each dictionary
            should contain:
                - 'image': Mandatory key with an image array.
                - 'mask': Optional key with a mask array.
                - 'global_label': Optional key with a class label array.
            If None or an empty sequence is provided, no operation is performed and a warning is issued.
        read_fn (Callable[[ReferenceImage], Dict[str, Any]]):
            A function to process items from reference_data. It should accept a dictionary from reference_data
            and return a processed dictionary containing 'image', and optionally 'mask' and 'global_label',
            each as numpy arrays. Defaults to a no-op lambda function.
        alpha (float):
            The alpha parameter for the Beta distribution, influencing the mix's balance. Must be ≥ 0.
            Higher values lead to more uniform mixing. Defaults to 0.4.
        p (float):
            The probability of applying the transformation. Defaults to 0.5.

    Targets:
        - image: The input image to augment.
        - mask: An optional segmentation mask corresponding to the input image.
        - global_label: An optional global label associated with the input image.

    Image types:
        - uint8, float32

    Raises:
    ------
        - ValueError: If the alpha parameter is negative.

    Notes:
    -----
        - If no reference data is provided, a warning is issued, and the transform acts as a no-op.

    """

    def __init__(
        self,
        reference_data: Optional[Union[Generator[ReferenceImage, None, None], Sequence[ReferenceImage]]] = None,
        read_fn: Callable[[ReferenceImage], Dict[str, Any]] = lambda x: {"image": x, "mask": None, "class_label": None},
        alpha: float = 0.4,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        if alpha < 0:
            msg = "Alpha must be >= 0."
            raise ValueError(msg)

        self.read_fn = read_fn
        self.alpha = alpha

        if reference_data is None:
            warnings.warn("No reference data provided for MixUp. This transform will act as a no-op.")
            # Create an empty generator
        self.reference_data = reference_data or []

    def apply(self, img: np.ndarray, mix_data: ReferenceImage, mix_coef: float, **params: Any) -> np.ndarray:
        mix_img = mix_data.get("image")
        return mix_arrays(img, mix_img, mix_coef) if mix_img is not None else img

    def apply_to_mask(self, mask: np.ndarray, mix_data: ReferenceImage, mix_coef: float, **params: Any) -> np.ndarray:
        mix_mask = mix_data.get("mask")
        return mix_arrays(mask, mix_mask, mix_coef) if mix_mask is not None else mask

    def apply_to_global_label(
        self, label: np.ndarray, mix_data: ReferenceImage, mix_coef: float, **params: Any
    ) -> np.ndarray:
        mix_label = mix_data.get("global_label")
        if mix_label is not None and label is not None:
            return mix_coef * label + (1 - mix_coef) * mix_label
        return label

    def get_params(self) -> Dict[str, Union[None, float, Dict[str, Any]]]:
        if self.reference_data and isinstance(self.reference_data, Sequence):
            mix_idx = random.randint(0, len(self.reference_data) - 1)
            mix_data = self.reference_data[mix_idx]
        elif self.reference_data and isinstance(self.reference_data, Iterator):
            try:
                mix_data = next(self.reference_data)  # Get the next item from the iterator
            except StopIteration:
                warnings.warn(
                    "Reference data iterator/generator has been exhausted. "
                    "Further MixUp augmentations will not be applied.",
                    RuntimeWarning,
                )
                return {"mix_data": None, "mix_coef": 1}

        mix_coef = beta(self.alpha, self.alpha) if mix_data else 1
        return {"mix_data": self.read_fn(mix_data) if mix_data else None, "mix_coef": mix_coef}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "reference_data", "alpha"
