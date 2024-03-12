import random
import types
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np

from albumentations.augmentations.utils import is_grayscale_image
from albumentations.core.transforms_interface import ReferenceBasedTransform
from albumentations.core.types import BoxType, KeypointType, ReferenceImage, Targets
from albumentations.random_utils import beta

from .functional import mix_arrays

__all__ = ["MixUp"]


class MixUp(ReferenceBasedTransform):
    """Performs MixUp data augmentation, blending images, masks, and class labels with reference data.

    MixUp augmentation linearly combines an input (image, mask, and class label) with another set from a predefined
    reference dataset. The mixing degree is controlled by a parameter λ (lambda), sampled from a Beta distribution.
    This method is known for improving model generalization by promoting linear behavior between classes and
    smoothing decision boundaries.

    Reference:
        Zhang, H., Cisse, M., Dauphin, Y.N., and Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization.
        In International Conference on Learning Representations. https://arxiv.org/abs/1710.09412

    Args:
        reference_data (Optional[Union[Generator[ReferenceImage, None, None], Sequence[Any]]]):
            A sequence or generator of dictionaries containing the reference data for mixing
            If None or an empty sequence is provided, no operation is performed and a warning is issued.
        read_fn (Callable[[ReferenceImage], Dict[str, Any]]):
            A function to process items from reference_data. It should accept items from reference_data
            and return a dictionary containing processed data:
                - The returned dictionary must include an 'image' key with a numpy array value.
                - It may also include 'mask', 'global_label' each associated with numpy array values.
            Defaults to a function that assumes input dictionary contains numpy arrays and directly returns it.
         mix_coef_return_name (str): Name used for the applied alpha coefficient in the returned dictionary.
            Defaults to "mix_coef".
        alpha (float):
            The alpha parameter for the Beta distribution, influencing the mix's balance. Must be ≥ 0.
            Higher values lead to more uniform mixing. Defaults to 0.4.
        p (float):
            The probability of applying the transformation. Defaults to 0.5.

    Targets:
        image, mask, global_label

    Image types:
        - uint8, float32

    Raises:
        - ValueError: If the alpha parameter is negative.
        - NotImplementedError: If the transform is applied to bounding boxes or keypoints.

    Notes:
        - If no reference data is provided, a warning is issued, and the transform acts as a no-op.
        - Notes if images are in float32 format, they should be within [0, 1] range.
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.GLOBAL_LABEL)

    def __init__(
        self,
        reference_data: Optional[Union[Generator[ReferenceImage, None, None], Sequence[Any]]] = None,
        read_fn: Callable[[ReferenceImage], Any] = lambda x: {"image": x, "mask": None, "class_label": None},
        alpha: float = 0.4,
        mix_coef_return_name: str = "mix_coef",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.mix_coef_return_name = mix_coef_return_name

        if alpha < 0:
            msg = "Alpha must be >= 0."
            raise ValueError(msg)

        self.read_fn = read_fn
        self.alpha = alpha

        if reference_data is None:
            warn("No reference data provided for MixUp. This transform will act as a no-op.")
            # Create an empty generator
            self.reference_data: List[Any] = []
        elif (
            isinstance(reference_data, types.GeneratorType)
            or isinstance(reference_data, Iterable)
            and not isinstance(reference_data, str)
        ):
            self.reference_data = reference_data  # type: ignore[assignment]
        else:
            msg = "reference_data must be a list, tuple, generator, or None."
            raise TypeError(msg)

    def apply(self, img: np.ndarray, mix_data: ReferenceImage, mix_coef: float, **params: Any) -> np.ndarray:
        mix_img = mix_data.get("image")

        if not is_grayscale_image(img) and img.shape != img.shape:
            msg = "The shape of the reference image should be the same as the input image."
            raise ValueError(msg)

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

    def apply_to_bboxes(self, bboxes: Sequence[BoxType], mix_data: ReferenceImage, **params: Any) -> Sequence[BoxType]:
        msg = "MixUp does not support bounding boxes yet, feel free to submit pull request to https://github.com/albumentations-team/albumentations/."
        raise NotImplementedError(msg)

    def apply_to_keypoints(
        self, keypoints: Sequence[KeypointType], *args: Any, **params: Any
    ) -> Sequence[KeypointType]:
        msg = "MixUp does not support keypoints yet, feel free to submit pull request to https://github.com/albumentations-team/albumentations/."
        raise NotImplementedError(msg)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "reference_data", "alpha"

    def get_params(self) -> Dict[str, Union[None, float, Dict[str, Any]]]:
        mix_data = None
        # Check if reference_data is not empty and is a sequence (list, tuple, np.array)
        if isinstance(self.reference_data, Sequence) and not isinstance(self.reference_data, (str, bytes)):
            if len(self.reference_data) > 0:  # Additional check to ensure it's not empty
                mix_idx = random.randint(0, len(self.reference_data) - 1)
                mix_data = self.reference_data[mix_idx]
        # Check if reference_data is an iterator or generator
        elif isinstance(self.reference_data, Iterator):
            try:
                mix_data = next(self.reference_data)  # Attempt to get the next item
            except StopIteration:
                warn(
                    "Reference data iterator/generator has been exhausted. "
                    "Further mixing augmentations will not be applied.",
                    RuntimeWarning,
                )
                return {"mix_data": {}, "mix_coef": 1}

        # If mix_data is None or empty after the above checks, return default values
        if mix_data is None:
            return {"mix_data": {}, "mix_coef": 1}

        # If mix_data is not None, calculate mix_coef and apply read_fn
        mix_coef = beta(self.alpha, self.alpha)  # Assuming beta is defined elsewhere
        return {"mix_data": self.read_fn(mix_data), "mix_coef": mix_coef}

    def apply_with_params(self, params: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        res = super().apply_with_params(params, *args, **kwargs)
        if self.mix_coef_return_name:
            res[self.mix_coef_return_name] = params["mix_coef"]
        return res
