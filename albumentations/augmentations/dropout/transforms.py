from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from albumentations.augmentations.dropout.functional import cutout, filter_bboxes_by_holes, filter_keypoints_in_holes
from albumentations.core.bbox_utils import BboxProcessor, denormalize_bboxes, normalize_bboxes
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import ColorType, Targets


class BaseDropout(DualTransform):
    """Base class for dropout-style transformations.

    This class provides common functionality for various dropout techniques,
    including applying cutouts to images and masks.

    Args:
        fill_value (Union[int, float, list[int], list[float], str]): Value to fill dropped regions.
            If "random", fills with random values.
        mask_fill_value (Union[int, float, list[int], list[float]] | None): Value to fill
            dropped regions in the mask. If None, the mask is not modified.
        p (float): Probability of applying the transform.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        fill_value: ColorType | Literal["random"]
        mask_fill_value: ColorType | None

    def __init__(
        self,
        fill_value: ColorType | Literal["random"],
        mask_fill_value: ColorType | None,
        p: float,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value

    def apply(self, img: np.ndarray, holes: np.ndarray, seed: int, **params: Any) -> np.ndarray:
        return cutout(img, holes, self.fill_value, np.random.default_rng(seed))

    def apply_to_mask(self, mask: np.ndarray, holes: np.ndarray, seed: int, **params: Any) -> np.ndarray:
        if self.mask_fill_value is None:
            return mask
        return cutout(mask, holes, self.mask_fill_value, np.random.default_rng(seed))

    def apply_to_bboxes(
        self,
        bboxes: np.ndarray,
        holes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        processor = cast(BboxProcessor, self.get_processor("bboxes"))
        if processor is None:
            return bboxes

        image_shape = params["shape"][:2]

        denormalized_bboxes = denormalize_bboxes(bboxes, image_shape)

        return normalize_bboxes(
            filter_bboxes_by_holes(
                denormalized_bboxes,
                holes,
                image_shape,
                min_area=processor.params.min_area,
                min_visibility=processor.params.min_visibility,
            ),
            image_shape,
        )

    def apply_to_keypoints(
        self,
        keypoints: np.ndarray,
        holes: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        processor = cast(KeypointsProcessor, self.get_processor("keypoints"))

        if processor is None or not processor.params.remove_invisible:
            return keypoints

        return filter_keypoints_in_holes(keypoints, holes)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Get parameters of the transform conditioned on the input image.

        Args:
            params (dict): Parameters given to the transform.
            data (dict): Additional data given to the transform.

        Returns:
            dict: Parameters required to apply the transform.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Get the names of the arguments used in __init__.

        Returns:
            tuple: Names of the arguments.
        """
        return ("fill_value", "mask_fill_value")
