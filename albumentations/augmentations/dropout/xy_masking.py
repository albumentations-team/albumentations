import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from albumentations.core.types import ColorType, KeypointType, ScaleIntType

from ...core.transforms_interface import DualTransform
from .functional import cutout, keypoint_in_hole

__all__ = ["XYMasking"]


class XYMasking(DualTransform):
    """
    Applies masking strips to an image, either horizontally (X axis) or vertically (Y axis),
    simulating occlusions. This transform is useful for training models to recognize images
    with varied visibility conditions. It's particularly effective for spectrogram images,
    allowing spectral and frequency masking to improve model robustness.

    At least one of `max_x_length` or `max_y_length` must be specified, dictating the mask's
    maximum size along each axis.

    Args:
        min_masks_x (int): Minimum number of horizontal masks. Defaults to 0.
        max_masks_x (int): Maximum number of horizontal masks. Defaults to 0.
        min_masks_y (int): Minimum number of vertical masks. Defaults to 1.
        max_masks_y (int): Maximum number of vertical masks. Defaults to 1.
        mask_x_length (Optional[Union[int, Tuple[int, int]]]): Specifies the length of the masks along
            the X (horizontal) axis. If an integer is provided, it sets a fixed mask length.
            If a tuple of two integers (min, max) is provided,
            the mask length is randomly chosen within this range for each mask.
            This allows for variable-length masks in the horizontal direction.
        mask_y_length (Optional[Union[int, Tuple[int, int]]]): Specifies the height of the masks along
            the Y (vertical) axis. Similar to `mask_x_length`, an integer sets a fixed mask height,
            while a tuple (min, max) allows for variable-height masks, chosen randomly
            within the specified range for each mask. This flexibility facilitates creating masks of various
            sizes in the vertical direction.
        fill_value (Union[int, float, List[int], List[float]]): Value to fill image masks. Defaults to 0.
        mask_fill_value (Optional[Union[int, float, List[int], List[float]]]): Value to fill masks in the mask.
            If `None`, uses fill_value. Default: `None`.
        p (float): Probability of applying the transform. Defaults to 0.5.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32

    Note: Either `max_x_length` or `max_y_length` or both must be defined.
    """

    def __init__(
        self,
        min_masks_x: int = 0,
        max_masks_x: int = 1,
        min_masks_y: int = 0,
        max_masks_y: int = 1,
        mask_x_length: Optional[ScaleIntType] = None,
        mask_y_length: Optional[ScaleIntType] = None,
        fill_value: ColorType = 0,
        mask_fill_value: ColorType = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        if mask_x_length is None and mask_y_length is None:
            raise ValueError("At least one of `mask_x_length` or `mask_y_length` must be specified.")

        if not 0 <= min_masks_x <= max_masks_x:
            raise ValueError("`min_masks_x` must be >= 0 and <= `max_masks_x`.")
        if not 0 <= min_masks_y <= max_masks_y:
            raise ValueError("`min_masks_y` must be >= 0 and <= `max_masks_y`.")

        self.min_masks_x = min_masks_x
        self.max_masks_x = max_masks_x
        self.min_masks_y = min_masks_y
        self.max_masks_y = max_masks_y
        self.mask_x_length = mask_x_length
        self.mask_y_length = mask_y_length
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value if mask_fill_value is not None else fill_value

    def apply(
        self,
        img: np.ndarray,
        masks_x: List[Tuple[int, int, int, int]],
        masks_y: List[Tuple[int, int, int, int]],
        **params: Any,
    ) -> np.ndarray:
        return cutout(img, masks_x + masks_y, self.fill_value)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        masks_x: List[Tuple[int, int, int, int]],
        masks_y: List[Tuple[int, int, int, int]],
        **params: Any,
    ) -> np.ndarray:
        if self.mask_fill_value is None:
            return mask
        return cutout(mask, masks_x + masks_y, self.mask_fill_value)

    def validate_mask_length(
        self, mask_length: Optional[ScaleIntType], dimension_size: int, dimension_name: str
    ) -> None:
        """
        Validate the mask length against the corresponding image dimension size.

        Args:
            mask_length (Optional[Union[int, Tuple[int, int]]]): The length of the mask to be validated.
            dimension_size (int): The size of the image dimension (width or height)
                against which to validate the mask length.
            dimension_name (str): The name of the dimension ('width' or 'height') for error messaging.
        """
        if mask_length is not None:
            if isinstance(mask_length, tuple):
                if mask_length[0] < 0 or mask_length[1] > dimension_size:
                    raise ValueError(
                        f"{dimension_name} range {mask_length} is out of valid range [0, {dimension_size}]"
                    )
            elif mask_length < 0 or mask_length > dimension_size:
                raise ValueError(f"{dimension_name} {mask_length} exceeds image {dimension_name} {dimension_size}")

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, List[Tuple[int, int, int, int]]]:
        img = params["image"]
        height, width = img.shape[:2]

        # Use the helper method to validate mask lengths against image dimensions
        self.validate_mask_length(self.mask_x_length, width, "mask_x_length")
        self.validate_mask_length(self.mask_y_length, height, "mask_y_length")

        masks_x = self.generate_masks(self.min_masks_x, self.max_masks_x, width, height, self.mask_x_length, axis="x")
        masks_y = self.generate_masks(self.min_masks_y, self.max_masks_y, width, height, self.mask_y_length, axis="y")

        return {"masks_x": masks_x, "masks_y": masks_y}

    @staticmethod
    def generate_mask_size(mask_length: Union[ScaleIntType]) -> int:
        if isinstance(mask_length, int):
            return mask_length  # Use fixed size or adjust to dimension size

        return random.randint(min(mask_length), max(mask_length))

    def generate_masks(
        self,
        min_masks: int,
        max_masks: int,
        width: int,
        height: int,
        max_length: Optional[ScaleIntType],
        axis: str,
    ) -> List[Tuple[int, int, int, int]]:
        if max_length is None or max_length == 0 or max_masks == 0:
            return []

        masks = []
        num_masks = random.randint(min_masks, max_masks)
        for _ in range(num_masks):
            length = self.generate_mask_size(max_length)

            if axis == "x":
                x1 = random.randint(0, width - length)
                y1 = 0
                x2, y2 = x1 + length, height
            else:  # axis == 'y'
                y1 = random.randint(0, height - length)
                x1 = 0
                x2, y2 = width, y1 + length

            masks.append((x1, y1, x2, y2))
        return masks

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def apply_to_keypoints(
        self,
        keypoints: Sequence[KeypointType],
        masks_x: List[Tuple[int, int, int, int]],
        masks_y: List[Tuple[int, int, int, int]],
        **params: Any,
    ) -> List[KeypointType]:
        filtered_keypoints = []
        for keypoint in keypoints:
            if not any(keypoint_in_hole(keypoint, hole) for hole in masks_x + masks_y):
                filtered_keypoints.append(keypoint)
        return filtered_keypoints

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return (
            "min_masks_x",
            "max_masks_x",
            "min_masks_y",
            "max_masks_y",
            "mask_x_length",
            "mask_y_length",
            "fill_value",
            "mask_fill_value",
        )
