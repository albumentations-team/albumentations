"""Implementation of XY masking for time-frequency domain transformations.

This module provides the XYMasking transform, which applies masking strips along the X and Y axes
of an image. This is particularly useful for audio spectrograms, time-series data visualizations,
and other grid-like data representations where masking in specific directions (time or frequency)
can improve model robustness and generalization.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from pydantic import model_validator
from typing_extensions import Self

from albumentations.augmentations.dropout.transforms import BaseDropout
from albumentations.core.pydantic import NonNegativeIntRangeType
from albumentations.core.transforms_interface import BaseTransformInitSchema

__all__ = ["XYMasking"]


class XYMasking(BaseDropout):
    """Applies masking strips to an image, either horizontally (X axis) or vertically (Y axis),
    simulating occlusions. This transform is useful for training models to recognize images
    with varied visibility conditions. It's particularly effective for spectrogram images,
    allowing spectral and frequency masking to improve model robustness.

    At least one of `max_x_length` or `max_y_length` must be specified, dictating the mask's
    maximum size along each axis.

    Args:
        num_masks_x (int | tuple[int, int]): Number or range of horizontal regions to mask. Defaults to 0.
        num_masks_y (int | tuple[int, int]): Number or range of vertical regions to mask. Defaults to 0.
        mask_x_length (int | tuple[int, int]): Specifies the length of the masks along
            the X (horizontal) axis. If an integer is provided, it sets a fixed mask length.
            If a tuple of two integers (min, max) is provided,
            the mask length is randomly chosen within this range for each mask.
            This allows for variable-length masks in the horizontal direction.
        mask_y_length (int | tuple[int, int]): Specifies the height of the masks along
            the Y (vertical) axis. Similar to `mask_x_length`, an integer sets a fixed mask height,
            while a tuple (min, max) allows for variable-height masks, chosen randomly
            within the specified range for each mask. This flexibility facilitates creating masks of various
            sizes in the vertical direction.
        fill (tuple[float, float] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]):
            Value for the dropped pixels. Can be:
            - int or float: all channels are filled with this value
            - tuple: tuple of values for each channel
            - 'random': each pixel is filled with random values
            - 'random_uniform': each hole is filled with a single random color
            - 'inpaint_telea': uses OpenCV Telea inpainting method
            - 'inpaint_ns': uses OpenCV Navier-Stokes inpainting method
            Default: 0
        fill_mask (tuple[float, float] | float | None): Fill value for dropout regions in the mask.
            If None, mask regions corresponding to image dropouts are unchanged. Default: None
        p (float): Probability of applying the transform. Defaults to 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note: Either `max_x_length` or `max_y_length` or both must be defined.

    """

    class InitSchema(BaseTransformInitSchema):
        num_masks_x: NonNegativeIntRangeType
        num_masks_y: NonNegativeIntRangeType
        mask_x_length: NonNegativeIntRangeType
        mask_y_length: NonNegativeIntRangeType

        fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]
        fill_mask: tuple[float, ...] | float | None

        @model_validator(mode="after")
        def _check_mask_length(self) -> Self:
            if (
                isinstance(self.mask_x_length, int)
                and self.mask_x_length <= 0
                and isinstance(self.mask_y_length, int)
                and self.mask_y_length <= 0
            ):
                msg = "At least one of `mask_x_length` or `mask_y_length` Should be a positive number."
                raise ValueError(msg)

            return self

    def __init__(
        self,
        num_masks_x: tuple[int, int] | int = 0,
        num_masks_y: tuple[int, int] | int = 0,
        mask_x_length: tuple[int, int] | int = 0,
        mask_y_length: tuple[int, int] | int = 0,
        fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"] = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, fill=fill, fill_mask=fill_mask)
        self.num_masks_x = cast("tuple[int, int]", num_masks_x)
        self.num_masks_y = cast("tuple[int, int]", num_masks_y)

        self.mask_x_length = cast("tuple[int, int]", mask_x_length)
        self.mask_y_length = cast("tuple[int, int]", mask_y_length)

    def _validate_mask_length(
        self,
        mask_length: tuple[int, int] | None,
        dimension_size: int,
        dimension_name: str,
    ) -> None:
        """Validate the mask length against the corresponding image dimension size."""
        if mask_length is not None:
            if isinstance(mask_length, (tuple, list)):
                if mask_length[0] < 0 or mask_length[1] > dimension_size:
                    raise ValueError(
                        f"{dimension_name} range {mask_length} is out of valid range [0, {dimension_size}]",
                    )
            elif mask_length < 0 or mask_length > dimension_size:
                raise ValueError(f"{dimension_name} {mask_length} exceeds image {dimension_name} {dimension_size}")

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Get parameters dependent on the data.

        Args:
            params (dict[str, Any]): Dictionary containing parameters.
            data (dict[str, Any]): Dictionary containing data.

        Returns:
            dict[str, np.ndarray]: Dictionary with parameters for transformation.

        """
        image_shape = params["shape"][:2]

        height, width = image_shape

        self._validate_mask_length(self.mask_x_length, width, "mask_x_length")
        self._validate_mask_length(self.mask_y_length, height, "mask_y_length")

        masks_x = self._generate_masks(self.num_masks_x, image_shape, self.mask_x_length, axis="x")
        masks_y = self._generate_masks(self.num_masks_y, image_shape, self.mask_y_length, axis="y")

        holes = np.array(masks_x + masks_y)

        return {"holes": holes, "seed": self.random_generator.integers(0, 2**32 - 1)}

    def _generate_mask_size(self, mask_length: tuple[int, int]) -> int:
        return self.py_random.randint(*mask_length)

    def _generate_masks(
        self,
        num_masks: tuple[int, int],
        image_shape: tuple[int, int],
        max_length: tuple[int, int] | None,
        axis: str,
    ) -> list[tuple[int, int, int, int]]:
        if max_length is None or max_length == 0 or (isinstance(num_masks, (int, float)) and num_masks == 0):
            return []

        masks = []
        num_masks_integer = (
            num_masks if isinstance(num_masks, int) else self.py_random.randint(num_masks[0], num_masks[1])
        )

        height, width = image_shape

        for _ in range(num_masks_integer):
            length = self._generate_mask_size(max_length)

            if axis == "x":
                x_min = self.py_random.randint(0, width - length)
                y_min = 0
                x_max, y_max = x_min + length, height
            else:  # axis == 'y'
                y_min = self.py_random.randint(0, height - length)
                x_min = 0
                x_max, y_max = width, y_min + length

            masks.append((x_min, y_min, x_max, y_max))
        return masks
