"""Implementation of grid-based dropout augmentation.

This module provides GridDropout, which creates a regular grid over the image and drops out
rectangular regions according to the specified grid pattern. Unlike random dropout methods,
grid dropout enforces a structured pattern of occlusions that can help models learn spatial
relationships and context across the entire image space.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field

import albumentations.augmentations.dropout.functional as fdropout
from albumentations.augmentations.dropout.transforms import BaseDropout
from albumentations.core.pydantic import check_range_bounds, nondecreasing

__all__ = ["GridDropout"]


class GridDropout(BaseDropout):
    """Apply GridDropout augmentation to images, masks, bounding boxes, and keypoints.

    GridDropout drops out rectangular regions of an image and the corresponding mask in a grid fashion.
    This technique can help improve model robustness by forcing the network to rely on a broader context
    rather than specific local features.

    Args:
        ratio (float): The ratio of the mask holes to the unit size (same for horizontal and vertical directions).
            Must be between 0 and 1. Default: 0.5.
        unit_size_range (tuple[int, int] | None): Range from which to sample grid size. Default: None.
            Must be between 2 and the image's shorter edge. If None, grid size is calculated based on image size.
        holes_number_xy (tuple[int, int] | None): The number of grid units in x and y directions.
            First value should be between 1 and image width//2,
            Second value should be between 1 and image height//2.
            Default: None. If provided, overrides unit_size_range.
        random_offset (bool): Whether to offset the grid randomly between 0 and (grid unit size - hole size).
            If True, entered shift_xy is ignored and set randomly. Default: True.
        fill (tuple[float, float] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]):
            Value for the dropped pixels. Can be:
            - int or float: all channels are filled with this value
            - tuple: tuple of values for each channel
            - 'random': each pixel is filled with random values
            - 'random_uniform': each hole is filled with a single random color
            - 'inpaint_telea': uses OpenCV Telea inpainting method
            - 'inpaint_ns': uses OpenCV Navier-Stokes inpainting method
            Default: 0
        fill_mask (tuple[float, float] | float | None): Value for the dropped pixels in mask.
            If None, the mask is not modified. Default: None.
        shift_xy (tuple[int, int]): Offsets of the grid start in x and y directions from (0,0) coordinate.
            Only used when random_offset is False. Default: (0, 0).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Note:
        - If both unit_size_range and holes_number_xy are None, the grid size is calculated based on the image size.
        - The actual number of dropped regions may differ slightly from holes_number_xy due to rounding.
        - Inpainting methods ('inpaint_telea', 'inpaint_ns') work only with grayscale or RGB images.
        - For 'random_uniform' fill, each grid cell gets a single random color, unlike 'random' where each pixel
            gets its own random value.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> # Example with standard fill value
        >>> aug_basic = A.GridDropout(
        ...     ratio=0.3,
        ...     unit_size_range=(10, 20),
        ...     random_offset=True,
        ...     p=1.0
        ... )
        >>> # Example with random uniform fill
        >>> aug_random = A.GridDropout(
        ...     ratio=0.3,
        ...     unit_size_range=(10, 20),
        ...     fill="random_uniform",
        ...     p=1.0
        ... )
        >>> # Example with inpainting
        >>> aug_inpaint = A.GridDropout(
        ...     ratio=0.3,
        ...     unit_size_range=(10, 20),
        ...     fill="inpaint_ns",
        ...     p=1.0
        ... )
        >>> transformed = aug_random(image=image, mask=mask)
        >>> transformed_image, transformed_mask = transformed["image"], transformed["mask"]

    Reference:
        - Paper: https://arxiv.org/abs/2001.04086
        - OpenCV Inpainting methods: https://docs.opencv.org/master/df/d3d/tutorial_py_inpainting.html

    """

    class InitSchema(BaseDropout.InitSchema):
        ratio: float = Field(gt=0, le=1)

        random_offset: bool

        unit_size_range: (
            Annotated[tuple[int, int], AfterValidator(check_range_bounds(2, None)), AfterValidator(nondecreasing)]
            | None
        )
        shift_xy: Annotated[tuple[int, int], AfterValidator(check_range_bounds(0, None))]

        holes_number_xy: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))] | None

    def __init__(
        self,
        ratio: float = 0.5,
        random_offset: bool = True,
        unit_size_range: tuple[int, int] | None = None,
        holes_number_xy: tuple[int, int] | None = None,
        shift_xy: tuple[int, int] = (0, 0),
        fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"] = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.ratio = ratio
        self.unit_size_range = unit_size_range
        self.holes_number_xy = holes_number_xy
        self.random_offset = random_offset
        self.shift_xy = shift_xy

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Get parameters dependent on the data.

        Args:
            params (dict[str, Any]): Dictionary containing parameters.
            data (dict[str, Any]): Dictionary containing data.

        Returns:
            dict[str, Any]: Dictionary with parameters for transformation.

        """
        image_shape = params["shape"]
        if self.holes_number_xy:
            grid = self.holes_number_xy
        else:
            # Calculate grid based on unit_size_range or default
            unit_height, unit_width = fdropout.calculate_grid_dimensions(
                image_shape,
                self.unit_size_range,
                self.holes_number_xy,
                self.random_generator,
            )
            grid = (image_shape[0] // unit_height, image_shape[1] // unit_width)

        holes = fdropout.generate_grid_holes(
            image_shape,
            grid,
            self.ratio,
            self.random_offset,
            self.shift_xy,
            self.random_generator,
        )
        return {"holes": holes, "seed": self.random_generator.integers(0, 2**32 - 1)}
