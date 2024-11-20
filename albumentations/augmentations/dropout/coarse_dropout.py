from __future__ import annotations

from typing import Annotated, Any
from warnings import warn

import numpy as np
from pydantic import AfterValidator, Field, model_validator
from typing_extensions import Self

from albumentations.augmentations.dropout.transforms import BaseDropout
from albumentations.core.pydantic import check_0plus, check_1plus, nondecreasing
from albumentations.core.types import ColorType, DropoutFillValue, Number, ScalarType

__all__ = ["CoarseDropout", "Erasing"]


class CoarseDropout(BaseDropout):
    """CoarseDropout randomly drops out rectangular regions from the image and optionally,
    the corresponding regions in an associated mask, to simulate occlusion and
    varied object sizes found in real-world settings.

    This transformation is an evolution of CutOut and RandomErasing, offering more
    flexibility in the size, number of dropout regions, and fill values.

    Args:
        num_holes_range (tuple[int, int]): Range (min, max) for the number of rectangular
            regions to drop out. Default: (1, 1)
        hole_height_range (tuple[Real, Real]): Range (min, max) for the height
            of dropout regions. If int, specifies absolute pixel values. If float,
            interpreted as a fraction of the image height. Default: (8, 8)
        hole_width_range (tuple[Real, Real]): Range (min, max) for the width
            of dropout regions. If int, specifies absolute pixel values. If float,
            interpreted as a fraction of the image width. Default: (8, 8)
        fill (ColorType | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]):
            Value for the dropped pixels. Can be:
            - int or float: all channels are filled with this value
            - tuple: tuple of values for each channel
            - 'random': each pixel is filled with random values
            - 'random_uniform': each hole is filled with a single random color
            - 'inpaint_telea': uses OpenCV Telea inpainting method
            - 'inpaint_ns': uses OpenCV Navier-Stokes inpainting method
            Default: 0
        mask_fill_value (ColorType | None): Fill value for dropout regions in the mask.
            If None, mask regions corresponding to image dropouts are unchanged. Default: None
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The actual number and size of dropout regions are randomly chosen within the specified ranges for each
            application.
        - When using float values for hole_height_range and hole_width_range, ensure they are between 0 and 1.
        - This implementation includes deprecation warnings for older parameter names (min_holes, max_holes, etc.).
        - Inpainting methods ('inpaint_telea', 'inpaint_ns') work only with grayscale or RGB images.
        - For 'random_uniform' fill, each hole gets a single random color, unlike 'random' where each pixel
            gets its own random value.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> # Example with random uniform fill
        >>> aug_random = A.CoarseDropout(
        ...     num_holes_range=(3, 6),
        ...     hole_height_range=(10, 20),
        ...     hole_width_range=(10, 20),
        ...     fill="random_uniform",
        ...     p=1.0
        ... )
        >>> # Example with inpainting
        >>> aug_inpaint = A.CoarseDropout(
        ...     num_holes_range=(3, 6),
        ...     hole_height_range=(10, 20),
        ...     hole_width_range=(10, 20),
        ...     fill="inpaint_ns",
        ...     p=1.0
        ... )
        >>> transformed = aug_random(image=image, mask=mask)
        >>> transformed_image, transformed_mask = transformed["image"], transformed["mask"]

    References:
        - CutOut: https://arxiv.org/abs/1708.04552
        - Random Erasing: https://arxiv.org/abs/1708.04896
        - OpenCV Inpainting methods: https://docs.opencv.org/master/df/d3d/tutorial_py_inpainting.html
    """

    class InitSchema(BaseDropout.InitSchema):
        min_holes: int | None = Field(ge=0)
        max_holes: int | None = Field(ge=0)
        num_holes_range: Annotated[tuple[int, int], AfterValidator(check_1plus), AfterValidator(nondecreasing)]

        min_height: ScalarType | None = Field(ge=0)
        max_height: ScalarType | None = Field(ge=0)
        hole_height_range: tuple[ScalarType, ScalarType]

        min_width: ScalarType | None = Field(ge=0)
        max_width: ScalarType | None = Field(ge=0)
        hole_width_range: tuple[ScalarType, ScalarType]

        @staticmethod
        def update_range(
            min_value: Number | None,
            max_value: Number | None,
            default_range: tuple[Number, Number],
        ) -> tuple[Number, Number]:
            return (min_value or max_value, max_value) if max_value is not None else default_range

        @staticmethod
        def validate_range(range_value: tuple[float, float], range_name: str, minimum: float = 0) -> None:
            if not minimum <= range_value[0] <= range_value[1]:
                raise ValueError(
                    f"First value in {range_name} should be less or equal than the second value "
                    f"and at least {minimum}. Got: {range_value}",
                )
            if isinstance(range_value[0], float) and not all(0 <= x <= 1 for x in range_value):
                raise ValueError(f"All values in {range_name} should be in [0, 1] range. Got: {range_value}")

        @model_validator(mode="after")
        def check_num_holes_and_dimensions(self) -> Self:
            if self.min_holes is not None:
                warn("`min_holes` is deprecated. Use num_holes_range instead.", DeprecationWarning, stacklevel=2)
            if self.max_holes is not None:
                warn("`max_holes` is deprecated. Use num_holes_range instead.", DeprecationWarning, stacklevel=2)
            if self.min_height is not None:
                warn("`min_height` is deprecated. Use hole_height_range instead.", DeprecationWarning, stacklevel=2)
            if self.max_height is not None:
                warn("`max_height` is deprecated. Use hole_height_range instead.", DeprecationWarning, stacklevel=2)
            if self.min_width is not None:
                warn("`min_width` is deprecated. Use hole_width_range instead.", DeprecationWarning, stacklevel=2)
            if self.max_width is not None:
                warn("`max_width` is deprecated. Use hole_width_range instead.", DeprecationWarning, stacklevel=2)

            if self.max_holes is not None:
                self.num_holes_range = self.update_range(self.min_holes, self.max_holes, self.num_holes_range)

            self.validate_range(self.num_holes_range, "num_holes_range", minimum=1)

            if self.max_height is not None:
                self.hole_height_range = self.update_range(self.min_height, self.max_height, self.hole_height_range)
            self.validate_range(self.hole_height_range, "hole_height_range")

            if self.max_width is not None:
                self.hole_width_range = self.update_range(self.min_width, self.max_width, self.hole_width_range)
            self.validate_range(self.hole_width_range, "hole_width_range")

            return self

    def __init__(
        self,
        max_holes: int | None = None,
        max_height: ScalarType | None = None,
        max_width: ScalarType | None = None,
        min_holes: int | None = None,
        min_height: ScalarType | None = None,
        min_width: ScalarType | None = None,
        fill_value: DropoutFillValue | None = None,
        mask_fill_value: ColorType | None = None,
        num_holes_range: tuple[int, int] = (1, 1),
        hole_height_range: tuple[ScalarType, ScalarType] = (8, 8),
        hole_width_range: tuple[ScalarType, ScalarType] = (8, 8),
        fill: DropoutFillValue = 0,
        fill_mask: ColorType | None = None,
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range

    def calculate_hole_dimensions(
        self,
        image_shape: tuple[int, int],
        height_range: tuple[float, float],
        width_range: tuple[float, float],
        size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate random hole dimensions based on the provided ranges."""
        height, width = image_shape[:2]

        if isinstance(height_range[0], int):
            min_height = height_range[0]
            max_height = min(height_range[1], height)

            min_width = width_range[0]
            max_width = min(width_range[1], width)

            hole_heights = self.random_generator.integers(int(min_height), int(max_height + 1), size=size)
            hole_widths = self.random_generator.integers(int(min_width), int(max_width + 1), size=size)

        else:  # Assume float
            hole_heights = (height * self.random_generator.uniform(*height_range, size=size)).astype(int)
            hole_widths = (width * self.random_generator.uniform(*width_range, size=size)).astype(int)

        return hole_heights, hole_widths

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_shape = params["shape"][:2]

        num_holes = self.py_random.randint(*self.num_holes_range)

        hole_heights, hole_widths = self.calculate_hole_dimensions(
            image_shape,
            self.hole_height_range,
            self.hole_width_range,
            size=num_holes,
        )

        height, width = image_shape[:2]

        y_min = self.random_generator.integers(0, height - hole_heights + 1, size=num_holes)
        x_min = self.random_generator.integers(0, width - hole_widths + 1, size=num_holes)
        y_max = y_min + hole_heights
        x_max = x_min + hole_widths

        holes = np.stack([x_min, y_min, x_max, y_max], axis=-1)

        return {"holes": holes, "seed": self.random_generator.integers(0, 2**32 - 1)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (*super().get_transform_init_args_names(), "num_holes_range", "hole_height_range", "hole_width_range")


class Erasing(BaseDropout):
    """Randomly erases rectangular regions in an image, following the Random Erasing Data Augmentation technique.

    This augmentation helps improve model robustness by randomly masking out rectangular regions in the image,
    simulating occlusions and encouraging the model to learn from partial information. It's particularly
    effective for image classification and person re-identification tasks.

    Args:
        scale (tuple[float, float]): Range for the proportion of image area to erase.
            The actual area will be randomly sampled from (scale[0] * image_area, scale[1] * image_area).
            Default: (0.02, 0.33)
        ratio (tuple[float, float]): Range for the aspect ratio (width/height) of the erased region.
            The actual ratio will be randomly sampled from (ratio[0], ratio[1]).
            Default: (0.3, 3.3)
        fill (ColorType | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]):
            Value used to fill the erased regions. Can be:
            - int or float: fills all channels with this value
            - tuple: fills each channel with corresponding value
            - "random": fills each pixel with random values
            - "random_uniform": fills entire erased region with a single random color
            - "inpaint_telea": uses OpenCV Telea inpainting method
            - "inpaint_ns": uses OpenCV Navier-Stokes inpainting method
            Default: 0
        mask_fill (ColorType | None): Value used to fill erased regions in the mask.
            If None, mask regions are not modified. Default: None
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - The transform attempts to find valid erasing parameters up to 10 times.
          If unsuccessful, no erasing is performed.
        - The actual erased area and aspect ratio are randomly sampled within
          the specified ranges for each application.
        - When using inpainting methods, only grayscale or RGB images are supported.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> # Basic usage with default parameters
        >>> transform = A.Erasing()
        >>> transformed = transform(image=image)
        >>> # Custom configuration
        >>> transform = A.Erasing(
        ...     scale=(0.1, 0.4),
        ...     ratio=(0.5, 2.0),
        ...     fill_value="random_uniform",
        ...     p=1.0
        ... )
        >>> transformed = transform(image=image)

    References:
        - Paper: https://arxiv.org/abs/1708.04896
        - Implementation inspired by torchvision:
          https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomErasing
    """

    class InitSchema(BaseDropout.InitSchema):
        scale: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_0plus)]
        ratio: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_0plus)]

    def __init__(
        self,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (0.3, 3.3),
        fill: DropoutFillValue = 0,
        fill_mask: ColorType | None = None,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)

        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Calculate erasing parameters using direct mathematical derivation.

        Given:
        - Image dimensions (H, W)
        - Target area (A)
        - Aspect ratio (r = w/h)

        We know:
        - h * w = A (area equation)
        - w = r * h (aspect ratio equation)

        Therefore:
        - h * (r * h) = A
        - h² = A/r
        - h = sqrt(A/r)
        - w = r * sqrt(A/r) = sqrt(A*r)
        """
        height, width = params["shape"][:2]
        total_area = height * width

        # Calculate maximum valid area based on dimensions and aspect ratio
        max_area = total_area * self.scale[1]
        min_area = total_area * self.scale[0]

        # For each aspect ratio r, the maximum area is constrained by:
        # h = sqrt(A/r) ≤ H and w = sqrt(A*r) ≤ W
        # Therefore: A ≤ min(r*H², W²/r)
        r_min, r_max = self.ratio

        def area_constraint_h(r: float) -> float:
            return r * height * height

        def area_constraint_w(r: float) -> float:
            return width * width / r

        # Find maximum valid area considering aspect ratio constraints
        max_area_h = min(area_constraint_h(r_min), area_constraint_h(r_max))
        max_area_w = min(area_constraint_w(r_min), area_constraint_w(r_max))
        max_valid_area = min(max_area, max_area_h, max_area_w)

        if max_valid_area < min_area:
            return {"holes": np.array([], dtype=np.int32).reshape((0, 4))}

        # Sample valid area and aspect ratio
        erase_area = self.py_random.uniform(min_area, max_valid_area)

        # Calculate valid aspect ratio range for this area
        max_r = min(r_max, width * width / erase_area)
        min_r = max(r_min, erase_area / (height * height))

        if min_r > max_r:
            return {"holes": np.array([], dtype=np.int32).reshape((0, 4))}

        aspect_ratio = self.py_random.uniform(min_r, max_r)

        # Calculate dimensions
        h = int(round(np.sqrt(erase_area / aspect_ratio)))
        w = int(round(np.sqrt(erase_area * aspect_ratio)))

        # Sample position
        top = self.py_random.randint(0, height - h)
        left = self.py_random.randint(0, width - w)

        holes = np.array([[left, top, left + w, top + h]], dtype=np.int32)
        return {"holes": holes, "seed": self.random_generator.integers(0, 2**32 - 1)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "scale", "ratio", "fill", "fill_mask"
