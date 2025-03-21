"""Implementation of coarse dropout and random erasing augmentations.

This module provides several variations of coarse dropout augmentations, which drop out
rectangular regions from images. It includes CoarseDropout for randomly placed dropouts,
ConstrainedCoarseDropout for dropping out regions based on masks or bounding boxes,
and Erasing for random erasing augmentation. These techniques help models become more
robust to occlusions and varying object completeness.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from warnings import warn

import numpy as np
from pydantic import AfterValidator

import albumentations.augmentations.dropout.functional as fdropout
from albumentations.augmentations.dropout.transforms import BaseDropout
from albumentations.core.bbox_utils import denormalize_bboxes
from albumentations.core.pydantic import check_range_bounds, nondecreasing

__all__ = ["CoarseDropout", "ConstrainedCoarseDropout", "Erasing"]


class CoarseDropout(BaseDropout):
    """CoarseDropout randomly drops out rectangular regions from the image and optionally,
    the corresponding regions in an associated mask, to simulate occlusion and
    varied object sizes found in real-world settings.

    This transformation is an evolution of CutOut and RandomErasing, offering more
    flexibility in the size, number of dropout regions, and fill values.

    Args:
        num_holes_range (tuple[int, int]): Range (min, max) for the number of rectangular
            regions to drop out. Default: (1, 1)
        hole_height_range (tuple[int, int] | tuple[float, float]): Range (min, max) for the height
            of dropout regions. If int, specifies absolute pixel values. If float,
            interpreted as a fraction of the image height. Default: (0.1, 0.2)
        hole_width_range (tuple[int, int] | tuple[float, float]): Range (min, max) for the width
            of dropout regions. If int, specifies absolute pixel values. If float,
            interpreted as a fraction of the image width. Default: (0.1, 0.2)
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
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

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
        num_holes_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]

        hole_height_range: Annotated[
            tuple[float, float] | tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, None)),
        ]

        hole_width_range: Annotated[
            tuple[float, float] | tuple[int, int],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, None)),
        ]

    def __init__(
        self,
        num_holes_range: tuple[int, int] = (1, 2),
        hole_height_range: tuple[float, float] | tuple[int, int] = (0.1, 0.2),
        hole_width_range: tuple[float, float] | tuple[int, int] = (0.1, 0.2),
        fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"] = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range

    def calculate_hole_dimensions(
        self,
        image_shape: tuple[int, int],
        height_range: tuple[float, float] | tuple[int, int],
        width_range: tuple[float, float] | tuple[int, int],
        size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate random hole dimensions based on the provided ranges."""
        height, width = image_shape[:2]

        if height_range[1] >= 1:
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
        """Get parameters dependent on the data.

        Args:
            params (dict[str, Any]): Dictionary containing parameters.
            data (dict[str, Any]): Dictionary containing data.

        Returns:
            dict[str, Any]: Dictionary with parameters for transformation.

        """
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
        fill (tuple[float, float] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]):
            Value used to fill the erased regions. Can be:
            - int or float: fills all channels with this value
            - tuple: fills each channel with corresponding value
            - "random": fills each pixel with random values
            - "random_uniform": fills entire erased region with a single random color
            - "inpaint_telea": uses OpenCV Telea inpainting method
            - "inpaint_ns": uses OpenCV Navier-Stokes inpainting method
            Default: 0
        fill_mask (tuple[float, float] | float | None): Value used to fill erased regions in the mask.
            If None, mask regions are not modified. Default: None
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

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
        scale: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, None)),
        ]
        ratio: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, None)),
        ]

    def __init__(
        self,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (0.3, 3.3),
        fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"] = 0,
        fill_mask: tuple[float, ...] | float | None = None,
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
        h = round(np.sqrt(erase_area / aspect_ratio))
        w = round(np.sqrt(erase_area * aspect_ratio))

        # Sample position
        top = self.py_random.randint(0, height - h)
        left = self.py_random.randint(0, width - w)

        holes = np.array([[left, top, left + w, top + h]], dtype=np.int32)
        return {"holes": holes, "seed": self.random_generator.integers(0, 2**32 - 1)}


class ConstrainedCoarseDropout(BaseDropout):
    """Applies coarse dropout to regions containing specific objects in the image.

    This augmentation creates holes (dropout regions) for each target object in the image.
    Objects can be specified either by their class indices in a segmentation mask or
    by their labels in bounding box annotations.

    The hole generation differs between mask and box modes:

    Mask mode:
    1. For each connected component in the mask matching target indices:
        - Samples N points randomly from within the object region (with replacement)
        - Creates holes centered at these points
        - Hole sizes are proportional to sqrt(component area), not total object area
        - Each component's holes are sized based on its own area

    Box mode:
    1. For each bounding box matching target labels:
        - Creates N holes with random positions inside the box
        - Hole sizes are proportional to the box dimensions

    In both modes:
    - N is sampled once from num_holes_range and used for all objects
    - For example, if num_holes_range=(2,4) and 3 is sampled:
        * With 3 target objects, you'll get exactly 3 holes per object (9 total)
        * Holes may overlap within or between objects
        * All holes are clipped to image boundaries

    Args:
        num_holes_range (tuple[int, int]): Range for number of holes per object (min, max)
        hole_height_range (tuple[float, float]): Range for hole height as proportion
            of object height/size (min, max). E.g., (0.2, 0.4) means:
            - For boxes: 20-40% of box height
            - For masks: 20-40% of sqrt(component area)
        hole_width_range (tuple[float, float]): Range for hole width, similar to height
        fill (tuple[float, float] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"]):
            Value used to fill the erased regions. Can be:
            - int or float: fills all channels with this value
            - tuple: fills each channel with corresponding value
            - "random": fills each pixel with random values
            - "random_uniform": fills entire erased region with a single random color
            - "inpaint_telea": uses OpenCV Telea inpainting method
            - "inpaint_ns": uses OpenCV Navier-Stokes inpainting method
            Default: 0
        fill_mask (tuple[float, float] | float | None): Value used to fill erased regions in the mask.
            If None, mask regions are not modified. Default: None
        p (float): Probability of applying the transform
        mask_indices (List[int], optional): List of class indices in segmentation mask to target.
            Only objects of these classes will be considered for hole placement.
        bbox_labels (List[str | int | float], optional): List of object labels in bbox
            annotations to target. String labels will be automatically encoded.
            When multiple label fields are specified in BboxParams, only the first
            label field is used for filtering.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Requires one of:
        - 'mask' key with segmentation mask where:
            * 0 represents background
            * Non-zero values represent different object instances/classes
            * Values must correspond to mask_indices
        - 'bboxes' key with bounding boxes in format [x_min, y_min, x_max, y_max, label, ...]

    Note:
        At least one of mask_indices or bbox_labels must be provided.
        If both are provided, mask_indices takes precedence.

    Examples:
        >>> # Using segmentation mask
        >>> transform = ConstrainedCoarseDropout(
        ...     num_holes_range=(2, 4),        # 2-4 holes per object
        ...     hole_height_range=(0.2, 0.4),  # 20-40% of sqrt(object area)
        ...     hole_width_range=(0.2, 0.4),   # 20-40% of sqrt(object area)
        ...     mask_indices=[1, 2],           # Target objects of class 1 and 2
        ...     fill=0,                        # Fill holes with black
        ... )
        >>> # Apply to image and its segmentation mask
        >>> transformed = transform(image=image, mask=mask)

        >>> # Using bounding boxes with Compose
        >>> transform = A.Compose([
        ...     ConstrainedCoarseDropout(
        ...         num_holes_range=(1, 3),
        ...         hole_height_range=(0.3, 0.5),  # 30-50% of box height
        ...         hole_width_range=(0.3, 0.5),   # 30-50% of box width
        ...         bbox_labels=['person'],        # Target people
        ...         fill=127,                      # Fill holes with gray
        ...     )
        ... ], bbox_params=A.BboxParams(
        ...     format='pascal_voc',  # [x_min, y_min, x_max, y_max]
        ...     label_fields=['labels']  # Specify field containing labels
        ... ))
        >>> # Apply to image and its bounding boxes
        >>> transformed = transform(
        ...     image=image,
        ...     bboxes=[[0, 0, 100, 100, 'car'], [150, 150, 300, 300, 'person']],
        ...     labels=['car', 'person']
        ... )

    """

    class InitSchema(BaseDropout.InitSchema):
        num_holes_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]

        hole_height_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]

        hole_width_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0.0, 1.0)),
        ]

        mask_indices: Annotated[
            list[int] | None,
            AfterValidator(check_range_bounds(1, None)),
        ]

        bbox_labels: list[str | int | float] | None = None

    def __init__(
        self,
        num_holes_range: tuple[int, int] = (1, 1),
        hole_height_range: tuple[float, float] = (0.1, 0.1),
        hole_width_range: tuple[float, float] = (0.1, 0.1),
        fill: tuple[float, ...] | float | Literal["random", "random_uniform", "inpaint_telea", "inpaint_ns"] = 0,
        fill_mask: tuple[float, ...] | float | None = None,
        p: float = 0.5,
        mask_indices: list[int] | None = None,
        bbox_labels: list[str | int | float] | None = None,
    ):
        super().__init__(fill=fill, fill_mask=fill_mask, p=p)
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range
        self.mask_indices = mask_indices
        self.bbox_labels = bbox_labels

    def get_boxes_from_bboxes(self, bboxes: np.ndarray) -> np.ndarray | None:
        """Get bounding boxes that match specified labels.

        Uses BboxProcessor's label encoder if bbox_labels contain strings.
        """
        if len(bboxes) == 0 or self.bbox_labels is None:
            return None

        # Get label encoder from BboxProcessor if needed
        bbox_processor = self.get_processor("bboxes")
        if bbox_processor is None:
            return None

        if not all(isinstance(label, (int, float)) for label in self.bbox_labels):
            label_fields = bbox_processor.params.label_fields
            if label_fields is None:
                raise ValueError("BboxParams.label_fields must be specified when using string labels")

            first_class_label = label_fields[0]
            # Access encoder through label_manager's metadata
            metadata = bbox_processor.label_manager.metadata["bboxes"][first_class_label]
            if metadata.encoder is None:
                raise ValueError(f"No encoder found for label field {first_class_label}")

            target_labels = metadata.encoder.transform(self.bbox_labels)
        else:
            target_labels = np.array(self.bbox_labels)

        # Filter boxes by labels (usually in column 4)
        mask = np.isin(bboxes[:, 4], target_labels)
        filtered_boxes = bboxes[mask, :4]

        return filtered_boxes if len(filtered_boxes) > 0 else None

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Get hole parameters based on either mask indices or bbox labels."""
        num_holes_per_obj = self.py_random.randint(*self.num_holes_range)

        if self.mask_indices is not None and "mask" in data:
            holes = fdropout.get_holes_from_mask(
                data["mask"],
                num_holes_per_obj,
                self.mask_indices,
                self.hole_height_range,
                self.hole_width_range,
                self.random_generator,
            )
        elif self.bbox_labels is not None and "bboxes" in data:
            target_boxes = self.get_boxes_from_bboxes(data["bboxes"])
            if target_boxes is None:
                holes = np.array([], dtype=np.int32).reshape((0, 4))
            else:
                target_boxes = denormalize_bboxes(target_boxes, data["image"].shape[:2])
                holes = fdropout.get_holes_from_boxes(
                    target_boxes,
                    num_holes_per_obj,
                    self.hole_height_range,
                    self.hole_width_range,
                    self.random_generator,
                )
        else:
            warn("Neither valid mask nor bboxes provided, do not apply Constrained Coarse Dropout", stacklevel=2)
            holes = np.array([], dtype=np.int32).reshape((0, 4))

        return {
            "holes": holes,
            "seed": self.random_generator.integers(0, 2**32 - 1),
        }
