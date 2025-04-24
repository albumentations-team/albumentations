"""Utilities for handling bounding box operations during image augmentation.

This module provides tools for processing bounding boxes in various formats (COCO, Pascal VOC, YOLO),
converting between coordinate systems, normalizing and denormalizing coordinates, filtering
boxes based on visibility and size criteria, and performing transformations on boxes to match
image augmentations. It forms the core functionality for all bounding box-related operations
in the albumentations library.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.type_definitions import MONO_CHANNEL_DIMENSIONS, NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS

from .utils import DataProcessor, Params, ShapeType

__all__ = [
    "BboxParams",
    "BboxProcessor",
    "check_bboxes",
    "convert_bboxes_from_albumentations",
    "convert_bboxes_to_albumentations",
    "denormalize_bboxes",
    "filter_bboxes",
    "normalize_bboxes",
    "union_of_bboxes",
]

BBOX_WITH_LABEL_SHAPE = 5


class BboxParams(Params):
    """Parameters for bounding box transforms.

    Args:
        format (Literal["coco", "pascal_voc", "albumentations", "yolo"]): Format of bounding boxes.
            Should be one of:
            - 'coco': [x_min, y_min, width, height], e.g. [97, 12, 150, 200].
            - 'pascal_voc': [x_min, y_min, x_max, y_max], e.g. [97, 12, 247, 212].
            - 'albumentations': like pascal_voc but normalized in [0, 1] range, e.g. [0.2, 0.3, 0.4, 0.5].
            - 'yolo': [x_center, y_center, width, height] normalized in [0, 1] range, e.g. [0.1, 0.2, 0.3, 0.4].

        label_fields (Sequence[str] | None): List of fields that are joined with boxes,
            e.g., ['class_labels', 'scores']. Default: None.

        min_area (float): Minimum area of a bounding box. All bounding boxes whose visible area in pixels is less than
            this value will be removed. Default: 0.0.

        min_visibility (float): Minimum fraction of area for a bounding box to remain this box in the result.
            Should be in [0.0, 1.0] range. Default: 0.0.

        min_width (float): Minimum width of a bounding box in pixels or normalized units. Bounding boxes with width
            less than this value will be removed. Default: 0.0.

        min_height (float): Minimum height of a bounding box in pixels or normalized units. Bounding boxes with height
            less than this value will be removed. Default: 0.0.

        check_each_transform (bool): If True, performs checks for each dual transform. Default: True.

        clip (bool): If True, clips bounding boxes to image boundaries before applying any transform. Default: False.

        filter_invalid_bboxes (bool): If True, filters out invalid bounding boxes (e.g., boxes with negative dimensions
            or boxes where x_max < x_min or y_max < y_min) at the beginning of the pipeline. If clip=True, filtering
            is applied after clipping. Default: False.

        max_accept_ratio (float | None): Maximum allowed aspect ratio for bounding boxes. The aspect ratio is calculated
            as max(width/height, height/width), so it's always >= 1. Boxes with aspect ratio greater than this value
            will be filtered out. For example, if max_accept_ratio=3.0, boxes with width:height or height:width ratios
            greater than 3:1 will be removed. Set to None to disable aspect ratio filtering. Default: None.


    Note:
        The processing order for bounding boxes is:
        1. Convert to albumentations format (normalized pascal_voc)
        2. Clip boxes to image boundaries (if clip=True)
        3. Filter invalid boxes (if filter_invalid_bboxes=True)
        4. Apply transformations
        5. Filter boxes based on min_area, min_visibility, min_width, min_height
        6. Convert back to the original format

    Examples:
        >>> # Create BboxParams for COCO format with class labels
        >>> bbox_params = BboxParams(
        ...     format='coco',
        ...     label_fields=['class_labels'],
        ...     min_area=1024,
        ...     min_visibility=0.1
        ... )

        >>> # Create BboxParams that clips and filters invalid boxes
        >>> bbox_params = BboxParams(
        ...     format='pascal_voc',
        ...     clip=True,
        ...     filter_invalid_bboxes=True
        ... )
        >>> # Create BboxParams that filters extremely elongated boxes
        >>> bbox_params = BboxParams(
        ...     format='yolo',
        ...     max_accept_ratio=5.0,  # Filter boxes with aspect ratio > 5:1
        ...     clip=True
        ... )

    """

    def __init__(
        self,
        format: Literal["coco", "pascal_voc", "albumentations", "yolo"],  # noqa: A002
        label_fields: Sequence[Any] | None = None,
        min_area: float = 0.0,
        min_visibility: float = 0.0,
        min_width: float = 0.0,
        min_height: float = 0.0,
        check_each_transform: bool = True,
        clip: bool = False,
        filter_invalid_bboxes: bool = False,
        max_accept_ratio: float | None = None,
    ):
        super().__init__(format, label_fields)
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.min_width = min_width
        self.min_height = min_height
        self.check_each_transform = check_each_transform
        self.clip = clip
        self.filter_invalid_bboxes = filter_invalid_bboxes
        if max_accept_ratio is not None and max_accept_ratio < 1.0:
            raise ValueError(
                "max_accept_ratio must be >= 1.0 when provided, as aspect ratio is calculated as max(w/h, h/w)",
            )
        self.max_accept_ratio = max_accept_ratio  # e.g., 5.0

    def to_dict_private(self) -> dict[str, Any]:
        """Get the private dictionary representation of bounding box parameters.

        Returns:
            dict[str, Any]: Dictionary containing the bounding box parameters.

        """
        data = super().to_dict_private()
        data.update(
            {
                "min_area": self.min_area,
                "min_visibility": self.min_visibility,
                "min_width": self.min_width,
                "min_height": self.min_height,
                "check_each_transform": self.check_each_transform,
                "clip": self.clip,
                "max_accept_ratio": self.max_accept_ratio,
            },
        )
        return data

    @classmethod
    def is_serializable(cls) -> bool:
        """Check if the bounding box parameters are serializable.

        Returns:
            bool: Always returns True as BboxParams is serializable.

        """
        return True

    @classmethod
    def get_class_fullname(cls) -> str:
        """Get the full name of the class.

        Returns:
            str: The string "BboxParams".

        """
        return "BboxParams"

    def __repr__(self) -> str:
        return (
            f"BboxParams(format={self.format}, label_fields={self.label_fields}, min_area={self.min_area},"
            f" min_visibility={self.min_visibility}, min_width={self.min_width}, min_height={self.min_height},"
            f" check_each_transform={self.check_each_transform}, clip={self.clip})"
        )


class BboxProcessor(DataProcessor):
    """Processor for bounding box transformations.

    This class handles the preprocessing and postprocessing of bounding boxes during augmentation pipeline,
    including format conversion, validation, clipping, and filtering.

    Args:
        params (BboxParams): Parameters that control bounding box processing.
            See BboxParams class for details.
        additional_targets (dict[str, str] | None): Dictionary with additional targets to process.
            Keys are names of additional targets, values are their types.
            For example: {'bbox2': 'bboxes'} will handle 'bbox2' as another bounding box target.
            Default: None.

    Note:
        The processing order for bounding boxes is:
        1. Convert to albumentations format (normalized pascal_voc)
        2. Clip boxes to image boundaries (if params.clip=True)
        3. Filter invalid boxes (if params.filter_invalid_bboxes=True)
        4. Apply transformations
        5. Filter boxes based on min_area, min_visibility, min_width, min_height
        6. Convert back to the original format

    Examples:
        >>> import albumentations as A
        >>> # Process COCO format bboxes with class labels
        >>> params = A.BboxParams(
        ...     format='coco',
        ...     label_fields=['class_labels'],
        ...     min_area=1024,
        ...     min_visibility=0.1
        ... )
        >>> processor = BboxProcessor(params)
        >>>
        >>> # Process multiple bbox fields
        >>> params = A.BboxParams('pascal_voc')
        >>> processor = BboxProcessor(
        ...     params,
        ...     additional_targets={'bbox2': 'bboxes'}
        ... )

    """

    def __init__(self, params: BboxParams, additional_targets: dict[str, str] | None = None):
        super().__init__(params, additional_targets)

    @property
    def default_data_name(self) -> str:
        """Returns the default key for bounding box data in transformations.

        Returns:
            str: The string 'bboxes'.

        """
        return "bboxes"

    def ensure_data_valid(self, data: dict[str, Any]) -> None:
        """Validates the input bounding box data.

        Checks that:
        - Bounding boxes have labels (either in the bbox array or in label_fields)
        - All specified label_fields exist in the data

        Args:
            data (dict[str, Any]): Dict with bounding boxes and optional label fields.

        Raises:
            ValueError: If bounding boxes don't have labels or if label_fields are invalid.

        """
        if self.params.label_fields and not all(i in data for i in self.params.label_fields):
            msg = "Your 'label_fields' are not valid - them must have same names as params in dict"
            raise ValueError(msg)

    def filter(self, data: np.ndarray, shape: ShapeType) -> np.ndarray:
        """Filter bounding boxes based on size and visibility criteria.

        Args:
            data (np.ndarray): Array of bounding boxes in Albumentations format.
            shape (ShapeType): Shape information for validation.

        Returns:
            np.ndarray: Filtered bounding boxes that meet the criteria.

        """
        self.params: BboxParams
        return filter_bboxes(
            data,
            shape,
            min_area=self.params.min_area,
            min_visibility=self.params.min_visibility,
            min_width=self.params.min_width,
            min_height=self.params.min_height,
            max_accept_ratio=self.params.max_accept_ratio,
        )

    def check_and_convert(
        self,
        data: np.ndarray,
        shape: ShapeType,
        direction: Literal["to", "from"] = "to",
    ) -> np.ndarray:
        """Converts bounding boxes between formats and applies preprocessing/postprocessing.

        Args:
            data (np.ndarray): Array of bounding boxes to process.
            shape (ShapeType): Image shape as dict with height and width keys.
            direction (Literal["to", "from"]): Direction of conversion:
                - "to": Convert from original format to albumentations format
                - "from": Convert from albumentations format to original format
                Default: "to".

        Returns:
            np.ndarray: Processed bounding boxes.

        Note:
            When direction="to":
            1. Converts to albumentations format
            2. Clips boxes if params.clip=True
            3. Filters invalid boxes if params.filter_invalid_bboxes=True
            4. Validates remaining boxes

            When direction="from":
            1. Validates boxes
            2. Converts back to original format

        """
        if direction == "to":
            # First convert to albumentations format
            if self.params.format == "albumentations":
                converted_data = data
            else:
                converted_data = convert_bboxes_to_albumentations(
                    data,
                    self.params.format,
                    shape,
                    check_validity=False,  # Don't check validity yet
                )

            if self.params.clip and converted_data.size > 0:
                converted_data[:, :4] = np.clip(converted_data[:, :4], 0, 1)

            # Then filter invalid boxes if requested
            if self.params.filter_invalid_bboxes:
                converted_data = filter_bboxes(
                    converted_data,
                    shape,
                    min_area=0,
                    min_visibility=0,
                    min_width=0,
                    min_height=0,
                )

            # Finally check the remaining boxes
            self.check(converted_data, shape)
            return converted_data
        self.check(data, shape)
        if self.params.format == "albumentations":
            return data
        return convert_bboxes_from_albumentations(data, self.params.format, shape)

    def check(self, data: np.ndarray, shape: ShapeType) -> None:
        """Check if bounding boxes are valid.

        Args:
            data (np.ndarray): Array of bounding boxes to validate.
            shape (ShapeType): Shape to check against.

        """
        check_bboxes(data)

    def convert_from_albumentations(self, data: np.ndarray, shape: ShapeType) -> np.ndarray:
        """Convert bounding boxes from internal Albumentations format to the specified format.

        Args:
            data (np.ndarray): Bounding boxes in Albumentations format.
            shape (ShapeType): Shape information for validation.

        Returns:
            np.ndarray: Converted bounding boxes in the target format.

        """
        return np.array(
            convert_bboxes_from_albumentations(data, self.params.format, shape, check_validity=True),
            dtype=data.dtype,
        )

    def convert_to_albumentations(self, data: np.ndarray, shape: ShapeType) -> np.ndarray:
        """Convert bounding boxes from the specified format to internal Albumentations format.

        Args:
            data (np.ndarray): Bounding boxes in source format.
            shape (ShapeType): Shape information for validation.

        Returns:
            np.ndarray: Converted bounding boxes in Albumentations format.

        """
        if self.params.clip:
            data_np = convert_bboxes_to_albumentations(data, self.params.format, shape, check_validity=False)
            data_np = filter_bboxes(data_np, shape, min_area=0, min_visibility=0, min_width=0, min_height=0)
            check_bboxes(data_np)
            return data_np

        return convert_bboxes_to_albumentations(data, self.params.format, shape, check_validity=True)


@handle_empty_array("bboxes")
def normalize_bboxes(bboxes: np.ndarray, shape: ShapeType | tuple[int, int]) -> np.ndarray:
    """Normalize array of bounding boxes.

    Args:
        bboxes (np.ndarray): Denormalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.
        shape (ShapeType | tuple[int, int]): Image shape `(height, width)`.

    Returns:
        np.ndarray: Normalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.

    """
    if isinstance(shape, tuple):
        rows, cols = shape[:2]
    else:
        rows, cols = shape["height"], shape["width"]

    normalized = bboxes.copy().astype(float)
    normalized[:, [0, 2]] /= cols
    normalized[:, [1, 3]] /= rows
    return normalized


@handle_empty_array("bboxes")
def denormalize_bboxes(
    bboxes: np.ndarray,
    shape: ShapeType | tuple[int, int],
) -> np.ndarray:
    """Denormalize array of bounding boxes.

    Args:
        bboxes (np.ndarray): Normalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.
        shape (ShapeType | tuple[int, int]): Image shape `(height, width)`.

    Returns:
        np.ndarray: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.

    """
    scale_factors = (shape[1], shape[0]) if isinstance(shape, tuple) else (shape["width"], shape["height"])

    # Vectorized scaling of bbox coordinates
    return bboxes * np.array([*scale_factors, *scale_factors, *[1] * (bboxes.shape[1] - 4)], dtype=float)


def calculate_bbox_areas_in_pixels(bboxes: np.ndarray, shape: ShapeType) -> np.ndarray:
    """Calculate areas for multiple bounding boxes.
    This function computes the areas of bounding boxes given their normalized coordinates
    and the dimensions of the image they belong to. The bounding boxes are expected to be
    in the format [x_min, y_min, x_max, y_max] with normalized coordinates (0 to 1).

    Args:
        bboxes (np.ndarray): A numpy array of shape (N, 4+) where N is the number of bounding boxes.
                             Each row contains [x_min, y_min, x_max, y_max] in normalized coordinates.
                             Additional columns beyond the first 4 are ignored.
        shape (ShapeType): A tuple containing the height and width of the image (height, width).

    Returns:
        np.ndarray: A 1D numpy array of shape (N,) containing the areas of the bounding boxes in pixels.
                    Returns an empty array if the input `bboxes` is empty.

    Note:
        - The function assumes that the input bounding boxes are valid (i.e., x_max > x_min and y_max > y_min).
          Invalid bounding boxes may result in negative areas.
        - The function preserves the input array and creates a copy for internal calculations.
        - The returned areas are in pixel units, not normalized.

    Example:
        >>> bboxes = np.array([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
        >>> image_shape = (100, 100)
        >>> areas = calculate_bbox_areas(bboxes, image_shape)
        >>> print(areas)
        [1600. 3600.]

    """
    if len(bboxes) == 0:
        return np.array([], dtype=np.float32)

    # Unpack shape to variables
    height, width = shape["height"], shape["width"]

    # Directly compute denormalized bbox dimensions and areas
    widths = (bboxes[:, 2] - bboxes[:, 0]) * width
    heights = (bboxes[:, 3] - bboxes[:, 1]) * height

    return widths * heights


@handle_empty_array("bboxes")
def convert_bboxes_to_albumentations(
    bboxes: np.ndarray,
    source_format: Literal["coco", "pascal_voc", "yolo"],
    shape: ShapeType,
    check_validity: bool = False,
) -> np.ndarray:
    """Convert bounding boxes from a specified format to the format used by albumentations:
    normalized coordinates of top-left and bottom-right corners of the bounding box in the form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.

    Args:
        bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).
        source_format (Literal["coco", "pascal_voc", "yolo"]): Format of the input bounding boxes.
        shape (ShapeType): Image shape (height, width).
        check_validity (bool): Check if all boxes are valid boxes.

    Returns:
        np.ndarray: An array of bounding boxes in albumentations format with shape (num_bboxes, 4+).

    Raises:
        ValueError: If `source_format` is not 'coco', 'pascal_voc', or 'yolo'.
        ValueError: If in YOLO format, any coordinates are not in the range (0, 1].

    """
    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'",
        )

    bboxes = bboxes.copy().astype(np.float32)
    converted_bboxes = np.zeros_like(bboxes)
    converted_bboxes[:, 4:] = bboxes[:, 4:]  # Preserve additional columns

    if source_format == "coco":
        converted_bboxes[:, 0] = bboxes[:, 0]  # x_min
        converted_bboxes[:, 1] = bboxes[:, 1]  # y_min
        converted_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x_max
        converted_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y_max
    elif source_format == "yolo":
        if check_validity and np.any((bboxes[:, :4] <= 0) | (bboxes[:, :4] > 1)):
            raise ValueError(f"In YOLO format all coordinates must be float and in range (0, 1], got {bboxes}")

        w_half, h_half = bboxes[:, 2] / 2, bboxes[:, 3] / 2
        converted_bboxes[:, 0] = bboxes[:, 0] - w_half  # x_min
        converted_bboxes[:, 1] = bboxes[:, 1] - h_half  # y_min
        converted_bboxes[:, 2] = bboxes[:, 0] + w_half  # x_max
        converted_bboxes[:, 3] = bboxes[:, 1] + h_half  # y_max
    else:  # pascal_voc
        converted_bboxes[:, :4] = bboxes[:, :4]

    if source_format != "yolo":
        converted_bboxes[:, :4] = normalize_bboxes(converted_bboxes[:, :4], shape)

    if check_validity:
        check_bboxes(converted_bboxes)

    return converted_bboxes


@handle_empty_array("bboxes")
def convert_bboxes_from_albumentations(
    bboxes: np.ndarray,
    target_format: Literal["coco", "pascal_voc", "yolo"],
    shape: ShapeType,
    check_validity: bool = False,
) -> np.ndarray:
    """Convert bounding boxes from the format used by albumentations to a specified format.

    Args:
        bboxes (np.ndarray): A numpy array of albumentations bounding boxes with shape (num_bboxes, 4+).
                The first 4 columns are [x_min, y_min, x_max, y_max].
        target_format (Literal["coco", "pascal_voc", "yolo"]): Required format of the output bounding boxes.
        shape (ShapeType): Image shape (height, width).
        check_validity (bool): Check if all boxes are valid boxes.

    Returns:
        np.ndarray: An array of bounding boxes in the target format with shape (num_bboxes, 4+).

    Raises:
        ValueError: If `target_format` is not 'coco', 'pascal_voc' or 'yolo'.

    """
    if target_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown target_format {target_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'",
        )

    if check_validity:
        check_bboxes(bboxes)

    converted_bboxes = np.zeros_like(bboxes)
    converted_bboxes[:, 4:] = bboxes[:, 4:]  # Preserve additional columns

    denormalized_bboxes = denormalize_bboxes(bboxes[:, :4], shape) if target_format != "yolo" else bboxes[:, :4]

    if target_format == "coco":
        converted_bboxes[:, 0] = denormalized_bboxes[:, 0]  # x_min
        converted_bboxes[:, 1] = denormalized_bboxes[:, 1]  # y_min
        converted_bboxes[:, 2] = denormalized_bboxes[:, 2] - denormalized_bboxes[:, 0]  # width
        converted_bboxes[:, 3] = denormalized_bboxes[:, 3] - denormalized_bboxes[:, 1]  # height
    elif target_format == "yolo":
        converted_bboxes[:, 0] = (denormalized_bboxes[:, 0] + denormalized_bboxes[:, 2]) / 2  # x_center
        converted_bboxes[:, 1] = (denormalized_bboxes[:, 1] + denormalized_bboxes[:, 3]) / 2  # y_center
        converted_bboxes[:, 2] = denormalized_bboxes[:, 2] - denormalized_bboxes[:, 0]  # width
        converted_bboxes[:, 3] = denormalized_bboxes[:, 3] - denormalized_bboxes[:, 1]  # height
    else:  # pascal_voc
        converted_bboxes[:, :4] = denormalized_bboxes

    return converted_bboxes


@handle_empty_array("bboxes")
def check_bboxes(bboxes: np.ndarray) -> None:
    """Check if bounding boxes are valid.

    Args:
        bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).

    Raises:
        ValueError: If any bounding box is invalid.

    """
    # Check if all values are in range [0, 1]
    in_range = (bboxes[:, :4] >= 0) & (bboxes[:, :4] <= 1)
    close_to_zero = np.isclose(bboxes[:, :4], 0)
    close_to_one = np.isclose(bboxes[:, :4], 1)
    valid_range = in_range | close_to_zero | close_to_one

    if not np.all(valid_range):
        invalid_idx = np.where(~np.all(valid_range, axis=1))[0][0]
        invalid_bbox = bboxes[invalid_idx]
        invalid_coord = ["x_min", "y_min", "x_max", "y_max"][np.where(~valid_range[invalid_idx])[0][0]]
        invalid_value = invalid_bbox[np.where(~valid_range[invalid_idx])[0][0]]
        raise ValueError(
            f"Expected {invalid_coord} for bbox {invalid_bbox} to be in the range [0.0, 1.0], got {invalid_value}.",
        )

    # Check if x_max > x_min and y_max > y_min
    valid_order = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])

    if not np.all(valid_order):
        invalid_idx = np.where(~valid_order)[0][0]
        invalid_bbox = bboxes[invalid_idx]
        if invalid_bbox[2] <= invalid_bbox[0]:
            raise ValueError(f"x_max is less than or equal to x_min for bbox {invalid_bbox}.")

        raise ValueError(f"y_max is less than or equal to y_min for bbox {invalid_bbox}.")


@handle_empty_array("bboxes")
def clip_bboxes(bboxes: np.ndarray, shape: ShapeType) -> np.ndarray:
    """Clip bounding boxes to the image shape.

    Args:
        bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).
        shape (ShapeType): The shape of the image/volume:
                           - For 2D: {'height': int, 'width': int}
                           - For 3D: {'height': int, 'width': int, 'depth': int}

    Returns:
        np.ndarray: A numpy array of bounding boxes with shape (num_bboxes, 4+).

    """
    height, width = shape["height"], shape["width"]

    # Denormalize bboxes
    denorm_bboxes = denormalize_bboxes(bboxes, shape)

    ## Note:
    # It could be tempting to use cols - 1 and rows - 1 as the upper bounds for the clipping

    # But this would cause the bounding box to be clipped to the image dimensions - 1 which is not what we want.
    # Bounding box lives not in the middle of pixels but between them.

    # Example: for image with height 100, width 100, the pixel values are in the range [0, 99]
    # but if we want bounding box to be 1 pixel width and height and lie on the boundary of the image
    # it will be described as [99, 99, 100, 100] => clip by image_size - 1 will lead to [99, 99, 99, 99]
    # which is incorrect

    # It could be also tempting to clip `x_min`` to `cols - 1`` and `y_min` to `rows - 1`, but this also leads
    # to another error. If image fully lies outside of the visible area and min_area is set to 0, then
    # the bounding box will be clipped to the image size - 1 and will be 1 pixel in size and fully visible,
    # but it should be completely removed.

    # Clip coordinates
    denorm_bboxes[:, [0, 2]] = np.clip(denorm_bboxes[:, [0, 2]], 0, width, out=denorm_bboxes[:, [0, 2]])
    denorm_bboxes[:, [1, 3]] = np.clip(denorm_bboxes[:, [1, 3]], 0, height, out=denorm_bboxes[:, [1, 3]])

    # Normalize clipped bboxes
    return normalize_bboxes(denorm_bboxes, shape)


def filter_bboxes(
    bboxes: np.ndarray,
    shape: ShapeType,
    min_area: float = 0.0,
    min_visibility: float = 0.0,
    min_width: float = 1.0,
    min_height: float = 1.0,
    max_accept_ratio: float | None = None,
) -> np.ndarray:
    """Remove bounding boxes that either lie outside of the visible area by more than min_visibility
    or whose area in pixels is under the threshold set by `min_area`. Also crops boxes to final image size.

    Args:
        bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).
        shape (ShapeType): The shape of the image/volume:
                           - For 2D: {'height': int, 'width': int}
                           - For 3D: {'height': int, 'width': int, 'depth': int}
        min_area (float): Minimum area of a bounding box in pixels. Default: 0.0.
        min_visibility (float): Minimum fraction of area for a bounding box to remain. Default: 0.0.
        min_width (float): Minimum width of a bounding box in pixels. Default: 0.0.
        min_height (float): Minimum height of a bounding box in pixels. Default: 0.0.
        max_accept_ratio (float | None): Maximum allowed aspect ratio, calculated as max(width/height, height/width).
            Boxes with higher ratios will be filtered out. Default: None.

    Returns:
        np.ndarray: Filtered bounding boxes.

    """
    epsilon = 1e-7

    if len(bboxes) == 0:
        return np.array([], dtype=np.float32).reshape(0, 4)

    # Calculate areas of bounding boxes before clipping in pixels
    denormalized_box_areas = calculate_bbox_areas_in_pixels(bboxes, shape)

    # Clip bounding boxes in ratio
    clipped_bboxes = clip_bboxes(bboxes, shape)

    # Calculate areas of clipped bounding boxes in pixels
    clipped_box_areas = calculate_bbox_areas_in_pixels(clipped_bboxes, shape)

    # Calculate width and height of the clipped bounding boxes
    denormalized_bboxes = denormalize_bboxes(clipped_bboxes[:, :4], shape)

    clipped_widths = denormalized_bboxes[:, 2] - denormalized_bboxes[:, 0]
    clipped_heights = denormalized_bboxes[:, 3] - denormalized_bboxes[:, 1]

    # Calculate aspect ratios if needed
    if max_accept_ratio is not None:
        aspect_ratios = np.maximum(
            clipped_widths / (clipped_heights + epsilon),
            clipped_heights / (clipped_widths + epsilon),
        )
        valid_ratios = aspect_ratios <= max_accept_ratio
    else:
        valid_ratios = np.ones_like(denormalized_box_areas, dtype=bool)

    # Create a mask for bboxes that meet all criteria
    mask = (
        (denormalized_box_areas >= epsilon)
        & (clipped_box_areas >= min_area - epsilon)
        & (clipped_box_areas / (denormalized_box_areas + epsilon) >= min_visibility)
        & (clipped_widths >= min_width - epsilon)
        & (clipped_heights >= min_height - epsilon)
        & valid_ratios
    )

    # Apply the mask to get the filtered bboxes
    filtered_bboxes = clipped_bboxes[mask]

    return np.array([], dtype=np.float32).reshape(0, 4) if len(filtered_bboxes) == 0 else filtered_bboxes


def union_of_bboxes(bboxes: np.ndarray, erosion_rate: float) -> np.ndarray | None:
    """Calculate union of bounding boxes. Boxes could be in albumentations or Pascal Voc format.

    Args:
        bboxes (np.ndarray): List of bounding boxes
        erosion_rate (float): How much each bounding box can be shrunk, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox lose its volume.

    Returns:
        np.ndarray | None: A bounding box `(x_min, y_min, x_max, y_max)` or None if no bboxes are given or if
                    the bounding boxes become invalid after erosion.

    """
    if not bboxes.size:
        return None

    if erosion_rate == 1:
        return None

    if bboxes.shape[0] == 1:
        return bboxes[0][:4]

    epsilon = 1e-6

    x_min, y_min = np.min(bboxes[:, :2], axis=0)
    x_max, y_max = np.max(bboxes[:, 2:4], axis=0)

    width = x_max - x_min
    height = y_max - y_min

    erosion_x = width * erosion_rate * 0.5
    erosion_y = height * erosion_rate * 0.5

    x_min += erosion_x
    y_min += erosion_y
    x_max -= erosion_x
    y_max -= erosion_y

    if abs(x_max - x_min) < epsilon or abs(y_max - y_min) < epsilon:
        return None

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def bboxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """Create bounding boxes from binary masks (fast version)

    Args:
        masks (np.ndarray): Binary masks of shape (H, W) or (N, H, W) where N is the number of masks,
                           and H, W are the height and width of each mask.

    Returns:
        np.ndarray: An array of bounding boxes with shape (N, 4), where each row is
                   (x_min, y_min, x_max, y_max).

    """
    # Handle single mask case by adding batch dimension
    if len(masks.shape) == MONO_CHANNEL_DIMENSIONS:
        masks = masks[np.newaxis, ...]

    rows = np.any(masks, axis=2)
    cols = np.any(masks, axis=1)

    bboxes = np.zeros((masks.shape[0], 4), dtype=np.int32)

    for i, (row, col) in enumerate(zip(rows, cols)):
        if not np.any(row) or not np.any(col):
            bboxes[i] = [-1, -1, -1, -1]
        else:
            y_min, y_max = np.where(row)[0][[0, -1]]
            x_min, x_max = np.where(col)[0][[0, -1]]
            bboxes[i] = [x_min, y_min, x_max + 1, y_max + 1]

    return bboxes


def masks_from_bboxes(bboxes: np.ndarray, shape: ShapeType | tuple[int, int]) -> np.ndarray:
    """Convert bounding boxes to masks.

    Args:
        bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).
        shape (ShapeType | tuple[int, int]): Image shape (height, width).

    Returns:
        np.ndarray: A numpy array of masks with shape (num_bboxes, height, width).

    """
    if isinstance(shape, dict):
        height, width = shape["height"], shape["width"]
    else:
        height, width = shape[:2]

    masks = np.zeros((len(bboxes), height, width), dtype=np.uint8)
    y, x = np.ogrid[:height, :width]

    for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes[:, :4].astype(int)):
        masks[i] = (x_min <= x) & (x < x_max) & (y_min <= y) & (y < y_max)

    return masks


def bboxes_to_mask(
    bboxes: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert bounding boxes to a single mask.

    Args:
        bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).
        image_shape (tuple[int, int]): Image shape (height, width).

    Returns:
        np.ndarray: A numpy array of shape (height, width) with 1s where any bounding box is present.

    """
    height, width = image_shape[:2]
    num_boxes = len(bboxes)

    # Create multi-channel mask where each channel represents one bbox
    bbox_masks = np.zeros((height, width, num_boxes), dtype=np.uint8)

    # Fill each bbox in its channel
    for idx, box in enumerate(bboxes):
        x_min, y_min, x_max, y_max = map(round, box[:4])
        x_min = max(0, min(width - 1, x_min))
        x_max = max(0, min(width - 1, x_max))
        y_min = max(0, min(height - 1, y_min))
        y_max = max(0, min(height - 1, y_max))
        bbox_masks[y_min : y_max + 1, x_min : x_max + 1, idx] = 1

    return bbox_masks


def mask_to_bboxes(
    masks: np.ndarray,
    original_bboxes: np.ndarray,
) -> np.ndarray:
    """Convert masks back to bounding boxes.

    Args:
        masks (np.ndarray): A numpy array of masks with shape (num_masks, height, width).
        original_bboxes (np.ndarray): Original bounding boxes with shape (num_bboxes, 4+).

    Returns:
        np.ndarray: A numpy array of bounding boxes with shape (num_masks, 4+).

    """
    num_boxes = masks.shape[-1]
    new_bboxes = []

    num_boxes = masks.shape[-1]

    if num_boxes == 0:
        # Return empty array with correct shape
        return np.zeros((0, original_bboxes.shape[1]), dtype=original_bboxes.dtype)

    for idx in range(num_boxes):
        mask = masks[..., idx]
        if np.any(mask):
            y_coords, x_coords = np.where(mask)
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            new_bboxes.append([x_min, y_min, x_max, y_max])
        else:
            # If bbox disappeared, use original coords
            new_bboxes.append(original_bboxes[idx, :4])

    new_bboxes = np.array(new_bboxes)

    return (
        np.column_stack([new_bboxes, original_bboxes[:, 4:]])
        if original_bboxes.shape[1] > NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS
        else new_bboxes
    )
