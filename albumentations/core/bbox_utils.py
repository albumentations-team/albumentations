from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from albumentations.augmentations.utils import handle_empty_array
from albumentations.core.types import MONO_CHANNEL_DIMENSIONS

from .utils import DataProcessor, Params

__all__ = [
    "normalize_bboxes",
    "denormalize_bboxes",
    "convert_bboxes_to_albumentations",
    "convert_bboxes_from_albumentations",
    "check_bboxes",
    "filter_bboxes",
    "union_of_bboxes",
    "BboxProcessor",
    "BboxParams",
]

BBOX_WITH_LABEL_SHAPE = 5


class BboxParams(Params):
    """Parameters of bounding boxes

    Args:
        format Literal["coco", "pascal_voc", "albumentations", "yolo"]: format of bounding boxes.

            The `coco` format
                `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
            The `pascal_voc` format
                `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
            The `albumentations` format
                is like `pascal_voc`, but normalized,
                in other words: `[x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
            The `yolo` format
                `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
                `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.

        label_fields (list): List of fields joined with boxes, e.g., labels.
        min_area (float): Minimum area of a bounding box in pixels or normalized units.
            Bounding boxes with an area less than this value will be removed. Default: 0.0.
        min_visibility (float): Minimum fraction of area for a bounding box to remain in the list.
            Bounding boxes with a visible area less than this fraction will be removed. Default: 0.0.
        min_width (float): Minimum width of a bounding box in pixels or normalized units.
            Bounding boxes with a width less than this value will be removed. Default: 0.0.
        min_height (float): Minimum height of a bounding box in pixels or normalized units.
            Bounding boxes with a height less than this value will be removed. Default: 0.0.
        check_each_transform (bool): If True, bounding boxes will be checked after each dual transform. Default: True.
        clip (bool): If True, bounding boxes will be clipped to the image borders before applying any transform.
            Default: False.

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
    ):
        super().__init__(format, label_fields)
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.min_width = min_width
        self.min_height = min_height
        self.check_each_transform = check_each_transform
        self.clip = clip

    def to_dict_private(self) -> dict[str, Any]:
        data = super().to_dict_private()
        data.update(
            {
                "min_area": self.min_area,
                "min_visibility": self.min_visibility,
                "min_width": self.min_width,
                "min_height": self.min_height,
                "check_each_transform": self.check_each_transform,
                "clip": self.clip,
            },
        )
        return data

    @classmethod
    def is_serializable(cls) -> bool:
        return True

    @classmethod
    def get_class_fullname(cls) -> str:
        return "BboxParams"

    def __repr__(self) -> str:
        return (
            f"BboxParams(format={self.format}, label_fields={self.label_fields}, min_area={self.min_area},"
            f" min_visibility={self.min_visibility}, min_width={self.min_width}, min_height={self.min_height},"
            f" check_each_transform={self.check_each_transform}, clip={self.clip})"
        )


class BboxProcessor(DataProcessor):
    def __init__(self, params: BboxParams, additional_targets: dict[str, str] | None = None):
        super().__init__(params, additional_targets)

    @property
    def default_data_name(self) -> str:
        return "bboxes"

    def ensure_data_valid(self, data: dict[str, Any]) -> None:
        for data_name in self.data_fields:
            data_exists = data_name in data and len(data[data_name])
            if data_exists and len(data[data_name][0]) < BBOX_WITH_LABEL_SHAPE and self.params.label_fields is None:
                msg = (
                    "Please specify 'label_fields' in 'bbox_params' or add labels to the end of bbox "
                    "because bboxes must have labels"
                )
                raise ValueError(msg)
        if self.params.label_fields and not all(i in data for i in self.params.label_fields):
            msg = "Your 'label_fields' are not valid - them must have same names as params in dict"
            raise ValueError(msg)

    def filter(self, data: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        self.params: BboxParams
        return filter_bboxes(
            data,
            image_shape,
            min_area=self.params.min_area,
            min_visibility=self.params.min_visibility,
            min_width=self.params.min_width,
            min_height=self.params.min_height,
        )

    def check(self, data: np.ndarray, image_shape: tuple[int, int]) -> None:
        check_bboxes(data)

    def convert_from_albumentations(self, data: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        return np.array(
            convert_bboxes_from_albumentations(data, self.params.format, image_shape, check_validity=True),
            dtype=np.float32,
        )

    def convert_to_albumentations(self, data: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        if self.params.clip:
            data_np = convert_bboxes_to_albumentations(data, self.params.format, image_shape, check_validity=False)
            data_np = filter_bboxes(data_np, image_shape, min_area=0, min_visibility=0, min_width=0, min_height=0)
            check_bboxes(data_np)
            return data_np

        return convert_bboxes_to_albumentations(data, self.params.format, image_shape, check_validity=True)


@handle_empty_array
def normalize_bboxes(bboxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Normalize array of bounding boxes.

    Args:
        bboxes: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.
        image_shape: Image shape `(height, width)`.

    Returns:
        Normalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.

    """
    rows, cols = image_shape[:2]
    normalized = bboxes.copy().astype(float)
    normalized[:, [0, 2]] /= cols
    normalized[:, [1, 3]] /= rows
    return normalized


@handle_empty_array
def denormalize_bboxes(
    bboxes: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Denormalize  array of bounding boxes.

    Args:
        bboxes: Normalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.
        image_shape: Image shape `(height, width)`.

    Returns:
        Denormalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.

    """
    rows, cols = image_shape[:2]

    denormalized = bboxes.copy().astype(float)
    denormalized[:, [0, 2]] *= cols
    denormalized[:, [1, 3]] *= rows
    return denormalized


def calculate_bbox_areas_in_pixels(bboxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Calculate areas for multiple bounding boxes.

    This function computes the areas of bounding boxes given their normalized coordinates
    and the dimensions of the image they belong to. The bounding boxes are expected to be
    in the format [x_min, y_min, x_max, y_max] with normalized coordinates (0 to 1).

    Args:
        bboxes (np.ndarray): A numpy array of shape (N, 4+) where N is the number of bounding boxes.
                             Each row contains [x_min, y_min, x_max, y_max] in normalized coordinates.
                             Additional columns beyond the first 4 are ignored.
        image_shape (tuple[int, int]): A tuple containing the height and width of the image (height, width).

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

    height, width = image_shape
    bboxes_denorm = bboxes.copy()
    bboxes_denorm[:, [0, 2]] *= width
    bboxes_denorm[:, [1, 3]] *= height
    return (bboxes_denorm[:, 2] - bboxes_denorm[:, 0]) * (bboxes_denorm[:, 3] - bboxes_denorm[:, 1])


@handle_empty_array
def convert_bboxes_to_albumentations(
    bboxes: np.ndarray,
    source_format: Literal["coco", "pascal_voc", "yolo"],
    image_shape: tuple[int, int],
    check_validity: bool = False,
) -> np.ndarray:
    """Convert bounding boxes from a specified format to the format used by albumentations:
    normalized coordinates of top-left and bottom-right corners of the bounding box in the form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.

    Args:
        bboxes: A numpy array of bounding boxes with shape (num_bboxes, 4+).
        source_format: Format of the input bounding boxes. Should be 'coco', 'pascal_voc', or 'yolo'.
        image_shape: Image shape (height, width).
        check_validity: Check if all boxes are valid boxes.

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
        converted_bboxes[:, :4] = normalize_bboxes(converted_bboxes[:, :4], image_shape)

    if check_validity:
        check_bboxes(converted_bboxes)

    return converted_bboxes


@handle_empty_array
def convert_bboxes_from_albumentations(
    bboxes: np.ndarray,
    target_format: Literal["coco", "pascal_voc", "yolo"],
    image_shape: tuple[int, int],
    check_validity: bool = False,
) -> np.ndarray:
    """Convert bounding boxes from the format used by albumentations to a specified format.

    Args:
        bboxes: A numpy array of albumentations bounding boxes with shape (num_bboxes, 4+).
                The first 4 columns are [x_min, y_min, x_max, y_max].
        target_format: Required format of the output bounding boxes. Should be 'coco', 'pascal_voc' or 'yolo'.
        image_shape: Image shape (height, width).
        check_validity: Check if all boxes are valid boxes.

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

    denormalized_bboxes = denormalize_bboxes(bboxes[:, :4], image_shape) if target_format != "yolo" else bboxes[:, :4]

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


@handle_empty_array
def check_bboxes(bboxes: np.ndarray) -> None:
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser than maximums.

    Args:
        bboxes: numpy array of shape (num_bboxes, 4+) where first 4 coordinates are x_min, y_min, x_max, y_max.

    Raises:
        ValueError: If any bbox is invalid.
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


@handle_empty_array
def clip_bboxes(bboxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Clips the bounding box coordinates to ensure they fit within the boundaries of an image.

    Parameters:
        bboxes (np.ndarray): Array of bounding boxes with shape (num_boxes, 4+) in normalized format.
                             The first 4 columns are [x_min, y_min, x_max, y_max].
        image_shape (Tuple[int, int]): Image shape (height, width).

    Returns:
        np.ndarray: The clipped bounding boxes, normalized to the image dimensions.

    """
    height, width = image_shape[:2]

    # Denormalize bboxes
    denorm_bboxes = denormalize_bboxes(bboxes, image_shape)

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
    denorm_bboxes[:, [0, 2]] = np.clip(denorm_bboxes[:, [0, 2]], 0, width)
    denorm_bboxes[:, [1, 3]] = np.clip(denorm_bboxes[:, [1, 3]], 0, height)

    # Normalize clipped bboxes
    return normalize_bboxes(denorm_bboxes, image_shape)


def filter_bboxes(
    bboxes: np.ndarray,
    image_shape: tuple[int, int],
    min_area: float = 0.0,
    min_visibility: float = 0.0,
    min_width: float = 1.0,
    min_height: float = 1.0,
) -> np.ndarray:
    """Remove bounding boxes that either lie outside of the visible area by more than min_visibility
    or whose area in pixels is under the threshold set by `min_area`. Also crops boxes to final image size.

    Args:
        bboxes: numpy array of bounding boxes with shape (num_bboxes, 4+).
                The first 4 columns are [x_min, y_min, x_max, y_max].
        image_shape: Image shape (height, width).
        min_area: Minimum area of a bounding box in pixels. Default: 0.0.
        min_visibility: Minimum fraction of area for a bounding box to remain. Default: 0.0.
        min_width: Minimum width of a bounding box in pixels. Default: 0.0.
        min_height: Minimum height of a bounding box in pixels. Default: 0.0.

    Returns:
        numpy array of filtered bounding boxes.
    """
    epsilon = 1e-7

    if len(bboxes) == 0:
        return np.array([], dtype=np.float32).reshape(0, 4)

    # Calculate areas of bounding boxes before clipping in pixels
    denormalized_box_areas = calculate_bbox_areas_in_pixels(bboxes, image_shape)

    # Clip bounding boxes in ratio
    clipped_bboxes = clip_bboxes(bboxes, image_shape)

    # Calculate areas of clipped bounding boxes in pixels
    clipped_box_areas = calculate_bbox_areas_in_pixels(clipped_bboxes, image_shape)

    # Calculate width and height of the clipped bounding boxes
    denormalized_bboxes = denormalize_bboxes(clipped_bboxes[:, :4], image_shape)

    clipped_widths = denormalized_bboxes[:, 2] - denormalized_bboxes[:, 0]
    clipped_heights = denormalized_bboxes[:, 3] - denormalized_bboxes[:, 1]

    # Create a mask for bboxes that meet all criteria
    mask = (
        (denormalized_box_areas >= epsilon)
        & (clipped_box_areas >= min_area - epsilon)
        & (clipped_box_areas / denormalized_box_areas >= min_visibility - epsilon)
        & (clipped_widths >= min_width - epsilon)
        & (clipped_heights >= min_height - epsilon)
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


def masks_from_bboxes(bboxes: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
    """Create binary masks from multiple bounding boxes

    Args:
        bboxes: Array of bounding boxes with shape (N, 4), where N is the number of boxes
        img_shape: Image shape (height, width)

    Returns:
        masks: Array of binary masks with shape (N, height, width)

    """
    height, width = img_shape[:2]
    masks = np.zeros((len(bboxes), height, width), dtype=np.uint8)
    y, x = np.ogrid[:height, :width]

    for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes[:, :4].astype(int)):
        masks[i] = (x_min <= x) & (x < x_max) & (y_min <= y) & (y < y_max)

    return masks
