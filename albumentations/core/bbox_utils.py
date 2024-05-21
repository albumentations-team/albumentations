from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from .types import BoxInternalType, BoxType
from .utils import DataProcessor, Params

__all__ = [
    "normalize_bbox",
    "denormalize_bbox",
    "normalize_bboxes",
    "denormalize_bboxes",
    "calculate_bbox_area",
    "filter_bboxes_by_visibility",
    "convert_bbox_to_albumentations",
    "convert_bbox_from_albumentations",
    "convert_bboxes_to_albumentations",
    "convert_bboxes_from_albumentations",
    "check_bbox",
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
        format (str): format of bounding boxes. Should be `coco`, `pascal_voc`, `albumentations` or `yolo`.

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
        format: str,
        label_fields: Optional[Sequence[Any]] = None,
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

    def to_dict_private(self) -> Dict[str, Any]:
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


class BboxProcessor(DataProcessor):
    def __init__(self, params: BboxParams, additional_targets: Optional[Dict[str, str]] = None):
        super().__init__(params, additional_targets)

    @property
    def default_data_name(self) -> str:
        return "bboxes"

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
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

    def filter(self, data: Sequence[BoxType], rows: int, cols: int) -> List[BoxType]:
        self.params: BboxParams
        return filter_bboxes(
            data,
            rows,
            cols,
            min_area=self.params.min_area,
            min_visibility=self.params.min_visibility,
            min_width=self.params.min_width,
            min_height=self.params.min_height,
        )

    def check(self, data: Sequence[BoxType], rows: int, cols: int) -> None:
        check_bboxes(data)

    def convert_from_albumentations(self, data: Sequence[BoxType], rows: int, cols: int) -> List[BoxType]:
        return convert_bboxes_from_albumentations(data, self.params.format, rows, cols, check_validity=True)

    def convert_to_albumentations(self, data: Sequence[BoxType], rows: int, cols: int) -> List[BoxType]:
        if self.params.clip:
            data = convert_bboxes_to_albumentations(data, self.params.format, rows, cols, check_validity=False)
            data = filter_bboxes(data, rows, cols, min_area=0, min_visibility=0, min_width=0, min_height=0)
            for bbox in data:
                check_bbox(bbox)
            return data

        return convert_bboxes_to_albumentations(data, self.params.format, rows, cols, check_validity=True)


def normalize_bbox(bbox: BoxType, rows: int, cols: int) -> BoxType:
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.

    Args:
        bbox: Denormalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.

    Returns:
        Normalized bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If rows or cols is less or equal zero

    """
    if rows <= 0:
        msg = "Argument rows must be positive integer"
        raise ValueError(msg)
    if cols <= 0:
        msg = "Argument cols must be positive integer"
        raise ValueError(msg)

    tail: Tuple[Any, ...]
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])
    x_min /= cols
    x_max /= cols
    y_min /= rows
    y_max /= rows

    return cast(BoxType, (x_min, y_min, x_max, y_max, *tail))


def denormalize_bbox(bbox: BoxType, rows: int, cols: int) -> BoxType:
    """Denormalize coordinates of a bounding box. Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bbox`.

    Args:
        bbox: Normalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.

    Returns:
        Denormalized bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If rows or cols is less or equal zero

    """
    tail: Tuple[Any, ...]
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    if rows <= 0:
        msg = "Argument rows must be positive integer"
        raise ValueError(msg)
    if cols <= 0:
        msg = "Argument cols must be positive integer"
        raise ValueError(msg)

    x_min, x_max = x_min * cols, x_max * cols
    y_min, y_max = y_min * rows, y_max * rows

    return cast(BoxType, (x_min, y_min, x_max, y_max, *tail))


def normalize_bboxes(bboxes: Sequence[BoxType], rows: int, cols: int) -> List[BoxType]:
    """Normalize a list of bounding boxes.

    Args:
        bboxes: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        rows: Image height.
        cols: Image width.

    Returns:
        Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.

    """
    return [normalize_bbox(bbox, rows, cols) for bbox in bboxes]


def denormalize_bboxes(bboxes: Sequence[BoxType], rows: int, cols: int) -> List[BoxType]:
    """Denormalize a list of bounding boxes.

    Args:
        bboxes: Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        rows: Image height.
        cols: Image width.

    Returns:
        List: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.

    """
    return [denormalize_bbox(bbox, rows, cols) for bbox in bboxes]


def calculate_bbox_area(bbox: BoxType, rows: int, cols: int) -> float:
    """Calculate the area of a bounding box in (fractional) pixels.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.

    Return:
        Area in (fractional) pixels of the (denormalized) bounding box.

    """
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox[:4]
    return (x_max - x_min) * (y_max - y_min)


def filter_bboxes_by_visibility(
    original_shape: Sequence[int],
    bboxes: Sequence[BoxType],
    transformed_shape: Sequence[int],
    transformed_bboxes: Sequence[BoxType],
    threshold: float = 0.0,
    min_area: float = 0.0,
) -> List[BoxType]:
    """Filter bounding boxes and return only those boxes whose visibility after transformation is above
    the threshold and minimal area of bounding box in pixels is more then min_area.

    Args:
        original_shape: Original image shape `(height, width, ...)`.
        bboxes: Original bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        transformed_shape: Transformed image shape `(height, width)`.
        transformed_bboxes: Transformed bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        threshold: visibility threshold. Should be a value in the range [0.0, 1.0].
        min_area: Minimal area threshold.

    Returns:
        Filtered bounding boxes `[(x_min, y_min, x_max, y_max)]`.

    """
    img_height, img_width = original_shape[:2]
    transformed_img_height, transformed_img_width = transformed_shape[:2]

    visible_bboxes = []
    for bbox, transformed_bbox in zip(bboxes, transformed_bboxes):
        if not all(0.0 <= value <= 1.0 for value in transformed_bbox[:4]):
            continue
        bbox_area = calculate_bbox_area(bbox, img_height, img_width)
        transformed_bbox_area = calculate_bbox_area(transformed_bbox, transformed_img_height, transformed_img_width)
        if transformed_bbox_area < min_area:
            continue
        visibility = transformed_bbox_area / bbox_area
        if visibility >= threshold:
            visible_bboxes.append(transformed_bbox)
    return visible_bboxes


def convert_bbox_to_albumentations(
    bbox: BoxType,
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
) -> BoxType:
    """Convert a bounding box from a format specified in `source_format` to the format used by albumentations:
    normalized coordinates of top-left and bottom-right corners of the bounding box in a form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.

    Args:
        bbox: A bounding box tuple.
        source_format: format of the bounding box. Should be 'coco', 'pascal_voc', or 'yolo'.
        check_validity: Check if all boxes are valid boxes.
        rows: Image height.
        cols: Image width.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    Note:
        The `coco` format of a bounding box looks like `(x_min, y_min, width, height)`, e.g. (97, 12, 150, 200).
        The `pascal_voc` format of a bounding box looks like `(x_min, y_min, x_max, y_max)`, e.g. (97, 12, 247, 212).
        The `yolo` format of a bounding box looks like `(x, y, width, height)`, e.g. (0.3, 0.1, 0.05, 0.07);
        where `x`, `y` coordinates of the center of the box, all values normalized to 1 by image height and width.

    Raises:
        ValueError: if `target_format` is not equal to `coco` or `pascal_voc`, or `yolo`.
        ValueError: If in YOLO format all labels not in range (0, 1).

    """
    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'",
        )

    if source_format == "coco":
        (x_min, y_min, width, height), tail = bbox[:4], bbox[4:]
        x_max = x_min + width
        y_max = y_min + height
    elif source_format == "yolo":
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/scripts/voc_label.py#L12
        _bbox = np.array(bbox[:4])
        if check_validity and np.any((_bbox <= 0) | (_bbox > 1)):
            msg = "In YOLO format all coordinates must be float and in range (0, 1]"
            raise ValueError(msg)

        (x, y, width, height), tail = bbox[:4], bbox[4:]

        w_half, h_half = width / 2, height / 2
        x_min = x - w_half
        y_min = y - h_half
        x_max = x_min + width
        y_max = y_min + height
    else:
        (x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]

    bbox = (x_min, y_min, x_max, y_max, *tuple(tail))

    if source_format != "yolo":
        bbox = normalize_bbox(bbox, rows, cols)
    if check_validity:
        check_bbox(bbox)
    return bbox


def convert_bbox_from_albumentations(
    bbox: BoxType,
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
) -> BoxType:
    """Convert a bounding box from the format used by albumentations to a format, specified in `target_format`.

    Args:
        bbox: An albumentations bounding box `(x_min, y_min, x_max, y_max)`.
        target_format: required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.
        rows: Image height.
        cols: Image width.
        check_validity: Check if all boxes are valid boxes.

    Returns:
        tuple: A bounding box.

    Note:
        The `coco` format of a bounding box looks like `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
        The `pascal_voc` format of a bounding box looks like `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
        The `yolo` format of a bounding box looks like `[x, y, width, height]`, e.g. [0.3, 0.1, 0.05, 0.07].

    Raises:
        ValueError: if `target_format` is not equal to `coco`, `pascal_voc` or `yolo`.

    """
    if target_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown target_format {target_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'",
        )
    if check_validity:
        check_bbox(bbox)

    if target_format != "yolo":
        bbox = denormalize_bbox(bbox, rows, cols)
    if target_format == "coco":
        (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])
        width = x_max - x_min
        height = y_max - y_min
        bbox = cast(BoxType, (x_min, y_min, width, height, *tail))
    elif target_format == "yolo":
        (x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        bbox = cast(BoxType, (x, y, width, height, *tail))
    return bbox


def convert_bboxes_to_albumentations(
    bboxes: Sequence[BoxType],
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
) -> List[BoxType]:
    """Convert a list bounding boxes from a format specified in `source_format` to the format used by albumentations"""
    return [convert_bbox_to_albumentations(bbox, source_format, rows, cols, check_validity) for bbox in bboxes]


def convert_bboxes_from_albumentations(
    bboxes: Sequence[BoxType],
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
) -> List[BoxType]:
    """Convert a list of bounding boxes from the format used by albumentations to a format, specified
    in `target_format`.

    Args:
        bboxes: List of albumentations bounding box `(x_min, y_min, x_max, y_max)`.
        target_format: required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.
        rows: Image height.
        cols: Image width.
        check_validity: Check if all boxes are valid boxes.

    Returns:
        List of bounding boxes.

    """
    return [convert_bbox_from_albumentations(bbox, target_format, rows, cols, check_validity) for bbox in bboxes]


def check_bbox(bbox: BoxType) -> None:
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
        if not 0 <= value <= 1 and not np.isclose(value, 0) and not np.isclose(value, 1):
            raise ValueError(f"Expected {name} for bbox {bbox} to be in the range [0.0, 1.0], got {value}.")
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max <= x_min:
        raise ValueError(f"x_max is less than or equal to x_min for bbox {bbox}.")
    if y_max <= y_min:
        raise ValueError(f"y_max is less than or equal to y_min for bbox {bbox}.")


def check_bboxes(bboxes: Sequence[BoxType]) -> None:
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for bbox in bboxes:
        check_bbox(bbox)


def clip_bbox(bbox: BoxType, rows: int, cols: int) -> BoxType:
    """Clips the bounding box coordinates to ensure they fit within the boundaries of an image.

    The function first denormalizes the bounding box coordinates from relative to absolute (pixel) values.
    Each coordinate is then clipped to the respective dimension of the image to ensure that the bounding box
    does not exceed the image's boundaries. Finally, the bounding box is normalized back to relative values.

    Parameters:
        bbox (BoxInternalType): The bounding box in normalized format (relative to image dimensions).
        rows (int): The number of rows (height) in the image.
        cols (int): The number of columns (width) in the image.

    Returns:
        BoxInternalType: The clipped bounding box, normalized to the image dimensions.
    """
    x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]

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

    x_min = np.clip(x_min, 0, cols)
    x_max = np.clip(x_max, 0, cols)
    y_min = np.clip(y_min, 0, rows)
    y_max = np.clip(y_max, 0, rows)
    return cast(BoxType, normalize_bbox((x_min, y_min, x_max, y_max), rows, cols) + tuple(bbox[4:]))


def filter_bboxes(
    bboxes: Sequence[BoxType],
    rows: int,
    cols: int,
    min_area: float = 0.0,
    min_visibility: float = 0.0,
    min_width: float = 0.0,
    min_height: float = 0.0,
) -> List[BoxType]:
    """Remove bounding boxes that either lie outside of the visible area by more then min_visibility
    or whose area in pixels is under the threshold set by `min_area`. Also it crops boxes to final image size.

    Args:
        bboxes: List of albumentations bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.
        min_area: Minimum area of a bounding box. All bounding boxes whose visible area in pixels.
            is less than this value will be removed. Default: 0.0.
        min_visibility: Minimum fraction of area for a bounding box to remain this box in list. Default: 0.0.
        min_width: Minimum width of a bounding box. All bounding boxes whose width is
            less than this value will be removed. Default: 0.0.
        min_height: Minimum height of a bounding box. All bounding boxes whose height is
            less than this value will be removed. Default: 0.0.

    Returns:
        List of bounding boxes.

    """
    resulting_boxes: List[BoxType] = []
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        # Calculate areas of bounding box before and after clipping.
        transformed_box_area = calculate_bbox_area(bbox, rows, cols)
        clipped_bbox = clip_bbox(bbox, rows, cols)

        bbox, tail = clipped_bbox[:4], clipped_bbox[4:]

        clipped_box_area = calculate_bbox_area(bbox, rows, cols)

        # Calculate width and height of the clipped bounding box.
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
        clipped_width, clipped_height = x_max - x_min, y_max - y_min

        if (
            clipped_box_area != 0  # to ensure transformed_box_area!=0 and to handle min_area=0 or min_visibility=0
            and clipped_box_area >= min_area
            and clipped_box_area / transformed_box_area >= min_visibility
            and clipped_width >= min_width
            and clipped_height >= min_height
        ):
            resulting_boxes.append(cast(BoxType, bbox + tail))
    return resulting_boxes


def union_of_bboxes(height: int, width: int, bboxes: Sequence[BoxType], erosion_rate: float = 0.0) -> BoxInternalType:
    """Calculate union of bounding boxes.

    Args:
        height (float): Height of image or space.
        width (float): Width of image or space.
        bboxes (List[tuple]): List like bounding boxes. Format is `[(x_min, y_min, x_max, y_max)]`.
        erosion_rate (float): How much each bounding box can be shrunk, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x1, y1 = width, height
    x2, y2 = 0, 0
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox[:4]
        bbox_width, bbox_height = x_max - x_min, y_max - y_min
        lim_x1, lim_y1 = x_min + erosion_rate * bbox_width, y_min + erosion_rate * bbox_height
        lim_x2, lim_y2 = x_max - erosion_rate * bbox_width, y_max - erosion_rate * bbox_height
        x1, y1 = np.min([x1, lim_x1]), np.min([y1, lim_y1])
        x2, y2 = np.max([x2, lim_x2]), np.max([y2, lim_y2])
    return x1, y1, x2, y2
