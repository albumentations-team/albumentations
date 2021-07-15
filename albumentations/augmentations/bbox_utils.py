from __future__ import division
from albumentations.core.utils import DataProcessor

import numpy as np

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
    "BboxProcessor",
]


class BboxProcessor(DataProcessor):
    @property
    def default_data_name(self):
        return "bboxes"

    def ensure_data_valid(self, data):
        for data_name in self.data_fields:
            data_exists = data_name in data and len(data[data_name])
            if data_exists and len(data[data_name][0]) < 5:
                if self.params.label_fields is None:
                    raise ValueError(
                        "Please specify 'label_fields' in 'bbox_params' or add labels to the end of bbox "
                        "because bboxes must have labels"
                    )
        if self.params.label_fields:
            if not all(i in data.keys() for i in self.params.label_fields):
                raise ValueError("Your 'label_fields' are not valid - them must have same names as params in dict")

    def filter(self, data, rows, cols):
        return filter_bboxes(
            data, rows, cols, min_area=self.params.min_area, min_visibility=self.params.min_visibility
        )

    def check(self, data, rows, cols):
        return check_bboxes(data)

    def convert_from_albumentations(self, data, rows, cols):
        return convert_bboxes_from_albumentations(data, self.params.format, rows, cols, check_validity=True)

    def convert_to_albumentations(self, data, rows, cols):
        return convert_bboxes_to_albumentations(data, self.params.format, rows, cols, check_validity=True)


def normalize_bbox(bbox, rows, cols):
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.

    Args:
        bbox (tuple): Denormalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: Normalized bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If rows or cols is less or equal zero

    """
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")

    x_min, x_max = x_min / cols, x_max / cols
    y_min, y_max = y_min / rows, y_max / rows

    return (x_min, y_min, x_max, y_max) + tail


def denormalize_bbox(bbox, rows, cols):
    """Denormalize coordinates of a bounding box. Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bbox`.

    Args:
        bbox (tuple): Normalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: Denormalized bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If rows or cols is less or equal zero

    """
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")

    x_min, x_max = x_min * cols, x_max * cols
    y_min, y_max = y_min * rows, y_max * rows

    return (x_min, y_min, x_max, y_max) + tail


def normalize_bboxes(bboxes, rows, cols):
    """Normalize a list of bounding boxes.

    Args:
        bboxes (List[tuple]): Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        List[tuple]: Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.

    """
    return [normalize_bbox(bbox, rows, cols) for bbox in bboxes]


def denormalize_bboxes(bboxes, rows, cols):
    """Denormalize a list of bounding boxes.

    Args:
        bboxes (List[tuple]): Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        List[tuple]: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.

    """
    return [denormalize_bbox(bbox, rows, cols) for bbox in bboxes]


def calculate_bbox_area(bbox, rows, cols):
    """Calculate the area of a bounding box in pixels.

    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image height.
        cols (int): Image width.

    Return:
        int: Area of a bounding box in pixels.

    """
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox[:4]
    area = (x_max - x_min) * (y_max - y_min)
    return area


def filter_bboxes_by_visibility(
    original_shape, bboxes, transformed_shape, transformed_bboxes, threshold=0.0, min_area=0.0
):
    """Filter bounding boxes and return only those boxes whose visibility after transformation is above
    the threshold and minimal area of bounding box in pixels is more then min_area.

    Args:
        original_shape (tuple): Original image shape `(height, width)`.
        bboxes (List[tuple]): Original bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        transformed_shape (tuple): Transformed image shape `(height, width)`.
        transformed_bboxes (List[tuple]): Transformed bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        threshold (float): visibility threshold. Should be a value in the range [0.0, 1.0].
        min_area (float): Minimal area threshold.

    Returns:
        List[tuple]: Filtered bounding boxes `[(x_min, y_min, x_max, y_max)]`.

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


def convert_bbox_to_albumentations(bbox, source_format, rows, cols, check_validity=False):
    """Convert a bounding box from a format specified in `source_format` to the format used by albumentations:
    normalized coordinates of top-left and bottom-right corners of the bounding box in a form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.

    Args:
        bbox (tuple): A bounding box tuple.
        source_format (str): format of the bounding box. Should be 'coco', 'pascal_voc', or 'yolo'.
        check_validity (bool): Check if all boxes are valid boxes.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    Note:
        The `coco` format of a bounding box looks like `(x_min, y_min, width, height)`, e.g. (97, 12, 150, 200).
        The `pascal_voc` format of a bounding box looks like `(x_min, y_min, x_max, y_max)`, e.g. (97, 12, 247, 212).
        The `yolo` format of a bounding box looks like `(x, y, width, height)`, e.g. (0.3, 0.1, 0.05, 0.07);
        where `x`, `y` coordinates of the center of the box, all values normalized to 1 by image height and width.

    Raises:
        ValueError: if `target_format` is not equal to `coco` or `pascal_voc`, ot `yolo`.
        ValueError: If in YOLO format all labels not in range (0, 1).

    """
    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            "Unknown source_format {}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'".format(source_format)
        )
    if isinstance(bbox, np.ndarray):
        bbox = bbox.tolist()

    if source_format == "coco":
        (x_min, y_min, width, height), tail = bbox[:4], tuple(bbox[4:])
        x_max = x_min + width
        y_max = y_min + height
    elif source_format == "yolo":
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/scripts/voc_label.py#L12
        bbox, tail = bbox[:4], tuple(bbox[4:])
        _bbox = np.array(bbox[:4])
        if check_validity and np.any((_bbox <= 0) | (_bbox > 1)):
            raise ValueError("In YOLO format all coordinates must be float and in range (0, 1]")

        x, y, w, h = bbox

        w_half, h_half = w / 2, h / 2
        x_min = x - w_half
        y_min = y - h_half
        x_max = x_min + w
        y_max = y_min + h
    else:
        (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    bbox = (x_min, y_min, x_max, y_max) + tail

    if source_format != "yolo":
        bbox = normalize_bbox(bbox, rows, cols)
    if check_validity:
        check_bbox(bbox)
    return bbox


def convert_bbox_from_albumentations(bbox, target_format, rows, cols, check_validity=False):
    """Convert a bounding box from the format used by albumentations to a format, specified in `target_format`.

    Args:
        bbox (tuple): An albumentation bounding box `(x_min, y_min, x_max, y_max)`.
        target_format (str): required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.
        rows (int): Image height.
        cols (int): Image width.
        check_validity (bool): Check if all boxes are valid boxes.

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
            "Unknown target_format {}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'".format(target_format)
        )
    if check_validity:
        check_bbox(bbox)

    if target_format != "yolo":
        bbox = denormalize_bbox(bbox, rows, cols)
    if target_format == "coco":
        (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])
        width = x_max - x_min
        height = y_max - y_min
        bbox = (x_min, y_min, width, height) + tail
    elif target_format == "yolo":
        (x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        bbox = (x, y, w, h) + tail
    return bbox


def convert_bboxes_to_albumentations(bboxes, source_format, rows, cols, check_validity=False):
    """Convert a list bounding boxes from a format specified in `source_format` to the format used by albumentations"""
    return [convert_bbox_to_albumentations(bbox, source_format, rows, cols, check_validity) for bbox in bboxes]


def convert_bboxes_from_albumentations(bboxes, target_format, rows, cols, check_validity=False):
    """Convert a list of bounding boxes from the format used by albumentations to a format, specified
    in `target_format`.

    Args:
        bboxes (List[tuple]): List of albumentation bounding box `(x_min, y_min, x_max, y_max)`.
        target_format (str): required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.
        rows (int): Image height.
        cols (int): Image width.
        check_validity (bool): Check if all boxes are valid boxes.

    Returns:
        list[tuple]: List of bounding box.

    """
    return [convert_bbox_from_albumentations(bbox, target_format, rows, cols, check_validity) for bbox in bboxes]


def check_bbox(bbox):
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
        if not 0 <= value <= 1 and not np.isclose(value, 0) and not np.isclose(value, 1):
            raise ValueError(
                "Expected {name} for bbox {bbox} "
                "to be in the range [0.0, 1.0], got {value}.".format(bbox=bbox, name=name, value=value)
            )
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max <= x_min:
        raise ValueError("x_max is less than or equal to x_min for bbox {bbox}.".format(bbox=bbox))
    if y_max <= y_min:
        raise ValueError("y_max is less than or equal to y_min for bbox {bbox}.".format(bbox=bbox))


def check_bboxes(bboxes):
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for bbox in bboxes:
        check_bbox(bbox)


def filter_bboxes(bboxes, rows, cols, min_area=0.0, min_visibility=0.0):
    """Remove bounding boxes that either lie outside of the visible area by more then min_visibility
    or whose area in pixels is under the threshold set by `min_area`. Also it crops boxes to final image size.

    Args:
        bboxes (List[tuple]): List of albumentation bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image height.
        cols (int): Image width.
        min_area (float): Minimum area of a bounding box. All bounding boxes whose visible area in pixels.
            is less than this value will be removed. Default: 0.0.
        min_visibility (float): Minimum fraction of area for a bounding box to remain this box in list. Default: 0.0.

    Returns:
        List[tuple]: List of bounding box.

    """
    resulting_boxes = []
    for bbox in bboxes:
        transformed_box_area = calculate_bbox_area(bbox, rows, cols)
        bbox, tail = tuple(np.clip(bbox[:4], 0, 1.0)), tuple(bbox[4:])
        clipped_box_area = calculate_bbox_area(bbox, rows, cols)
        if not transformed_box_area or clipped_box_area / transformed_box_area <= min_visibility:
            continue
        else:
            bbox = tuple(np.clip(bbox[:4], 0, 1.0))
        if calculate_bbox_area(bbox, rows, cols) <= min_area:
            continue
        resulting_boxes.append(bbox + tail)
    return resulting_boxes


def union_of_bboxes(height, width, bboxes, erosion_rate=0.0):
    """Calculate union of bounding boxes.

    Args:
        height (float): Height of image or space.
        width (float): Width of image or space.
        bboxes (List[tuple]): List like bounding boxes. Format is `[(x_min, y_min, x_max, y_max)]`.
        erosion_rate (float): How much each bounding box can be shrinked, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x1, y1 = width, height
    x2, y2 = 0, 0
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox[:4]
        w, h = x_max - x_min, y_max - y_min
        lim_x1, lim_y1 = x_min + erosion_rate * w, y_min + erosion_rate * h
        lim_x2, lim_y2 = x_max - erosion_rate * w, y_max - erosion_rate * h
        x1, y1 = np.min([x1, lim_x1]), np.min([y1, lim_y1])
        x2, y2 = np.max([x2, lim_x2]), np.max([y2, lim_y2])
    return x1, y1, x2, y2
