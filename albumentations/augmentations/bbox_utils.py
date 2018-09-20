from __future__ import division

import numpy as np

__all__ = ['normalize_bbox', 'denormalize_bbox', 'normalize_bboxes', 'denormalize_bboxes', 'calculate_bbox_area',
           'filter_bboxes_by_visibility', 'convert_bbox_to_albumentations', 'convert_bbox_from_albumentations',
           'convert_bboxes_to_albumentations', 'convert_bboxes_from_albumentations']


def normalize_bbox(bbox, rows, cols):
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    normalized_bbox = [x_min / cols, y_min / rows, x_max / cols, y_max / rows]
    return normalized_bbox + list(bbox[4:])


def denormalize_bbox(bbox, rows, cols):
    """Denormalize coordinates of a bounding box. Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bbox`.
    """
    x_min, y_min, x_max, y_max = bbox[:4]
    denormalized_bbox = [x_min * cols, y_min * rows, x_max * cols, y_max * rows]
    return denormalized_bbox + list(bbox[4:])


def normalize_bboxes(bboxes, rows, cols):
    """Normalize a list of bounding boxes."""
    return [normalize_bbox(bbox, rows, cols) for bbox in bboxes]


def denormalize_bboxes(bboxes, rows, cols):
    """Denormalize a list of bounding boxes."""
    return [denormalize_bbox(bbox, rows, cols) for bbox in bboxes]


def calculate_bbox_area(bbox, rows, cols):
    """Calculate the area of a bounding box in pixels."""
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox[:4]
    area = (x_max - x_min) * (y_max - y_min)
    return area


def filter_bboxes_by_visibility(original_shape, bboxes, transformed_shape, transformed_bboxes,
                                threshold=0., min_area=0.):
    """Filter bounding boxes and return only those boxes whose visibility after transformation is above
    the threshold and minimal area of bounding box in pixels is more then min_area.

    Args:
        original_shape (tuple): original image shape
        bboxes (list): original bounding boxes
        transformed_shape(tuple): transformed image
        transformed_bboxes (list): transformed bounding boxes
        threshold (float): visibility threshold. Should be a value in the range [0.0, 1.0].
        min_area (float): Minimal area threshold.
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
    normalized coordinates of bottom-left and top-right corners of the bounding box in a form of
    `[x_min, y_min, x_max, y_max]` e.g. `[0.15, 0.27, 0.67, 0.5]`.

    Args:
        bbox (list): bounding box
        source_format (str): format of the bounding box. Should be 'coco' or 'pascal_voc'.
        check_validity (bool): check if all boxes are valid boxes
        rows (int): image height
        cols (int): image width

    Note:
        The `coco` format of a bounding box looks like `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
        The `pascal_voc` format of a bounding box looks like `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].

    Raises:
        ValueError: if `target_format` is not equal to `coco` or `pascal_voc`.

    """
    if source_format not in {'coco', 'pascal_voc'}:
        raise ValueError(
            "Unknown source_format {}. Supported formats are: 'coco' and 'pascal_voc'".format(source_format)
        )
    if source_format == 'coco':
        x_min, y_min, width, height = bbox[:4]
        x_max = x_min + width
        y_max = y_min + height
    else:
        x_min, y_min, x_max, y_max = bbox[:4]
    bbox = [x_min, y_min, x_max, y_max] + list(bbox[4:])
    bbox = normalize_bbox(bbox, rows, cols)
    if check_validity:
        check_bbox(bbox)
    return bbox


def convert_bbox_from_albumentations(bbox, target_format, rows, cols, check_validity=False):
    """Convert a bounding box from the format used by albumentations to a format, specified in `target_format`.

    Args:
        shape (tuple): input image shape. Image must have at least 2 dims
        bbox (list): bounding box with coordinates in the format used by albumentations
        target_format (str): required format of the output bounding box. Should be 'coco' or 'pascal_voc'.
        check_validity (bool): check if all boxes are valid boxes
        rows (int): image height
        cols (int): image width

    Note:
        The `coco` format of a bounding box looks like `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
        The `pascal_voc` format of a bounding box looks like `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].

    Raises:
        ValueError: if `target_format` is not equal to `coco` or `pascal_voc`.

    """
    if target_format not in {'coco', 'pascal_voc'}:
        raise ValueError(
            "Unknown target_format {}. Supported formats are: 'coco' and 'pascal_voc'".format(target_format)
        )
    if check_validity:
        check_bbox(bbox)
    bbox = denormalize_bbox(bbox, rows, cols)
    if target_format == 'coco':
        x_min, y_min, x_max, y_max = bbox[:4]
        width = x_max - x_min
        height = y_max - y_min
        bbox = [x_min, y_min, width, height] + list(bbox[4:])
    return bbox


def convert_bboxes_to_albumentations(bboxes, source_format, rows, cols, check_validity=False):
    """Convert a list bounding boxes from a format specified in `source_format` to the format used by albumentations
    """
    return [convert_bbox_to_albumentations(bbox, source_format, rows, cols, check_validity) for bbox in bboxes]


def convert_bboxes_from_albumentations(bboxes, target_format, rows, cols, check_validity=False):
    """Convert a list of bounding boxes from the format used by albumentations to a format, specified
    in `target_format`.

    Args:
        shape (tuple): input image shape. Image must have at least 2 dims
        bboxes (list): List of bounding box with coordinates in the format used by albumentations
        target_format (str): required format of the output bounding box. Should be 'coco' or 'pascal_voc'.
    """
    return [convert_bbox_from_albumentations(bbox, target_format, rows, cols, check_validity) for bbox in bboxes]


def check_bbox(bbox):
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for name, value in zip(['x_min', 'y_min', 'x_max', 'y_max'], bbox[:4]):
        if not 0 <= value <= 1:
            raise ValueError(
                'Expected {name} for bbox {bbox} '
                'to be in the range [0.0, 1.0], got {value}.'.format(
                    bbox=bbox,
                    name=name,
                    value=value,
                )
            )
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max <= x_min:
        raise ValueError('x_max is less than or equal to x_min for bbox {bbox}.'.format(
            bbox=bbox,
        ))
    if y_max <= y_min:
        raise ValueError('y_max is less than or equal to y_min for bbox {bbox}.'.format(
            bbox=bbox,
        ))


def check_bboxes(bboxes):
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for bbox in bboxes:
        check_bbox(bbox)


def filter_bboxes(bboxes, rows, cols, min_area=0., min_visibility=0.):
    """Remove bounding boxes that either lie outside of the visible area by more then min_visibility
    or whose area in pixels is under the threshold set by `min_area`. Also it crops boxes to final image size.

    Args:
        min_area (float): minimum area of a bounding box. All bounding boxes whose visible area in pixels
            is less than this value will be removed. Default: 0.0.
        min_visibility (float): minimum fraction of area for a bounding box to remain this box in list. Default: 0.0.
        rows (int): Image rows.
        rows (int): Image cols.
    """
    resulting_boxes = []
    for bbox in bboxes:
        transformed_box_area = calculate_bbox_area(bbox, rows, cols)
        bbox[:4] = np.clip(bbox[:4], 0, 1.)
        clipped_box_area = calculate_bbox_area(bbox, rows, cols)
        if not transformed_box_area or clipped_box_area / transformed_box_area <= min_visibility:
            continue
        else:
            bbox[:4] = np.clip(bbox[:4], 0, 1.)
        if calculate_bbox_area(bbox, rows, cols) <= min_area:
            continue
        resulting_boxes.append(bbox)
    return resulting_boxes
