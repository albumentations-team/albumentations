import numpy as np

from typing import List, Union, Tuple

from ..bbox_utils import denormalize_bbox, normalize_bbox

BboxType = Union[List[int], List[float], Tuple[int, ...], Tuple[float, ...], np.ndarray]
KeypointType = Union[List[int], List[float], Tuple[int, ...], Tuple[float, ...], np.ndarray]


def get_random_crop_coords(height: int, width: int, crop_height: int, crop_width: int, h_start: float, w_start: float):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(img: np.ndarray, crop_height: int, crop_width: int, h_start: float, w_start: float):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img


def crop_bbox_by_coords(
    bbox: BboxType, crop_coords: Tuple[int, int, int, int], crop_height: int, crop_width: int, rows: int, cols: int
):
    """Crop a bounding box using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.

    Args:
        bbox (tuple): A cropped box `(x_min, y_min, x_max, y_max)`.
        crop_coords (tuple): Crop coordinates `(x1, y1, x2, y2)`.
        crop_height (int):
        crop_width (int):
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        tuple: A cropped bounding box `(x_min, y_min, x_max, y_max)`.

    """
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox[:4]
    x1, y1, _, _ = crop_coords
    cropped_bbox = x_min - x1, y_min - y1, x_max - x1, y_max - y1
    return normalize_bbox(cropped_bbox, crop_height, crop_width)


def bbox_random_crop(
    bbox: BboxType, crop_height: int, crop_width: int, h_start: float, w_start: float, rows: int, cols: int
):
    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def crop_keypoint_by_coords(keypoint: KeypointType, crop_coords: Tuple[int, int, int, int]):  # skipcq: PYL-W0613
    """Crop a keypoint using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        crop_coords (tuple): Crop box coords `(x1, x2, y1, y2)`.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    x1, y1, _, _ = crop_coords
    return x - x1, y - y1, angle, scale


def keypoint_random_crop(
    keypoint: KeypointType, crop_height: int, crop_width: int, h_start: float, w_start: float, rows: int, cols: int
):
    """Keypoint random crop.

    Args:
        keypoint: (tuple): A keypoint `(x, y, angle, scale)`.
        crop_height (int): Crop height.
        crop_width (int): Crop width.
        h_start (int): Crop height start.
        w_start (int): Crop width start.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    return crop_keypoint_by_coords(keypoint, crop_coords)


def get_center_crop_coords(height: int, width: int, crop_height: int, crop_width: int):
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def center_crop(img: np.ndarray, crop_height: int, crop_width: int):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    img = img[y1:y2, x1:x2]
    return img


def bbox_center_crop(bbox: BboxType, crop_height: int, crop_width: int, rows: int, cols: int):
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def keypoint_center_crop(keypoint: KeypointType, crop_height: int, crop_width: int, rows: int, cols: int):
    """Keypoint center crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        crop_height (int): Crop height.
        crop_width (int): Crop width.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_keypoint_by_coords(keypoint, crop_coords)


def crop(img: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int):
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            "(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            "height = {height}, width = {width})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, height=height, width=width
            )
        )

    return img[y_min:y_max, x_min:x_max]


def bbox_crop(bbox: BboxType, x_min: int, y_min: int, x_max: int, y_max: int, rows: int, cols: int):
    """Crop a bounding box.

    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        x_min (int):
        y_min (int):
        x_max (int):
        y_max (int):
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        tuple: A cropped bounding box `(x_min, y_min, x_max, y_max)`.

    """
    crop_coords = x_min, y_min, x_max, y_max
    crop_height = y_max - y_min
    crop_width = x_max - x_min
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def clamping_crop(img: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int):
    h, w = img.shape[:2]
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if y_max >= h:
        y_max = h - 1
    if x_max >= w:
        x_max = w - 1
    return img[int(y_min) : int(y_max), int(x_min) : int(x_max)]
