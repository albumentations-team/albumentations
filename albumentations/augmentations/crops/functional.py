from typing import Optional, Sequence, Tuple, cast

import cv2
import numpy as np
from albucore.utils import maybe_process_in_chunks, preserve_channel_dim

from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox
from albumentations.core.types import BoxInternalType, KeypointInternalType

__all__ = [
    "get_random_crop_coords",
    "random_crop",
    "crop_bbox_by_coords",
    "bbox_random_crop",
    "crop_keypoint_by_coords",
    "keypoint_random_crop",
    "get_center_crop_coords",
    "center_crop",
    "bbox_center_crop",
    "keypoint_center_crop",
    "crop",
    "bbox_crop",
    "clamping_crop",
    "crop_and_pad",
    "crop_and_pad_bbox",
    "crop_and_pad_keypoint",
]


def get_random_crop_coords(
    height: int,
    width: int,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
) -> Tuple[int, int, int, int]:
    # h_start is [0, 1) and should map to [0, (height - crop_height)]  (note inclusive)
    # This is conceptually equivalent to mapping onto `range(0, (height - crop_height + 1))`
    # See: https://github.com/albumentations-team/albumentations/pull/1080
    y1 = int((height - crop_height + 1) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width + 1) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(img: np.ndarray, crop_height: int, crop_width: int, h_start: float, w_start: float) -> np.ndarray:
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            f"Requested crop size ({crop_height}, {crop_width}) is larger than the image size ({height}, {width})",
        )
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    return img[y1:y2, x1:x2]


def crop_bbox_by_coords(
    bbox: BoxInternalType,
    crop_coords: Tuple[int, int, int, int],
    crop_height: int,
    crop_width: int,
    rows: int,
    cols: int,
) -> BoxInternalType:
    """Crop a bounding box using the provided coordinates of the crop area and the original image dimensions.

    Args:
        bbox (BoxInternalType): A bounding box in the format `(x_min, y_min, x_max, y_max)`.
        crop_coords (Tuple[int, int, int, int]): The coordinates of the crop area in the format `(x1, y1, x2, y2)`.
        crop_height (int): The height of the crop area in pixels.
        crop_width (int): The width of the crop area in pixels.
        rows (int): The number of rows (height) in the original image.
        cols (int): The number of columns (width) in the original image.

    Returns:
        BoxInternalType: A cropped bounding box in the format `(x_min, y_min, x_max, y_max)`.
    """
    normalized_bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = normalized_bbox[:4]
    x1, y1 = crop_coords[:2]
    cropped_bbox = x_min - x1, y_min - y1, x_max - x1, y_max - y1
    return cast(BoxInternalType, normalize_bbox(cropped_bbox, crop_height, crop_width))


def bbox_random_crop(
    bbox: BoxInternalType,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
    rows: int,
    cols: int,
) -> BoxInternalType:
    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def crop_keypoint_by_coords(
    keypoint: KeypointInternalType,
    crop_coords: Tuple[int, int, int, int],
) -> KeypointInternalType:
    """Crop a keypoint using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        crop_coords (tuple): Crop box coords `(x1, x2, y1, y2)`.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    x1, y1 = crop_coords[:2]
    return x - x1, y - y1, angle, scale


def keypoint_random_crop(
    keypoint: KeypointInternalType,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
    rows: int,
    cols: int,
) -> KeypointInternalType:
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


def get_center_crop_coords(height: int, width: int, crop_height: int, crop_width: int) -> Tuple[int, int, int, int]:
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def center_crop(img: np.ndarray, crop_height: int, crop_width: int) -> np.ndarray:
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            f"Requested crop size ({crop_height}, {crop_width}) is larger than the image size ({height}, {width})",
        )
    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    return img[y1:y2, x1:x2]


def bbox_center_crop(bbox: BoxInternalType, crop_height: int, crop_width: int, rows: int, cols: int) -> BoxInternalType:
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def keypoint_center_crop(
    keypoint: KeypointInternalType,
    crop_height: int,
    crop_width: int,
    rows: int,
    cols: int,
) -> KeypointInternalType:
    """Keypoint center crop.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        crop_height: Crop height.
        crop_width: Crop width.
        rows: Image height.
        cols: Image width.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_keypoint_by_coords(keypoint, crop_coords)


def crop(img: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int) -> np.ndarray:
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            f" (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})",
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            f"(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            f"height = {height}, width = {width})",
        )

    return img[y_min:y_max, x_min:x_max]


def bbox_crop(
    bbox: BoxInternalType,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    rows: int,
    cols: int,
) -> BoxInternalType:
    """Crop a bounding box using the provided coordinates.

    Args:
        bbox (BoxInternalType): A bounding box in the format `(x_min, y_min, x_max, y_max)`.
        x_min (int): The minimum x-coordinate of the crop area.
        y_min (int): The minimum y-coordinate of the crop area.
        x_max (int): The maximum x-coordinate of the crop area.
        y_max (int): The maximum y-coordinate of the crop area.
        rows (int): The number of rows (height) in the original image.
        cols (int): The number of columns (width) in the original image.

    Returns:
        BoxInternalType: A cropped bounding box in the format `(x_min, y_min, x_max, y_max)`.
    """
    crop_coords = x_min, y_min, x_max, y_max
    crop_height = y_max - y_min
    crop_width = x_max - x_min
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def clamping_crop(img: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int) -> np.ndarray:
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


@preserve_channel_dim
def crop_and_pad(
    img: np.ndarray,
    crop_params: Optional[Sequence[int]],
    pad_params: Optional[Sequence[int]],
    pad_value: Optional[float],
    rows: int,
    cols: int,
    interpolation: int,
    pad_mode: int,
    keep_size: bool,
) -> np.ndarray:
    if crop_params is not None and any(i != 0 for i in crop_params):
        img = crop(img, *crop_params)
    if pad_params is not None and any(i != 0 for i in pad_params):
        img = fgeometric.pad_with_params(
            img,
            pad_params[0],
            pad_params[1],
            pad_params[2],
            pad_params[3],
            border_mode=pad_mode,
            value=pad_value,
        )

    if keep_size:
        resize_fn = maybe_process_in_chunks(cv2.resize, dsize=(cols, rows), interpolation=interpolation)
        return resize_fn(img)

    return img


def crop_and_pad_bbox(
    bbox: BoxInternalType,
    crop_params: Optional[Sequence[int]],
    pad_params: Optional[Sequence[int]],
    rows: int,
    cols: int,
    result_rows: int,
    result_cols: int,
) -> BoxInternalType:
    x1, y1, x2, y2 = denormalize_bbox(bbox, rows, cols)[:4]

    if crop_params is not None:
        crop_x, crop_y = crop_params[:2]

        x1 -= crop_x
        y1 -= crop_y
        x2 -= crop_x
        y2 -= crop_y

    if pad_params is not None:
        top = pad_params[0]
        left = pad_params[2]

        x1 += left
        y1 += top
        x2 += left
        y2 += top

    return cast(BoxInternalType, normalize_bbox((x1, y1, x2, y2), result_rows, result_cols))


def crop_and_pad_keypoint(
    keypoint: KeypointInternalType,
    crop_params: Optional[Sequence[int]],
    pad_params: Optional[Sequence[int]],
    rows: int,
    cols: int,
    result_rows: int,
    result_cols: int,
    keep_size: bool,
) -> KeypointInternalType:
    x, y, angle, scale = keypoint[:4]

    if crop_params is not None:
        crop_x1, crop_y1 = crop_params[:2]
        x, y = x - crop_x1, y - crop_y1
    if pad_params is not None:
        top = pad_params[0]
        left = pad_params[2]

        x += left
        y += top

    if keep_size and (result_cols != cols or result_rows != rows):
        scale_x = cols / result_cols
        scale_y = rows / result_rows
        return fgeometric.keypoint_scale((x, y, angle, scale), scale_x, scale_y)

    return x, y, angle, scale
