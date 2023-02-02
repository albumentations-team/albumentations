from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from albumentations.augmentations.utils import (
    _maybe_process_in_chunks,
    preserve_channel_dim,
)

from ...core.bbox_utils import denormalize_bboxes_np, normalize_bboxes_np
from ...core.transforms_interface import KeypointInternalType
from ..geometric import functional as FGeometric

__all__ = [
    "get_random_crop_coords",
    "random_crop",
    "crop_bboxes_by_coords",
    "bboxes_random_crop",
    "crop_keypoint_by_coords",
    "keypoint_random_crop",
    "get_center_crop_coords",
    "center_crop",
    "bboxes_center_crop",
    "keypoint_center_crop",
    "crop",
    "bboxes_crop",
    "clamping_crop",
    "crop_and_pad",
    "crop_and_pad_bboxes",
    "crop_and_pad_keypoint",
]


def get_random_crop_coords(height: int, width: int, crop_height: int, crop_width: int, h_start: float, w_start: float):
    # h_start is [0, 1) and should map to [0, (height - crop_height)]  (note inclusive)
    # This is conceptually equivalent to mapping onto `range(0, (height - crop_height + 1))`
    # See: https://github.com/albumentations-team/albumentations/pull/1080
    y1 = int((height - crop_height + 1) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width + 1) * w_start)
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


def crop_bboxes_by_coords(
    bboxes: np.ndarray,
    crop_coords: Union[Sequence[Tuple[int, int, int, int]], np.ndarray],
    crop_height: Union[Sequence[int], int, np.ndarray],
    crop_width: Union[Sequence[int], int, np.ndarray],
    rows: int,
    cols: int,
) -> np.ndarray:
    """Crop a batch of bounding boxes using the provided coordinates in pixels and the
    required height and width of the crop.

    Args:
        bboxes (numpy.ndarray): A batch of bounding boxes in `albumentations` format.
        crop_coords (list[tuple[int, int, int, int]]): A sequence of coordinates.
        crop_height (Sequence[int] | int):
        crop_width (Sequence[int] | int):
        rows (int): Image rows.
        cols (int): Image columns.

    Returns:
        numpy.ndarray: A batch of cropped bounding boxes with `albumentations` format.

    """

    np_bboxes = denormalize_bboxes_np(bboxes, rows, cols)
    crop_coords = np.tile(np.array(crop_coords)[:, :2], 2)
    cropped_bboxes = np_bboxes - crop_coords
    return normalize_bboxes_np(cropped_bboxes, crop_width, crop_height)


def bboxes_random_crop(
    bboxes: np.ndarray,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
    rows: int,
    cols: int,
) -> np.ndarray:
    num_bboxes = len(bboxes)
    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    return crop_bboxes_by_coords(bboxes, [crop_coords] * num_bboxes, crop_height, crop_width, rows, cols)


def crop_keypoint_by_coords(
    keypoint: KeypointInternalType, crop_coords: Tuple[int, int, int, int]
):  # skipcq: PYL-W0613
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
    keypoint: KeypointInternalType,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
    rows: int,
    cols: int,
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


def bboxes_center_crop(bboxes: np.ndarray, crop_height: int, crop_width: int, rows: int, cols: int):
    num_bboxes = len(bboxes)
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_bboxes_by_coords(bboxes, [crop_coords] * num_bboxes, crop_height, crop_width, rows, cols)


def keypoint_center_crop(keypoint: KeypointInternalType, crop_height: int, crop_width: int, rows: int, cols: int):
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


def bboxes_crop(
    bboxes: np.ndarray,
    x_min: Union[np.ndarray, int],
    y_min: Union[np.ndarray, int],
    x_max: Union[np.ndarray, int],
    y_max: Union[np.ndarray, int],
    rows: int,
    cols: int,
) -> np.ndarray:
    """Crop a batch of bounding boxes in `albumentations` format.

    Args:
        bboxes (numpy.ndarray): A batch of bounding boxes with `albumentations` format.
        x_min (numpy.ndarray | int):
        y_min (numpy.ndarray | int):
        x_max (numpy.ndarray | int):
        y_max (numpy.ndarray | int):
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        numpy.ndarray: A batch of cropped bounding boxes in `albumentations` format.

    """

    assert (
        isinstance(x_min, np.ndarray)
        and isinstance(y_min, np.ndarray)
        and isinstance(x_max, np.ndarray)
        and isinstance(y_max, np.ndarray)
    ) or (isinstance(x_min, int) and isinstance(y_min, int) and isinstance(x_max, int) and isinstance(y_max, int))
    if isinstance(x_min, np.ndarray):
        crop_coords = np.stack([x_min, y_min, x_max, y_max], axis=1)
    else:
        len_bboxes = len(bboxes)
        x_mins: Sequence[int] = [x_min] * len_bboxes
        y_mins: Sequence[int] = [x_min] * len_bboxes
        x_maxs: Sequence[int] = [x_min] * len_bboxes
        y_maxs: Sequence[int] = [x_min] * len_bboxes
        crop_coords = np.stack(
            [x_mins, y_mins, x_maxs, y_maxs],
            axis=1,
        )
    crop_heights = y_max - y_min
    crop_widths = x_max - x_min

    return crop_bboxes_by_coords(bboxes, crop_coords, crop_heights, crop_widths, rows=rows, cols=cols)


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
        img = FGeometric.pad_with_params(
            img, pad_params[0], pad_params[1], pad_params[2], pad_params[3], border_mode=pad_mode, value=pad_value
        )

    if keep_size:
        resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(cols, rows), interpolation=interpolation)
        img = resize_fn(img)

    return img


def crop_and_pad_bboxes(
    bboxes: np.ndarray,
    crop_params: Optional[Sequence[int]],
    pad_params: Optional[Sequence[int]],
    rows: int,
    cols: int,
    result_rows: int,
    result_cols: int,
) -> np.ndarray:

    bboxes = denormalize_bboxes_np(bboxes, rows, cols)

    if crop_params is not None:
        crop_x, crop_y = crop_params[:2]
        bboxes -= np.array([crop_x, crop_y, crop_x, crop_y])
    if pad_params is not None:
        top, bottom, left, right = pad_params
        bboxes += np.array([left, top, left, top])

    return normalize_bboxes_np(bboxes, result_rows, result_cols)


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
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_params
        x, y = x - crop_x1, y - crop_y1
    if pad_params is not None:
        top, bottom, left, right = pad_params
        x, y = x + left, y + top

    if keep_size and (result_cols != cols or result_rows != rows):
        scale_x = cols / result_cols
        scale_y = rows / result_rows
        return FGeometric.keypoint_scale((x, y, angle, scale), scale_x, scale_y)

    return x, y, angle, scale
