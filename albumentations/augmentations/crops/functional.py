from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from albumentations.augmentations.utils import (
    _maybe_process_in_chunks,
    preserve_channel_dim,
)

from ...core.bbox_utils import (
    denormalize_bboxes_np,
    ensure_bboxes_format,
    normalize_bboxes_np,
    use_bboxes_ndarray,
)
from ...core.keypoints_utils import ensure_keypoints_format, use_keypoints_ndarray
from ...core.transforms_interface import (
    BBoxesInternalType,
    BoxesArray,
    KeypointsArray,
    KeypointsInternalType,
)
from ..geometric import functional as FGeometric

__all__ = [
    "get_random_crop_coords",
    "random_crop",
    "crop_bboxes_by_coords",
    "bboxes_random_crop",
    "crop_keypoints_by_coords",
    "keypoints_random_crop",
    "get_center_crop_coords",
    "center_crop",
    "bboxes_center_crop",
    "keypoints_center_crop",
    "crop",
    "bboxes_crop",
    "clamping_crop",
    "crop_and_pad",
    "crop_and_pad_bboxes",
    "crop_and_pad_keypoints",
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


@use_bboxes_ndarray(return_array=True)
def crop_bboxes_by_coords(
    bboxes: BoxesArray,
    crop_coords: Union[Sequence[Tuple[int, int, int, int]], np.ndarray],
    crop_height: Union[Sequence[int], int, np.ndarray],
    crop_width: Union[Sequence[int], int, np.ndarray],
    rows: int,
    cols: int,
) -> BoxesArray:
    """Crop a batch of bounding boxes using the provided coordinates in pixels and the
    required height and width of the crop.

    Args:
        bboxes (BoxesArray): A batch of bounding boxes in `albumentations` format.
        crop_coords (list[tuple[int, int, int, int]]): A sequence of coordinates.
        crop_height (Sequence[int] | int):
        crop_width (Sequence[int] | int):
        rows (int): Image rows.
        cols (int): Image columns.

    Returns:
        BoxesArray: A batch of cropped bounding boxes with `albumentations` format.

    """
    if not len(bboxes):
        return bboxes
    np_bboxes = denormalize_bboxes_np(bboxes, rows, cols)
    crop_coords = np.tile(np.array(crop_coords)[:, :2], 2)
    cropped_bboxes = np_bboxes - crop_coords
    return normalize_bboxes_np(cropped_bboxes, crop_width, crop_height)


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def bboxes_random_crop(
    bboxes: BoxesArray,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
    rows: int,
    cols: int,
) -> BoxesArray:
    num_bboxes = len(bboxes)
    if not num_bboxes:
        return bboxes
    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    return crop_bboxes_by_coords(bboxes, [crop_coords] * num_bboxes, crop_height, crop_width, rows, cols)


@use_keypoints_ndarray(return_array=True)
def crop_keypoints_by_coords(
    keypoints: KeypointsArray,
    crop_coords: np.ndarray,
) -> KeypointsArray:
    """Crop a batch of keypoints using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.

    Args:
        keypoints (KeypointsArray): A batch of keypoints in `(x, y, angle, scale)` format.
        crop_coords (np.ndarray): Crop box coords `(x1, x2, y1, y2)`.

    Returns:
        KeypointsArray, A batch of keypoints in `(x, y, angle, scale)` format.

    """

    if not len(keypoints):
        return keypoints
    keypoints[..., [0, 1]] -= crop_coords[:2]
    return keypoints


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
def keypoints_random_crop(
    keypoints: KeypointsArray,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
    rows: int,
    cols: int,
) -> KeypointsArray:
    """
    Keypoints random crop.
    Args:
        keypoints (KeypointsArray): A batch of keypoints in `x, y, angle, scale` format.
        crop_height (int): Crop height.
        crop_width (int): Crop width.
        h_start (float): Crop height start.
        w_start (float): Crop width start.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        KeypointsArray, A batch of keypoints in `x, y, angle, scale` format.

    """
    if not len(keypoints):
        return keypoints

    crop_coords = get_random_crop_coords(rows, cols, crop_height, crop_width, h_start, w_start)
    return crop_keypoints_by_coords(keypoints, np.array(crop_coords))


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


@use_bboxes_ndarray(return_array=True)
def bboxes_center_crop(bboxes: BoxesArray, crop_height: int, crop_width: int, rows: int, cols: int) -> BoxesArray:
    num_bboxes = len(bboxes)
    if not num_bboxes:
        return bboxes
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_bboxes_by_coords(bboxes, [crop_coords] * num_bboxes, crop_height, crop_width, rows, cols)


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
def keypoints_center_crop(
    keypoints: KeypointsArray,
    crop_height: int,
    crop_width: int,
    rows: int,
    cols: int,
) -> KeypointsArray:
    """Keypoints center crop.

    Args:
        keypoints (KeypointsArray): A batch of keypoints in `x, y, angle, scale` format.
        crop_height (int): Crop height.
        crop_width (int): Crop width.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        KeypointsArray, A batch of keypoints in `x, y, angle, scale` format.
    """
    if not len(keypoints):
        return keypoints
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_keypoints_by_coords(keypoints, np.array(crop_coords))


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


@ensure_bboxes_format
def bboxes_crop(
    bboxes: BBoxesInternalType,
    x_min: Union[np.ndarray, int],
    y_min: Union[np.ndarray, int],
    x_max: Union[np.ndarray, int],
    y_max: Union[np.ndarray, int],
    rows: int,
    cols: int,
) -> BBoxesInternalType:
    """Crop a batch of bounding boxes in `albumentations` format.

    Args:
        bboxes (BBoxesInternalType): A batch of bounding boxes with `albumentations` format.
        x_min (numpy.ndarray | int):
        y_min (numpy.ndarray | int):
        x_max (numpy.ndarray | int):
        y_max (numpy.ndarray | int):
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        BBoxesInternalType: A batch of cropped bounding boxes in `albumentations` format.

    """
    if not len(bboxes):
        return bboxes

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


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def crop_and_pad_bboxes(
    bboxes: BoxesArray,
    crop_params: Optional[Sequence[int]],
    pad_params: Optional[Sequence[int]],
    rows: int,
    cols: int,
    result_rows: int,
    result_cols: int,
) -> BoxesArray:
    if not len(bboxes):
        return bboxes

    bboxes = denormalize_bboxes_np(bboxes, rows, cols)

    if crop_params is not None:
        crop_x, crop_y = crop_params[:2]
        bboxes -= np.array([crop_x, crop_y, crop_x, crop_y])
    if pad_params is not None:
        top, bottom, left, right = pad_params
        bboxes += np.array([left, top, left, top])

    return normalize_bboxes_np(bboxes, result_rows, result_cols)


@ensure_keypoints_format
@use_keypoints_ndarray(return_array=True)
def crop_and_pad_keypoints(
    keypoints: KeypointsArray,
    crop_params: Optional[Sequence[int]],
    pad_params: Optional[Sequence[int]],
    rows: int,
    cols: int,
    result_rows: int,
    result_cols: int,
    keep_size: bool,
) -> KeypointsArray:

    if crop_params is not None:
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_params
        keypoints[..., :2] -= np.array([crop_x1, crop_y1])
    if pad_params is not None:
        top, bottom, left, right = pad_params
        keypoints[..., :2] += np.array([left, top])

    if keep_size and (result_cols != cols or result_rows != rows):
        scale_x = cols / result_cols
        scale_y = rows / result_rows
        return FGeometric.keypoints_scale(keypoints, scale_x, scale_y)

    return keypoints
