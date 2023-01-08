from __future__ import division

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np

from .transforms_interface import BoxInternalType, BoxType
from .utils import DataProcessor, Params

__all__ = [
    "normalize_bbox",
    "denormalize_bbox",
    "normalize_bboxes",
    "normalize_bboxes_np",
    "denormalize_bboxes",
    "denormalize_bboxes_np",
    "calculate_bbox_area",
    "calculate_bboxes_area",
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
    "bboxes_to_array",
    "array_to_bboxes",
    "assert_np_bboxes_format",
]

TBox = TypeVar("TBox", BoxType, BoxInternalType)


def assert_np_bboxes_format(bboxes: np.ndarray):
    assert (
        isinstance(bboxes, np.ndarray) and len(bboxes.shape) == 2 and bboxes.shape[-1] == 4
    ), "An array of bboxes should be 2 dimension, and the last dimension must has 4 elements."


def bboxes_to_array(bboxes: Sequence[BoxType]) -> np.ndarray:
    return np.array([bbox[:4] for bbox in bboxes])


def array_to_bboxes(np_bboxes: np.ndarray, ori_bboxes: Sequence[BoxType]) -> List[BoxType]:
    return [cast(BoxType, tuple(np_bbox) + tuple(bbox[4:])) for bbox, np_bbox in zip(ori_bboxes, np_bboxes)]


class BboxParams(Params):
    """
    Parameters of bounding boxes

    Args:
        format (str): format of bounding boxes. Should be 'coco', 'pascal_voc', 'albumentations' or 'yolo'.

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
        label_fields (list): list of fields that are joined with boxes, e.g labels.
            Should be same type as boxes.
        min_area (float): minimum area of a bounding box. All bounding boxes whose
            visible area in pixels is less than this value will be removed. Default: 0.0.
        min_visibility (float): minimum fraction of area for a bounding box
            to remain this box in list. Default: 0.0.
        min_width (float): Minimum width of a bounding box. All bounding boxes whose width is
            less than this value will be removed. Default: 0.0.
        min_height (float): Minimum height of a bounding box. All bounding boxes whose height is
            less than this value will be removed. Default: 0.0.
        check_each_transform (bool): if `True`, then bboxes will be checked after each dual transform.
            Default: `True`
    """

    def __init__(
        self,
        format: str,
        label_fields: Optional[Sequence[str]] = None,
        min_area: float = 0.0,
        min_visibility: float = 0.0,
        min_width: float = 0.0,
        min_height: float = 0.0,
        check_each_transform: bool = True,
    ):
        super(BboxParams, self).__init__(format, label_fields)
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.min_width = min_width
        self.min_height = min_height
        self.check_each_transform = check_each_transform

    def _to_dict(self) -> Dict[str, Any]:
        data = super(BboxParams, self)._to_dict()
        data.update(
            {
                "min_area": self.min_area,
                "min_visibility": self.min_visibility,
                "min_width": self.min_width,
                "min_height": self.min_height,
                "check_each_transform": self.check_each_transform,
            }
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
            if data_exists and len(data[data_name][0]) < 5:
                if self.params.label_fields is None:
                    raise ValueError(
                        "Please specify 'label_fields' in 'bbox_params' or add labels to the end of bbox "
                        "because bboxes must have labels"
                    )
        if self.params.label_fields:
            if not all(i in data.keys() for i in self.params.label_fields):
                raise ValueError("Your 'label_fields' are not valid - them must have same names as params in dict")

    def filter(self, data: Sequence, rows: int, cols: int) -> List:
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

    def check(self, data: Sequence, rows: int, cols: int) -> None:
        check_bboxes(data)

    def convert_from_albumentations(self, data: Sequence, rows: int, cols: int) -> List[BoxType]:
        return convert_bboxes_from_albumentations(data, self.params.format, rows, cols, check_validity=True)

    def convert_to_albumentations(self, data: Sequence[BoxType], rows: int, cols: int) -> List[BoxType]:
        return convert_bboxes_to_albumentations(data, self.params.format, rows, cols, check_validity=True)


def normalize_bbox(bbox: TBox, rows: int, cols: int) -> TBox:
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
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")

    tail: Tuple[Any, ...]
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    x_min, x_max = x_min / cols, x_max / cols
    y_min, y_max = y_min / rows, y_max / rows

    return cast(BoxType, (x_min, y_min, x_max, y_max) + tail)  # type: ignore


def denormalize_bbox(bbox: TBox, rows: int, cols: int) -> TBox:
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
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")

    x_min, x_max = x_min * cols, x_max * cols
    y_min, y_max = y_min * rows, y_max * rows

    return cast(BoxType, (x_min, y_min, x_max, y_max) + tail)  # type: ignore


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


def _convert_to_array(dim: Union[Sequence[int], np.ndarray], length: int, dim_name: str):
    if not isinstance(dim, np.ndarray):
        dim = np.array(
            [
                dim,
            ]
        ).transpose()
    elif isinstance(dim, np.ndarray) and len(dim.shape) == 1:
        dim = np.expand_dims(dim, axis=0).transpose()
    assert isinstance(dim, np.ndarray) and dim.shape[0] == length

    if np.any(dim <= 0):
        raise ValueError(f"Argument {dim_name} must be all positive integer")
    return dim.astype(float)


def normalize_bboxes_np(
    bboxes: np.ndarray, rows: Union[int, Sequence[int], np.ndarray], cols: Union[int, Sequence[int], np.ndarray]
) -> np.ndarray:
    """Normalize a list of bounding boxes.

    Args:
        bboxes: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        rows: Image height.
        cols: Image width.

    Returns:
        Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
    """
    if not len(bboxes):
        return bboxes

    assert_np_bboxes_format(bboxes)

    if not isinstance(rows, int):
        rows = _convert_to_array(rows, len(bboxes), "rows")
    if not isinstance(cols, int):
        cols = _convert_to_array(cols, len(bboxes), "cols")

    bboxes_ = bboxes.copy().astype(float)
    bboxes_[:, 0::2] /= cols
    bboxes_[:, 1::2] /= rows
    return bboxes_


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


def denormalize_bboxes_np(bboxes: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Denormalize a list of bounding boxes.

    Args:
        bboxes: Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        rows: Image height.
        cols: Image width.

    Returns:
        List: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.

    """
    if not len(bboxes):
        return bboxes
    assert_np_bboxes_format(bboxes)

    if not isinstance(rows, int):
        rows = _convert_to_array(rows, len(bboxes), "rows")
    if not isinstance(cols, int):
        cols = _convert_to_array(cols, len(bboxes), "cols")
    bboxes_ = bboxes.copy()

    bboxes_[:, 0::2] *= cols
    bboxes_[:, 1::2] *= rows
    return bboxes_


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
    area = (x_max - x_min) * (y_max - y_min)
    return area


def calculate_bboxes_area(bboxes: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Calculate the area of bounding boxes in (fractional) pixels.

    Args:
        bboxes: numpy.ndarray
            A 2D ndarray
        rows: int
            Image height
        cols: int
            Image width

    Returns:
        numpy.ndarray, area in (fractional) pixels of the denormalized bounding boxes.

    """
    bboxes = bboxes if isinstance(bboxes, np.ndarray) else np.array(bboxes)
    assert_np_bboxes_format(bboxes)
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) * cols * rows
    return bboxes_area


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
        bbox_area = calculate_bbox_area(bbox[:4], img_height, img_width)
        transformed_bbox_area = calculate_bbox_area(transformed_bbox[:4], transformed_img_height, transformed_img_width)
        if transformed_bbox_area < min_area:
            continue
        visibility = transformed_bbox_area / bbox_area
        if visibility >= threshold:
            visible_bboxes.append(transformed_bbox)
    return visible_bboxes


def convert_bbox_to_albumentations(
    bbox: BoxType, source_format: str, rows: int, cols: int, check_validity: bool = False
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
        ValueError: if `target_format` is not equal to `coco` or `pascal_voc`, ot `yolo`.
        ValueError: If in YOLO format all labels not in range (0, 1).

    """
    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'"
        )

    if source_format == "coco":
        (x_min, y_min, width, height), tail = bbox[:4], bbox[4:]
        x_max = x_min + width
        y_max = y_min + height
    elif source_format == "yolo":
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/scripts/voc_label.py#L12
        _bbox = np.array(bbox[:4])
        if check_validity and np.any((_bbox <= 0) | (_bbox > 1)):
            raise ValueError("In YOLO format all coordinates must be float and in range (0, 1]")

        (x, y, w, h), tail = bbox[:4], bbox[4:]

        w_half, h_half = w / 2, h / 2
        x_min = x - w_half
        y_min = y - h_half
        x_max = x_min + w
        y_max = y_min + h
    else:
        (x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]

    bbox = (x_min, y_min, x_max, y_max) + tuple(tail)  # type: ignore

    if source_format != "yolo":
        bbox = normalize_bbox(bbox, rows, cols)
    if check_validity:
        check_bbox(bbox)
    return bbox


def convert_bbox_from_albumentations(
    bbox: BoxType, target_format: str, rows: int, cols: int, check_validity: bool = False
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
            f"Unknown target_format {target_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'"
        )
    if check_validity:
        check_bbox(bbox)

    if target_format != "yolo":
        bbox = denormalize_bbox(bbox, rows, cols)
    if target_format == "coco":
        (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])
        width = x_max - x_min
        height = y_max - y_min
        bbox = cast(BoxType, (x_min, y_min, width, height) + tail)
    elif target_format == "yolo":
        (x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        bbox = cast(BoxType, (x, y, w, h) + tail)
    return bbox


def convert_bboxes_to_albumentations(
    bboxes: Sequence[BoxType], source_format, rows, cols, check_validity=False
) -> List[BoxType]:
    """Convert a list bounding boxes from a format specified in `source_format` to the format used by albumentations"""
    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'"
        )
    np_bboxes = np.array([bbox[:4] for bbox in bboxes])

    if source_format == "coco":

        np_bboxes[:, 2:] += np_bboxes[:, :2]
    elif source_format == "yolo":
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/scripts/voc_label.py#L12

        if check_validity and np.any((np_bboxes <= 0) | (np_bboxes > 1)):
            raise ValueError("In YOLO format all coordinates must be float and in range (0, 1]")

        np_bboxes[:, :2] -= np_bboxes[:, 2:] / 2
        np_bboxes[:, 2:] += np_bboxes[:, :2]

    if source_format != "yolo":
        np_bboxes = normalize_bboxes_np(np_bboxes, rows, cols)
    if check_validity:
        check_bboxes(np_bboxes)

    return [cast(BoxType, tuple(np_bbox) + tuple(bbox[4:])) for np_bbox, bbox in zip(np_bboxes, bboxes)]


def convert_bboxes_from_albumentations(
    bboxes: Sequence[BoxType], target_format: str, rows: int, cols: int, check_validity: bool = False
) -> List[BoxType]:
    """Convert a list of bounding boxes from the format used by albumentations to a format, specified
    in `target_format`.

    Args:
        bboxes: List of albumentation bounding box `(x_min, y_min, x_max, y_max)`.
        target_format: required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.
        rows: Image height.
        cols: Image width.
        check_validity: Check if all boxes are valid boxes.

    Returns:
        List of bounding boxes.

    """
    if not len(bboxes):
        return []
    if target_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown target_format {target_format}. Supported formats are `coco`, `pascal_voc`, and `yolo`."
        )

    np_bboxes = np.array([bbox[:4] for bbox in bboxes])
    if check_validity:
        check_bboxes(np_bboxes)

    if target_format != "yolo":
        np_bboxes = denormalize_bboxes_np(np_bboxes, rows=rows, cols=cols)
    if target_format == "coco":
        np_bboxes[:, 2] -= np_bboxes[:, 0]
        np_bboxes[:, 3] -= np_bboxes[:, 1]
    elif target_format == "yolo":
        np_bboxes[:, 2:] -= np_bboxes[:, :2]
        np_bboxes[:, :2] += np_bboxes[:, 2:] / 2.0

    return [cast(BoxType, tuple(np_bbox) + tuple(bbox[4:])) for np_bbox, bbox in zip(np_bboxes, bboxes)]


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
    if not len(bboxes):
        return

    np_bboxes = bboxes if isinstance(bboxes, np.ndarray) else np.array([bbox[:4] for bbox in bboxes])
    row_idx, col_idx = np.where(
        (~np.logical_and(0 <= np_bboxes, np_bboxes <= 1)) & (~np.isclose(np_bboxes, 0)) & (~np.isclose(np_bboxes, 1))
    )
    if len(row_idx) and len(col_idx):
        name = {
            0: "x_min",
            1: "y_min",
            2: "x_max",
            3: "y_max",
        }[col_idx[0]]
        raise ValueError(
            f"Expected {name} for bbox {bboxes[row_idx[0]]} to be "
            f"in the range [0.0, 1.0], got {bboxes[row_idx[0]][col_idx[0]]}."
        )

    x_idx = np.where(np_bboxes[:, 0] >= np_bboxes[:, 2])[0]
    y_idx = np.where(np_bboxes[:, 1] >= np_bboxes[:, 3])[0]

    if len(x_idx):
        raise ValueError(f"x_max is less than or equal to x_min for bbox {bboxes[x_idx[0]]}.")
    if len(y_idx):
        raise ValueError(f"y_max is less than or equal to y_min for bbox {bboxes[y_idx[0]]}.")


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
        bboxes: List of albumentation bounding box `(x_min, y_min, x_max, y_max)`.
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

    if not len(bboxes):
        return []
    np_bboxes = np.array([bbox[:4] for bbox in bboxes])
    clipped_norm_bboxes = np.clip(np_bboxes, 0.0, 1.0)

    clipped_width = (clipped_norm_bboxes[:, 2] - clipped_norm_bboxes[:, 0]) * cols
    clipped_height = (clipped_norm_bboxes[:, 3] - clipped_norm_bboxes[:, 1]) * rows

    # denormalize bbox
    bboxes_area = calculate_bboxes_area(clipped_norm_bboxes, rows=rows, cols=cols)

    transform_bboxes_area = calculate_bboxes_area(np_bboxes, rows=rows, cols=cols)

    idx, *_ = np.where(
        (bboxes_area >= min_area)
        & (bboxes_area / transform_bboxes_area >= min_visibility)
        & (clipped_width >= min_width)
        & (clipped_height >= min_height)
    )

    resulting_boxes = [cast(BoxType, tuple(clipped_norm_bboxes[i]) + bboxes[i][4:]) for i in idx]

    return resulting_boxes


def union_of_bboxes(height: int, width: int, bboxes: Sequence[BoxType], erosion_rate: float = 0.0) -> BoxType:
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

    np_bboxes = bboxes if isinstance(bboxes, np.ndarray) else np.array([bbox[:4] for bbox in bboxes])
    w, h = np_bboxes[:, 2] - np_bboxes[:, 0], np_bboxes[:, 3] - np_bboxes[:, 1]

    limits = np.tile(
        np.concatenate((np.expand_dims(w, 0).transpose(), np.expand_dims(h, 0).transpose()), 1) * erosion_rate, 2
    )
    limits[2:] *= -1

    limits += np_bboxes

    limits = np.concatenate((limits, [[width, height, 0, 0]]))

    x1, y1 = np.min(limits[:, 0:2], axis=0)
    x2, y2 = np.max(limits[:, 2:4], axis=0)

    return x1, y1, x2, y2
