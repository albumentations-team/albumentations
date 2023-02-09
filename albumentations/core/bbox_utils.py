from __future__ import division

from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from .transforms_interface import BBoxesInternalType, BoxesArray, BoxType
from .utils import DataProcessor, Params

__all__ = [
    "normalize_bboxes_np",
    "denormalize_bboxes_np",
    "calculate_bboxes_area",
    "convert_bboxes_to_albumentations",
    "convert_bboxes_from_albumentations",
    "check_bboxes",
    "filter_bboxes",
    "union_of_bboxes",
    "BboxProcessor",
    "BboxParams",
    "bboxes_to_array",
    "array_to_bboxes",
    "ensure_bboxes_format",
    "use_bboxes_ndarray",
]

TBox = TypeVar("TBox", BoxType, BBoxesInternalType)


def bboxes_to_array(bboxes: Sequence[BoxType]) -> np.ndarray:
    return np.array([bbox[:4] for bbox in bboxes])


def array_to_bboxes(np_bboxes: np.ndarray, ori_bboxes: Sequence[BoxType]) -> List[BoxType]:
    return [cast(BoxType, tuple(np_bbox) + tuple(bbox[4:])) for bbox, np_bbox in zip(ori_bboxes, np_bboxes)]


def ensure_bboxes_format(func: Callable) -> Callable:
    """Ensure the return bboxes from the provided function check its consistency.

    Args:
        func (Callable): a callable with the first argument being bboxes.

    Returns:
        Callable
    """

    @wraps(func)
    def wrapper(bboxes, *args, **kwargs):
        bboxes = func(bboxes, *args, **kwargs)
        if isinstance(bboxes, BBoxesInternalType):
            bboxes.check_consistency()
        return bboxes

    return wrapper


def use_bboxes_ndarray(return_array: bool = True) -> Callable:
    def dec(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            bboxes: Union[BBoxesInternalType, np.ndarray], *args, **kwargs
        ) -> Union[BBoxesInternalType, np.ndarray]:
            if isinstance(bboxes, BBoxesInternalType):
                ret = func(bboxes.array, *args, **kwargs)
                if not return_array:
                    return ret
                if not isinstance(ret, np.ndarray):
                    raise TypeError(f"The return from {func.__name__} must be a numpy ndarray.")
                bboxes.array = ret
            elif isinstance(bboxes, np.ndarray):
                bboxes = func(bboxes.astype(float), *args, **kwargs)
            return bboxes

        return wrapper

    return dec


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

    def convert_to_internal_type(self, data):
        box_array = []
        targets = []
        for _data in data:
            box_array.append(_data[:4])
            targets.append(_data[4:])
        return BBoxesInternalType(array=np.array(box_array).astype(float), targets=targets)

    def convert_to_original_type(self, data):
        return [tuple(bbox.array[0].tolist()) + tuple(bbox.targets[0]) for bbox in data]  # type: ignore[attr-defined]

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

    def filter(self, data, rows: int, cols: int, target_name: str):
        self.params: BboxParams
        data = filter_bboxes(
            data,
            rows,
            cols,
            min_area=self.params.min_area,
            min_visibility=self.params.min_visibility,
            min_width=self.params.min_width,
            min_height=self.params.min_height,
        )

        return data

    def check(self, data, rows: int, cols: int) -> None:
        check_bboxes(data)

    def convert_from_albumentations(self, data, rows: int, cols: int):
        return convert_bboxes_from_albumentations(data, self.params.format, rows, cols, check_validity=True)

    def convert_to_albumentations(self, data, rows: int, cols: int):
        return convert_bboxes_to_albumentations(data, self.params.format, rows, cols, check_validity=True)


def _convert_to_array(dim: Sequence[Union[int, float]], length: int, dim_name: str) -> np.ndarray:
    if not isinstance(dim, np.ndarray):
        arr = np.array(dim) if isinstance(dim, Sequence) and isinstance(dim[0], (float, int)) else np.array([dim])
    elif isinstance(dim, np.ndarray) and len(dim.shape) == 1:
        arr = np.expand_dims(dim, axis=1)
    else:
        raise ValueError("dim should be a numpy ndarray")
    arr = arr.transpose()
    assert isinstance(arr, np.ndarray) and arr.shape[0] == length

    if np.any(arr <= 0):
        raise ValueError(f"Argument {dim_name} must be all positive integer")
    return dim


@use_bboxes_ndarray(return_array=True)
def normalize_bboxes_np(
    bboxes: BoxesArray, rows: Union[int, Sequence[int], np.ndarray], cols: Union[int, Sequence[int], np.ndarray]
) -> BoxesArray:
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
    if not isinstance(rows, (float, int)):
        rows = _convert_to_array(rows, len(bboxes), "rows").astype(float)
    if not isinstance(cols, (float, int)):
        cols = _convert_to_array(cols, len(bboxes), "cols").astype(float)

    bboxes_ = bboxes.copy().astype(float)
    bboxes_[:, 0::2] /= cols
    bboxes_[:, 1::2] /= rows
    return bboxes_


@use_bboxes_ndarray(return_array=True)
def denormalize_bboxes_np(bboxes: BoxesArray, rows: int, cols: int) -> BoxesArray:
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
    if not isinstance(rows, int):
        rows = _convert_to_array(rows, len(bboxes), "rows").astype(float)
    if not isinstance(cols, int):
        cols = _convert_to_array(cols, len(bboxes), "cols").astype(float)
    bboxes_ = bboxes.copy().astype(float)

    bboxes_[:, 0::2] *= cols
    bboxes_[:, 1::2] *= rows
    return bboxes_


@use_bboxes_ndarray(return_array=True)
def calculate_bboxes_area(bboxes: BoxesArray, rows: int, cols: int) -> np.ndarray:
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
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) * cols * rows
    return bboxes_area


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def convert_bboxes_to_albumentations(bboxes: BoxesArray, source_format, rows, cols, check_validity=False) -> BoxesArray:
    """Convert a list bounding boxes from a format specified in `source_format` to the format used by albumentations"""
    if not len(bboxes):
        return bboxes

    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'"
        )

    if source_format == "coco":

        bboxes[:, 2:] += bboxes[:, :2]
    elif source_format == "yolo":
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/scripts/voc_label.py#L12

        if check_validity and np.any((bboxes <= 0) | (bboxes > 1)):
            raise ValueError("In YOLO format all coordinates must be float and in range (0, 1]")

        bboxes[:, :2] -= bboxes[:, 2:] / 2
        bboxes[:, 2:] += bboxes[:, :2]

    if source_format != "yolo":
        bboxes = normalize_bboxes_np(bboxes, rows, cols)
    if check_validity:
        check_bboxes(bboxes)

    return bboxes


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=True)
def convert_bboxes_from_albumentations(
    bboxes: BoxesArray, target_format: str, rows: int, cols: int, check_validity: bool = False
) -> BoxesArray:
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
        return bboxes
    if target_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown target_format {target_format}. Supported formats are `coco`, `pascal_voc`, and `yolo`."
        )

    if check_validity:
        check_bboxes(bboxes)

    if target_format != "yolo":
        bboxes = denormalize_bboxes_np(bboxes, rows=rows, cols=cols)
    if target_format == "coco":
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]
    elif target_format == "yolo":
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes[:, :2] += bboxes[:, 2:] / 2.0

    return bboxes


@ensure_bboxes_format
@use_bboxes_ndarray(return_array=False)
def check_bboxes(bboxes: BoxesArray) -> None:
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    if not len(bboxes):
        return

    row_idx, col_idx = np.where(
        (~np.logical_and(0 <= bboxes, bboxes <= 1)) & (~np.isclose(bboxes, 0)) & (~np.isclose(bboxes, 1))
    )
    if len(row_idx) and len(col_idx):
        name = {
            0: "x_min",
            1: "y_min",
            2: "x_max",
            3: "y_max",
        }[col_idx[0]]
        raise ValueError(
            f"Expected {name} for bbox {bboxes[row_idx[0]].tolist()} to be "
            f"in the range [0.0, 1.0], got {bboxes[row_idx[0]][col_idx[0]]}."
        )

    x_idx = np.where(bboxes[:, 0] >= bboxes[:, 2])[0]
    y_idx = np.where(bboxes[:, 1] >= bboxes[:, 3])[0]

    if len(x_idx):
        raise ValueError(f"x_max is less than or equal to x_min for bbox {bboxes[x_idx[0]].tolist()}.")
    if len(y_idx):
        raise ValueError(f"y_max is less than or equal to y_min for bbox {bboxes[y_idx[0]].tolist()}.")


@ensure_bboxes_format
def filter_bboxes(
    bboxes: BBoxesInternalType,
    rows: int,
    cols: int,
    min_area: float = 0.0,
    min_visibility: float = 0.0,
    min_width: float = 0.0,
    min_height: float = 0.0,
) -> BBoxesInternalType:
    """Remove bounding boxes that either lie outside of the visible area by more than min_visibility
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
        return bboxes

    boxes_array = bboxes.array.copy()

    bboxes.array = np.clip(bboxes.array, 0.0, 1.0)

    clipped_width = (bboxes.array[:, 2] - bboxes.array[:, 0]) * cols
    clipped_height = (bboxes.array[:, 3] - bboxes.array[:, 1]) * rows

    # denormalize bbox
    bboxes_area = calculate_bboxes_area(bboxes.array, rows=rows, cols=cols)

    transform_bboxes_area = calculate_bboxes_area(boxes_array, rows=rows, cols=cols)

    idx, *_ = np.where(
        (bboxes_area >= min_area)
        & (bboxes_area / transform_bboxes_area >= min_visibility)
        & (clipped_width >= min_width)
        & (clipped_height >= min_height)
    )

    return bboxes[idx] if len(idx) != len(bboxes) else bboxes


@use_bboxes_ndarray(return_array=False)
def union_of_bboxes(bboxes: BoxesArray, height: int, width: int, erosion_rate: float = 0.0) -> BoxType:
    """Calculate union of bounding boxes.

    Args:
        bboxes (BBoxesInternalType): List like bounding boxes. Format is `[(x_min, y_min, x_max, y_max)]`.
        height (float): Height of image or space.
        width (float): Width of image or space.
        erosion_rate (float): How much each bounding box can be shrinked, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    w, h = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]

    limits = np.tile(
        np.concatenate((np.expand_dims(w, 0).transpose(), np.expand_dims(h, 0).transpose()), 1) * erosion_rate, 2
    )
    limits[2:] *= -1

    limits += bboxes

    limits = np.concatenate((limits, np.array([[width, height, 0, 0]])))

    x1, y1 = np.min(limits[:, 0:2], axis=0)
    x2, y2 = np.max(limits[:, 2:4], axis=0)

    return x1, y1, x2, y2
