from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np

from .types import BBoxesInternalType, BoxInternalType, BoxType, TBBoxesOrKeypoints, TRawBboxesOrKeypoints
from .utils import DataProcessor, Params, get_numpy_2d_array

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
        super().__init__(format, label_fields)
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.min_width = min_width
        self.min_height = min_height
        self.check_each_transform = check_each_transform

    def to_dict_private(self) -> Dict[str, Any]:
        data = super().to_dict_private()
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
    def internal_type(self) -> Type[BBoxesInternalType]:
        return BBoxesInternalType

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

    def filter(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> TBBoxesOrKeypoints:
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

    def check(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> None:
        check_bboxes(data.data)

    def convert_from_albumentations(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> TBBoxesOrKeypoints:
        data.data = convert_bboxes_from_albumentations(
            data.data,
            self.params.format,
            rows,
            cols,
            check_validity=True,
        )
        return data

    def convert_to_albumentations(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> TBBoxesOrKeypoints:
        data.data = convert_bboxes_to_albumentations(
            data.data,
            self.params.format,
            rows,
            cols,
            check_validity=True,
        )
        return data


def normalize_bbox(bbox: BoxType, rows: int, cols: int) -> BoxType:
    return normalize_bboxes([bbox], rows, cols)[0]


def denormalize_bbox(bbox: BoxType, rows: int, cols: int) -> BoxType:
    return denormalize_bboxes([bbox], rows, cols)[0]


def normalize_bboxes(bboxes: TRawBboxesOrKeypoints, rows: int, cols: int) -> TRawBboxesOrKeypoints:
    """Normalize a np.ndarray of bounding boxes.

    Args:
        bboxes: Denormalized bounding boxes `[N, (x_min, y_min, x_max, y_max)]`.
        rows: Image height.
        cols: Image width.

    Returns:
        Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.

    """
    if not isinstance(bboxes, np.ndarray):  # Backward compatibility
        return [  # type: ignore[return-value]
            tuple(normalize_bboxes(get_numpy_2d_array([i[:4]]), rows, cols)[0]) + tuple(i[4:]) for i in bboxes
        ]

    result = np.empty_like(bboxes, dtype=float)
    result[:, 0::2] = bboxes[:, 0::2] * (1 / cols)
    result[:, 1::2] = bboxes[:, 1::2] * (1 / rows)
    return result


def denormalize_bboxes(bboxes: TRawBboxesOrKeypoints, rows: int, cols: int) -> TRawBboxesOrKeypoints:
    """Denormalize coordinates of a bounding boxes. Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bboxes`.

    Args:
        bboxes: Normalized bounding boxes `[N, (x_min, y_min, x_max, y_max)]`.
        rows: Image height.
        cols: Image width.

    Returns:
        Denormalized bounding boxes `[N, (x_min, y_min, x_max, y_max)]`.

    Raises:
        ValueError: If rows or cols is less or equal zero
    """
    if not isinstance(bboxes, np.ndarray):  # Backward compatibility
        return [  # type: ignore[return-value]
            tuple(denormalize_bboxes(get_numpy_2d_array([i[:4]]), rows, cols)[0]) + tuple(i[4:]) for i in bboxes
        ]

    if rows <= 0:
        msg = "Argument rows must be positive integer"
        raise ValueError(msg)
    if cols <= 0:
        msg = "Argument cols must be positive integer"
        raise ValueError(msg)

    result = np.empty_like(bboxes)
    result[:, 0::2] = bboxes[:, 0::2] * cols
    result[:, 1::2] = bboxes[:, 1::2] * rows

    return result


def calculate_bbox_area(bbox: BoxType, rows: int, cols: int) -> float:
    """Calculate the area of a bounding box in (fractional) pixels.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.

    Return:
        Area in (fractional) pixels of the (denormalized) bounding box.

    """
    return float(calculate_bboxes_area(np.array([bbox[:4]]), rows, cols)[0])


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


def convert_bboxes_to_albumentations(
    bboxes: TRawBboxesOrKeypoints, source_format: str, rows: int, cols: int, check_validity: bool = False
) -> TRawBboxesOrKeypoints:
    """Convert a bounding box from a format specified in `source_format` to the format used by albumentations:
    normalized coordinates of top-left and bottom-right corners of the bounding box in a form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.

    Args:
        bboxes: A [N, 4] dimension array represents bounding boxes.
        source_format: format of the bounding box. Should be 'coco', 'pascal_voc', or 'yolo'.
        check_validity: Check if all boxes are valid boxes.
        rows: Image height.
        cols: Image width.

    Returns:
        np.ndarray: A bounding boxes `[N, (x_min, y_min, x_max, y_max)]`.

    Note:
        The `coco` format of a bounding box looks like `(x_min, y_min, width, height)`, e.g. (97, 12, 150, 200).
        The `pascal_voc` format of a bounding box looks like `(x_min, y_min, x_max, y_max)`, e.g. (97, 12, 247, 212).
        The `yolo` format of a bounding box looks like `(x, y, width, height)`, e.g. (0.3, 0.1, 0.05, 0.07);
        where `x`, `y` coordinates of the center of the box, all values normalized to 1 by image height and width.

    Raises:
        ValueError: if `target_format` is not equal to `coco` or `pascal_voc`, or `yolo`.
        ValueError: If in YOLO format all labels not in range (0, 1).

    """
    if not isinstance(bboxes, np.ndarray):  # Backward compatibility
        return [  # type: ignore[return-value]
            tuple(
                convert_bboxes_to_albumentations(
                    get_numpy_2d_array([i[:4]]), source_format, rows, cols, check_validity
                )[0]
            )
            + tuple(i[4:])
            for i in bboxes
        ]

    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'"
        )

    if source_format == "coco":
        x_min, y_min, width, height = bboxes.T
        x_max = x_min + width
        y_max = y_min + height
    elif source_format == "yolo":
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/scripts/voc_label.py#L12
        if check_validity and np.any((bboxes <= 0) | (bboxes > 1)):
            msg = "In YOLO format all coordinates must be float and in range (0, 1]"
            raise ValueError(msg)

        x, y, w, h = bboxes.T

        w_half, h_half = w / 2, h / 2
        x_min = x - w_half
        y_min = y - h_half
        x_max = x_min + w
        y_max = y_min + h
    else:
        x_min, y_min, x_max, y_max = bboxes.T

    bboxes = np.stack([x_min, y_min, x_max, y_max]).T

    if source_format != "yolo":
        bboxes = normalize_bboxes(bboxes, rows, cols)
    if check_validity:
        check_bboxes(bboxes)
    return bboxes


def convert_bboxes_from_albumentations(
    bboxes: TRawBboxesOrKeypoints, target_format: str, rows: int, cols: int, check_validity: bool = False
) -> TRawBboxesOrKeypoints:
    """Convert a bounding box from the format used by albumentations to a format, specified in `target_format`.

    Args:
        bboxes: An albumentations bounding box `[N, (x_min, y_min, x_max, y_max)]`.
        target_format: required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.
        rows: Image height.
        cols: Image width.
        check_validity: Check if all boxes are valid boxes.

    Returns:
        np.ndarray: A bounding boxes.

    Note:
        The `coco` format of a bounding box looks like `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
        The `pascal_voc` format of a bounding box looks like `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
        The `yolo` format of a bounding box looks like `[x, y, width, height]`, e.g. [0.3, 0.1, 0.05, 0.07].

    Raises:
        ValueError: if `target_format` is not equal to `coco`, `pascal_voc` or `yolo`.

    """
    if not isinstance(bboxes, np.ndarray):  # Backward compatibility
        return [  # type: ignore[return-value]
            tuple(
                convert_bboxes_from_albumentations(
                    get_numpy_2d_array([i[:4]]), target_format, rows, cols, check_validity
                )[0]
            )
            + tuple(i[4:])
            for i in bboxes
        ]

    if target_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            f"Unknown target_format {target_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'"
        )
    if check_validity:
        check_bboxes(bboxes)

    result_bboxes = denormalize_bboxes(bboxes, rows, cols) if target_format != "yolo" else bboxes

    if target_format == "coco":
        x_min, y_min, x_max, y_max = result_bboxes.T
        result_bboxes[:, 2] = x_max - x_min
        result_bboxes[:, 3] = y_max - y_min
    elif target_format == "yolo":
        result_bboxes = np.empty_like(bboxes)
        x_min, y_min, x_max, y_max = bboxes.T
        result_bboxes[:, 0] = (x_min + x_max) / 2.0
        result_bboxes[:, 1] = (y_min + y_max) / 2.0
        result_bboxes[:, 2] = x_max - x_min
        result_bboxes[:, 3] = y_max - y_min
    return result_bboxes


def convert_bbox_to_albumentations(
    bbox: BoxType, source_format: str, rows: int, cols: int, check_validity: bool = False
) -> BoxType:
    return convert_bboxes_to_albumentations([bbox], source_format, rows, cols, check_validity)[0]


def convert_bbox_from_albumentations(
    bbox: BoxType, target_format: str, rows: int, cols: int, check_validity: bool = False
) -> BoxType:
    return convert_bboxes_from_albumentations([bbox], target_format, rows, cols, check_validity)[0]


def check_bbox(bbox: BoxType) -> None:
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    check_bboxes([bbox])


def check_bboxes(bboxes: TRawBboxesOrKeypoints) -> None:
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    if not isinstance(bboxes, np.ndarray):  # Backward compatibility
        for i in bboxes:
            check_bboxes(get_numpy_2d_array([i[:4]]))
        return

    for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bboxes[:, :4].T):
        cond = ~np.bitwise_and(value >= 0, value <= 1)
        if np.any(cond):
            cond &= ~(np.isclose(value, 0) | np.isclose(value, 1))
            if np.any(cond):
                raise ValueError(
                    f"Expected {name} for bboxes to be in the range [0.0, 1.0]. Wrong bboxes: {bboxes[cond]}"
                )
    x_min, y_min, x_max, y_max = bboxes.T
    if np.any(x_max <= x_min):
        raise ValueError(f"x_max is less than or equal to x_min for bboxes {bboxes[x_max <= x_min]}.")
    if np.any(y_max <= y_min):
        raise ValueError(f"y_max is less than or equal to y_min for bbox {bboxes[y_max <= y_min]}.")


def calculate_bboxes_area(bbox: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Calculate the area of a bounding box in (fractional) pixels.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.

    Return:
        Area in (fractional) pixels of the (denormalized) bounding box.
    """
    bbox = denormalize_bboxes(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox.T
    return (x_max - x_min) * (y_max - y_min)


def filter_bboxes(
    bboxes: BBoxesInternalType,
    rows: int,
    cols: int,
    min_area: float = 0.0,
    min_visibility: float = 0.0,
    min_width: float = 0.0,
    min_height: float = 0.0,
) -> BBoxesInternalType:
    """Remove bounding boxes that either lie outside the visible area by more than min_visibility
    or whose area in pixels is under the threshold set by `min_area`. Also, it crops boxes to final image size.

    Args:
        bboxes: BBoxesInternalType of albumentations bounding box `[N, (x_min, y_min, x_max, y_max)]`.
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
        BBoxesInternalType of bounding boxes.

    """
    if not isinstance(bboxes, BBoxesInternalType):
        msg = f"filter_bboxes works only with BBoxesInternalType. Got: {type(bboxes)}"
        raise TypeError(msg)

    # Calculate areas of bounding box before and after clipping.
    transformed_box_area = calculate_bboxes_area(bboxes.data, rows, cols)
    bboxes_data = np.clip(bboxes.data, 0, 1.0)
    bboxes = BBoxesInternalType(bboxes_data, bboxes.labels)
    clipped_boxes_area = calculate_bboxes_area(bboxes_data, rows, cols)

    # Calculate width and height of the clipped bounding box.
    x_min, y_min, x_max, y_max = denormalize_bboxes(bboxes.data, rows, cols).T
    clipped_width, clipped_height = x_max - x_min, y_max - y_min

    cond = clipped_boxes_area != 0  # to ensure transformed_box_area!=0 and to handle min_area=0 or min_visibility=0
    cond &= clipped_boxes_area >= min_area
    cond &= clipped_boxes_area / transformed_box_area >= min_visibility
    cond &= clipped_width > min_width
    cond &= clipped_height > min_height

    result = BBoxesInternalType(bboxes.data[cond])
    for k, v in bboxes.labels.items():
        val = v if isinstance(v, np.ndarray) else np.array(v, dtype=object)
        result.labels[k] = val[cond]
    return result


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
        w, h = x_max - x_min, y_max - y_min
        lim_x1, lim_y1 = x_min + erosion_rate * w, y_min + erosion_rate * h
        lim_x2, lim_y2 = x_max - erosion_rate * w, y_max - erosion_rate * h
        x1, y1 = np.min([x1, lim_x1]), np.min([y1, lim_y1])
        x2, y2 = np.max([x2, lim_x2]), np.max([y2, lim_y2])
    return x1, y1, x2, y2
