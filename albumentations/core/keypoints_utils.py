from __future__ import division

import math
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from .transforms_interface import KeypointsInternalType, KeypointType
from .utils import DataProcessor, Params

__all__ = [
    "angle_to_2pi_range",
    "check_keypoints",
    "convert_keypoints_from_albumentations",
    "convert_keypoints_to_albumentations",
    "filter_keypoints",
    "KeypointsProcessor",
    "KeypointParams",
]

keypoint_formats = {"xy", "yx", "xya", "xys", "xyas", "xysa"}


def angle_to_2pi_range(angle: Union[np.ndarray, float]):
    two_pi = 2 * math.pi
    return angle % two_pi


def assert_np_keypoints_format(keypoints: KeypointsInternalType):  # noqa
    assert isinstance(keypoints, np.ndarray), "Keypoints should be represented by a 2D numpy array."
    if len(keypoints):
        assert (
            len(keypoints.shape) == 2 and 2 <= keypoints.shape[-1] <= 4
        ), "An array of keypoints should be 2 dimension, and the last dimension must has at least 2 elements."


def keypoints_to_array(keypoints: Sequence[KeypointType]) -> KeypointsInternalType:
    return np.array([keypoint[:4] for keypoint in keypoints])


def array_to_keypoints(
    np_keypoints: KeypointsInternalType,
    ori_keypoints: Sequence[KeypointType],
) -> Sequence[KeypointType]:
    return [
        cast(KeypointType, tuple(np_keypoint) + tuple(keypoint))
        for np_keypoint, keypoint in zip(np_keypoints, ori_keypoints)
    ]


def ensure_and_convert_keypoints(func: Callable) -> Callable:
    """Ensure keypoints in inputs of the provided function can be properly converted to numpy array.

    Args:
        func (Callable): a callable with the first argument being keypoints.

    Returns:
        Callable, a callable with the first argument being keypoints in numpy ndarray.
    """

    @wraps(func)
    def wrapper(keypoints, *args, **kwargs):  # noqa
        if not isinstance(keypoints, np.ndarray):
            keypoints = keypoints_to_array(keypoints).astype(float)
        assert_np_keypoints_format(keypoints)
        return func(keypoints, *args, **kwargs)

    return wrapper


class KeypointParams(Params):
    """
    Parameters of keypoints

    Args:
        format (str): format of keypoints. Should be 'xy', 'yx', 'xya', 'xys', 'xyas', 'xysa'.

            x - X coordinate,

            y - Y coordinate

            s - Keypoint scale

            a - Keypoint orientation in radians or degrees (depending on KeypointParams.angle_in_degrees)
        label_fields (list): list of fields that are joined with keypoints, e.g labels.
            Should be same type as keypoints.
        remove_invisible (bool): to remove invisible points after transform or not
        angle_in_degrees (bool): angle in degrees or radians in 'xya', 'xyas', 'xysa' keypoints
        check_each_transform (bool): if `True`, then keypoints will be checked after each dual transform.
            Default: `True`
    """

    def __init__(
        self,
        format: str,  # skipcq: PYL-W0622
        label_fields: Optional[Sequence[str]] = None,
        remove_invisible: bool = True,
        angle_in_degrees: bool = True,
        check_each_transform: bool = True,
    ):
        super(KeypointParams, self).__init__(format, label_fields)
        self.remove_invisible = remove_invisible
        self.angle_in_degrees = angle_in_degrees
        self.check_each_transform = check_each_transform

    def _to_dict(self) -> Dict[str, Any]:
        data = super(KeypointParams, self)._to_dict()
        data.update(
            {
                "remove_invisible": self.remove_invisible,
                "angle_in_degrees": self.angle_in_degrees,
                "check_each_transform": self.check_each_transform,
            }
        )
        return data

    @classmethod
    def is_serializable(cls) -> bool:
        return True

    @classmethod
    def get_class_fullname(cls) -> str:
        return "KeypointParams"


class KeypointsProcessor(DataProcessor):
    def __init__(self, params: KeypointParams, additional_targets: Optional[Dict[str, str]] = None):
        assert isinstance(params, KeypointParams)
        super().__init__(params, additional_targets)

    @property
    def default_data_name(self) -> str:
        return "keypoints"

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        if self.params.label_fields:
            if not all(i in data.keys() for i in self.params.label_fields):
                raise ValueError(
                    "Your 'label_fields' are not valid - them must have same names as params in "
                    "'keypoint_params' dict"
                )

    def ensure_transforms_valid(self, transforms: Sequence[object]) -> None:
        # IAA-based augmentations supports only transformation of xy keypoints.
        # If your keypoints formats is other than 'xy' we emit warning to let user
        # be aware that angle and size will not be modified.

        try:
            from albumentations.imgaug.transforms import DualIAATransform
        except ImportError:
            # imgaug is not installed so we skip imgaug checks.
            return

        if self.params.format is not None and self.params.format != "xy":
            for transform in transforms:
                if isinstance(transform, DualIAATransform):
                    warnings.warn(
                        "{} transformation supports only 'xy' keypoints "
                        "augmentation. You have '{}' keypoints format. Scale "
                        "and angle WILL NOT BE transformed.".format(transform.__class__.__name__, self.params.format)
                    )
                    break

    def filter(self, data: KeypointsInternalType, rows: int, cols: int, target_name: str) -> KeypointsInternalType:
        self.params: KeypointParams
        data, idx = filter_keypoints(data, rows, cols, remove_invisible=self.params.remove_invisible)
        self.filter_labels(target_name=target_name, indices=idx)
        return data

    def separate_label_from_data(self, data: Sequence) -> Tuple[Sequence, Sequence]:
        keypoints = []
        additional_data = []

        data_length = len(self.params.format)

        for _data in data:
            keypoints.append(_data[:data_length])
            additional_data.append(_data[data_length:])
        return keypoints, additional_data

    def check(self, data: Sequence[Sequence], rows: int, cols: int) -> None:
        check_keypoints(data, rows, cols)

    def convert_from_albumentations(self, data: KeypointsInternalType, rows: int, cols: int) -> KeypointsInternalType:
        return convert_keypoints_from_albumentations(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )

    def convert_to_albumentations(self, data: KeypointsInternalType, rows: int, cols: int) -> KeypointsInternalType:
        return convert_keypoints_to_albumentations(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )


@ensure_and_convert_keypoints
def check_keypoints(keypoints: KeypointsInternalType, rows: int, cols: int) -> None:
    """Check if keypoints boundaries are less than image shapes"""

    if not len(keypoints):
        return

    row_idx, *_ = np.where(
        ~np.logical_and(0 <= keypoints[..., 0], keypoints[..., 0] < cols)
        | ~np.logical_and(0 <= keypoints[..., 1], keypoints[..., 1] < rows)
    )
    if row_idx:
        raise ValueError(
            f"Expected keypoints `x` in the range [0.0, {cols}] and `y` in the range [0.0, {rows}]. "
            f"Got {keypoints[row_idx]}."
        )
    row_idx, *_ = np.where(~np.logical_and(0 <= keypoints[..., 2], keypoints[..., 2] < 2 * math.pi))
    if len(row_idx):
        raise ValueError(f"Keypoint angle must be in range [0, 2 * PI). Got: {keypoints[row_idx, 2]}.")


@ensure_and_convert_keypoints
def filter_keypoints(
    keypoints: KeypointsInternalType, rows: int, cols: int, remove_invisible: bool
) -> Tuple[KeypointsInternalType, Sequence[int]]:
    """Remove keypoints that are not visible.
    Args:
        keypoints: A batch of keypoints in `x, y, a, s` format.
        rows: Image height.
        cols: Image width.
        remove_invisible: whether to remove invisible keypoints or not.

    Returns:
        A batch of keypoints in `x, y, a, s` format.

    """
    if not remove_invisible:
        return keypoints, list(range(len(keypoints)))
    if not len(keypoints):
        return keypoints, []

    x = keypoints[..., 0]
    y = keypoints[..., 1]
    idx, *_ = np.where(np.logical_and(0 <= x, x < cols) & np.logical_and(0 <= y, y < rows))

    return keypoints[idx], idx


@ensure_and_convert_keypoints
def convert_keypoints_to_albumentations(
    keypoints: KeypointsInternalType,
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointsInternalType:
    if source_format not in keypoint_formats:
        raise ValueError(f"Unknown source_format {source_format}. " f"Supported formats are {keypoint_formats}.")
    if not len(keypoints):
        return keypoints

    if source_format == "xy":
        keypoints = np.concatenate((keypoints[..., :2], np.zeros_like(keypoints[..., :2])), axis=1)
    elif source_format == "yx":
        keypoints = np.concatenate((keypoints[..., :2][..., ::-1], np.zeros_like(keypoints[..., :2])), axis=1)
    elif source_format == "xya":
        keypoints = np.concatenate((keypoints[..., :3], np.zeros_like(keypoints[..., 0][..., np.newaxis])), axis=1)
    elif source_format == "xys":
        keypoints = np.insert(keypoints[..., :3], 2, np.zeros_like(keypoints[..., 0]), axis=1)
    elif source_format == "xyas":
        keypoints = keypoints[..., :4]
    elif source_format == "xysa":
        keypoints = keypoints[..., [0, 1, 3, 2]]
    else:
        raise ValueError(f"Unsupported source format. Got {source_format}.")

    if angle_in_degrees:
        keypoints[..., 2] = np.radians(keypoints[..., 2])
    keypoints[..., 2] = angle_to_2pi_range(keypoints[..., 2])
    if check_validity:
        check_keypoints(keypoints, rows=rows, cols=cols)
    return keypoints


@ensure_and_convert_keypoints
def convert_keypoints_from_albumentations(
    keypoints: KeypointsInternalType,
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointsInternalType:
    if target_format not in keypoint_formats:
        raise ValueError(f"Unknown target_format {target_format}. " f"Supported formats are: {keypoint_formats}.")

    if not len(keypoints):
        return keypoints

    keypoints[..., 2] = angle_to_2pi_range(keypoints[..., 2])
    if check_validity:
        check_keypoints(keypoints, rows, cols)

    if angle_in_degrees:
        keypoints[..., 2] = np.degrees(keypoints[..., 2])

    if target_format == "xy":
        keypoints = keypoints[..., :2]
    elif target_format == "yx":
        keypoints = keypoints[..., :2][..., ::-1]
    elif target_format == "xya":
        keypoints = keypoints[..., [0, 1, 2]]
    elif target_format == "xys":
        keypoints = keypoints[..., [0, 1, 3]]
    elif target_format == "xyas":
        keypoints = keypoints[..., :4]
    elif target_format == "xysa":
        keypoints = keypoints[..., [0, 1, 3, 2]]
    else:
        raise ValueError(f"Invalid target format. Got: {target_format}.")

    return keypoints
