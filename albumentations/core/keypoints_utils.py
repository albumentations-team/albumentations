from __future__ import division

import math
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from .transforms_interface import KeypointsArray, KeypointsInternalType
from .utils import DataProcessor, Params

__all__ = [
    "angle_to_2pi_range",
    "check_keypoints",
    "convert_keypoints_from_albumentations",
    "convert_keypoints_to_albumentations",
    "filter_keypoints",
    "KeypointsProcessor",
    "KeypointParams",
    "ensure_keypoints_format",
    "use_keypoints_ndarray",
]

keypoint_formats = {"xy", "yx", "xya", "xys", "xyas", "xysa"}


def angle_to_2pi_range(angle: Union[np.ndarray, float]):
    two_pi = 2 * math.pi
    return angle % two_pi


def ensure_keypoints_format(func: Callable) -> Callable:
    """Ensure keypoints in inputs of the provided function is KeypointsInternalType,
    and ensure its data consistency.

    Args:
        func (Callable): a callable with the first argument being keypoints.

    Returns:
        Callable, a callable with the first argument being keypoints as KeypointsInternalType.
    """

    @wraps(func)
    def wrapper(keypoints, *args, **kwargs):  # noqa
        if not isinstance(keypoints, KeypointsInternalType):
            raise TypeError(
                "keypoints should already converted to `KeypointsInternalType`. " f"Get {type(keypoints)} instead."
            )
        keypoints.check_consistency()
        return func(keypoints, *args, **kwargs)

    return wrapper


def use_keypoints_ndarray(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        keypoints: Union[KeypointsInternalType, np.ndarray], **kwargs
    ) -> Union[KeypointsInternalType, np.ndarray]:
        if isinstance(keypoints, KeypointsInternalType):
            ret = func(keypoints.array, **kwargs)
            if not isinstance(ret, np.ndarray):
                raise TypeError(f"The return from {func.__name__} must be a numpy ndarray.")
            keypoints.array = ret
            keypoints.check_consistency()
        elif isinstance(keypoints, np.ndarray):
            keypoints = func(keypoints, **kwargs)
        return keypoints

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

    def convert_to_internal_type(self, data):
        ...

    def convert_to_original_type(self, data):
        return [tuple(kp.array[0].tolist()) + tuple(kp.targets[0]) for kp in data]  # type: ignore[attr-defined]

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

    def filter(self, data, rows: int, cols: int, target_name: str):
        self.params: KeypointParams
        data = filter_keypoints(data, rows, cols, remove_invisible=self.params.remove_invisible)
        return data

    def check(self, data, rows: int, cols: int) -> None:
        check_keypoints(data, rows, cols)

    def convert_from_albumentations(self, data, rows: int, cols: int):
        return convert_keypoints_from_albumentations(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )

    def convert_to_albumentations(self, data, rows: int, cols: int):
        return convert_keypoints_to_albumentations(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )


@use_keypoints_ndarray
def check_keypoints(keypoints: KeypointsArray, rows: int, cols: int) -> None:
    """Check if keypoints boundaries are less than image shapes"""

    if not len(keypoints):
        return

    row_idx, *_ = np.where(
        ~np.logical_and(0 <= keypoints.array[..., 0], keypoints.array[..., 0] < cols)
        | ~np.logical_and(0 <= keypoints.array[..., 1], keypoints.array[..., 1] < rows)
    )
    if row_idx:
        raise ValueError(
            f"Expected keypoints `x` in the range [0.0, {cols}] and `y` in the range [0.0, {rows}]. "
            f"Got {keypoints.array[row_idx]}."
        )
    row_idx, *_ = np.where(~np.logical_and(0 <= keypoints.array[..., 2], keypoints.array[..., 2] < 2 * math.pi))
    if len(row_idx):
        raise ValueError(f"Keypoint angle must be in range [0, 2 * PI). Got: {keypoints.array[row_idx, 2]}.")


@ensure_keypoints_format
@use_keypoints_ndarray
def filter_keypoints(keypoints: KeypointsArray, rows: int, cols: int, remove_invisible: bool) -> KeypointsArray:
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
        return keypoints
    if not len(keypoints):
        return keypoints

    x = keypoints.array[..., 0]
    y = keypoints.array[..., 1]
    idx, *_ = np.where(np.logical_and(0 <= x, x < cols) & np.logical_and(0 <= y, y < rows))

    return keypoints[idx]


@ensure_keypoints_format
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


@ensure_keypoints_format
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
