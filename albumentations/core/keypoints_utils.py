import math
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .transforms_interface import KeypointsArray, KeypointsInternalType, KeypointType
from .utils import DataProcessor, InternalDtype, Params, ensure_internal_format

__all__ = [
    "angle_to_2pi_range",
    "check_keypoints",
    "convert_keypoints_from_albumentations",
    "convert_keypoints_to_albumentations",
    "filter_keypoints",
    "KeypointsProcessor",
    "KeypointParams",
    "use_keypoints_ndarray",
]

keypoint_formats = {"xy", "yx", "xya", "xys", "xyas", "xysa"}


def angle_to_2pi_range(angle: Union[np.ndarray, float]):
    two_pi = 2 * math.pi
    return angle % two_pi


def split_keypoints_targets(keypoints: Sequence[KeypointType], coord_length: int) -> Tuple[np.ndarray, List[Any]]:
    kps_array, targets = [], []
    for kp in keypoints:
        kps_array.append(kp[:coord_length])
        targets.append(kp[coord_length:])
    return np.array(kps_array, dtype=float), targets


def use_keypoints_ndarray(return_array: bool = True) -> Callable:
    """Decorate a function and return a decorator.
    Since most transformation functions does not alter the amount of bounding boxes, only update the internal
    keypoints' coordinates, thus this function provides a way to interact directly with
    the KeypointsInternalType's internal array member.

    Args:
        return_array (bool): whether the return of the decorated function is a KeypointsArray.

    Returns:
        Callable: A decorator function.
    """

    def dec(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            keypoints: Union[KeypointsInternalType, np.ndarray], *args, **kwargs
        ) -> Union[KeypointsInternalType, np.ndarray]:
            if isinstance(keypoints, KeypointsInternalType):
                ret = func(keypoints.array, *args, **kwargs)
                if not return_array:
                    return ret
                if not isinstance(ret, np.ndarray):
                    raise TypeError(f"The return from {func.__name__} must be a numpy ndarray.")
                keypoints.array = ret
            elif isinstance(keypoints, np.ndarray):
                keypoints = func(keypoints.astype(float), *args, **kwargs)
            else:
                raise TypeError(
                    f"The first input of {func.__name__} must be either a `KeypointsInternalType` or a `np.ndarray`. "
                    f"Given {type(keypoints)} instead."
                )
            return keypoints

        return wrapper

    return dec


class KeypointParams(Params):
    """Parameters of keypoints

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
        if not len(data):
            return KeypointsInternalType(array=np.empty(0))
        kps_array = []
        targets = []
        ori_kp_len = len(self.params.format)
        for _data in data:
            kps_array.append(_data[:ori_kp_len])
            targets.append(_data[ori_kp_len:])
        if ori_kp_len != 4:
            kps_array = np.pad(kps_array, [(0, 0), (0, 4 - ori_kp_len)], mode="constant").astype(float)
        else:
            kps_array = np.array(kps_array, dtype=float)
        return KeypointsInternalType(array=kps_array, targets=np.array(targets))

    def convert_to_original_type(self, data: InternalDtype) -> Any:
        dl = len(self.params.format)
        return [tuple(kp.array[0].tolist()[:dl]) + tuple(kp.targets[0].tolist()) for kp in data]  # type: ignore[attr-defined]

    @property
    def default_data_name(self) -> str:
        return "keypoints"

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        if self.params.label_fields:
            if not all(i in data for i in self.params.label_fields):
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
                        f"{transform.__class__.__name__} transformation supports only 'xy' keypoints "
                        f"augmentation. You have '{self.params.format}' keypoints format. Scale "
                        "and angle WILL NOT BE transformed."
                    )
                    break

    def filter(self, data: KeypointsArray, rows: int, cols: int, target_name: str):
        self.params: KeypointParams
        data = filter_keypoints(data, rows, cols, remove_invisible=self.params.remove_invisible)
        return data

    def check(self, data: KeypointsArray, rows: int, cols: int) -> None:
        check_keypoints(data, rows, cols)

    def convert_from_albumentations(self, data: KeypointsArray, rows: int, cols: int):
        return convert_keypoints_from_albumentations(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )

    def convert_to_albumentations(self, data: KeypointsArray, rows: int, cols: int):
        return convert_keypoints_to_albumentations(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )


@use_keypoints_ndarray(return_array=False)
def check_keypoints(keypoints: KeypointsArray, rows: int, cols: int) -> None:
    """Check if keypoints boundaries are less than image shapes"""
    if not len(keypoints):
        return

    row_idx, *_ = np.where(
        ~np.logical_and(keypoints[..., 0] >= 0, keypoints[..., 0] < cols)
        | ~np.logical_and(keypoints[..., 1] >= 0, keypoints[..., 1] < rows)
    )
    if row_idx:
        raise ValueError(
            f"Expected keypoints `x` in the range [0.0, {cols}] and `y` in the range [0.0, {rows}]. "
            f"Got {keypoints[row_idx]}."
        )
    row_idx, *_ = np.where(~np.logical_and(keypoints[..., 2] >= 0, keypoints[..., 2] < 2 * math.pi))
    if len(row_idx):
        raise ValueError(f"Keypoint angle must be in range [0, 2 * PI). Got: {keypoints[row_idx, 2]}.")


@ensure_internal_format
def filter_keypoints(
    keypoints: KeypointsInternalType, rows: int, cols: int, remove_invisible: bool
) -> KeypointsInternalType:
    """Remove keypoints that are not visible.

    Args:
        keypoints (KeypointsInternalType): A batch of keypoints in `x, y, a, s` format.
        rows (int): Image height.
        cols (int): Image width.
        remove_invisible (bool): whether to remove invisible keypoints or not.

    Returns:
        KeypointsInternalType: A batch of keypoints in `x, y, a, s` format.

    """
    if not remove_invisible:
        return keypoints
    if not len(keypoints):
        return keypoints

    x = keypoints.array[..., 0]
    y = keypoints.array[..., 1]
    idx, *_ = np.where(np.logical_and(x >= 0, x < cols) & np.logical_and(y >= 0, y < rows))

    return keypoints[idx] if len(idx) != len(keypoints) else keypoints


@ensure_internal_format
@use_keypoints_ndarray(return_array=True)
def convert_keypoints_to_albumentations(
    keypoints: KeypointsArray,
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointsArray:
    """Convert a batch of keypoints from source format to the format used by albumentations.

    Args:
        keypoints (KeypointsArray): a batch of keypoints in source format.
        source_format (str):
        rows (int):
        cols (int):
        check_validity (bool):
        angle_in_degrees (bool):

    Returns:
        KeypointsArray: A batch of keypoints in `albumentations` format, which is [x, y, a, s].

    Raises:
        ValueError: Unknown keypoint format is given.

    """
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


@ensure_internal_format
@use_keypoints_ndarray(return_array=True)
def convert_keypoints_from_albumentations(
    keypoints: KeypointsArray,
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointsArray:
    """Convert a batch of keypoints from `albumentations` format to target format.

    Args:
        keypoints (KeypointsArray): A batch of keypoints in `albumentations` format, which is [x, y, a, s].
        target_format (str):
        rows (int):
        cols (int):
        check_validity (bool):
        angle_in_degrees (bool):

    Returns:
        KeypointsArray: A batch of keypoints in target format.

    Raises:
        ValueError: Unknown target format is given.

    """
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
