import math
from typing import Any, Dict, Optional, Sequence, Type, TypeVar, Union

import numpy as np

from .types import KeypointsInternalType, KeypointType, TBBoxesOrKeypoints, TRawBboxesOrKeypoints
from .utils import DATA_DIM, DataProcessor, Params, get_numpy_2d_array

__all__ = [
    "angle_to_2pi_range",
    "check_keypoints",
    "convert_keypoints_from_albumentations",
    "convert_keypoints_to_albumentations",
    "filter_keypoints",
    "KeypointsProcessor",
    "KeypointParams",
]

KEYPOINT_FORMATS = {"xy", "yx", "xya", "xys", "xyas", "xysa"}

TArrayOrFloat = TypeVar("TArrayOrFloat", bound=Union[np.ndarray, float, int])


def angle_to_2pi_range(angle: TArrayOrFloat) -> TArrayOrFloat:
    two_pi = 2 * math.pi
    return angle % two_pi  # type: ignore[return-value]


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
        format: str,
        label_fields: Optional[Sequence[str]] = None,
        remove_invisible: bool = True,
        angle_in_degrees: bool = True,
        check_each_transform: bool = True,
    ):
        super().__init__(format, label_fields)
        self.remove_invisible = remove_invisible
        self.angle_in_degrees = angle_in_degrees
        self.check_each_transform = check_each_transform

    def to_dict_private(self) -> Dict[str, Any]:
        data = super().to_dict_private()
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
        super().__init__(params, additional_targets)

    @property
    def values_dim(self) -> int:
        return len(self.params.format)

    @property
    def internal_type(self) -> Type[KeypointsInternalType]:
        return KeypointsInternalType

    @property
    def default_data_name(self) -> str:
        return "keypoints"

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        if self.params.label_fields and not all(i in data for i in self.params.label_fields):
            msg = "Your 'label_fields' are not valid - them must have same names as params in " "'keypoint_params' dict"
            raise ValueError(msg)

    def filter(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> TBBoxesOrKeypoints:
        """The function filters a sequence of data based on the number of rows and columns, and returns a
        sequence of keypoints.

        Args:
            data: KeypointsInternalType of albumentations keypoints `[N, (x, y, a, s)]`.
            rows: The `rows` parameter represents the number of rows in the data matrix. It specifies
                  the number of rows that will be used for filtering the keypoints
            cols: The parameter "cols" represents the number of columns in the grid that the
                  keypoints will be filtered on

        Returns:
            A KeypointsInternalType object.
        """
        self.params: KeypointParams
        return filter_keypoints(
            data,
            rows,
            cols,
            remove_invisible=self.params.remove_invisible,
        )

    def check(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> None:
        check_keypoints(data.data, rows, cols, self.values_dim)

    def convert_from_albumentations(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> TBBoxesOrKeypoints:
        data.data = convert_keypoints_from_albumentations(
            data.data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )
        return data

    def convert_to_albumentations(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> TBBoxesOrKeypoints:
        data.data = convert_keypoints_to_albumentations(
            data.data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )
        return data


def check_keypoint(kp: KeypointType, rows: int, cols: int, values_dim: int = DATA_DIM) -> None:
    """Check if keypoint coordinates are less than image shapes"""
    return check_keypoints([kp], rows, cols, values_dim)


def check_keypoints(keypoints: TRawBboxesOrKeypoints, rows: int, cols: int, values_dim: int = DATA_DIM) -> None:
    """Check if keypoints boundaries are less than image shapes"""
    if not isinstance(keypoints, np.ndarray):  # Backward compatibility
        for i in keypoints:
            check_keypoints(get_numpy_2d_array([i[:values_dim]]), rows, cols, values_dim)
        return

    for name, value, size in zip(["x", "y"], keypoints[:, :2].T, [cols, rows]):
        cond = (value < 0) | (value >= size)
        if np.any(cond):
            raise ValueError(f"Expected {name} to be in the range [0.0, {size}]. Problem keypoints: {keypoints[cond]}")

    angle = keypoints[:, 2]
    cond = (angle < 0) | (angle >= 2 * math.pi)
    if np.any(cond):
        raise ValueError(f"Keypoint angle must be in range [0, 2 * PI). Problem keypoints: {keypoints[cond]}")


def filter_keypoints(
    keypoints: KeypointsInternalType, rows: int, cols: int, remove_invisible: bool
) -> KeypointsInternalType:
    if not remove_invisible:
        return keypoints

    if not isinstance(keypoints, KeypointsInternalType):
        msg = f"filter_keypoints works only with KeypointsInternalType. Got: {type(keypoints)}"
        raise TypeError(msg)

    x, y = keypoints.data[:, :2].T
    cond = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)

    result = KeypointsInternalType(keypoints.data[cond])
    for k, v in keypoints.labels.items():
        val = v if isinstance(v, np.ndarray) else np.array(v, dtype=object)
        result.labels[k] = val[cond]
    return result


def convert_keypoint_to_albumentations(
    keypoint: KeypointType,
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointType:
    return convert_keypoints_to_albumentations([keypoint], source_format, rows, cols, check_validity, angle_in_degrees)[
        0
    ]


def convert_keypoint_from_albumentations(
    keypoint: KeypointType,
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointType:
    return convert_keypoints_from_albumentations(
        [keypoint], target_format, rows, cols, check_validity, angle_in_degrees
    )[0]


def convert_keypoints_to_albumentations(  # noqa: C901
    keypoints: TRawBboxesOrKeypoints,
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> TRawBboxesOrKeypoints:
    if not isinstance(keypoints, np.ndarray):  # Backward compatibility
        num_values = len(source_format)
        return [  # type: ignore[return-value]
            tuple(
                convert_keypoints_to_albumentations(
                    get_numpy_2d_array([i[:num_values]]), source_format, rows, cols, check_validity
                )[0]
            )
            + tuple(i[num_values:])
            for i in keypoints
        ]

    if source_format not in KEYPOINT_FORMATS:
        raise ValueError(f"Unknown target_format {source_format}. Supported formats are: {KEYPOINT_FORMATS}")

    if source_format == "xy":
        x, y = keypoints.T
        a, s = np.zeros_like(x), np.zeros_like(x)
    elif source_format == "yx":
        y, x = keypoints.T
        a, s = np.zeros_like(x), np.zeros_like(x)
    elif source_format == "xya":
        x, y, a = keypoints.T
        s = np.zeros_like(x)
    elif source_format == "xys":
        x, y, s = keypoints.T
        a = np.zeros_like(x)
    elif source_format in "xyas":
        x, y, a, s = keypoints.T
    elif source_format in "xysa":
        x, y, s, a = keypoints.T
    else:
        raise ValueError(f"Unsupported source format. Got {source_format}")

    if angle_in_degrees:
        a = np.deg2rad(a)

    keypoints = np.stack([x, y, angle_to_2pi_range(a), s], axis=1)
    if check_validity:
        check_keypoints(keypoints, rows, cols, DATA_DIM)
    return keypoints


def convert_keypoints_from_albumentations(  # noqa: C901
    keypoints: TRawBboxesOrKeypoints,
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> TRawBboxesOrKeypoints:
    if not isinstance(keypoints, np.ndarray):  # Backward compatibility
        return [  # type: ignore[return-value]
            tuple(
                convert_keypoints_from_albumentations(
                    get_numpy_2d_array([i[:4]]), target_format, rows, cols, check_validity, angle_in_degrees
                )[0]
            )
            + tuple(i[4:])
            for i in keypoints
        ]

    if target_format not in KEYPOINT_FORMATS:
        raise ValueError(f"Unknown target_format {target_format}. Supported formats are: {KEYPOINT_FORMATS}")

    x, y, angle, scale = keypoints.T
    angle = angle_to_2pi_range(angle)
    if check_validity:
        check_keypoints(np.stack([x, y, angle, scale], axis=1), rows, cols, DATA_DIM)
    if angle_in_degrees:
        angle = np.degrees(angle)

    if target_format == "xy":
        res = [x, y]
    elif target_format == "yx":
        res = [y, x]
    elif target_format == "xya":
        res = [x, y, angle]
    elif target_format == "xys":
        res = [x, y, scale]
    elif target_format == "xyas":
        res = [x, y, angle, scale]
    elif target_format == "xysa":
        res = [x, y, scale, angle]
    else:
        raise ValueError(f"Invalid target format. Got: {target_format}")

    return np.stack(res, axis=1)
