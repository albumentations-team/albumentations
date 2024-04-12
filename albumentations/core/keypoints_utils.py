import math
from typing import Any, Dict, List, Optional, Sequence

from .types import KeypointType
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


def angle_to_2pi_range(angle: float) -> float:
    two_pi = 2 * math.pi
    return angle % two_pi


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
            },
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
    def default_data_name(self) -> str:
        return "keypoints"

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        if self.params.label_fields and not all(i in data for i in self.params.label_fields):
            msg = "Your 'label_fields' are not valid - them must have same names as params in 'keypoint_params' dict"
            raise ValueError(msg)

    def filter(self, data: Sequence[KeypointType], rows: int, cols: int) -> Sequence[KeypointType]:
        """The function filters a sequence of data based on the number of rows and columns, and returns a
        sequence of keypoints.

        :param data: The `data` parameter is a sequence of sequences. Each inner sequence represents a
        set of keypoints
        :type data: Sequence[Sequence]
        :param rows: The `rows` parameter represents the number of rows in the data matrix. It specifies
        the number of rows that will be used for filtering the keypoints
        :type rows: int
        :param cols: The parameter "cols" represents the number of columns in the grid that the
        keypoints will be filtered on
        :type cols: int
        :return: a sequence of KeypointType objects.
        """
        self.params: KeypointParams
        return filter_keypoints(data, rows, cols, remove_invisible=self.params.remove_invisible)

    def check(self, data: Sequence[KeypointType], rows: int, cols: int) -> None:
        check_keypoints(data, rows, cols)

    def convert_from_albumentations(self, data: Sequence[KeypointType], rows: int, cols: int) -> List[KeypointType]:
        params = self.params
        return convert_keypoints_from_albumentations(
            data,
            params.format,
            rows,
            cols,
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )

    def convert_to_albumentations(self, data: Sequence[KeypointType], rows: int, cols: int) -> List[KeypointType]:
        params = self.params
        return convert_keypoints_to_albumentations(
            data,
            params.format,
            rows,
            cols,
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )


def check_keypoint(kp: KeypointType, rows: int, cols: int) -> None:
    """Check if keypoint coordinates are less than image shapes"""
    for name, value, size in zip(["x", "y"], kp[:2], [cols, rows]):
        if not 0 <= value < size:
            raise ValueError(f"Expected {name} for keypoint {kp} to be in the range [0.0, {size}], got {value}.")

    angle = kp[2]
    if not (0 <= angle < 2 * math.pi):
        raise ValueError(f"Keypoint angle must be in range [0, 2 * PI). Got: {angle}")


def check_keypoints(keypoints: Sequence[KeypointType], rows: int, cols: int) -> None:
    """Check if keypoints boundaries are less than image shapes"""
    for kp in keypoints:
        check_keypoint(kp, rows, cols)


def filter_keypoints(
    keypoints: Sequence[KeypointType],
    rows: int,
    cols: int,
    remove_invisible: bool,
) -> Sequence[KeypointType]:
    if not remove_invisible:
        return keypoints

    resulting_keypoints = []
    for kp in keypoints:
        x, y = kp[:2]
        if x < 0 or x >= cols:
            continue
        if y < 0 or y >= rows:
            continue
        resulting_keypoints.append(kp)
    return resulting_keypoints


def convert_keypoint_to_albumentations(
    keypoint: KeypointType,
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointType:
    if source_format not in keypoint_formats:
        raise ValueError(f"Unknown target_format {source_format}. Supported formats are: {keypoint_formats}")

    if source_format == "xy":
        (x, y), tail = keypoint[:2], tuple(keypoint[2:])
        a, s = 0.0, 0.0
    elif source_format == "yx":
        (y, x), tail = keypoint[:2], tuple(keypoint[2:])
        a, s = 0.0, 0.0
    elif source_format == "xya":
        (x, y, a), tail = keypoint[:3], tuple(keypoint[3:])
        s = 0.0
    elif source_format == "xys":
        (x, y, s), tail = keypoint[:3], tuple(keypoint[3:])
        a = 0.0
    elif source_format == "xyas":
        (x, y, a, s), tail = keypoint[:4], tuple(keypoint[4:])
    elif source_format == "xysa":
        (x, y, s, a), tail = keypoint[:4], tuple(keypoint[4:])
    else:
        raise ValueError(f"Unsupported source format. Got {source_format}")

    if angle_in_degrees:
        a = math.radians(a)

    keypoint = (x, y, angle_to_2pi_range(a), s, *tail)
    if check_validity:
        check_keypoint(keypoint, rows, cols)
    return keypoint


def convert_keypoint_from_albumentations(
    keypoint: KeypointType,
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> KeypointType:
    if target_format not in keypoint_formats:
        raise ValueError(f"Unknown target_format {target_format}. Supported formats are: {keypoint_formats}")

    (x, y, angle, scale), tail = keypoint[:4], tuple(keypoint[4:])
    angle = angle_to_2pi_range(angle)
    if check_validity:
        check_keypoint((x, y, angle, scale), rows, cols)
    if angle_in_degrees:
        angle = math.degrees(angle)

    if target_format == "xy":
        return (x, y, *tail)
    if target_format == "yx":
        return (y, x, *tail)
    if target_format == "xya":
        return (x, y, angle, *tail)
    if target_format == "xys":
        return (x, y, scale, *tail)
    if target_format == "xyas":
        return (x, y, angle, scale, *tail)
    if target_format == "xysa":
        return (x, y, scale, angle, *tail)

    raise ValueError(f"Invalid target format. Got: {target_format}")


def convert_keypoints_to_albumentations(
    keypoints: Sequence[KeypointType],
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> List[KeypointType]:
    return [
        convert_keypoint_to_albumentations(kp, source_format, rows, cols, check_validity, angle_in_degrees)
        for kp in keypoints
    ]


def convert_keypoints_from_albumentations(
    keypoints: Sequence[KeypointType],
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> List[KeypointType]:
    return [
        convert_keypoint_from_albumentations(kp, target_format, rows, cols, check_validity, angle_in_degrees)
        for kp in keypoints
    ]
