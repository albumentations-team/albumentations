from __future__ import division

import math
import typing
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    def filter(self, data: Sequence[Sequence], rows: int, cols: int) -> Sequence[Sequence]:
        self.params: KeypointParams
        return filter_keypoints(data, rows, cols, remove_invisible=self.params.remove_invisible)

    def check(self, data: Sequence[Sequence], rows: int, cols: int) -> None:
        check_keypoints(data, rows, cols)

    def convert_from_albumentations(self, data: Sequence[Sequence], rows: int, cols: int) -> List[Tuple]:
        params = self.params
        return convert_keypoints_from_albumentations(
            data,
            params.format,
            rows,
            cols,
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )

    def convert_to_albumentations(self, data: Sequence[Sequence], rows: int, cols: int) -> List[Tuple]:
        params = self.params
        return convert_keypoints_to_albumentations(
            data,
            params.format,
            rows,
            cols,
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )


def check_keypoint(kp: Sequence, rows: int, cols: int) -> None:
    """Check if keypoint coordinates are less than image shapes"""
    for name, value, size in zip(["x", "y"], kp[:2], [cols, rows]):
        if not 0 <= value < size:
            raise ValueError(
                "Expected {name} for keypoint {kp} "
                "to be in the range [0.0, {size}], got {value}.".format(kp=kp, name=name, value=value, size=size)
            )

    angle = kp[2]
    if not (0 <= angle < 2 * math.pi):
        raise ValueError("Keypoint angle must be in range [0, 2 * PI). Got: {angle}".format(angle=angle))


def check_keypoints(keypoints: Sequence[Sequence], rows: int, cols: int) -> None:
    """Check if keypoints boundaries are less than image shapes"""
    for kp in keypoints:
        check_keypoint(kp, rows, cols)


def filter_keypoints(keypoints: Sequence[Sequence], rows: int, cols: int, remove_invisible: bool) -> Sequence[Sequence]:
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
    keypoint: Sequence,
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> Tuple:
    if source_format not in keypoint_formats:
        raise ValueError("Unknown target_format {}. Supported formats are: {}".format(source_format, keypoint_formats))

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

    keypoint = (x, y, angle_to_2pi_range(a), s) + tail
    if check_validity:
        check_keypoint(keypoint, rows, cols)
    return keypoint


def convert_keypoint_from_albumentations(
    keypoint: Sequence,
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> Tuple:
    if target_format not in keypoint_formats:
        raise ValueError("Unknown target_format {}. Supported formats are: {}".format(target_format, keypoint_formats))

    (x, y, angle, scale), tail = keypoint[:4], tuple(keypoint[4:])
    angle = angle_to_2pi_range(angle)
    if check_validity:
        check_keypoint((x, y, angle, scale), rows, cols)
    if angle_in_degrees:
        angle = math.degrees(angle)

    kp: Tuple
    if target_format == "xy":
        kp = (x, y)
    elif target_format == "yx":
        kp = (y, x)
    elif target_format == "xya":
        kp = (x, y, angle)
    elif target_format == "xys":
        kp = (x, y, scale)
    elif target_format == "xyas":
        kp = (x, y, angle, scale)
    elif target_format == "xysa":
        kp = (x, y, scale, angle)
    else:
        raise ValueError(f"Invalid target format. Got: {target_format}")

    return kp + tail


def convert_keypoints_to_albumentations(
    keypoints: Sequence[Sequence],
    source_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> List[Tuple]:
    return [
        convert_keypoint_to_albumentations(kp, source_format, rows, cols, check_validity, angle_in_degrees)
        for kp in keypoints
    ]


def convert_keypoints_from_albumentations(
    keypoints: Sequence[Sequence],
    target_format: str,
    rows: int,
    cols: int,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> List[Tuple]:
    return [
        convert_keypoint_from_albumentations(kp, target_format, rows, cols, check_validity, angle_in_degrees)
        for kp in keypoints
    ]
