from __future__ import division
import math
import warnings

from albumentations.core.utils import DataProcessor

__all__ = [
    "angle_to_2pi_range",
    "check_keypoints",
    "convert_keypoints_from_albumentations",
    "convert_keypoints_to_albumentations",
    "filter_keypoints",
    "KeypointsProcessor",
]

keypoint_formats = {"xy", "yx", "xya", "xys", "xyas", "xysa"}


def angle_to_2pi_range(angle):
    two_pi = 2 * math.pi
    return angle % two_pi


class KeypointsProcessor(DataProcessor):
    @property
    def default_data_name(self):
        return "keypoints"

    def ensure_data_valid(self, data):
        if self.params.label_fields:
            if not all(i in data.keys() for i in self.params.label_fields):
                raise ValueError(
                    "Your 'label_fields' are not valid - them must have same names as params in "
                    "'keypoint_params' dict"
                )

    def ensure_transforms_valid(self, transforms):
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

    def filter(self, data, rows, cols):
        return filter_keypoints(data, rows, cols, remove_invisible=self.params.remove_invisible)

    def check(self, data, rows, cols):
        return check_keypoints(data, rows, cols)

    def convert_from_albumentations(self, data, rows, cols):
        return convert_keypoints_from_albumentations(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )

    def convert_to_albumentations(self, data, rows, cols):
        return convert_keypoints_to_albumentations(
            data,
            self.params.format,
            rows,
            cols,
            check_validity=self.params.remove_invisible,
            angle_in_degrees=self.params.angle_in_degrees,
        )


def check_keypoint(kp, rows, cols):
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


def check_keypoints(keypoints, rows, cols):
    """Check if keypoints boundaries are less than image shapes"""
    for kp in keypoints:
        check_keypoint(kp, rows, cols)


def filter_keypoints(keypoints, rows, cols, remove_invisible):
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


def keypoint_has_extra_data(kp, format):
    return len(kp) > len(format)


def convert_keypoint_to_albumentations(
    keypoint, source_format, rows, cols, check_validity=False, angle_in_degrees=True
):
    if source_format not in keypoint_formats:
        raise ValueError("Unknown target_format {}. Supported formats are: {}".format(source_format, keypoint_formats))

    if source_format == "xy":
        (x, y), tail = keypoint[:2], tuple(keypoint[2:])
        a, s = 0, 0
    elif source_format == "yx":
        (y, x), tail = keypoint[:2], tuple(keypoint[2:])
        a, s = 0, 0
    elif source_format == "xya":
        (x, y, a), tail = keypoint[:3], tuple(keypoint[3:])
        s = 0
    elif source_format == "xys":
        (x, y, s), tail = keypoint[:3], tuple(keypoint[3:])
        a = 0
    elif source_format == "xyas":
        (x, y, a, s), tail = keypoint[:4], tuple(keypoint[4:])
    elif source_format == "xysa":
        (x, y, s, a), tail = keypoint[:4], tuple(keypoint[4:])

    if angle_in_degrees:
        a = math.radians(a)

    keypoint = (x, y, angle_to_2pi_range(a), s) + tail
    if check_validity:
        check_keypoint(keypoint, rows, cols)
    return keypoint


def convert_keypoint_from_albumentations(
    keypoint, target_format, rows, cols, check_validity=False, angle_in_degrees=True
):
    # type (tuple, str, int, int, bool, bool) -> tuple
    if target_format not in keypoint_formats:
        raise ValueError("Unknown target_format {}. Supported formats are: {}".format(target_format, keypoint_formats))
    if check_validity:
        check_keypoint(keypoint, rows, cols)

    (x, y, angle, scale), tail = keypoint[:4], tuple(keypoint[4:])
    angle = angle_to_2pi_range(angle)
    if angle_in_degrees:
        angle = math.degrees(angle)

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

    return kp + tail


def convert_keypoints_to_albumentations(
    keypoints, source_format, rows, cols, check_validity=False, angle_in_degrees=True
):
    return [
        convert_keypoint_to_albumentations(kp, source_format, rows, cols, check_validity, angle_in_degrees)
        for kp in keypoints
    ]


def convert_keypoints_from_albumentations(
    keypoints, target_format, rows, cols, check_validity=False, angle_in_degrees=True
):
    return [
        convert_keypoint_from_albumentations(kp, target_format, rows, cols, check_validity, angle_in_degrees)
        for kp in keypoints
    ]
