from __future__ import division
import math
import warnings

from albumentations.core.utils import Params, DataProcessor


class KeypointsParams(Params):
    def __init__(self, format=None, label_fields=None, remove_invisible=True):
        super().__init__(format, label_fields)
        self.remove_invisible = remove_invisible


class KeypointsProcessor(DataProcessor):
    def ensure_data_valid(self, data):
        if self.params.label_fields:
            if not all(l in data.keys() for l in self.params.label_fields):
                raise Exception("Your 'label_fields' are not valid - them must have same names as params in "
                                "'keypoint_params' dict")

    def ensure_transforms_valid(self, transforms):
        pass
        # IAA-based augmentations supports only transformation of xy keypoints.
        # If your keypoints formats is other than 'xy' we emit warning to let user
        # be aware that angle and size will not be modified.

        # TODO Circular import
        # if self.params.data_format is not None and self.params.data_format != 'xy':
        #     for transform in transforms:
        #         if isinstance(transform, DualIAATransform):
        #             warnings.warn("{} transformation supports only 'xy' keypoints "
        #                           "augmentation. You have '{}' keypoints format. Scale "
        #                           "and angle WILL NOT BE transformed.".format(transform.__class__.__name__,
        #                                                                       self.params.data_format))
        #             break

    def preprocess(self, data):
        rows, cols = data['image'].shape[:2]

        for data_name in self.data_fields:
            data[data_name] = filter_keypoints(data[data_name], rows, cols, remove_invisible=self.params.remove_invisible)

            if self.params.format == 'albumentations':
                check_keypoints(data[data_name], rows, cols)
            else:
                data[data_name] = convert_keypoints_to_albumentations(data[data_name], self.params.format, rows, cols,
                                                                      check_validity=self.params.remove_invisible)

            data = self.remove_label_fields_from_data(data)
        return data

    def postprocess(self, data):
        data = self.add_label_fields_to_data(data)

        rows, cols = data['image'].shape[:2]
        for data_name in self.data_fields:
            if self.params.format == 'albumentations':
                check_keypoints(data[data_name], rows, cols)
            else:
                data[data_name] = convert_keypoint_from_albumentations(data[data_name], self.params.format, rows, cols,
                                                                       check_validity=self.params.remove_invisible)

        return data


def check_keypoint(kp, rows, cols):
    """Check if keypoint coordinates are in range [0, 1)"""
    for name, value, size in zip(['x', 'y'], kp[:2], [cols, rows]):
        if not 0 <= value < size:
            raise ValueError(
                'Expected {name} for keypoint {kp} '
                'to be in the range [0.0, {size}], got {value}.'.format(
                    kp=kp,
                    name=name,
                    value=value,
                    size=size
                )
            )


def check_keypoints(keypoints, rows, cols):
    """Check if keypoints boundaries are in range [0, 1)"""
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


keypoint_formats = {'xy', 'yx', 'xya', 'xys', 'xyas', 'xysa'}


def keypoint_has_extra_data(kp, format):
    return len(kp) > len(format)


def convert_keypoint_to_albumentations(keypoint, source_format, rows, cols,
                                       check_validity=False, angle_in_degrees=True):
    if source_format not in keypoint_formats:
        raise ValueError(
            "Unknown target_format {}. Supported formats are: {}".format(source_format, keypoint_formats)
        )

    if source_format == 'xy':
        x, y = keypoint[:2]
        a, s = 0, 0
        tail = list(keypoint[2:])
    elif source_format == 'yx':
        y, x = keypoint[:2]
        a, s = 0, 0
        tail = list(keypoint[2:])
    elif source_format == 'xya':
        x, y, a = keypoint[:3]
        s = 0
        tail = list(keypoint[3:])
    elif source_format == 'xys':
        x, y, s = keypoint[:3]
        a = 0
        tail = list(keypoint[3:])
    elif source_format == 'xyas':
        x, y, a, s = keypoint[:4]
        tail = list(keypoint[4:])
    elif source_format == 'xysa':
        x, y, s, a = keypoint[:4]
        tail = list(keypoint[4:])

    if angle_in_degrees:
        a = math.radians(a)

    keypoint = [x, y, a, s] + tail
    if check_validity:
        check_keypoint(keypoint, rows, cols)
    return keypoint


def normalize_angle(a):
    two_pi = 2. * math.pi
    while a < 0:
        a += two_pi
    while a > two_pi:
        a -= two_pi
    return a


def convert_keypoint_from_albumentations(keypoint, target_format, rows, cols,
                                         check_validity=False, angle_in_degrees=True):
    if target_format not in keypoint_formats:
        raise ValueError(
            "Unknown target_format {}. Supported formats are: {}".format(target_format, keypoint_formats)
        )
    if check_validity:
        check_keypoint(keypoint, rows, cols)
    x, y, a, s = keypoint[:4]
    a = normalize_angle(a)
    if angle_in_degrees:
        a = math.degrees(a)

    if target_format == 'xy':
        kp = [x, y]
    elif target_format == 'yx':
        kp = [y, x]
    elif target_format == 'xya':
        kp = [x, y, a]
    elif target_format == 'xys':
        kp = [x, y, s]
    elif target_format == 'xyas':
        kp = [x, y, a, s]
    elif target_format == 'xysa':
        kp = [x, y, s, a]

    return kp + list(keypoint[4:])


def convert_keypoints_to_albumentations(keypoints, source_format, rows, cols, check_validity=False):
    return [convert_keypoint_to_albumentations(kp, source_format, rows, cols, check_validity) for kp in keypoints]


def convert_keypoints_from_albumentations(keypoints, target_format, rows, cols, check_validity=False):
    return [convert_keypoint_from_albumentations(kp, target_format, rows, cols, check_validity) for kp in keypoints]
