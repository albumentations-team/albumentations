from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import numpy as np

from albumentations.core.types import NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS, PAIR

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


def angle_to_2pi_range(angles: np.ndarray) -> np.ndarray:
    return angles % (2 * np.pi)


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
        format: str,  # noqa: A002
        label_fields: Sequence[str] | None = None,
        remove_invisible: bool = True,
        angle_in_degrees: bool = True,
        check_each_transform: bool = True,
    ):
        super().__init__(format, label_fields)
        self.remove_invisible = remove_invisible
        self.angle_in_degrees = angle_in_degrees
        self.check_each_transform = check_each_transform

    def to_dict_private(self) -> dict[str, Any]:
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
    def __init__(self, params: KeypointParams, additional_targets: dict[str, str] | None = None):
        super().__init__(params, additional_targets)

    @property
    def default_data_name(self) -> str:
        return "keypoints"

    def ensure_data_valid(self, data: dict[str, Any]) -> None:
        if self.params.label_fields and not all(i in data for i in self.params.label_fields):
            msg = "Your 'label_fields' are not valid - them must have same names as params in 'keypoint_params' dict"
            raise ValueError(msg)

    def filter(self, data: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        self.params: KeypointParams
        return filter_keypoints(data, image_shape, remove_invisible=self.params.remove_invisible)

    def check(self, data: np.ndarray, image_shape: tuple[int, int]) -> None:
        check_keypoints(data, image_shape)

    def convert_from_albumentations(
        self,
        data: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        params = self.params
        return convert_keypoints_from_albumentations(
            data,
            params.format,
            image_shape,
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )

    def convert_to_albumentations(
        self,
        data: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        params = self.params
        return convert_keypoints_to_albumentations(
            data,
            params.format,
            image_shape,
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )


def check_keypoints(keypoints: np.ndarray, image_shape: tuple[int, int]) -> None:
    """Check if keypoint coordinates are less than image shapes for multiple keypoints"""
    rows, cols = image_shape[:2]

    # Check x and y coordinates
    x, y = keypoints[:, 0], keypoints[:, 1]
    if np.any((x < 0) | (x >= cols)) or np.any((y < 0) | (y >= rows)):
        invalid_x = np.where((x < 0) | (x >= cols))[0]
        invalid_y = np.where((y < 0) | (y >= rows))[0]

        error_messages = []

        error_messages = [
            f"Expected {'x' if idx in invalid_x else 'y'} for keypoint {keypoints[idx]} to be "
            f"in the range [0.0, {cols if idx in invalid_x else rows}], got {x[idx] if idx in invalid_x else y[idx]}."
            for idx in sorted(set(invalid_x) | set(invalid_y))
        ]

        raise ValueError("\n".join(error_messages))

    # Check angles
    if keypoints.shape[1] > PAIR:
        angles = keypoints[:, 2]
        invalid_angles = np.where((angles < 0) | (angles >= 2 * math.pi))[0]
        if len(invalid_angles) > 0:
            error_messages = [
                f"Keypoint angle must be in range [0, 2 * PI). Got: {angles[idx]} for keypoint {keypoints[idx]}"
                for idx in invalid_angles
            ]
            raise ValueError("\n".join(error_messages))


def filter_keypoints(
    keypoints: np.ndarray,
    image_shape: tuple[int, int],
    remove_invisible: bool,
) -> np.ndarray:
    """Filter keypoints to remove those outside the image boundaries.

    Args:
        keypoints: A numpy array of shape (N, 2+) where N is the number of keypoints.
                   Each row represents a keypoint (x, y, ...).
        image_shape: A tuple (height, width) representing the image dimensions.
        remove_invisible: If True, remove keypoints outside the image boundaries.

    Returns:
        A numpy array of filtered keypoints.
    """
    if not remove_invisible:
        return keypoints

    rows, cols = image_shape[:2]

    # Create boolean mask for visible keypoints
    x, y = keypoints[:, 0], keypoints[:, 1]
    visible = (x >= 0) & (x < cols) & (y >= 0) & (y < rows)

    # Apply the mask to filter keypoints
    return keypoints[visible]


def convert_keypoints_to_albumentations(
    keypoints: np.ndarray,
    source_format: Literal["xy", "yx", "xya", "xys", "xyas", "xysa"],
    image_shape: tuple[int, int],
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> np.ndarray:
    if source_format not in keypoint_formats:
        raise ValueError(f"Unknown source_format {source_format}. Supported formats are: {keypoint_formats}")

    format_to_indices: dict[str, list[int | None]] = {
        "xy": [0, 1, None, None],
        "yx": [1, 0, None, None],
        "xya": [0, 1, 2, None],
        "xys": [0, 1, None, 2],
        "xyas": [0, 1, 2, 3],
        "xysa": [0, 1, 3, 2],
    }

    indices: list[int | None] = format_to_indices[source_format]

    processed_keypoints = np.zeros((keypoints.shape[0], 4))

    for i, idx in enumerate(indices):
        if idx is not None:
            processed_keypoints[:, i] = keypoints[:, idx]

    if angle_in_degrees and indices[2] is not None:
        processed_keypoints[:, 2] = np.radians(processed_keypoints[:, 2])

    processed_keypoints[:, 2] = angle_to_2pi_range(processed_keypoints[:, 2])

    if keypoints.shape[1] > len(source_format):
        processed_keypoints = np.column_stack((processed_keypoints, keypoints[:, len(source_format) :]))

    if check_validity:
        check_keypoints(processed_keypoints, image_shape)

    return processed_keypoints


def convert_keypoints_from_albumentations(
    keypoints: np.ndarray,
    target_format: Literal["xy", "yx", "xya", "xys", "xyas", "xysa"],
    image_shape: tuple[int, int],
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> np.ndarray:
    if target_format not in keypoint_formats:
        raise ValueError(f"Unknown target_format {target_format}. Supported formats are: {keypoint_formats}")

    x, y, angle, scale = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], keypoints[:, 3]
    angle = angle_to_2pi_range(angle)

    if check_validity:
        check_keypoints(np.column_stack((x, y, angle, scale)), image_shape)

    if angle_in_degrees:
        angle = np.degrees(angle)

    format_to_columns = {
        "xy": [x, y],
        "yx": [y, x],
        "xya": [x, y, angle],
        "xys": [x, y, scale],
        "xyas": [x, y, angle, scale],
        "xysa": [x, y, scale, angle],
    }

    result = np.column_stack(format_to_columns[target_format])

    # Add any additional columns from the original keypoints
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        result = np.column_stack((result, keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:]))

    return result
