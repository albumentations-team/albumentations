from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Literal, TypeVar, Union

import cv2
import numpy as np
from albucore.utils import MAX_VALUES_BY_DTYPE
from typing_extensions import NotRequired, TypedDict

ScalarType = Union[int, float]
ColorType = Union[float, Sequence[float]]

NumericType = TypeVar("NumericType", float, int)

ScaleIntType = Union[int, tuple[int, int]]
ScaleFloatType = Union[float, tuple[float, float]]

ScaleType = Union[ScaleIntType, ScaleFloatType]

NumType = Union[ScalarType, np.ndarray]

IntNumType = Union[np.integer, np.ndarray]
FloatNumType = Union[np.floating, np.ndarray]

ImageMode = Literal["cv", "pil"]
SpatterMode = Literal["rain", "mud"]
ChromaticAberrationMode = Literal["green_purple", "red_blue", "random"]
RainMode = Literal["drizzle", "heavy", "torrential", "default"]

MorphologyMode = Literal["erosion", "dilation"]

d4_group_elements = ["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]
D4Type = Literal["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]


class ReferenceImage(TypedDict):
    image: np.ndarray
    mask: NotRequired[np.ndarray]
    bbox: NotRequired[tuple[float, ...] | np.ndarray]
    keypoints: NotRequired[tuple[float, ...] | np.ndarray]


class Targets(Enum):
    IMAGE = "Image"
    MASK = "Mask"
    BBOXES = "BBoxes"
    KEYPOINTS = "Keypoints"


NUM_MULTI_CHANNEL_DIMENSIONS = 3
MONO_CHANNEL_DIMENSIONS = 2
NUM_RGB_CHANNELS = 3

PAIR = 2
TWO = 2
THREE = 3
FOUR = 4
EIGHT = 8
THREE_SIXTY = 360

BIG_INTEGER = MAX_VALUES_BY_DTYPE[np.uint32]
MAX_RAIN_ANGLE = 45  # Maximum angle for rain augmentation in degrees
MIN_UNIT_SIZE = 2

LENGTH_RAW_BBOX = 4

PercentType = Union[
    float,
    tuple[float, float],
    tuple[float, float, float, float],
    tuple[
        Union[float, tuple[float, float], list[float]],
        Union[float, tuple[float, float], list[float]],
        Union[float, tuple[float, float], list[float]],
        Union[float, tuple[float, float], list[float]],
    ],
]


PxType = Union[
    int,
    tuple[int, int],
    tuple[int, int, int, int],
    tuple[
        Union[int, tuple[int, int], list[int]],
        Union[int, tuple[int, int], list[int]],
        Union[int, tuple[int, int], list[int]],
        Union[int, tuple[int, int], list[int]],
    ],
]


REFLECT_BORDER_MODES = {
    cv2.BORDER_REFLECT101,
    cv2.BORDER_REFLECT_101,
    cv2.BORDER_REFLECT,
}

NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS = 4
NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS = 4


PositionType = Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"]
