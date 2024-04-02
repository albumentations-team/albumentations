from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence, Tuple, TypedDict, Union

import numpy as np
from typing_extensions import NotRequired

ScalarType = Union[int, float]
ColorType = Union[int, float, Sequence[int], Sequence[float]]
SizeType = Sequence[int]

BoxInternalType = Tuple[float, float, float, float]
BoxType = Union[BoxInternalType, Tuple[float, float, float, float, Any], Tuple[float, float, float, float]]
KeypointInternalType = Tuple[float, float, float, float]
KeypointType = Union[KeypointInternalType, Tuple[float, float, float, float, Any]]

BoxOrKeypointType = Union[BoxType, KeypointType]

ScaleFloatType = Union[float, Tuple[float, float]]
ScaleIntType = Union[int, Tuple[int, int]]

ScaleType = Union[ScaleFloatType, ScaleIntType]

NumType = Union[int, float, np.ndarray]

ImageColorType = Union[float, Sequence[float]]

IntNumType = Union[np.integer, np.ndarray]
FloatNumType = Union[np.floating, np.ndarray]


image_modes = ["cv", "pil"]
ImageMode = Literal["cv", "pil"]


SpatterMode = Literal["rain", "mud"]
ChromaticAberrationMode = Literal["green_purple", "red_blue", "random"]

MorphologyMode = Literal["erosion", "dilation"]


class ReferenceImage(TypedDict):
    image: Union[str, Path]
    mask: NotRequired[np.ndarray]
    global_label: NotRequired[np.ndarray]
    bbox: NotRequired[BoxType]
    keypoints: NotRequired[KeypointType]


class Targets(Enum):
    IMAGE = "Image"
    MASK = "Mask"
    BBOXES = "BBoxes"
    KEYPOINTS = "Keypoints"
    GLOBAL_LABEL = "Global Label"
