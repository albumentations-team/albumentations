from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, TypedDict, TypeVar, Union

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


class DataWithLabels:
    def __init__(self, data: np.ndarray, labels: Optional[Dict[str, Any]] = None):
        self.data = data.astype(float)
        self.labels = labels if labels is not None else {}

    def __repr__(self) -> str:
        return f"data={self.data} labels={self.labels}"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: Union[int, slice]) -> Union[int, float, np.ndarray]:
        return self.data[item]


BBoxesInternalType = DataWithLabels
KeypointsInternalType = DataWithLabels


TBBoxesOrKeypoints = DataWithLabels
TRawBboxesOrKeypoints = TypeVar(
    "TRawBboxesOrKeypoints", bound=Union[np.ndarray, Sequence[Union[BoxInternalType, KeypointInternalType]]]
)
