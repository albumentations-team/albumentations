from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Tuple, TypedDict, TypeVar, Union

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

BoxesArray = np.ndarray
KeypointsArray = np.ndarray
InternalDtype = TypeVar("InternalDtype", bound=Union[BoxesArray, KeypointsArray])

ScaleFloatType = Union[float, Tuple[float, float]]
ScaleIntType = Union[int, Tuple[int, int]]

FillValueType = Optional[Union[int, float, Sequence[int], Sequence[float]]]


image_modes = ["cv", "pil"]
ImageMode = Literal["cv", "pil"]


SpatterMode = Literal["rain", "mud"]
ChromaticAberrationMode = Literal["green_purple", "red_blue", "random"]

TWO = 2
FOUR = 4


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


@dataclass
class BatchInternalType:
    array: np.ndarray
    targets: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=object))

    def __post_init__(self):
        if not isinstance(self.array, np.ndarray):
            self.array = np.array(self.array, dtype=float)
        elif isinstance(self.array, np.ndarray):
            self.array = self.array.astype(float)
        if not isinstance(self.targets, np.ndarray):
            self.targets = np.array(self.targets, dtype=object)
        if len(self.array) and not self.targets.shape[0]:
            self.targets = np.empty((len(self.array), 0), dtype=object)
        self.check_consistency()

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "array":
            self.assert_array_format(value)
        super().__setattr__(key, value)

    def __len__(self) -> int:
        if len(self.array) != len(self.targets):
            msg = (
                f"Length if array and targets are not equal."
                f" len(array)={len(self.array)} len(targets)={len(self.targets)}"
            )
            raise ValueError(msg)
        return len(self.array)

    @abstractmethod
    def __getitem__(self, item: Union[int, ...]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key: Union[int, ...], value: Union[float, np.ndarray]) -> None:
        raise NotImplementedError

    @abstractmethod
    def check_consistency(self) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def assert_array_format(array: np.ndarray) -> None:
        raise NotImplementedError


@dataclass
class Floats4Internal(BatchInternalType):
    def __eq__(self, other: "Floats4Internal") -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"`{self.__class__}` is only comparable with another `{self.__class__}`, "
                f"given {type(other)} instead."
            )

        if len(self.array) == len(other.array) == len(self.targets) == len(other.targets) == 0:
            # This's because numpy does not treat array([], dtype=float64)
            # and array=array([], shape=(0, 4), dtype=float64) equally.
            return True
        return np.array_equal(self.array, other.array) and np.array_equal(self.targets, other.targets)

    def __getitem__(self, item: Union[int, ...]) -> np.ndarray:
        _arr = self.array[item].astype(float)
        _target = self.targets[item]
        if isinstance(item, int):
            _arr = _arr[np.newaxis, :]
            _target = _target[np.newaxis, :]
        return self.__class__(array=_arr, targets=_target)

    def __setitem__(self, idx: Union[int, ...], value: "Floats4Internal") -> None:
        self.array[idx] = value.array
        self.targets[idx] = value.targets


@dataclass(eq=False)
class BBoxesInternalType(Floats4Internal):
    array: BoxesArray = field(default_factory=lambda: np.empty((0, 4)))

    @staticmethod
    def assert_array_format(bboxes: np.ndarray) -> None:
        if not isinstance(bboxes, np.ndarray):
            msg = "Bboxes should be a numpy ndarray."
            raise TypeError(msg)
        if len(bboxes) and not (len(bboxes.shape) == TWO and bboxes.shape[-1] == FOUR):
            msg = (
                f"An array of bboxes should be 2 dimension, and the last dimension must has 4 elements."
                f" Received {bboxes.shape}."
            )
            raise ValueError(msg)

    def check_consistency(self) -> None:
        if len(self.array) != len(self.targets):
            raise ValueError(
                "The amount of bboxes and additional targets should be the same. "
                f"Get {len(self.array)} bboxes and {len(self.targets)} additional targets."
            )
        self.assert_array_format(self.array)


@dataclass(eq=False)
class KeypointsInternalType(Floats4Internal):
    array: KeypointsArray = field(default_factory=lambda: np.empty((0, 4)))

    @staticmethod
    def assert_array_format(keypoints: np.ndarray) -> None:
        if not isinstance(keypoints, np.ndarray):
            msg = "keypoints should be a numpy ndarray."
            raise TypeError(msg)
        if len(keypoints) and not (len(keypoints.shape) == TWO and TWO <= keypoints.shape[-1] <= FOUR):
            msg = (
                "An array of keypoints should be 2 dimension, "
                "and the last dimension must has at least 2 elements at most 4 elements. "
                f"Received {keypoints.shape}."
            )
            raise ValueError(msg)

    def check_consistency(self) -> None:
        if self.targets is not None and len(self.array) != len(self.targets):
            raise ValueError(
                "The amount of keypoints and additional targets should be the same. "
                f"Get {len(self.array)} keypoints and {len(self.targets)} additional targets."
            )
        self.assert_array_format(self.array)
