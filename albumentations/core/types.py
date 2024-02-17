from typing import Any, Sequence, Tuple, Union

import numpy as np

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
