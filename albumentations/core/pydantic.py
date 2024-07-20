from __future__ import annotations

from typing import Tuple

import cv2
from pydantic import Field
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from albumentations.core.types import NumericType, ScalarType, ScaleType
from albumentations.core.utils import to_tuple

valid_interpolations = [
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_CUBIC,
    cv2.INTER_AREA,
    cv2.INTER_LANCZOS4,
    cv2.INTER_LINEAR_EXACT,
    cv2.INTER_MAX,
]


def check_valid_interpolation(value: int) -> int:
    if value not in valid_interpolations:
        raise ValueError(f"Interpolation should be one of {valid_interpolations}, got {value} instead")
    return value


InterpolationType = Annotated[int, Field(description="Interpolation"), AfterValidator(check_valid_interpolation)]

valid_border_modes = [
    cv2.BORDER_CONSTANT,  # Adds a constant colored border. The value should be given as next argument.
    cv2.BORDER_REPLICATE,  # Replicates the last element.
    cv2.BORDER_REFLECT,  # Mirrors the image border without repeating the last element.
    cv2.BORDER_WRAP,  # Wraps the image around like a tiled texture.
    cv2.BORDER_REFLECT_101,  # Also known as cv2.BORDER_DEFAULT. Mirrors the image border, repeating the last element.
    cv2.BORDER_TRANSPARENT,  # Makes the border transparent
]


def check_valid_border_modes(value: int) -> int:
    if value not in valid_interpolations:
        raise ValueError(f"Border mode should be one of {valid_border_modes}, got {value} instead")
    return value


BorderModeType = Annotated[int, Field(description="Border Mode"), AfterValidator(check_valid_border_modes)]


valid_font_types = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    cv2.FONT_ITALIC,
]


def check_valid_font_types(value: int) -> int:
    if value not in valid_font_types:
        raise ValueError(f"Font type should be one of {valid_font_types}, got {value} instead")
    return value


FontType = Annotated[int, Field(description="Font type"), AfterValidator(check_valid_border_modes)]

ProbabilityType = Annotated[float, Field(description="Probability of applying the transform", ge=0, le=1)]


def process_non_negative_range(value: ScaleType | None) -> tuple[float, float]:
    result = to_tuple(value if value is not None else 0, 0)
    if not all(x >= 0 for x in result):
        msg = "All values in the non negative range should be non negative"
        raise ValueError(msg)
    return result


def float2int(value: tuple[float, float]) -> tuple[int, int]:
    return int(value[0]), int(value[1])


NonNegativeFloatRangeType = Annotated[ScaleType, AfterValidator(process_non_negative_range)]
NonNegativeIntRangeType = Annotated[ScaleType, AfterValidator(process_non_negative_range), AfterValidator(float2int)]


def create_symmetric_range(value: ScaleType) -> tuple[float, float]:
    return to_tuple(value)


SymmetricRangeType = Annotated[ScaleType, AfterValidator(create_symmetric_range)]


def convert_to_1plus_range(value: ScaleType) -> tuple[float, float]:
    return to_tuple(value, low=1)


def check_1plus(value: tuple[NumericType, NumericType]) -> tuple[NumericType, NumericType]:
    if any(x < 1 for x in value):
        raise ValueError(f"All values should be >= 1, got {value} instead")
    return value


def check_0plus(value: tuple[NumericType, NumericType]) -> tuple[NumericType, NumericType]:
    if any(x < 0 for x in value):
        raise ValueError(f"All values should be >= 0, got {value} instead")
    return value


def nondecreasing(value: tuple[NumericType, NumericType]) -> tuple[NumericType, NumericType]:
    if not value[0] <= value[1]:
        raise ValueError(f"First value should be less than the second value, got {value} instead")
    return value


OnePlusFloatRangeType = Annotated[ScaleType, AfterValidator(convert_to_1plus_range), AfterValidator(check_1plus)]
OnePlusIntRangeType = Annotated[
    ScaleType,
    AfterValidator(convert_to_1plus_range),
    AfterValidator(check_1plus),
    AfterValidator(float2int),
]

OnePlusIntNonDecreasingRangeType = Annotated[
    Tuple[ScalarType, ScalarType],
    AfterValidator(check_1plus),
    AfterValidator(nondecreasing),
    AfterValidator(float2int),
]


def convert_to_0plus_range(value: ScaleType) -> tuple[float, float]:
    return to_tuple(value, low=0)


def check_01(value: tuple[NumericType, NumericType]) -> tuple[NumericType, NumericType]:
    if not all(0 <= x <= 1 for x in value):
        raise ValueError(f"All values should be in [0, 1], got {value} instead")
    return value


ZeroOneRangeType = Annotated[ScaleType, AfterValidator(convert_to_0plus_range), AfterValidator(check_01)]
