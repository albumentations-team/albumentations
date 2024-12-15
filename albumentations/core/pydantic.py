from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, TypeVar, overload

import cv2
from pydantic import Field
from pydantic.functional_validators import AfterValidator

from albumentations.core.types import Number, ScaleFloatType, ScaleIntType, ScaleType
from albumentations.core.utils import to_tuple

valid_interpolations = {
    cv2.INTER_NEAREST,
    cv2.INTER_NEAREST_EXACT,
    cv2.INTER_LINEAR,
    cv2.INTER_CUBIC,
    cv2.INTER_AREA,
    cv2.INTER_LANCZOS4,
    cv2.INTER_LINEAR_EXACT,
    cv2.INTER_MAX,
}


def check_valid_interpolation(value: int) -> int:
    if value not in valid_interpolations:
        raise ValueError(f"Interpolation should be one of {valid_interpolations}, got {value} instead")
    return value


InterpolationType = Annotated[int, Field(description="Interpolation"), AfterValidator(check_valid_interpolation)]

valid_border_modes = {
    cv2.BORDER_CONSTANT,  # Adds a constant colored border. The value should be given as next argument.
    cv2.BORDER_REPLICATE,  # Replicates the last element.
    cv2.BORDER_REFLECT,  # Mirrors the image border without repeating the last element.
    cv2.BORDER_WRAP,  # Wraps the image around like a tiled texture.
    cv2.BORDER_REFLECT_101,  # Also known as cv2.BORDER_DEFAULT. Mirrors the image border, repeating the last element.
    cv2.BORDER_REFLECT101,  # Also known as cv2.BORDER_DEFAULT. Mirrors the image border, repeating the last element.
    cv2.BORDER_TRANSPARENT,  # Makes the border transparent
}


def check_valid_border_modes(value: int) -> int:
    if value not in valid_interpolations:
        raise ValueError(f"Border mode should be one of {valid_border_modes}, got {value} instead")
    return value


def nondecreasing(value: tuple[Number, Number]) -> tuple[Number, Number]:
    if not value[0] <= value[1]:
        raise ValueError(f"First value should be less than the second value, got {value} instead")
    return value


BorderModeType = Annotated[int, Field(description="Border Mode"), AfterValidator(check_valid_border_modes)]

ProbabilityType = Annotated[float, Field(description="Probability of applying the transform", ge=0, le=1)]


def process_non_negative_range(value: ScaleType | None) -> tuple[float, float]:
    result = to_tuple(value if value is not None else 0, 0)
    if not all(x >= 0 for x in result):
        msg = "All values in the non negative range should be non negative"
        raise ValueError(msg)
    return result


def float2int(value: tuple[float, float]) -> tuple[int, int]:
    return int(value[0]), int(value[1])


NonNegativeFloatRangeType = Annotated[
    ScaleType,
    AfterValidator(process_non_negative_range),
    AfterValidator(nondecreasing),
]
NonNegativeIntRangeType = Annotated[ScaleType, AfterValidator(process_non_negative_range), AfterValidator(float2int)]


@overload
def create_symmetric_range(value: ScaleIntType) -> tuple[int, int]: ...


@overload
def create_symmetric_range(value: ScaleFloatType) -> tuple[float, float]: ...


def create_symmetric_range(value: ScaleType) -> tuple[int, int] | tuple[float, float]:
    return to_tuple(value)


SymmetricRangeType = Annotated[ScaleType, AfterValidator(create_symmetric_range)]


def convert_to_1plus_range(value: ScaleType) -> tuple[float, float]:
    return to_tuple(value, low=1)


def convert_to_0plus_range(value: ScaleType) -> tuple[float, float]:
    return to_tuple(value, low=0)


def repeat_if_scalar(value: ScaleType) -> tuple[float, float]:
    return (value, value) if isinstance(value, (int, float)) else value


T = TypeVar("T", int, float)


def check_range_bounds(
    min_val: Number,
    max_val: Number | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> Callable[[tuple[T, ...] | None], tuple[T, ...] | None]:
    """Validates that all values in a tuple are within specified bounds.

    Args:
        min_val: Minimum allowed value
        max_val: Maximum allowed value. If None, only lower bound is checked.
        min_inclusive: If True, min_val is inclusive (>=). If False, exclusive (>).
        max_inclusive: If True, max_val is inclusive (<=). If False, exclusive (<).

    Returns:
        Validator function that checks if all values in tuple are within bounds.
        Returns None if input is None.

    Raises:
        ValueError: If any value in tuple is outside the allowed range

    Examples:
        >>> validator = check_range_bounds(0, 1)  # For [0, 1] range
        >>> validator((0.1, 0.5))  # Valid 2D
        (0.1, 0.5)
        >>> validator((0.1, 0.5, 0.7))  # Valid 3D
        (0.1, 0.5, 0.7)
        >>> validator((1.1, 0.5))  # Raises ValueError - outside range
        >>> validator = check_range_bounds(0, 1, max_inclusive=False)  # For [0, 1) range
        >>> validator((0, 1))  # Raises ValueError - 1 not included
    """

    def validator(value: tuple[T, ...] | None) -> tuple[T, ...] | None:
        if value is None:
            return None

        min_op = (lambda x, y: x >= y) if min_inclusive else (lambda x, y: x > y)
        max_op = (lambda x, y: x <= y) if max_inclusive else (lambda x, y: x < y)

        if max_val is None:
            if not all(min_op(x, min_val) for x in value):
                op_symbol = ">=" if min_inclusive else ">"
                raise ValueError(f"All values in {value} must be {op_symbol} {min_val}")
        else:
            min_symbol = ">=" if min_inclusive else ">"
            max_symbol = "<=" if max_inclusive else "<"
            if not all(min_op(x, min_val) and max_op(x, max_val) for x in value):
                raise ValueError(f"All values in {value} must be {min_symbol} {min_val} and {max_symbol} {max_val}")
        return value

    return validator


ZeroOneRangeType = Annotated[
    ScaleType,
    AfterValidator(convert_to_0plus_range),
    AfterValidator(check_range_bounds(0, 1)),
    AfterValidator(nondecreasing),
]


OnePlusFloatRangeType = Annotated[
    ScaleType,
    AfterValidator(convert_to_1plus_range),
    AfterValidator(check_range_bounds(1, None)),
]
OnePlusIntRangeType = Annotated[
    ScaleType,
    AfterValidator(convert_to_1plus_range),
    AfterValidator(check_range_bounds(1, None)),
    AfterValidator(float2int),
]

OnePlusIntNonDecreasingRangeType = Annotated[
    tuple[Number, Number],
    AfterValidator(check_range_bounds(1, None)),
    AfterValidator(nondecreasing),
    AfterValidator(float2int),
]
