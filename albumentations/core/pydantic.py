"""Module containing Pydantic validation utilities for Albumentations.

This module provides a collection of validators and utility functions used for validating
parameters in the Pydantic models throughout the Albumentations library. It includes
functions for ensuring numeric ranges are valid, handling type conversions, and creating
standardized validation patterns that are reused across the codebase.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, TypeVar, Union, overload

from pydantic.functional_validators import AfterValidator

from albumentations.core.type_definitions import Number
from albumentations.core.utils import to_tuple


def nondecreasing(value: tuple[Number, Number]) -> tuple[Number, Number]:
    """Ensure a tuple of two numbers is in non-decreasing order.

    Args:
        value (tuple[Number, Number]): Tuple of two numeric values to validate.

    Returns:
        tuple[Number, Number]: The original tuple if valid.

    Raises:
        ValueError: If the first value is greater than the second value.

    """
    if not value[0] <= value[1]:
        raise ValueError(f"First value should be less than the second value, got {value} instead")
    return value


def process_non_negative_range(value: tuple[float, float] | float | None) -> tuple[float, float]:
    """Process and validate a non-negative range.

    Args:
        value (tuple[float, float] | float | None): Value to process. Can be:
            - A tuple of two floats
            - A single float (converted to symmetric range)
            - None (defaults to 0)

    Returns:
        tuple[float, float]: Validated non-negative range.

    Raises:
        ValueError: If any values in the range are negative.

    """
    result = to_tuple(value if value is not None else 0, 0)
    if not all(x >= 0 for x in result):
        msg = "All values in the non negative range should be non negative"
        raise ValueError(msg)
    return result


def float2int(value: tuple[float, float]) -> tuple[int, int]:
    """Convert a tuple of floats to a tuple of integers.

    Args:
        value (tuple[float, float]): Tuple of two float values.

    Returns:
        tuple[int, int]: Tuple of two integer values.

    """
    return int(value[0]), int(value[1])


NonNegativeFloatRangeType = Annotated[
    Union[tuple[float, float], float],
    AfterValidator(process_non_negative_range),
    AfterValidator(nondecreasing),
]

NonNegativeIntRangeType = Annotated[
    Union[tuple[int, int], int],
    AfterValidator(process_non_negative_range),
    AfterValidator(nondecreasing),
    AfterValidator(float2int),
]


@overload
def create_symmetric_range(value: tuple[int, int] | int) -> tuple[int, int]: ...


@overload
def create_symmetric_range(value: tuple[float, float] | float) -> tuple[float, float]: ...


def create_symmetric_range(value: tuple[float, float] | float) -> tuple[float, float]:
    """Create a symmetric range around zero or use provided range.

    Args:
        value (tuple[float, float] | float): Input value, either:
            - A tuple of two floats (used directly)
            - A single float (converted to (-value, value))

    Returns:
        tuple[float, float]: Symmetric range.

    """
    return to_tuple(value)


SymmetricRangeType = Annotated[Union[tuple[float, float], float], AfterValidator(create_symmetric_range)]


def convert_to_1plus_range(value: tuple[float, float] | float) -> tuple[float, float]:
    """Convert value to a range with lower bound of 1.

    Args:
        value (tuple[float, float] | float): Input value.

    Returns:
        tuple[float, float]: Range with minimum value of at least 1.

    """
    return to_tuple(value, low=1)


def convert_to_0plus_range(value: tuple[float, float] | float) -> tuple[float, float]:
    """Convert value to a range with lower bound of 0.

    Args:
        value (tuple[float, float] | float): Input value.

    Returns:
        tuple[float, float]: Range with minimum value of at least 0.

    """
    return to_tuple(value, low=0)


def repeat_if_scalar(value: tuple[float, float] | float) -> tuple[float, float]:
    """Convert a scalar value to a tuple by repeating it, or return the tuple as is.

    Args:
        value (tuple[float, float] | float): Input value, either a scalar or tuple.

    Returns:
        tuple[float, float]: If input is scalar, returns (value, value), otherwise returns input unchanged.

    """
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
        min_val (int | float):
            Minimum allowed value.
        max_val (int | float | None):
            Maximum allowed value. If None, only lower bound is checked.
        min_inclusive (bool):
            If True, min_val is inclusive (>=). If False, exclusive (>).
        max_inclusive (bool):
            If True, max_val is inclusive (<=). If False, exclusive (<).

    Returns:
        Callable[[tuple[T, ...] | None], tuple[T, ...] | None]: Validator function that
            checks if all values in tuple are within bounds. Returns None if input is None.

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
    Union[tuple[float, float], float],
    AfterValidator(convert_to_0plus_range),
    AfterValidator(check_range_bounds(0, 1)),
    AfterValidator(nondecreasing),
]


OnePlusFloatRangeType = Annotated[
    Union[tuple[float, float], float],
    AfterValidator(convert_to_1plus_range),
    AfterValidator(check_range_bounds(1, None)),
]
OnePlusIntRangeType = Annotated[
    Union[tuple[float, float], float],
    AfterValidator(convert_to_1plus_range),
    AfterValidator(check_range_bounds(1, None)),
    AfterValidator(float2int),
]

OnePlusIntNonDecreasingRangeType = Annotated[
    tuple[int, int],
    AfterValidator(check_range_bounds(1, None)),
    AfterValidator(nondecreasing),
    AfterValidator(float2int),
]
