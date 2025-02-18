from inspect import Parameter, signature
from typing import Any, Dict, Optional, Tuple, Union

import pytest
from pydantic import BaseModel

import albumentations as A
from albumentations.core.pydantic import (
    NonNegativeFloatRangeType,
    NonNegativeIntRangeType,
    OnePlusFloatRangeType,
    OnePlusIntRangeType,
    SymmetricRangeType,
    ZeroOneRangeType,
    check_range_bounds,
    create_symmetric_range,
    process_non_negative_range,
)
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.core.validation import ValidatedTransformMeta


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, (0.0, 0.0)),
        ((0, 5), (0.0, 5.0)),
    ],
)
def test_process_non_negative_range_valid(
    value: Optional[Tuple[float, float]],
    expected: Tuple[float, float],
) -> None:
    assert process_non_negative_range(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        (-1, 1),  # Invalid because it contains a negative number
        # You can add more invalid cases here as needed
    ],
)
def test_process_non_negative_range_invalid(value: Tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        process_non_negative_range(value)


# Symmetric Range Tests
@pytest.mark.parametrize(
    "value, expected",
    [
        (5, (-5.0, 5.0)),
        ((-1, 1), (-1.0, 1.0)),
    ],
)
def test_create_symmetric_range(
    value: Union[int, Tuple[int, int]],
    expected: Tuple[float, float],
) -> None:
    assert create_symmetric_range(value) == expected


# Correcting the handling of expected exceptions in tests
@pytest.mark.parametrize(
    "value, expected",
    [
        (None, (0.0, 0.0)),
        ((0, 5), (0.0, 5.0)),
    ],
)
def test_process_non_negative_range_with_valid_input(
    value: Optional[Tuple[float, float]],
    expected: Tuple[float, float],
) -> None:
    assert process_non_negative_range(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        (-1, 1),  # Invalid input
    ],
)
def test_process_non_negative_range_with_invalid_input(value: Tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        process_non_negative_range(value)

class ValidationModel(BaseModel):
    non_negative_range_float: Optional[NonNegativeFloatRangeType] = None
    non_negative_range_int: Optional[NonNegativeIntRangeType] = None
    symmetric_range: Optional[SymmetricRangeType] = None
    one_plus_range_float: Optional[OnePlusFloatRangeType] = None
    one_plus_range_int: Optional[OnePlusIntRangeType] = None
    zero_one_range: Optional[ZeroOneRangeType] = None


@pytest.mark.parametrize("probability", [0, 0.5, 1])
def test_probability_valid(probability: float) -> None:
    assert ValidationModel(probability=probability)


@pytest.mark.parametrize("non_negative_range", [(0, 5), 10, None])
def test_non_negative_range_valid(non_negative_range: Union[int, Tuple[int, int]]) -> None:
    assert ValidationModel(non_negative_range_float=non_negative_range)
    assert ValidationModel(non_negative_range_int=non_negative_range)


@pytest.mark.parametrize("symmetric_range", [(-10, 10), 5])
def test_symmetric_range_valid(symmetric_range: Union[int, Tuple[int, int]]) -> None:
    assert ValidationModel(symmetric_range=symmetric_range)


@pytest.mark.parametrize(
    "non_negative_range",
    [(-1, 5), -10],
)  # Invalid non-negative ranges
def test_non_negative_range_invalid(non_negative_range: Union[int, Tuple[int, int]]) -> None:
    with pytest.raises(ValueError):
        ValidationModel(non_negative_range_float=non_negative_range)

    with pytest.raises(ValueError):
        ValidationModel(non_negative_range_int=non_negative_range)


@pytest.mark.parametrize("one_plus_range", [(0, 5), 0.5])  # Invalid 1+ ranges
def test_one_plus_range_invalid(one_plus_range: Union[int, Tuple[int, int]]) -> None:
    with pytest.raises(ValueError):
        ValidationModel(one_plus_range_float=one_plus_range)

    with pytest.raises(ValueError):
        ValidationModel(one_plus_range_int=one_plus_range)


@pytest.mark.parametrize("zero_one_range", [(-0.1, 0.5), 1.1])  # Invalid 0-1 ranges
def test_zero_one_range_invalid(zero_one_range: Union[int, Tuple[int, int]]) -> None:
    with pytest.raises(ValueError):
        ValidationModel(zero_one_range=zero_one_range)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interpolation": 999, "size": (1, 1)},
        {"interpolation": -1, "size": (1, 1)},
        {"scale": -4, "size": (1, 1)},
        {"ratio": (-1, 2), "size": (1, 1)},
        {"size": (-1, 1)},
        {"size": (0, 1)},
    ],
)
def test_RandomResizedCrop(kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        A.RandomResizedCrop(**kwargs)


class MyTransformInitSchema(BaseModel):
    param_a: int
    param_b: float = 1.0


class MyTransform(metaclass=ValidatedTransformMeta):
    class InitSchema(MyTransformInitSchema):
        pass

    def __init__(self, param_a: int, param_b: float = 1.0):
        self.param_a = param_a
        self.param_b = param_b


# Test successful initialization with valid parameters
def test_my_transform_valid_initialization() -> None:
    # This should not raise an exception
    transform = MyTransform(param_a=10, param_b=2.0)
    assert transform.param_a == 10
    assert transform.param_b == 2.0

    # Test with defaults
    transform = MyTransform(param_a=5)
    assert transform.param_a == 5
    assert transform.param_b == 1.0  # Default value


# Test initialization with missing required parameters
def test_my_transform_missing_required_param() -> None:
    with pytest.raises(ValueError):
        MyTransform()


# Test initialization with invalid types
@pytest.mark.parametrize(
    "invalid_a, invalid_b",
    [
        ("not an int", 2.0),  # invalid param_a
        (10, "not a float"),  # invalid param_b
    ],
)
def test_my_transform_invalid_types(invalid_a: int, invalid_b: float) -> None:
    with pytest.raises(ValueError):
        MyTransform(param_a=invalid_a, param_b=invalid_b)


class SimpleTransform(metaclass=ValidatedTransformMeta):
    def __init__(self, param_a: int, param_b: str = "default"):
        self.param_a = param_a
        self.param_b = param_b


def test_transform_without_schema() -> None:
    # Test instantiation with only the required parameter
    transform = SimpleTransform(param_a=10)
    assert transform.param_a == 10
    assert transform.param_b == "default", "Default parameter should be unchanged"

    # Test instantiation with both parameters
    transform = SimpleTransform(param_a=20, param_b="custom")
    assert transform.param_a == 20
    assert transform.param_b == "custom", "Custom parameter should be correctly assigned"

    # Test instantiation with an incorrect type for param_a, acknowledging no validation is performed
    # This demonstrates the class's behavior without InitSchema validation,
    # but since Python does not enforce type annotations at runtime, this won't raise a TypeError.
    transform = SimpleTransform(
        param_a="should not fail due to type annotations not enforcing type checks at runtime",
    )
    assert (
        transform.param_a == "should not fail due to type annotations not enforcing type checks at runtime"
    ), "Parameter accepted without type validation"


class CustomImageTransform(ImageOnlyTransform):
    def __init__(self, custom_param: int, p: float = 0.5):
        super().__init__(p=p)
        self.custom_param = custom_param


def test_custom_image_transform_signature() -> None:
    expected_signature = signature(CustomImageTransform)
    expected_params = expected_signature.parameters

    assert "custom_param" in expected_params
    assert expected_params["custom_param"] == Parameter(
        "custom_param",
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=Parameter.empty,
        annotation=int,
    )

    assert "p" in expected_params
    assert expected_params["p"] == Parameter(
        "p",
        kind=Parameter.POSITIONAL_OR_KEYWORD,
        default=0.5,
        annotation=float,
    )

    # Ensure the correct defaults and types
    assert expected_params["p"].default == 0.5
    assert expected_params["custom_param"].annotation is int


def test_check_range_bounds_doctest():
    # Test the examples from the docstring
    validator = check_range_bounds(0, 1)
    assert validator((0.1, 0.5)) == (0.1, 0.5)
    assert validator((0.1, 0.5, 0.7)) == (0.1, 0.5, 0.7)

    with pytest.raises(ValueError):
        validator((1.1, 0.5))

    validator_exclusive = check_range_bounds(0, 1, max_inclusive=False)
    with pytest.raises(ValueError):
        validator_exclusive((0, 1))

@pytest.mark.parametrize(
    ["min_val", "max_val", "min_inclusive", "max_inclusive", "test_value", "expected"],
    [
        # Basic cases
        (0, 1, True, True, (0, 0.5, 1), (0, 0.5, 1)),  # All inclusive bounds
        (0, 1, False, False, (0.1, 0.5, 0.9), (0.1, 0.5, 0.9)),  # All exclusive bounds

        # None cases
        (0, None, True, True, None, None),  # None input
        (0, None, True, True, (1, 2, 3), (1, 2, 3)),  # Only min bound

        # Different number of elements
        (0, 1, True, True, (0.5,), (0.5,)),  # Single element
        (0, 1, True, True, (0.2, 0.4), (0.2, 0.4)),  # Two elements
        (0, 1, True, True, (0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8)),  # Four elements

        # Edge cases
        (0, 1, True, True, (0, 1), (0, 1)),  # Inclusive bounds
        (-1, 1, True, True, (-1, 0, 1), (-1, 0, 1)),  # Negative values
        (0.5, 1.5, True, True, (0.5, 1.0, 1.5), (0.5, 1.0, 1.5)),  # Float bounds
    ]
)
def test_check_range_bounds_valid(min_val, max_val, min_inclusive, max_inclusive, test_value, expected):
    validator = check_range_bounds(min_val, max_val, min_inclusive, max_inclusive)
    assert validator(test_value) == expected

@pytest.mark.parametrize(
    ["min_val", "max_val", "min_inclusive", "max_inclusive", "test_value", "error_pattern"],
    [
        # Min bound violations
        (0, 1, True, True, (-0.1, 0.5), "must be >= 0"),
        (0, 1, False, True, (0, 0.5), "must be > 0"),

        # Max bound violations
        (0, 1, True, True, (0.5, 1.1), "must be >= 0 and <= 1"),
        (0, 1, True, False, (0.5, 1), "must be >= 0 and < 1"),

        # Both bounds violations
        (0, 1, True, True, (-0.1, 1.1), "must be >= 0 and <= 1"),
        (0, 1, False, False, (0, 1), "must be > 0 and < 1"),

        # Only min bound with violations
        (0, None, True, True, (-1, -0.5), "must be >= 0"),
        (0, None, False, True, (0, 1), "must be > 0"),
    ]
)
def test_check_range_bounds_invalid(min_val, max_val, min_inclusive, max_inclusive, test_value, error_pattern):
    validator = check_range_bounds(min_val, max_val, min_inclusive, max_inclusive)
    with pytest.raises(ValueError, match=error_pattern):
        validator(test_value)

@pytest.mark.parametrize(
    ["min_val", "max_val", "values"],
    [
        (0, 1, [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]),  # Multiple valid tuples
        (0, None, [(1, 2), (3, 4), (5, 6)]),  # Multiple valid tuples with no max
    ]
)
def test_check_range_bounds_multiple_calls(min_val, max_val, values):
    validator = check_range_bounds(min_val, max_val)
    for value in values:
        assert validator(value) == value

def test_check_range_bounds_type_preservation():
    # Test that the function preserves input types
    validator = check_range_bounds(0, 1)

    int_tuple = (0, 1)
    assert isinstance(validator(int_tuple)[0], int)

    float_tuple = (0.5, 0.7)
    assert isinstance(validator(float_tuple)[0], float)
