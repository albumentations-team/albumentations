from typing import Optional
import warnings
import pytest
import cv2

from pydantic import BaseModel, ValidationError
import albumentations as A
from inspect import signature, Parameter
from albumentations.core.transforms_interface import ImageOnlyTransform

from albumentations.core.pydantic import (
    BorderModeType,
    InterpolationType,
    NonNegativeFloatRangeType,
    NonNegativeIntRangeType,
    OnePlusFloatRangeType,
    OnePlusIntRangeType,
    ProbabilityType,
    SymmetricRangeType,
    ZeroOneRangeType,
    check_valid_interpolation,
    check_valid_border_modes,
    process_non_negative_range,
    create_symmetric_range,
    check_1plus,
    check_01,
    valid_interpolations,
    valid_border_modes
)
from albumentations.core.validation import ValidatedTransformMeta

# Interpolation Tests
@pytest.mark.parametrize("interpolation, exception", [
    (cv2.INTER_NEAREST, False),
    (cv2.INTER_LINEAR, False),
    (999, True),  # Invalid interpolation
])
def test_check_valid_interpolation(interpolation, exception):
    if exception:
        with pytest.raises(ValueError):
            check_valid_interpolation(interpolation)
    else:
        assert check_valid_interpolation(interpolation) == interpolation

# Border Mode Tests
@pytest.mark.parametrize("border_mode, exception", [
    (cv2.BORDER_CONSTANT, False),
    (cv2.BORDER_REPLICATE, False),
    (999, True),  # Invalid border mode
])
def test_check_valid_border_modes(border_mode, exception):
    if exception:
        with pytest.raises(ValueError):
            check_valid_border_modes(border_mode)
    else:
        assert check_valid_border_modes(border_mode) == border_mode

@pytest.mark.parametrize("value, expected", [
    (None, (0.0, 0.0)),
    ((0, 5), (0.0, 5.0)),
])
def test_process_non_negative_range_valid(value, expected):
    assert process_non_negative_range(value) == expected

@pytest.mark.parametrize("value", [
    (-1, 1),  # Invalid because it contains a negative number
    # You can add more invalid cases here as needed
])
def test_process_non_negative_range_invalid(value):
    with pytest.raises(ValueError):
        process_non_negative_range(value)

# Symmetric Range Tests
@pytest.mark.parametrize("value,expected", [
    (5, (5.0, 5.0)),
    ((-1, 1), (-1.0, 1.0)),
])
def test_create_symmetric_range(value, expected):
    assert create_symmetric_range(value) == expected

# Correcting the handling of expected exceptions in tests
@pytest.mark.parametrize("value,expected", [
    (None, (0.0, 0.0)),
    ((0, 5), (0.0, 5.0)),
])
def test_process_non_negative_range_with_valid_input(value, expected):
    assert process_non_negative_range(value) == expected

@pytest.mark.parametrize("value", [
    (-1, 1),  # Invalid input
])
def test_process_non_negative_range_with_invalid_input(value):
    with pytest.raises(ValueError):
        process_non_negative_range(value)

@pytest.mark.parametrize("value,expected", [
    (5, (-5.0, 5.0)),
    ((-1, 1), (-1.0, 1.0)),
])
def test_create_symmetric_range(value, expected):
    assert create_symmetric_range(value) == expected

@pytest.mark.parametrize("value", [
    (0, 0.9),  # Invalid input
    (0, 1.1),  # Invalid input
])
def test_check_1plus_range_with_invalid_input(value):
    with pytest.raises(ValueError):
        check_1plus(value)

@pytest.mark.parametrize("value,expected", [
    ((1, 2), (1.0, 2.0)),
])
def test_check_1plus_range_with_valid_input(value, expected):
    assert check_1plus(value) == expected

@pytest.mark.parametrize("value", [
    (0, -0.1),
    (2, 1.2),
])
def test_check_01_range_with_invalid_input(value):
    with pytest.raises(ValueError):
        check_01(value)

@pytest.mark.parametrize("value,expected", [
    ((0, 1), (0.0, 1.0)),
    ((0, 0.3), (0.0, 0.3)),
])
def test_check_01_range_with_valid_input(value, expected):
    assert check_01(value) == expected

class ValidationModel(BaseModel):
    interpolation: Optional[InterpolationType] = None
    border_mode: Optional[BorderModeType] = None
    probability: Optional[ProbabilityType] = None
    non_negative_range_float: Optional[NonNegativeFloatRangeType] = None
    non_negative_range_int: Optional[NonNegativeIntRangeType] = None
    symmetric_range: Optional[SymmetricRangeType] = None
    one_plus_range_float: Optional[OnePlusFloatRangeType] = None
    one_plus_range_int: Optional[OnePlusIntRangeType] = None
    zero_one_range: Optional[ZeroOneRangeType] = None

# Valid Cases
@pytest.mark.parametrize("interpolation", valid_interpolations)
def test_interpolation_valid(interpolation):
    assert ValidationModel(interpolation=interpolation)

@pytest.mark.parametrize("border_mode", valid_border_modes)
def test_border_mode_valid(border_mode):
    assert ValidationModel(border_mode=border_mode)

@pytest.mark.parametrize("probability", [0, 0.5, 1])
def test_probability_valid(probability):
    assert ValidationModel(probability=probability)

@pytest.mark.parametrize("non_negative_range", [(0, 5), 10, None])
def test_non_negative_range_valid(non_negative_range):
    assert ValidationModel(non_negative_range_float=non_negative_range)
    assert ValidationModel(non_negative_range_int=non_negative_range)

@pytest.mark.parametrize("symmetric_range", [(-10, 10), 5])
def test_symmetric_range_valid(symmetric_range):
    assert ValidationModel(symmetric_range=symmetric_range)

# Invalid Cases
@pytest.mark.parametrize("interpolation", [999, -1])  # Invalid interpolation values
def test_interpolation_invalid(interpolation):
    with pytest.raises(ValueError):
        ValidationModel(interpolation=interpolation)

@pytest.mark.parametrize("border_mode", [999, -1])  # Invalid border mode values
def test_border_mode_invalid(border_mode):
    with pytest.raises(ValueError):
        ValidationModel(border_mode=border_mode)

@pytest.mark.parametrize("probability", [-0.1, 1.1])  # Invalid probabilities
def test_probability_invalid(probability):
    with pytest.raises(ValueError):
        ValidationModel(probability=probability)

@pytest.mark.parametrize("non_negative_range", [(-1, 5), -10])  # Invalid non-negative ranges
def test_non_negative_range_invalid(non_negative_range):
    with pytest.raises(ValueError):
        ValidationModel(non_negative_range_float=non_negative_range)

    with pytest.raises(ValueError):
        ValidationModel(non_negative_range_int=non_negative_range)

@pytest.mark.parametrize("one_plus_range", [(0, 5), 0.5])  # Invalid 1+ ranges
def test_one_plus_range_invalid(one_plus_range):
    with pytest.raises(ValueError):
        ValidationModel(one_plus_range_float=one_plus_range)

    with pytest.raises(ValueError):
        ValidationModel(one_plus_range_int=one_plus_range)


@pytest.mark.parametrize("zero_one_range", [(-0.1, 0.5), 1.1])  # Invalid 0-1 ranges
def test_zero_one_range_invalid(zero_one_range):
    with pytest.raises(ValueError):
        ValidationModel(zero_one_range=zero_one_range)

@pytest.mark.parametrize("kwargs", [{"interpolation": 999, "size": (1, 1)},
                                    {"interpolation": -1, "size": (1, 1)},
                                    {"scale": -4, "size": (1, 1)},
                                    {"ratio": (-1, 2), "size": (1, 1)},
                                    {"size": (-1, 1)},
                                    {"size": (0, 1)},])

def test_RandomResizedCrop(kwargs):
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
def test_my_transform_valid_initialization():
    # This should not raise an exception
    transform = MyTransform(param_a=10, param_b=2.0)
    assert transform.param_a == 10
    assert transform.param_b == 2.0

    # Test with defaults
    transform = MyTransform(param_a=5)
    assert transform.param_a == 5
    assert transform.param_b == 1.0  # Default value

# Test initialization with missing required parameters
def test_my_transform_missing_required_param():
    with pytest.raises(ValueError):
        MyTransform()

# Test initialization with invalid types
@pytest.mark.parametrize("invalid_a, invalid_b", [
    ("not an int", 2.0),  # invalid param_a
    (10, "not a float"),  # invalid param_b
])
def test_my_transform_invalid_types(invalid_a, invalid_b):
    with pytest.raises(ValueError):
        MyTransform(param_a=invalid_a, param_b=invalid_b)


class SimpleTransform(metaclass=ValidatedTransformMeta):
    def __init__(self, param_a: int, param_b: str = "default"):
        self.param_a = param_a
        self.param_b = param_b

def test_transform_without_schema():
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
    transform = SimpleTransform(param_a="should not fail due to type annotations not enforcing type checks at runtime")
    assert transform.param_a == "should not fail due to type annotations not enforcing type checks at runtime", \
        "Parameter accepted without type validation"

@pytest.mark.parametrize("invalid_a, invalid_b", [
    ("not an int", 2.0),  # invalid param_a
    (10, "not a float"),  # invalid param_b
])
def test_my_transform_invalid_types(invalid_a, invalid_b):
    with pytest.raises(ValidationError):
        MyTransform(param_a=invalid_a, param_b=invalid_b)


class CustomImageTransform(ImageOnlyTransform):
    def __init__(self, custom_param: int, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.custom_param = custom_param


def test_custom_image_transform_signature():
    expected_signature = signature(CustomImageTransform)
    expected_params = expected_signature.parameters

    assert 'custom_param' in expected_params
    assert expected_params['custom_param'] == Parameter('custom_param', kind=Parameter.POSITIONAL_OR_KEYWORD, default=Parameter.empty, annotation=int)

    assert 'always_apply' in expected_params
    assert expected_params['always_apply'] == Parameter('always_apply', kind=Parameter.POSITIONAL_OR_KEYWORD, default=False, annotation=bool)

    assert 'p' in expected_params
    assert expected_params['p'] == Parameter('p', kind=Parameter.POSITIONAL_OR_KEYWORD, default=0.5, annotation=float)

    # Ensure the correct defaults and types
    assert expected_params['always_apply'].default is False
    assert expected_params['p'].default == 0.5
    assert expected_params['custom_param'].annotation is int


def test_wrong_argument() -> None:
    """Test that pas Transform will get warning"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        transform = A.Crop(wrong_param=10)
        assert not hasattr(transform, "wrong_param")
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert str(w[0].message) == "Argument 'wrong_param' is not valid and will be ignored."
    warnings.resetwarnings()
