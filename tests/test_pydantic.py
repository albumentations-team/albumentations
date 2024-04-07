from typing import Optional
import pytest
import cv2

from pydantic import BaseModel
import albumentations as A

from albumentations.core.pydantic import (
    BorderModeType,
    InterpolationType,
    OnePlusRangeType,
    ProbabilityType,
    RangeNonNegativeType,
    RangeSymmetricType,
    ZeroOneRangeType,
    check_valid_interpolation,
    check_valid_border_modes,
    process_non_negative_range,
    create_symmetric_range,
    check_1plus_range,
    check_01_range,
    valid_interpolations,
    valid_border_modes
)

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
    (0.9),  # Invalid input
])
def test_check_1plus_range_with_invalid_input(value):
    with pytest.raises(ValueError):
        check_1plus_range(value)

@pytest.mark.parametrize("value,expected", [
    ((1, 2), (1.0, 2.0)),
    (2, (1.0, 2.0)),
])
def test_check_1plus_range_with_valid_input(value, expected):
    assert check_1plus_range(value) == expected

@pytest.mark.parametrize("value", [
    (-0.1),
    (1.2),
])
def test_check_01_range_with_invalid_input(value):
    with pytest.raises(ValueError):
        check_01_range(value)

@pytest.mark.parametrize("value,expected", [
    ((0, 1), (0.0, 1.0)),
    ((0.3), (0.0, 0.3)),
])
def test_check_01_range_with_valid_input(value, expected):
    assert check_01_range(value) == expected

class ValidationModel(BaseModel):
    interpolation: Optional[InterpolationType] = None
    border_mode: Optional[BorderModeType] = None
    probability: Optional[ProbabilityType] = None
    non_negative_range: Optional[RangeNonNegativeType] = None
    symmetric_range: Optional[RangeSymmetricType] = None
    one_plus_range: Optional[OnePlusRangeType] = None
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
    assert ValidationModel(non_negative_range=non_negative_range)

@pytest.mark.parametrize("symmetric_range", [(-10, 10), 5])
def test_symmetric_range_valid(symmetric_range):
    assert ValidationModel(symmetric_range=symmetric_range)

@pytest.mark.parametrize("one_plus_range", [(1, 5), 2])
def test_one_plus_range_valid(one_plus_range):
    assert ValidationModel(one_plus_range=one_plus_range)

@pytest.mark.parametrize("zero_one_range", [(0, 1), 0.5])
def test_zero_one_range_valid(zero_one_range):
    assert ValidationModel(zero_one_range=zero_one_range)


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
        ValidationModel(non_negative_range=non_negative_range)

@pytest.mark.parametrize("one_plus_range", [(0, 5), 0.5])  # Invalid 1+ ranges
def test_one_plus_range_invalid(one_plus_range):
    with pytest.raises(ValueError):
        ValidationModel(one_plus_range=one_plus_range)

@pytest.mark.parametrize("zero_one_range", [(-0.1, 0.5), 1.1])  # Invalid 0-1 ranges
def test_zero_one_range_invalid(zero_one_range):
    with pytest.raises(ValueError):
        ValidationModel(zero_one_range=zero_one_range)

@pytest.mark.parametrize("kwargs", [{"interpolation": 999, "height": 1, "width": 1},
                                    {"interpolation": -1, "height": 1, "width": 1},
                                    {"scale": -4, "height": 1, "width": 1},
                                    {"ratio": (-1, 2), "height": 1, "width": 1},
                                    {"height": -1, "width": 1},
                                    {"width": 0, "height": 1},])

def test_RandomResizedCrop(kwargs):
    with pytest.raises(ValueError):
        A.RandomResizedCrop(**kwargs)
