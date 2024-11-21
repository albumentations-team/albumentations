from random import Random
import pytest
from albumentations.augmentations.blur import functional as fblur


@pytest.mark.parametrize(
    "low, high, expected_range",
    [
        (-8, 7, {3, 5, 7}),           # negative low
        (2, 6, {3, 5, 7}),               # even values
        (1, 4, {3, 5}),                  # low < 3
        (4, 4, {5}),                  # same even value
        (3, 3, {3}),                  # same odd value
        (2, 2, {3}),                  # same even value < 3
        (-4, -2, {3}),                # all negative values
    ],
    ids=[
        "negative_low",
        "even_values",
        "low_less_than_3",
        "same_even_value",
        "same_odd_value",
        "same_even_value_less_than_3",
        "all_negative",
    ]
)
def test_sample_odd_from_range(low: int, high: int, expected_range: set[int]):
    """Test sampling odd numbers from a range."""
    random_state = Random(42)

    results = set()
    for _ in range(50):  # Sample multiple times to get all possible values
        value = fblur.sample_odd_from_range(random_state, low, high)
        results.add(value)
        # Verify each value is odd
        assert value % 2 == 1
        # Verify value is >= 3
        assert value >= 3

    assert results == expected_range, f"Failed for low={low}, high={high}"
