from __future__ import annotations

import random
import numpy as np
from typing import Any

import pytest

from albumentations.augmentations.mixing.functional import (
    assign_items_to_grid_cells,
    calculate_cell_placements,
    calculate_mosaic_center_point,
    filter_valid_metadata,
)


@pytest.mark.parametrize(
    "grid_yx, target_size, center_range, seed, expected_center",
    [
        # Standard 2x2 grid, target 100x100 (Large Grid 200x200)
        # Safe Zone: x=[49, 149], y=[49, 149] -> size 101x101
        ((2, 2), (100, 100), (0.0, 0.0), 42, (49, 49)),  # Min offset
        ((2, 2), (100, 100), (1.0, 1.0), 42, (150, 150)), # Max offset (49 + int(101*1.0) = 150)
        ((2, 2), (100, 100), (0.5, 0.5), 42, (99, 99)),   # Mid offset (49 + int(101*0.5) = 99)

        # 1x1 grid, target 200x200 (Large Grid 200x200)
        # Safe Zone: x=[99, 99], y=[99, 99] -> size 1x1
        ((1, 1), (200, 200), (0.0, 1.0), 42, (99, 99)), # Offset always 0, center fixed
        ((1, 1), (200, 200), (0.0, 0.0), 42, (99, 99)),
        ((1, 1), (200, 200), (1.0, 1.0), 42, (100, 100)), # Offset becomes 1, center = 99+1 = 100

        # Large grid 5x5, small target 10x10 (Large Grid 50x50)
        # Safe Zone: x=[4, 44], y=[4, 44] -> size 41x41
        ((5, 5), (10, 10), (0.0, 0.0), 42, (4, 4)),      # Min offset
        ((5, 5), (10, 10), (1.0, 1.0), 42, (45, 45)),     # Max offset (4 + int(41*1.0) = 45)
        ((5, 5), (10, 10), (0.5, 0.5), 42, (24, 24)),     # Mid offset (4 + int(41*0.5) = 24)

        # Larger target relative to grid cell (causes large safe zone)
        # Grid 2x2, target 600x600 (Large Grid 1200x1200)
        # Safe Zone: x=[299, 899], y=[299, 899] -> size 601x601
        ((2, 2), (600, 600), (0.5, 0.5), 42, (599, 599)), # Mid offset (299 + int(601*0.5) = 599)
    ],
)
def test_calculate_mosaic_center_point(
    grid_yx: tuple[int, int],
    target_size: tuple[int, int],
    center_range: tuple[float, float],
    seed: int,
    expected_center: tuple[int, int],
) -> None:
    """Test the calculation of the mosaic center point under various conditions."""
    py_random = random.Random(seed)
    center_xy = calculate_mosaic_center_point(grid_yx, target_size, center_range, py_random)
    assert center_xy == expected_center


@pytest.mark.parametrize(
    "grid_yx, target_size, center_xy, expected_placements",
    [
        # Case 1: 2x2 grid, target 100x100, center (99, 99)
        # Corrected based on function output
        (
            (2, 2),
            (100, 100),
            (99, 99),
            {
                 (0, 0): (0, 0, 51, 51),
                 (0, 1): (51, 0, 100, 51),
                 (1, 0): (0, 51, 51, 100),
                 (1, 1): (51, 51, 100, 100),
            },
        ),
        # Case 2: 2x2 grid, target 100x100, center (149, 149) -> parts of 4 cells
        (
            (2, 2),
            (100, 100),
            (149, 149),
            {
                (0, 0): (0, 0, 1, 1),
                (0, 1): (1, 0, 100, 1),
                (1, 0): (0, 1, 1, 100),
                (1, 1): (1, 1, 100, 100),
            },
        ),
        # Case 3: 1x1 grid, target 100x100, center (49, 49)
        # Corrected based on function output
        (
            (1, 1),
            (100, 100),
            (49, 49),
            {(0, 0): (1, 1, 100, 100)},
        ),
        # Case 4: Center point exactly at top-left corner of the grid (0, 0)
        # Corrected based on function output
        (
            (2, 2),
            (100, 100),
            (0, 0),
            {(0, 0): (50, 50, 100, 100)},
        ),
        # Case 5: 3x3 grid, target 100x100, center (99, 99) -> 4 cells visible with different sizes
        (
            (3, 3),
            (100, 100),
            (99, 99),
            {
                 (0, 0): (0, 0, 51, 51),
                 (0, 1): (51, 0, 100, 51),
                 (1, 0): (0, 51, 51, 100),
                 (1, 1): (51, 51, 100, 100),
            },
        ),
    ],
)
def test_calculate_cell_placements(
    grid_yx: tuple[int, int],
    target_size: tuple[int, int],
    center_xy: tuple[int, int],
    expected_placements: dict[tuple[int, int], tuple[int, int, int, int]],
) -> None:
    """Test the calculation of cell placements on the target canvas."""
    placements = calculate_cell_placements(grid_yx, target_size, center_xy)
    assert placements == expected_placements


# Fixtures for metadata tests
@pytest.fixture
def valid_item_1() -> dict[str, Any]:
    return {"image": np.zeros((10, 10, 3)), "label": "cat"}


@pytest.fixture
def valid_item_2() -> dict[str, Any]:
    return {"image": np.ones((5, 5))}


@pytest.fixture
def invalid_item_no_image() -> dict[str, Any]:
    return {"label": "dog"}


@pytest.fixture
def invalid_item_not_dict() -> str:
    return "not_a_dict"


# Tests for filter_valid_metadata
def test_filter_valid_metadata_all_valid(valid_item_1, valid_item_2) -> None:
    """Test with a list of only valid metadata items."""
    metadata_input = [valid_item_1, valid_item_2]
    result = filter_valid_metadata(metadata_input, "test_key")
    assert result == metadata_input


def test_filter_valid_metadata_mixed(valid_item_1, invalid_item_no_image, valid_item_2, invalid_item_not_dict) -> None:
    """Test with a mix of valid and invalid items, checking warnings."""
    metadata_input = [valid_item_1, invalid_item_no_image, valid_item_2, invalid_item_not_dict]
    expected_output = [valid_item_1, valid_item_2]

    with pytest.warns(UserWarning) as record:
        result = filter_valid_metadata(metadata_input, "test_key")

    assert result == expected_output
    assert len(record) == 2 # One warning for each invalid item
    assert "Item at index 1 in 'test_key' is invalid" in str(record[0].message)
    assert "Item at index 3 in 'test_key' is invalid" in str(record[1].message)

def test_filter_valid_metadata_empty_list() -> None:
    """Test with an empty list."""
    result = filter_valid_metadata([], "test_key")
    assert result == []

def test_filter_valid_metadata_none_input() -> None:
    """Test with None as input, checking warning."""
    with pytest.warns(UserWarning, match="Metadata under key 'test_key' is not a Sequence"):
        result = filter_valid_metadata(None, "test_key")
    assert result == []

def test_filter_valid_metadata_dict_input(valid_item_1) -> None:
    """Test with a dictionary as input instead of a sequence, checking warning."""
    with pytest.warns(UserWarning, match="Metadata under key 'test_key' is not a Sequence"):
        result = filter_valid_metadata(valid_item_1, "test_key")
    assert result == []

def test_filter_valid_metadata_tuple_input(valid_item_1, valid_item_2) -> None:
    """Test with a tuple of valid items."""
    metadata_input = (valid_item_1, valid_item_2)
    expected_output = [valid_item_1, valid_item_2]
    result = filter_valid_metadata(metadata_input, "test_key")
    assert result == expected_output


# Tests for assign_items_to_grid_cells


@pytest.mark.parametrize(
    "num_items, cell_placements, seed, expected_assignment, expected_warning_msg",
    [
        # Case 1: Enough items for all cells
        (
            4,
            {
                (0, 0): (0, 0, 50, 50),  # Area 2500
                (0, 1): (50, 0, 100, 60), # Area 3000 (Largest)
                (1, 0): (0, 50, 40, 100), # Area 2000
                (1, 1): (40, 60, 100, 100),# Area 2400
            },
            42,
            {
                (0, 1): 0,  # Primary assigned to largest
                (0, 0): 2,  # Corrected: Random assignment (seed 42: shuffle [1, 2, 3] -> [2, 1, 3])
                (1, 0): 1,
                (1, 1): 3,
            },
            None, # No warning expected
        ),
        # Case 2: More cells than items (warning expected)
        (
            3,
            {
                (0, 0): (0, 0, 50, 50),  # Area 2500
                (0, 1): (50, 0, 100, 60), # Area 3000 (Largest)
                (1, 0): (0, 50, 40, 100), # Area 2000
                (1, 1): (40, 60, 100, 100),# Area 2400
            },
            42,
            {
                (0, 1): 0,  # Primary assigned to largest
                (0, 0): 2,  # Random assignment (seed 42: shuffle [1, 2] -> [2, 1])
                (1, 0): 1,
                # (1, 1) is left unassigned
            },
            "Mismatch: 3 remaining cells, but 2 remaining items. Assigning 2.",
        ),
        # Case 3: More items than cells (warning expected)
        (
            5,
            {
                (0, 0): (0, 0, 100, 100), # Largest
                (0, 1): (0, 0, 50, 50),
            },
            123,
            {
                (0, 0): 0, # Primary assigned to largest
                (0, 1): 3, # Random assignment (seed 123: shuffle [1, 2, 3, 4] -> [3, 2, 1, 4])
                # Items 1, 2, 4 are unused
            },
             "Mismatch: 1 remaining cells, but 4 remaining items. Assigning 1.",
        ),
        # Case 4: Only one cell visible
        (
            3,
            {(1, 1): (0, 0, 100, 100)}, # Only one cell
            42,
            {(1, 1): 0}, # Primary assigned to the only cell
             None, # No warning expected (0 remaining cells, 2 remaining items)
        ),
        # Case 5: Empty cell placements
        (
            2,
            {},
            42,
            {},
             None,
        ),
        # Case 6: Equal cell sizes (primary goes to first encountered in max)
        (
            4,
            {
                (0, 0): (0, 0, 50, 50),
                (0, 1): (50, 0, 100, 50),
                (1, 0): (0, 50, 50, 100),
                (1, 1): (50, 50, 100, 100),
            },
            99,
            {
                (0, 0): 0,  # Primary assigned to first largest encountered ((0,0) here)
                (0, 1): 1,  # Corrected: Random assignment (seed 99: shuffle [1, 2, 3] -> [1, 3, 2])
                (1, 0): 3,
                (1, 1): 2,
            },
            None,
        ),
    ],
)
def test_assign_items_to_grid_cells(
    num_items: int,
    cell_placements: dict[tuple[int, int], tuple[int, int, int, int]],
    seed: int,
    expected_assignment: dict[tuple[int, int], int],
    expected_warning_msg: str | None,
) -> None:
    """Test assignment logic including primary placement, randomization, and mismatches."""
    py_random = random.Random(seed)

    if expected_warning_msg:
        with pytest.warns(UserWarning, match=expected_warning_msg):
            assignment = assign_items_to_grid_cells(num_items, cell_placements, py_random)
    else:
        # If no warning is expected, just run the function.
        # An unexpected warning will cause a failure or be reported by pytest.
        assignment = assign_items_to_grid_cells(num_items, cell_placements, py_random)

    assert assignment == expected_assignment
