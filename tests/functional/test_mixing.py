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
    process_cell_geometry,
    process_all_mosaic_geometries,
    assemble_mosaic_from_processed_cells,
)


@pytest.mark.parametrize(
    "grid_yx, target_size, center_range, seed, expected_center",
    [
        # Standard 2x2 grid, target 100x100 (Large Grid 200x200)
        # Safe Zone: x=[49, 149], y=[49, 149] -> size 101x101
        ((2, 2), (100, 100), (0.0, 0.0), 42, (50, 50)),  # Min offset
        ((2, 2), (100, 100), (1.0, 1.0), 42, (150, 150)), # Max offset (49 + int(101*1.0) = 150)
        ((2, 2), (100, 100), (0.5, 0.5), 42, (100, 100)),   # Mid offset (49 + int(101*0.5) = 99)

        # 1x1 grid, target 200x200 (Large Grid 200x200)
        # Safe Zone: x=[99, 99], y=[99, 99] -> size 1x1
        ((1, 1), (200, 200), (0.0, 1.0), 42, (100, 100)), # Offset always 0, center fixed
        ((1, 1), (200, 200), (0.0, 0.0), 42, (100, 100)),
        ((1, 1), (200, 200), (1.0, 1.0), 42, (100, 100)), # Offset becomes 1, center = 99+1 = 100

        # Large grid 5x5, small target 10x10 (Large Grid 50x50)
        # Safe Zone: x=[4, 44], y=[4, 44] -> size 41x41
        ((5, 5), (10, 10), (0.0, 0.0), 42, (5, 5)),      # Min offset
        ((5, 5), (10, 10), (1.0, 1.0), 42, (45, 45)),
        ((5, 5), (10, 10), (0.5, 0.5), 42, (25, 25)),

        # Larger target relative to grid cell (causes large safe zone)
        # Grid 2x2, target 600x600 (Large Grid 1200x1200)
        # Safe Zone: x=[299, 899], y=[299, 899] -> size 601x601
        ((2, 2), (600, 600), (0.5, 0.5), 42, (600, 600)),
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
        ((1, 1), (100, 100), (50, 50), [(0, 0, 100, 100)]),
        # Case 1: 2x2 grid, target 100x100,
        (
            (2, 2),
            (100, 100),
            (99, 99),
            [(0, 0, 51, 51), (51, 0, 100, 51), (0, 51, 51, 100), (51, 51, 100, 100)]
        ),
        # Case 2:
        (
            (2, 2),
            (100, 100),
            (149, 149),
            [(0, 0, 1, 1), (1, 0, 100, 1), (0, 1, 1, 100), (1, 1, 100, 100)]
        ),
        # Case 5: 3x3 grid, target 100x100
        # Linspace Y: [0, 33, 67, 100], X: [0, 33, 67, 100]
        (
            (3, 3),
            (100, 100),
            (99, 99),
            [(0, 0, 51, 51), (51, 0, 100, 51), (0, 51, 51, 100), (51, 51, 100, 100)]
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
    assert placements == expected_placements, f"Placements {placements} do not match expected {expected_placements}"


    # Check for exact coverage of the target area
    target_h, target_w = target_size
    coverage_mask = np.zeros((target_h, target_w), dtype=bool)
    total_placement_area = 0

    for (x1, y1, x2, y2) in placements:
        # Ensure placement is within bounds (should be guaranteed by calculate_cell_placements)
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(target_w, x2), min(target_h, y2)

        # Check for overlaps by verifying the mask area is currently False before setting to True
        # If the design *guarantees* no overlaps, this check can be simplified/removed.
        # For now, let's assume non-overlap is expected and check it.
        if np.any(coverage_mask[y1_c:y2_c, x1_c:x2_c]):
             pytest.fail(f"Overlapping placement detected: {(x1, y1, x2, y2)} in target_size {target_size}")

        coverage_mask[y1_c:y2_c, x1_c:x2_c] = True
        total_placement_area += (x2_c - x1_c) * (y2_c - y1_c)

    # Assert that the total area covered by placements equals the target area
    expected_area = target_h * target_w
    assert total_placement_area == expected_area, \
        f"Total placement area {total_placement_area} does not match target area {expected_area}"

    # Assert that the entire coverage mask is True (ensures no gaps)
    assert np.all(coverage_mask), "Coverage mask has gaps (False values)"


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
    "num_items, cell_placements, seed, expected_assignment",
    [
        # Case 1: Enough items for all cells
        (
            4,
            [(0, 0, 50, 50), (50, 0, 100, 60), (0, 50, 40, 100), (40, 60, 100, 100)],
            42,
            {
                (50, 0, 100, 60): 0,  # Primary assigned to largest area placement
                (0, 0, 50, 50): 2,    # Random assignment (seed 42: shuffle [1, 2, 3] -> [2, 1, 3])
                (0, 50, 40, 100): 1,
                (40, 60, 100, 100): 3,
            },
        ),
        # Case 2: More cells than items
        (
            3,
            [(0, 0, 50, 50), (50, 0, 100, 60), (0, 50, 40, 100), (40, 60, 100, 100)],
            42,
            {
                (50, 0, 100, 60): 0,  # Primary assigned to largest
                (0, 0, 50, 50): 2,    # Random assignment (seed 42: shuffle [1, 2] -> [2, 1])
                (0, 50, 40, 100): 1,
                # Placement (40, 60, 100, 100) is left unassigned
            },
        ),
        # Case 3: More items than cells
        (
            5,
            [(0, 0, 100, 100), (0, 0, 50, 50)], # List of placements
            123,
            {
                (0, 0, 100, 100): 0, # Primary assigned to largest
                (0, 0, 50, 50): 3,    # Random assignment (seed 123: shuffle [1, 2, 3, 4] -> [3, 2, 1, 4])
                # Items 1, 2, 4 are unused
            },
        ),
        # Case 4: Only one cell visible
        (
            3,
            [(0, 0, 100, 100)], # List with one placement
            42,
            {(0, 0, 100, 100): 0}, # Primary assigned to the only placement
        ),
        # Case 5: Empty cell placements
        (
            2,
            [], # Empty list
            42,
            {},
        ),
        # Case 6: Equal cell sizes
        (
            4,
            [(0, 0, 50, 50), (50, 0, 100, 50), (0, 50, 50, 100), (50, 50, 100, 100)],
            99,
            {
                (0, 0, 50, 50): 0,      # Primary assigned to first largest encountered ((0,0) here)
                (50, 0, 100, 50): 1,    # Random assignment (seed 99: shuffle [1, 2, 3] -> [1, 3, 2])
                (0, 50, 50, 100): 3,
                (50, 50, 100, 100): 2,
            },
        ),
    ],
)
def test_assign_items_to_grid_cells(
    num_items: int,
    cell_placements: list[tuple[int, int, int, int]],
    seed: int,
    expected_assignment: dict[tuple[int, int, int, int], int],
) -> None:
    """Test assignment logic including primary placement and randomization."""
    py_random = random.Random(seed)

    # Directly call the function without checking for warnings
    assignment = assign_items_to_grid_cells(num_items, cell_placements, py_random)

    assert assignment == expected_assignment


# Helper fixtures for process_cell_geometry tests
@pytest.fixture
def base_item_geom() -> dict[str, Any]:
    """A standard 100x100 image item for geometry tests."""
    return {
        "image": np.arange(100 * 100 * 3).reshape(100, 100, 3).astype(np.uint8),
        "mask": (np.arange(100 * 100).reshape(100, 100) % 2).astype(np.uint8),
        "bboxes": None, # Not testing annotation geometry here
        "keypoints": None,
    }

@pytest.fixture
def small_item_geom() -> dict[str, Any]:
    """A smaller 50x50 image item for geometry tests."""
    return {
        "image": np.arange(50 * 50 * 3).reshape(50, 50, 3).astype(np.uint8),
        "mask": (np.arange(50 * 50).reshape(50, 50) % 3).astype(np.uint8),
        "bboxes": None,
        "keypoints": None,
    }


# Tests for process_cell_geometry

def test_process_cell_geometry_identity(base_item_geom) -> None:
    """Test identity case: target size matches item size."""
    item = base_item_geom
    target_h, target_w = 100, 100

    processed = process_cell_geometry(item, target_h, target_w, fill=0, fill_mask=0)

    assert processed["image"].shape == (target_h, target_w, 3)
    assert processed["mask"].shape == (target_h, target_w)
    np.testing.assert_array_equal(processed["image"], item["image"])
    np.testing.assert_array_equal(processed["mask"], item["mask"])

def test_process_cell_geometry_crop(base_item_geom) -> None:
    """Test cropping case: target size is smaller than item size."""
    item = base_item_geom
    target_h, target_w = 60, 50

    processed = process_cell_geometry(item, target_h, target_w, fill=0, fill_mask=0)

    assert processed["image"].shape == (target_h, target_w, 3)
    assert processed["mask"].shape == (target_h, target_w)

    # Exact content depends on RandomCrop, but check if it's a subset of the original
    # We can check if the sum of the processed is less than the original sum
    # (This is a weak check, but better than nothing without mocking RandomCrop)
    assert np.sum(processed["image"]) < np.sum(item["image"])
    assert np.sum(processed["mask"]) < np.sum(item["mask"])

def test_process_cell_geometry_pad(small_item_geom) -> None:
    """Test padding case: target size is larger than item size."""
    item = small_item_geom # 50x50
    target_h, target_w = 70, 80
    fill_value = 111
    mask_fill_value = 5

    processed = process_cell_geometry(item, target_h, target_w, fill=fill_value, fill_mask=mask_fill_value)

    assert processed["image"].shape == (target_h, target_w, 3)
    assert processed["mask"].shape == (target_h, target_w)

    # Check if top-left corner matches original
    np.testing.assert_array_equal(processed["image"][:50, :50], item["image"])
    np.testing.assert_array_equal(processed["mask"][:50, :50], item["mask"])

    # Check padding values
    assert np.all(processed["image"][50:, 50:] == fill_value)
    assert np.all(processed["mask"][50:, 50:] == mask_fill_value)

# Note: Annotations (bboxes, keypoints) are not directly tested here
# as process_cell_geometry relies on the Compose([RandomCrop(...)]) call,
# and testing the RandomCrop's annotation handling is done elsewhere.

# Tests for process_all_mosaic_geometries

# Use the same fixtures as process_cell_geometry tests
# (base_item_geom, small_item_geom)

@pytest.fixture
def items_list_geom(base_item_geom, small_item_geom) -> list[dict[str, Any]]:
    return [base_item_geom, small_item_geom]

def test_process_all_geometry_identity(base_item_geom) -> None:
    """Test process_all for a single cell identity case."""
    # Setup: Map the placement directly to the item index
    placement_to_item_index = {(0, 0, 100, 100): 0}
    final_items = [base_item_geom]

    processed = process_all_mosaic_geometries(
        placement_to_item_index=placement_to_item_index,
        final_items_for_grid=final_items,
        # Removed cell_placements argument
        fill=0,
        fill_mask=0,
    )

    assert len(processed) == 1
    placement_key = (0, 0, 100, 100)
    assert placement_key in processed

    processed_item = processed[placement_key]
    assert processed_item["image"].shape == (100, 100, 3)
    assert processed_item["mask"].shape == (100, 100)
    # process_cell_geometry should handle identity correctly now
    np.testing.assert_array_equal(processed_item["image"], base_item_geom["image"])
    np.testing.assert_array_equal(processed_item["mask"], base_item_geom["mask"])

def test_process_all_geometry_crop(base_item_geom) -> None:
    """Test process_all for a single cell requiring cropping."""
    # Setup: Map the placement directly to the item index
    placement_to_item_index = {(10, 20, 60, 80): 0} # Placement is 50x60
    final_items = [base_item_geom] # Item is 100x100

    processed = process_all_mosaic_geometries(
        placement_to_item_index=placement_to_item_index,
        final_items_for_grid=final_items,
        # Removed cell_placements argument
        fill=0,
        fill_mask=0,
    )

    assert len(processed) == 1
    placement_key = (10, 20, 60, 80)
    assert placement_key in processed
    processed_item = processed[placement_key]
    assert processed_item["image"].shape == (60, 50, 3) # Target H=60, W=50
    assert processed_item["mask"].shape == (60, 50)

def test_process_all_geometry_pad(small_item_geom) -> None:
    """Test process_all for a single cell requiring padding."""
    # Setup: Map the placement directly to the item index
    placement_to_item_index = {(0, 0, 80, 70): 0} # Placement 80x70
    final_items = [small_item_geom]  # Item is 50x50
    fill_value = 111
    mask_fill_value = 5

    processed = process_all_mosaic_geometries(
        placement_to_item_index=placement_to_item_index,
        final_items_for_grid=final_items,
        # Removed cell_placements argument
        fill=fill_value,
        fill_mask=mask_fill_value,
    )

    assert len(processed) == 1
    placement_key = (0, 0, 80, 70)
    assert placement_key in processed
    processed_item = processed[placement_key]
    assert processed_item["image"].shape == (70, 80, 3) # Target H=70, W=80
    assert processed_item["mask"].shape == (70, 80)
    # Check if top-left matches (assuming pad_position="top_left" in process_cell_geometry)
    np.testing.assert_array_equal(processed_item["image"][:50, :50], small_item_geom["image"])
    np.testing.assert_array_equal(processed_item["mask"][:50, :50], small_item_geom["mask"])
    assert np.all(processed_item["image"][50:, 50:] == fill_value)
    assert np.all(processed_item["mask"][50:, 50:] == mask_fill_value)

def test_process_all_geometry_multiple_cells(items_list_geom) -> None:
    """Test process_all processing two different items for two cells."""
    # Setup: Map placements directly to item indices
    placement_to_item_index = {
        (0, 0, 50, 50): 0,    # Crop base_item_geom (idx 0) to 50x50
        (50, 0, 110, 60): 1,  # Pad small_item_geom (idx 1) to 60x60
    }
    final_items = items_list_geom

    processed = process_all_mosaic_geometries(
        placement_to_item_index=placement_to_item_index,
        final_items_for_grid=final_items,
        # Removed cell_placements argument
        fill=0,
        fill_mask=0,
    )

    assert len(processed) == 2
    placement1 = (0, 0, 50, 50)
    placement2 = (50, 0, 110, 60)
    assert placement1 in processed
    assert placement2 in processed

    # Check cell 1 (cropped from base_item_geom)
    processed1 = processed[placement1]
    assert processed1["image"].shape == (50, 50, 3)
    assert processed1["mask"].shape == (50, 50)

    # Check cell 2 (padded small_item_geom)
    processed2 = processed[placement2]
    assert processed2["image"].shape == (60, 60, 3) # Target H=60, W=60
    assert processed2["mask"].shape == (60, 60)
    np.testing.assert_array_equal(processed2["image"][:50, :50], items_list_geom[1]["image"])
    np.testing.assert_array_equal(processed2["mask"][:50, :50], items_list_geom[1]["mask"])
    assert np.all(processed2["image"][50:, 50:] == 0)
    assert np.all(processed2["mask"][50:, 50:] == 0)


# Tests for assemble_mosaic_from_processed_cells

@pytest.fixture
def processed_cell_data_1() -> dict[str, Any]:
    return {
        "image": np.ones((50, 50, 3), dtype=np.uint8) * 1,
        "mask": np.ones((50, 50), dtype=np.uint8) * 11,
    }

@pytest.fixture
def processed_cell_data_2() -> dict[str, Any]:
    return {
        "image": np.ones((60, 60, 3), dtype=np.uint8) * 2,
        "mask": np.ones((60, 60), dtype=np.uint8) * 22,
    }

def test_assemble_single_cell(processed_cell_data_1) -> None:
    """Test assembling a mosaic from a single processed cell (identity case)."""
    processed_cells = {(0, 0, 50, 50): processed_cell_data_1}
    target_shape = (50, 50, 3) # RGB image
    dtype = np.uint8

    # Test for image
    canvas_img = assemble_mosaic_from_processed_cells(processed_cells, target_shape, dtype, "image")
    assert canvas_img.shape == target_shape
    assert canvas_img.dtype == dtype
    np.testing.assert_array_equal(canvas_img, processed_cell_data_1["image"])

    # Test for mask
    canvas_mask = assemble_mosaic_from_processed_cells(
        processed_cells, target_shape[:2], dtype, "mask"
    )
    assert canvas_mask.shape == target_shape[:2]
    assert canvas_mask.dtype == dtype
    np.testing.assert_array_equal(canvas_mask, processed_cell_data_1["mask"])
    # Ensure canvas is exactly the cell data (no extra non-zero pixels)
    assert np.count_nonzero(canvas_img) == np.count_nonzero(processed_cell_data_1["image"])
    assert np.count_nonzero(canvas_mask) == np.count_nonzero(processed_cell_data_1["mask"])

def test_assemble_multiple_non_overlapping(processed_cell_data_1, processed_cell_data_2) -> None:
    """Test assembling from multiple non-overlapping cells."""
    processed_cells = {
        (0, 0, 50, 50): processed_cell_data_1,    # Top-left 50x50
        (50, 60, 110, 120): processed_cell_data_2, # Bottom-right 60x60
    }
    target_shape = (120, 120, 3) # Target canvas size
    dtype = np.uint8

    # Test for image
    canvas_img = assemble_mosaic_from_processed_cells(processed_cells, target_shape, dtype, "image")
    assert canvas_img.shape == target_shape
    # Check content of cell 1
    np.testing.assert_array_equal(canvas_img[0:50, 0:50], processed_cell_data_1["image"])
    # Check content of cell 2
    np.testing.assert_array_equal(canvas_img[60:120, 50:110], processed_cell_data_2["image"])
    # Check empty areas are zero
    assert np.all(canvas_img[50:60, :] == 0)
    assert np.all(canvas_img[:, 50:50] == 0)
    assert np.all(canvas_img[0:60, 110:] == 0)
    assert np.all(canvas_img[110:, 0:50] == 0)
    # Ensure total non-zero area matches sum of cell areas
    expected_img_pixels = np.count_nonzero(processed_cell_data_1["image"]) + np.count_nonzero(processed_cell_data_2["image"])
    assert np.count_nonzero(canvas_img) == expected_img_pixels

    # Test for mask
    canvas_mask = assemble_mosaic_from_processed_cells(
        processed_cells, target_shape[:2], dtype, "mask"
    )
    assert canvas_mask.shape == target_shape[:2]
    np.testing.assert_array_equal(canvas_mask[0:50, 0:50], processed_cell_data_1["mask"])
    np.testing.assert_array_equal(canvas_mask[60:120, 50:110], processed_cell_data_2["mask"])
    assert np.all(canvas_mask[50:60, :] == 0)
    assert np.all(canvas_mask[:, 50:50] == 0)
    # Ensure total non-zero area matches sum of cell areas
    expected_mask_pixels = np.count_nonzero(processed_cell_data_1["mask"]) + np.count_nonzero(processed_cell_data_2["mask"])
    assert np.count_nonzero(canvas_mask) == expected_mask_pixels
