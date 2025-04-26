from __future__ import annotations

import random
import numpy as np
from typing import Any, Literal

import pytest
import cv2
from albumentations.core.bbox_utils import BboxParams, BboxProcessor
from albumentations.core.keypoints_utils import KeypointParams, KeypointsProcessor

from albumentations.augmentations.mixing.functional import (
    assign_items_to_grid_cells,
    calculate_cell_placements,
    calculate_mosaic_center_point,
    filter_valid_metadata,
    process_cell_geometry,
    process_all_mosaic_geometries,
    assemble_mosaic_from_processed_cells,
    preprocess_selected_mosaic_items,
    get_cell_relative_position,
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
    center_xy = calculate_mosaic_center_point(grid_yx=grid_yx, cell_shape=target_size,
                                              target_size=target_size,
                                              center_range=center_range, py_random=py_random)
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
    placements = calculate_cell_placements(grid_yx=grid_yx, cell_shape=target_size,
                                           target_size=target_size,
                                            center_xy=center_xy)
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
    # Use empty primary data so no compatibility checks are enforced for this basic test
    primary_data = {}
    result = filter_valid_metadata(metadata_input, "test_key", primary_data)
    assert result == metadata_input


def test_filter_valid_metadata_mixed(valid_item_1, invalid_item_no_image, valid_item_2, invalid_item_not_dict) -> None:
    """Test with a mix of valid and invalid items, checking warnings."""
    metadata_input = [valid_item_1, invalid_item_no_image, valid_item_2, invalid_item_not_dict]
    # Expected output should include both valid items if primary data is empty
    expected_output = [valid_item_1, valid_item_2]
    # Use empty primary data so no compatibility checks are enforced beyond basic structure
    primary_data = {}

    with pytest.warns(UserWarning) as record:
        result = filter_valid_metadata(metadata_input, "test_key", primary_data)

    assert result == expected_output
    # Check the warning messages based on the refactored logic
    assert len(record) == 2 # Expecting two warnings
    # Warning 1: invalid_item_no_image (missing required key 'image')
    assert "skipped due to incompatibility in 'image': Item is missing required key 'image'" in str(record[0].message)
    assert "index 1" in str(record[0].message)
    # Warning 2: invalid_item_not_dict (not a dict)
    assert "Item at index 3 in 'test_key' is not a dict and will be skipped." in str(record[1].message)

def test_filter_valid_metadata_empty_list() -> None:
    """Test with an empty list."""
    result = filter_valid_metadata([], "test_key", data={})
    assert result == []

def test_filter_valid_metadata_none_input() -> None:
    """Test with None as input, checking warning."""
    with pytest.warns(UserWarning, match="Metadata under key 'test_key' is not a Sequence"):
        result = filter_valid_metadata(None, "test_key", data={})
    assert result == []

def test_filter_valid_metadata_dict_input(valid_item_1) -> None:
    """Test with a dictionary as input instead of a sequence, checking warning."""
    with pytest.warns(UserWarning, match="Metadata under key 'test_key' is not a Sequence"):
        result = filter_valid_metadata(valid_item_1, "test_key", data={})
    assert result == []

def test_filter_valid_metadata_tuple_input(valid_item_1, valid_item_2) -> None:
    """Test with a tuple of valid items."""
    metadata_input = (valid_item_1, valid_item_2)
    expected_output = [valid_item_1, valid_item_2]
    # Use empty primary data so no compatibility checks are enforced
    primary_data = {}
    result = filter_valid_metadata(metadata_input, "test_key", primary_data)
    assert result == expected_output


# === New tests for data compatibility checks ===

@pytest.fixture
def primary_data_rgb_mask_grayscale() -> dict[str, Any]:
    return {"image": np.zeros((100, 100, 3), dtype=np.uint8), "mask": np.zeros((100, 100), dtype=np.uint8)}

@pytest.fixture
def primary_data_rgb_mask_single_channel() -> dict[str, Any]:
    return {"image": np.zeros((100, 100, 3), dtype=np.uint8), "mask": np.zeros((100, 100, 1), dtype=np.uint8)}

@pytest.fixture
def item_rgb_mask_grayscale() -> dict[str, Any]:
    return {"image": np.ones((50, 50, 3), dtype=np.uint8), "mask": np.ones((50, 50), dtype=np.uint8)}

@pytest.fixture
def item_gray_mask_grayscale() -> dict[str, Any]:
    return {"image": np.ones((50, 50), dtype=np.uint8), "mask": np.ones((50, 50), dtype=np.uint8)}

@pytest.fixture
def item_rgb_mask_single_channel() -> dict[str, Any]:
    return {"image": np.ones((50, 50, 3), dtype=np.uint8), "mask": np.ones((50, 50, 1), dtype=np.uint8)}

@pytest.fixture
def item_rgb_mask_multi_channel() -> dict[str, Any]:
    return {"image": np.ones((50, 50, 3), dtype=np.uint8), "mask": np.ones((50, 50, 2), dtype=np.uint8)}

@pytest.fixture
def item_rgb_no_mask() -> dict[str, Any]:
    return {"image": np.ones((50, 50, 3), dtype=np.uint8)}


def test_filter_valid_metadata_image_incompatible_channels(primary_data_rgb_mask_grayscale, item_gray_mask_grayscale):
    """Test image incompatibility: primary RGB, item Grayscale."""
    metadata = [item_gray_mask_grayscale]
    with pytest.warns(UserWarning, match="incompatibility in 'image'.*Item 'image' has 2 dimensions, but primary has 3"):
        result = filter_valid_metadata(metadata, "test", primary_data_rgb_mask_grayscale)
    assert result == []

def test_filter_valid_metadata_image_compatible_channels(primary_data_rgb_mask_grayscale, item_rgb_mask_grayscale):
    """Test image compatibility: both RGB."""
    metadata = [item_rgb_mask_grayscale]
    # No warning expected
    result = filter_valid_metadata(metadata, "test", primary_data_rgb_mask_grayscale)
    assert result == [item_rgb_mask_grayscale]


def test_filter_valid_metadata_mask_incompatible_dims(primary_data_rgb_mask_grayscale, item_rgb_mask_single_channel):
    """Test mask incompatibility: primary 2D, item 3D."""
    metadata = [item_rgb_mask_single_channel]
    with pytest.warns(UserWarning, match="incompatibility in 'mask'.*Item 'mask' has 3 dimensions, but primary has 2"):
        result = filter_valid_metadata(metadata, "test", primary_data_rgb_mask_grayscale)
    assert result == []

def test_filter_valid_metadata_mask_incompatible_channels(primary_data_rgb_mask_single_channel, item_rgb_mask_multi_channel):
    """Test mask incompatibility: both 3D, different channels."""
    metadata = [item_rgb_mask_multi_channel]
    with pytest.warns(UserWarning, match="incompatibility in 'mask'.*Item 'mask' has 2 channels, but primary has 1"):
        result = filter_valid_metadata(metadata, "test", primary_data_rgb_mask_single_channel)
    assert result == []

def test_filter_valid_metadata_mask_compatible_dims_channels(primary_data_rgb_mask_single_channel, item_rgb_mask_single_channel):
    """Test mask compatibility: both 3D, same channels."""
    metadata = [item_rgb_mask_single_channel]
    # No warning expected
    result = filter_valid_metadata(metadata, "test", primary_data_rgb_mask_single_channel)
    assert result == [item_rgb_mask_single_channel]

def test_filter_valid_metadata_mask_compatible_primary_has_item_missing(primary_data_rgb_mask_grayscale, item_rgb_no_mask):
     """Test mask compatibility: primary has mask, item does not (valid)."""
     metadata = [item_rgb_no_mask]
     # No warning expected
     result = filter_valid_metadata(metadata, "test", primary_data_rgb_mask_grayscale)
     assert result == [item_rgb_no_mask]

def test_filter_valid_metadata_mask_compatible_primary_missing_item_has(item_rgb_mask_grayscale):
    """Test mask compatibility: primary has no mask, item does (valid)."""
    primary_data = {"image": np.zeros((10,10,3))} # No mask in primary
    metadata = [item_rgb_mask_grayscale]
    # No warning expected
    result = filter_valid_metadata(metadata, "test", primary_data)
    assert result == [item_rgb_mask_grayscale]

# === End new tests ===


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

    processed = process_cell_geometry(item=item, cell_shape=(target_h, target_w),
                                      target_shape=(target_h, target_w),
                                      fill=0, fill_mask=0, fit_mode="contain",
                                      interpolation=cv2.INTER_NEAREST, mask_interpolation=cv2.INTER_NEAREST,
                                      cell_position="center")

    assert processed["image"].shape == (target_h, target_w, 3)
    assert processed["mask"].shape == (target_h, target_w)
    np.testing.assert_array_equal(processed["image"], item["image"])
    np.testing.assert_array_equal(processed["mask"], item["mask"])


def test_process_cell_geometry_crop(base_item_geom) -> None:
    """Test cropping case: target size is smaller than item size."""
    item = base_item_geom
    target_h, target_w = 60, 50

    processed = process_cell_geometry(item=item, cell_shape=(target_h, target_w),
                                      target_shape=(target_h, target_w),
                                      fill=0, fill_mask=0, fit_mode="contain",
                                      interpolation=cv2.INTER_NEAREST, mask_interpolation=cv2.INTER_NEAREST,
                                      cell_position="center")

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
    cell_position = "center"

    processed = process_cell_geometry(item=item, cell_shape=(target_h, target_w),
                                      target_shape=(target_h, target_w),
                                      fill=fill_value, fill_mask=mask_fill_value,
                                      fit_mode="contain", interpolation=cv2.INTER_NEAREST, mask_interpolation=cv2.INTER_NEAREST,
                                      cell_position=cell_position)

    assert processed["image"].shape == (target_h, target_w, 3)
    assert processed["mask"].shape == (target_h, target_w)

    # Calculate expected size after LongestMaxSize (maintaining aspect ratio)
    # Original 50x50 -> Target 70x80. Longest=80. Scale = 80/50 = 1.6? No, longest is 70 for height.
    # Scale = 70/50 = 1.4. New size = (50*1.4, 50*1.4) = (70, 70)
    # If target was 80x70, scale = 80/50 = 1.6 -> (80, 80). Need LongestMaxSize logic.
    # Simpler: find the non-padded region. PadIfNeeded(center) adds symmetric padding.
    original_h, original_w = item["image"].shape[:2]
    scale = min(target_h / original_h, target_w / original_w)
    resized_h, resized_w = int(original_h * scale), int(original_w * scale)

    pad_top = (target_h - resized_h) // 2
    pad_left = (target_w - resized_w) // 2
    y_slice_img = slice(pad_top, pad_top + resized_h)
    x_slice_img = slice(pad_left, pad_left + resized_w)

    # Check that the central (image) area does NOT contain fill values
    assert not np.all(processed["image"][y_slice_img, x_slice_img] == fill_value)
    assert not np.all(processed["mask"][y_slice_img, x_slice_img] == mask_fill_value)

    # Check padding values in corners (or other known padded areas)
    assert np.all(processed["image"][:pad_top, :pad_left] == fill_value)
    assert np.all(processed["image"][pad_top + resized_h:, pad_left + resized_w:] == fill_value)
    assert np.all(processed["mask"][:pad_top, :pad_left] == mask_fill_value)
    assert np.all(processed["mask"][pad_top + resized_h:, pad_left + resized_w:] == mask_fill_value)

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
    canvas_shape = (100, 100) # Added canvas_shape

    processed = process_all_mosaic_geometries(
        canvas_shape=canvas_shape,
        cell_shape=canvas_shape,
        placement_to_item_index=placement_to_item_index,
        final_items_for_grid=final_items,
        # Removed cell_placements argument
        fill=0,
        fill_mask=0,
        fit_mode="contain",
        interpolation=cv2.INTER_NEAREST,
        mask_interpolation=cv2.INTER_NEAREST,
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
    placement_to_item_index = {(10, 20, 60, 80): 0} # Placement is 50x60 within 80x60 canvas
    final_items = [base_item_geom] # Item is 100x100
    canvas_shape = (80, 60) # Added canvas_shape (height, width)

    processed = process_all_mosaic_geometries(
        canvas_shape=canvas_shape,
        cell_shape=canvas_shape,
        placement_to_item_index=placement_to_item_index,
        final_items_for_grid=final_items,
        fill=0,
        fill_mask=0,
        fit_mode="contain",
        interpolation=cv2.INTER_NEAREST,
        mask_interpolation=cv2.INTER_NEAREST,
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
    canvas_shape = (70, 80) # Added canvas_shape (height, width)

    processed = process_all_mosaic_geometries(
        canvas_shape=canvas_shape,
        cell_shape=canvas_shape,
        placement_to_item_index=placement_to_item_index,
        final_items_for_grid=final_items,
        fill=fill_value,
        fill_mask=mask_fill_value,
        fit_mode="contain",
        interpolation=cv2.INTER_NEAREST,
        mask_interpolation=cv2.INTER_NEAREST,
    )

    assert len(processed) == 1
    placement_key = (0, 0, 80, 70)
    target_h, target_w = 70, 80
    assert placement_key in processed
    processed_item = processed[placement_key]
    assert processed_item["image"].shape == (target_h, target_w, 3)
    assert processed_item["mask"].shape == (target_h, target_w)

    # Similar check as in test_process_cell_geometry_pad
    # Calculate expected size after LongestMaxSize
    original_h, original_w = small_item_geom["image"].shape[:2]
    scale = min(target_h / original_h, target_w / original_w)
    resized_h, resized_w = int(original_h * scale), int(original_w * scale)
    pad_top = (target_h - resized_h) // 2
    pad_left = (target_w - resized_w) // 2
    y_slice_img = slice(pad_top, pad_top + resized_h)
    x_slice_img = slice(pad_left, pad_left + resized_w)

    # Check the central (image) area does NOT contain fill values
    assert not np.all(processed_item["image"][y_slice_img, x_slice_img] == fill_value)
    assert not np.all(processed_item["mask"][y_slice_img, x_slice_img] == mask_fill_value)

    # Check padding values in corners
    assert np.all(processed_item["image"][:pad_top, :pad_left] == fill_value)
    assert np.all(processed_item["image"][pad_top + resized_h:, pad_left + resized_w:] == fill_value)
    assert np.all(processed_item["mask"][:pad_top, :pad_left] == mask_fill_value)
    assert np.all(processed_item["mask"][pad_top + resized_h:, pad_left + resized_w:] == mask_fill_value)

def test_process_all_geometry_multiple_cells(items_list_geom) -> None:
    """Test process_all processing two different items for two cells."""
    # Setup: Map placements directly to item indices
    placement_to_item_index = {
        (0, 0, 50, 50): 0,    # Crop base_item_geom (idx 0) to 50x50
        (50, 0, 110, 60): 1,  # Pad small_item_geom (idx 1) to 60x60
    }
    final_items = items_list_geom
    canvas_shape = (60, 110) # Added canvas_shape (height=60, width=110)
    fill_value = 0 # Using default fill=0 for this test
    mask_fill_value = 0

    processed = process_all_mosaic_geometries(
        canvas_shape=canvas_shape,
        cell_shape=canvas_shape,
        placement_to_item_index=placement_to_item_index,
        final_items_for_grid=final_items,
        fill=fill_value,
        fill_mask=mask_fill_value,
        fit_mode="contain",
        interpolation=cv2.INTER_NEAREST,
        mask_interpolation=cv2.INTER_NEAREST,
    )

    assert len(processed) == 2
    placement1 = (0, 0, 50, 50)
    placement2 = (50, 0, 110, 60)
    target_h1, target_w1 = 50, 50
    target_h2, target_w2 = 60, 60 # Cell 2 shape is 60x60 (110-50, 60-0)
    assert placement1 in processed
    assert placement2 in processed

    # Check cell 1 (cropped from base_item_geom)
    processed1 = processed[placement1]
    assert processed1["image"].shape == (target_h1, target_w1, 3)
    assert processed1["mask"].shape == (target_h1, target_w1)
    # Cropping might change content, so just check shape is correct

    # Check cell 2 (padded small_item_geom)
    processed2 = processed[placement2]
    assert processed2["image"].shape == (target_h2, target_w2, 3)
    assert processed2["mask"].shape == (target_h2, target_w2)

    # Calculate expected size after LongestMaxSize for item 1 (small_item_geom)
    original_h, original_w = items_list_geom[1]["image"].shape[:2] # 50x50
    scale = min(target_h2 / original_h, target_w2 / original_w) # min(60/50, 60/50) = 1.2
    resized_h, resized_w = int(original_h * scale), int(original_w * scale) # 60x60

    pad_top = (target_h2 - resized_h) // 2 # (60-60)//2 = 0
    pad_left = (target_w2 - resized_w) // 2 # (60-60)//2 = 0
    y_slice_img = slice(pad_top, pad_top + resized_h) # slice(0, 60)
    x_slice_img = slice(pad_left, pad_left + resized_w) # slice(0, 60)

    # Check the central (image) area does NOT contain fill values
    assert not np.all(processed2["image"][y_slice_img, x_slice_img] == fill_value)
    assert not np.all(processed2["mask"][y_slice_img, x_slice_img] == mask_fill_value)

    # Since resized_h/w match target_h2/w2, there should be no padding.
    # Verify this implicitly by checking the whole image doesn't equal fill value.
    assert not np.all(processed2["image"] == fill_value)
    assert not np.all(processed2["mask"] == mask_fill_value)


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
    target_shape_img = (50, 50, 3) # RGB image
    target_shape_mask = (50, 50) # 2D Mask
    dtype = np.uint8
    fill_value = 0 # Define fill value

    # Test for image
    canvas_img = assemble_mosaic_from_processed_cells(processed_cells, target_shape_img, dtype, "image", fill=fill_value)
    assert canvas_img.shape == target_shape_img
    assert canvas_img.dtype == dtype
    np.testing.assert_array_equal(canvas_img, processed_cell_data_1["image"])

    # Test for mask
    canvas_mask = assemble_mosaic_from_processed_cells(
        processed_cells, target_shape_mask, dtype, "mask", fill=fill_value
    )
    assert canvas_mask.shape == target_shape_mask
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
    target_shape_img = (120, 120, 3) # Target canvas size
    target_shape_mask = (120, 120)
    dtype = np.uint8
    fill_value = 0 # Define fill value

    # Test for image
    canvas_img = assemble_mosaic_from_processed_cells(processed_cells, target_shape_img, dtype, "image", fill=fill_value)
    assert canvas_img.shape == target_shape_img
    # Check content of cell 1
    np.testing.assert_array_equal(canvas_img[0:50, 0:50], processed_cell_data_1["image"])
    # Check content of cell 2
    np.testing.assert_array_equal(canvas_img[60:120, 50:110], processed_cell_data_2["image"])
    # Check empty areas are zero (or fill_value if non-zero)
    assert np.all(canvas_img[50:60, :] == fill_value)
    assert np.all(canvas_img[0:60, 110:] == fill_value)
    assert np.all(canvas_img[110:, 0:50] == fill_value)
    # Ensure total non-zero area matches sum of cell areas (only if fill_value is 0)
    if fill_value == 0:
        expected_img_pixels = np.count_nonzero(processed_cell_data_1["image"]) + np.count_nonzero(processed_cell_data_2["image"])
        assert np.count_nonzero(canvas_img) == expected_img_pixels

    # Test for mask
    canvas_mask = assemble_mosaic_from_processed_cells(
        processed_cells, target_shape_mask, dtype, "mask", fill=fill_value
    )
    assert canvas_mask.shape == target_shape_mask
    np.testing.assert_array_equal(canvas_mask[0:50, 0:50], processed_cell_data_1["mask"])
    np.testing.assert_array_equal(canvas_mask[60:120, 50:110], processed_cell_data_2["mask"])
    assert np.all(canvas_mask[50:60, :] == fill_value)
    assert np.all(canvas_mask[0:60, 110:] == fill_value)
    assert np.all(canvas_mask[110:, 0:50] == fill_value)
    # Ensure total non-zero area matches sum of cell areas (only if fill_value is 0)
    if fill_value == 0:
        expected_mask_pixels = np.count_nonzero(processed_cell_data_1["mask"]) + np.count_nonzero(processed_cell_data_2["mask"])
        assert np.count_nonzero(canvas_mask) == expected_mask_pixels



def test_preprocess_selected_mosaic_items_basic():
    # Setup processors
    bbox_processor = BboxProcessor(BboxParams(format='pascal_voc', label_fields=['class_labels']))
    keypoint_processor = KeypointsProcessor(KeypointParams(format='xy', label_fields=['kp_labels']))

    # Setup input data items with different shapes and unique labels
    item1 = {
        "image": np.zeros((50, 60, 3), dtype=np.uint8),
        "mask": np.zeros((50, 60), dtype=np.uint8),
        "bboxes": np.array([[10, 10, 20, 20]], dtype=np.float32),
        "class_labels": ["cat"],
        "keypoints": np.array([[15, 15]], dtype=np.float32),
        "kp_labels": ["eye"],
    }
    item2 = {
        "image": np.zeros((80, 70, 3), dtype=np.uint8),
        # no mask
        "bboxes": np.array([[30, 30, 40, 40]], dtype=np.float32),
        "class_labels": ["dog"],
        "keypoints": np.array([[35, 35]], dtype=np.float32),
        "kp_labels": ["nose"],
    }
    selected_raw_items = [item1, item2]

    # --- Call the function ---
    result = preprocess_selected_mosaic_items(selected_raw_items, bbox_processor, keypoint_processor)

    # --- Assertions ---
    assert isinstance(result, list)
    assert len(result) == 2

    # Check item 1 structure and data
    res1 = result[0]
    assert isinstance(res1, dict)
    assert sorted(list(res1.keys())) == sorted(["image", "mask", "bboxes", "keypoints"]) # Check keys regardless of order
    np.testing.assert_array_equal(res1["image"], item1["image"])
    np.testing.assert_array_equal(res1["mask"], item1["mask"])
    assert res1["bboxes"] is not None
    assert isinstance(res1["bboxes"], np.ndarray)
    # Expect format change (pascal_voc -> albumentations) + label encoding
    assert res1["bboxes"].shape == (1, 5) # [x_min, y_min, x_max, y_max, label_id]
    assert res1["keypoints"] is not None
    assert isinstance(res1["keypoints"], np.ndarray)
    # Expect format change (xy -> xy + angle + scale) + label encoding + kp_labels
    assert res1["keypoints"].shape == (1, 6) # [x, y, angle, scale, label_id, kp_label_id]

    # Check item 2 structure and data
    res2 = result[1]
    assert isinstance(res2, dict)
    assert sorted(list(res2.keys())) == sorted(["image", "mask", "bboxes", "keypoints"])
    np.testing.assert_array_equal(res2["image"], item2["image"])
    assert res2["mask"] is None # Original item had no mask
    assert res2["bboxes"] is not None
    assert isinstance(res2["bboxes"], np.ndarray)
    assert res2["bboxes"].shape == (1, 5)
    assert res2["keypoints"] is not None
    assert isinstance(res2["keypoints"], np.ndarray)
    # Expect format change (xy -> xy + angle + scale) + label encoding + kp_labels
    assert res2["keypoints"].shape == (1, 6) # [x, y, angle, scale, label_id, kp_label_id]

    # Check LabelEncoder state
    # BBox labels: cat (0), dog (1)
    bbox_encoder = bbox_processor.label_manager.get_encoder("bboxes", "class_labels")
    assert bbox_encoder is not None
    assert len(bbox_encoder.classes_) == 2
    assert bbox_encoder.transform(["cat", "dog"]).tolist() == [0, 1]
    assert res1["bboxes"][0, 4] == 0 # cat
    assert res2["bboxes"][0, 4] == 1 # dog

    # Keypoint labels: eye (0), nose (1)
    kp_encoder = keypoint_processor.label_manager.get_encoder("keypoints", "kp_labels")
    assert kp_encoder is not None
    assert len(kp_encoder.classes_) == 2
    assert kp_encoder.transform(["eye", "nose"]).tolist() == [0, 1]
    # Check base label ID (column 4) - Seems to default to 0
    assert res1["keypoints"][0, 4] == 0
    assert res2["keypoints"][0, 4] == 0
    # Check kp_labels encoded value (column 5)
    assert res1["keypoints"][0, 5] == kp_encoder.classes_["eye"] # Should be 0
    assert res2["keypoints"][0, 5] == kp_encoder.classes_["nose"] # Should be 1

def test_preprocess_selected_mosaic_items_missing_data():
    # Setup processors
    bbox_processor = BboxProcessor(BboxParams(format='pascal_voc', label_fields=['class_labels']))
    keypoint_processor = KeypointsProcessor(KeypointParams(format='xy', label_fields=['kp_labels']))

    # Setup input data items
    item_bbox_only = {
        "image": np.zeros((50, 50, 3), dtype=np.uint8),
        "bboxes": np.array([[10, 10, 20, 20]], dtype=np.float32),
        "class_labels": ["apple"], # Label MUST be present if declared in BboxParams
    }
    item_kp_only = {
        "image": np.zeros((60, 60, 3), dtype=np.uint8),
        "keypoints": np.array([[15, 15]], dtype=np.float32),
        "kp_labels": ["fruit_stem"], # Label MUST be present if declared in KeypointParams
    }
    item_img_only = {
        "image": np.zeros((90, 90, 3), dtype=np.uint8),
    }
    # Removed item_bbox_no_label and item_kp_no_label as they violate the contract

    selected_raw_items = [item_bbox_only, item_kp_only, item_img_only]

    # --- Call the function ---
    result = preprocess_selected_mosaic_items(selected_raw_items, bbox_processor, keypoint_processor)

    # --- Assertions ---
    assert len(result) == 3

    # Check item_bbox_only (index 0)
    res0 = result[0]
    assert res0["bboxes"] is not None and res0["bboxes"].shape == (1, 5) # Has bbox + class_label
    assert res0["keypoints"] is None
    assert res0["bboxes"][0, 4] == 0 # 'apple' encoded as 0

    # Check item_kp_only (index 1)
    res1 = result[1]
    assert res1["bboxes"] is None
    assert res1["keypoints"] is not None and res1["keypoints"].shape == (1, 6) # Has kp + label_id + kp_label
    assert res1["keypoints"][0, 4] == 0 # 'fruit_stem' encoded as 0 for main label_id

    # Check item_img_only (index 2)
    res2 = result[2]
    assert res2["bboxes"] is None
    assert res2["keypoints"] is None

    # Check label encoders
    bbox_encoder = bbox_processor.label_manager.get_encoder("bboxes", "class_labels")
    assert bbox_encoder is not None
    assert len(bbox_encoder.classes_) == 1 # Only saw 'apple'
    assert "apple" in bbox_encoder.classes_

    kp_encoder = keypoint_processor.label_manager.get_encoder("keypoints", "kp_labels")
    assert kp_encoder is not None
    assert len(kp_encoder.classes_) == 1 # Only saw 'fruit_stem'
    assert "fruit_stem" in kp_encoder.classes_

def test_preprocess_selected_mosaic_items_shared_labels():
    bbox_processor = BboxProcessor(BboxParams(format='pascal_voc', label_fields=['class_labels']))
    keypoint_processor = KeypointsProcessor(KeypointParams(format='xy')) # No kp labels needed

    item1 = {"image": np.zeros((10, 10, 3)), "bboxes": np.array([[1,1,2,2]]), "class_labels": ["cat"]}
    item2 = {"image": np.zeros((10, 10, 3)), "bboxes": np.array([[3,3,4,4]]), "class_labels": ["dog"]}
    item3 = {"image": np.zeros((10, 10, 3)), "bboxes": np.array([[5,5,6,6]]), "class_labels": ["cat"]}

    selected_raw_items = [item1, item2, item3]
    result = preprocess_selected_mosaic_items(selected_raw_items, bbox_processor, keypoint_processor)

    assert len(result) == 3
    # Check labels learned: cat (0), dog (1)
    label_encoder = bbox_processor.label_manager.get_encoder("bboxes", "class_labels")
    assert label_encoder is not None
    assert len(label_encoder.classes_) == 2
    np.testing.assert_array_equal(sorted(label_encoder.classes_.keys()), ["cat", "dog"])
    # Check assigned labels in output
    assert result[0]["bboxes"][0, 4] == label_encoder.classes_["cat"]
    assert result[1]["bboxes"][0, 4] == label_encoder.classes_["dog"]
    assert result[2]["bboxes"][0, 4] == label_encoder.classes_["cat"]

def test_preprocess_selected_mosaic_items_no_processors():
    item = {
        "image": np.zeros((50, 60, 3), dtype=np.uint8),
        "bboxes": np.array([[10, 10, 20, 20]], dtype=np.float32), # pascal_voc
        "class_labels": ["cat"],
        "keypoints": np.array([[15, 15]], dtype=np.float32), # xy
        "kp_labels": ["eye"],
    }
    selected_raw_items = [item]

    # --- Test with None processors ---
    result = preprocess_selected_mosaic_items(selected_raw_items, None, None)

    assert len(result) == 1
    res = result[0]
    # Bboxes/Keypoints should be unchanged (original format, no labels added)
    assert res["bboxes"] is not None
    np.testing.assert_array_equal(res["bboxes"], item["bboxes"])
    assert res["bboxes"].shape == (1, 4)

    assert res["keypoints"] is not None
    np.testing.assert_array_equal(res["keypoints"], item["keypoints"])
    assert res["keypoints"].shape == (1, 2)

    # --- Test with only bbox processor ---
    bbox_processor = BboxProcessor(BboxParams(format='pascal_voc', label_fields=['class_labels']))
    # Need copies because processors modify the temp dicts which might affect subsequent calls
    # if we reused the original selected_raw_items list directly.
    # However, preprocess_selected_mosaic_items makes internal copies, so this is safe.
    result_bbox_only = preprocess_selected_mosaic_items(selected_raw_items, bbox_processor, None)
    assert len(result_bbox_only) == 1
    res_bbox = result_bbox_only[0]
    assert res_bbox["bboxes"] is not None
    assert res_bbox["bboxes"].shape == (1, 5) # Processed
    assert res_bbox["keypoints"] is not None
    np.testing.assert_array_equal(res_bbox["keypoints"], item["keypoints"]) # Unchanged

    # --- Test with only keypoint processor ---
    keypoint_processor = KeypointsProcessor(KeypointParams(format='xy', label_fields=['kp_labels']))
    # Need a fresh item dict because the previous call might have altered labels if passed directly
    item_copy = {
        "image": np.zeros((50, 60, 3), dtype=np.uint8),
        "bboxes": np.array([[10, 10, 20, 20]], dtype=np.float32),
        "class_labels": ["cat"],
        "keypoints": np.array([[15, 15]], dtype=np.float32),
        "kp_labels": ["eye"],
    }
    selected_raw_items_copy = [item_copy]

    result_kp_only = preprocess_selected_mosaic_items(selected_raw_items_copy, None, keypoint_processor)
    assert len(result_kp_only) == 1
    res_kp = result_kp_only[0]
    assert res_kp["bboxes"] is not None
    np.testing.assert_array_equal(res_kp["bboxes"], item["bboxes"]) # Unchanged
    assert res_kp["keypoints"] is not None
    assert res_kp["keypoints"].shape == (1, 6) # Processed: x,y,a,s,id,kp_id


@pytest.mark.parametrize(
    ("placement_coords", "target_shape", "expected_position"),
    [
        # Target canvas 100x100, center at (50, 50)
        ((0, 0, 40, 40), (100, 100), "top_left"),  # Cell center (20, 20)
        ((60, 0, 100, 40), (100, 100), "top_right"),  # Cell center (80, 20)
        ((0, 60, 40, 100), (100, 100), "bottom_left"),  # Cell center (20, 80)
        ((60, 60, 100, 100), (100, 100), "bottom_right"),  # Cell center (80, 80)
        # Centered cases
        ((25, 25, 75, 75), (100, 100), "center"),  # Cell center (50, 50)
        # Edge cases - touching center lines
        ((0, 0, 50, 50), (100, 100), "top_left"),  # Cell center (25, 25) -> top_left
        ((50, 0, 100, 50), (100, 100), "top_right"),  # Cell center (75, 25) -> top_right
        ((0, 50, 50, 100), (100, 100), "bottom_left"),  # Cell center (25, 75) -> bottom_left
        ((50, 50, 100, 100), (100, 100), "bottom_right"),  # Cell center (75, 75) -> bottom_right
        # Cells exactly on center lines -> Expected: "center"
        ((25, 0, 75, 100), (100, 100), "center"),  # Cell center (50, 50)
        ((0, 25, 100, 75), (100, 100), "center"),  # Cell center (50, 50)
        # Cells crossing center lines but centered -> Expected: "center"
        ((40, 40, 60, 60), (100, 100), "center"), # Cell center (50, 50)
        # Cells whose center is exactly on a center line -> Expected: "center"
        ((40, 10, 60, 40), (100, 100), "center"),     # Cell center (50, 25)
        ((10, 40, 40, 60), (100, 100), "center"),    # Cell center (25, 50)
        ((60, 40, 90, 60), (100, 100), "center"),   # Cell center (75, 50)
        ((40, 60, 60, 90), (100, 100), "center"),  # Cell center (50, 75)

        # Odd dimensions - Target canvas 101x101, center at (50.5, 50.5)
        ((0, 0, 40, 40), (101, 101), "top_left"),      # Cell center (20, 20) < (50.5, 50.5)
        ((60, 0, 101, 40), (101, 101), "top_right"),    # Cell center (80.5, 20)
        ((0, 60, 40, 101), (101, 101), "bottom_left"),  # Cell center (20, 80.5)
        ((60, 60, 101, 101), (101, 101), "bottom_right"),# Cell center (80.5, 80.5)
        ((25, 25, 76, 76), (101, 101), "center"), # Cell center (50.5, 50.5) - exactly centered

        # Cases exactly on the center line with odd dimensions
        ((0, 0, 50, 50), (101, 101), "top_left"), # Cell center (25, 25) < (50.5, 50.5)
        ((51, 0, 101, 50), (101, 101), "top_right"), # Cell center (76, 25)
        ((0, 51, 50, 101), (101, 101), "bottom_left"), # Cell center (25, 76)
        ((51, 51, 101, 101), (101, 101), "bottom_right"), # Cell center (76, 76)
    ],
    ids=[
        "top_left_100",
        "top_right_100",
        "bottom_left_100",
        "bottom_right_100",
        "center_100",
        "edge_tl_100",
        "edge_tr_100",
        "edge_bl_100",
        "edge_br_100",
        "on_hline_100",
        "on_vline_100",
        "cross_center_exact_100",
        "on_vline_top_100", # Renamed ID
        "on_hline_left_100", # Renamed ID
        "on_hline_right_100", # Renamed ID
        "on_vline_bottom_100", # Renamed ID
        "top_left_101",
        "top_right_101",
        "bottom_left_101",
        "bottom_right_101",
        "center_101",
        "edge_tl_101",
        "edge_tr_101",
        "edge_bl_101",
        "edge_br_101",
    ]
)
def test_get_cell_relative_position(
    placement_coords: tuple[int, int, int, int],
    target_shape: tuple[int, int],
    expected_position: Literal["top_left", "top_right", "center", "bottom_left", "bottom_right"],
):
    # The current implementation classifies cells whose center falls *exactly* on a dividing line
    # as "center". This test verifies that behavior directly against the expected value.
    actual_position = get_cell_relative_position(placement_coords, target_shape)

    assert actual_position == expected_position, \
           f"Placement: {placement_coords}, Target: {target_shape}, Expected: {expected_position}, Got: {actual_position}"
