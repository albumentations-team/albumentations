"""Functional implementations for image mixing operations.

This module provides utility functions for blending and combining images,
such as copy-and-paste operations with masking.
"""

from __future__ import annotations

import random
import warnings
from typing import Any, Literal, cast

import cv2
import numpy as np

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.crops.transforms import Crop
from albumentations.augmentations.geometric.resize import LongestMaxSize
from albumentations.augmentations.geometric.transforms import PadIfNeeded
from albumentations.core.composition import Compose

__all__ = [
    "_determine_padding_position",
    "assemble_mosaic_image_mask",
    "calculate_mosaic_center_point",
    "copy_and_paste_blend",
    "get_mosaic_transforms_coords",
    "prepare_mosaic_inputs",
    "process_mosaic_cell",
    "process_mosaic_grid",
]


def copy_and_paste_blend(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    overlay_mask: np.ndarray,
    offset: tuple[int, int],
) -> np.ndarray:
    """Blend images by copying pixels from an overlay image to a base image using a mask.

    This function copies pixels from the overlay image to the base image only where
    the mask has non-zero values. The overlay is placed at the specified offset
    from the top-left corner of the base image.

    Args:
        base_image (np.ndarray): The destination image that will be modified.
        overlay_image (np.ndarray): The source image containing pixels to copy.
        overlay_mask (np.ndarray): Binary mask indicating which pixels to copy from the overlay.
            Pixels are copied where mask > 0.
        offset (tuple[int, int]): The (y, x) offset specifying where to place the
            top-left corner of the overlay relative to the base image.

    Returns:
        np.ndarray: The blended image with the overlay applied to the base image.

    """
    y_offset, x_offset = offset

    blended_image = base_image.copy()
    mask_indices = np.where(overlay_mask > 0)
    blended_image[mask_indices[0] + y_offset, mask_indices[1] + x_offset] = overlay_image[
        mask_indices[0],
        mask_indices[1],
    ]
    return blended_image


def _determine_final_additional_data(
    primary_data: dict[str, Any],
    additional_data_list: list[dict[str, Any]],
    num_needed_additional: int,
    py_random: random.Random,
) -> list[dict[str, Any]]:
    """Determines the final list of additional data items by sampling or replication."""
    num_provided_additional = len(additional_data_list)
    final_additional_data = []

    if num_provided_additional == num_needed_additional:
        final_additional_data = additional_data_list
    elif num_provided_additional > num_needed_additional:
        final_additional_data = py_random.sample(additional_data_list, num_needed_additional)
    else:  # num_provided_additional < num_needed_additional
        final_additional_data = additional_data_list  # Start with the valid provided
        num_replications = num_needed_additional - num_provided_additional

        # Prepare replication source from primary data (only copy necessary keys)
        primary_replication_source = {"image": primary_data["image"].copy()}
        optional_keys_to_copy = {"mask", "bboxes", "keypoints"}
        for key in optional_keys_to_copy:
            if key in primary_data and primary_data[key] is not None:
                arr = primary_data[key]
                if not isinstance(arr, np.ndarray):
                    arr = np.array(arr)
                primary_replication_source[key] = arr.copy()

        for _ in range(num_replications):
            replication_dict = {
                k: v.copy() if isinstance(v, np.ndarray) else v for k, v in primary_replication_source.items()
            }
            final_additional_data.append(replication_dict)
    return final_additional_data


def _prepare_data_item(
    data_item: dict[str, Any],
    present_optional_keys: set[str],
) -> dict[str, Any]:
    """Ensures consistent keys and deep copies arrays for a single data item."""
    prepared_item = data_item.copy()  # Shallow copy first

    # Deep copy image array
    if "image" in prepared_item:
        prepared_item["image"] = prepared_item["image"].copy()

    # Ensure optional keys exist and deep copy their arrays if present
    for key in present_optional_keys:
        if key not in prepared_item or prepared_item[key] is None:
            prepared_item[key] = None
        elif isinstance(prepared_item[key], np.ndarray):
            # Make sure it's copied, even if shallow copy was enough initially
            prepared_item[key] = prepared_item[key].copy()
        # If key is present but not None and not ndarray, try converting (e.g., list of lists for bbox)
        # This part might need refinement based on expected input types for bboxes/keypoints
        elif key in ["bboxes", "keypoints"]:
            try:
                # Ensure numpy conversion also happens if needed
                if not isinstance(prepared_item[key], np.ndarray):
                    prepared_item[key] = np.array(prepared_item[key])
                # np.array creates a copy, so no extra copy needed here
            except (ValueError, TypeError) as e:  # Catch specific conversion errors
                warnings.warn(
                    f"Could not convert {key} to numpy array ({e}). Keeping original type.",
                    UserWarning,
                    stacklevel=3,
                )

    # Ensure all optional keys are present, even if None
    all_optional_keys = {"mask", "bboxes", "keypoints"}
    for key in all_optional_keys:
        if key not in prepared_item:
            prepared_item[key] = None

    return prepared_item


def _determine_primary_placement(
    grid_yx: tuple[int, int],
    py_random: random.Random,
) -> tuple[int, int]:
    """Determines the grid cell coordinates for placing the primary image."""
    rows, cols = grid_yx
    center_row_start = (rows - 1) // 2
    center_row_end = rows // 2
    center_col_start = (cols - 1) // 2
    center_col_end = cols // 2

    possible_primary_placements = [
        (r, c) for r in range(center_row_start, center_row_end + 1) for c in range(center_col_start, center_col_end + 1)
    ]
    return py_random.choice(possible_primary_placements)


def prepare_mosaic_inputs(
    primary_data: dict[str, Any],
    additional_data_list: list[dict[str, Any]],
    grid_yx: tuple[int, int],
    py_random: random.Random,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Prepares and assigns input data dictionaries to grid positions for mosaic."""
    rows, cols = grid_yx
    num_total_images = rows * cols
    num_needed_additional = num_total_images - 1

    # --- 1. Determine final list of additional items ---
    final_additional_data = _determine_final_additional_data(
        primary_data,
        additional_data_list,
        num_needed_additional,
        py_random,
    )

    # --- 2. Identify optional keys present in primary data ---
    optional_keys = {"mask", "bboxes", "keypoints"}
    present_optional_keys = {k for k in optional_keys if k in primary_data and primary_data[k] is not None}

    # --- 3. Prepare all items (ensure consistent keys, copy arrays) ---
    prepared_primary = _prepare_data_item(primary_data, present_optional_keys)
    prepared_additional = [_prepare_data_item(item, present_optional_keys) for item in final_additional_data]
    all_data_items_prepared = [prepared_primary, *prepared_additional]

    # --- 4. Check final count ---
    if len(all_data_items_prepared) != num_total_images:
        warnings.warn(
            f"Internal error after preparation: Expected {num_total_images} items, got {len(all_data_items_prepared)}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}  # Indicate failure

    # --- 5. Determine primary image placement ---
    primary_placement = _determine_primary_placement(grid_yx, py_random)

    # --- 6. Assign items to grid positions ---
    output_grid = {}
    items_to_place = prepared_additional  # Use prepared additional list
    py_random.shuffle(items_to_place)
    item_iterator = iter(items_to_place)

    for r in range(rows):
        for c in range(cols):
            grid_pos = (r, c)
            if grid_pos == primary_placement:
                output_grid[grid_pos] = prepared_primary
            else:
                try:
                    output_grid[grid_pos] = next(item_iterator)
                except StopIteration as e:
                    raise RuntimeError(
                        "Mismatch between grid size and number of prepared data items during assignment.",
                    ) from e

    return output_grid


def get_mosaic_transforms_coords(
    grid_yx: tuple[int, int],
    target_size: tuple[int, int],
    center_point: tuple[int, int],
) -> dict[tuple[int, int], dict[str, tuple[int, int, int, int]]]:
    """Calculates source crop and target placement coordinates for each mosaic grid cell.

    Assumes a conceptual large grid formed by placing target_size images,
    and calculates the coordinates based on a central crop.

    Args:
        grid_yx (tuple[int, int]): The (rows, cols) of the mosaic grid.
        target_size (tuple[int, int]): The final output (height, width). This is also
                                       the assumed size of each cell's source image
                                       for coordinate calculations.
        center_point (tuple[int, int]): The (x, y) center of the final crop window,
                                        relative to the top-left of the conceptual
                                        large grid (size: rows*target_h, cols*target_w).

    Returns:
        dict[tuple[int, int], dict[str, tuple[int, int, int, int]]]:
            A dictionary where keys are grid cell coordinates (row, col) and
            values are dictionaries containing:
            - 'source_crop': (x_min, y_min, x_max, y_max) relative to cell's image.
            - 'target_placement': (x_min, y_min, x_max, y_max) relative to final output canvas.
            Returns empty dict if calculation is not possible (e.g., invalid center).

    """
    rows, cols = grid_yx
    target_h, target_w = target_size
    center_x, center_y = center_point

    # Calculate the boundaries of the final crop window on the large conceptual grid
    # Ensure center point is valid (simplistic check, might need refinement based on center_range)
    large_grid_h = rows * target_h
    large_grid_w = cols * target_w
    if not (0 <= center_x < large_grid_w and 0 <= center_y < large_grid_h):
        warnings.warn(
            f"Center point {center_point} is outside the conceptual large grid dimensions "
            f"({large_grid_h}x{large_grid_w}). Cannot calculate transforms.",
            UserWarning,
            stacklevel=2,
        )
        return {}

    crop_x_min = center_x - (target_w - 1) // 2
    crop_y_min = center_y - (target_h - 1) // 2
    crop_x_max = crop_x_min + target_w
    crop_y_max = crop_y_min + target_h

    # Clip crop window to the large grid boundaries (important if center is near edge)
    crop_x_min_clipped = max(0, crop_x_min)
    crop_y_min_clipped = max(0, crop_y_min)
    crop_x_max_clipped = min(large_grid_w, crop_x_max)
    crop_y_max_clipped = min(large_grid_h, crop_y_max)

    transforms_coords = {}

    for r in range(rows):
        for c in range(cols):
            # Boundaries of the current cell on the large grid
            cell_x_min = c * target_w
            cell_y_min = r * target_h
            cell_x_max = (c + 1) * target_w
            cell_y_max = (r + 1) * target_h

            # Find the intersection of the final crop window and the current cell
            inter_x_min = max(crop_x_min_clipped, cell_x_min)
            inter_y_min = max(crop_y_min_clipped, cell_y_min)
            inter_x_max = min(crop_x_max_clipped, cell_x_max)
            inter_y_max = min(crop_y_max_clipped, cell_y_max)

            # If there is a valid intersection (positive area)
            if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                # Source crop coordinates (relative to the cell's image top-left)
                # These coordinates are within the [0, target_w] and [0, target_h] range
                src_x1 = inter_x_min - cell_x_min
                src_y1 = inter_y_min - cell_y_min
                src_x2 = inter_x_max - cell_x_min
                src_y2 = inter_y_max - cell_y_min

                # Target placement coordinates (relative to the final output image top-left)
                # These coordinates are within the [0, target_w] and [0, target_h] range
                # Need to account for cases where the overall crop starts negative
                tgt_x1 = inter_x_min - crop_x_min
                tgt_y1 = inter_y_min - crop_y_min
                tgt_x2 = inter_x_max - crop_x_min
                tgt_y2 = inter_y_max - crop_y_min

                transforms_coords[(r, c)] = {
                    "source_crop": (src_x1, src_y1, src_x2, src_y2),
                    "target_placement": (tgt_x1, tgt_y1, tgt_x2, tgt_y2),
                }

    return transforms_coords


def _determine_padding_position(grid_pos: tuple[int, int], grid_yx: tuple[int, int]) -> str:
    """Determines the padding position to push content towards the mosaic center."""
    r, c = grid_pos
    rows, cols = grid_yx

    # Determine vertical position relative to center midpoint ((rows - 1) / 2)
    is_top_half = r < (rows - 1) / 2
    is_bottom_half = r > (rows - 1) / 2
    # If neither top nor bottom half, it's on the center row (for odd rows)
    # or one of the two center rows (for even rows) - treat as vertically centered.

    # Determine horizontal position relative to center midpoint ((cols - 1) / 2)
    is_left_half = c < (cols - 1) / 2
    is_right_half = c > (cols - 1) / 2
    # If neither left nor right half, it's on the center col (for odd cols)
    # or one of the two center cols (for even cols) - treat as horizontally centered.

    # Combine positions to determine padding strategy
    if is_top_half and is_left_half:
        # Top-left quadrant -> pad bottom-right
        return "bottom_right"
    if is_top_half and is_right_half:
        # Top-right quadrant -> pad bottom-left
        return "bottom_left"
    if is_bottom_half and is_left_half:
        # Bottom-left quadrant -> pad top-right
        return "top_right"
    if is_bottom_half and is_right_half:
        # Bottom-right quadrant -> pad top-left
        return "top_left"

    # Handle edges (cells that are in a center row or center column but not true center)
    if is_top_half:  # And thus center col(s)
        # Top edge, middle column(s) -> pad bottom
        return "bottom"
    if is_bottom_half:  # And thus center col(s)
        # Bottom edge, middle column(s) -> pad top
        return "top"
    if is_left_half:  # And thus center row(s)
        # Left edge, middle row(s) -> pad right
        return "right"
    if is_right_half:  # And thus center row(s)
        # Right edge, middle row(s) -> pad left
        return "left"

    # If none of the above, it must be the true center cell(s)
    return "center"


def process_mosaic_cell(
    cell_data: dict[str, Any],
    transform_coords: dict[str, tuple[int, int, int, int]],
    grid_pos: tuple[int, int],
    grid_yx: tuple[int, int],
    interpolation: int,
    mask_interpolation: int,
    fill: float | tuple[float, ...],
    fill_mask: float | tuple[float, ...],
) -> dict[str, Any]:
    """Processes a single mosaic cell's data using Crop, LongestMaxSize, and PadIfNeeded.

    Args:
        cell_data (dict): Data dictionary for the cell (must contain 'image').
        transform_coords (dict): Dictionary containing 'source_crop' and
                                 'target_placement' coordinates.
        grid_pos (tuple[int, int]): The (row, col) position of this cell in the grid.
        grid_yx (tuple[int, int]): The total (rows, cols) of the mosaic grid.
        interpolation (int): OpenCV interpolation flag for resizing images.
        mask_interpolation (int): OpenCV interpolation flag for resizing masks.
        fill (...): Fill value for image padding.
        fill_mask (...): Fill value for mask padding.

    Returns:
        dict[str, Any]: The processed data dictionary for the cell.

    """
    source_crop = transform_coords["source_crop"]
    target_placement = transform_coords["target_placement"]

    src_x1, src_y1, src_x2, src_y2 = source_crop
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target_placement

    # Target dimensions for the final padded output for this cell
    target_h = tgt_y2 - tgt_y1
    target_w = tgt_x2 - tgt_x1

    # Max size for LongestMaxSize corresponds to the dimensions of the target slot
    # Ensure max_size is at least 1, handle cases where target_h/w might be 0
    max_size = max(1, target_h, target_w)

    # Determine padding position
    pad_position = _determine_padding_position(grid_pos, grid_yx)

    # Create the processing pipeline
    # Note: Using p=1 as we always want these steps applied here
    pipeline = Compose(
        [
            Crop(x_min=src_x1, y_min=src_y1, x_max=src_x2, y_max=src_y2, p=1.0),
            LongestMaxSize(
                max_size=max_size,
                interpolation=interpolation,
                mask_interpolation=mask_interpolation,
                p=1.0,
            ),
            PadIfNeeded(
                min_height=target_h,
                min_width=target_w,
                border_mode=cv2.BORDER_CONSTANT,
                fill=fill,
                fill_mask=fill_mask,
                position=cast(
                    "Literal['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'random']",
                    pad_position,
                ),
                p=1.0,
            ),
        ],
        p=1.0,
    )  # Explicitly set p=1 for the Compose itself

    # Prepare data for the pipeline (needs image, potentially mask, bboxes, keypoints)
    pipeline_input = {"image": cell_data["image"]}
    if "mask" in cell_data and cell_data["mask"] is not None:
        pipeline_input["mask"] = cell_data["mask"]
    # Warning: Bbox/Keypoint handling might be inaccurate without Compose setup
    if "bboxes" in cell_data and cell_data["bboxes"] is not None:
        pipeline_input["bboxes"] = cell_data["bboxes"]
    if "keypoints" in cell_data and cell_data["keypoints"] is not None:
        pipeline_input["keypoints"] = cell_data["keypoints"]

    try:
        # Apply the pipeline
        processed_data = pipeline(**pipeline_input)
    except (ValueError, IndexError, cv2.error) as e:  # Catch more specific errors
        warnings.warn(
            f"Error processing mosaic cell {grid_pos} with Compose: {e}. Returning empty data.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    # Ensure the output image has the exact target dimensions
    if "image" not in processed_data:
        warnings.warn(
            f"Compose pipeline failed to return an image for cell {grid_pos}. Returning empty data.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    out_h, out_w = processed_data["image"].shape[:2]
    if out_h != target_h or out_w != target_w:
        msg = (
            f"Processed cell {grid_pos} size ({out_h}x{out_w}) doesn't match target ({target_h}x{target_w}). Check pad."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # Shift bboxes and keypoints
    tgt_x1, tgt_y1, _, _ = target_placement

    # Define shift vectors (using int32 for safety with pixel coords)
    kp_shift_vector = np.array([tgt_x1, tgt_y1, 0], dtype=np.int32)
    # Assuming shift_bboxes needs [shift_x_min, shift_y_min, shift_x_max, shift_y_max]
    bbox_shift_vector = np.array([tgt_x1, tgt_y1, tgt_x1, tgt_y1], dtype=np.int32)

    if "bboxes" in processed_data and processed_data["bboxes"] is not None and len(processed_data["bboxes"]) > 0:
        bboxes_np = (
            np.array(processed_data["bboxes"])
            if not isinstance(processed_data["bboxes"], np.ndarray)
            else processed_data["bboxes"]
        )
        # Correct call using shift_vector
        processed_data["bboxes"] = fgeometric.shift_bboxes(bboxes_np, bbox_shift_vector)

    if (
        "keypoints" in processed_data
        and processed_data["keypoints"] is not None
        and len(processed_data["keypoints"]) > 0
    ):
        keypoints_np = (
            np.array(processed_data["keypoints"])
            if not isinstance(processed_data["keypoints"], np.ndarray)
            else processed_data["keypoints"]
        )
        # Correct call using shift_vector
        processed_data["keypoints"] = fgeometric.shift_keypoints(keypoints_np, kp_shift_vector)

    # Return directly (Fix RET504)
    return {k: v for k, v in processed_data.items() if k in pipeline_input}


def assemble_mosaic_image_mask(
    processed_data: dict[tuple[int, int], dict[str, Any]],
    placements: dict[tuple[int, int], tuple[int, int, int, int]],
    target_size: tuple[int, int],
    num_channels: int,
    fill: float | tuple[float, ...],
    dtype: np.dtype,
    data_key: str,
) -> np.ndarray | None:
    """Assembles the final mosaic image or mask from processed cell data."""
    target_h, target_w = target_size

    # Create blank canvas
    canvas_shape = (target_h, target_w, num_channels) if num_channels > 0 else (target_h, target_w)
    extended_fill_value = fgeometric.extend_value(fill, num_channels)
    canvas = np.full(canvas_shape, extended_fill_value, dtype=dtype)

    for grid_pos, placement_coords in placements.items():
        if grid_pos not in processed_data:
            continue  # Should not happen if called correctly

        cell_data = processed_data[grid_pos]
        segment = cell_data.get(data_key)

        if segment is None:
            # If assembling masks, we might have already checked consistency.
            # If assembling images, this indicates an error in processing.
            if data_key == "image":
                warnings.warn(
                    f"Missing '{data_key}' in processed data for cell {grid_pos}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue  # Skip if the required data is missing

        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = placement_coords
        place_h = tgt_y2 - tgt_y1
        place_w = tgt_x2 - tgt_x1

        # Check if placement area is valid
        if place_h <= 0 or place_w <= 0:
            continue  # Skip pasting if target area is non-positive

        # Check consistency and paste
        if segment.shape[0] == place_h and segment.shape[1] == place_w:
            try:
                # Ensure target slice is valid within canvas bounds
                y_slice = slice(max(0, tgt_y1), min(target_h, tgt_y2))
                x_slice = slice(max(0, tgt_x1), min(target_w, tgt_x2))

                # Ensure source segment slice matches target slice dimensions
                seg_h, seg_w = segment.shape[:2]
                seg_y_start = max(0, -tgt_y1)
                seg_x_start = max(0, -tgt_x1)
                seg_y_end = seg_y_start + (y_slice.stop - y_slice.start)
                seg_x_end = seg_x_start + (x_slice.stop - x_slice.start)

                # Double check slice validity before assignment
                if 0 <= seg_y_start < seg_y_end <= seg_h and 0 <= seg_x_start < seg_x_end <= seg_w:
                    canvas[y_slice, x_slice] = segment[seg_y_start:seg_y_end, seg_x_start:seg_x_end]
                else:
                    warnings.warn(
                        f"Calculated invalid slice for segment/canvas for cell {grid_pos}. Skipping paste.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            except (ValueError, IndexError) as e:  # Catch more specific errors for pasting
                warnings.warn(f"Error pasting segment for cell {grid_pos}: {e}", RuntimeWarning, stacklevel=2)

        else:
            msg = (
                f"Size mismatch for {data_key} at {grid_pos}: "
                f"target {place_h}x{place_w}, got {segment.shape[:2]}. Skip."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return canvas


def calculate_mosaic_center_point(
    grid_yx: tuple[int, int],
    target_size: tuple[int, int],
    center_range: tuple[float, float],
    py_random: random.Random,
) -> tuple[int, int]:
    """Calculates the center point for the mosaic crop.

    Ensures the center point allows a crop of target_size to overlap
    all grid cells, applying randomness based on center_range within
    the valid zone.

    Args:
        grid_yx (tuple[int, int]): The (rows, cols) of the mosaic grid.
        target_size (tuple[int, int]): The final output (height, width).
        center_range (tuple[float, float]): Range [0.0-1.0] for sampling center proportionally
                                            within the valid zone.
        py_random (random.Random): Random state instance.

    Returns:
        tuple[int, int]: The calculated (x, y) center point relative to the
                         top-left of the conceptual large grid.

    """
    rows, cols = grid_yx
    target_h, target_w = target_size
    large_grid_h = rows * target_h
    large_grid_w = cols * target_w

    # Define valid center range on the large grid to ensure full target crop
    # Top-left corner of the crop window should be <= center
    # Bottom-right corner should be >= center
    min_center_x = (target_w - 1) // 2
    max_center_x_incl = large_grid_w - 1 - (target_w // 2)  # Inclusive max index
    min_center_y = (target_h - 1) // 2
    max_center_y_incl = large_grid_h - 1 - (target_h // 2)

    if max_center_x_incl < min_center_x or max_center_y_incl < min_center_y:
        warnings.warn(
            f"Target size {target_size} is too large for the conceptual grid {grid_yx}. "
            "Cannot guarantee overlap with all cells. Centering crop instead.",
            UserWarning,
            stacklevel=3,  # Adjust stacklevel based on call depth
        )
        center_x = large_grid_w // 2
        center_y = large_grid_h // 2
    else:
        # Calculate the valid range (width/height of the safe zone)
        safe_zone_w = max_center_x_incl - min_center_x + 1  # +1 because range is inclusive
        safe_zone_h = max_center_y_incl - min_center_y + 1

        # Sample offsets within the safe zone based on center_range
        # Using uniform(a, b) samples in [a, b), so adjust range slightly if needed,
        # but int conversion handles it okay.
        offset_x = int(safe_zone_w * py_random.uniform(*center_range))
        offset_y = int(safe_zone_h * py_random.uniform(*center_range))

        # Calculate final center point
        center_x = min_center_x + offset_x
        center_y = min_center_y + offset_y

        # Safety clip to ensure center is within the large grid bounds
        # This shouldn't be strictly necessary if calculations above are correct, but good practice
        center_x = max(0, min(center_x, large_grid_w - 1))
        center_y = max(0, min(center_y, large_grid_h - 1))

    return center_x, center_y


def process_mosaic_grid(
    prepared_grid_data: dict[tuple[int, int], dict[str, Any]],
    transforms_coords: dict[tuple[int, int], dict[str, tuple[int, int, int, int]]],
    grid_yx: tuple[int, int],
    interpolation: int,
    mask_interpolation: int,
    fill: float | tuple[float, ...],
    fill_mask: float | tuple[float, ...],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], tuple[int, int, int, int]]]:
    """Processes each cell in the mosaic grid using process_mosaic_cell.

    Iterates through the calculated transform coordinates, processes the
    corresponding cell data (cropping, resizing, padding), and shifts labels.

    Args:
        prepared_grid_data: Dict mapping (row, col) to prepared cell data.
        transforms_coords: Dict mapping (row, col) to source/target coords.
        grid_yx: The (rows, cols) of the mosaic grid.
        interpolation: OpenCV interpolation for images.
        mask_interpolation: OpenCV interpolation for masks.
        fill: Fill value for image padding.
        fill_mask: Fill value for mask padding.

    Returns:
        tuple containing:
            - final_processed_data: Dict mapping (row, col) to fully processed cell data.
            - final_placements: Dict mapping (row, col) to target placement coordinates.
              Returns empty dicts if processing fails for all cells with coordinates.

    """
    final_processed_data = {}
    final_placements = {}

    for grid_pos, coords_dict in transforms_coords.items():
        if grid_pos not in prepared_grid_data:
            warnings.warn(
                f"Mismatch: Transform coords exist for {grid_pos} but prepared data does not.",
                RuntimeWarning,
                stacklevel=3,
            )  # Adjusted stacklevel
            continue  # Skip this cell

        cell_data = prepared_grid_data[grid_pos]

        # Process image and mask using the temporary Compose pipeline
        processed_cell_data = process_mosaic_cell(  # Assumes process_mosaic_cell exists
            cell_data=cell_data,
            transform_coords=coords_dict,
            grid_pos=grid_pos,
            grid_yx=grid_yx,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            fill=fill,
            fill_mask=fill_mask,
        )

        # Check if cell processing failed (returned empty dict)
        if not processed_cell_data:
            warnings.warn(
                f"Processing failed for cell {grid_pos}. Skipping.",
                RuntimeWarning,
                stacklevel=3,
            )  # Adjusted stacklevel
            continue  # Skip this cell if processing failed

        # Get target placement offset (shift is handled inside process_mosaic_cell)
        target_placement = coords_dict["target_placement"]

        final_processed_data[grid_pos] = processed_cell_data
        final_placements[grid_pos] = target_placement  # Store placement coords

    return final_processed_data, final_placements
