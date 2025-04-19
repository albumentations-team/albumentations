"""Functional implementations for image mixing operations.

This module provides utility functions for blending and combining images,
such as copy-and-paste operations with masking.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal, TypedDict, cast
from warnings import warn

import cv2
import numpy as np

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.crops.transforms import Crop
from albumentations.augmentations.geometric.transforms import PadIfNeeded
from albumentations.core.bbox_utils import BboxProcessor
from albumentations.core.composition import Compose
from albumentations.core.keypoints_utils import KeypointsProcessor


# Type definition for a processed mosaic item
class ProcessedMosaicItem(TypedDict):
    """Represents a single data item (primary or additional) after preprocessing.

    Includes the original image/mask and the *preprocessed* annotations.
    """

    image: np.ndarray  # Image is mandatory
    mask: np.ndarray | None
    bboxes: np.ndarray | None
    keypoints: np.ndarray | None


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


def calculate_mosaic_center_point(
    grid_yx: tuple[int, int],
    target_size: tuple[int, int],
    center_range: tuple[float, float],
    py_random: random.Random,
) -> tuple[int, int]:
    """Calculates the center point for the mosaic crop using proportional sampling within the valid zone.

    Ensures the center point allows a crop of target_size to overlap
    all grid cells, applying randomness based on center_range proportionally
    within the valid region where the center can lie.

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

    # Define valid center range bounds (inclusive)
    # The center must be far enough from edges so the crop window fits
    min_cx = target_w // 2
    max_cx = large_grid_w - (target_w + 1) // 2
    min_cy = target_h // 2
    max_cy = large_grid_h - (target_h + 1) // 2

    # Calculate valid range dimensions (size of the safe zone)
    valid_w = max_cx - min_cx + 1
    valid_h = max_cy - min_cy + 1

    # Sample relative position within the valid range using center_range
    rel_x = py_random.uniform(*center_range)
    rel_y = py_random.uniform(*center_range)

    # Calculate center coordinates by scaling relative position within valid range
    # Add the minimum bound to shift the range start
    center_x = min_cx + int(valid_w * rel_x)
    center_y = min_cy + int(valid_h * rel_y)

    # Ensure the result is strictly within the calculated bounds after int conversion
    # (This clip is mostly a safety measure, shouldn't be needed with correct int conversion)
    center_x = max(min_cx, min(center_x, max_cx))
    center_y = max(min_cy, min(center_y, max_cy))

    return center_x, center_y


def calculate_cell_placements(
    grid_yx: tuple[int, int],
    target_size: tuple[int, int],
    center_xy: tuple[int, int],
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """Calculates the placement coordinates for grid cells that appear in the final crop.

    Args:
        grid_yx (tuple[int, int]): The (rows, cols) of the mosaic grid.
        target_size (tuple[int, int]): The final output (height, width).
        center_xy (tuple[int, int]): The calculated (x, y) center of the final crop window,
                                        relative to the top-left of the conceptual large grid.

    Returns:
        dict[tuple[int, int], tuple[int, int, int, int]]:
            A dictionary mapping grid cell coordinates `(r, c)` of visible cells to
            their placement coordinates `(x_min, y_min, x_max, y_max)` on the
            final output canvas.

    """
    rows, cols = grid_yx
    target_h, target_w = target_size
    center_x, center_y = center_xy

    # Calculate the boundaries of the final crop window on the large conceptual grid
    large_grid_h = rows * target_h
    large_grid_w = cols * target_w

    # Ensure center point is within valid calculation range relative to target size
    if not (0 <= center_x <= large_grid_w and 0 <= center_y <= large_grid_h):
        warn(
            f"Center point {center_xy} is outside the conceptual large grid dimensions "
            f"({large_grid_h}x{large_grid_w}). Cannot calculate placements.",
            UserWarning,
            stacklevel=2,
        )
        return {}

    # Determine the crop window coordinates relative to the large grid's top-left (0,0)
    crop_x_min = center_x - target_w // 2
    crop_y_min = center_y - target_h // 2
    crop_x_max = crop_x_min + target_w
    crop_y_max = crop_y_min + target_h

    cell_placements = {}

    for row_idx in range(rows):
        for col_idx in range(cols):
            # Boundaries of the current cell on the large grid
            cell_x_min = col_idx * target_w
            cell_y_min = row_idx * target_h
            cell_x_max = cell_x_min + target_w
            cell_y_max = cell_y_min + target_h

            # Find the intersection of the final crop window and the current cell
            inter_x_min = max(crop_x_min, cell_x_min)
            inter_y_min = max(crop_y_min, cell_y_min)
            inter_x_max = min(crop_x_max, cell_x_max)
            inter_y_max = min(crop_y_max, cell_y_max)

            # If there is a valid intersection (positive area)
            if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                # Calculate placement coordinates relative to the final output canvas's top-left (0,0)
                tgt_x1 = inter_x_min - crop_x_min
                tgt_y1 = inter_y_min - crop_y_min
                tgt_x2 = inter_x_max - crop_x_min
                tgt_y2 = inter_y_max - crop_y_min

                # Clip placement coords to target canvas boundaries
                tgt_x1_clipped = max(0, tgt_x1)
                tgt_y1_clipped = max(0, tgt_y1)
                tgt_x2_clipped = min(target_w, tgt_x2)
                tgt_y2_clipped = min(target_h, tgt_y2)

                # Only add if the *clipped* placement still has positive area
                if tgt_x2_clipped > tgt_x1_clipped and tgt_y2_clipped > tgt_y1_clipped:
                    cell_placements[(row_idx, col_idx)] = (
                        tgt_x1_clipped,
                        tgt_y1_clipped,
                        tgt_x2_clipped,
                        tgt_y2_clipped,
                    )

    return cell_placements


def _gather_mosaic_item_data(
    item: dict[str, Any],
    data_key: Literal["bboxes", "keypoints"],
    label_fields: list[str],
    all_raw_list: list[np.ndarray],
    labels_gathered_dict: dict[str, list[Any]],
) -> int:
    """Gathers raw data (bboxes/keypoints) and associated labels for a single item.

    Returns:
        int: The number of items (bboxes or keypoints) found for this item.

    """
    item_data = item.get(data_key)  # Assume np.ndarray or None
    count = 0
    if item_data is not None and item_data.size > 0:
        count = len(item_data)
        all_raw_list.append(item_data)
        # Gather corresponding labels
        for field in label_fields:
            labels = item.get(field)
            if labels is not None and len(labels) == count:
                labels_gathered_dict[field].extend(labels)
            else:
                # Pad with None if mismatch or missing - processor must handle Nones if required
                labels_gathered_dict[field].extend([None] * count)
    return count


def _preprocess_combined_data(
    all_raw_items: list[np.ndarray],
    labels_gathered: dict[str, list[Any]],
    processor: BboxProcessor | KeypointsProcessor,
    data_key: Literal["bboxes", "keypoints"],
) -> np.ndarray:
    """Helper to preprocess combined lists of bboxes or keypoints using a processor."""
    combined_raw_np = np.vstack(all_raw_items)

    # Prepare input for the processor
    input_combined: dict[str, Any] = {data_key: combined_raw_np}
    total_items = len(combined_raw_np)
    for field, labels in labels_gathered.items():
        if len(labels) == total_items:
            input_combined[field] = labels
        else:
            warn(
                f"Length mismatch for combined {data_key} label field '{field}'. "
                f"Expected {total_items}, got {len(labels)}. Skipping field.",
                UserWarning,
                stacklevel=3,
            )

    processor.preprocess(input_combined)
    return np.array(input_combined[data_key])


def filter_valid_metadata(
    metadata_input: Sequence[dict[str, Any]] | None,
    metadata_key_name: str,  # Need the key name for warnings
) -> list[dict[str, Any]]:
    """Filters a list of metadata dicts, keeping only valid ones.

    Valid items must be dictionaries and contain the 'image' key.
    """
    if not isinstance(metadata_input, Sequence):
        warn(
            f"Metadata under key '{metadata_key_name}' is not a Sequence (e.g., list or tuple). "
            f"Returning empty list for additional items.",
            UserWarning,
            stacklevel=3,
        )
        return []

    valid_items = []
    for i, item in enumerate(metadata_input):
        if isinstance(item, dict) and "image" in item:
            valid_items.append(item)
        else:
            warn(
                f"Item at index {i} in '{metadata_key_name}' is invalid "
                f"(not a dict or lacks 'image' key) and will be skipped.",
                UserWarning,
                stacklevel=4,
            )
    return valid_items


def assign_items_to_grid_cells(
    num_items: int,
    cell_placements: dict[tuple[int, int], tuple[int, int, int, int]],
    py_random: random.Random,
) -> dict[tuple[int, int], int]:
    """Assigns item indices to grid cells based on placement data.

    Assigns the primary item (index 0) to the cell with the largest area,
    and assigns the remaining items (indices 1 to num_items-1) randomly to the
    remaining cells specified in cell_placements.

    Args:
        num_items (int): The total number of items to assign (primary + additional + replicas).
        cell_placements (dict): Maps grid coords (r, c) to placement
                                coords (x1, y1, x2, y2) for cells to be filled.
        py_random (random.Random): Random state instance.

    Returns:
        dict[tuple[int, int], int]: Dict mapping grid coords (r, c) to assigned item index.

    """
    grid_positions = list(cell_placements.keys())
    if not grid_positions:
        return {}

    # Find the grid cell with the largest area for primary placement
    primary_placement_pos = max(
        grid_positions,
        key=lambda pos: (cell_placements[pos][2] - cell_placements[pos][0])
        * (cell_placements[pos][3] - cell_placements[pos][1]),
    )

    grid_coords_to_item_index: dict[tuple[int, int], int] = {}
    grid_coords_to_item_index[primary_placement_pos] = 0  # Assign primary item (index 0)

    remaining_grid_positions = [pos for pos in grid_positions if pos != primary_placement_pos]
    # Indices for additional/replicated items start from 1
    remaining_item_indices = list(range(1, num_items))
    py_random.shuffle(remaining_item_indices)

    num_to_assign = min(len(remaining_grid_positions), len(remaining_item_indices))
    if len(remaining_grid_positions) != len(remaining_item_indices):
        msg = (
            f"Mismatch: {len(remaining_grid_positions)} remaining cells, but {len(remaining_item_indices)} "
            f"remaining items. Assigning {num_to_assign}."
        )
        warn(
            msg,
            UserWarning,
            stacklevel=3,
        )

    for i in range(num_to_assign):
        grid_coords_to_item_index[remaining_grid_positions[i]] = remaining_item_indices[i]

    return grid_coords_to_item_index


def preprocess_selected_mosaic_items(
    selected_raw_items: list[dict[str, Any]],
    bbox_processor: BboxProcessor,
    keypoint_processor: KeypointsProcessor,
) -> list[ProcessedMosaicItem]:
    """Preprocesses bboxes/keypoints for selected raw additional items combined.

    Gathers raw bboxes/kps/labels, preprocesses them together using the processors
    (updating label encoders), and returns a list of dicts with original image/mask
    and the corresponding chunk of preprocessed bboxes/keypoints.
    """
    if not selected_raw_items:
        return []

    # Gather combined raw data and counts from selected items
    all_raw_bboxes: list[np.ndarray] = []
    all_raw_keypoints: list[np.ndarray] = []

    bbox_labels_gathered: dict[str, list[Any]] = defaultdict(list)
    kp_labels_gathered: dict[str, list[Any]] = defaultdict(list)
    bbox_counts_per_item: list[int] = []
    keypoint_counts_per_item: list[int] = []

    bbox_label_fields = cast(
        "list[str]",
        bbox_processor.params.label_fields if bbox_processor and bbox_processor.params.label_fields else [],
    )
    kp_label_fields = cast(
        "list[str]",
        keypoint_processor.params.label_fields if keypoint_processor and keypoint_processor.params.label_fields else [],
    )

    for item in selected_raw_items:
        num_bboxes = _gather_mosaic_item_data(
            item,
            "bboxes",
            bbox_label_fields,
            all_raw_bboxes,
            bbox_labels_gathered,
        )
        bbox_counts_per_item.append(num_bboxes)
        num_keypoints = _gather_mosaic_item_data(
            item,
            "keypoints",
            kp_label_fields,
            all_raw_keypoints,
            kp_labels_gathered,
        )
        keypoint_counts_per_item.append(num_keypoints)

    # Preprocess combined data using the helper
    combined_preprocessed_bboxes = _preprocess_combined_data(
        all_raw_bboxes,
        bbox_labels_gathered,
        bbox_processor,
        "bboxes",
    )
    combined_preprocessed_keypoints = _preprocess_combined_data(
        all_raw_keypoints,
        kp_labels_gathered,
        keypoint_processor,
        "keypoints",
    )

    # Split back into per-item dictionaries
    result_data_items: list[ProcessedMosaicItem] = []
    bbox_start_idx = 0
    kp_start_idx = 0

    for i, item in enumerate(selected_raw_items):
        processed_item_dict: ProcessedMosaicItem = {
            "image": item["image"],  # Keep original image
            "mask": item.get("mask"),  # Keep original mask
            "bboxes": None,
            "keypoints": None,
        }
        num_bboxes = bbox_counts_per_item[i]
        if num_bboxes > 0:
            bbox_end_idx = bbox_start_idx + num_bboxes
            processed_item_dict["bboxes"] = combined_preprocessed_bboxes[bbox_start_idx:bbox_end_idx]
            bbox_start_idx = bbox_end_idx
        else:
            processed_item_dict["bboxes"] = np.empty(
                (0, combined_preprocessed_bboxes.shape[1]),
                dtype=combined_preprocessed_bboxes.dtype,
            )
        num_keypoints = keypoint_counts_per_item[i]
        if num_keypoints > 0:
            kp_end_idx = kp_start_idx + num_keypoints
            processed_item_dict["keypoints"] = combined_preprocessed_keypoints[kp_start_idx:kp_end_idx]
            kp_start_idx = kp_end_idx
        else:
            processed_item_dict["keypoints"] = np.empty(
                (0, combined_preprocessed_keypoints.shape[1]),
                dtype=combined_preprocessed_keypoints.dtype,
            )
        result_data_items.append(processed_item_dict)

    return result_data_items


def process_cell_geometry(
    item: ProcessedMosaicItem,
    target_h: int,
    target_w: int,
    fill: float | tuple[float, ...],
    fill_mask: float | tuple[float, ...],
    pad_position: Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"] = "top_left",
    border_mode: int = cv2.BORDER_CONSTANT,
) -> ProcessedMosaicItem:
    """Applies geometric transformations (padding and/or cropping) to a single mosaic item.

    Uses a Compose pipeline with PadIfNeeded and Crop to ensure the output
    matches the target cell dimensions exactly, handling both padding and cropping cases.

    Args:
        item: (ProcessedMosaicItem): The preprocessed mosaic item dictionary.
        target_h: (int): Target height of the cell.
        target_w: (int): Target width of the cell.
        fill: (float | tuple[float, ...]): Fill value for image padding.
        fill_mask: (float | tuple[float, ...]): Fill value for mask padding.
        pad_position: (Literal[...]): Position for padding if item is smaller than target. Default: "top_left".
        border_mode: (int): OpenCV border mode for padding.

    Returns: (ProcessedMosaicItem): Dictionary containing the geometrically processed image,
        mask, bboxes, and keypoints, fitting the target dimensions.

    """
    # Define the pipeline: PadIfNeeded first, then Crop
    compose_kwargs: dict[str, Any] = {"p": 1.0}
    if item.get("bboxes") is not None:
        compose_kwargs["bbox_params"] = {"format": "albumentations"}
    if item.get("keypoints") is not None:
        compose_kwargs["keypoint_params"] = {"format": "albumentations"}

    geom_pipeline = Compose(
        [
            PadIfNeeded(
                min_height=target_h,
                min_width=target_w,
                position=pad_position,
                border_mode=border_mode,
                fill=fill,
                fill_mask=fill_mask,
                p=1.0,
            ),
            Crop(
                x_min=0,
                y_min=0,
                x_max=target_w,
                y_max=target_h,
                p=1.0,
            ),
        ],
        **compose_kwargs,
    )

    # Prepare input data for the pipeline
    geom_input = {"image": item["image"]}
    if item.get("mask") is not None:
        geom_input["mask"] = item["mask"]
    if item.get("bboxes") is not None:
        # Compose expects bboxes in a specific format, ensure it's compatible
        # Assuming item['bboxes'] is already preprocessed correctly
        geom_input["bboxes"] = item["bboxes"]
    if item.get("keypoints") is not None:
        geom_input["keypoints"] = item["keypoints"]

    # Apply the pipeline
    processed_item = geom_pipeline(**geom_input)

    # Ensure output dict has the same structure as ProcessedMosaicItem
    # Compose might not return None for missing keys, handle explicitly
    return {
        "image": processed_item["image"],
        "mask": processed_item.get("mask"),
        "bboxes": processed_item.get("bboxes"),
        "keypoints": processed_item.get("keypoints"),
    }


def shift_cell_coordinates(
    processed_item_geom: ProcessedMosaicItem,
    placement_coords: tuple[int, int, int, int],
) -> ProcessedMosaicItem:
    """Shifts the coordinates of geometrically processed bboxes and keypoints.

    Args:
        processed_item_geom: (ProcessedMosaicItem): The output from process_cell_geometry.
        placement_coords: (tuple[int, int, int, int]): The (x1, y1, x2, y2) placement on the final canvas.

    Returns: (ProcessedMosaicItem): A dictionary with keys 'bboxes' and 'keypoints', containing the shifted
        numpy arrays (potentially empty).

    """
    tgt_x1, tgt_y1, _, _ = placement_coords

    bboxes_geom = processed_item_geom.get("bboxes")
    if bboxes_geom is not None and np.asarray(bboxes_geom).size > 0:
        bboxes_geom_arr = np.asarray(bboxes_geom)  # Ensure it's an array
        bbox_shift_vector = np.array([tgt_x1, tgt_y1, tgt_x1, tgt_y1], dtype=np.int32)
        shifted_bboxes = fgeometric.shift_bboxes(bboxes_geom_arr, bbox_shift_vector)

    keypoints_geom = processed_item_geom.get("keypoints")
    if keypoints_geom is not None and np.asarray(keypoints_geom).size > 0:
        keypoints_geom_arr = np.asarray(keypoints_geom)  # Ensure it's an array
        kp_shift_vector = np.array([tgt_x1, tgt_y1, 0], dtype=np.int32)
        shifted_keypoints = fgeometric.shift_keypoints(keypoints_geom_arr, kp_shift_vector)

    return {
        "bboxes": shifted_bboxes,
        "keypoints": shifted_keypoints,
        "image": processed_item_geom["image"],
        "mask": processed_item_geom.get("mask"),
    }


def assemble_mosaic_from_processed_cells(
    processed_cells: dict[tuple[int, int, int, int], dict[str, Any]],
    target_shape: tuple[int, int],
    dtype: np.dtype,
    data_key: Literal["image", "mask"],
) -> np.ndarray:
    """Assembles the final mosaic image or mask from processed cell data onto a zero canvas.
    Assumes input data is valid and correctly sized.
    """
    canvas = np.zeros(target_shape, dtype=dtype)

    # Iterate and paste
    for placement_coords, cell_data in processed_cells.items():
        segment = cell_data.get(data_key)
        if segment is not None:  # Minimal check to avoid error on .get() returning None
            tgt_x1, tgt_y1, tgt_x2, tgt_y2 = placement_coords
            # Use direct colon notation for slicing
            canvas[tgt_y1:tgt_y2, tgt_x1:tgt_x2] = segment

    return canvas


def process_all_mosaic_geometries(
    grid_coords_to_item_index: dict[tuple[int, int], int],
    final_items_for_grid: list[ProcessedMosaicItem],
    cell_placements: dict[tuple[int, int], tuple[int, int, int, int]],
    fill: float | tuple[float, ...],
    fill_mask: float | tuple[float, ...],
) -> dict[tuple[int, int, int, int], ProcessedMosaicItem]:  # Key is placement coords, value is processed item
    """Processes the geometry (cropping/padding) for all assigned mosaic cells.

    Iterates through assigned cells, applies geometric transforms, and returns
    a dictionary mapping final placement coordinates to the processed item data.
    The bbox/keypoint coordinates in the returned dict are *not* shifted yet.

    Args:
        grid_coords_to_item_index (dict[tuple[int, int], int]): Mapping from grid position (r, c) to item index.
        final_items_for_grid (list[ProcessedMosaicItem]): List of all preprocessed items available for the grid.
        cell_placements (dict[tuple[int, int], tuple[int, int, int, int]]): Mapping from grid
            position (r, c) to placement coords (x1, y1, x2, y2).
        fill (float | tuple[float, ...]): Fill value for image padding.
        fill_mask (float | tuple[float, ...]): Fill value for mask padding.

    Returns:
        dict[tuple[int, int, int, int], ProcessedMosaicItem]: Dictionary mapping final placement
        coordinates (x1, y1, x2, y2) to the geometrically processed item data (image, mask, un-shifted bboxes/kps).

    """
    processed_cells_geom: dict[tuple[int, int, int, int], ProcessedMosaicItem] = {}

    for grid_pos, item_idx in grid_coords_to_item_index.items():
        item = final_items_for_grid[item_idx]
        placement_coords = cell_placements[grid_pos]
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = placement_coords
        target_h = tgt_y2 - tgt_y1
        target_w = tgt_x2 - tgt_x1

        # Apply geometric processing (crop/pad)
        processed_cells_geom[placement_coords] = process_cell_geometry(
            item=item,
            target_h=target_h,
            target_w=target_w,
            fill=fill,
            fill_mask=fill_mask,
        )

    return processed_cells_geom


def shift_all_coordinates(
    processed_cells_geom: dict[tuple[int, int, int, int], ProcessedMosaicItem],
) -> dict[tuple[int, int, int, int], ProcessedMosaicItem]:  # Return type matches input, but values are updated
    """Shifts coordinates for all geometrically processed cells.

    Iterates through the processed cells (keyed by placement coords), applies coordinate
    shifting to bboxes/keypoints, and returns a new dictionary with the same keys
    but updated ProcessedMosaicItem values containing the *shifted* coordinates.

    Args:
        processed_cells_geom (dict[tuple[int, int, int, int], ProcessedMosaicItem]):
             Output from process_all_mosaic_geometries (keyed by placement coords).

    Returns:
        dict[tuple[int, int, int, int], ProcessedMosaicItem]: Final dictionary mapping
        placement coords (x1, y1, x2, y2) to processed cell data with shifted coordinates.

    """
    final_processed_cells: dict[tuple[int, int, int, int], ProcessedMosaicItem] = {}

    for placement_coords, cell_data_geom in processed_cells_geom.items():
        tgt_x1, tgt_y1 = placement_coords[:2]

        # Extract geometrically processed bboxes/keypoints
        bboxes_geom = cell_data_geom.get("bboxes")
        keypoints_geom = cell_data_geom.get("keypoints")

        final_cell_data = {
            "image": cell_data_geom["image"],
            "mask": cell_data_geom.get("mask"),
        }

        # Perform shifting if data exists
        if bboxes_geom is not None and bboxes_geom.size > 0:
            bboxes_geom_arr = np.asarray(bboxes_geom)
            bbox_shift_vector = np.array([tgt_x1, tgt_y1, tgt_x1, tgt_y1], dtype=np.int32)
            shifted_bboxes = fgeometric.shift_bboxes(bboxes_geom_arr, bbox_shift_vector)
            final_cell_data["bboxes"] = shifted_bboxes

        if keypoints_geom is not None and keypoints_geom.size > 0:
            keypoints_geom_arr = np.asarray(keypoints_geom)
            kp_shift_vector = np.array([tgt_x1, tgt_y1, 0], dtype=np.int32)
            shifted_keypoints = fgeometric.shift_keypoints(keypoints_geom_arr, kp_shift_vector)
            final_cell_data["keypoints"] = shifted_keypoints

        final_processed_cells[placement_coords] = cast("ProcessedMosaicItem", final_cell_data)

    return final_processed_cells
