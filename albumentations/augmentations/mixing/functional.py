"""Functional implementations for image mixing operations.

This module provides utility functions for blending and combining images,
such as copy-and-paste operations with masking.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, Literal, TypedDict, cast
from warnings import warn

import cv2
import numpy as np

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.crops.transforms import Crop
from albumentations.augmentations.geometric.resize import LongestMaxSize, SmallestMaxSize
from albumentations.core.bbox_utils import BboxProcessor, denormalize_bboxes, normalize_bboxes
from albumentations.core.composition import Compose
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.type_definitions import (
    NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS,
    NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS,
)


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
    cell_shape: tuple[int, int],
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
        cell_shape (tuple[int, int]): Shape of each cell in the mosaic grid.
        target_size (tuple[int, int]): The final output (height, width).
        center_range (tuple[float, float]): Range [0.0-1.0] for sampling center proportionally
                                            within the valid zone.
        py_random (random.Random): Random state instance.

    Returns:
        tuple[int, int]: The calculated (x, y) center point relative to the
                         top-left of the conceptual large grid.

    """
    rows, cols = grid_yx
    cell_h, cell_w = cell_shape
    target_h, target_w = target_size

    large_grid_h = rows * cell_h
    large_grid_w = cols * cell_w

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
    cell_shape: tuple[int, int],
    target_size: tuple[int, int],
    center_xy: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Calculates placements by clipping arange-defined grid lines to the crop window.

    Args:
        grid_yx (tuple[int, int]): The (rows, cols) of the mosaic grid.
        cell_shape (tuple[int, int]): Shape of each cell in the mosaic grid.
        target_size (tuple[int, int]): The final output (height, width).
        center_xy (tuple[int, int]): The calculated (x, y) center of the final crop window,
                                        relative to the top-left of the conceptual large grid.

    Returns:
        list[tuple[int, int, int, int]]:
            A list containing placement coordinates `(x_min, y_min, x_max, y_max)`
            for each resulting cell part on the final output canvas.

    """
    rows, cols = grid_yx
    cell_h, cell_w = cell_shape
    target_h, target_w = target_size
    center_x, center_y = center_xy

    # 1. Generate grid line coordinates using arange for the large grid
    y_coords_large = np.arange(rows + 1) * cell_h
    x_coords_large = np.arange(cols + 1) * cell_w

    # 2. Calculate Crop Window boundaries
    crop_x_min = center_x - target_w // 2
    crop_y_min = center_y - target_h // 2
    crop_x_max = crop_x_min + target_w
    crop_y_max = crop_y_min + target_h

    # 3. Clip grid lines to the crop window
    y_coords_clipped = np.clip(y_coords_large, crop_y_min, crop_y_max)
    x_coords_clipped = np.clip(x_coords_large, crop_x_min, crop_x_max)

    # 4. Get unique sorted coordinates
    y_coords_unique = np.unique(y_coords_clipped) - crop_y_min
    x_coords_unique = np.unique(x_coords_clipped) - crop_x_min

    # 5. Form all cell coordinates using numpy operations
    x_mins = x_coords_unique[:-1]
    x_maxs = x_coords_unique[1:]
    y_mins = y_coords_unique[:-1]
    y_maxs = y_coords_unique[1:]

    num_x_intervals = len(x_mins)
    num_y_intervals = len(y_mins)

    # Create all combinations of intervals (Cartesian product)
    all_x_mins = np.tile(x_mins, num_y_intervals)
    all_y_mins = np.repeat(y_mins, num_x_intervals)
    all_x_maxs = np.tile(x_maxs, num_y_intervals)
    all_y_maxs = np.repeat(y_maxs, num_x_intervals)

    # Stack into (N, 4) array of [x_min, y_min, x_max, y_max] mapped to target origin
    np_result = np.stack([all_x_mins, all_y_mins, all_x_maxs, all_y_maxs], axis=-1)

    # Convert final numpy array to list of tuples of ints
    # Note: Assumes np_result contains valid, non-zero area coordinates within target bounds
    return cast("list[tuple[int, int, int, int]]", [tuple(map(int, row)) for row in np_result])


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
    cell_placements: list[tuple[int, int, int, int]],
    py_random: random.Random,
) -> dict[tuple[int, int, int, int], int]:
    """Assigns item indices to placement coordinate tuples.

    Assigns the primary item (index 0) to the placement with the largest area,
    and assigns the remaining items (indices 1 to num_items-1) randomly to the
    remaining placements.

    Args:
        num_items (int): The total number of items to assign (primary + additional + replicas).
        cell_placements (list[tuple[int, int, int, int]]): List of placement
                                coords (x1, y1, x2, y2) for cells to be filled.
        py_random (random.Random): Random state instance.

    Returns:
        dict[tuple[int, int, int, int], int]: Dict mapping placement coords (x1, y1, x2, y2)
                                            to assigned item index.

    """
    if not cell_placements:
        return {}

    # Find the placement tuple with the largest area for primary assignment
    primary_placement = max(
        cell_placements,
        key=lambda coords: (coords[2] - coords[0]) * (coords[3] - coords[1]),
    )

    placement_to_item_index: dict[tuple[int, int, int, int], int] = {}
    placement_to_item_index[primary_placement] = 0  # Assign primary item (index 0)

    # Use list comprehension for potentially better performance
    remaining_placements = [coords for coords in cell_placements if coords != primary_placement]

    # Indices for additional/replicated items start from 1
    remaining_item_indices = list(range(1, num_items))
    py_random.shuffle(remaining_item_indices)

    num_to_assign = min(len(remaining_placements), len(remaining_item_indices))
    for i in range(num_to_assign):
        placement_to_item_index[remaining_placements[i]] = remaining_item_indices[i]

    return placement_to_item_index


def _preprocess_item_annotations(
    item: dict[str, Any],
    processor: BboxProcessor | KeypointsProcessor | None,
    data_key: Literal["bboxes", "keypoints"],
) -> np.ndarray | None:
    """Helper to preprocess annotations (bboxes or keypoints) for a single item."""
    original_data = item.get(data_key)

    # Check if processor exists and the relevant data key is in the item
    if processor and data_key in item and item.get(data_key) is not None:
        # === Add validation for required label fields ===
        required_labels = processor.params.label_fields
        if required_labels:
            missing_labels = [field for field in required_labels if field not in item]
            if missing_labels:
                raise ValueError(
                    f"Item contains '{data_key}' but is missing required label fields: {missing_labels}. "
                    f"Ensure all label fields declared in {type(processor.params).__name__} "
                    f"({required_labels}) are present in the item dictionary when '{data_key}' is present.",
                )
        # === End validation ===

        # Create a temporary minimal dict for the processor
        temp_data = {
            "image": item["image"],
            data_key: item[data_key],
        }

        # Add declared label fields if they exist in the item (already validated above)
        if required_labels:
            for field in required_labels:
                # Check again just in case validation logic changes, avoids KeyError
                if field in item:
                    temp_data[field] = item[field]

        # Preprocess modifies temp_data in-place
        processor.preprocess(temp_data)
        # Return the potentially modified data from the temp dict
        return temp_data.get(data_key)

    # Return original data if no processor or data key wasn't in item
    return original_data


def preprocess_selected_mosaic_items(
    selected_raw_items: list[dict[str, Any]],
    bbox_processor: BboxProcessor | None,  # Allow None
    keypoint_processor: KeypointsProcessor | None,  # Allow None
) -> list[ProcessedMosaicItem]:
    """Preprocesses bboxes/keypoints for selected raw additional items.

    Iterates through items, preprocesses annotations individually using processors
    (updating label encoders), and returns a list of dicts with original image/mask
    and the corresponding preprocessed bboxes/keypoints.
    """
    if not selected_raw_items:
        return []

    result_data_items: list[ProcessedMosaicItem] = []

    for item in selected_raw_items:
        processed_bboxes = _preprocess_item_annotations(item, bbox_processor, "bboxes")
        processed_keypoints = _preprocess_item_annotations(item, keypoint_processor, "keypoints")

        # Construct the final processed item dict
        processed_item_dict: ProcessedMosaicItem = {
            "image": item["image"],
            "mask": item.get("mask"),
            "bboxes": processed_bboxes,  # Already np.ndarray or None
            "keypoints": processed_keypoints,  # Already np.ndarray or None
        }
        result_data_items.append(processed_item_dict)

    return result_data_items


def get_opposite_crop_coords(
    cell_size: tuple[int, int],
    crop_size: tuple[int, int],
    cell_position: Literal["top_left", "top_right", "center", "bottom_left", "bottom_right"],
) -> tuple[int, int, int, int]:
    """Calculates crop coordinates positioned opposite to the specified cell_position.

    Given a cell of `cell_size`, this function determines the top-left (x_min, y_min)
    and bottom-right (x_max, y_max) coordinates for a crop of `crop_size`, such
    that the crop is located in the corner or center opposite to `cell_position`.

    For example, if `cell_position` is "top_left", the crop coordinates will
    correspond to the bottom-right region of the cell.

    Args:
        cell_size: The (height, width) of the cell from which to crop.
        crop_size: The (height, width) of the desired crop.
        cell_position: The reference position within the cell. The crop will be
            taken from the opposite position.

    Returns:
        tuple[int, int, int, int]: (x_min, y_min, x_max, y_max) representing the crop coordinates.

    Raises:
        ValueError: If crop_size is larger than cell_size in either dimension.

    """
    cell_h, cell_w = cell_size
    crop_h, crop_w = crop_size

    if crop_h > cell_h or crop_w > cell_w:
        raise ValueError(f"Crop size {crop_size} cannot be larger than cell size {cell_size}")

    # Determine top-left corner (x_min, y_min) based on the OPPOSITE position
    if cell_position == "top_left":  # Crop from bottom_right
        x_min = cell_w - crop_w
        y_min = cell_h - crop_h
    elif cell_position == "top_right":  # Crop from bottom_left
        x_min = 0
        y_min = cell_h - crop_h
    elif cell_position == "bottom_left":  # Crop from top_right
        x_min = cell_w - crop_w
        y_min = 0
    elif cell_position == "bottom_right":  # Crop from top_left
        x_min = 0
        y_min = 0
    elif cell_position == "center":  # Crop from center
        x_min = (cell_w - crop_w) // 2
        y_min = (cell_h - crop_h) // 2
    else:
        # Should be unreachable due to Literal type hint, but good practice
        raise ValueError(f"Invalid cell_position: {cell_position}")

    # Calculate bottom-right corner
    x_max = x_min + crop_w
    y_max = y_min + crop_h

    return x_min, y_min, x_max, y_max


def process_cell_geometry(
    cell_shape: tuple[int, int],
    item: ProcessedMosaicItem,
    target_shape: tuple[int, int],
    fill: float | tuple[float, ...],
    fill_mask: float | tuple[float, ...],
    fit_mode: Literal["cover", "contain"],
    interpolation: int,
    mask_interpolation: int,
    cell_position: Literal["top_left", "top_right", "center", "bottom_left", "bottom_right"],
) -> ProcessedMosaicItem:
    """Applies geometric transformations (padding and/or cropping) to a single mosaic item.

    Uses a Compose pipeline with PadIfNeeded and Crop to ensure the output
    matches the target cell dimensions exactly, handling both padding and cropping cases.

    Args:
        cell_shape: (tuple[int, int]): Shape of the cell.
        item: (ProcessedMosaicItem): The preprocessed mosaic item dictionary.
        target_shape: (tuple[int, int]): Target shape of the cell.
        fill: (float | tuple[float, ...]): Fill value for image padding.
        fill_mask: (float | tuple[float, ...]): Fill value for mask padding.
        fit_mode: (Literal["cover", "contain"]): Fit mode for the mosaic.
        interpolation: (int): Interpolation method for image.
        mask_interpolation: (int): Interpolation method for mask.
        cell_position: (Literal["top_left", "top_right", "center", "bottom_left", "bottom_right"]): Position
            of the cell.

    Returns: (ProcessedMosaicItem): Dictionary containing the geometrically processed image,
        mask, bboxes, and keypoints, fitting the target dimensions.

    """
    # Define the pipeline: PadIfNeeded first, then Crop
    compose_kwargs: dict[str, Any] = {"p": 1.0}
    if item.get("bboxes") is not None:
        compose_kwargs["bbox_params"] = {"format": "albumentations"}
    if item.get("keypoints") is not None:
        compose_kwargs["keypoint_params"] = {"format": "albumentations"}

    crop_coords = get_opposite_crop_coords(cell_shape, target_shape, cell_position)

    if fit_mode == "cover":
        geom_pipeline = Compose(
            [
                SmallestMaxSize(
                    max_size_hw=cell_shape,
                    interpolation=interpolation,
                    mask_interpolation=mask_interpolation,
                    p=1.0,
                ),
                Crop(
                    x_min=crop_coords[0],
                    y_min=crop_coords[1],
                    x_max=crop_coords[2],
                    y_max=crop_coords[3],
                ),
            ],
            **compose_kwargs,
        )
    elif fit_mode == "contain":
        geom_pipeline = Compose(
            [
                LongestMaxSize(
                    max_size_hw=cell_shape,
                    interpolation=interpolation,
                    mask_interpolation=mask_interpolation,
                    p=1.0,
                ),
                Crop(
                    x_min=crop_coords[0],
                    y_min=crop_coords[1],
                    x_max=crop_coords[2],
                    y_max=crop_coords[3],
                    pad_if_needed=True,
                    fill=fill,
                    fill_mask=fill_mask,
                    p=1.0,
                ),
            ],
            **compose_kwargs,
        )
    else:
        raise ValueError(f"Invalid fit_mode: {fit_mode}. Must be 'cover' or 'contain'.")

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
        kp_shift_vector = np.array([tgt_x1, tgt_y1, 0], dtype=keypoints_geom_arr.dtype)
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
    canvas_shape: tuple[int, int],
    cell_shape: tuple[int, int],
    placement_to_item_index: dict[tuple[int, int, int, int], int],
    final_items_for_grid: list[ProcessedMosaicItem],
    fill: float | tuple[float, ...],
    fill_mask: float | tuple[float, ...],
    fit_mode: Literal["cover", "contain"],
    interpolation: Literal[
        cv2.INTER_NEAREST,
        cv2.INTER_NEAREST_EXACT,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR_EXACT,
    ],
    mask_interpolation: Literal[
        cv2.INTER_NEAREST,
        cv2.INTER_NEAREST_EXACT,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR_EXACT,
    ],
) -> dict[tuple[int, int, int, int], ProcessedMosaicItem]:
    """Processes the geometry (cropping/padding) for all assigned mosaic cells.

    Iterates through assigned placements, applies geometric transforms via process_cell_geometry,
    and returns a dictionary mapping final placement coordinates to the processed item data.
    The bbox/keypoint coordinates in the returned dict are *not* shifted yet.

    Args:
        canvas_shape (tuple[int, int]): The shape of the canvas.
        cell_shape (tuple[int, int]): Shape of each cell in the mosaic grid.
        placement_to_item_index (dict[tuple[int, int, int, int], int]): Mapping from placement
            coordinates (x1, y1, x2, y2) to assigned item index.
        final_items_for_grid (list[ProcessedMosaicItem]): List of all preprocessed items available.
        fill (float | tuple[float, ...]): Fill value for image padding.
        fill_mask (float | tuple[float, ...]): Fill value for mask padding.
        fit_mode (Literal["cover", "contain"]): Fit mode for the mosaic.
        interpolation (int): Interpolation method for image.
        mask_interpolation (int): Interpolation method for mask.

    Returns:
        dict[tuple[int, int, int, int], ProcessedMosaicItem]: Dictionary mapping final placement
        coordinates (x1, y1, x2, y2) to the geometrically processed item data (image, mask, un-shifted bboxes/kps).

    """
    processed_cells_geom: dict[tuple[int, int, int, int], ProcessedMosaicItem] = {}

    # Iterate directly over placements and their assigned item indices
    for placement_coords, item_idx in placement_to_item_index.items():
        item = final_items_for_grid[item_idx]
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = placement_coords
        target_h = tgt_y2 - tgt_y1
        target_w = tgt_x2 - tgt_x1

        cell_position = get_cell_relative_position(placement_coords, canvas_shape)

        # Apply geometric processing (crop/pad)
        processed_cells_geom[placement_coords] = process_cell_geometry(
            cell_shape=cell_shape,
            item=item,
            target_shape=(target_h, target_w),
            fill=fill,
            fill_mask=fill_mask,
            fit_mode=fit_mode,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            cell_position=cell_position,
        )

    return processed_cells_geom


def get_cell_relative_position(
    placement_coords: tuple[int, int, int, int],
    target_shape: tuple[int, int],
) -> Literal["top_left", "top_right", "center", "bottom_left", "bottom_right"]:
    """Determines the position of a cell relative to the center of the target canvas.

    Compares the cell center to the canvas center and returns its quadrant
    or "center" if it lies on or very close to a central axis.

    Args:
        placement_coords (tuple[int, int, int, int]): The (x_min, y_min, x_max, y_max) coordinates
            of the cell.
        target_shape (tuple[int, int]): The (height, width) of the overall target canvas.

    Returns:
        Literal["top_left", "top_right", "center", "bottom_left", "bottom_right"]:
            The position of the cell relative to the center of the target canvas.

    """
    target_h, target_w = target_shape
    x1, y1, x2, y2 = placement_coords

    # Use floating point for potentially more accurate center calculation
    canvas_center_x = target_w / 2.0
    canvas_center_y = target_h / 2.0

    cell_center_x = (x1 + x2) / 2.0
    cell_center_y = (y1 + y2) / 2.0

    # Determine vertical position (using a small tolerance might be robust, but direct comparison first)
    if cell_center_y < canvas_center_y:
        v_pos = "top"
    elif cell_center_y > canvas_center_y:
        v_pos = "bottom"
    else:  # Exactly on the horizontal center line
        v_pos = "center"

    # Determine horizontal position
    if cell_center_x < canvas_center_x:
        h_pos = "left"
    elif cell_center_x > canvas_center_x:
        h_pos = "right"
    else:  # Exactly on the vertical center line
        h_pos = "center"

    # Combine positions
    if v_pos == "top" and h_pos == "left":
        return "top_left"
    if v_pos == "top" and h_pos == "right":
        return "top_right"
    if v_pos == "bottom" and h_pos == "left":
        return "bottom_left"
    if v_pos == "bottom" and h_pos == "right":
        return "bottom_right"

    # Default to "center" if perfectly centered or aligned with one central axis
    return "center"


def shift_all_coordinates(
    processed_cells_geom: dict[tuple[int, int, int, int], ProcessedMosaicItem],
    canvas_shape: tuple[int, int],
) -> dict[tuple[int, int, int, int], ProcessedMosaicItem]:  # Return type matches input, but values are updated
    """Shifts coordinates for all geometrically processed cells.

    Iterates through the processed cells (keyed by placement coords), applies coordinate
    shifting to bboxes/keypoints, and returns a new dictionary with the same keys
    but updated ProcessedMosaicItem values containing the *shifted* coordinates.

    Args:
        processed_cells_geom (dict[tuple[int, int, int, int], ProcessedMosaicItem]):
             Output from process_all_mosaic_geometries (keyed by placement coords).
        canvas_shape (tuple[int, int]): The shape of the canvas.

    Returns:
        dict[tuple[int, int, int, int], ProcessedMosaicItem]: Final dictionary mapping
        placement coords (x1, y1, x2, y2) to processed cell data with shifted coordinates.

    """
    final_processed_cells: dict[tuple[int, int, int, int], ProcessedMosaicItem] = {}
    canvas_h, canvas_w = canvas_shape

    for placement_coords, cell_data_geom in processed_cells_geom.items():
        tgt_x1, tgt_y1 = placement_coords[:2]

        cell_width = placement_coords[2] - placement_coords[0]
        cell_height = placement_coords[3] - placement_coords[1]

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
            bbox_denoramlized = denormalize_bboxes(bboxes_geom_arr, {"height": cell_height, "width": cell_width})
            bbox_shift_vector = np.array([tgt_x1, tgt_y1, tgt_x1, tgt_y1], dtype=np.float32)

            shifted_bboxes_denormalized = fgeometric.shift_bboxes(bbox_denoramlized, bbox_shift_vector)
            shifted_bboxes = normalize_bboxes(shifted_bboxes_denormalized, {"height": canvas_h, "width": canvas_w})
            final_cell_data["bboxes"] = shifted_bboxes
        else:
            final_cell_data["bboxes"] = np.empty((0, NUM_BBOXES_COLUMNS_IN_ALBUMENTATIONS))

        if keypoints_geom is not None and keypoints_geom.size > 0:
            keypoints_geom_arr = np.asarray(keypoints_geom)

            # Ensure shift vector matches keypoint dtype (usually float)
            kp_shift_vector = np.array([tgt_x1, tgt_y1, 0], dtype=keypoints_geom_arr.dtype)

            shifted_keypoints = fgeometric.shift_keypoints(keypoints_geom_arr, kp_shift_vector)

            final_cell_data["keypoints"] = shifted_keypoints
        else:
            final_cell_data["keypoints"] = np.empty((0, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS))

        final_processed_cells[placement_coords] = cast("ProcessedMosaicItem", final_cell_data)

    return final_processed_cells
