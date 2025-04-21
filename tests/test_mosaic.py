import random
import numpy as np
import pytest

from albumentations.augmentations.mixing.transforms import Mosaic
from albumentations.core.composition import Compose
from albumentations.core.bbox_utils import BboxParams, check_bboxes, filter_bboxes as filter_bboxes_helper
from albumentations.core.keypoints_utils import KeypointParams, check_keypoints
from albumentations.augmentations.mixing import functional as Fmixing
from albumentations.augmentations.mixing.functional import (
    calculate_mosaic_center_point,
    calculate_cell_placements,
    process_cell_geometry,
    assemble_mosaic_from_processed_cells,
    ProcessedMosaicItem,
)


@pytest.mark.parametrize(
    "img_shape, target_size",
    [
        ((100, 80, 3), (100, 80)),  # Standard RGB
        ((64, 64, 1), (64, 64)),  # Grayscale
        ((128, 50), (128, 50)),   # Grayscale without channel dim
    ],
)
def test_mosaic_identity_single_image(img_shape: tuple[int, ...], target_size: tuple[int, int]) -> None:
    """Check Mosaic returns the original image when metadata is empty and target_size matches."""
    if len(img_shape) == 2:
        img = np.random.randint(0, 256, size=img_shape, dtype=np.uint8)
    else:
        img = np.random.randint(0, 256, size=img_shape, dtype=np.uint8)

    transform = Mosaic(target_size=target_size, grid_yx=(1, 1), p=1.0)

    # Input data structure expects a list for metadata
    data = {"image": img, "mosaic_metadata": []}

    result = transform(**data)
    transformed_img = result["image"]

    assert transformed_img.shape == img.shape
    np.testing.assert_array_equal(transformed_img, img)


# Separate parametrize for shapes, sizes, and fill values
@pytest.mark.parametrize(
    "img_shape, target_size, fill, fill_mask",
    [
        # Matching sizes
        ((100, 80, 3), (100, 80), 128, 1), # RGB
        ((64, 64, 1), (64, 64), 50, 2),   # Grayscale
        ((128, 50), (128, 50), 0, 3),     # Grayscale 2D
        # Target smaller (cropping)
        ((100, 100, 3), (80, 80), 100, 4),
        # Target larger (padding)
        ((50, 50, 1), (70, 70), 200, 5),
        ((80, 60), (100, 100), 30, 6), # Grayscale 2D padding
    ],
)
# Separate parametrize for grid dimensions
@pytest.mark.parametrize(
    "grid_yx",
    [
        (1, 1),
        (2, 2),
        (1, 2),
        (3, 2),
        (1, 3),
    ],
)
def test_mosaic_identity_monochromatic(
    img_shape: tuple[int, ...],
    target_size: tuple[int, int],
    grid_yx: tuple[int, int],
    fill: int,
    fill_mask: int,
) -> None:
    """Check Mosaic returns a uniform image/mask if input is uniform (no metadata)."""
    # --- Image Setup ---
    if len(img_shape) == 2:
        img = np.full(img_shape, fill_value=fill, dtype=np.uint8)
        expected_output_shape_img = (*target_size,)
    else:
        img = np.full(img_shape, fill_value=fill, dtype=np.uint8)
        expected_output_shape_img = (*target_size, img_shape[-1])

    # --- Mask Setup ---
    mask_shape = img_shape[:2]
    mask = np.full(mask_shape, fill_value=fill_mask, dtype=np.uint8)
    expected_output_shape_mask = target_size

    # --- Transform --- (Use 0 for padding values to test persistence)
    transform = Mosaic(
        target_size=target_size,
        grid_yx=grid_yx,
        p=1.0,
        fill=0,
        fill_mask=0
    )

    # --- Apply ---
    data = {"image": img, "mask": mask, "mosaic_metadata": []}
    result = transform(**data)
    transformed_img = result["image"]
    transformed_mask = result["mask"]

    # --- Assertions (Image) ---
    assert transformed_img.shape == expected_output_shape_img
    assert transformed_img.dtype == img.dtype

    is_padded_h = target_size[0] > img_shape[0]
    is_padded_w = target_size[1] > img_shape[1]

    if not is_padded_h and not is_padded_w:
        expected_output_img = np.full(expected_output_shape_img, fill_value=fill, dtype=np.uint8)
        np.testing.assert_array_equal(transformed_img, expected_output_img)
    else:
        assert np.all((transformed_img == fill) | (transformed_img == 0))
        orig_h, orig_w = img_shape[:2]
        assert np.all(transformed_img[:orig_h, :orig_w] == fill)

    # --- Assertions (Mask) ---
    assert transformed_mask.shape == expected_output_shape_mask
    assert transformed_mask.dtype == mask.dtype

    if not is_padded_h and not is_padded_w:
        expected_output_mask = np.full(expected_output_shape_mask, fill_value=fill_mask, dtype=np.uint8)
        np.testing.assert_array_equal(transformed_mask, expected_output_mask)
    else:
        assert np.all((transformed_mask == fill_mask) | (transformed_mask == 0))
        orig_h, orig_w = mask_shape[:2]
        assert np.all(transformed_mask[:orig_h, :orig_w] == fill_mask)


def test_mosaic_identity_with_targets() -> None:
    """Check Mosaic returns original image, mask, and bboxes when grid is (1, 1) and no metadata."""
    img_size = (8, 6)
    img = np.random.randint(0, 256, size=(*img_size, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, size=img_size, dtype=np.uint8)
    # Bbox in albumentations format [x_min, y_min, x_max, y_max, class_id]
    bboxes = np.array([
        [0.2, 0.3, 0.8, 0.7, 1],
        [0.1, 0.1, 0.5, 0.5, 2],
        [0.6, 0.2, 0.9, 0.4, 0]
    ], dtype=np.float32)

    transform = Mosaic(target_size=img_size, grid_yx=(1, 1), p=1.0)

    # Use Compose to handle bbox processing
    pipeline = Compose([
        transform
    ], bbox_params=BboxParams(format='albumentations', label_fields=['class_labels']))

    data = {
        "image": img.copy(),
        "mask": mask.copy(),
        "bboxes": bboxes.copy()[:, :4], # Pass only coords
        "class_labels": bboxes[:, 4].tolist(), # Pass labels separately
        "mosaic_metadata": []
    }

    result = pipeline(**data)

    # Check image
    assert result["image"].shape == img.shape
    np.testing.assert_array_equal(result["image"], img)

    # Check mask
    assert result["mask"].shape == mask.shape
    np.testing.assert_array_equal(result["mask"], mask)

    # Check bboxes (coords + label)
    expected_bboxes_coords = np.array([
        [0.2, 0.3, 0.8, 0.7],
        [0.1, 0.1, 0.5, 0.5],
        [0.6, 0.2, 0.9, 0.4]
    ], dtype=np.float32)
    expected_labels = [1.0, 2.0, 0.0]

    assert "bboxes" in result
    assert len(result["bboxes"]) == 3
    # Check coordinates only
    np.testing.assert_allclose(result["bboxes"], expected_bboxes_coords, atol=1e-6)

    assert "class_labels" in result # Check label field is returned
    # Check labels separately
    assert result["class_labels"] == expected_labels


@pytest.mark.parametrize(
    "grid_yx",
    [
        (1, 2),
        (2, 1),
    ],
)
@pytest.mark.parametrize("center_range", [(0.4, 0.4), (0.6, 0.6)])
def test_mosaic_deterministic_with_metadata(
    grid_yx: tuple[int, int],
    center_range: tuple[tuple[float, float], tuple[float, float]], # Add center_range to signature
) -> None:
    """Test Mosaic with metadata, fixed center, and multiple targets including labels."""
    target_size = (100, 100)

    # --- Calculate Expected Placements using functional helpers ---
    rng = random.Random(0)
    center_xy = Fmixing.calculate_mosaic_center_point(grid_yx, target_size, center_range, rng)

    cell_placements = Fmixing.calculate_cell_placements(grid_yx, target_size, center_xy)

    # Determine which cell gets the primary image (largest area)
    # cell_placements is a list of tuples: [(y1, x1, y2, x2), ...]
    primary_placement = max(
        cell_placements,
        key=lambda coords: (coords[2] - coords[0]) * (coords[3] - coords[1]) # coords is (y1, x1, y2, x2)
    )

    # Filter out the primary placement to find the metadata placement
    remaining_placements = [coords for coords in cell_placements if coords != primary_placement]
    if not remaining_placements:
        raise ValueError("Expected at least one remaining cell placement for metadata.")
    meta_placement = remaining_placements[0] # Assume only one remaining for this test setup


    py1, px1, py2, px2 = primary_placement
    my1, mx1, my2, mx2 = meta_placement
    place_h_meta, place_w_meta = my2 - my1, mx2 - mx1

    # --- Primary Data ---
    img_primary = np.ones((*target_size, 3), dtype=np.uint8) * 1
    mask_primary = np.ones(target_size, dtype=np.uint8) * 1
    # pascal_voc format [x_min, y_min, x_max, y_max] (absolute pixels)
    bboxes_primary = np.array([[10, 10, 30, 30], [50, 50, 70, 70]], dtype=np.float32)
    labels_primary = ['cat', 10]
    # xy format [x, y] (absolute pixels)
    keypoints_primary = np.array([[20, 20], [60, 60]], dtype=np.float32)
    kp_labels_primary = ['eye', 5]

    # --- Metadata ---
    meta_size = (80, 120) # height, width
    img_meta = np.ones((*meta_size, 3), dtype=np.uint8) * 2
    mask_meta = np.ones(meta_size, dtype=np.uint8) * 2
    bboxes_meta = np.array([[5, 5, 25, 25], [40, 60, 80, 75]], dtype=np.float32) # pascal_voc
    labels_meta = ['cat', 'fish']
    keypoints_meta = np.array([[15, 15], [60, 70]], dtype=np.float32) # xy
    kp_labels_meta = ['nose', 6]

    metadata_list = [
        {
            "image": img_meta,
            "mask": mask_meta,
            "bboxes": bboxes_meta,
            "class_labels": labels_meta,
            "keypoints": keypoints_meta,
            "kp_labels": kp_labels_meta,
        }
    ]

    # --- Transform ---
    transform = Mosaic(
        target_size=target_size,
        grid_yx=grid_yx,
        center_range=center_range,
        p=1.0,
        fill=0,
        fill_mask=0,
    )

    pipeline = Compose([
        transform
    ],
    bbox_params=BboxParams(format='pascal_voc', label_fields=['class_labels']),
    keypoint_params=KeypointParams(format='xy', label_fields=['kp_labels']))

    # --- Input Data ---
    data = {
        "image": img_primary,
        "mask": mask_primary,
        "bboxes": bboxes_primary,
        "class_labels": labels_primary,
        "keypoints": keypoints_primary,
        "kp_labels": kp_labels_primary,
        "mosaic_metadata": metadata_list,
    }

    # --- Apply ---
    result = pipeline(**data)

    # --- Calculate Expected ---
    # Use functional components to build expected image/mask

    # 1. Prepare input items in ProcessedMosaicItem format (bboxes/kps are not needed for image/mask calculation)
    primary_item_input: ProcessedMosaicItem = {
        "image": img_primary,
        "mask": mask_primary,
        "bboxes": None, # Not needed for geom
        "keypoints": None, # Not needed for geom
    }
    meta_item_input: ProcessedMosaicItem = {
        "image": img_meta,
        "mask": mask_meta,
        "bboxes": None,
        "keypoints": None,
    }

    # 2. Process geometry for each item based on its placement dimensions
    primary_h, primary_w = py2 - py1, px2 - px1
    processed_primary_geom = process_cell_geometry(
        item=primary_item_input,
        target_h=primary_h,
        target_w=primary_w,
        fill=0, # Match transform fill
        fill_mask=0, # Match transform fill_mask
    )
    meta_h, meta_w = my2 - my1, mx2 - mx1
    processed_meta_geom = process_cell_geometry(
        item=meta_item_input,
        target_h=meta_h,
        target_w=meta_w,
        fill=0,
        fill_mask=0,
    )

    # 3. Create the dictionary mapping placements to processed geometric data
    processed_cells_for_assembly = {
        primary_placement: processed_primary_geom,
        meta_placement: processed_meta_geom,
    }

    # 4. Assemble the expected image and mask
    expected_image = assemble_mosaic_from_processed_cells(
        processed_cells=processed_cells_for_assembly,
        target_shape=(*target_size, 3),
        dtype=np.uint8,
        data_key="image",
    )
    expected_mask = assemble_mosaic_from_processed_cells(
        processed_cells=processed_cells_for_assembly,
        target_shape=target_size,
        dtype=np.uint8,
        data_key="mask",
    )

    # --- Expected BBoxes/Keypoints (Absolute Pixels in Original Formats) ---

    # Primary Targets (Shifted)
    exp_bboxes_primary = bboxes_primary.copy()
    exp_bboxes_primary[:, [0, 2]] += px1 # Shift x_min, x_max
    exp_bboxes_primary[:, [1, 3]] += py1 # Shift y_min, y_max

    exp_kps_primary = keypoints_primary.copy()
    exp_kps_primary[:, 0] += px1 # Shift x
    exp_kps_primary[:, 1] += py1 # Shift y

    # Metadata Targets (Cropped/Filtered + Shifted in absolute pixels)
    # Simulate the effect of placing the meta_image into the meta_placement cell

    # Determine the crop size applied *implicitly* to the metadata image
    crop_h = place_h_meta
    crop_w = place_w_meta

    # -- Metadata BBoxes (pascal_voc format) --
    bboxes_meta_filtered = []
    labels_meta_filtered = []
    for i, bbox in enumerate(bboxes_meta):
        x_min, y_min, x_max, y_max = bbox
        label = labels_meta[i]

        # Check for overlap with the implicit crop area (0, 0, crop_w, crop_h)
        if x_min < crop_w and x_max > 0 and y_min < crop_h and y_max > 0:
            # Clip coordinates to the crop area
            x_min_clipped = max(0, x_min)
            y_min_clipped = max(0, y_min)
            x_max_clipped = min(crop_w, x_max)
            y_max_clipped = min(crop_h, y_max)

            # Check if area is still valid after clipping
            if x_max_clipped > x_min_clipped and y_max_clipped > y_min_clipped:
                 # Shift the clipped coordinates to the final canvas position
                 shifted_bbox = [
                     x_min_clipped + mx1,
                     y_min_clipped + my1,
                     x_max_clipped + mx1,
                     y_max_clipped + my1,
                 ]
                 bboxes_meta_filtered.append(shifted_bbox)
                 labels_meta_filtered.append(label)

    exp_bboxes_meta = np.array(bboxes_meta_filtered, dtype=np.float32)

    # -- Metadata Keypoints (xy format) --
    keypoints_meta_filtered = []
    kp_labels_meta_filtered = []
    for i, kp in enumerate(keypoints_meta):
        x, y = kp
        label = kp_labels_meta[i]

        # Check if keypoint is within the implicit crop area
        if 0 <= x < crop_w and 0 <= y < crop_h:
            # Shift the keypoint to the final canvas position
            shifted_kp = [x + mx1, y + my1]
            keypoints_meta_filtered.append(shifted_kp)
            kp_labels_meta_filtered.append(label)

    exp_kps_meta = np.array(keypoints_meta_filtered, dtype=np.float32)


    # Combine expected results
    expected_bboxes = np.vstack((exp_bboxes_primary, exp_bboxes_meta)) if exp_bboxes_meta.size > 0 else exp_bboxes_primary
    expected_labels = labels_primary + labels_meta_filtered

    if exp_kps_meta.size > 0:
       expected_keypoints = np.vstack((exp_kps_primary, exp_kps_meta))
       expected_kp_labels = kp_labels_primary + kp_labels_meta_filtered
    else:
       expected_keypoints = exp_kps_primary
       expected_kp_labels = kp_labels_primary


    # --- Assertions ---
    assert result['image'].shape == (*target_size, 3)
    np.testing.assert_array_equal(result['image'], expected_image)

    assert result['mask'].shape == target_size
    np.testing.assert_array_equal(result['mask'], expected_mask)

    assert 'bboxes' in result
    # Check if result bboxes are empty before comparing length/values
    if not expected_bboxes.size:
        assert not result['bboxes'].size, "Expected empty bboxes but got some."
    else:
        assert result['bboxes'].shape[0] == expected_bboxes.shape[0], f"Expected {expected_bboxes.shape[0]} bboxes, got {result['bboxes'].shape[0]}"
        np.testing.assert_allclose(result['bboxes'], expected_bboxes, atol=1) # Use tolerance for pixel coords
        assert result['class_labels'] == expected_labels

    assert 'keypoints' in result
    if not expected_keypoints.size:
       assert not result['keypoints'].size, "Expected empty keypoints but got some."
    else:
       assert result['keypoints'].shape[0] == expected_keypoints.shape[0], f"Expected {expected_keypoints.shape[0]} keypoints, got {result['keypoints'].shape[0]}"
       np.testing.assert_allclose(result['keypoints'], expected_keypoints, atol=1) # Use tolerance for pixel coords
       assert result['kp_labels'] == expected_kp_labels
