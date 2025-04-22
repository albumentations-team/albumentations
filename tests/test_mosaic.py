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


def test_mosaic_simplified_deterministic() -> None:
    """Test Mosaic with fixed parameters, albumentations format, no labels."""
    target_size = (100, 100)
    grid_yx = (1, 2)
    center_range = (0.5, 0.5)
    meta_size = (100, 100) # H, W

    # --- Primary Data ---
    img_primary = np.ones((*target_size, 3), dtype=np.uint8) * 1
    mask_primary = np.ones(target_size, dtype=np.uint8) * 11
    # BBoxes: Albumentations format [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
    bboxes_primary = np.array([[0, 0, 1, 1]], dtype=np.float32)
    # Keypoints: Albumentations format [x, y, Z, angle, scale]
    keypoints_primary = np.array([[0, 0, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=np.float32)

    # --- Metadata ---
    img_meta = np.ones((*meta_size, 3), dtype=np.uint8) * 2
    mask_meta = np.ones(meta_size, dtype=np.uint8) * 22
    bboxes_meta = np.array([[0, 0, 1, 1]], dtype=np.float32) # rel to meta_size
    keypoints_meta = np.array([[0, 0, 0, 0, 0], [99, 99, 0, 0, 0]], dtype=np.float32) # rel to meta_size

    metadata_list = [
        {
            "image": img_meta,
            "mask": mask_meta,
            "bboxes": bboxes_meta,
            "keypoints": keypoints_meta,
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
    # Use albumentations format, no label fields
    bbox_params=BboxParams(format='albumentations', min_visibility=0.0, min_area=0.0),
    keypoint_params=KeypointParams(format='albumentations')) # No labels, use correct format

    # --- Input Data ---
    data = {
        "image": img_primary,
        "mask": mask_primary,
        "bboxes": bboxes_primary,
        "keypoints": keypoints_primary,
        "mosaic_metadata": metadata_list,
    }

    # --- Apply ---
    result = pipeline(**data)

    # --- Calculate Expected Image/Mask (Simplified) ---
    # Primary fills left (0,0,50,100), Meta fills right (50,0,100,100)
    expected_image = np.zeros((*target_size, 3), dtype=np.uint8)
    expected_mask = np.zeros(target_size, dtype=np.uint8)

    # Primary part (cropped from img_primary)
    expected_image[0:100, 0:50] = img_primary[:, :50]
    expected_mask[0:100, 0:50] = mask_primary[:, :50]

    # Meta part (padded then cropped from img_meta)
    padded_h, padded_w = 100, 120
    img_meta_padded = np.full((padded_h, padded_w, 3), 0, dtype=np.uint8)
    mask_meta_padded = np.full((padded_h, padded_w), 0, dtype=np.uint8)
    img_meta_padded[:meta_size[0], :meta_size[1]] = img_meta
    mask_meta_padded[:meta_size[0], :meta_size[1]] = mask_meta
    expected_image[0:100, 50:100] = img_meta_padded[:100, :50]
    expected_mask[0:100, 50:100] = mask_meta_padded[:100, :50]


    # --- Calculate Expected Annotations (derived manually above) ---
    expected_bboxes = np.array([[0. , 0. , 0.5, 1. ], [0.5, 0. , 1. , 1.]], dtype=np.float32)
    expected_keypoints = np.array([[ 0.,  0., 0, 0, 0], [ 1.,  1., 0, 0, 0], [50.,  0., 0, 0, 0]], dtype=np.float32)

    # --- Assertions ---
    assert result['image'].shape == (*target_size, 3)
    np.testing.assert_array_equal(result['image'], expected_image)

    assert result['mask'].shape == target_size
    np.testing.assert_array_equal(result['mask'], expected_mask)

    assert 'bboxes' in result
    assert result['bboxes'].shape[0] == expected_bboxes.shape[0]
    np.testing.assert_allclose(result['bboxes'], expected_bboxes, atol=1e-6)
    assert result['bboxes'].shape[1] == 4 # No label column

    assert 'keypoints' in result
    assert result['keypoints'].shape[0] == expected_keypoints.shape[0]
    np.testing.assert_allclose(result['keypoints'], expected_keypoints, atol=1e-6)
    assert result['keypoints'].shape[1] == 5 # x, y, Z, angle, scale
