"""Tests for compose operators (+, -, __radd__)."""

from __future__ import annotations

import numpy as np
import pytest

import albumentations as A
from albumentations.core.composition import BboxParams, KeypointParams


# Sample data fixtures
@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample mask for testing."""
    return np.random.randint(0, 2, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_bboxes():
    """Create sample bounding boxes in pascal_voc format."""
    return [[10, 10, 50, 50, 1], [60, 60, 90, 90, 2]]


@pytest.fixture
def sample_keypoints():
    """Create sample keypoints in xy format."""
    return [[20, 30], [70, 80]]


@pytest.fixture
def sample_bbox_labels():
    """Create sample bbox labels."""
    return [1, 2]


@pytest.fixture
def sample_keypoint_labels():
    """Create sample keypoint labels."""
    return [0, 1]


# Test data combinations
@pytest.mark.parametrize("compose_class", [A.Compose, A.Sequential, A.OneOf])
@pytest.mark.parametrize(
    "transform_sets",
    [
        # Simple case: [A, B] + C = [A, B, C]
        ([A.HorizontalFlip, A.VerticalFlip], A.Blur),

        # Different transform types
        ([A.RandomBrightnessContrast], A.HorizontalFlip),

        # Single transform
        ([A.HorizontalFlip], A.VerticalFlip),
    ]
)
def test_compose_add_single_transform_equivalence(
    compose_class,
    transform_sets,
    sample_image,
    sample_mask,
):
    """Test that compose + transform equals compose([original_transforms, added_transform])."""
    base_transform_classes, additional_transform_class = transform_sets

    # Base compose kwargs (no seed during init)
    compose_kwargs = {"p": 1.0}

    # Create fresh instances for base compose
    base_transforms = [cls(p=1.0) for cls in base_transform_classes]
    base_compose = compose_class(base_transforms, **compose_kwargs)

    # Create fresh instances for expected compose
    expected_transforms = [cls(p=1.0) for cls in base_transform_classes] + [additional_transform_class(p=1.0)]
    expected_compose = compose_class(expected_transforms, **compose_kwargs)

    # Test the operator with fresh instance
    additional_transform = additional_transform_class(p=1.0)
    result_compose = base_compose + additional_transform

    # Set the same random seed on both composes before applying
    expected_compose.set_random_seed(137)
    result_compose.set_random_seed(137)

    # Apply both compositions to the same data
    data = {"image": sample_image, "mask": sample_mask}

    expected_result = expected_compose(**data)
    actual_result = result_compose(**data)

    # Compare results
    np.testing.assert_array_equal(expected_result["image"], actual_result["image"])
    np.testing.assert_array_equal(expected_result["mask"], actual_result["mask"])


@pytest.mark.parametrize("compose_class", [A.Compose, A.Sequential])
@pytest.mark.parametrize(
    "transform_sets",
    [
        # Multiple transforms: [A, B] + [C, D] = [A, B, C, D]
        ([A.HorizontalFlip], [A.VerticalFlip, A.Blur]),

        # Empty base: [] + [A, B] = [A, B]
        ([], [A.HorizontalFlip, A.VerticalFlip]),

        # Different transform types
        ([A.RandomBrightnessContrast], [A.HorizontalFlip]),
    ]
)
def test_compose_add_multiple_transforms_equivalence(
    compose_class,
    transform_sets,
    sample_image,
    sample_mask,
):
    """Test that compose + [transforms] equals compose([original_transforms, *added_transforms])."""
    base_transform_classes, additional_transform_classes = transform_sets

    # Base compose kwargs (no seed during init)
    compose_kwargs = {"p": 1.0}

    # Create fresh instances for base compose
    base_transforms = [cls(p=1.0) for cls in base_transform_classes]
    base_compose = compose_class(base_transforms, **compose_kwargs)

    # Create fresh instances for expected compose
    expected_transforms = ([cls(p=1.0) for cls in base_transform_classes] +
                          [cls(p=1.0) for cls in additional_transform_classes])
    expected_compose = compose_class(expected_transforms, **compose_kwargs)

    # Test the operator with fresh instances
    additional_transforms = [cls(p=1.0) for cls in additional_transform_classes]
    result_compose = base_compose + additional_transforms

    # Set the same random seed on both composes before applying
    expected_compose.set_random_seed(137)
    result_compose.set_random_seed(137)

    # Apply both compositions to the same data
    data = {"image": sample_image, "mask": sample_mask}

    expected_result = expected_compose(**data)
    actual_result = result_compose(**data)

    # Compare results
    np.testing.assert_array_equal(expected_result["image"], actual_result["image"])
    np.testing.assert_array_equal(expected_result["mask"], actual_result["mask"])


@pytest.mark.parametrize("compose_class", [A.Compose, A.Sequential])
@pytest.mark.parametrize(
    "transform_sets",
    [
        # Simple case: A + [B, C] = [A, B, C]
        (A.HorizontalFlip, [A.VerticalFlip, A.Blur]),

        # Different transform types
        (A.RandomBrightnessContrast, [A.HorizontalFlip]),
    ]
)
def test_compose_radd_single_transform_equivalence(
    compose_class,
    transform_sets,
    sample_image,
    sample_mask,
):
    """Test that transform + compose equals compose([added_transform, *original_transforms])."""
    additional_transform_class, base_transform_classes = transform_sets

    # Base compose kwargs (no seed during init)
    compose_kwargs = {"p": 1.0}

    # Create fresh instances for base compose
    base_transforms = [cls(p=1.0) for cls in base_transform_classes]
    base_compose = compose_class(base_transforms, **compose_kwargs)

    # Create fresh instances for expected compose
    expected_transforms = [additional_transform_class(p=1.0)] + [cls(p=1.0) for cls in base_transform_classes]
    expected_compose = compose_class(expected_transforms, **compose_kwargs)

    # Test the operator with fresh instance
    additional_transform = additional_transform_class(p=1.0)
    result_compose = additional_transform + base_compose

    # Set the same random seed on both composes before applying
    expected_compose.set_random_seed(137)
    result_compose.set_random_seed(137)

    # Apply both compositions to the same data
    data = {"image": sample_image, "mask": sample_mask}

    expected_result = expected_compose(**data)
    actual_result = result_compose(**data)

    # Compare results
    np.testing.assert_array_equal(expected_result["image"], actual_result["image"])
    np.testing.assert_array_equal(expected_result["mask"], actual_result["mask"])


@pytest.mark.parametrize("compose_class", [A.Compose, A.Sequential])
@pytest.mark.parametrize(
    "transform_sets",
    [
        # Multiple prepends: [A, B] + [C, D] = [A, B, C, D]
        ([A.HorizontalFlip, A.VerticalFlip], [A.Blur]),

        # Empty base: [A, B] + [] = [A, B]
        ([A.HorizontalFlip, A.VerticalFlip], []),
    ]
)
def test_compose_radd_multiple_transforms_equivalence(
    compose_class,
    transform_sets,
    sample_image,
    sample_mask,
):
    """Test that [transforms] + compose equals compose([*added_transforms, *original_transforms])."""
    additional_transform_classes, base_transform_classes = transform_sets

    # Base compose kwargs (no seed during init)
    compose_kwargs = {"p": 1.0}

    # Create fresh instances for base compose
    base_transforms = [cls(p=1.0) for cls in base_transform_classes]
    base_compose = compose_class(base_transforms, **compose_kwargs)

    # Create fresh instances for expected compose
    expected_transforms = ([cls(p=1.0) for cls in additional_transform_classes] +
                          [cls(p=1.0) for cls in base_transform_classes])
    expected_compose = compose_class(expected_transforms, **compose_kwargs)

    # Test the operator with fresh instances
    additional_transforms = [cls(p=1.0) for cls in additional_transform_classes]
    result_compose = additional_transforms + base_compose

    # Set the same random seed on both composes before applying
    expected_compose.set_random_seed(137)
    result_compose.set_random_seed(137)

    # Apply both compositions to the same data
    data = {"image": sample_image, "mask": sample_mask}

    expected_result = expected_compose(**data)
    actual_result = result_compose(**data)

    # Compare results
    np.testing.assert_array_equal(expected_result["image"], actual_result["image"])
    np.testing.assert_array_equal(expected_result["mask"], actual_result["mask"])


def test_compose_subtract_transform_equivalence(sample_image, sample_mask):
    """Test that compose - transform removes the specific transform instance."""
    # Create fresh instances for full compose
    transform_a = A.HorizontalFlip(p=1.0)
    transform_b = A.VerticalFlip(p=1.0)
    transform_c = A.Blur(p=1.0)

    # Create compose with three transforms
    full_compose = A.Compose([transform_a, transform_b, transform_c], p=1.0)

    # Remove the middle transform
    reduced_compose = full_compose - transform_b

    # Expected compose without transform_b - use fresh instances
    expected_compose = A.Compose([A.HorizontalFlip(p=1.0), A.Blur(p=1.0)], p=1.0)

    # Set the same random seed on both composes before applying
    reduced_compose.set_random_seed(137)
    expected_compose.set_random_seed(137)

    # Apply both compositions to the same data
    data = {"image": sample_image, "mask": sample_mask}

    expected_result = expected_compose(**data)
    actual_result = reduced_compose(**data)

    # Compare results
    np.testing.assert_array_equal(expected_result["image"], actual_result["image"])
    np.testing.assert_array_equal(expected_result["mask"], actual_result["mask"])

    # Verify the transform was actually removed
    assert len(reduced_compose.transforms) == 2
    assert transform_b not in reduced_compose.transforms
    assert transform_a in reduced_compose.transforms
    assert transform_c in reduced_compose.transforms


def test_compose_subtract_nonexistent_transform_raises_error():
    """Test that subtracting a non-existent transform raises ValueError."""
    transform_a = A.HorizontalFlip(p=1.0)
    transform_b = A.VerticalFlip(p=1.0)
    transform_c = A.Blur(p=1.0)  # Not in the compose

    compose = A.Compose([transform_a, transform_b], p=1.0)

    with pytest.raises(ValueError, match="Transform .* not found in compose"):
        compose - transform_c


@pytest.mark.parametrize(
    "bbox_params,keypoint_params",
    [
        (None, None),
        (BboxParams(format="pascal_voc", label_fields=["bbox_labels"]), None),
        (None, KeypointParams(format="xy", label_fields=["keypoint_labels"])),
        (
            BboxParams(format="pascal_voc", label_fields=["bbox_labels"]),
            KeypointParams(format="xy", label_fields=["keypoint_labels"])
        ),
    ]
)
def test_compose_operators_preserve_params(
    bbox_params,
    keypoint_params,
    sample_image,
    sample_mask,
    sample_bboxes,
    sample_keypoints,
    sample_bbox_labels,
    sample_keypoint_labels,
):
    """Test that bbox_params and keypoint_params are preserved in operator results."""
    base_compose = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        p=1.0,
    )

    additional_transform = A.VerticalFlip(p=1.0)
    result_compose = base_compose + additional_transform

    # Verify params are preserved
    if bbox_params:
        assert result_compose.processors.get("bboxes") is not None
        assert result_compose.processors["bboxes"].params.format == bbox_params.format
    else:
        assert result_compose.processors.get("bboxes") is None

    if keypoint_params:
        assert result_compose.processors.get("keypoints") is not None
        assert result_compose.processors["keypoints"].params.format == keypoint_params.format
    else:
        assert result_compose.processors.get("keypoints") is None

    # Test functionality with data
    data = {"image": sample_image, "mask": sample_mask}

    if bbox_params:
        data["bboxes"] = sample_bboxes
        data["bbox_labels"] = sample_bbox_labels

    if keypoint_params:
        data["keypoints"] = sample_keypoints
        data["keypoint_labels"] = sample_keypoint_labels

    # Should not raise any errors
    result = result_compose(**data)

    # Check that outputs have expected keys
    assert "image" in result
    assert "mask" in result

    if bbox_params:
        assert "bboxes" in result
        assert "bbox_labels" in result

    if keypoint_params:
        assert "keypoints" in result
        assert "keypoint_labels" in result


def test_compose_operators_preserve_additional_targets(sample_image):
    """Test that additional_targets are preserved in operator results."""
    additional_targets = {"image2": "image", "mask2": "mask"}

    base_compose = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        additional_targets=additional_targets,
        p=1.0,
    )

    result_compose = base_compose + A.VerticalFlip(p=1.0)

    # Verify additional_targets are preserved
    assert result_compose.additional_targets == additional_targets

    # Test functionality
    mask2 = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    data = {
        "image": sample_image,
        "image2": sample_image.copy(),
        "mask2": mask2,
    }

    result = result_compose(**data)

    assert "image" in result
    assert "image2" in result
    assert "mask2" in result


def test_compose_operators_preserve_other_params(sample_image, sample_mask):
    """Test that other Compose parameters (strict, is_check_shapes, etc.) are preserved."""
    base_compose = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        p=0.8,
        strict=True,
        is_check_shapes=False,
    )

    result_compose = base_compose + A.VerticalFlip(p=1.0)

    # Verify other params are preserved
    assert result_compose.p == 0.8
    assert result_compose.strict == True
    assert result_compose.is_check_shapes == False


def test_compose_operators_immutability(sample_image, sample_mask):
    """Test that operators return new instances and don't modify originals."""
    transform_a = A.HorizontalFlip(p=1.0)
    transform_b = A.VerticalFlip(p=1.0)
    transform_c = A.Blur(p=1.0)

    original_compose = A.Compose([transform_a, transform_b], p=1.0)
    original_length = len(original_compose.transforms)

    # Test addition doesn't modify original
    new_compose = original_compose + transform_c
    assert len(original_compose.transforms) == original_length
    assert len(new_compose.transforms) == original_length + 1
    assert new_compose is not original_compose

    # Test prepending doesn't modify original
    prepended_compose = transform_c + original_compose
    assert len(original_compose.transforms) == original_length
    assert len(prepended_compose.transforms) == original_length + 1
    assert prepended_compose is not original_compose

    # Test subtraction doesn't modify original
    reduced_compose = original_compose - transform_a
    assert len(original_compose.transforms) == original_length
    assert len(reduced_compose.transforms) == original_length - 1
    assert reduced_compose is not original_compose
    assert transform_a in original_compose.transforms


@pytest.mark.parametrize("compose_class", [A.OneOf])
def test_special_compose_classes_add_operations(compose_class, sample_image, sample_mask):
    """Test that special compose classes work with operators."""
    # No seed during initialization
    compose_kwargs = {"p": 1.0}

    # Create fresh instances for base compose
    base_transforms = [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]
    base_compose = compose_class(base_transforms, **compose_kwargs)

    # Create fresh instances for expected compose
    expected_transforms = [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Blur(p=1.0)]
    expected_compose = compose_class(expected_transforms, **compose_kwargs)

    # Use fresh instance for operator
    additional_transform = A.Blur(p=1.0)
    result_compose = base_compose + additional_transform

    # Check that the result has correct number of transforms
    assert len(result_compose.transforms) == len(expected_compose.transforms)
    assert type(result_compose) == type(expected_compose)


def test_selective_channel_transform_operators(sample_image):
    """Test operators work with SelectiveChannelTransform."""
    base_compose = A.SelectiveChannelTransform(
        [A.HorizontalFlip(p=1.0)],
        channels=[0, 1],
        p=1.0
    )

    result_compose = base_compose + A.VerticalFlip(p=1.0)

    # Check that channels parameter is preserved
    assert result_compose.channels == [0, 1]
    assert len(result_compose.transforms) == 2

    # Test functionality
    data = {"image": sample_image}
    result = result_compose(**data)
    assert "image" in result


def test_some_of_operators_behavior(sample_image, sample_mask):
    """Test SomeOf operator behavior - n parameter should be preserved, not incremented."""
    # SomeOf([A, B], n=1) + C should equal SomeOf([A, B, C], n=1)
    # Both select 1 transform, but from different pools

    base_transforms = [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]
    base_compose = A.SomeOf(base_transforms, n=1, p=1.0)

    additional_transform = A.Blur(p=1.0)
    result_compose = base_compose + additional_transform

    # Check that n is preserved
    assert result_compose.n == 1  # Should stay 1, not become 2
    assert len(result_compose.transforms) == 3  # But pool size increases

    # Check that the transforms are correctly added
    transform_classes = [t.__class__ for t in result_compose.transforms]
    expected_classes = [A.HorizontalFlip, A.VerticalFlip, A.Blur]
    assert transform_classes == expected_classes

    # Test that it works functionally (both should select 1 transform from their pools)
    data = {"image": sample_image, "mask": sample_mask}

    # Set same seed and apply multiple times to see different selections
    result_compose.set_random_seed(137)
    result1 = result_compose(**data)

    result_compose.set_random_seed(138)
    result2 = result_compose(**data)

    # Should work without errors
    assert "image" in result1 and "image" in result2
    assert "mask" in result1 and "mask" in result2


@pytest.mark.parametrize("compose_class", [A.Compose, A.Sequential])
def test_compose_operators_associativity(compose_class, sample_image, sample_mask):
    """Test functional associativity: (A + B) + C produces same results as A + (B + C)."""

    # Create fresh transform instances for each path
    transform_a1 = A.HorizontalFlip(p=1.0)
    transform_b1 = A.VerticalFlip(p=1.0)
    transform_c1 = A.Blur(p=1.0)

    transform_a2 = A.HorizontalFlip(p=1.0)
    transform_b2 = A.VerticalFlip(p=1.0)
    transform_c2 = A.Blur(p=1.0)

    # Path 1: (A + B) + C  -> [A, B, C] (flat)
    ab_compose = compose_class([transform_a1], p=1.0) + transform_b1
    abc_left = ab_compose + transform_c1

    # Path 2: A + (B + C)  -> [A, compose(B, C)] (nested)
    bc_compose = compose_class([transform_b2], p=1.0) + transform_c2
    abc_right = compose_class([transform_a2], p=1.0) + bc_compose

    # Structure is different (flat vs nested), but function should be equivalent
    assert len(abc_left.transforms) == 3  # Flat: [A, B, C]
    assert len(abc_right.transforms) == 2  # Nested: [A, compose(B, C)]

    # Apply with same seed - should produce identical results despite different structure
    abc_left.set_random_seed(137)
    abc_right.set_random_seed(137)

    data = {"image": sample_image, "mask": sample_mask}

    result_left = abc_left(**data)
    result_right = abc_right(**data)

    # Results should be functionally identical
    np.testing.assert_array_equal(result_left["image"], result_right["image"])
    np.testing.assert_array_equal(result_left["mask"], result_right["mask"])


@pytest.mark.parametrize("compose_class", [A.Compose, A.Sequential])
def test_compose_operators_mixed_associativity(compose_class, sample_image, sample_mask):
    """Test mixed associativity with transforms and lists."""

    # Create fresh instances
    transform_a1 = A.HorizontalFlip(p=1.0)
    transform_b1 = A.VerticalFlip(p=1.0)
    transform_c1 = A.Blur(p=1.0)

    transform_a2 = A.HorizontalFlip(p=1.0)
    transform_b2 = A.VerticalFlip(p=1.0)
    transform_c2 = A.Blur(p=1.0)

    # Path 1: (A + [B]) + C
    a_compose = compose_class([transform_a1], p=1.0)
    ab_compose = a_compose + [transform_b1]
    abc_left = ab_compose + transform_c1

    # Path 2: A + ([B] + C) - this creates: A + compose([B, C])
    c_compose = compose_class([transform_c2], p=1.0)
    bc_compose = [transform_b2] + c_compose  # [B] + compose([C]) -> compose([B, compose([C])])
    abc_right = compose_class([transform_a2], p=1.0) + bc_compose

    # Different structures but should produce same results
    abc_left.set_random_seed(137)
    abc_right.set_random_seed(137)

    data = {"image": sample_image, "mask": sample_mask}

    result_left = abc_left(**data)
    result_right = abc_right(**data)

    # Results should be functionally identical
    np.testing.assert_array_equal(result_left["image"], result_right["image"])
    np.testing.assert_array_equal(result_left["mask"], result_right["mask"])
