from random import Random
import pytest
from albumentations.augmentations.blur import functional as fblur
import numpy as np

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


def create_pil_kernel(radius):
    """Helper function to extract PIL's Gaussian kernel for comparison"""
    size = int(radius * 3.5) * 2 + 1
    kernel = []
    sigma2 = radius * radius
    scale = 1.0 / (radius * np.sqrt(2.0 * np.pi))

    for i in range(-size//2 + 1, size//2 + 1):
        x = i * 1.0
        kernel.append(scale * np.exp(-(x * x) / (2.0 * sigma2)))

    # Normalize 1D kernel
    kernel = np.array(kernel) / np.sum(kernel)
    # Create 2D kernel
    return kernel[:, np.newaxis] @ kernel[np.newaxis, :]

@pytest.mark.parametrize(
    ["sigma", "ksize", "expected_shape"],
    [
        (0.5, 0, (3, 3)),     # Small sigma
        (1.0, 0, (7, 7)),     # Medium sigma
        (2.0, 0, (15, 15)),   # Large sigma
        (3.0, 0, (21, 21)),   # Very large sigma
        (1.0, 5, (5, 5)),     # Fixed kernel size
        (2.0, 7, (7, 7)),     # Different fixed kernel size
    ]
)
def test_kernel_shapes(sigma, ksize, expected_shape):
    kernel = fblur.create_gaussian_kernel(sigma, ksize)
    assert kernel.shape == expected_shape

@pytest.mark.parametrize(
    ["sigma", "ksize", "expected_shape"],
    [
        (0.5, 0, (3,)),     # Small sigma
        (1.0, 0, (7,)),     # Medium sigma
        (2.0, 0, (15,)),   # Large sigma
        (3.0, 0, (21,)),   # Very large sigma
        (1.0, 5, (5,)),     # Fixed kernel size
        (2.0, 7, (7,)),     # Different fixed kernel size
    ]
)
def test_1d_kernel_shapes(sigma, ksize, expected_shape):
    kernel = fblur.create_gaussian_kernel_1d(sigma, ksize)
    assert kernel.shape == expected_shape

@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.0])
def test_kernel_normalization(sigma):
    """Test that kernel sums to 1 to preserve luminance"""
    kernel = fblur.create_gaussian_kernel(sigma, 0)
    kernel_1d = fblur.create_gaussian_kernel_1d(sigma, 0)

    np.testing.assert_allclose(kernel.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(kernel_1d.sum(), 1.0, atol=1e-6)

@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.0])
def test_kernel_symmetry(sigma):
    """Test that kernel is symmetric both horizontally and vertically"""
    kernel = fblur.create_gaussian_kernel(sigma, 0)
    np.testing.assert_allclose(kernel, kernel.T)  # Vertical symmetry
    np.testing.assert_allclose(kernel, np.flip(kernel))  # Horizontal symmetry

@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.0])
def test_matches_pil_kernel(sigma):
    """Test that our kernel matches PIL's kernel"""
    our_kernel = fblur.create_gaussian_kernel(sigma, 0)
    pil_kernel = create_pil_kernel(sigma)
    np.testing.assert_allclose(our_kernel, pil_kernel, rtol=1e-5)

@pytest.mark.parametrize(
    ["sigma", "ksize", "expected_max_value"],
    [
        (0.5, 0, 0.619),  # Small sigma has highest peak
        (2.0, 0, 0.040),  # Larger sigma has lower peak
        (3.0, 0, 0.018),  # Even larger sigma has even lower peak
    ]
)
def test_kernel_peak_values(sigma, ksize, expected_max_value):
    """Test that kernel peak values decrease with increasing sigma"""
    kernel = fblur.create_gaussian_kernel(sigma, ksize)

    assert np.abs(kernel.max() - expected_max_value) < 0.01

@pytest.mark.parametrize(
    ["sigma", "ksize", "expected_max_value"],
    [
        (0.5, 0, 0.786),  # Small sigma has highest peak
        (2.0, 0, 0.199),  # Larger sigma has lower peak
        (3.0, 0, 0.133),  # Even larger sigma has even lower peak
    ]
)
def test_1d_kernel_peak_values(sigma, ksize, expected_max_value):
    """Test that kernel peak values decrease with increasing sigma"""
    kernel = fblur.create_gaussian_kernel_1d(sigma, ksize)

    assert np.abs(kernel.max() - expected_max_value) < 0.01

def test_kernel_visual_comparison():
    """Visual test to compare kernels (useful for debugging)"""
    sigma = 2.0
    our_kernel = fblur.create_gaussian_kernel(sigma, 0)
    pil_kernel = create_pil_kernel(sigma)

    np.testing.assert_allclose(our_kernel, pil_kernel, rtol=1e-5)


# === Motion Kernel Tests ===

@pytest.mark.parametrize(
    "kernel_size, expected_shape",
    [
        (3, (3, 3)),
        (5, (5, 5)),
        (7, (7, 7)),
        (9, (9, 9)),
    ]
)
def test_create_motion_kernel_shape(kernel_size, expected_shape):
    """Test that motion kernel has the correct shape."""
    random_state = Random(42)
    kernel = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,
        direction=0.0,
        allow_shifted=False,
        random_state=random_state
    )
    assert kernel.shape == expected_shape


@pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
def test_create_motion_kernel_normalization(kernel_size):
    """Test that motion kernel has equal weight distribution along motion line."""
    random_state = Random(42)
    kernel = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,
        direction=0.0,
        allow_shifted=False,
        random_state=random_state
    )
    # For symmetric horizontal motion, the kernel sum should equal the kernel size
    # (each pixel along the horizontal line gets weight 1.0)
    np.testing.assert_allclose(kernel.sum(), kernel_size, atol=1e-6)


@pytest.mark.parametrize(
    "direction, expected_behavior",
    [
        (-1.0, "backward"),   # Should have more weight toward start of line
        (-0.5, "backward"),   # Moderate backward bias
        (0.0, "symmetric"),   # Should be symmetric
        (0.5, "forward"),     # Moderate forward bias
        (1.0, "forward"),     # Should have more weight toward end of line
    ]
)
def test_create_motion_kernel_direction_bias(direction, expected_behavior):
    """Test that direction parameter controls blur direction correctly."""
    random_state = Random(42)
    kernel_size = 7
    kernel = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,  # Horizontal line for easier testing
        direction=direction,
        allow_shifted=False,
        random_state=random_state
    )

    # For horizontal motion, check the middle row
    middle_row = kernel_size // 2
    row = kernel[middle_row, :]

    # Find center and calculate weights
    center_col = kernel_size // 2
    left_weight = np.sum(row[:center_col])
    right_weight = np.sum(row[center_col + 1:])

    if expected_behavior == "backward":
        assert left_weight >= right_weight, f"direction={direction} should have more weight on the left"
        assert left_weight > 0, f"direction={direction} should have some weight on the left"
    elif expected_behavior == "symmetric":
        assert abs(left_weight - right_weight) < 1e-6, f"direction={direction} should be symmetric"
        assert left_weight > 0 and right_weight > 0, f"direction={direction} should have weight on both sides"
    elif expected_behavior == "forward":
        assert right_weight >= left_weight, f"direction={direction} should have more weight on the right"
        assert right_weight > 0, f"direction={direction} should have some weight on the right"


@pytest.mark.parametrize(
    "angle, expected_orientation",
    [
        (0.0, "horizontal"),
        (90.0, "vertical"),
        (45.0, "diagonal"),
        (135.0, "diagonal"),
        (180.0, "horizontal"),
        (270.0, "vertical"),
    ]
)
def test_create_motion_kernel_angles(angle, expected_orientation):
    """Test that angle parameter controls motion direction correctly."""
    random_state = Random(42)
    kernel_size = 7
    kernel = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=angle,
        direction=0.0,  # Symmetric for easier testing
        allow_shifted=False,
        random_state=random_state
    )

    # Check that kernel has non-zero values
    assert kernel.sum() > 0, f"Kernel should have non-zero values for angle={angle}"

    # For horizontal angles (0, 180), expect motion in middle row
    if expected_orientation == "horizontal":
        middle_row = kernel_size // 2
        row_sum = np.sum(kernel[middle_row, :])
        assert np.isclose(row_sum, kernel_size, atol=1.5) or row_sum > kernel_size * 0.8, (
            f"Horizontal motion should be concentrated in middle row for angle={angle} "
            f"(row_sum={row_sum}, expected≈{kernel_size})"
        )

    # For vertical angles (90, 270), expect motion in middle column
    elif expected_orientation == "vertical":
        middle_col = kernel_size // 2
        col_sum = np.sum(kernel[:, middle_col])
        assert np.isclose(col_sum, kernel_size, atol=1.5) or col_sum > kernel_size * 0.8, (
            f"Vertical motion should be concentrated in middle column for angle={angle} "
            f"(col_sum={col_sum}, expected≈{kernel_size})"
        )


@pytest.mark.parametrize("allow_shifted", [True, False])
def test_create_motion_kernel_allow_shifted(allow_shifted):
    """Test that allow_shifted parameter works correctly."""
    random_state = Random(42)
    kernel_size = 7

    # Generate multiple kernels to test shifting behavior
    kernels = []
    for _ in range(10):
        kernel = fblur.create_motion_kernel(
            kernel_size=kernel_size,
            angle=0.0,
            direction=0.0,
            allow_shifted=allow_shifted,
            random_state=random_state
        )
        kernels.append(kernel)

    if not allow_shifted:
        # All kernels should be identical when shifting is disabled
        for i in range(1, len(kernels)):
            np.testing.assert_array_equal(kernels[0], kernels[i],
                                        "Kernels should be identical when allow_shifted=False")
    # Note: With shifting enabled, kernels might be different
    # (This is harder to test deterministically due to randomness)


@pytest.mark.parametrize(
    "direction1, direction2",
    [
        (-1.0, 0.0),
        (-1.0, 1.0),
        (0.0, 1.0),
        (-0.5, 0.5),
    ]
)
def test_create_motion_kernel_different_directions_produce_different_kernels(direction1, direction2):
    """Test that different direction values produce different kernels."""
    random_state = Random(42)
    kernel_size = 5

    kernel1 = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,
        direction=direction1,
        allow_shifted=False,
        random_state=random_state
    )

    # Reset random state to ensure same other parameters
    random_state = Random(42)
    kernel2 = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,
        direction=direction2,
        allow_shifted=False,
        random_state=random_state
    )

    assert not np.array_equal(kernel1, kernel2), \
        f"direction={direction1} and direction={direction2} should produce different kernels"


def test_create_motion_kernel_extreme_directions():
    """Test motion kernel with extreme direction values."""
    random_state = Random(42)
    kernel_size = 7

    # Test direction = -1 (fully backward)
    kernel_backward = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,
        direction=-1.0,
        allow_shifted=False,
        random_state=random_state
    )

    # Test direction = 1 (fully forward)
    random_state = Random(42)  # Reset for consistency
    kernel_forward = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,
        direction=1.0,
        allow_shifted=False,
        random_state=random_state
    )

    # For horizontal motion, check middle row
    middle_row = kernel_size // 2
    center_col = kernel_size // 2

    # Backward should have no weight strictly to the right of center
    backward_row = kernel_backward[middle_row, :]
    assert np.sum(backward_row[center_col+1:]) == 0, "Fully backward motion should have no weight strictly to the right of center"
    assert np.sum(backward_row[:center_col+1]) > 0, "Fully backward motion should have weight on left side and center"

    # Forward should have no weight strictly to the left of center
    forward_row = kernel_forward[middle_row, :]
    assert np.sum(forward_row[:center_col]) == 0, "Fully forward motion should have no weight strictly to the left of center"
    assert np.sum(forward_row[center_col:]) > 0, "Fully forward motion should have weight on center and right side"


def test_create_motion_kernel_center_pixel_behavior():
    """Test that center pixel is handled correctly for symmetric motion."""
    random_state = Random(42)
    kernel_size = 5  # Use odd size for clear center
    kernel = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,
        direction=0.0,  # Symmetric
        allow_shifted=False,
        random_state=random_state
    )

    # For symmetric horizontal motion, center pixel should have weight
    center = kernel_size // 2
    assert kernel[center, center] > 0, "Center pixel should have weight for symmetric motion"


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_create_motion_kernel_non_empty(kernel_size):
    """Test that kernel always has at least one non-zero pixel."""
    random_state = Random(42)

    # Test with extreme parameters that might create empty kernels
    kernel = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,
        direction=0.0,
        allow_shifted=False,
        random_state=random_state
    )

    assert kernel.sum() > 0, "Kernel should never be completely empty"
    assert np.any(kernel > 0), "Kernel should have at least one non-zero pixel"


@pytest.mark.parametrize(
    "direction_input, expected_clamped",
    [
        (-2.0, -1.0),  # Should be clamped to -1.0
        (-1.5, -1.0),  # Should be clamped to -1.0
        (1.5, 1.0),    # Should be clamped to 1.0
        (2.0, 1.0),    # Should be clamped to 1.0
        (0.5, 0.5),    # Should remain unchanged
        (-0.5, -0.5),  # Should remain unchanged
    ]
)
def test_create_motion_kernel_direction_validation(direction_input, expected_clamped):
    """Test that direction values are properly clamped to [-1, 1] range."""
    random_state = Random(42)
    kernel_size = 5

    # Create kernel with extreme direction value
    kernel = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,
        direction=direction_input,
        allow_shifted=False,
        random_state=random_state
    )

    # Create reference kernel with expected clamped value
    random_state = Random(42)  # Reset for consistency
    reference_kernel = fblur.create_motion_kernel(
        kernel_size=kernel_size,
        angle=0.0,
        direction=expected_clamped,
        allow_shifted=False,
        random_state=random_state
    )

    # Kernels should be identical (direction was clamped to valid range)
    np.testing.assert_array_equal(kernel, reference_kernel,
                                  f"direction={direction_input} should be clamped to {expected_clamped}")
