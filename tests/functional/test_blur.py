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
