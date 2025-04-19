import random

import pytest

from albumentations.augmentations.mixing.functional import calculate_mosaic_center_point


@pytest.mark.parametrize(
    "grid_yx, target_size, center_range, seed, expected_center",
    [
        # Standard 2x2 grid, target 100x100 (Large Grid 200x200)
        # Safe Zone: x=[49, 149], y=[49, 149] -> size 101x101
        ((2, 2), (100, 100), (0.0, 0.0), 42, (49, 49)),  # Min offset
        ((2, 2), (100, 100), (1.0, 1.0), 42, (150, 150)), # Max offset (49 + int(101*1.0) = 150)
        ((2, 2), (100, 100), (0.5, 0.5), 42, (99, 99)),   # Mid offset (49 + int(101*0.5) = 99)

        # 1x1 grid, target 200x200 (Large Grid 200x200)
        # Safe Zone: x=[99, 99], y=[99, 99] -> size 1x1
        ((1, 1), (200, 200), (0.0, 1.0), 42, (99, 99)), # Offset always 0, center fixed
        ((1, 1), (200, 200), (0.0, 0.0), 42, (99, 99)),
        ((1, 1), (200, 200), (1.0, 1.0), 42, (100, 100)), # Offset becomes 1, center = 99+1 = 100

        # Large grid 5x5, small target 10x10 (Large Grid 50x50)
        # Safe Zone: x=[4, 44], y=[4, 44] -> size 41x41
        ((5, 5), (10, 10), (0.0, 0.0), 42, (4, 4)),      # Min offset
        ((5, 5), (10, 10), (1.0, 1.0), 42, (45, 45)),     # Max offset (4 + int(41*1.0) = 45)
        ((5, 5), (10, 10), (0.5, 0.5), 42, (24, 24)),     # Mid offset (4 + int(41*0.5) = 24)

        # Larger target relative to grid cell (causes large safe zone)
        # Grid 2x2, target 600x600 (Large Grid 1200x1200)
        # Safe Zone: x=[299, 899], y=[299, 899] -> size 601x601
        ((2, 2), (600, 600), (0.5, 0.5), 42, (599, 599)), # Mid offset (299 + int(601*0.5) = 599)
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
