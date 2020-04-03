from __future__ import absolute_import

import numpy as np
import pytest

import albumentations.pcaug.functional as F


@pytest.mark.parametrize(
    ["points", "axis", "angle", "expected_points"],
    [
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            [1, 0, 0],
            np.pi / 2,
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        )
    ],
)
def test_rotate(points, angle, axis, expected_points):
    # rotation around axis x
    processed_points = F.rotate_around_axis(points, axis=axis, angle=angle)
    assert np.allclose(expected_points, processed_points)


@pytest.mark.parametrize(
    ["points", "expected_points", "min_max"],
    [
        (
            np.array([[100, 0, 0], [-100, 0, 0], [0, 0, 0], [0, 1, 100], [0, 100, 1]]),
            np.array([0, 0, 0]),
            (-10, 10, -10, 10, -10, 10),
        )
    ],
)
def test_crop(points, expected_points, min_max):
    processed_points = F.crop(
        points,
        x_min=min_max[0],
        x_max=min_max[1],
        y_min=min_max[2],
        y_max=min_max[3],
        z_min=min_max[4],
        z_max=min_max[5],
    )
    assert np.allclose(expected_points, processed_points)


@pytest.mark.parametrize(
    ["points", "expected_points"],
    [(np.array([[2, 2, 2], [0, 0, 0]], dtype=np.float), np.array([[1, 1, 1], [-1, -1, -1]], dtype=np.float))],
)
def test_center(points, expected_points):
    processed_points = F.center(points)
    assert np.allclose(expected_points, processed_points)


@pytest.mark.parametrize(
    ["points", "expected_points", "offset"],
    [
        (
            np.array([[2, 2, 2], [0, 0, 0]], dtype=np.float),
            np.array([[102, 142, 52], [100, 140, 50]], dtype=np.float),
            np.array([100, 140, 50]),
        )
    ],
)
def test_move(points, expected_points, offset):
    processed_points = F.move(points, offset=offset)
    assert np.allclose(expected_points, processed_points)


@pytest.mark.parametrize(
    ["points", "expected_points", "scale_factor"],
    [
        (
            np.array([[2, 1, 1], [0, 0, 0]], dtype=np.float),
            np.array([[4, 1, 1], [0, 0, 0]], dtype=np.float),
            np.array([2, 1, 1], dtype=np.float),
        )
    ],
)
def test_scale(points, expected_points, scale_factor):
    processed_points = F.scale(points, scale_factor)
    assert np.allclose(expected_points, processed_points)
