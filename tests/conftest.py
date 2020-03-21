import multiprocessing
import sys
import warnings

import numpy as np
import pytest
from hypothesis.extra.numpy import arrays as h_array
from hypothesis.strategies import composite
from hypothesis.strategies import floats as h_float
from hypothesis.strategies import integers as h_int

try:
    import torch
    import torchvision

    torch_available = True
except ImportError:
    torch_available = False


def pytest_ignore_collect(path):
    if not torch_available and path.fnmatch("test_pytorch.py"):
        warnings.warn(
            UserWarning(
                "Tests that require PyTorch and torchvision were skipped because those libraries are not installed."
            )
        )
        return True
    return False


@pytest.fixture
def bboxes():
    return [[15, 12, 75, 30, 1], [55, 25, 90, 90, 2]]


@pytest.fixture
def albumentations_bboxes():
    return [[0.15, 0.12, 0.75, 0.30, 1], [0.55, 0.25, 0.90, 0.90, 2]]


@pytest.fixture
def keypoints():
    return [[20, 30, 40, 50, 1], [20, 30, 60, 80, 2]]


@pytest.fixture
def multiprocessing_context():
    # Usage of `fork` as a start method for multiprocessing could lead to deadlocks on macOS.
    # Because `fork` was the default start method for macOS until Python 3.8
    # we had to manually set the start method to `spawn` to avoid those issues.
    if sys.platform == "darwin":
        method = "spawn"
    else:
        method = None
    return multiprocessing.get_context(method)


@composite
def image(draw, width=100, height=100, num_channels=3, dtype=np.uint8):
    return draw(
        h_array(
            dtype=dtype,
            shape=(height, width, num_channels),
            elements=h_int(min_value=0, max_value=np.iinfo(dtype).max - 1),
        )
    )


@composite
def mask(draw, width=100, height=100, dtype=np.uint8):
    return draw(
        h_array(dtype=dtype, shape=(height, width), elements=h_int(min_value=0, max_value=np.iinfo(dtype).max - 1))
    )


@composite
def float_image(draw, width=100, height=100, num_channels=3, dtype=np.float32):
    return draw(
        h_array(
            dtype=dtype,
            shape=(height, width, num_channels),
            elements=h_float(min_value=0, allow_nan=False, max_value=1, width=32),
        )
    )
