import os
import warnings
import sys
import multiprocessing

import numpy as np
import pytest


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
def image():
    return np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mask():
    return np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)


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
def float_image():
    return np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype("float32")


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
