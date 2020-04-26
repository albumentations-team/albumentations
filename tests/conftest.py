import sys
import warnings
import multiprocessing

import numpy as np
import pytest


try:
    import torch  # skipcq: PYL-W0611
    import torchvision  # skipcq: PYL-W0611

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
def points():
    return np.random.random((100, 3))


@pytest.fixture
def features():
    return np.random.random((100, 4))


@pytest.fixture
def labels():
    return np.random.random((100))


@pytest.fixture
def cameras():
    return [np.random.random((3, 3)), np.random.random((3, 3))]


@pytest.fixture
def normals():
    return np.random.random((100, 3))


@pytest.fixture
def bboxes():
    return [np.random.random((7)), np.random.random((7))]


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
