import multiprocessing
import sys
import warnings

import numpy as np
import pytest

from albumentations.core.transforms_interface import (
    BBoxesInternalType,
    KeypointsInternalType,
)

try:
    import torch  # skipcq: PYL-W0611
    import torchvision  # skipcq: PYL-W0611

    torch_available = True
except ImportError:
    torch_available = False


try:
    import imgaug

    imgaug_available = True
except ImportError:
    imgaug_available = False


skipif_imgaug = pytest.mark.skipif(imgaug_available, reason="The test was skipped because imgaug is installed")
skipif_no_imgaug = pytest.mark.skipif(
    not imgaug_available, reason="The test was skipped because imgaug is not installed"
)
skipif_no_torch = pytest.mark.skipif(
    not torch_available, reason="The test was skipped because PyTorch and torchvision are not installed"
)


def pytest_ignore_collect(path):
    if not torch_available and path.fnmatch("test_pytorch.py"):
        warnings.warn(
            UserWarning(
                "Tests that require PyTorch and torchvision were skipped because those libraries are not installed."
            )
        )
        return True

    if not imgaug_available and path.fnmatch("test_imgaug.py"):
        warnings.warn(UserWarning("Tests that require imgaug were skipped because this library is not installed."))
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
    bboxes = BBoxesInternalType(
        array=np.array([[15, 12, 75, 30], [55, 25, 90, 90]]),
        targets=[(1,), (2,)],
    )
    return bboxes


@pytest.fixture
def albumentations_bboxes():
    bboxes = BBoxesInternalType(
        array=np.array([[0.15, 0.12, 0.75, 0.30], [0.55, 0.25, 0.90, 0.90]]), targets=[(1,), (2,)]
    )
    return bboxes


@pytest.fixture
def keypoints():
    kps = KeypointsInternalType(array=np.array([[20, 30, 40, 50], [20, 30, 60, 80]]), targets=[(1,), (2,)])
    return kps


@pytest.fixture
def float_image():
    return np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype("float32")


@pytest.fixture
def template():
    return np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)


@pytest.fixture
def float_template():
    return np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype("float32")


@pytest.fixture(scope="package")
def mp_pool():
    # Usage of `fork` as a start method for multiprocessing could lead to deadlocks on macOS.
    # Because `fork` was the default start method for macOS until Python 3.8
    # we had to manually set the start method to `spawn` to avoid those issues.
    if sys.platform == "darwin":
        method = "spawn"
    else:
        method = None
    return multiprocessing.get_context(method).Pool(8)
