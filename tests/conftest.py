import multiprocessing
import sys

import numpy as np
import pytest

@pytest.fixture
def global_label():
    return np.array([1, 0, 0])

@pytest.fixture
def mask():
    return np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)

@pytest.fixture
def bboxes():
    return np.array([[15, 12, 75, 30, 1], [55, 25, 90, 90, 2]])


@pytest.fixture
def albumentations_bboxes():
    return np.array([[0.15, 0.12, 0.75, 0.30, 1], [0.55, 0.25, 0.90, 0.90, 2]])


@pytest.fixture
def keypoints():
    return np.array([[30, 20, 0, 50, 1], [20, 30, 60, 80, 2]], dtype=np.float32)

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
    return multiprocessing.get_context(method).Pool(4)


SQUARE_UINT8_IMAGE = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
RECTANGULAR_UINT8_IMAGE = np.random.randint(low=0, high=256, size=(101, 99, 3), dtype=np.uint8)

SQUARE_FLOAT_IMAGE = np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype(np.float32)
RECTANGULAR_FLOAT_IMAGE = np.random.uniform(low=0.0, high=1.0, size=(101, 99, 3)).astype(np.float32)

UINT8_IMAGES = [SQUARE_UINT8_IMAGE, RECTANGULAR_UINT8_IMAGE]

FLOAT32_IMAGES = [SQUARE_FLOAT_IMAGE, RECTANGULAR_FLOAT_IMAGE]

IMAGES = UINT8_IMAGES + FLOAT32_IMAGES

SQUARE_IMAGES = [SQUARE_UINT8_IMAGE, SQUARE_FLOAT_IMAGE]
RECTANGULAR_IMAGES = [RECTANGULAR_UINT8_IMAGE, RECTANGULAR_FLOAT_IMAGE]

SQUARE_MULTI_UINT8_IMAGE = np.random.randint(low=0, high=256, size=(100, 100, 5), dtype=np.uint8)
SQUARE_MULTI_FLOAT_IMAGE = np.random.uniform(low=0.0, high=1.0, size=(100, 100, 5)).astype(np.float32)
