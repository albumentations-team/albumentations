import multiprocessing
import sys

import numpy as np
import pytest
import cv2

cv2.setRNGSeed(137)

np.random.seed(137)

@pytest.fixture
def mask():
    return cv2.randu(np.empty((100, 100), dtype=np.uint8), 0, 2)

@pytest.fixture
def image():
    return cv2.randu(np.zeros((100, 100, 3), dtype=np.uint8),
                       low=np.array([0, 0, 0]),
                       high=np.array([255, 255, 255]))


@pytest.fixture
def bboxes():
    return np.array([[15, 12, 75, 30, 1], [55, 25, 90, 90, 2]])

@pytest.fixture
def volume():
    return np.random.randint(0, 256, (10, 100, 100), dtype=np.uint8)

@pytest.fixture
def mask3d():
    return np.random.randint(0, 2, (10, 100, 100), dtype=np.uint8)


@pytest.fixture
def albumentations_bboxes():
    return np.array([[0.15, 0.12, 0.75, 0.30, 1], [0.55, 0.25, 0.90, 0.90, 2]])


@pytest.fixture
def keypoints():
    return np.array([[30, 20, 0, 0.5, 1], [20, 30, 60, 2.5, 2]], dtype=np.float32)


@pytest.fixture
def template():
    return cv2.randu(np.zeros((100, 100, 3), dtype=np.uint8), 0, 255)


@pytest.fixture
def float_template():
    return cv2.randu(np.zeros((100, 100, 3), dtype=np.float32), 0, 1)


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

SQUARE_UINT8_IMAGE = cv2.randu(np.zeros((100, 100, 3), dtype=np.uint8), 0, 255)
RECTANGULAR_UINT8_IMAGE = cv2.randu(np.zeros((101, 99, 3), dtype=np.uint8), 0, 255)

SQUARE_FLOAT_IMAGE = cv2.randu(np.zeros((100, 100, 3), dtype=np.float32), 0, 1)
RECTANGULAR_FLOAT_IMAGE = cv2.randu(np.zeros((101, 99, 3), dtype=np.float32), 0, 1)

UINT8_IMAGES = [SQUARE_UINT8_IMAGE, RECTANGULAR_UINT8_IMAGE]

FLOAT32_IMAGES = [SQUARE_FLOAT_IMAGE, RECTANGULAR_FLOAT_IMAGE]

IMAGES = UINT8_IMAGES + FLOAT32_IMAGES
VOLUME = np.random.randint(0, 255, (4, 101, 99, 3), dtype=np.uint8)

SQUARE_IMAGES = [SQUARE_UINT8_IMAGE, SQUARE_FLOAT_IMAGE]
RECTANGULAR_IMAGES = [RECTANGULAR_UINT8_IMAGE, RECTANGULAR_FLOAT_IMAGE]

SQUARE_MULTI_UINT8_IMAGE = np.random.randint(low=0, high=256, size=(100, 100, 5), dtype=np.uint8)
SQUARE_MULTI_FLOAT_IMAGE = np.random.uniform(low=0.0, high=1.0, size=(100, 100, 5)).astype(np.float32)

MULTI_IMAGES = [SQUARE_MULTI_UINT8_IMAGE, SQUARE_MULTI_FLOAT_IMAGE]
