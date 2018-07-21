import numpy as np
import pytest


@pytest.fixture
def image():
    return np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mask():
    return np.random.randint(low=0, high=2, size=(100, 100), dtype=np.uint8)


@pytest.fixture
def float_image():
    return np.random.uniform(low=0.0, high=1.0, size=(100, 100, 3)).astype('float32')
