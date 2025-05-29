import numpy as np
import pytest
from albucore.utils import get_max_value

from albumentations.augmentations import FromFloat, ToFloat


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float32", "float64"])
@pytest.mark.parametrize(
    "param, shape",
    [
        ("image", (8, 7, 6)),
        ("image", (8, 7)),
        ("images", (4, 8, 7, 6)),
        ("images", (4, 8, 7)),
    ]
)
def test_to_float(param, shape, dtype):
    rng = np.random.default_rng()
    data = rng.uniform(0, 10, size=shape).astype(dtype)

    aug = ToFloat()
    result = aug(**{param: data})[param]

    assert result.dtype == np.float32
    assert result.shape == data.shape
    np.testing.assert_allclose(data, result * get_max_value(np.dtype(dtype)))


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize(
    "param, shape",
    [
        ("image", (8, 7, 6)),
        ("image", (8, 7)),
        ("images", (4, 8, 7, 6)),
        ("images", (4, 8, 7)),
    ]
)
def test_from_float(param, shape, dtype):
    rng = np.random.default_rng()
    data = rng.random(size=shape, dtype=np.float32)

    aug = FromFloat(dtype=dtype)
    result = aug(**{param: data})[param]

    assert result.dtype == np.dtype(dtype)
    assert result.shape == data.shape
    # Because FromFloat has to round to the nearest integer, we get an absolute difference up to 0.5
    np.testing.assert_allclose(data * get_max_value(np.dtype(dtype)), result, atol=0.5)
