import numpy as np
import pytest

from albumentations import random_utils
from .utils import set_seed


def _calc(args):
    return args[0](*args[1])

@pytest.mark.parametrize(
    ["func", "args"],
    [
        [random_utils.uniform, [-(1 << 15), 1 << 15, 100]],
        [random_utils.rand, [10, 10]],
        [random_utils.randn, [10, 10]],
        [random_utils.normal, [0, 1, 100]],
        [random_utils.poisson, [1 << 15, 100]],
        [random_utils.permutation, [np.arange(1000)]],
        [random_utils.randint, [-(1 << 15), 1 << 15, 100]],
        [random_utils.random, [100]],
        [random_utils.choice, [np.arange(1000), 100]],
        [random_utils.beta, [0.3, 0.4]],
        [random_utils.shuffle, [np.arange(1000)]]
    ],
)
def test_multiprocessing(func, args, mp_pool):
    set_seed(0)

    n = 4
    status = False
    for _ in range(n):
        res = mp_pool.map(_calc, [(func, args), (func, args)])
        status = not np.allclose(res[0], res[1])
        if status:
            break
    assert status


@pytest.mark.parametrize("array, random_state", [
    (np.array([1, 2, 3, 4, 5]), np.random.RandomState(42)),
    (np.array([10, 20, 30]), np.random.RandomState(99)),
    (np.array([1.5, 2.5, 3.5]), np.random.RandomState(43)),  # Float array
])
def test_shuffle_effectiveness(array, random_state):
    shuffled_array = random_utils.shuffle(array, random_state)
    assert len(array) == len(shuffled_array), "Shuffled array length differs"

@pytest.mark.parametrize("array", [
    (np.array([1, 2, 3, 4, 5])),
    (np.array([10, 20, 30])),
])
def test_inplace_shuffle_check(array):
    original_array = array.copy()
    random_utils.shuffle(array)
    assert not np.array_equal(original_array, array), "Shuffle should be in-place, but original array was not modified"
