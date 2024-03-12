import numpy as np
import pytest

from albumentations import random_utils
from .utils import set_seed


def _calc(args):
    return args[0](*args[1])

def _calc_shuffle(args):
    func, arr = args
    func(arr)  # Perform the shuffle in-place
    return arr  # Return the shuffled array for verification

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

    n = 10
    status = False
    for _ in range(n):
        res = mp_pool.map(_calc, [(func, args), (func, args)])
        status = not np.allclose(res[0], res[1])
        if status:
            break
    assert status
