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
        [random_utils.beta, [0.3, 0.4]]
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


def test_shuffle_multiprocessing(mp_pool):
    set_seed(0)
    original_array = np.arange(1000)
    arrays_to_shuffle = [np.copy(original_array) for _ in range(10)]  # Create copies to shuffle independently

    # Shuffle arrays using multiprocessing
    shuffled_arrays = mp_pool.map(_calc_shuffle, [(random_utils.shuffle, arr) for arr in arrays_to_shuffle])

    # Check that each shuffled array is not the same as the original (highly likely but not absolutely guaranteed)
    # and that they are permutations of the original (i.e., contain the same elements)
    for shuffled_array in shuffled_arrays:
        assert not np.array_equal(shuffled_array, original_array), "Array was not shuffled"
        assert np.array_equal(np.sort(shuffled_array), original_array), "Shuffled array elements differ from original"
