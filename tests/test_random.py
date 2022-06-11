import random
from multiprocessing.pool import Pool

import numpy as np
import pytest

from albumentations import random_utils


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
    ],
)
def test_multiprocessing(func, args):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    n = 10
    status = False
    for _ in range(n):
        with Pool(2) as pool:
            res = pool.map(_calc, [(func, args), (func, args)])
        status = not np.allclose(res[0], res[1])
        if status:
            break
    assert status
