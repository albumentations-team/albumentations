import random as py_random
from typing import Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import DTypeLike

from .core.types import FloatNumType, IntNumType, NumType, SizeType


def get_random_state() -> np.random.RandomState:
    return np.random.RandomState(py_random.randint(0, (1 << 32) - 1))


def uniform(
    low: NumType = 0.0,
    high: NumType = 1.0,
    size: Optional[Union[SizeType, int]] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> FloatNumType:
    if random_state is None:
        random_state = get_random_state()
    return random_state.uniform(low, high, size)


def rand(
    d0: NumType, d1: NumType, *more: Any, random_state: Optional[np.random.RandomState] = None, **kwargs: Any
) -> np.ndarray:
    if random_state is None:
        random_state = get_random_state()
    return random_state.rand(d0, d1, *more, **kwargs)


def randn(
    d0: NumType, d1: NumType, *more: Any, random_state: Optional[np.random.RandomState] = None, **kwargs: Any
) -> np.ndarray:
    if random_state is None:
        random_state = get_random_state()
    return random_state.randn(d0, d1, *more, **kwargs)


def normal(
    loc: NumType = 0.0,
    scale: NumType = 1.0,
    size: Optional[SizeType] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> FloatNumType:
    if random_state is None:
        random_state = get_random_state()
    return random_state.normal(loc, scale, size)


def poisson(
    lam: NumType = 1.0, size: Optional[SizeType] = None, random_state: Optional[np.random.RandomState] = None
) -> IntNumType:
    if random_state is None:
        random_state = get_random_state()
    return random_state.poisson(lam, size)


def permutation(
    x: Union[int, Sequence[float], np.ndarray], random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    if random_state is None:
        random_state = get_random_state()
    return random_state.permutation(x)


def randint(
    low: IntNumType,
    high: Optional[IntNumType] = None,
    size: Optional[Union[SizeType, int]] = None,
    dtype: DTypeLike = np.int32,
    random_state: Optional[np.random.RandomState] = None,
) -> IntNumType:
    if random_state is None:
        random_state = get_random_state()
    return random_state.randint(low, high, size, dtype)


def random(size: Optional[NumType] = None, random_state: Optional[np.random.RandomState] = None) -> FloatNumType:
    if random_state is None:
        random_state = get_random_state()
    return random_state.random(size)


def choice(
    a: NumType,
    size: Optional[Union[SizeType, int]] = None,
    replace: bool = True,
    p: Optional[Union[Sequence[float], np.ndarray]] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    if random_state is None:
        random_state = get_random_state()
    return random_state.choice(a, size, replace, p)
