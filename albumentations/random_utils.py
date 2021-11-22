import random as py_random
from typing import Optional, Sequence, TypeVar, Union

import numpy as np

NumType = Union[float, np.ndarray]
IntNumType = Union[int, np.ndarray]
Size = Union[int, Sequence[int]]
T = TypeVar("T")


def get_random_state() -> np.random.RandomState:
    return np.random.RandomState(py_random.randint(0, (1 << 32) - 1))


def uniform(
    low: NumType = 0.0,
    high: NumType = 1.0,
    size: Optional[Size] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> NumType:
    if random_state is None:
        random_state = get_random_state()
    return random_state.uniform(low, high, size)


def rand(d0: NumType, d1: NumType, *more, random_state: Optional[np.random.RandomState] = None, **kwargs) -> NumType:
    if random_state is None:
        random_state = get_random_state()
    return random_state.randn(d0, d1, *more, **kwargs)


def normal(
    loc: NumType = 0.0,
    scale: NumType = 1.0,
    size: Optional[Size] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> NumType:
    if random_state is None:
        random_state = get_random_state()
    return random_state.normal(loc, scale, size)


def poisson(
    lam: NumType = 1.0, size: Optional[Size] = None, random_state: Optional[np.random.RandomState] = None
) -> NumType:
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
    size: Optional[Size] = None,
    dtype: np.dtype = np.int,
    random_state: Optional[np.random.RandomState] = None,
) -> IntNumType:
    if random_state is None:
        random_state = get_random_state()
    return random_state.randint(low, high, size, dtype)


def random(size: Optional[NumType] = None, random_state: Optional[np.random.RandomState] = None) -> NumType:
    if random_state is None:
        random_state = get_random_state()
    return random_state.random(size)


def choice(
    a: Sequence[T],
    size: Optional[Size] = None,
    replace: bool = True,
    p: Optional[Union[Sequence[float], np.ndarray]] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Union[T, Sequence[T]]:
    if random_state is None:
        random_state = get_random_state()
    return random_state.choice(a, size, replace, p)
