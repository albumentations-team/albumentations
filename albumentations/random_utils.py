# Use `Any` as the return type to avoid mypy problems with Union data types,
# because numpy can return single number and ndarray

import random as py_random
from typing import Any, Optional, Sequence, Type, Union

import numpy as np

from .core.transforms_interface import NumType

IntNumType = Union[int, np.ndarray]
Size = Union[int, Sequence[int]]


def get_random_state() -> np.random.RandomState:
    return np.random.RandomState(py_random.randint(0, (1 << 32) - 1))


def uniform(
    low: NumType = 0.0,
    high: NumType = 1.0,
    size: Optional[Size] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.uniform(low, high, size)


def rand(d0: NumType, d1: NumType, *more, random_state: Optional[np.random.RandomState] = None, **kwargs) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.rand(d0, d1, *more, **kwargs)  # type: ignore


def randn(d0: NumType, d1: NumType, *more, random_state: Optional[np.random.RandomState] = None, **kwargs) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.randn(d0, d1, *more, **kwargs)  # type: ignore


def normal(
    loc: NumType = 0.0,
    scale: NumType = 1.0,
    size: Optional[Size] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.normal(loc, scale, size)


def poisson(
    lam: NumType = 1.0, size: Optional[Size] = None, random_state: Optional[np.random.RandomState] = None
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.poisson(lam, size)


def permutation(
    x: Union[int, Sequence[float], np.ndarray], random_state: Optional[np.random.RandomState] = None
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.permutation(x)


def randint(
    low: IntNumType,
    high: Optional[IntNumType] = None,
    size: Optional[Size] = None,
    dtype: Type = np.int32,
    random_state: Optional[np.random.RandomState] = None,
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.randint(low, high, size, dtype)


def random(size: Optional[NumType] = None, random_state: Optional[np.random.RandomState] = None) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.random(size)  # type: ignore


def choice(
    a: NumType,
    size: Optional[Size] = None,
    replace: bool = True,
    p: Optional[Union[Sequence[float], np.ndarray]] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.choice(a, size, replace, p)  # type: ignore
