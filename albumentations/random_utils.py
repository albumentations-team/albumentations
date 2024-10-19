from __future__ import annotations

import random as py_random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from .core.types import FloatNumType, IntNumType, NumType


def get_random_seed() -> int:
    return py_random.randint(0, (1 << 32) - 1)


def get_random_generator(random_seed: int | None = None) -> np.random.Generator:
    if random_seed is None:
        random_seed = get_random_seed()
    return np.random.default_rng(random_seed)


def uniform(
    low: NumType = 0.0,
    high: NumType = 1.0,
    size: tuple[int, ...] | int | None = None,
    random_generator: np.random.Generator | None = None,
) -> FloatNumType:
    if random_generator is None:
        random_generator = get_random_generator()
    return random_generator.uniform(low, high, size)


def beta(
    alpha: NumType = 0.5,
    beta: NumType = 0.5,
    random_generator: np.random.Generator | None = None,
) -> FloatNumType:
    if random_generator is None:
        random_generator = get_random_generator()
    return random_generator.beta(alpha, beta)


def rand(
    d0: NumType,
    d1: NumType,
    *more: Any,
    random_generator: np.random.Generator | None = None,
    **kwargs: Any,
) -> np.ndarray:
    if random_generator is None:
        random_generator = get_random_generator()
    return random_generator.random((d0, d1, *more), **kwargs)


def randn(
    d0: NumType,
    d1: NumType,
    *more: Any,
    random_generator: np.random.Generator | None = None,
    **kwargs: Any,
) -> np.ndarray:
    if random_generator is None:
        random_generator = get_random_generator()
    return random_generator.standard_normal((d0, d1, *more), **kwargs)


def normal(
    loc: NumType = 0.0,
    scale: NumType = 1.0,
    size: tuple[int, ...] | int | None = None,
    random_generator: np.random.Generator | None = None,
) -> FloatNumType:
    if random_generator is None:
        random_generator = get_random_generator()
    return random_generator.normal(loc, scale, size)


def poisson(
    lam: NumType = 1.0,
    size: tuple[int, ...] | int | None = None,
    random_generator: np.random.Generator | None = None,
) -> IntNumType:
    if random_generator is None:
        random_generator = get_random_generator()
    return random_generator.poisson(lam, size)


def permutation(
    x: int | Sequence[float] | np.ndarray,
    random_generator: np.random.Generator | None = None,
) -> np.ndarray:
    if random_generator is None:
        random_generator = get_random_generator()
    return random_generator.permutation(x)


def randint(
    low: IntNumType,
    high: IntNumType | None = None,
    size: tuple[int, ...] | int | None = None,
    dtype: DTypeLike = np.int32,
    random_generator: np.random.Generator | None = None,
) -> IntNumType:
    if random_generator is None:
        random_generator = get_random_generator()
    return random_generator.integers(low, high, size, dtype=dtype)


def random(size: NumType | None = None, random_generator: np.random.Generator | None = None) -> FloatNumType:
    if random_generator is None:
        random_generator = get_random_generator()
    return random_generator.random(size)


def choice(
    a: NumType,
    size: tuple[int, int] | int | None = None,
    replace: bool = True,
    p: Sequence[float] | np.ndarray | None = None,
    random_generator: np.random.Generator | None = None,
) -> np.ndarray:
    if random_generator is None:
        random_generator = get_random_generator()
    return random_generator.choice(a, size, replace, p)


def shuffle(
    a: np.ndarray,
    random_generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Shuffles an array in-place, using a specified random generator or creating a new one if not provided.

    Args:
        a (np.ndarray): The array to be shuffled.
        random_generator (Optional[np.random.Generator], optional): The random generator used for shuffling.
        Defaults to None.

    Returns:
        np.ndarray: The shuffled array (note: the shuffle is in-place, so the original array is modified).
    """
    if random_generator is None:
        random_generator = get_random_generator()
    random_generator.shuffle(a)
    return a
