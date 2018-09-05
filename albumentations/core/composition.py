from __future__ import division

import random

import numpy as np


__all__ = ['Compose', 'OneOf', 'OneOrOther']


class Compose(object):
    """Composes transforms together"""

    def __init__(self, transforms, p=1.0):
        self.transforms = [t for t in transforms if t is not None]
        self.p = p

    def __call__(self, **data):
        if random.random() < self.p:
            for t in self.transforms:
                data = t(**data)
        return data


class OneOf(object):
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, **data):
        if random.random() < self.p:
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            t.p = 1.
            data = t(**data)
        return data


class OneOrOther(object):
    def __init__(self, first, second, p=0.5):
        self.first = first
        first.p = 1.
        self.second = second
        second.p = 1.
        self.p = p

    def __call__(self, **data):
        return self.first(**data) if random.random() < self.p else self.second(**data)
