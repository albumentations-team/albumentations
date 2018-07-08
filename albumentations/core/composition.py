from __future__ import division

import numpy as np


__all__ = ['Compose', 'OneOf', 'OneOrOther']


class Compose(object):
    def __init__(self, transforms, p=1.):
        self.transforms = [t for t in transforms if t is not None]
        self.p = p

    def __call__(self, **data):
        if np.random.random() < self.p:
            for t in self.transforms:
                data = t(**data)
        return data


class OneOf(object):
    def __init__(self, transforms, p=.5):
        self.transforms = transforms
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, **data):
        if np.random.random() < self.p:
            t = np.random.choice(self.transforms, p=self.transforms_ps)
            t.p = 1.
            data = t(**data)
        return data


class OneOrOther(object):
    def __init__(self, first, second, p=.5):
        self.first = first
        first.p = 1.
        self.second = second
        second.p = 1.
        self.p = p

    def __call__(self, **data):
        return self.first(**data) if np.random.random() < self.p else self.second(**data)
