import random


def to_tuple(param, low=None):
    if isinstance(param, tuple):
        return param
    else:
        return (-param if low is None else low, param)


class BasicTransform:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, **kwargs):
        if random.random() < self.p:
            params = self.get_params()
            return {k: self.apply(a, **params) if k in self.targets else a for k, a in kwargs.items()}
        return kwargs

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError


class DualTransform(BasicTransform):
    """
    transfrom for segmentation task
    """

    @property
    def targets(self):
        return 'image', 'mask'


class ImageOnlyTransform(BasicTransform):
    """
    transforms applied to image only
    """

    @property
    def targets(self):
        return 'image'
