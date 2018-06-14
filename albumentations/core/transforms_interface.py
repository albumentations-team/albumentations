import numpy as np


__all__ = ['to_tuple', 'BasicTransform', 'DualTransform', 'ImageOnlyTransform']


def to_tuple(param, low=None):
    if isinstance(param, tuple):
        return param
    else:
        return (-param if low is None else low, param)


class BasicTransform:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, **kwargs):
        if np.random.random() < self.p:
            params = self.get_params()
            params = self.update_params(params, **kwargs)
            return {key: self.targets.get(key, lambda x, **p: x)(arg, **params) for key, arg in kwargs.items()}
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

    def update_params(self, params, **kwargs):
        params.update({'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]})
        return params


class DualTransform(BasicTransform):
    """
    transform for segmentation task
    """

    @property
    def targets(self):
        return {'image': self.apply, 'mask': self.apply, 'bboxes': self.apply_to_bbox}

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError


class ImageOnlyTransform(BasicTransform):
    """
    transforms applied to image only
    """

    @property
    def targets(self):
        return {'image': self.apply}

