import random

import cv2

__all__ = ['to_tuple', 'BasicTransform', 'DualTransform', 'ImageOnlyTransform', 'NoOp']


def to_tuple(param, low=None):
    if isinstance(param, (list, tuple)):
        return tuple(param)
    elif param is not None:
        if low is None:
            return -param, param
        return (low, param) if low < param else (param, low)
    else:
        return param


class BasicTransform(object):
    def __init__(self, always_apply=False, p=0.5):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, **kwargs):
        if (random.random() < self.p) or self.always_apply:
            initial_params = self.get_initial_params(**kwargs)
            params = self.get_params(initial_params)
            params.update(initial_params)
            res = {}
            for key, arg in kwargs.items():
                if arg is not None:
                    target_function = self.targets.get(key, lambda x, **p: x)
                    target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                    res[key] = target_function(arg, **dict(params, **target_dependencies))
                else:
                    res[key] = None
            return res
        return kwargs

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self, params):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def get_initial_params(self, **kwargs):
        params = {}
        if hasattr(self, 'interpolation'):
            params['interpolation'] = self.interpolation
        params.update({'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]})
        params.update({k: kwargs[k] for k in self.targets_as_params})
        return params

    @property
    def target_dependence(self):
        return {}

    @property
    def targets_as_params(self):
        return []


class DualTransform(BasicTransform):
    """Transform for segmentation task."""

    @property
    def targets(self):
        return {'image': self.apply, 'mask': self.apply_to_mask, 'bboxes': self.apply_to_bboxes,
                'masks': self.apply_to_masks}

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def apply_to_bboxes(self, bboxes, **params):
        bboxes = [list(bbox) for bbox in bboxes]
        return [self.apply_to_bbox(bbox[:4], **params) + bbox[4:] for bbox in bboxes]

    def apply_to_mask(self, img, **params):
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == 'interpolation' else v for k, v in params.items()})

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    @property
    def targets(self):
        return {'image': self.apply}


class NoOp(DualTransform):
    """Does nothing"""

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, img, **params):
        return img
