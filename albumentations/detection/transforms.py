import random

from ..core.transforms_interface import BasicTransform
from ..augmentations import functional as F
from .functional import vflip_bbox, hflip_bbox


__all__ = ['DetectionTransform', 'VFlipWithBbox', 'HFlipWithBbox']


class DetectionTransform(BasicTransform):
    """
    transforms applied to images and boxes
    """

    def __init__(self, p=.5):
        super().__init__(p)

    @property
    def targets(self):
        return 'image', 'bboxes'

    def __call__(self, **kwargs):
        if random.random() < self.p:
            params = self.get_params()
            kwargs.update({'image': self.apply(kwargs['image'], **params)})
            params.update({'cols': kwargs['image'].shape[1], 'rows': kwargs['image'].shape[0]})
            kwargs.update({'bboxes': [self.apply_to_bbox(bbox, **params) for bbox in kwargs['bboxes']]})
        return kwargs

    def apply_to_bbox(self, bbox, **params):
        """
        :param bbox: tuple
        """
        raise NotImplementedError


class VFlipWithBbox(DetectionTransform):
    def apply(self, img, **params):
        return F.vflip(img)

    def apply_to_bbox(self, bbox, **params):
        return vflip_bbox(bbox, **params)


class HFlipWithBbox(DetectionTransform):
    def apply(self, img, **params):
        return F.hflip(img)

    def apply_to_bbox(self, bbox, **params):
        return hflip_bbox(bbox, **params)
