from __future__ import division

import random

import numpy as np
from albumentations.augmentations.bbox_utils import convert_bboxes_from_albumentations, \
    convert_bboxes_to_albumentations, filter_bboxes


__all__ = ['Compose', 'OneOf', 'OneOrOther']


class Compose(object):
    """Compose transforms together."""

    def __init__(self, transforms, p=1.0):
        self.transforms = [t for t in transforms if t is not None]
        self.p = p

    def __call__(self, **data):
        if random.random() < self.p:
            for t in self.transforms:
                data = t(**data)
        return data


class ComposeWithBoxes(object):
    """Compose transforms and handle all transformations regrading bounding boxes"""

    def __init__(self, transforms, bbox_format, label_fields=[], min_area=0., min_visibility=0., p=1.0):
        self.bboxe_format = bbox_format
        self.label_fields = label_fields
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.transforms = [t for t in transforms if t is not None]
        self.p = p

    def __call__(self, **data):
        if random.random() < self.p:
            if 'bboxes' not in data:
                raise Exception('Please name field with bounding boxes `bboxes`')
            if self.label_fields:
                for field in self.label_fields:
                    bboxes_with_added_field = []
                    for bbox, field_value in zip(data['bboxes'], data[field]):
                        bboxes_with_added_field.append(list(bbox) + [field_value])
                    data['bboxes'] = bboxes_with_added_field

            rows, cols = data['image'].shape[:2]
            data['bboxes'] = convert_bboxes_to_albumentations(data['bboxes'], self.bboxe_format, rows, cols,
                                                              check_validity=True)
            for t in self.transforms:
                data = t(**data)

            data['bboxes'] = filter_bboxes(data['bboxes'], rows, cols, self.min_area, self.min_visibility)

            data['bboxes'] = convert_bboxes_from_albumentations(data['bboxes'], self.bboxe_format, rows, cols,
                                                                check_validity=True)

            if self.label_fields:
                for idx, field in enumerate(self.label_fields):
                    field_values = []
                    for bbox in data['bboxes']:
                        field_values.append(bbox[4+idx])
                    data[field] = field_values
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
