from __future__ import division

import random

import numpy as np
from albumentations.augmentations.bbox_utils import convert_bboxes_from_albumentations, \
    convert_bboxes_to_albumentations, filter_bboxes


__all__ = ['Compose', 'OneOf', 'OneOrOther']


class Compose(object):
    """Compose transforms and handle all transformations regrading bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        preprocessing_transforms (list): list of transforms to run before transforms
        postprocessing_transforms (list): list of transforms to run after transforms
        p (float): probability of applying all list of transforms. Default: 1.0.
        bbox_params (dict): Parameters for bounding boxes transforms

    **bbox_params** dictionary contains the following keys:
        * **format** (*str*): format of bounding boxes. Should be 'coco' or 'pascal_voc'. If None - don't use bboxes.
          The `coco` format of a bounding box looks like `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
          The `pascal_voc` format of a bounding box looks like `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
        * | **label_fields** (*list*): list of fields that are joined with boxes, e.g labels.
          | Should be same type as boxes.
        * | **min_area** (*float*): minimum area of a bounding box. All bounding boxes whose
          | visible area in pixels is less than this value will be removed. Default: 0.0.
        * | **min_visibility** (*float*): minimum fraction of area for a bounding box
          | to remain this box in list. Default: 0.0.
    """

    def __init__(self, transforms, preprocessing_transforms=[], postprocessing_transforms=[], bbox_params={}, p=1.0):
        self.preprocessing_transforms = preprocessing_transforms
        self.postprocessing_transforms = postprocessing_transforms
        self.transforms = [t for t in transforms if t is not None]
        self.p = p
        self.bbox_format = bbox_params.get('format', None)
        self.label_fields = bbox_params.get('label_fields', [])
        self.min_area = bbox_params.get('min_area', 0.0)
        self.min_visibility = bbox_params.get('min_visibility', 0.0)

    def __call__(self, **data):
        need_to_run = random.random() < self.p
        if 'bboxes' in data:
            if self.bbox_format is None:
                raise Exception("Please specify 'pascal_voc' or 'coco' as 'format' in 'bbox_params'")
            if data['bboxes'] and len(data['bboxes'][0]) < 5:
                if not self.label_fields:
                    raise Exception("Please specify 'label_fields' in 'bbox_params' or add labels to the end of bbox "
                                    "because bboxes must have labels")
            if self.label_fields:
                if not all(l in data.keys() for l in self.label_fields):
                    raise Exception("Your 'label_fields' are not valid - them must have same names as params in dict")
        if self.preprocessing_transforms or need_to_run:
            if self.bbox_format is not None:
                data = self.boxes_preprocessing(data)

            data = self.run_transforms_if_needed(need_to_run, data)

            if self.bbox_format is not None:
                data = self.boxes_postprocessing(data)
        return data

    def run_transforms_if_needed(self, need_to_run, data):
        for t in self.preprocessing_transforms:
            data = t(**data)

        if need_to_run:
            for t in self.transforms:
                data = t(**data)

        for t in self.postprocessing_transforms:
            data = t(**data)
        return data

    def boxes_preprocessing(self, data):
        if 'bboxes' not in data:
            raise Exception('Please name field with bounding boxes `bboxes`')
        if self.label_fields:
            for field in self.label_fields:
                bboxes_with_added_field = []
                for bbox, field_value in zip(data['bboxes'], data[field]):
                    bboxes_with_added_field.append(list(bbox) + [field_value])
                data['bboxes'] = bboxes_with_added_field

        rows, cols = data['image'].shape[:2]
        data['bboxes'] = convert_bboxes_to_albumentations(data['bboxes'], self.bbox_format, rows, cols,
                                                          check_validity=True)

        return data

    def boxes_postprocessing(self, data):
        rows, cols = data['image'].shape[:2]
        data['bboxes'] = filter_bboxes(data['bboxes'], rows, cols, self.min_area, self.min_visibility)

        data['bboxes'] = convert_bboxes_from_albumentations(data['bboxes'], self.bbox_format, rows, cols,
                                                            check_validity=True)

        if self.label_fields:
            for idx, field in enumerate(self.label_fields):
                field_values = []
                for bbox in data['bboxes']:
                    field_values.append(bbox[4 + idx])
                data[field] = field_values
            data['bboxes'] = [bbox[:4] for bbox in data['bboxes']]
        return data


class OneOf(object):
    """Select on of transforms to apply

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

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
