from __future__ import division

import random
import warnings

import numpy as np

from albumentations.augmentations.keypoints_utils import KeypointsParams, KeypointsProcessor
from albumentations.core.serialization import SerializableMeta
from albumentations.core.six import add_metaclass
from albumentations.core.transforms_interface import DualTransform
from albumentations.core.utils import format_args
from albumentations.imgaug.transforms import DualIAATransform
from albumentations.augmentations.bbox_utils import BboxParams, BboxProcessor

__all__ = ['Compose', 'OneOf', 'OneOrOther']


REPR_INDENT_STEP = 2


class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms
        self.start_end = self._find_dual_start_end(transforms)

    def _find_dual_start_end(self, transforms):
        dual_start_end = None
        last_dual = None
        for idx, transform in enumerate(transforms):
            if isinstance(transform, DualTransform):
                last_dual = idx
                if dual_start_end is None:
                    dual_start_end = [idx]
            if isinstance(transform, BaseCompose):
                inside = self._find_dual_start_end(transform)
                if inside is not None:
                    last_dual = idx
                    if dual_start_end is None:
                        dual_start_end = [idx]
        if dual_start_end is not None:
            dual_start_end.append(last_dual)
        return dual_start_end

    def get_always_apply(self, transforms):
        new_transforms = []
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                new_transforms.extend(self.get_always_apply(transform))
            elif transform.always_apply:
                new_transforms.append(transform)
        return Transforms(new_transforms)

    def __getitem__(self, item):
        return self.transforms[item]


def set_always_apply(transforms):
    for t in transforms:
        t.always_apply = True


@add_metaclass(SerializableMeta)
class BaseCompose(object):
    def __init__(self, transforms, p):
        self.transforms = Transforms(transforms)
        self.p = p

    def __getitem__(self, item):
        return self.transforms[item]

    def __repr__(self):
        return self.indented_repr()

    def indented_repr(self, indent=REPR_INDENT_STEP):
        args = {k: v for k, v in self._to_dict().items() if not (k.startswith('__') or k == 'transforms')}
        repr_string = self.__class__.__name__ + '(['
        for t in self.transforms:
            repr_string += '\n'
            if hasattr(t, 'indented_repr'):
                t_repr = t.indented_repr(indent + REPR_INDENT_STEP)
            else:
                t_repr = repr(t)
            repr_string += ' ' * indent + t_repr + ','
        repr_string += '\n' + ' ' * (indent - REPR_INDENT_STEP) + '], {args})'.format(args=format_args(args))
        return repr_string

    @classmethod
    def get_class_fullname(cls):
        return '{cls.__module__}.{cls.__name__}'.format(cls=cls)

    def _to_dict(self):
        return {
            '__class_fullname__': self.get_class_fullname(),
            'p': self.p,
            'transforms': [t._to_dict() for t in self.transforms],
        }

    def add_targets(self, additional_targets):
        if additional_targets:
            for t in self.transforms:
                t.add_targets(additional_targets)


class Compose(BaseCompose):
    """Compose transforms and handle all transformations regrading bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        bbox_params (dict): Parameters for bounding boxes transforms
        keypoint_params (dict): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.

    **bbox_params** dictionary contains the following keys:
        * **format** (*str*): format of bounding boxes. Should be 'coco', 'pascal_voc' or 'albumentations'.
          If None - don't use bboxes.
          The `coco` format of a bounding box looks like `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
          The `pascal_voc` format of a bounding box looks like `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
          The `albumentations` format of a bounding box looks like `pascal_voc`, but between [0, 1], in other words:
          [x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
        * | **label_fields** (*list*): list of fields that are joined with boxes, e.g labels.
          | Should be same type as boxes.
        * | **min_area** (*float*): minimum area of a bounding box. All bounding boxes whose
          | visible area in pixels is less than this value will be removed. Default: 0.0.
        * | **min_visibility** (*float*): minimum fraction of area for a bounding box
          | to remain this box in list. Default: 0.0.
    """

    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1.0):
        super(Compose, self).__init__([t for t in transforms if t is not None], p)

        self.processors = []
        if bbox_params is not None:
            params = BboxParams(**bbox_params)
            self.processors.append(BboxProcessor(params, ['bboxes']))

        if keypoint_params is not None:
            params = KeypointsParams(**keypoint_params)
            self.processors.append(KeypointsProcessor(params, ['keypoints']))
        # todo additional

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        for proc in self.processors:
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

    def __call__(self, force_apply=False, **data):
        need_to_run = force_apply or random.random() < self.p
        for p in self.processors:
            p.ensure_data_valid(data)
        transforms = self.transforms if need_to_run else self.transforms.get_always_apply(self.transforms)
        dual_start_end = transforms.start_end if self.processors else None

        for idx, t in enumerate(transforms):
            if dual_start_end is not None and idx == dual_start_end[0]:
                for p in self.processors:
                    p.preprocess(data)

            data = t(force_apply=force_apply, **data)

            if dual_start_end is not None and idx == dual_start_end[1]:
                for p in self.processors:
                    p.postprocess(data)

        return data

    def _to_dict(self):
        #todo
        dictionary = super(Compose, self)._to_dict()
        dictionary.update({
            'bbox_params': self.params[self.bboxes_name],
            'keypoint_params': self.params[self.keypoints_name],
            'additional_targets': self.additional_targets,
        })
        return dictionary


class OneOf(BaseCompose):
    """Select on of transforms to apply

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transforms, p=0.5):
        super(OneOf, self).__init__(transforms, p)
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, force_apply=False, **data):
        if force_apply or random.random() < self.p:
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(force_apply=True, **data)
        return data


class OneOrOther(BaseCompose):

    def __init__(self, first=None, second=None, transforms=None, p=0.5):
        if transforms is None:
            transforms = [first, second]
        super(OneOrOther, self).__init__(transforms, p)

    def __call__(self, force_apply=False, **data):
        if random.random() < self.p:
            return self.transforms[0](force_apply=True, **data)
        else:
            return self.transforms[-1](force_apply=True, **data)


class PerChannel(BaseCompose):
    """Apply transformations per-channel

    Args:
        transforms (list): list of transformations to compose.
        channels (list): channels to apply the transform to. Pass None to apply to all.
                         Default: None (apply to all)
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, transforms, channels=None, p=0.5):
        super(PerChannel, self).__init__(transforms, p)
        self.channels = channels

    def __call__(self, force_apply=False, **data):
        if force_apply or random.random() < self.p:

            image = data['image']

            # Expan mono images to have a single channel
            if len(image.shape) == 2:
                image = np.expand_dims(image, -1)

            if self.channels is None:
                self.channels = range(image.shape[2])

            for c in self.channels:
                for t in self.transforms:
                    image[:, :, c] = t(image=image[:, :, c])['image']

            data['image'] = image

        return data
