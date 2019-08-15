from __future__ import division

import random
import warnings

import numpy as np

from albumentations.augmentations.keypoints_utils import convert_keypoints_from_albumentations, filter_keypoints, \
    convert_keypoints_to_albumentations, check_keypoints
from albumentations.core.serialization import SerializableMeta
from albumentations.core.six import add_metaclass
from albumentations.core.transforms_interface import DualTransform
from albumentations.core.utils import format_args
from albumentations.imgaug.transforms import DualIAATransform
from albumentations.augmentations.bbox_utils import convert_bboxes_from_albumentations, \
    convert_bboxes_to_albumentations, filter_bboxes, check_bboxes

__all__ = ['Compose', 'OneOf', 'OneOrOther']


REPR_INDENT_STEP = 2


def find_dual_start_end(transforms):
    dual_start_end = None
    last_dual = None
    for idx, transform in enumerate(transforms):
        if isinstance(transform, DualTransform):
            last_dual = idx
            if dual_start_end is None:
                dual_start_end = [idx]
        if isinstance(transform, BaseCompose):
            inside = find_dual_start_end(transform)
            if inside is not None:
                last_dual = idx
                if dual_start_end is None:
                    dual_start_end = [idx]
    if dual_start_end is not None:
        dual_start_end.append(last_dual)
    return dual_start_end


def find_always_apply_transforms(transforms):
    new_transforms = []
    for transform in transforms:
        if isinstance(transform, BaseCompose):
            new_transforms.extend(find_always_apply_transforms(transform))
        elif transform.always_apply:
            new_transforms.append(transform)
    return new_transforms


def set_always_apply(transforms):
    for t in transforms:
        t.always_apply = True


@add_metaclass(SerializableMeta)
class BaseCompose(object):
    def __init__(self, transforms, p):
        self.transforms = transforms
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

        if bbox_params is None:
            bbox_params = {}
        if keypoint_params is None:
            keypoint_params = {}
        if additional_targets is None:
            additional_targets = {}

        self.bboxes_name = 'bboxes'
        self.keypoints_name = 'keypoints'
        self.additional_targets = additional_targets
        self.params = {
            self.bboxes_name: bbox_params,
            self.keypoints_name: keypoint_params
        }

        self.bbox_format = bbox_params.get('format', None)
        self.bbox_label_fields = bbox_params.get('label_fields', [])

        self.keypoints_format = keypoint_params.get('format', None)
        self.keypoints_label_fields = keypoint_params.get('label_fields', [])

        # IAA-based augmentations supports only transformation of xy keypoints.
        # If your keypoints formats is other than 'xy' we emit warning to let user
        # be aware that angle and size will not be modified.
        if self.keypoints_format is not None and self.keypoints_format != 'xy':
            for transform in self.transforms:
                if isinstance(transform, DualIAATransform):
                    warnings.warn("{} transformation supports only 'xy' keypoints "
                                  "augmentation. You have '{}' keypoints format. Scale "
                                  "and angle WILL NOT BE transformed.".format(transform.__class__.__name__,
                                                                              self.keypoints_format))
                    break

        self.add_targets(additional_targets)

    def __call__(self, force_apply=False, **data):
        need_to_run = force_apply or random.random() < self.p
        transforms = self.transforms if need_to_run else find_always_apply_transforms(self.transforms)
        dual_start_end = None
        if self.params[self.bboxes_name] or self.params[self.keypoints_name]:
            dual_start_end = find_dual_start_end(transforms)

        if (self.params[self.bboxes_name] and
                len(data.get(self.bboxes_name, [])) and
                len(data[self.bboxes_name][0]) < 5):
            if 'label_fields' not in self.params[self.bboxes_name]:
                raise Exception("Please specify 'label_fields' in 'bbox_params' or add labels to the end of bbox "
                                "because bboxes must have labels")
        if 'label_fields' in self.params[self.bboxes_name]:
            if not all(l in data.keys() for l in self.params[self.bboxes_name]['label_fields']):
                raise Exception("Your 'label_fields' are not valid - them must have same names as params in dict")

        if 'label_fields' in self.params[self.keypoints_name]:
            if not all(l in data.keys() for l in self.params[self.keypoints_name]['label_fields']):
                raise Exception("Your 'label_fields' are not valid - them must have same names as params in "
                                "'keypoint_params' dict")

        for idx, t in enumerate(transforms):
            if dual_start_end is not None and idx == dual_start_end[0]:
                if self.params[self.bboxes_name]:
                    data = data_preprocessing(self.bboxes_name, self.params[self.bboxes_name], check_bboxes,
                                              convert_bboxes_to_albumentations, data)
                if self.params[self.keypoints_name]:
                    data = data_preprocessing(self.keypoints_name, self.params[self.keypoints_name], check_keypoints,
                                              convert_keypoints_to_albumentations, data)

            data = t(force_apply=force_apply, **data)

            if dual_start_end is not None and idx == dual_start_end[1]:
                if self.params[self.bboxes_name]:
                    data = data_postprocessing(self.bboxes_name, self.params[self.bboxes_name], check_bboxes,
                                               filter_bboxes, convert_bboxes_from_albumentations, data)
                if self.params[self.keypoints_name]:
                    data = data_postprocessing(self.keypoints_name, self.params[self.keypoints_name], check_keypoints,
                                               filter_keypoints, convert_keypoints_from_albumentations, data)

        return data

    def _to_dict(self):
        dictionary = super(Compose, self)._to_dict()
        dictionary.update({
            'bbox_params': self.params[self.bboxes_name],
            'keypoint_params': self.params[self.keypoints_name],
            'additional_targets': self.additional_targets,
        })
        return dictionary


def data_postprocessing(data_name, params, check_fn, filter_fn, convert_fn, data):
    rows, cols = data['image'].shape[:2]

    additional_params = {}
    if data_name == 'bboxes':
        additional_params['min_area'] = params.get('min_area', 0.0),
        additional_params['min_visibility'] = params.get('min_visibility', 0.0)
    elif data_name == 'keypoints':
        additional_params['remove_invisible'] = bool(params.get('remove_invisible', True))
    else:
        raise Exception('Not known data_name')

    data[data_name] = filter_fn(data[data_name], rows, cols, **additional_params)

    if params['format'] == 'albumentations':
        check_fn(data[data_name])
    else:
        data[data_name] = convert_fn(data[data_name], params['format'], rows, cols,
                                     check_validity=bool(params.get('remove_invisible', True)))

    data = remove_label_fields_from_data(data_name, params.get('label_fields', []), data)
    return data


def data_preprocessing(data_name, params, check_fn, convert_fn, data):
    if data_name not in data:
        raise Exception('Please name field with {} `{}`'.format(data_name, data_name))
    data = add_label_fields_to_data(data_name, params.get('label_fields', []), data)

    rows, cols = data['image'].shape[:2]
    if params['format'] == 'albumentations':
        check_fn(data[data_name])
    else:
        data[data_name] = convert_fn(data[data_name], params['format'], rows, cols,
                                     check_validity=bool(params.get('remove_invisible', True)))

    return data


def add_label_fields_to_data(data_name, label_fields, data):
    for field in label_fields:
        data_with_added_field = []
        for d, field_value in zip(data[data_name], data[field]):
            data_with_added_field.append(list(d) + [field_value])
        data[data_name] = data_with_added_field
    return data


def remove_label_fields_from_data(data_name, label_fields, data):
    for idx, field in enumerate(label_fields):
        field_values = []
        for bbox in data[data_name]:
            field_values.append(bbox[4 + idx])
        data[field] = field_values
    if label_fields:
        data[data_name] = [d[:4] for d in data[data_name]]
    return data


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
        self.transforms = transforms
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
