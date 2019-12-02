from __future__ import division
from collections import defaultdict

import random

import numpy as np

from albumentations.augmentations.keypoints_utils import KeypointsProcessor
from albumentations.core.serialization import SerializableMeta
from albumentations.core.six import add_metaclass
from albumentations.core.transforms_interface import DualTransform
from albumentations.core.utils import format_args, Params
from albumentations.augmentations.bbox_utils import BboxProcessor
from albumentations.core.serialization import SERIALIZABLE_REGISTRY, instantiate_lambda

__all__ = ["Compose", "OneOf", "OneOrOther", "BboxParams", "KeypointParams", "ReplayCompose"]


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

        self.replay_mode = False
        self.applied_in_replay = False

    def __getitem__(self, item):
        return self.transforms[item]

    def __repr__(self):
        return self.indented_repr()

    def indented_repr(self, indent=REPR_INDENT_STEP):
        args = {k: v for k, v in self._to_dict().items() if not (k.startswith("__") or k == "transforms")}
        repr_string = self.__class__.__name__ + "(["
        for t in self.transforms:
            repr_string += "\n"
            if hasattr(t, "indented_repr"):
                t_repr = t.indented_repr(indent + REPR_INDENT_STEP)
            else:
                t_repr = repr(t)
            repr_string += " " * indent + t_repr + ","
        repr_string += "\n" + " " * (indent - REPR_INDENT_STEP) + "], {args})".format(args=format_args(args))
        return repr_string

    @classmethod
    def get_class_fullname(cls):
        return "{cls.__module__}.{cls.__name__}".format(cls=cls)

    def _to_dict(self):
        return {
            "__class_fullname__": self.get_class_fullname(),
            "p": self.p,
            "transforms": [t._to_dict() for t in self.transforms],
        }

    def get_dict_with_id(self):
        return {
            "__class_fullname__": self.get_class_fullname(),
            "id": id(self),
            "params": None,
            "transforms": [t.get_dict_with_id() for t in self.transforms],
        }

    def add_targets(self, additional_targets):
        if additional_targets:
            for t in self.transforms:
                t.add_targets(additional_targets)

    def set_deterministic(self, flag, save_key="replay"):
        for t in self.transforms:
            t.set_deterministic(flag, save_key)


class Compose(BaseCompose):
    """Compose transforms and handle all transformations regrading bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
    """

    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1.0):
        super(Compose, self).__init__([t for t in transforms if t is not None], p)

        self.processors = {}
        if bbox_params:
            if isinstance(bbox_params, dict):
                params = BboxParams(**bbox_params)
            elif isinstance(bbox_params, BboxParams):
                params = bbox_params
            else:
                raise ValueError("unknown format of bbox_params, please use `dict` or `BboxParams`")
            self.processors["bboxes"] = BboxProcessor(params, additional_targets)

        if keypoint_params:
            if isinstance(keypoint_params, dict):
                params = KeypointParams(**keypoint_params)
            elif isinstance(keypoint_params, KeypointParams):
                params = keypoint_params
            else:
                raise ValueError("unknown format of keypoint_params, please use `dict` or `KeypointParams`")
            self.processors["keypoints"] = KeypointsProcessor(params, additional_targets)

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

    def __call__(self, force_apply=False, **data):
        assert isinstance(force_apply, (bool, int)), "force_apply must have bool or int type"
        need_to_run = force_apply or random.random() < self.p
        for p in self.processors.values():
            p.ensure_data_valid(data)
        transforms = self.transforms if need_to_run else self.transforms.get_always_apply(self.transforms)
        dual_start_end = transforms.start_end if self.processors else None

        for idx, t in enumerate(transforms):
            if dual_start_end is not None and idx == dual_start_end[0]:
                for p in self.processors.values():
                    p.preprocess(data)

            data = t(force_apply=force_apply, **data)

            if dual_start_end is not None and idx == dual_start_end[1]:
                for p in self.processors.values():
                    p.postprocess(data)

        return data

    def _to_dict(self):
        dictionary = super(Compose, self)._to_dict()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params._to_dict() if bbox_processor else None,
                "keypoint_params": keypoints_processor.params._to_dict() if keypoints_processor else None,
                "additional_targets": self.additional_targets,
            }
        )
        return dictionary


class OneOf(BaseCompose):
    """Select one of transforms to apply

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
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if self.transforms_ps and (force_apply or random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms.transforms, p=self.transforms_ps)
            data = t(force_apply=True, **data)
        return data


class OneOrOther(BaseCompose):
    def __init__(self, first=None, second=None, transforms=None, p=0.5):
        if transforms is None:
            transforms = [first, second]
        super(OneOrOther, self).__init__(transforms, p)

    def __call__(self, force_apply=False, **data):
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

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

            image = data["image"]

            # Expan mono images to have a single channel
            if len(image.shape) == 2:
                image = np.expand_dims(image, -1)

            if self.channels is None:
                self.channels = range(image.shape[2])

            for c in self.channels:
                for t in self.transforms:
                    image[:, :, c] = t(image=image[:, :, c])["image"]

            data["image"] = image

        return data


class ReplayCompose(Compose):
    def __init__(
        self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1.0, save_key="replay"
    ):
        super(ReplayCompose, self).__init__(transforms, bbox_params, keypoint_params, additional_targets, p)
        self.set_deterministic(True, save_key=save_key)
        self.save_key = save_key

    def __call__(self, force_apply=False, **kwargs):
        kwargs[self.save_key] = defaultdict(dict)
        result = super(ReplayCompose, self).__call__(force_apply=force_apply, **kwargs)
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, result[self.save_key])
        self.fill_applied(serialized)
        result[self.save_key] = serialized
        return result

    @staticmethod
    def replay(saved_augmentations, **kwargs):
        augs = ReplayCompose._restore_for_replay(saved_augmentations)
        return augs(force_apply=True, **kwargs)

    @staticmethod
    def _restore_for_replay(transform_dict, lambda_transforms=None):
        """
        Args:
            transform (dict): A dictionary with serialized transform pipeline.
            lambda_transforms (dict): A dictionary that contains lambda transforms, that
            is instances of the Lambda class.
                This dictionary is required when you are restoring a pipeline that contains lambda transforms. Keys
                in that dictionary should be named same as `name` arguments in respective lambda transforms from
                a serialized pipeline.
        """
        transform = transform_dict
        applied = transform["applied"]
        params = transform["params"]
        lmbd = instantiate_lambda(transform, lambda_transforms)
        if lmbd:
            transform = lmbd
        else:
            name = transform["__class_fullname__"]
            args = {k: v for k, v in transform.items() if k not in ["__class_fullname__", "applied", "params"]}
            cls = SERIALIZABLE_REGISTRY[name]
            if "transforms" in args:
                args["transforms"] = [
                    ReplayCompose._restore_for_replay(t, lambda_transforms=lambda_transforms)
                    for t in args["transforms"]
                ]
            transform = cls(**args)

        transform.params = params
        transform.replay_mode = True
        transform.applied_in_replay = applied
        return transform

    def fill_with_params(self, serialized, all_params):
        params = all_params.get(serialized.get("id"))
        serialized["params"] = params
        del serialized["id"]
        for transform in serialized.get("transforms", []):
            self.fill_with_params(transform, all_params)

    def fill_applied(self, serialized):
        if "transforms" in serialized:
            applied = [self.fill_applied(t) for t in serialized["transforms"]]
            serialized["applied"] = any(applied)
        else:
            serialized["applied"] = serialized.get("params") is not None
        return serialized["applied"]

    def _to_dict(self):
        raise NotImplementedError("You cannot serialize ReplayCompose")


class BboxParams(Params):
    """
    Parameters of bounding boxes

    Args:
        format (str): format of bounding boxes. Should be 'coco', 'pascal_voc', 'albumentations' or 'yolo'.

            The `coco` format
                `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
            The `pascal_voc` format
                `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
            The `albumentations` format
                is like `pascal_voc`, but normalized,
                in other words: [x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
            The `yolo` format
                `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
                `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
        label_fields (list): list of fields that are joined with boxes, e.g labels.
            Should be same type as boxes.
        min_area (float): minimum area of a bounding box. All bounding boxes whose
            visible area in pixels is less than this value will be removed. Default: 0.0.
        min_visibility (float): minimum fraction of area for a bounding box
            to remain this box in list. Default: 0.0.
    """

    def __init__(self, format, label_fields=None, min_area=0.0, min_visibility=0.0):
        super(BboxParams, self).__init__(format, label_fields)
        self.min_area = min_area
        self.min_visibility = min_visibility

    def _to_dict(self):
        data = super(BboxParams, self)._to_dict()
        data.update({"min_area": self.min_area, "min_visibility": self.min_visibility})
        return data


class KeypointParams(Params):
    """
    Parameters of keypoints

    Args:
        format (str): format of keypoints. Should be 'xy', 'yx', 'xya', 'xys', 'xyas', 'xysa'.

            x - X coordinate,

            y - Y coordinate

            s - Keypoint scale

            a - Keypoint orientation in radians or degrees (depending on KeypointParams.angle_in_degrees)
        label_fields (list): list of fields that are joined with keypoints, e.g labels.
            Should be same type as keypoints.
        remove_invisible (bool): to remove invisible points after transform or not
        angle_in_degrees (bool): angle in degrees or radians in 'xya', 'xyas', 'xysa' keypoints
    """

    def __init__(self, format, label_fields=None, remove_invisible=True, angle_in_degrees=True):
        super(KeypointParams, self).__init__(format, label_fields)
        self.remove_invisible = remove_invisible
        self.angle_in_degrees = angle_in_degrees

    def _to_dict(self):
        data = super(KeypointParams, self)._to_dict()
        data.update({"remove_invisible": self.remove_invisible, "angle_in_degrees": self.angle_in_degrees})
        return data
