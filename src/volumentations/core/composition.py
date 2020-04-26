from __future__ import division
from collections import defaultdict

import random

import numpy as np

from volumentations.core.serialization import SerializableMeta
from volumentations.core.six import add_metaclass
from volumentations.core.utils import format_args
from volumentations.core.serialization import SERIALIZABLE_REGISTRY

__all__ = [
    "Compose",
    "OneOf",
    "OneOrOther",
    "ReplayCompose",
]


REPR_INDENT_STEP = 2


class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms
        self.start_end = self._find_dual_start_end(transforms)

    def _find_dual_start_end(self, transforms):
        dual_start_end = None
        last_dual = None
        for idx, transform in enumerate(transforms):
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
class BaseCompose:
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
        args = {
            k: v
            for k, v in self._to_dict().items()
            if not (k.startswith("__") or k == "transforms")
        }
        repr_string = self.__class__.__name__ + "(["
        for t in self.transforms:
            repr_string += "\n"
            if hasattr(t, "indented_repr"):
                t_repr = t.indented_repr(indent + REPR_INDENT_STEP)
            else:
                t_repr = repr(t)
            repr_string += " " * indent + t_repr + ","
        repr_string += (
            "\n"
            + " " * (indent - REPR_INDENT_STEP)
            + "], {args})".format(args=format_args(args))
        )
        return repr_string

    @classmethod
    def get_class_fullname(cls):
        return "{cls.__module__}.{cls.__name__}".format(cls=cls)

    def _to_dict(self):
        return {
            "__class_fullname__": self.get_class_fullname(),
            "p": self.p,
            "transforms": [t._to_dict() for t in self.transforms],  # skipcq: PYL-W0212
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

    def __init__(
        self,
        transforms,
        bbox_params=None,
        keypoint_params=None,
        additional_targets=None,
        p=1.0,
    ):
        super(Compose, self).__init__([t for t in transforms if t is not None], p)

        self.processors = {}

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

    def __call__(self, force_apply=False, **data):
        assert isinstance(
            force_apply, (bool, int)
        ), "force_apply must have bool or int type"
        need_to_run = force_apply or random.random() < self.p
        for p in self.processors.values():
            p.ensure_data_valid(data)
        transforms = (
            self.transforms
            if need_to_run
            else self.transforms.get_always_apply(self.transforms)
        )
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
                "bbox_params": bbox_processor.params._to_dict()
                if bbox_processor
                else None,  # skipcq: PYL-W0212
                "keypoint_params": keypoints_processor.params._to_dict()  # skipcq: PYL-W0212
                if keypoints_processor
                else None,
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

        return self.transforms[-1](force_apply=True, **data)


class ReplayCompose(Compose):
    def __init__(
        self,
        transforms,
        bbox_params=None,
        keypoint_params=None,
        additional_targets=None,
        p=1.0,
        save_key="replay",
    ):
        super(ReplayCompose, self).__init__(
            transforms, bbox_params, keypoint_params, additional_targets, p
        )
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
    def _restore_for_replay(transform_dict):
        """
        Args:
            transform (dict): A dictionary with serialized transform pipeline.
        """
        transform = transform_dict
        applied = transform["applied"]
        params = transform["params"]
        name = transform["__class_fullname__"]
        args = {
            k: v
            for k, v in transform.items()
            if k not in ["__class_fullname__", "applied", "params"]
        }
        cls = SERIALIZABLE_REGISTRY[name]
        if "transforms" in args:
            args["transforms"] = [
                ReplayCompose._restore_for_replay(t) for t in args["transforms"]
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
