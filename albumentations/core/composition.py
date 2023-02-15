from __future__ import division

import random
import typing
import warnings
from collections import defaultdict

import numpy as np

from .. import random_utils
from .bbox_utils import BboxParams, BboxProcessor
from .keypoints_utils import KeypointParams, KeypointsProcessor
from .serialization import (
    SERIALIZABLE_REGISTRY,
    Serializable,
    get_shortest_class_fullname,
    instantiate_nonserializable,
)
from .transforms_interface import BasicTransform
from .utils import format_args, get_shape

__all__ = [
    "BaseCompose",
    "Compose",
    "SomeOf",
    "OneOf",
    "OneOrOther",
    "BboxParams",
    "KeypointParams",
    "ReplayCompose",
    "Sequential",
]


REPR_INDENT_STEP = 2
TransformType = typing.Union[BasicTransform, "BaseCompose"]
TransformsSeqType = typing.Sequence[TransformType]


def get_always_apply(transforms: typing.Union["BaseCompose", TransformsSeqType]) -> TransformsSeqType:
    new_transforms: typing.List[TransformType] = []
    for transform in transforms:  # type: ignore
        if isinstance(transform, BaseCompose):
            new_transforms.extend(get_always_apply(transform))
        elif transform.always_apply:
            new_transforms.append(transform)
    return new_transforms


class BaseCompose(Serializable):
    def __init__(self, transforms: TransformsSeqType, p: float):
        if isinstance(transforms, (BaseCompose, BasicTransform)):
            warnings.warn(
                "transforms is single transform, but a sequence is expected! Transform will be wrapped into list."
            )
            transforms = [transforms]

        self.transforms = transforms
        self.p = p

        self.replay_mode = False
        self.applied_in_replay = False

    def __len__(self) -> int:
        return len(self.transforms)

    def __call__(self, *args, **data) -> typing.Dict[str, typing.Any]:
        raise NotImplementedError

    def __getitem__(self, item: int) -> TransformType:  # type: ignore
        return self.transforms[item]

    def __repr__(self) -> str:
        return self.indented_repr()

    def indented_repr(self, indent: int = REPR_INDENT_STEP) -> str:
        args = {k: v for k, v in self._to_dict().items() if not (k.startswith("__") or k == "transforms")}
        repr_string = self.__class__.__name__ + "(["
        for t in self.transforms:
            repr_string += "\n"
            if hasattr(t, "indented_repr"):
                t_repr = t.indented_repr(indent + REPR_INDENT_STEP)  # type: ignore
            else:
                t_repr = repr(t)
            repr_string += " " * indent + t_repr + ","
        repr_string += "\n" + " " * (indent - REPR_INDENT_STEP) + "], {args})".format(args=format_args(args))
        return repr_string

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls) -> bool:
        return True

    def _to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            "__class_fullname__": self.get_class_fullname(),
            "p": self.p,
            "transforms": [t._to_dict() for t in self.transforms],  # skipcq: PYL-W0212
        }

    def get_dict_with_id(self) -> typing.Dict[str, typing.Any]:
        return {
            "__class_fullname__": self.get_class_fullname(),
            "id": id(self),
            "params": None,
            "transforms": [t.get_dict_with_id() for t in self.transforms],
        }

    def add_targets(self, additional_targets: typing.Optional[typing.Dict[str, str]]) -> None:
        if additional_targets:
            for t in self.transforms:
                t.add_targets(additional_targets)

    def set_deterministic(self, flag: bool, save_key: str = "replay") -> None:
        for t in self.transforms:
            t.set_deterministic(flag, save_key)


class Compose(BaseCompose):
    """Compose transforms and handle all transformations regarding bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
        is_check_shapes (bool): If True shapes consistency of images/mask/masks would be checked on each call. If you
            would like to disable this check - pass False (do it only if you are sure in your data consistency).
    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: typing.Optional[typing.Union[dict, "BboxParams"]] = None,
        keypoint_params: typing.Optional[typing.Union[dict, "KeypointParams"]] = None,
        additional_targets: typing.Optional[typing.Dict[str, str]] = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
    ):
        super(Compose, self).__init__(transforms, p)

        self.processors: typing.Dict[str, typing.Union[BboxProcessor, KeypointsProcessor]] = {}
        if bbox_params:
            if isinstance(bbox_params, dict):
                b_params = BboxParams(**bbox_params)
            elif isinstance(bbox_params, BboxParams):
                b_params = bbox_params
            else:
                raise ValueError("unknown format of bbox_params, please use `dict` or `BboxParams`")
            self.processors["bboxes"] = BboxProcessor(b_params, additional_targets)

        if keypoint_params:
            if isinstance(keypoint_params, dict):
                k_params = KeypointParams(**keypoint_params)
            elif isinstance(keypoint_params, KeypointParams):
                k_params = keypoint_params
            else:
                raise ValueError("unknown format of keypoint_params, please use `dict` or `KeypointParams`")
            self.processors["keypoints"] = KeypointsProcessor(k_params, additional_targets)

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

        self.is_check_args = True
        self._disable_check_args_for_transforms(self.transforms)

        self.is_check_shapes = is_check_shapes

    @staticmethod
    def _disable_check_args_for_transforms(transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                Compose._disable_check_args_for_transforms(transform.transforms)
            if isinstance(transform, Compose):
                transform._disable_check_args()

    def _disable_check_args(self) -> None:
        self.is_check_args = False

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        if self.is_check_args:
            self._check_args(**data)
        assert isinstance(force_apply, (bool, int)), "force_apply must have bool or int type"
        need_to_run = force_apply or random.random() < self.p
        for p in self.processors.values():
            p.ensure_data_valid(data)
        transforms = self.transforms if need_to_run else get_always_apply(self.transforms)

        check_each_transform = any(
            getattr(item.params, "check_each_transform", False) for item in self.processors.values()
        )

        for p in self.processors.values():
            p.preprocess(data)

        for idx, t in enumerate(transforms):
            data = t(**data)

            if check_each_transform:
                data = self._check_data_post_transform(data)
        data = Compose._make_targets_contiguous(data)  # ensure output targets are contiguous

        for p in self.processors.values():
            p.postprocess(data)

        return data

    def _check_data_post_transform(self, data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        rows, cols = get_shape(data["image"])

        for p in self.processors.values():
            if not getattr(p.params, "check_each_transform", False):
                continue

            for data_name in p.data_fields:
                data[data_name] = p.filter(data[data_name], rows, cols)
        return data

    def _to_dict(self) -> typing.Dict[str, typing.Any]:
        dictionary = super(Compose, self)._to_dict()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params._to_dict() if bbox_processor else None,  # skipcq: PYL-W0212
                "keypoint_params": keypoints_processor.params._to_dict()  # skipcq: PYL-W0212
                if keypoints_processor
                else None,
                "additional_targets": self.additional_targets,
                "is_check_shapes": self.is_check_shapes,
            }
        )
        return dictionary

    def get_dict_with_id(self) -> typing.Dict[str, typing.Any]:
        dictionary = super().get_dict_with_id()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params._to_dict() if bbox_processor else None,  # skipcq: PYL-W0212
                "keypoint_params": keypoints_processor.params._to_dict()  # skipcq: PYL-W0212
                if keypoints_processor
                else None,
                "additional_targets": self.additional_targets,
                "params": None,
                "is_check_shapes": self.is_check_shapes,
            }
        )
        return dictionary

    def _check_args(self, **kwargs) -> None:
        checked_single = ["image", "mask"]
        checked_multi = ["masks"]
        check_bbox_param = ["bboxes"]
        # ["bboxes", "keypoints"] could be almost any type, no need to check them
        shapes = []
        for data_name, data in kwargs.items():
            internal_data_name = self.additional_targets.get(data_name, data_name)
            if internal_data_name in checked_single:
                if not isinstance(data, np.ndarray):
                    raise TypeError("{} must be numpy array type".format(data_name))
                shapes.append(data.shape[:2])
            if internal_data_name in checked_multi:
                if data is not None:
                    if not isinstance(data[0], np.ndarray):
                        raise TypeError("{} must be list of numpy arrays".format(data_name))
                    shapes.append(data[0].shape[:2])
            if internal_data_name in check_bbox_param and self.processors.get("bboxes") is None:
                raise ValueError("bbox_params must be specified for bbox transformations")

        if self.is_check_shapes and shapes and shapes.count(shapes[0]) != len(shapes):
            raise ValueError(
                "Height and Width of image, mask or masks should be equal. You can disable shapes check "
                "by setting a parameter is_check_shapes=False of Compose class (do it only if you are sure "
                "about your data consistency)."
            )

    @staticmethod
    def _make_targets_contiguous(data: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            result[key] = value
        return result


class OneOf(BaseCompose):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transforms: TransformsSeqType, p: float = 0.5):
        super(OneOf, self).__init__(transforms, p)
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if self.transforms_ps and (force_apply or random.random() < self.p):
            idx: int = random_utils.choice(len(self.transforms), p=self.transforms_ps)
            t = self.transforms[idx]
            data = t(force_apply=True, **data)
        return data


class SomeOf(BaseCompose):
    """Select N transforms to apply. Selected transforms will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        n (int): number of transforms to apply.
        replace (bool): Whether the sampled transforms are with or without replacement. Default: True.
        p (float): probability of applying selected transform. Default: 1.
    """

    def __init__(self, transforms: TransformsSeqType, n: int, replace: bool = True, p: float = 1):
        super(SomeOf, self).__init__(transforms, p)
        self.n = n
        self.replace = replace
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if self.transforms_ps and (force_apply or random.random() < self.p):
            idx = random_utils.choice(len(self.transforms), size=self.n, replace=self.replace, p=self.transforms_ps)
            for i in idx:  # type: ignore
                t = self.transforms[i]
                data = t(force_apply=True, **data)
        return data

    def _to_dict(self) -> typing.Dict[str, typing.Any]:
        dictionary = super(SomeOf, self)._to_dict()
        dictionary.update({"n": self.n, "replace": self.replace})
        return dictionary


class OneOrOther(BaseCompose):
    """Select one or another transform to apply. Selected transform will be called with `force_apply=True`."""

    def __init__(
        self,
        first: typing.Optional[TransformType] = None,
        second: typing.Optional[TransformType] = None,
        transforms: typing.Optional[TransformsSeqType] = None,
        p: float = 0.5,
    ):
        if transforms is None:
            if first is None or second is None:
                raise ValueError("You must set both first and second or set transforms argument.")
            transforms = [first, second]
        super(OneOrOther, self).__init__(transforms, p)
        if len(self.transforms) != 2:
            warnings.warn("Length of transforms is not equal to 2.")

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if random.random() < self.p:
            return self.transforms[0](force_apply=True, **data)

        return self.transforms[-1](force_apply=True, **data)


class PerChannel(BaseCompose):
    """Apply transformations per-channel

    Args:
        transforms (list): list of transformations to compose.
        channels (sequence): channels to apply the transform to. Pass None to apply to all.
                         Default: None (apply to all)
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self, transforms: TransformsSeqType, channels: typing.Optional[typing.Sequence[int]] = None, p: float = 0.5
    ):
        super(PerChannel, self).__init__(transforms, p)
        self.channels = channels

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:
        if force_apply or random.random() < self.p:

            image = data["image"]

            # Expand mono images to have a single channel
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
        self,
        transforms: TransformsSeqType,
        bbox_params: typing.Optional[typing.Union[dict, "BboxParams"]] = None,
        keypoint_params: typing.Optional[typing.Union[dict, "KeypointParams"]] = None,
        additional_targets: typing.Optional[typing.Dict[str, str]] = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        save_key: str = "replay",
    ):
        super(ReplayCompose, self).__init__(
            transforms, bbox_params, keypoint_params, additional_targets, p, is_check_shapes
        )
        self.set_deterministic(True, save_key=save_key)
        self.save_key = save_key

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> typing.Dict[str, typing.Any]:
        kwargs[self.save_key] = defaultdict(dict)
        result = super(ReplayCompose, self).__call__(force_apply=force_apply, **kwargs)
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, result[self.save_key])
        self.fill_applied(serialized)
        result[self.save_key] = serialized
        return result

    @staticmethod
    def replay(saved_augmentations: typing.Dict[str, typing.Any], **kwargs) -> typing.Dict[str, typing.Any]:
        augs = ReplayCompose._restore_for_replay(saved_augmentations)
        return augs(force_apply=True, **kwargs)

    @staticmethod
    def _restore_for_replay(
        transform_dict: typing.Dict[str, typing.Any], lambda_transforms: typing.Optional[dict] = None
    ) -> TransformType:
        """
        Args:
            lambda_transforms (dict): A dictionary that contains lambda transforms, that
            is instances of the Lambda class.
                This dictionary is required when you are restoring a pipeline that contains lambda transforms. Keys
                in that dictionary should be named same as `name` arguments in respective lambda transforms from
                a serialized pipeline.
        """
        applied = transform_dict["applied"]
        params = transform_dict["params"]
        lmbd = instantiate_nonserializable(transform_dict, lambda_transforms)
        if lmbd:
            transform = lmbd
        else:
            name = transform_dict["__class_fullname__"]
            args = {k: v for k, v in transform_dict.items() if k not in ["__class_fullname__", "applied", "params"]}
            cls = SERIALIZABLE_REGISTRY[name]
            if "transforms" in args:
                args["transforms"] = [
                    ReplayCompose._restore_for_replay(t, lambda_transforms=lambda_transforms)
                    for t in args["transforms"]
                ]
            transform = cls(**args)

        transform = typing.cast(BasicTransform, transform)
        if isinstance(transform, BasicTransform):
            transform.params = params
        transform.replay_mode = True
        transform.applied_in_replay = applied
        return transform

    def fill_with_params(self, serialized: dict, all_params: dict) -> None:
        params = all_params.get(serialized.get("id"))
        serialized["params"] = params
        del serialized["id"]
        for transform in serialized.get("transforms", []):
            self.fill_with_params(transform, all_params)

    def fill_applied(self, serialized: typing.Dict[str, typing.Any]) -> bool:
        if "transforms" in serialized:
            applied = [self.fill_applied(t) for t in serialized["transforms"]]
            serialized["applied"] = any(applied)
        else:
            serialized["applied"] = serialized.get("params") is not None
        return serialized["applied"]

    def _to_dict(self) -> typing.Dict[str, typing.Any]:
        dictionary = super(ReplayCompose, self)._to_dict()
        dictionary.update({"save_key": self.save_key})
        return dictionary


class Sequential(BaseCompose):
    """Sequentially applies all transforms to targets.

    Note:
        This transform is not intended to be a replacement for `Compose`. Instead, it should be used inside `Compose`
        the same way `OneOf` or `OneOrOther` are used. For instance, you can combine `OneOf` with `Sequential` to
        create an augmentation pipeline that contains multiple sequences of augmentations and applies one randomly
        chose sequence to input data (see the `Example` section for an example definition of such pipeline).

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        >>>    A.OneOf([
        >>>        A.Sequential([
        >>>            A.HorizontalFlip(p=0.5),
        >>>            A.ShiftScaleRotate(p=0.5),
        >>>        ]),
        >>>        A.Sequential([
        >>>            A.VerticalFlip(p=0.5),
        >>>            A.RandomBrightnessContrast(p=0.5),
        >>>        ]),
        >>>    ], p=1)
        >>> ])
    """

    def __init__(self, transforms: TransformsSeqType, p: float = 0.5):
        super().__init__(transforms, p)

    def __call__(self, *args, **data) -> typing.Dict[str, typing.Any]:
        for t in self.transforms:
            data = t(**data)
        return data
