import random
import warnings
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Union, cast

import cv2
import numpy as np

from albumentations import random_utils

from .bbox_utils import BboxParams, BboxProcessor
from .hub_mixin import HubMixin
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
    "TransformType",
    "TransformsSeqType",
    "SelectiveChannelTransform",
]

NUM_ONEOF_TRANSFORMS = 2
REPR_INDENT_STEP = 2

TransformType = Union[BasicTransform, "BaseCompose"]
TransformsSeqType = List[TransformType]

AVAILABLE_KEYS = ("image", "mask", "masks", "bboxes", "keypoints", "global_label")


def get_transforms_dict(transforms: TransformsSeqType) -> Dict[int, BasicTransform]:
    result = {}
    for transform in transforms:
        if isinstance(transform, BaseCompose):
            result.update(get_transforms_dict(transform.transforms))
        else:
            result[id(transform)] = transform
    return result


class BaseCompose(Serializable):
    _transforms_dict: Optional[Dict[int, BasicTransform]] = None

    def __init__(self, transforms: TransformsSeqType, p: float):
        if isinstance(transforms, (BaseCompose, BasicTransform)):
            warnings.warn(
                "transforms is single transform, but a sequence is expected! Transform will be wrapped into list.",
                stacklevel=2,
            )
            transforms = [transforms]

        self.transforms = transforms
        self.p = p

        self.replay_mode = False
        self.applied_in_replay = False
        self._additional_targets: Dict[str, str] = {}
        self._available_keys: Set[str] = set()
        self.processors: Dict[str, Union[BboxProcessor, KeypointsProcessor]] = {}
        self._set_keys()

    def __iter__(self) -> Iterator[TransformType]:
        return iter(self.transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __call__(self, *args: Any, **data: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, item: int) -> TransformType:
        return self.transforms[item]

    def __repr__(self) -> str:
        return self.indented_repr()

    @property
    def additional_targets(self) -> Dict[str, str]:
        return self._additional_targets

    @property
    def available_keys(self) -> Set[str]:
        return self._available_keys

    def indented_repr(self, indent: int = REPR_INDENT_STEP) -> str:
        args = {k: v for k, v in self.to_dict_private().items() if not (k.startswith("__") or k == "transforms")}
        repr_string = self.__class__.__name__ + "(["
        for t in self.transforms:
            repr_string += "\n"
            t_repr = t.indented_repr(indent + REPR_INDENT_STEP) if hasattr(t, "indented_repr") else repr(t)
            repr_string += " " * indent + t_repr + ","
        repr_string += "\n" + " " * (indent - REPR_INDENT_STEP) + f"], {format_args(args)})"
        return repr_string

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls) -> bool:
        return True

    def to_dict_private(self) -> Dict[str, Any]:
        return {
            "__class_fullname__": self.get_class_fullname(),
            "p": self.p,
            "transforms": [t.to_dict_private() for t in self.transforms],
        }

    def get_dict_with_id(self) -> Dict[str, Any]:
        return {
            "__class_fullname__": self.get_class_fullname(),
            "id": id(self),
            "params": None,
            "transforms": [t.get_dict_with_id() for t in self.transforms],
        }

    def add_targets(self, additional_targets: Optional[Dict[str, str]]) -> None:
        if additional_targets:
            for k, v in additional_targets.items():
                if k in self._additional_targets and v != self._additional_targets[k]:
                    raise ValueError(
                        f"Trying to overwrite existed additional targets. "
                        f"Key={k} Exists={self._additional_targets[k]} New value: {v}",
                    )
            self._additional_targets.update(additional_targets)
            for t in self.transforms:
                t.add_targets(additional_targets)
            for proc in self.processors.values():
                proc.add_targets(additional_targets)
        self._set_keys()

    def _set_keys(self) -> None:
        """Set _available_keys"""
        self._available_keys.update(self._additional_targets.keys())
        for t in self.transforms:
            self._available_keys.update(t.available_keys)
            if hasattr(t, "targets_as_params"):
                self._available_keys.update(t.targets_as_params)
        if self.processors:
            self._available_keys.update(["labels"])
            for proc in self.processors.values():
                if proc.default_data_name not in self._available_keys:  # if no transform to process this data
                    warnings.warn(
                        f"Got processor for {proc.default_data_name}, but no transform to process it.",
                        stacklevel=2,
                    )
                self._available_keys.update(proc.data_fields)
                if proc.params.label_fields:
                    self._available_keys.update(proc.params.label_fields)

    def set_deterministic(self, flag: bool, save_key: str = "replay") -> None:
        for t in self.transforms:
            t.set_deterministic(flag, save_key)


class Compose(BaseCompose, HubMixin):
    """Compose transforms and handle all transformations regarding bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
        is_check_shapes (bool): If True shapes consistency of images/mask/masks would be checked on each call. If you
            would like to disable this check - pass False (do it only if you are sure in your data consistency).
        return_params (bool): if True returns params of each applied transform
        save_key (str): key to save applied params, default is 'applied_params'

    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: Optional[Union[Dict[str, Any], "BboxParams"]] = None,
        keypoint_params: Optional[Union[Dict[str, Any], "KeypointParams"]] = None,
        additional_targets: Optional[Dict[str, str]] = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        return_params: bool = False,
        save_key: str = "applied_params",
    ):
        super().__init__(transforms, p)

        if bbox_params:
            if isinstance(bbox_params, dict):
                b_params = BboxParams(**bbox_params)
            elif isinstance(bbox_params, BboxParams):
                b_params = bbox_params
            else:
                msg = "unknown format of bbox_params, please use `dict` or `BboxParams`"
                raise ValueError(msg)
            self.processors["bboxes"] = BboxProcessor(b_params)

        if keypoint_params:
            if isinstance(keypoint_params, dict):
                k_params = KeypointParams(**keypoint_params)
            elif isinstance(keypoint_params, KeypointParams):
                k_params = keypoint_params
            else:
                msg = "unknown format of keypoint_params, please use `dict` or `KeypointParams`"
                raise ValueError(msg)
            self.processors["keypoints"] = KeypointsProcessor(k_params)

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)
        if not self.transforms:  # if no transforms -> do nothing, all keys will be available
            self._available_keys.update(AVAILABLE_KEYS)

        self.is_check_args = True
        self._disable_check_args_for_transforms(self.transforms)

        self.is_check_shapes = is_check_shapes
        self._check_each_transform = tuple(  # processors that checks after each transform
            proc for proc in self.processors.values() if getattr(proc.params, "check_each_transform", False)
        )

        self.return_params = return_params
        if return_params:
            self.save_key = save_key
            self._available_keys.add(save_key)
            self._transforms_dict = get_transforms_dict(self.transforms)
            self.set_deterministic(True, save_key=save_key)

    @staticmethod
    def _disable_check_args_for_transforms(transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                Compose._disable_check_args_for_transforms(transform.transforms)
            if isinstance(transform, Compose):
                transform.disable_check_args_private()

    def disable_check_args_private(self) -> None:
        self.is_check_args = False

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> Dict[str, Any]:
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)

        if not isinstance(force_apply, (bool, int)):
            msg = "force_apply must have bool or int type"
            raise TypeError(msg)

        if self.return_params:
            data[self.save_key] = OrderedDict()

        need_to_run = force_apply or random.random() < self.p
        if not need_to_run:
            return data

        self.preprocess(data)

        for t in self.transforms:
            data = t(**data)

            if self._check_each_transform:
                data = self._check_data_post_transform(data)

        return self.postprocess(data)

    def run_with_params(self, *, params: Dict[int, Dict[str, Any]], **data: Any) -> Dict[str, Any]:
        """Run transforms with given parameters. Available only for Compose with `return_params=True`."""
        if self._transforms_dict is None:
            raise RuntimeError("`run_with_params` is not available for Compose with `return_params=False`.")

        self.preprocess(data)

        for tr_id, param in params.items():
            tr = self._transforms_dict[tr_id]
            data = tr.apply_with_params(param, **data)

            if self._check_each_transform:
                data = self._check_data_post_transform(data)

        return self.postprocess(data)

    def preprocess(self, data: Any) -> None:
        if self.is_check_args:
            self._check_args(**data)
        for p in self.processors.values():
            p.ensure_data_valid(data)
        for p in self.processors.values():
            p.preprocess(data)

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = Compose._make_targets_contiguous(data)  # ensure output targets are contiguous
        for p in self.processors.values():
            p.postprocess(data)
        return data

    def _check_data_post_transform(self, data: Any) -> Dict[str, Any]:
        rows, cols = get_shape(data["image"])

        for p in self._check_each_transform:
            for data_name in data:
                if data_name in p.data_fields or (
                    data_name in self._additional_targets and self._additional_targets[data_name] in p.data_fields
                ):
                    data[data_name] = p.filter(data[data_name], rows, cols)
        return data

    def to_dict_private(self) -> Dict[str, Any]:
        dictionary = super().to_dict_private()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params.to_dict_private() if bbox_processor else None,
                "keypoint_params": (keypoints_processor.params.to_dict_private() if keypoints_processor else None),
                "additional_targets": self.additional_targets,
                "is_check_shapes": self.is_check_shapes,
            },
        )
        return dictionary

    def get_dict_with_id(self) -> Dict[str, Any]:
        dictionary = super().get_dict_with_id()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params.to_dict_private() if bbox_processor else None,
                "keypoint_params": (keypoints_processor.params.to_dict_private() if keypoints_processor else None),
                "additional_targets": self.additional_targets,
                "params": None,
                "is_check_shapes": self.is_check_shapes,
            },
        )
        return dictionary

    def _check_args(self, **kwargs: Any) -> None:
        checked_single = ["image", "mask"]
        checked_multi = ["masks"]
        check_bbox_param = ["bboxes"]
        check_keypoints_param = ["keypoints"]
        shapes = []

        for data_name, data in kwargs.items():
            if data_name not in self._available_keys and data_name not in ["mask", "masks"]:
                msg = f"Key {data_name} is not in available keys."
                raise ValueError(msg)
            internal_data_name = self._additional_targets.get(data_name, data_name)
            if internal_data_name in checked_single:
                if not isinstance(data, np.ndarray):
                    raise TypeError(f"{data_name} must be numpy array type")
                shapes.append(data.shape[:2])
            if internal_data_name in checked_multi and data is not None and len(data):
                if not isinstance(data[0], np.ndarray):
                    raise TypeError(f"{data_name} must be list of numpy arrays")
                shapes.append(data[0].shape[:2])
            if internal_data_name in check_bbox_param and self.processors.get("bboxes") is None:
                msg = "bbox_params must be specified for bbox transformations"
                raise ValueError(msg)

            if internal_data_name in check_keypoints_param and self.processors.get("keypoints") is None:
                msg = "keypoints_params must be specified for keypoint transformations"
                raise ValueError(msg)

        if self.is_check_shapes and shapes and shapes.count(shapes[0]) != len(shapes):
            msg = (
                "Height and Width of image, mask or masks should be equal. You can disable shapes check "
                "by setting a parameter is_check_shapes=False of Compose class (do it only if you are sure "
                "about your data consistency)."
            )
            raise ValueError(msg)

    @staticmethod
    def _make_targets_contiguous(data: Any) -> Dict[str, Any]:
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                result[key] = np.ascontiguousarray(value)
            else:
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
        super().__init__(transforms, p)
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> Dict[str, Any]:
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
        super().__init__(transforms, p)
        self.n = n
        self.replace = replace
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *arg: Any, force_apply: bool = False, **data: Any) -> Dict[str, Any]:
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if self.transforms_ps and (force_apply or random.random() < self.p):
            idx = random_utils.choice(len(self.transforms), size=self.n, replace=self.replace, p=self.transforms_ps)
            for i in idx:
                t = self.transforms[i]
                data = t(force_apply=True, **data)
        return data

    def to_dict_private(self) -> Dict[str, Any]:
        dictionary = super().to_dict_private()
        dictionary.update({"n": self.n, "replace": self.replace})
        return dictionary


class OneOrOther(BaseCompose):
    """Select one or another transform to apply. Selected transform will be called with `force_apply=True`."""

    def __init__(
        self,
        first: Optional[TransformType] = None,
        second: Optional[TransformType] = None,
        transforms: Optional[TransformsSeqType] = None,
        p: float = 0.5,
    ):
        if transforms is None:
            if first is None or second is None:
                msg = "You must set both first and second or set transforms argument."
                raise ValueError(msg)
            transforms = [first, second]
        super().__init__(transforms, p)
        if len(self.transforms) != NUM_ONEOF_TRANSFORMS:
            warnings.warn("Length of transforms is not equal to 2.", stacklevel=2)

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> Dict[str, Any]:
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if random.random() < self.p:
            return self.transforms[0](force_apply=True, **data)

        return self.transforms[-1](force_apply=True, **data)


class SelectiveChannelTransform(BaseCompose):
    """A transformation class to apply specified transforms to selected channels of an image.

    This class extends BaseCompose to allow selective application of transformations to
    specified image channels. It extracts the selected channels, applies the transformations,
    and then reinserts the transformed channels back into their original positions in the image.

    Parameters:
        transforms (TransformsSeqType):
            A sequence of transformations (from Albumentations) to be applied to the specified channels.
        channels (Sequence[int]):
            A sequence of integers specifying the indices of the channels to which the transforms should be applied.
        p (float):
            Probability that the transform will be applied; the default is 1.0 (always apply).

    Methods:
        __call__(*args, **kwargs):
            Applies the transforms to the image according to the specified channels.
            The input data should include 'image' key with the image array.

    Returns:
        Dict[str, Any]: The transformed data dictionary, which includes the transformed 'image' key.
    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        channels: Sequence[int] = (0, 1, 2),
        p: float = 1.0,
    ) -> None:
        super().__init__(transforms, p)
        self.channels = channels

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> Dict[str, Any]:
        if force_apply or random.random() < self.p:
            image = data["image"]

            selected_channels = image[:, :, self.channels]
            sub_image = np.ascontiguousarray(selected_channels)

            for t in self.transforms:
                sub_image = t(image=sub_image)["image"]

            transformed_channels = cv2.split(sub_image)
            output_img = image.copy()

            for idx, channel in zip(self.channels, transformed_channels):
                output_img[:, :, idx] = channel

            data["image"] = np.ascontiguousarray(output_img)

        return data


class ReplayCompose(Compose):
    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: Optional[Union[Dict[str, Any], "BboxParams"]] = None,
        keypoint_params: Optional[Union[Dict[str, Any], "KeypointParams"]] = None,
        additional_targets: Optional[Dict[str, str]] = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        save_key: str = "replay",
    ):
        super().__init__(transforms, bbox_params, keypoint_params, additional_targets, p, is_check_shapes)
        self.set_deterministic(True, save_key=save_key)
        self.save_key = save_key
        self._available_keys.add(save_key)

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> Dict[str, Any]:
        kwargs[self.save_key] = defaultdict(dict)
        result = super().__call__(force_apply=force_apply, **kwargs)
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, result[self.save_key])
        self.fill_applied(serialized)
        result[self.save_key] = serialized
        return result

    @staticmethod
    def replay(saved_augmentations: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        augs = ReplayCompose._restore_for_replay(saved_augmentations)
        return augs(force_apply=True, **kwargs)

    @staticmethod
    def _restore_for_replay(
        transform_dict: Dict[str, Any],
        lambda_transforms: Optional[Dict[str, Any]] = None,
    ) -> TransformType:
        """Args:
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

        transform = cast(BasicTransform, transform)
        if isinstance(transform, BasicTransform):
            transform.params = params
        transform.replay_mode = True
        transform.applied_in_replay = applied
        return transform

    def fill_with_params(self, serialized: Dict[str, Any], all_params: Any) -> None:
        params = all_params.get(serialized.get("id"))
        serialized["params"] = params
        del serialized["id"]
        for transform in serialized.get("transforms", []):
            self.fill_with_params(transform, all_params)

    def fill_applied(self, serialized: Dict[str, Any]) -> bool:
        if "transforms" in serialized:
            applied = [self.fill_applied(t) for t in serialized["transforms"]]
            serialized["applied"] = any(applied)
        else:
            serialized["applied"] = serialized.get("params") is not None
        return serialized["applied"]

    def to_dict_private(self) -> Dict[str, Any]:
        dictionary = super().to_dict_private()
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

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> Dict[str, Any]:
        if self.replay_mode or force_apply or random.random() < self.p:
            for t in self.transforms:
                data = t(**data)
        return data
