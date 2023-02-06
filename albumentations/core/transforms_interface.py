from __future__ import absolute_import

import random
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import cv2
import numpy as np
from numpy import typing as npt

from .serialization import Serializable, get_shortest_class_fullname
from .utils import format_args

__all__ = [
    "to_tuple",
    "BasicTransform",
    "DualTransform",
    "ImageOnlyTransform",
    "NoOp",
    "BoxType",
    "BatchInternalType",
    "BBoxesInternalType",
    "BoxesArray",
    "KeypointsInternalType",
    "KeypointsArray",
    "ImageColorType",
    "ScaleFloatType",
    "ScaleIntType",
    "ImageColorType",
]

NumType = Union[int, float, np.ndarray]
BoxType = Union[Tuple[float, float, float, float], Tuple[float, float, float, float, Any]]
BoxesArray = Annotated[npt.NDArray, Literal["N", 4]]
KeypointsArray = Annotated[npt.NDArray, Literal["N", 4]]
ImageColorType = Union[float, Sequence[float]]

ScaleFloatType = Union[float, Tuple[float, float]]
ScaleIntType = Union[int, Tuple[int, int]]

FillValueType = Optional[Union[int, float, Sequence[int], Sequence[float]]]


@dataclass
class BatchInternalType:
    array: np.ndarray

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key, value):
        raise NotImplementedError

    @abstractmethod
    def check_consistency(self):
        raise NotImplementedError


@dataclass
class BBoxesInternalType(BatchInternalType):
    array: BoxesArray
    targets: List[Any] = field(default_factory=list)

    def __post_init__(self):
        self.check_consistency()

    def __len__(self):
        assert len(self.array) == len(self.targets)
        return len(self.array)

    def __setattr__(self, key, value):
        if key == "bboxes":
            self.assert_np_bboxes_format(value)
        super().__setattr__(key, value)

    def __getitem__(self, item) -> "BBoxesInternalType":
        _bboxes = self.array[item].astype(float)
        if isinstance(item, int):
            _bboxes = _bboxes[np.newaxis, ...]
            _target = [self.targets[item]] if self.targets else []
        else:
            _target = [self.targets[i] for i in item] if self.targets is not None else None
        return BBoxesInternalType(array=_bboxes, targets=_target)

    def __setitem__(self, idx, value: "BBoxesInternalType"):
        self.array[idx] = value.array
        if isinstance(idx, int):
            self.targets[idx] = value.targets[0]
        else:
            for i, target in zip(idx, value.targets):
                self.targets[i] = target

    @staticmethod
    def assert_np_bboxes_format(bboxes: np.ndarray):  # noqa
        if not isinstance(bboxes, np.ndarray):
            raise TypeError("Bboxes should be a numpy ndarray.")
        if len(bboxes):
            if not (len(bboxes.shape) == 2 and bboxes.shape[-1] == 4):
                raise ValueError(
                    "An array of bboxes should be 2 dimension, and the last dimension must has 4 elements."
                )

    def check_consistency(self):
        if len(self.array) != len(self.targets):
            raise ValueError(
                "The amount of bboxes and additional targets should be the same. "
                f"Get {len(self.array)} bboxes and {len(self.targets)} additional targets."
            )
        self.assert_np_bboxes_format(self.array)


@dataclass
class KeypointsInternalType(BatchInternalType):
    array: KeypointsArray
    targets: List[Any] = field(default_factory=list)

    def __post_init__(self):
        self.check_consistency()

    def __len__(self):
        assert len(self.array) == len(self.targets)
        return len(self.array)

    def __setattr__(self, key, value):
        if key == "keypoints":
            self.assert_np_keypoints_format(value)
        super().__setattr__(key, value)

    def __getitem__(self, idx) -> "KeypointsInternalType":
        _kps = self.array[idx].astype(float)
        if isinstance(idx, int):
            _kps = _kps[np.newaxis, ...]
            _target = [self.targets[idx]] if self.targets else []
        else:
            _target = [self.targets[i] for i in idx] if self.targets is not None else None
        return KeypointsInternalType(array=_kps, targets=_target)

    def __setitem__(self, idx, value: "KeypointsInternalType"):
        self.array[idx] = value.array
        if isinstance(idx, int):
            self.targets[idx] = value.targets[0]
        else:
            for i, target in zip(idx, value.targets):
                self.targets[i] = target

    @staticmethod
    def assert_np_keypoints_format(keypoints: np.ndarray):  # noqa
        if not isinstance(keypoints, np.ndarray):
            raise TypeError("Bboxes should be a numpy ndarray.")
        if len(keypoints):
            if not (len(keypoints.shape) == 2 and 2 <= keypoints.shape[-1] <= 4):
                raise ValueError(
                    "An array of keypoints should be 2 dimension, "
                    "and the last dimension must has at least 2 elements at most 4 elements. "
                )

    def check_consistency(self):
        if self.targets is not None and len(self.array) != len(self.targets):
            raise ValueError(
                "The amount of keypoints and additional targets should be the same. "
                f"Get {len(self.array)} keypoints and {len(self.targets)} additional targets."
            )
        self.assert_np_keypoints_format(self.array)


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        if len(param) != 2:
            raise ValueError("to_tuple expects 1 or 2 values")
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)


class BasicTransform(Serializable):
    call_backup = None
    interpolation: Any
    fill_value: Any
    mask_fill_value: Any

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.p = p
        self.always_apply = always_apply
        self._additional_targets: Dict[str, str] = {}

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params: Dict[Any, Any] = {}
        self.replay_mode = False
        self.applied_in_replay = False

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(key in kwargs for key in self.targets_as_params), "{} requires {}".format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname() + " could work incorrectly in ReplayMode for other input data"
                        " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def apply_with_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def set_deterministic(self, flag: bool, save_key: str = "replay") -> "BasicTransform":
        assert save_key != "params", "params save_key is reserved"
        self.deterministic = flag
        self.save_key = save_key
        return self

    def __repr__(self) -> str:
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return "{name}({args})".format(name=self.__class__.__name__, args=format_args(state))

    def _get_target_function(self, key: str) -> Callable:
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, key)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> Dict:
        return {}

    @property
    def targets(self) -> Dict[str, Callable]:
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        if hasattr(self, "mask_fill_value"):
            params["mask_fill_value"] = self.mask_fill_value
        params.update({"cols": kwargs["image"].shape[1], "rows": kwargs["image"].shape[0]})
        return params

    @property
    def target_dependence(self) -> Dict:
        return {}

    def add_targets(self, additional_targets: Dict[str, str]):
        """Add targets to transform them the same way as one of existing targets
        ex: {'target_image': 'image'}
        ex: {'obj1_mask': 'mask', 'obj2_mask': 'mask'}
        by the way you must have at least one object with key 'image'

        Args:
            additional_targets (dict): keys - new target name, values - old target name. ex: {'image2': 'image'}
        """
        self._additional_targets = additional_targets

    @property
    def targets_as_params(self) -> List[str]:
        return []

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "Method get_params_dependent_on_targets is not implemented in class " + self.__class__.__name__
        )

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls):
        return True

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        raise NotImplementedError(
            "Class {name} is not serializable because the `get_transform_init_args_names` method is not "
            "implemented".format(name=self.get_class_fullname())
        )

    def get_base_init_args(self) -> Dict[str, Any]:
        return {"always_apply": self.always_apply, "p": self.p}

    def get_transform_init_args(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def _to_dict(self) -> Dict[str, Any]:
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state

    def get_dict_with_id(self) -> Dict[str, Any]:
        d = self._to_dict()
        d["id"] = id(self)
        return d


class DualTransform(BasicTransform):
    """Transform for segmentation task."""

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }

    def apply_to_bbox(self, bbox: BBoxesInternalType, **params) -> BBoxesInternalType:
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint: KeypointsInternalType, **params) -> KeypointsInternalType:
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)

    def apply_to_bboxes(self, bboxes: BBoxesInternalType, **params) -> BBoxesInternalType:
        bbox: BBoxesInternalType
        for i, bbox in enumerate(bboxes):  # type: ignore[arg-type]
            bboxes[i] = self.apply_to_bbox(bbox, **params)
        return bboxes

    def apply_to_keypoints(self, keypoints: KeypointsInternalType, **params) -> KeypointsInternalType:
        kpt: KeypointsInternalType
        for i, kpt in enumerate(keypoints):  # type: ignore[arg-type]
            keypoints[i] = self.apply_to_keypoint(kpt, **params)
        return keypoints

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_masks(self, masks: Sequence[np.ndarray], **params) -> List[np.ndarray]:
        return [self.apply_to_mask(mask, **params) for mask in masks]


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    @property
    def targets(self) -> Dict[str, Callable]:
        return {"image": self.apply}


class NoOp(DualTransform):
    """Does nothing"""

    def apply_to_keypoint(self, keypoint: KeypointsInternalType, **params) -> KeypointsInternalType:
        return keypoint

    def apply_to_bbox(self, bbox: BBoxesInternalType, **params) -> BBoxesInternalType:
        return bbox

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return img

    def get_transform_init_args_names(self) -> Tuple:
        return ()
