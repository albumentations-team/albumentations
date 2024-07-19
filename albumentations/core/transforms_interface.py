from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Callable, Sequence, cast
from warnings import warn

import cv2
from pydantic import BaseModel, ConfigDict, Field


from albumentations.core.validation import ValidatedTransformMeta

from .serialization import Serializable, SerializableMeta, get_shortest_class_fullname
from albumentations.core.pydantic import ProbabilityType
from .types import (
    BoxInternalType,
    BoxType,
    ColorType,
    KeypointInternalType,
    KeypointType,
    Targets,
)
from .utils import format_args

import numpy as np

__all__ = ["BasicTransform", "DualTransform", "ImageOnlyTransform", "NoOp", "ReferenceBasedTransform"]


class Interpolation:
    def __init__(self, downscale: int = cv2.INTER_NEAREST, upscale: int = cv2.INTER_NEAREST):
        self.downscale = downscale
        self.upscale = upscale


class BaseTransformInitSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    always_apply: bool | None = Field(
        default=None,
        deprecated="Deprecated. Use `p=1` instead to always apply the transform",
    )
    p: ProbabilityType = 0.5


class CombinedMeta(SerializableMeta, ValidatedTransformMeta):
    pass


class BasicTransform(Serializable, metaclass=CombinedMeta):
    _targets: tuple[Targets, ...] | Targets  # targets that this transform can work on
    _available_keys: set[str]  # targets that this transform, as string, lower-cased
    _key2func: dict[
        str,
        Callable[..., Any],
    ]  # mapping for targets (plus additional targets) and methods for which they depend
    call_backup = None
    interpolation: int
    fill_value: ColorType
    mask_fill_value: ColorType | None
    # replay mode params
    deterministic: bool = False
    save_key = "replay"
    replay_mode = False
    applied_in_replay = False

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(self, p: float = 0.5, always_apply: bool | None = None):
        self.p = p
        if always_apply is not None:
            if always_apply:
                warn(
                    "always_apply is deprecated. Use `p=1` if you want to always apply the transform."
                    " self.p will be set to 1.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self.p = 1.0
            else:
                warn(
                    "always_apply is deprecated.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        self._additional_targets: dict[str, str] = {}
        # replay mode params
        self.params: dict[Any, Any] = {}
        self._key2func = {}
        self._set_keys()

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> Any:
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if force_apply or (random.random() < self.p):
            params = self.get_params()
            params = self.update_params_shape(params=params, data=kwargs)
            params = self.update_params(params, **kwargs)  # remove after move parameters like interpolation

            if self.targets_as_params:
                missing_keys = set(self.targets_as_params).difference(kwargs.keys())
                if missing_keys and not (missing_keys == {"image"} and "images" in kwargs):
                    msg = f"{self.__class__.__name__} requires {self.targets_as_params} missing keys: {missing_keys}"
                    raise ValueError(msg)

                targets_as_params = {k: kwargs.get(k, None) for k in self.targets_as_params}
                if missing_keys:  # here we expecting case when missing_keys == {"image"} and "images" in kwargs
                    targets_as_params["image"] = kwargs["images"][0]
                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)
            if self.deterministic:
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def apply_with_params(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Apply transforms with parameters."""
        res = {}
        for key, arg in kwargs.items():
            if key in self._key2func and arg is not None:
                target_function = self._key2func[key]
                res[key] = target_function(arg, **params)
            else:
                res[key] = arg
        return res

    def set_deterministic(self, flag: bool, save_key: str = "replay") -> BasicTransform:
        """Set transform to be deterministic."""
        if save_key == "params":
            msg = "params save_key is reserved"
            raise KeyError(msg)

        self.deterministic = flag
        if self.deterministic and self.targets_as_params:
            warn(
                self.get_class_fullname() + " could work incorrectly in ReplayMode for other input data"
                " because its' params depend on targets.",
                stacklevel=2,
            )
        self.save_key = save_key
        return self

    def __repr__(self) -> str:
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return f"{self.__class__.__name__}({format_args(state)})"

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform on image."""
        raise NotImplementedError

    def apply_to_images(self, images: np.ndarray, **params: Any) -> list[np.ndarray]:
        """Apply transform on images."""
        return [self.apply(image, **params) for image in images]

    def get_params(self) -> dict[str, Any]:
        """Returns parameters independent of input."""
        return {}

    def update_params_shape(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Updates parameters with input image shape."""
        # here we expects `image` or `images` in kwargs. it's checked at Compose._check_args
        shape = data["image"].shape if "image" in data else data["images"][0].shape
        params["shape"] = shape
        params.update({"cols": shape[1], "rows": shape[0]})
        return params

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Returns parameters dependent on input."""
        return params

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        # mapping for targets and methods for which they depend
        # for example:
        # >>  {"image": self.apply}
        # >>  {"masks": self.apply_to_masks}
        raise NotImplementedError

    def _set_keys(self) -> None:
        """Set _available_keys."""
        if not hasattr(self, "_targets"):
            self._available_keys = set()
        else:
            self._available_keys = {
                target.value.lower()
                for target in (self._targets if isinstance(self._targets, tuple) else [self._targets])
            }
        self._available_keys.update(self.targets.keys())
        self._key2func = {key: self.targets[key] for key in self._available_keys if key in self.targets}

    @property
    def available_keys(self) -> set[str]:
        """Returns set of available keys."""
        return self._available_keys

    def update_params(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Update parameters with transform specific params.
        This method is deprecated, use:
        - `get_params` for transform specific params like interpolation and
        - `update_params_shape` for data like shape.
        """
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        if hasattr(self, "mask_fill_value"):
            params["mask_fill_value"] = self.mask_fill_value

        # here we expects `image` or `images` in kwargs. it's checked at Compose._check_args
        shape = kwargs["image"].shape if "image" in kwargs else kwargs["images"][0].shape
        params["shape"] = shape
        params.update({"cols": shape[1], "rows": shape[0]})
        return params

    def add_targets(self, additional_targets: dict[str, str]) -> None:
        """Add targets to transform them the same way as one of existing targets.
        ex: {'target_image': 'image'}
        ex: {'obj1_mask': 'mask', 'obj2_mask': 'mask'}
        by the way you must have at least one object with key 'image'

        Args:
            additional_targets (dict): keys - new target name, values - old target name. ex: {'image2': 'image'}

        """
        for k, v in additional_targets.items():
            if k in self._additional_targets and v != self._additional_targets[k]:
                raise ValueError(
                    f"Trying to overwrite existed additional targets. "
                    f"Key={k} Exists={self._additional_targets[k]} New value: {v}",
                )
            if v in self._available_keys:
                self._additional_targets[k] = v
                self._key2func[k] = self.targets[v]
                self._available_keys.add(k)

    @property
    def targets_as_params(self) -> list[str]:
        """Targets used to get params dependent on targets.
        This is used to check input has all required targets.
        """
        return []

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        """This method is deprecated.
        Use `get_params_dependent_on_data` instead.
        Returns parameters dependent on targets.
        Dependent target is defined in `self.targets_as_params`
        """
        return {}

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls) -> bool:
        return True

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Returns names of arguments that are used in __init__ method of the transform."""
        msg = (
            f"Class {self.get_class_fullname()} is not serializable because the `get_transform_init_args_names` "
            "method is not implemented"
        )
        raise NotImplementedError(msg)

    def get_base_init_args(self) -> dict[str, Any]:
        """Returns base init args - p"""
        return {"p": self.p}

    def get_transform_init_args(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def to_dict_private(self) -> dict[str, Any]:
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())

        return state

    def get_dict_with_id(self) -> dict[str, Any]:
        d = self.to_dict_private()
        d["id"] = id(self)
        return d


class DualTransform(BasicTransform):
    """A base class for transformations that should be applied both to an image and its corresponding properties
    such as masks, bounding boxes, and keypoints. This class ensures that when a transform is applied to an image,
    all associated entities are transformed accordingly to maintain consistency between the image and its annotations.

    Properties:
        targets (dict[str, Callable[..., Any]]): Defines the types of targets (e.g., image, mask, bboxes, keypoints)
            that the transform should be applied to and maps them to the corresponding methods.

    Methods:
        apply_to_bbox(bbox: BoxInternalType, *args: Any, **params: Any) -> BoxInternalType:
            Applies the transform to a single bounding box. Should be implemented in the subclass.

        apply_to_keypoint(keypoint: KeypointInternalType, *args: Any, **params: Any) -> KeypointInternalType:
            Applies the transform to a single keypoint. Should be implemented in the subclass.

        apply_to_bboxes(bboxes: Sequence[BoxType], *args: Any, **params: Any) -> Sequence[BoxType]:
            Applies the transform to a list of bounding boxes. Delegates to `apply_to_bbox` for each bounding box.

        apply_to_keypoints(keypoints: Sequence[KeypointType], *args: Any, **params: Any) -> Sequence[KeypointType]:
            Applies the transform to a list of keypoints. Delegates to `apply_to_keypoint` for each keypoint.

        apply_to_mask(mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
            Applies the transform specifically to a single mask.

        apply_to_masks(masks: Sequence[np.ndarray], **params: Any) -> list[np.ndarray]:
            Applies the transform to a list of masks. Delegates to `apply_to_mask` for each mask.

    Note:
        This class is intended to be subclassed and should not be used directly. Subclasses are expected to
        implement the specific logic for each type of target (e.g., image, mask, bboxes, keypoints) in the
        corresponding `apply_to_*` methods.

    """

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }

    def apply_to_bbox(self, bbox: BoxInternalType, *args: Any, **params: Any) -> BoxInternalType:
        msg = f"Method apply_to_bbox is not implemented in class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, *args: Any, **params: Any) -> KeypointInternalType:
        msg = f"Method apply_to_keypoint is not implemented in class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def apply_to_global_label(self, label: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        msg = f"Method apply_to_global_label is not implemented in class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def apply_to_bboxes(self, bboxes: Sequence[BoxType], *args: Any, **params: Any) -> Sequence[BoxType]:
        return [
            self.apply_to_bbox(cast(BoxInternalType, tuple(cast(BoxInternalType, bbox[:4]))), **params)
            + tuple(bbox[4:])
            for bbox in bboxes
        ]

    def apply_to_keypoints(
        self,
        keypoints: Sequence[KeypointType],
        *args: Any,
        **params: Any,
    ) -> Sequence[KeypointType]:
        return [
            self.apply_to_keypoint(cast(KeypointInternalType, tuple(keypoint[:4])), **params) + tuple(keypoint[4:])
            for keypoint in keypoints
        ]

    def apply_to_mask(self, mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(mask, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_masks(self, masks: Sequence[np.ndarray], **params: Any) -> list[np.ndarray]:
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def apply_to_global_labels(self, labels: Sequence[np.ndarray], **params: Any) -> list[np.ndarray]:
        return [self.apply_to_global_label(label, **params) for label in labels]


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    _targets = Targets.IMAGE

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {"image": self.apply, "images": self.apply_to_images}


class NoOp(DualTransform):
    """Identity transform (does nothing).

    Targets:
        image, mask, bboxes, keypoints, global_label
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS, Targets.GLOBAL_LABEL)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        return keypoint

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return bbox

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return img

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        return mask

    def apply_to_global_label(self, label: np.ndarray, **params: Any) -> np.ndarray:
        return label

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


class ReferenceBasedTransform(DualTransform):
    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "global_label": self.apply_to_global_label,
            "image": self.apply,
            "mask": self.apply_to_mask,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }
