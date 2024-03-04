import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import cv2
import numpy as np

from .serialization import Serializable, get_shortest_class_fullname
from .types import (
    BoxInternalType,
    BoxType,
    ColorType,
    KeypointInternalType,
    KeypointType,
    ScalarType,
    ScaleType,
    Targets,
)
from .utils import format_args

__all__ = ["to_tuple", "BasicTransform", "DualTransform", "ImageOnlyTransform", "NoOp", "ReferenceBasedTransform"]

PAIR = 2


def to_tuple(
    param: ScaleType,
    low: Optional[ScaleType] = None,
    bias: Optional[ScalarType] = None,
) -> Union[Tuple[int, int], Tuple[float, float]]:
    """Convert input argument to a min-max tuple.

    Args:
        param: Input value which could be a scalar or a sequence of exactly 2 scalars.
        low: Second element of the tuple, provided as an optional argument for when `param` is a scalar.
        bias: An offset added to both elements of the tuple.

    Returns:
        A tuple of two scalars, optionally adjusted by `bias`.
        Raises ValueError for invalid combinations or types of arguments.

    """
    # Validate mutually exclusive arguments
    if low is not None and bias is not None:
        msg = "Arguments 'low' and 'bias' cannot be used together."
        raise ValueError(msg)

    if isinstance(param, Sequence) and len(param) == PAIR:
        min_val, max_val = min(param), max(param)

    # Handle scalar input
    elif isinstance(param, (int, float)):
        if isinstance(low, (int, float)):
            # Use low and param to create a tuple
            min_val, max_val = (low, param) if low < param else (param, low)
        else:
            # Create a symmetric tuple around 0
            min_val, max_val = -param, param
    else:
        msg = "Argument 'param' must be either a scalar or a sequence of 2 elements."
        raise ValueError(msg)

    # Apply bias if provided
    if bias is not None:
        return (bias + min_val, bias + max_val)

    return min_val, max_val


class Interpolation:
    def __init__(self, downscale: int = cv2.INTER_NEAREST, upscale: int = cv2.INTER_NEAREST):
        self.downscale = downscale
        self.upscale = upscale


class BasicTransform(Serializable):
    call_backup = None
    interpolation: Union[int, Interpolation]
    fill_value: ColorType
    mask_fill_value: Optional[ColorType]

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

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> Any:
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if force_apply or self.always_apply or (random.random() < self.p):
            params = self.get_params()

            if self.targets_as_params:
                if not all(key in kwargs for key in self.targets_as_params):
                    msg = f"{self.__class__.__name__} requires {self.targets_as_params}"
                    raise ValueError(msg)

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

    def apply_with_params(self, params: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
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
        if save_key == "params":
            msg = "params save_key is reserved"
            raise KeyError(msg)

        self.deterministic = flag
        self.save_key = save_key
        return self

    def __repr__(self) -> str:
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return f"{self.__class__.__name__}({format_args(state)})"

    def _get_target_function(self, key: str) -> Callable[..., Any]:
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, key)

        return self.targets.get(transform_key, lambda x, **p: x)

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {}

    @property
    def targets(self) -> Dict[str, Callable[..., Any]]:
        # you must specify targets in subclass
        # foe example:
        # >>  ('image', 'mask')
        # >>  ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        if hasattr(self, "mask_fill_value"):
            params["mask_fill_value"] = self.mask_fill_value
        params.update({"cols": kwargs["image"].shape[1], "rows": kwargs["image"].shape[0]})
        return params

    @property
    def target_dependence(self) -> Dict[str, Any]:
        return {}

    def add_targets(self, additional_targets: Dict[str, str]) -> None:
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
    def is_serializable(cls) -> bool:
        return True

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        msg = f"Class {self.get_class_fullname()} is not serializable because the `get_transform_init_args_names` "
        "method is not implemented"
        raise NotImplementedError(msg)

    def get_base_init_args(self) -> Dict[str, Any]:
        return {"always_apply": self.always_apply, "p": self.p}

    def get_transform_init_args(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def to_dict_private(self) -> Dict[str, Any]:
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state

    def get_dict_with_id(self) -> Dict[str, Any]:
        d = self.to_dict_private()
        d["id"] = id(self)
        return d


class DualTransform(BasicTransform):
    """A base class for transformations that should be applied both to an image and its corresponding properties
    such as masks, bounding boxes, and keypoints. This class ensures that when a transform is applied to an image,
    all associated entities are transformed accordingly to maintain consistency between the image and its annotations.

    Properties:
        targets (Dict[str, Callable[..., Any]]): Defines the types of targets (e.g., image, mask, bboxes, keypoints)
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

        apply_to_masks(masks: Sequence[np.ndarray], **params: Any) -> List[np.ndarray]:
            Applies the transform to a list of masks. Delegates to `apply_to_mask` for each mask.

    Note:
        This class is intended to be subclassed and should not be used directly. Subclasses are expected to
        implement the specific logic for each type of target (e.g., image, mask, bboxes, keypoints) in the
        corresponding `apply_to_*` methods.

    """

    @property
    def targets(self) -> Dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
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
        self, keypoints: Sequence[KeypointType], *args: Any, **params: Any
    ) -> Sequence[KeypointType]:
        return [
            self.apply_to_keypoint(cast(KeypointInternalType, tuple(keypoint[:4])), **params) + tuple(keypoint[4:])
            for keypoint in keypoints
        ]

    def apply_to_mask(self, mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(mask, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_masks(self, masks: Sequence[np.ndarray], **params: Any) -> List[np.ndarray]:
        return [self.apply_to_mask(mask, **params) for mask in masks]

    def apply_to_global_labels(self, labels: Sequence[np.ndarray], **params: Any) -> List[np.ndarray]:
        return [self.apply_to_global_label(label, **params) for label in labels]


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    _targets = Targets.IMAGE

    @property
    def targets(self) -> Dict[str, Callable[..., Any]]:
        return {"image": self.apply}


class NoOp(DualTransform):
    """Does nothing

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

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()


class ReferenceBasedTransform(DualTransform):
    @property
    def targets(self) -> Dict[str, Callable[..., Any]]:
        return {
            "global_label": self.apply_to_global_label,
            "image": self.apply,
            "mask": self.apply_to_mask,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }
