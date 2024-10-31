from __future__ import annotations

import random
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, Literal
from warnings import warn

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from albumentations.core.bbox_utils import BboxProcessor
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.pydantic import ProbabilityType
from albumentations.core.validation import ValidatedTransformMeta

from .serialization import Serializable, SerializableMeta, get_shortest_class_fullname
from .types import (
    NUM_MULTI_CHANNEL_DIMENSIONS,
    ColorType,
    Targets,
)
from .utils import ensure_contiguous_output, format_args

__all__ = ["BasicTransform", "DualTransform", "ImageOnlyTransform", "NoOp"]


class Interpolation:
    def __init__(self, downscale: int = cv2.INTER_NEAREST, upscale: int = cv2.INTER_NEAREST):
        self.downscale = downscale
        self.upscale = upscale


class BaseTransformInitSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    always_apply: bool | None
    p: ProbabilityType


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
    fill_value: ColorType | Literal["random"]
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
        self.processors: dict[str, BboxProcessor | KeypointsProcessor] = {}
        self.seed: int | None = None
        self.random_generator = np.random.default_rng(self.seed)
        self.py_random = random.Random(self.seed)

    def set_random_state(
        self,
        random_generator: np.random.Generator,
        py_random: random.Random,
    ) -> None:
        """Set random state directly from generators.

        Args:
            random_generator: numpy random generator to use
            py_random: python random generator to use
        """
        self.random_generator = random_generator
        self.py_random = py_random

    def set_random_seed(self, seed: int | None) -> None:
        """Set random state from seed.

        Args:
            seed: Random seed to use
        """
        self.seed = seed
        self.random_generator = np.random.default_rng(seed)
        self.py_random = random.Random(seed)

    def get_dict_with_id(self) -> dict[str, Any]:
        d = self.to_dict_private()
        d["id"] = id(self)
        return d

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Returns names of arguments that are used in __init__ method of the transform."""
        msg = (
            f"Class {self.get_class_fullname()} is not serializable because the `get_transform_init_args_names` "
            "method is not implemented"
        )
        raise NotImplementedError(msg)

    def set_processors(self, processors: dict[str, BboxProcessor | KeypointsProcessor]) -> None:
        self.processors = processors

    def get_processor(self, key: str) -> BboxProcessor | KeypointsProcessor | None:
        return self.processors.get(key)

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> Any:
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)
            return kwargs

        # Reset params at the start of each call
        self.params = {}

        if self.should_apply(force_apply=force_apply):
            params = self.get_params()
            params = self.update_params_shape(params=params, data=kwargs)

            if self.targets_as_params:  # check if all required targets are in kwargs.
                missing_keys = set(self.targets_as_params).difference(kwargs.keys())
                if missing_keys and not (missing_keys == {"image"} and "images" in kwargs):
                    msg = f"{self.__class__.__name__} requires {self.targets_as_params} missing keys: {missing_keys}"
                    raise ValueError(msg)

            params_dependent_on_data = self.get_params_dependent_on_data(params=params, data=kwargs)
            params.update(params_dependent_on_data)

            if self.targets_as_params:  # this block will be removed after removing `get_params_dependent_on_targets`
                targets_as_params = {k: kwargs.get(k) for k in self.targets_as_params}
                if missing_keys:  # here we expecting case when missing_keys == {"image"} and "images" in kwargs
                    targets_as_params["image"] = kwargs["images"][0]
                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)

            # Store the final params
            self.params = params

            if self.deterministic:
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def get_applied_params(self) -> dict[str, Any]:
        """Returns the parameters that were used in the last transform application.
        Returns empty dict if transform was not applied.
        """
        return self.params

    def should_apply(self, force_apply: bool = False) -> bool:
        if self.p <= 0.0:
            return False
        if self.p >= 1.0 or force_apply:
            return True
        return self.py_random.random() < self.p

    def apply_with_params(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Apply transforms with parameters."""
        params = self.update_params(params, **kwargs)  # remove after move parameters like interpolation
        res = {}
        for key, arg in kwargs.items():
            if key in self._key2func and arg is not None:
                target_function = self._key2func[key]
                res[key] = ensure_contiguous_output(
                    target_function(ensure_contiguous_output(arg), **params),
                )
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

    def get_base_init_args(self) -> dict[str, Any]:
        """Returns base init args - p"""
        return {"p": self.p}

    def get_transform_init_args(self) -> dict[str, Any]:
        """Exclude seed from init args during serialization"""
        args = {k: getattr(self, k) for k in self.get_transform_init_args_names()}
        args.pop("seed", None)  # Remove seed from args
        return args

    def to_dict_private(self) -> dict[str, Any]:
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state


class DualTransform(BasicTransform):
    """A base class for transformations that should be applied both to an image and its corresponding properties
    such as masks, bounding boxes, and keypoints. This class ensures that when a transform is applied to an image,
    all associated entities are transformed accordingly to maintain consistency between the image and its annotations.

    Methods:
        apply(img: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to the image.

            img: Input image of shape (H, W, C) or (H, W) for grayscale.
            **params: Additional parameters specific to the transform.

            Returns Transformed image of the same shape as input.

        apply_to_mask(mask: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to a mask or sequence of masks.

            mask: Input mask of shape (H, W), (H, W, C) for multi-channel masks,
                    or a sequence of such arrays.
            **params: Additional parameters specific to the transform.

            Returns Transformed mask in the same format as input.

        apply_to_masks(masks: np.ndarray | Sequence[np.ndarray], **params: Any) -> np.ndarray | list[np.ndarray]:
            Apply the transform to multiple masks.

            masks: Either a 3D array of shape (num_masks, H, W) or (H, W, num_masks),
                    or a sequence of 2D/3D arrays each of shape (H, W) or (H, W, C).
            **params: Additional parameters specific to the transform.
            Returns Transformed masks in the same format as input.

        apply_to_keypoints(keypoints: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to keypoints.

            keypoints: Array of shape (N, 2+) where N is the number of keypoints.
                **params: Additional parameters specific to the transform.
            Returns Transformed keypoints array of shape (N, 2+).

        apply_to_bboxes(bboxes: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to bounding boxes.

            bboxes: Array of shape (N, 4+) where N is the number of bounding boxes,
                    and each row is in the format [x_min, y_min, x_max, y_max].
            **params: Additional parameters specific to the transform.

            Returns Transformed bounding boxes array of shape (N, 4+).

    Note:
        - All `apply_*` methods should maintain the input shape and format of the data.
        - The `apply_to_mask` and `apply_to_masks` methods handle both single arrays and sequences of arrays.
        - When applying transforms to masks, ensure that discrete values (e.g., class labels) are preserved.
        - For keypoints and bounding boxes, the transformation should maintain their relative positions
            with respect to the transformed image.
        - The difference between `apply_to_mask` and `apply_to_masks` is mainly in how they handle 3D arrays:
            `apply_to_mask` treats a 3D array as a multi-channel mask, while `apply_to_masks` treats it as
            multiple single-channel masks.

    Example:
        class MyTransform(DualTransform):
            def apply(self, img, **params):
                # Transform the image
                return transformed_img

            def apply_to_mask(self, mask, **params):
                # Transform the mask or sequence of masks
                if isinstance(mask, Sequence):
                    return [self._transform_single_mask(m, **params) for m in mask]
                return self._transform_single_mask(mask, **params)

            def apply_to_keypoints(self, keypoints, **params):
                # Transform the keypoints
                return transformed_keypoints

            def apply_to_bboxes(self, bboxes, **params):
                # Transform the bounding boxes
                return transformed_bboxes

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

    def apply_to_keypoints(self, keypoints: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        msg = f"Method apply_to_keypoints is not implemented in class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def apply_to_bboxes(self, bboxes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        raise NotImplementedError(f"BBoxes not implemented for {self.__class__.__name__}")

    def apply_to_mask(self, mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(mask, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_masks(self, masks: np.ndarray | Sequence[np.ndarray], **params: Any) -> list[np.ndarray] | np.ndarray:
        if isinstance(masks, np.ndarray):
            if masks.ndim == NUM_MULTI_CHANNEL_DIMENSIONS:
                # Transpose from (num_channels, height, width) to (height, width, num_channels)
                masks = np.transpose(masks, (1, 2, 0))
                masks = np.require(masks, requirements=["C_CONTIGUOUS"])
                transformed_masks = self.apply_to_mask(masks, **params)
                # Transpose back to (num_channels, height, width)
                return np.require(np.transpose(transformed_masks, (2, 0, 1)), requirements=["C_CONTIGUOUS"])

            return self.apply_to_mask(masks, **params)
        return [self.apply_to_mask(mask, **params) for mask in masks]


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    _targets = Targets.IMAGE

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {"image": self.apply, "images": self.apply_to_images}


class NoOp(DualTransform):
    """Identity transform (does nothing).

    Targets:
        image, mask, bboxes, keypoints
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return keypoints

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        return bboxes

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return img

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        return mask

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()
