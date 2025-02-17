from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Callable
from warnings import warn

import cv2
import numpy as np
from albucore import batch_transform
from pydantic import BaseModel, ConfigDict, Field

from albumentations.core.bbox_utils import BboxProcessor
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.validation import ValidatedTransformMeta

from .serialization import Serializable, SerializableMeta, get_shortest_class_fullname
from .type_definitions import ALL_TARGETS, Targets
from .utils import ensure_contiguous_output, format_args

__all__ = ["BasicTransform", "DualTransform", "ImageOnlyTransform", "NoOp", "Transform3D"]


class Interpolation:
    def __init__(self, downscale: int = cv2.INTER_NEAREST, upscale: int = cv2.INTER_NEAREST):
        self.downscale = downscale
        self.upscale = upscale


class BaseTransformInitSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    p: float = Field(ge=0, le=1)
    strict: bool = False


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
    fill: tuple[float, ...] | float
    fill_mask: tuple[float, ...] | float | None
    # replay mode params
    deterministic: bool = False
    save_key = "replay"
    replay_mode = False
    applied_in_replay = False

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(self, p: float = 0.5):
        self.p = p
        self._additional_targets: dict[str, str] = {}
        self.params: dict[Any, Any] = {}
        self._key2func = {}
        self._set_keys()
        self.processors: dict[str, BboxProcessor | KeypointsProcessor] = {}
        self.seed: int | None = None
        self.random_generator = np.random.default_rng(self.seed)
        self.py_random = random.Random(self.seed)
        self._strict = False  # Use private attribute
        self.invalid_args: list[str] = []  # Store invalid args found during init

    @property
    def strict(self) -> bool:
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        """Set strict mode and validate for invalid arguments if enabled."""
        if value == self._strict:
            return  # No change needed

        # Only validate if strict is being set to True and we have stored init args
        if value and hasattr(self, "_init_args"):
            # Get the list of valid arguments for this transform
            valid_args = {"p", "strict"}  # Base valid args
            if hasattr(self, "InitSchema"):
                valid_args.update(self.InitSchema.model_fields.keys())

            # Check for invalid arguments
            invalid_args = [name_arg for name_arg in self._init_args if name_arg not in valid_args]

            if invalid_args:
                message = (
                    f"Argument(s) '{', '.join(invalid_args)}' are not valid for transform {self.__class__.__name__}"
                )
                if value:  # In strict mode
                    raise ValueError(message)
                warn(message, stacklevel=2)

        self._strict = value

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
            params = self.update_transform_params(params=params, data=kwargs)

            if self.targets_as_params:  # check if all required targets are in kwargs.
                missing_keys = set(self.targets_as_params).difference(kwargs.keys())
                if missing_keys and not (missing_keys == {"image"} and "images" in kwargs):
                    msg = f"{self.__class__.__name__} requires {self.targets_as_params} missing keys: {missing_keys}"
                    raise ValueError(msg)

            params_dependent_on_data = self.get_params_dependent_on_data(params=params, data=kwargs)
            params.update(params_dependent_on_data)

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

    def apply_to_images(self, images: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform on images.

        Args:
            images: Input images as numpy array of shape:
                - (num_images, height, width, channels)
                - (num_images, height, width) for grayscale
            *args: Additional positional arguments
            **params: Additional parameters specific to the transform

        Returns:
            Transformed images as numpy array in the same format as input
        """
        # Handle batched numpy array input
        transformed = np.stack([self.apply(image, **params) for image in images])
        return np.require(transformed, requirements=["C_CONTIGUOUS"])

    def apply_to_volume(self, volume: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform slice by slice to a volume.

        Args:
            volume: Input volume of shape (depth, height, width) or (depth, height, width, channels)
            *args: Additional positional arguments
            **params: Additional parameters specific to the transform

        Returns:
            Transformed volume as numpy array in the same format as input
        """
        return self.apply_to_images(volume, *args, **params)

    def apply_to_volumes(self, volumes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform to multiple volumes."""
        return np.stack([self.apply_to_volume(vol, *args, **params) for vol in volumes])

    def get_params(self) -> dict[str, Any]:
        """Returns parameters independent of input."""
        return {}

    def update_transform_params(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Updates parameters with input shape and transform-specific params.

        Args:
            params: Parameters to be updated
            data: Input data dictionary containing images/volumes

        Returns:
            Updated parameters dictionary with shape and transform-specific params
        """
        # Extract shape from volume, volumes, image, or images
        if "volume" in data:
            shape = data["volume"][0].shape  # Take first slice of volume
        elif "volumes" in data:
            shape = data["volumes"][0][0].shape  # Take first slice of first volume
        elif "image" in data:
            shape = data["image"].shape
        else:
            shape = data["images"][0].shape

        # For volumes/images, shape will be either (H, W) or (H, W, C)
        params["shape"] = shape

        # Add transform-specific params
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill"):
            params["fill"] = self.fill
        if hasattr(self, "fill_mask"):
            params["fill_mask"] = self.fill_mask

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
        """Returns a dictionary representation of the transform, excluding internal parameters."""
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        # Remove strict from serialization
        state.pop("strict", None)
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

        apply_to_images(images: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to multiple images.

            images: Input images of shape (N, H, W, C) or (N, H, W) for grayscale.
            **params: Additional parameters specific to the transform.

            Returns Transformed images in the same format as input.

        apply_to_mask(mask: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to a mask.

            mask: Input mask of shape (H, W), (H, W, C) for multi-channel masks
            **params: Additional parameters specific to the transform.

            Returns Transformed mask in the same format as input.

        apply_to_masks(masks: np.ndarray, **params: Any) -> np.ndarray | list[np.ndarray]:
            Apply the transform to multiple masks.

            masks: Array of shape (N, H, W) or (N, H, W, C) where N is number of masks
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

        apply_to_volume(volume: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to a volume.

            volume: Input volume of shape (D, H, W) or (D, H, W, C).
            **params: Additional parameters specific to the transform.

            Returns Transformed volume of the same shape as input.

        apply_to_volumes(volumes: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to multiple volumes.

            volumes: Input volumes of shape (N, D, H, W) or (N, D, H, W, C).
            **params: Additional parameters specific to the transform.

            Returns Transformed volumes in the same format as input.

        apply_to_mask3d(mask: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to a 3D mask.

            mask: Input 3D mask of shape (D, H, W) or (D, H, W, C)
            **params: Additional parameters specific to the transform.

            Returns Transformed 3D mask in the same format as input.

        apply_to_masks3d(masks: np.ndarray, **params: Any) -> np.ndarray:
            Apply the transform to multiple 3D masks.

            masks: Input 3D masks of shape (N, D, H, W) or (N, D, H, W, C)
            **params: Additional parameters specific to the transform.

            Returns Transformed 3D masks in the same format as input.

    Note:
        - All `apply_*` methods should maintain the input shape and format of the data.
        - When applying transforms to masks, ensure that discrete values (e.g., class labels) are preserved.
        - For keypoints and bounding boxes, the transformation should maintain their relative positions
            with respect to the transformed image.
        - The difference between `apply_to_mask` and `apply_to_masks` is mainly in how they handle 3D arrays:
            `apply_to_mask` treats a 3D array as a multi-channel mask, while `apply_to_masks` treats it as
            multiple single-channel masks.

    """

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "mask3d": self.apply_to_mask3d,
            "masks3d": self.apply_to_masks3d,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "volume": self.apply_to_volume,
            "volumes": self.apply_to_volumes,
        }

    def apply_to_keypoints(self, keypoints: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        msg = f"Method apply_to_keypoints is not implemented in class {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def apply_to_bboxes(self, bboxes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        raise NotImplementedError(f"BBoxes not implemented for {self.__class__.__name__}")

    def apply_to_mask(self, mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(mask, *args, **params)

    def apply_to_masks(self, masks: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform to multiple masks.

        Args:
            masks: Array of shape (N, H, W) or (N, H, W, C) where N is number of masks
            *args: Additional positional arguments
            **params: Additional parameters specific to the transform

        Returns:
            Array of transformed masks with same shape as input
        """
        transformed = np.stack([self.apply_to_mask(mask, **params) for mask in masks])
        return np.require(transformed, requirements=["C_CONTIGUOUS"])

    @batch_transform("spatial", has_batch_dim=False, has_depth_dim=True)
    def apply_to_mask3d(self, mask3d: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform to single 3D mask.

        Args:
            mask3d: Input 3D mask of shape (D, H, W) or (D, H, W, C)
            *args: Additional positional arguments
            **params: Additional parameters specific to the transform

        Returns:
            Transformed 3D mask in the same format as input
        """
        return self.apply_to_mask(mask3d, *args, **params)

    @batch_transform("spatial", has_batch_dim=True, has_depth_dim=True)
    def apply_to_masks3d(self, masks3d: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform to batch of 3D masks.

        Args:
            masks3d: Input 3D masks of shape (N, D, H, W) or (N, D, H, W, C)
            *args: Additional positional arguments
            **params: Additional parameters specific to the transform

        Returns:
            Transformed 3D masks in the same format as input
        """
        return self.apply_to_mask(masks3d, *args, **params)


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    _targets = (Targets.IMAGE, Targets.VOLUME)

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "volume": self.apply_to_volume,
            "volumes": self.apply_to_volumes,
        }


class NoOp(DualTransform):
    """Identity transform (does nothing).

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d
    """

    _targets = ALL_TARGETS

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        return keypoints

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        return bboxes

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return img

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        return mask

    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        return volume

    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        return volumes

    def apply_to_mask3d(self, mask3d: np.ndarray, **params: Any) -> np.ndarray:
        return mask3d

    def apply_to_masks3d(self, masks3d: np.ndarray, **params: Any) -> np.ndarray:
        return masks3d

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


class Transform3D(DualTransform):
    """Base class for all 3D transforms.

    Transform3D inherits from DualTransform because 3D transforms can be applied to both
    volumes and masks, similar to how 2D DualTransforms work with images and masks.

    Targets:
        volume: 3D numpy array of shape (D, H, W) or (D, H, W, C)
        volumes: Batch of 3D arrays of shape (N, D, H, W) or (N, D, H, W, C)
        mask: 3D numpy array of shape (D, H, W)
        masks: Batch of 3D arrays of shape (N, D, H, W)
        keypoints: 3D numpy array of shape (N, 3)
    """

    def apply_to_volume(self, volume: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform to single 3D volume."""
        raise NotImplementedError

    @batch_transform("spatial", keep_depth_dim=True, has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform to batch of 3D volumes."""
        return self.apply_to_volume(volumes, *args, **params)

    def apply_to_mask3d(self, mask3d: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform to single 3D mask."""
        return self.apply_to_volume(mask3d, *args, **params)

    @batch_transform("spatial", keep_depth_dim=True, has_batch_dim=True, has_depth_dim=True)
    def apply_to_masks3d(self, masks3d: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply transform to batch of 3D masks."""
        return self.apply_to_mask3d(masks3d, *args, **params)

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        """Define valid targets for 3D transforms."""
        return {
            "volume": self.apply_to_volume,
            "volumes": self.apply_to_volumes,
            "mask3d": self.apply_to_mask3d,
            "masks3d": self.apply_to_masks3d,
            "keypoints": self.apply_to_keypoints,
        }
