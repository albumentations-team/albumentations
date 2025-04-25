"""Module for composing multiple transforms into augmentation pipelines.

This module provides classes for combining multiple transformations into cohesive
augmentation pipelines. It includes various composition strategies such as sequential
application, random selection, and conditional application of transforms. These
composition classes handle the coordination between different transforms, ensuring
proper data flow and maintaining consistent behavior across the augmentation pipeline.
"""

from __future__ import annotations

import random
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any, Union, cast

import cv2
import numpy as np

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
from .utils import DataProcessor, format_args, get_shape

__all__ = [
    "BaseCompose",
    "BboxParams",
    "Compose",
    "KeypointParams",
    "OneOf",
    "OneOrOther",
    "RandomOrder",
    "ReplayCompose",
    "SelectiveChannelTransform",
    "Sequential",
    "SomeOf",
]

NUM_ONEOF_TRANSFORMS = 2
REPR_INDENT_STEP = 2

TransformType = Union[BasicTransform, "BaseCompose"]
TransformsSeqType = list[TransformType]

AVAILABLE_KEYS = ("image", "mask", "masks", "bboxes", "keypoints", "volume", "volumes", "mask3d", "masks3d")

MASK_KEYS = (
    "mask",  # 2D mask
    "masks",  # Multiple 2D masks
    "mask3d",  # 3D mask
    "masks3d",  # Multiple 3D masks
)

# Keys related to image data
IMAGE_KEYS = {"image", "images"}
CHECKED_SINGLE = {"image", "mask"}
CHECKED_MULTI = {"masks", "images", "volumes", "masks3d"}
CHECK_BBOX_PARAM = {"bboxes"}
CHECK_KEYPOINTS_PARAM = {"keypoints"}
VOLUME_KEYS = {"volume", "volumes"}
CHECKED_VOLUME = {"volume"}
CHECKED_VOLUMES = {"volumes"}
CHECKED_MASK3D = {"mask3d"}
CHECKED_MASKS3D = {"masks3d"}


class BaseCompose(Serializable):
    """Base class for composing multiple transforms together.

    This class serves as a foundation for creating compositions of transforms
    in the Albumentations library. It provides basic functionality for
    managing a sequence of transforms and applying them to data.

    Attributes:
        transforms (List[TransformType]): A list of transforms to be applied.
        p (float): Probability of applying the compose. Should be in the range [0, 1].
        replay_mode (bool): If True, the compose is in replay mode.
        _additional_targets (Dict[str, str]): Additional targets for transforms.
        _available_keys (Set[str]): Set of available keys for data.
        processors (Dict[str, Union[BboxProcessor, KeypointsProcessor]]): Processors for specific data types.

    Args:
        transforms (TransformsSeqType): A sequence of transforms to compose.
        p (float): Probability of applying the compose.

    Raises:
        ValueError: If an invalid additional target is specified.

    Note:
        - Subclasses should implement the __call__ method to define how
          the composition is applied to data.
        - The class supports serialization and deserialization of transforms.
        - It provides methods for adding targets, setting deterministic behavior,
          and checking data validity post-transform.

    """

    _transforms_dict: dict[int, BasicTransform] | None = None
    check_each_transform: tuple[DataProcessor, ...] | None = None
    main_compose: bool = True

    def __init__(
        self,
        transforms: TransformsSeqType,
        p: float,
        mask_interpolation: int | None = None,
        seed: int | None = None,
        save_applied_params: bool = False,
        **kwargs: Any,
    ):
        if isinstance(transforms, (BaseCompose, BasicTransform)):
            warnings.warn(
                "transforms is single transform, but a sequence is expected! Transform will be wrapped into list.",
                stacklevel=2,
            )
            transforms = [transforms]

        self.transforms = transforms
        self.p = p

        self.replay_mode = False
        self._additional_targets: dict[str, str] = {}
        self._available_keys: set[str] = set()
        self.processors: dict[str, BboxProcessor | KeypointsProcessor] = {}
        self._set_keys()
        self.set_mask_interpolation(mask_interpolation)
        self.set_random_seed(seed)
        self.save_applied_params = save_applied_params

    def _track_transform_params(self, transform: TransformType, data: dict[str, Any]) -> None:
        """Track transform parameters if tracking is enabled."""
        if "applied_transforms" in data and hasattr(transform, "params") and transform.params:
            data["applied_transforms"].append((transform.__class__.__name__, transform.params.copy()))

    def set_random_state(
        self,
        random_generator: np.random.Generator,
        py_random: random.Random,
    ) -> None:
        """Set random state directly from generators.

        Args:
            random_generator (np.random.Generator): numpy random generator to use
            py_random (random.Random): python random generator to use

        """
        self.random_generator = random_generator
        self.py_random = py_random

        # Propagate both random states to all transforms
        for transform in self.transforms:
            if isinstance(transform, (BasicTransform, BaseCompose)):
                transform.set_random_state(random_generator, py_random)

    def set_random_seed(self, seed: int | None) -> None:
        """Set random state from seed.

        Args:
            seed (int | None): Random seed to use

        """
        self.seed = seed
        self.random_generator = np.random.default_rng(seed)
        self.py_random = random.Random(seed)
        # Propagate seed to all transforms
        for transform in self.transforms:
            if isinstance(transform, (BasicTransform, BaseCompose)):
                transform.set_random_seed(seed)

    def set_mask_interpolation(self, mask_interpolation: int | None) -> None:
        """Set interpolation mode for mask resizing operations.

        Args:
            mask_interpolation (int | None): OpenCV interpolation flag to use for mask transforms.
                If None, default interpolation for masks will be used.

        """
        self.mask_interpolation = mask_interpolation
        self._set_mask_interpolation_recursive(self.transforms)

    def _set_mask_interpolation_recursive(self, transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BasicTransform):
                if hasattr(transform, "mask_interpolation") and self.mask_interpolation is not None:
                    transform.mask_interpolation = self.mask_interpolation
            elif isinstance(transform, BaseCompose):
                transform.set_mask_interpolation(self.mask_interpolation)

    def __iter__(self) -> Iterator[TransformType]:
        return iter(self.transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __call__(self, *args: Any, **data: Any) -> dict[str, Any]:
        """Apply transforms.

        Args:
            *args (Any): Positional arguments are not supported.
            **data (Any): Named parameters with data to transform.

        Returns:
            dict[str, Any]: Transformed data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        """
        raise NotImplementedError

    def __getitem__(self, item: int) -> TransformType:
        return self.transforms[item]

    def __repr__(self) -> str:
        return self.indented_repr()

    @property
    def additional_targets(self) -> dict[str, str]:
        """Get additional targets dictionary.

        Returns:
            dict[str, str]: Dictionary containing additional targets mapping.

        """
        return self._additional_targets

    @property
    def available_keys(self) -> set[str]:
        """Get set of available keys.

        Returns:
            set[str]: Set of string keys available for transforms.

        """
        return self._available_keys

    def indented_repr(self, indent: int = REPR_INDENT_STEP) -> str:
        """Get an indented string representation of the composition.

        Args:
            indent (int): Indentation level. Default: REPR_INDENT_STEP.

        Returns:
            str: Formatted string representation with proper indentation.

        """
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
        """Get the full qualified name of the class.

        Returns:
            str: The shortest class fullname.

        """
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls) -> bool:
        """Check if the class is serializable.

        Returns:
            bool: True if the class is serializable, False otherwise.

        """
        return True

    def to_dict_private(self) -> dict[str, Any]:
        """Convert the composition to a dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the composition.

        """
        return {
            "__class_fullname__": self.get_class_fullname(),
            "p": self.p,
            "transforms": [t.to_dict_private() for t in self.transforms],
        }

    def get_dict_with_id(self) -> dict[str, Any]:
        """Get a dictionary representation with object IDs for replay mode.

        Returns:
            dict[str, Any]: Dictionary with composition data and object IDs.

        """
        return {
            "__class_fullname__": self.get_class_fullname(),
            "id": id(self),
            "params": None,
            "transforms": [t.get_dict_with_id() for t in self.transforms],
        }

    def add_targets(self, additional_targets: dict[str, str] | None) -> None:
        """Add additional targets to all transforms.

        Args:
            additional_targets (dict[str, str] | None): Dict of name -> type mapping for additional targets.
                If None, no additional targets will be added.

        """
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
        """Set deterministic mode for all transforms.

        Args:
            flag (bool): Whether to enable deterministic mode.
            save_key (str): Key to save replay parameters. Default: "replay".

        """
        for t in self.transforms:
            t.set_deterministic(flag, save_key)

    def check_data_post_transform(self, data: dict[str, Any]) -> dict[str, Any]:
        """Check and filter data after transformation.

        Args:
            data (dict[str, Any]): Dictionary containing transformed data

        Returns:
            dict[str, Any]: Filtered data dictionary

        """
        if self.check_each_transform:
            shape = get_shape(data)

            for proc in self.check_each_transform:
                for data_name, data_value in data.items():
                    if data_name in proc.data_fields or (
                        data_name in self._additional_targets
                        and self._additional_targets[data_name] in proc.data_fields
                    ):
                        data[data_name] = proc.filter(data_value, shape)
        return data


class Compose(BaseCompose, HubMixin):
    """Compose multiple transforms together and apply them sequentially to input data.

    This class allows you to chain multiple image augmentation transforms and apply them
    in a specified order. It also handles bounding box and keypoint transformations if
    the appropriate parameters are provided.

    Args:
        transforms (list[BasicTransform | BaseCompose]): A list of transforms to apply.
        bbox_params (dict[str, Any] | BboxParams | None): Parameters for bounding box transforms.
            Can be a dict of params or a BboxParams object. Default is None.
        keypoint_params (dict[str, Any] | KeypointParams | None): Parameters for keypoint transforms.
            Can be a dict of params or a KeypointParams object. Default is None.
        additional_targets (dict[str, str] | None): A dictionary mapping additional target names
            to their types. For example, {'image2': 'image'}. Default is None.
        p (float): Probability of applying all transforms. Should be in range [0, 1]. Default is 1.0.
        is_check_shapes (bool): If True, checks consistency of shapes for image/mask/masks on each call.
            Disable only if you are sure about your data consistency. Default is True.
        strict (bool): If True, enables strict mode which:
            1. Validates that all input keys are known/expected
            2. Validates that no transforms have invalid arguments
            3. Raises ValueError if any validation fails
            If False, these validations are skipped. Default is False.
        mask_interpolation (int | None): Interpolation method for mask transforms. When defined,
            it overrides the interpolation method specified in individual transforms. Default is None.
        seed (int | None): Controls reproducibility of random augmentations. Compose uses
            its own internal random state, completely independent from global random seeds.

            When seed is set (int):
            - Creates a fixed internal random state
            - Two Compose instances with the same seed and transforms will produce identical
              sequences of augmentations
            - Each call to the same Compose instance still produces random augmentations,
              but these sequences are reproducible between different Compose instances
            - Example: transform1 = A.Compose([...], seed=137) and
                      transform2 = A.Compose([...], seed=137) will produce identical sequences

            When seed is None (default):
            - Generates a new internal random state on each Compose creation
            - Different Compose instances will produce different sequences of augmentations
            - Example: transform = A.Compose([...])  # random results

            Important: Setting random seeds outside of Compose (like np.random.seed() or
            random.seed()) has no effect on augmentations as Compose uses its own internal
            random state.
        save_applied_params (bool): If True, saves the applied parameters of each transform. Default is False.
            You will need to use the `applied_transforms` key in the output dictionary to access the parameters.

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.RandomCrop(width=256, height=256),
        ...     A.HorizontalFlip(p=0.5),
        ...     A.RandomBrightnessContrast(p=0.2),
        ... ], seed=137)
        >>> transformed = transform(image=image)

    Note:
        - The class checks the validity of input data and shapes if is_check_args and is_check_shapes are True.
        - When bbox_params or keypoint_params are provided, it sets up the corresponding processors.
        - The transform can handle additional targets specified in the additional_targets dictionary.
        - When strict mode is enabled, it performs additional validation to ensure data and transform
          configuration correctness.

    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: dict[str, Any] | BboxParams | None = None,
        keypoint_params: dict[str, Any] | KeypointParams | None = None,
        additional_targets: dict[str, str] | None = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        strict: bool = False,
        mask_interpolation: int | None = None,
        seed: int | None = None,
        save_applied_params: bool = False,
    ):
        super().__init__(
            transforms=transforms,
            p=p,
            mask_interpolation=mask_interpolation,
            seed=seed,
            save_applied_params=save_applied_params,
        )

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
        self.strict = strict

        self.is_check_shapes = is_check_shapes
        self.check_each_transform = tuple(  # processors that checks after each transform
            proc for proc in self.processors.values() if getattr(proc.params, "check_each_transform", False)
        )
        self._set_check_args_for_transforms(self.transforms)

        self._set_processors_for_transforms(self.transforms)

        self.save_applied_params = save_applied_params
        self._images_was_list = False
        self._masks_was_list = False

    @property
    def strict(self) -> bool:
        """Get the current strict mode setting.

        Returns:
            bool: True if strict mode is enabled, False otherwise.

        """
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        # if value and not self._strict:
        if value:
            # Only validate when enabling strict mode
            self._validate_strict()
        self._strict = value

    def _validate_strict(self) -> None:
        """Validate that no transforms have invalid arguments when strict mode is enabled."""

        def check_transform(transform: TransformType) -> None:
            if hasattr(transform, "invalid_args") and transform.invalid_args:
                message = (
                    f"Argument(s) '{', '.join(transform.invalid_args)}' "
                    f"are not valid for transform {transform.__class__.__name__}"
                )
                raise ValueError(message)
            if isinstance(transform, BaseCompose):
                for t in transform.transforms:
                    check_transform(t)

        for transform in self.transforms:
            check_transform(transform)

    def _set_processors_for_transforms(self, transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BasicTransform):
                if hasattr(transform, "set_processors"):
                    transform.set_processors(self.processors)
            elif isinstance(transform, BaseCompose):
                self._set_processors_for_transforms(transform.transforms)

    def _set_check_args_for_transforms(self, transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                self._set_check_args_for_transforms(transform.transforms)
                transform.check_each_transform = self.check_each_transform
                transform.processors = self.processors
            if isinstance(transform, Compose):
                transform.disable_check_args_private()

    def disable_check_args_private(self) -> None:
        """Disable argument checking for transforms.

        This method disables strict mode and argument checking for all transforms in the composition.
        """
        self.is_check_args = False
        self.strict = False
        self.main_compose = False

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply transformations to data.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        Raises:
            KeyError: If positional arguments are provided.

        """
        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)

        if not isinstance(force_apply, (bool, int)):
            msg = "force_apply must have bool or int type"
            raise TypeError(msg)

        # Initialize applied_transforms only in top-level Compose if requested
        if self.save_applied_params and self.main_compose:
            data["applied_transforms"] = []

        need_to_run = force_apply or self.py_random.random() < self.p
        if not need_to_run:
            return data

        self.preprocess(data)

        for t in self.transforms:
            data = t(**data)
            self._track_transform_params(t, data)
            data = self.check_data_post_transform(data)

        return self.postprocess(data)

    def preprocess(self, data: Any) -> None:
        """Preprocess input data before applying transforms."""
        # Always validate shapes if is_check_shapes is True, regardless of strict mode
        if self.is_check_shapes:
            shapes = []  # For H,W checks
            volume_shapes = []  # For D,H,W checks

            for data_name, data_value in data.items():
                internal_name = self._additional_targets.get(data_name, data_name)

                # Skip empty data
                if data_value is None:
                    continue

                shape = self._get_data_shape(data_name, internal_name, data_value)
                if shape is not None:
                    if internal_name in CHECKED_VOLUME | CHECKED_MASK3D:
                        shapes.append(shape[1:3])  # H,W from (D,H,W)
                        volume_shapes.append(shape[:3])  # D,H,W
                    elif internal_name in {"volumes", "masks3d"}:
                        shapes.append(shape[2:4])  # H,W from (N,D,H,W)
                        volume_shapes.append(shape[1:4])  # D,H,W from (N,D,H,W)
                    else:
                        shapes.append(shape[:2])  # H,W

            self._check_shape_consistency(shapes, volume_shapes)

        # Do strict validation only if enabled
        if self.strict:
            self._validate_data(data)

        self._preprocess_processors(data)
        self._preprocess_arrays(data)

    def _validate_data(self, data: dict[str, Any]) -> None:
        """Validate input data keys and arguments."""
        if not self.strict:
            return

        for data_name in data:
            if not self._is_valid_key(data_name):
                raise ValueError(f"Key {data_name} is not in available keys.")

        if self.is_check_args:
            self._check_args(**data)

    def _is_valid_key(self, key: str) -> bool:
        """Check if the key is valid for processing."""
        return key in self._available_keys or key in MASK_KEYS or key in IMAGE_KEYS or key == "applied_transforms"

    def _preprocess_processors(self, data: dict[str, Any]) -> None:
        """Run preprocessors if this is the main compose."""
        if not self.main_compose:
            return

        for processor in self.processors.values():
            processor.ensure_data_valid(data)
        for processor in self.processors.values():
            processor.preprocess(data)

    def _preprocess_arrays(self, data: dict[str, Any]) -> None:
        """Convert lists to numpy arrays for images and masks, and ensure contiguity."""
        self._preprocess_images(data)
        self._preprocess_masks(data)

    def _preprocess_images(self, data: dict[str, Any]) -> None:
        """Convert image lists to numpy arrays."""
        if "images" not in data:
            return

        if isinstance(data["images"], (list, tuple)):
            self._images_was_list = True
            data["images"] = np.stack(data["images"])
        else:
            self._images_was_list = False

    def _preprocess_masks(self, data: dict[str, Any]) -> None:
        """Convert mask lists to numpy arrays."""
        if "masks" not in data:
            return

        if isinstance(data["masks"], (list, tuple)):
            self._masks_was_list = True
            data["masks"] = np.stack(data["masks"])
        else:
            self._masks_was_list = False

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply post-processing to data after all transforms have been applied.

        Args:
            data (dict[str, Any]): Data after transformation.

        Returns:
            dict[str, Any]: Post-processed data.

        """
        if self.main_compose:
            for p in self.processors.values():
                p.postprocess(data)

            # Convert back to list if original input was a list
            if "images" in data and self._images_was_list:
                data["images"] = list(data["images"])

            if "masks" in data and self._masks_was_list:
                data["masks"] = list(data["masks"])

        return data

    def to_dict_private(self) -> dict[str, Any]:
        """Convert the composition to a dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the composition.

        """
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

    def get_dict_with_id(self) -> dict[str, Any]:
        """Get a dictionary representation with object IDs for replay mode.

        Returns:
            dict[str, Any]: Dictionary with composition data and object IDs.

        """
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

    @staticmethod
    def _check_single_data(data_name: str, data: Any) -> tuple[int, int]:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{data_name} must be numpy array type")
        return data.shape[:2]

    @staticmethod
    def _check_masks_data(data_name: str, data: Any) -> tuple[int, int]:
        """Check masks data format and return shape.

        Args:
            data_name (str): Name of the data field being checked
            data (Any): Input data in one of these formats:
                - List of numpy arrays, each of shape (H, W) or (H, W, C)
                - Numpy array of shape (N, H, W) or (N, H, W, C)

        Returns:
            tuple[int, int]: (height, width) of the first mask
        Raises:
            TypeError: If data format is invalid

        """
        if isinstance(data, np.ndarray):
            if data.ndim not in [3, 4]:  # (N,H,W) or (N,H,W,C)
                raise TypeError(f"{data_name} as numpy array must be 3D or 4D")
            return data.shape[1:3]  # Return (H,W)

        if isinstance(data, (list, tuple)):
            if not data:
                raise ValueError(f"{data_name} cannot be empty")
            if not all(isinstance(m, np.ndarray) for m in data):
                raise TypeError(f"All elements in {data_name} must be numpy arrays")
            if any(m.ndim not in {2, 3} for m in data):
                raise TypeError(f"All masks in {data_name} must be 2D or 3D numpy arrays")
            return data[0].shape[:2]

        raise TypeError(f"{data_name} must be either a numpy array or a sequence of numpy arrays")

    @staticmethod
    def _check_multi_data(data_name: str, data: Any) -> tuple[int, int]:
        """Check multi-image data format and return shape.

        Args:
            data_name (str): Name of the data field being checked
            data (Any): Input data in one of these formats:
                - List-like of numpy arrays
                - Numpy array of shape (N, H, W, C) or (N, H, W)

        Returns:
            tuple[int, int]: (height, width) of the first image
        Raises:
            TypeError: If data format is invalid

        """
        if isinstance(data, np.ndarray):
            if data.ndim not in {3, 4}:  # (N,H,W) or (N,H,W,C)
                raise TypeError(f"{data_name} as numpy array must be 3D or 4D")
            return data.shape[1:3]  # Return (H,W)

        if not isinstance(data, Sequence) or not isinstance(data[0], np.ndarray):
            raise TypeError(f"{data_name} must be either a numpy array or a list of numpy arrays")
        return data[0].shape[:2]

    @staticmethod
    def _check_bbox_keypoint_params(internal_data_name: str, processors: dict[str, Any]) -> None:
        if internal_data_name in CHECK_BBOX_PARAM and processors.get("bboxes") is None:
            raise ValueError("bbox_params must be specified for bbox transformations")
        if internal_data_name in CHECK_KEYPOINTS_PARAM and processors.get("keypoints") is None:
            raise ValueError("keypoints_params must be specified for keypoint transformations")

    @staticmethod
    def _check_shapes(shapes: list[tuple[int, ...]], is_check_shapes: bool) -> None:
        if is_check_shapes and shapes and shapes.count(shapes[0]) != len(shapes):
            raise ValueError(
                "Height and Width of image, mask or masks should be equal. You can disable shapes check "
                "by setting a parameter is_check_shapes=False of Compose class (do it only if you are sure "
                "about your data consistency).",
            )

    def _check_args(self, **kwargs: Any) -> None:
        shapes = []  # For H,W checks
        volume_shapes = []  # For D,H,W checks

        for data_name, data in kwargs.items():
            internal_name = self._additional_targets.get(data_name, data_name)

            # For CHECKED_SINGLE, we must validate even if None
            if internal_name in CHECKED_SINGLE:
                if not isinstance(data, np.ndarray):
                    raise TypeError(f"{data_name} must be numpy array type")
                shapes.append(data.shape[:2])
                continue

            # Skip empty data or non-array/list inputs for other types
            if data is None:
                continue
            if not isinstance(data, (np.ndarray, list)):
                continue

            self._check_bbox_keypoint_params(internal_name, self.processors)

            shape = self._get_data_shape(data_name, internal_name, data)
            if shape is None:
                continue

            # Handle different shape types
            if internal_name in CHECKED_VOLUME | CHECKED_MASK3D:
                shapes.append(shape[1:3])  # H,W from (D,H,W)
                volume_shapes.append(shape[:3])  # D,H,W
            elif internal_name in {"volumes", "masks3d"}:
                shapes.append(shape[2:4])  # H,W from (N,D,H,W)
                volume_shapes.append(shape[1:4])  # D,H,W from (N,D,H,W)
            else:
                shapes.append(shape[:2])  # H,W

        self._check_shape_consistency(shapes, volume_shapes)

    def _get_data_shape(self, data_name: str, internal_name: str, data: Any) -> tuple[int, ...] | None:
        """Get shape of data based on its type."""
        if internal_name in CHECKED_SINGLE:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{data_name} must be numpy array type")
            return data.shape

        if internal_name in CHECKED_VOLUME:
            return self._check_volume_data(data_name, data)

        if internal_name in CHECKED_MASK3D:
            return self._check_mask3d_data(data_name, data)

        if internal_name in CHECKED_MULTI:
            if internal_name == "masks":
                return self._check_masks_data(data_name, data)
            if internal_name in {"volumes", "masks3d"}:  # Group these together
                if not isinstance(data, np.ndarray):
                    raise TypeError(f"{data_name} must be numpy array type")
                if data.ndim not in {4, 5}:  # (N,D,H,W) or (N,D,H,W,C)
                    raise TypeError(f"{data_name} must be 4D or 5D array")
                return data.shape  # Return full shape
            return self._check_multi_data(data_name, data)

        return None

    def _check_shape_consistency(self, shapes: list[tuple[int, ...]], volume_shapes: list[tuple[int, ...]]) -> None:
        """Check consistency of shapes."""
        # Check H,W consistency
        self._check_shapes(shapes, self.is_check_shapes)

        # Check D,H,W consistency for volumes and 3D masks
        if self.is_check_shapes and volume_shapes and volume_shapes.count(volume_shapes[0]) != len(volume_shapes):
            raise ValueError(
                "Depth, Height and Width of volume, mask3d, volumes and masks3d should be equal. "
                "You can disable shapes check by setting is_check_shapes=False.",
            )

    @staticmethod
    def _check_volume_data(data_name: str, data: np.ndarray) -> tuple[int, int, int]:
        if data.ndim not in {3, 4}:  # (D,H,W) or (D,H,W,C)
            raise TypeError(f"{data_name} must be 3D or 4D array")
        return data.shape[:3]  # Return (D,H,W)

    @staticmethod
    def _check_volumes_data(data_name: str, data: np.ndarray) -> tuple[int, int, int]:
        if data.ndim not in {4, 5}:  # (N,D,H,W) or (N,D,H,W,C)
            raise TypeError(f"{data_name} must be 4D or 5D array")
        return data.shape[1:4]  # Return (D,H,W)

    @staticmethod
    def _check_mask3d_data(data_name: str, data: np.ndarray) -> tuple[int, int, int]:
        """Check single volumetric mask data format and return shape."""
        if data.ndim not in {3, 4}:  # (D,H,W) or (D,H,W,C)
            raise TypeError(f"{data_name} must be 3D or 4D array")
        return data.shape[:3]  # Return (D,H,W)

    @staticmethod
    def _check_masks3d_data(data_name: str, data: np.ndarray) -> tuple[int, int, int]:
        """Check multiple volumetric masks data format and return shape."""
        if data.ndim not in [4, 5]:  # (N,D,H,W) or (N,D,H,W,C)
            raise TypeError(f"{data_name} must be 4D or 5D array")
        return data.shape[1:4]  # Return (D,H,W)


class OneOf(BaseCompose):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.

    """

    def __init__(self, transforms: TransformsSeqType, p: float = 0.5):
        super().__init__(transforms=transforms, p=p)
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply the OneOf composition to the input data.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        Raises:
            KeyError: If positional arguments are provided.

        """
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
            return data

        if self.transforms_ps and (force_apply or self.py_random.random() < self.p):
            idx: int = self.random_generator.choice(len(self.transforms), p=self.transforms_ps)
            t = self.transforms[idx]
            data = t(force_apply=True, **data)
            self._track_transform_params(t, data)
        return data


class SomeOf(BaseCompose):
    """Selects exactly `n` transforms from the given list and applies them.

    The selection of which `n` transforms to apply is done **uniformly at random**
    from the provided list. Each transform in the list has an equal chance of being selected.

    Once the `n` transforms are selected, each one is applied **based on its
    individual probability** `p`.

    Args:
        transforms (list[BasicTransform | BaseCompose]): A list of transforms to choose from.
        n (int): The exact number of transforms to select and potentially apply.
                 If `replace=False` and `n` is greater than the number of available transforms,
                 `n` will be capped at the number of transforms.
        replace (bool): Whether to sample transforms with replacement. If True, the same
                        transform can be selected multiple times (up to `n` times).
                        Default is False.
        p (float): The probability that this `SomeOf` composition will be applied.
                   If applied, it will select `n` transforms and attempt to apply them.
                   Default is 1.0.

    Note:
        - The overall probability `p` of the `SomeOf` block determines if *any* selection
          and application occurs.
        - The individual probability `p` of each transform inside the list determines if
          that specific transform runs *if it is selected*.
        - If `replace` is True, the same transform might be selected multiple times, and
          its individual probability `p` will be checked each time it's encountered.

    Example:
        >>> import albumentations as A
        >>> transform = A.SomeOf([
        ...     A.HorizontalFlip(p=0.5),  # 50% chance to apply if selected
        ...     A.VerticalFlip(p=0.8),    # 80% chance to apply if selected
        ...     A.RandomRotate90(p=1.0), # 100% chance to apply if selected
        ... ], n=2, replace=False, p=1.0) # Always select 2 transforms uniformly

        # In each call, 2 transforms out of 3 are chosen uniformly.
        # For example, if HFlip and VFlip are chosen:
        # - HFlip runs if random() < 0.5
        # - VFlip runs if random() < 0.8
        # If VFlip and Rotate90 are chosen:
        # - VFlip runs if random() < 0.8
        # - Rotate90 runs if random() < 1.0 (always)

    """

    def __init__(self, transforms: TransformsSeqType, n: int = 1, replace: bool = False, p: float = 1):
        super().__init__(transforms, p)
        self.n = n
        if not replace and n > len(self.transforms):
            self.n = len(self.transforms)
            warnings.warn(
                f"`n` is greater than number of transforms. `n` will be set to {self.n}.",
                UserWarning,
                stacklevel=2,
            )
        self.replace = replace

    def __call__(self, *arg: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply n randomly selected transforms from the list of transforms.

        Args:
            *arg (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        """
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
                data = self.check_data_post_transform(data)
            return data

        if self.py_random.random() < self.p:  # Check overall SomeOf probability
            # Get indices uniformly
            indices_to_consider = self._get_idx()
            for i in indices_to_consider:
                t = self.transforms[i]
                # Apply the transform respecting its own probability `t.p`
                data = t(**data)
                self._track_transform_params(t, data)
                data = self.check_data_post_transform(data)
        return data

    def _get_idx(self) -> np.ndarray[np.int_]:
        # Use uniform probability for selection, ignore individual p values here
        idx = self.random_generator.choice(
            len(self.transforms),
            size=self.n,
            replace=self.replace,
        )
        idx.sort()
        return idx

    def to_dict_private(self) -> dict[str, Any]:
        """Convert the SomeOf composition to a dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the composition.

        """
        dictionary = super().to_dict_private()
        dictionary.update({"n": self.n, "replace": self.replace})
        return dictionary


class RandomOrder(SomeOf):
    """Apply a random subset of transforms from the given list in a random order.

    Selects exactly `n` transforms uniformly at random from the list, and then applies
    the selected transforms in a random order. Each selected transform is applied
    based on its individual probability `p`.

    Attributes:
        transforms (TransformsSeqType): A list of transformations to choose from.
        n (int): The number of transforms to apply. If `n` is greater than the number of available transforms
                 and `replace` is False, `n` will be set to the number of available transforms.
        replace (bool): Whether to sample transforms with replacement. If True, the same transform can be
                        selected multiple times. Default is False.
        p (float): Probability of applying the selected transforms. Should be in the range [0, 1]. Default is 1.0.

    Example:
        >>> import albumentations as A
        >>> transform = A.RandomOrder([
        ...     A.HorizontalFlip(p=0.5),
        ...     A.VerticalFlip(p=1.0),
        ...     A.RandomBrightnessContrast(p=0.8),
        ... ], n=2, replace=False, p=1.0)
        >>> # This will uniformly select 2 transforms and apply them in a random order,
        >>> # respecting their individual probabilities (0.5, 1.0, 0.8).

    Note:
        - Inherits from SomeOf, but overrides `_get_idx` to ensure random order without sorting.
        - Selection is uniform; application depends on individual transform probabilities.

    """

    def __init__(self, transforms: TransformsSeqType, n: int = 1, replace: bool = False, p: float = 1):
        # Initialize using SomeOf's logic (which now does uniform selection setup)
        super().__init__(transforms=transforms, n=n, replace=replace, p=p)

    def _get_idx(self) -> np.ndarray[np.int_]:
        # Perform uniform random selection without replacement, like SomeOf
        # Crucially, DO NOT sort the indices here to maintain random order.
        return self.random_generator.choice(
            len(self.transforms),
            size=self.n,
            replace=self.replace,
        )


class OneOrOther(BaseCompose):
    """Select one or another transform to apply. Selected transform will be called with `force_apply=True`."""

    def __init__(
        self,
        first: TransformType | None = None,
        second: TransformType | None = None,
        transforms: TransformsSeqType | None = None,
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

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply one or another transform to the input data.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        """
        if self.replay_mode:
            for t in self.transforms:
                data = t(**data)
                self._track_transform_params(t, data)
            return data

        if self.py_random.random() < self.p:
            return self.transforms[0](force_apply=True, **data)

        return self.transforms[-1](force_apply=True, **data)


class SelectiveChannelTransform(BaseCompose):
    """A transformation class to apply specified transforms to selected channels of an image.

    This class extends BaseCompose to allow selective application of transformations to
    specified image channels. It extracts the selected channels, applies the transformations,
    and then reinserts the transformed channels back into their original positions in the image.

    Args:
        transforms (TransformsSeqType):
            A sequence of transformations (from Albumentations) to be applied to the specified channels.
        channels (Sequence[int]):
            A sequence of integers specifying the indices of the channels to which the transforms should be applied.
        p (float): Probability that the transform will be applied; the default is 1.0 (always apply).

    Returns:
        dict[str, Any]: The transformed data dictionary, which includes the transformed 'image' key.

    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        channels: Sequence[int] = (0, 1, 2),
        p: float = 1.0,
    ) -> None:
        super().__init__(transforms, p)
        self.channels = channels

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply transforms to specific channels of the image.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        """
        if force_apply or self.py_random.random() < self.p:
            image = data["image"]

            selected_channels = image[:, :, self.channels]
            sub_image = np.ascontiguousarray(selected_channels)

            for t in self.transforms:
                sub_image = t(image=sub_image)["image"]
                self._track_transform_params(t, sub_image)

            transformed_channels = cv2.split(sub_image)
            output_img = image.copy()

            for idx, channel in zip(self.channels, transformed_channels):
                output_img[:, :, idx] = channel

            data["image"] = np.ascontiguousarray(output_img)

        return data


class ReplayCompose(Compose):
    """Composition class that enables transform replay functionality.

    This class extends the Compose class with the ability to record and replay
    transformations. This is useful for applying the same sequence of random
    transformations to different data.

    Args:
        transforms (TransformsSeqType): List of transformations to compose.
        bbox_params (dict[str, Any] | BboxParams | None): Parameters for bounding box transforms.
        keypoint_params (dict[str, Any] | KeypointParams | None): Parameters for keypoint transforms.
        additional_targets (dict[str, str] | None): Dictionary of additional targets.
        p (float): Probability of applying the compose.
        is_check_shapes (bool): Whether to check shapes of different targets.
        save_key (str): Key for storing the applied transformations.

    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: dict[str, Any] | BboxParams | None = None,
        keypoint_params: dict[str, Any] | KeypointParams | None = None,
        additional_targets: dict[str, str] | None = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        save_key: str = "replay",
    ):
        super().__init__(transforms, bbox_params, keypoint_params, additional_targets, p, is_check_shapes)
        self.set_deterministic(True, save_key=save_key)
        self.save_key = save_key
        self._available_keys.add(save_key)

    def __call__(self, *args: Any, force_apply: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Apply transforms and record parameters for future replay.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **kwargs (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data and replay information.

        """
        kwargs[self.save_key] = defaultdict(dict)
        result = super().__call__(force_apply=force_apply, **kwargs)
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, result[self.save_key])
        self.fill_applied(serialized)
        result[self.save_key] = serialized
        return result

    @staticmethod
    def replay(saved_augmentations: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Replay previously saved augmentations.

        Args:
            saved_augmentations (dict[str, Any]): Previously saved augmentation parameters.
            **kwargs (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data using saved parameters.

        """
        augs = ReplayCompose._restore_for_replay(saved_augmentations)
        return augs(force_apply=True, **kwargs)

    @staticmethod
    def _restore_for_replay(
        transform_dict: dict[str, Any],
        lambda_transforms: dict[str, Any] | None = None,
    ) -> TransformType:
        """Args:
        transform_dict (dict[str, Any]): A dictionary that contains transform data.
        lambda_transforms (dict): A dictionary that contains lambda transforms, that
            is instances of the Lambda class.
        This dictionary is required when you are restoring a pipeline that contains lambda transforms.
        Keys in that dictionary should be named same as `name` arguments in respective lambda transforms
        from a serialized pipeline.

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

        transform = cast("BasicTransform", transform)
        if isinstance(transform, BasicTransform):
            transform.params = params
        transform.replay_mode = True
        transform.applied_in_replay = applied
        return transform

    def fill_with_params(self, serialized: dict[str, Any], all_params: Any) -> None:
        """Fill serialized transform data with parameters for replay.

        Args:
            serialized (dict[str, Any]): Serialized transform data.
            all_params (Any): Parameters to fill in.

        """
        params = all_params.get(serialized.get("id"))
        serialized["params"] = params
        del serialized["id"]
        for transform in serialized.get("transforms", []):
            self.fill_with_params(transform, all_params)

    def fill_applied(self, serialized: dict[str, Any]) -> bool:
        """Set 'applied' flag for transforms based on parameters.

        Args:
            serialized (dict[str, Any]): Serialized transform data.

        Returns:
            bool: True if any transform was applied, False otherwise.

        """
        if "transforms" in serialized:
            applied = [self.fill_applied(t) for t in serialized["transforms"]]
            serialized["applied"] = any(applied)
        else:
            serialized["applied"] = serialized.get("params") is not None
        return serialized["applied"]

    def to_dict_private(self) -> dict[str, Any]:
        """Convert the ReplayCompose to a dictionary for serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the composition.

        """
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
        super().__init__(transforms=transforms, p=p)

    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> dict[str, Any]:
        """Apply all transforms in sequential order.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            dict[str, Any]: Dictionary with transformed data.

        """
        if self.replay_mode or force_apply or self.py_random.random() < self.p:
            for t in self.transforms:
                data = t(**data)
                self._track_transform_params(t, data)
                data = self.check_data_post_transform(data)
        return data
