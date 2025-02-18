from __future__ import annotations

import functools
import inspect
import random
from io import StringIO

import numpy as np

import albumentations
from tests.aug_definitions import AUGMENTATION_CLS_PARAMS


def convert_2d_to_3d(arrays, num_channels=3):
    # Converts a 2D numpy array with shape (H, W) into a 3D array with shape (H, W, num_channels)
    # by repeating the existing values along the new axis.
    arrays = tuple(np.repeat(array[:, :, np.newaxis], repeats=num_channels, axis=2) for array in arrays)
    if len(arrays) == 1:
        return arrays[0]
    return arrays


def convert_2d_to_target_format(arrays, target):
    if target == "mask":
        return arrays[0] if len(arrays) == 1 else arrays
    if target == "image":
        return convert_2d_to_3d(arrays, num_channels=3)
    if target == "image_4_channels":
        return convert_2d_to_3d(arrays, num_channels=4)

    raise ValueError(f"Unknown target {target}")


class InMemoryFile(StringIO):
    def __init__(self, value, save_value, file):
        super().__init__(value)
        self.save_value = save_value
        self.file = file

    def close(self):
        self.save_value(self.getvalue(), self.file)
        super().close()


class OpenMock:
    """Mocks the `open` built-in function. A call to the instance of OpenMock returns an in-memory file which is
    readable and writable. The actual in-memory file implementation should call the passed `save_value` method
    to save the file content in the cache when the file is being closed to preserve the file content.
    """

    def __init__(self):
        self.values = {}

    def __call__(self, file, *args, **kwargs):
        value = self.values.get(file)
        return InMemoryFile(value, self.save_value, file)

    def save_value(self, value, file):
        self.values[file] = value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_all_valid_transforms(use_cache=False):
    """
    Find all transforms that are children of BasicTransform or BaseCompose,
    and do not have DeprecationWarning or FutureWarning.

    Args:
        use_cache (bool): Whether to cache the results using lru_cache. Default: False
    """
    if use_cache:
        return _get_all_valid_transforms_cached()
    return _get_all_valid_transforms()


@functools.lru_cache(maxsize=None)
def _get_all_valid_transforms_cached():
    return _get_all_valid_transforms()


def _get_all_valid_transforms():
    valid_transforms = []
    for _, cls in inspect.getmembers(albumentations):
        if not inspect.isclass(cls) or not issubclass(cls, (albumentations.BasicTransform, albumentations.BaseCompose)):
            continue

        valid_transforms.append(cls)
    return valid_transforms


def get_filtered_transforms(
    base_classes,
    custom_arguments=None,
    except_augmentations=None,
    exclude_base_classes=None,
):
    custom_arguments = custom_arguments or {}
    except_augmentations = except_augmentations or set()
    exclude_base_classes = exclude_base_classes or ()

    # Create a mapping of transform class to params from AUGMENTATION_CLS_PARAMS
    default_params = {}
    for transform_entry in AUGMENTATION_CLS_PARAMS:
        transform_cls = transform_entry[0]
        params = transform_entry[1]

        # Convert single dict to list for uniform handling
        if isinstance(params, dict):
            params = [params]

        if transform_cls not in default_params:
            default_params[transform_cls] = []
        default_params[transform_cls].extend(params)

    result = []
    for cls in get_all_valid_transforms():
        # Skip checks...
        if cls in except_augmentations:
            continue
        if any(cls == i for i in base_classes):
            continue
        if exclude_base_classes and issubclass(cls, exclude_base_classes):
            continue
        if not issubclass(cls, base_classes):
            continue

        # Get parameters for this transform
        if cls in custom_arguments:
            params = custom_arguments[cls]
            if isinstance(params, dict):
                params = [params]
            for param_set in params:
                result.append((cls, param_set))
        elif cls in default_params:
            for param_set in default_params[cls]:
                result.append((cls, param_set))
        else:
            result.append((cls, {}))

    return result


def get_image_only_transforms(
    custom_arguments: dict[type[albumentations.ImageOnlyTransform], dict] | None = None,
    except_augmentations: set[type[albumentations.ImageOnlyTransform]] | None = None,
) -> list[tuple[type, dict]]:
    return get_filtered_transforms((albumentations.ImageOnlyTransform,), custom_arguments, except_augmentations)


def get_dual_transforms(
    custom_arguments: dict[type[albumentations.DualTransform], dict] | None = None,
    except_augmentations: set[type[albumentations.DualTransform]] | None = None,
) -> list[tuple[type, dict]]:
    """Get all 2D dual transforms, excluding 3D transforms."""
    return get_filtered_transforms(
        base_classes=(albumentations.DualTransform,),
        custom_arguments=custom_arguments,
        except_augmentations=except_augmentations,
        exclude_base_classes=(albumentations.Transform3D,)
    )

def get_transforms(
    custom_arguments: dict[type[albumentations.BasicTransform], dict] | None = None,
    except_augmentations: set[type[albumentations.BasicTransform]] | None = None,
) -> list[tuple[type, dict]]:
    """Get all transforms (2D and 3D)."""
    return get_filtered_transforms(
        base_classes=(albumentations.ImageOnlyTransform, albumentations.DualTransform, albumentations.Transform3D),
        custom_arguments=custom_arguments,
        except_augmentations=except_augmentations,
    )

def get_2d_transforms(
    custom_arguments: dict[type[albumentations.BasicTransform], dict] | None = None,
    except_augmentations: set[type[albumentations.BasicTransform]] | None = None,
) -> list[tuple[type, dict]]:
    """Get all 2D transforms (both ImageOnly and Dual transforms), excluding 3D transforms."""
    return get_filtered_transforms(
        base_classes=(albumentations.ImageOnlyTransform, albumentations.DualTransform),
        custom_arguments=custom_arguments,
        except_augmentations=except_augmentations,
        exclude_base_classes=(albumentations.Transform3D,)  # Exclude Transform3D and its children
    )

def check_all_augs_exists(
    augmentations: list[list],
    except_augmentations: set | None = None,
) -> list[tuple[type, dict]]:
    existed_augs = {i[0] for i in augmentations}
    except_augmentations = except_augmentations or set()

    not_existed = []

    for cls, _ in get_transforms(except_augmentations=except_augmentations):
        if cls not in existed_augs:
            not_existed.append(cls.__name__)

    if not_existed:
        raise ValueError(f"These augmentations do not exist in augmentations and except_augmentations: {not_existed}")

    # Flatten the parameter sets into individual test cases
    flattened_augmentations = []
    for aug_cls, params in augmentations:
        if isinstance(params, list):
            # If params is a list, create a test case for each parameter set
            for param_set in params:
                flattened_augmentations.append((aug_cls, param_set))
        else:
            # If params is a single dict, keep as is
            flattened_augmentations.append((aug_cls, params))

    return flattened_augmentations


def get_3d_transforms(
    custom_arguments: dict[type[albumentations.Transform3D], dict] | None = None,
    except_augmentations: set[type[albumentations.Transform3D]] | None = None,
) -> list[tuple[type, dict]]:
    """Get all 3D transforms."""
    return get_filtered_transforms(
        base_classes=(albumentations.Transform3D,),
        custom_arguments=custom_arguments,
        except_augmentations=except_augmentations,
    )
