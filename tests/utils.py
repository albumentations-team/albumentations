from __future__ import annotations

import inspect
import random
from io import StringIO

import numpy as np

import albumentations


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


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


def get_filtered_transforms(
    base_classes: tuple[type, ...],
    custom_arguments: frozenset | None = None,
    except_augmentations: frozenset | None = None,
) -> tuple[tuple[type, dict], ...]:
    custom_arguments_dict = dict(custom_arguments) if custom_arguments else {}
    except_augmentations_set = set(except_augmentations) if except_augmentations else set()

    result = []

    for _, cls in inspect.getmembers(albumentations):
        if not isinstance(cls, type):
            continue

        try:
            if not issubclass(cls, (albumentations.BasicTransform, albumentations.BaseCompose)):
                continue
        except TypeError:
            continue  # Skip if issubclass raises a TypeError

        if "DeprecationWarning" in inspect.getsource(cls) or "FutureWarning" in inspect.getsource(cls):
            continue

        try:
            if (
                not issubclass(cls, base_classes)
                or any(cls == i for i in base_classes)
                or cls in except_augmentations_set
            ):
                continue
        except TypeError:
            continue  # Skip if issubclass raises a TypeError

        result.append((cls, custom_arguments_dict.get(cls, {})))
    return tuple(result)


def get_image_only_transforms(
    custom_arguments: dict[type[albumentations.ImageOnlyTransform], dict] | None = None,
    except_augmentations: set[type[albumentations.ImageOnlyTransform]] | None = None,
) -> list[tuple[type, dict]]:
    return get_filtered_transforms((albumentations.ImageOnlyTransform,), custom_arguments, except_augmentations)


def get_dual_transforms(
    custom_arguments: dict[type[albumentations.DualTransform], dict] | None = None,
    except_augmentations: set[type[albumentations.DualTransform]] | None = None,
) -> list[tuple[type, dict]]:
    return get_filtered_transforms((albumentations.DualTransform,), custom_arguments, except_augmentations)


def get_transforms(
    custom_arguments: dict[type[albumentations.BasicTransform], dict] | None = None,
    except_augmentations: set[type[albumentations.BasicTransform]] | None = None,
) -> list[tuple[type, dict]]:
    return get_filtered_transforms(
        (albumentations.ImageOnlyTransform, albumentations.DualTransform),
        custom_arguments,
        except_augmentations,
    )


def check_all_augs_exists(
    augmentations: list[list],
    except_augmentations: set | None = None,
) -> list[list]:
    existed_augs = {i[0] for i in augmentations}
    except_augmentations = except_augmentations or set()

    not_existed = []

    for cls, _ in get_transforms(except_augmentations=except_augmentations):
        if cls not in existed_augs:
            not_existed.append(cls.__name__)

    if not_existed:
        raise ValueError(f"These augmentations do not exist in augmentations and except_augmentations: {not_existed}")

    return augmentations
