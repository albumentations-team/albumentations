from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from .serialization import Serializable
from .types import PAIR, ScalarType, ScaleType

if TYPE_CHECKING:
    import torch


def get_shape(img: np.ndarray | torch.Tensor) -> tuple[int, int]:
    if isinstance(img, np.ndarray):
        return img.shape[:2]

    try:
        import torch

        if torch.is_tensor(img):
            return img.shape[-2:]
    except ImportError:
        pass

    raise RuntimeError(
        f"Albumentations supports only numpy.ndarray and torch.Tensor data type for image. Got: {type(img)}",
    )


def format_args(args_dict: dict[str, Any]) -> str:
    formatted_args = []
    for k, v in args_dict.items():
        v_formatted = f"'{v}'" if isinstance(v, str) else str(v)
        formatted_args.append(f"{k}={v_formatted}")
    return ", ".join(formatted_args)


class LabelEncoder:
    def __init__(self) -> None:
        self.classes_: dict[str | int | float, int] = {}
        self.inverse_classes_: dict[int, str | int | float] = {}
        self.num_classes: int = 0

    def fit(self, y: list[Any] | np.ndarray) -> LabelEncoder:
        unique_labels = sorted(set(y))
        for label in unique_labels:
            if label not in self.classes_:
                self.classes_[label] = self.num_classes
                self.inverse_classes_[self.num_classes] = label
                self.num_classes += 1
        return self

    def transform(self, y: list[Any] | np.ndarray) -> np.ndarray:
        return np.array([self.classes_[label] for label in y])

    def fit_transform(self, y: list[Any] | np.ndarray) -> np.ndarray:
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: list[Any] | np.ndarray) -> np.ndarray:
        return np.array([self.inverse_classes_[label] for label in y])


class Params(Serializable, ABC):
    def __init__(self, format: Any, label_fields: Sequence[str] | None):  # noqa: A002
        self.format = format
        self.label_fields = label_fields

    def to_dict_private(self) -> dict[str, Any]:
        return {"format": self.format, "label_fields": self.label_fields}


class DataProcessor(ABC):
    def __init__(self, params: Params, additional_targets: dict[str, str] | None = None):
        self.params = params
        self.data_fields = [self.default_data_name]
        self.label_encoders: dict[str, dict[str, LabelEncoder]] = defaultdict(dict)
        self.is_sequence_input: dict[str, bool] = {}

        if additional_targets is not None:
            self.add_targets(additional_targets)

    @property
    @abstractmethod
    def default_data_name(self) -> str:
        raise NotImplementedError

    def add_targets(self, additional_targets: dict[str, str]) -> None:
        """Add targets to transform them the same way as one of existing targets."""
        for k, v in additional_targets.items():
            if v == self.default_data_name and k not in self.data_fields:
                self.data_fields.append(k)

    def ensure_data_valid(self, data: dict[str, Any]) -> None:
        pass

    def ensure_transforms_valid(self, transforms: Sequence[object]) -> None:
        pass

    def postprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        image_shape = get_shape(data["image"])
        data = self.remove_label_fields_from_data(data)

        for data_name in set(self.data_fields) & set(data.keys()):
            data[data_name] = self.filter(data[data_name], image_shape)
            data[data_name] = self.check_and_convert(data[data_name], image_shape, direction="from")
            # Convert back to list of lists if original input was a list
            if self.is_sequence_input.get(data_name, False):
                data[data_name] = data[data_name].tolist()
        return data

    def preprocess(self, data: dict[str, Any]) -> None:
        image_shape = get_shape(data["image"])

        for data_name in set(self.data_fields) & set(data.keys()):  # Convert list of lists to numpy array if necessary
            if isinstance(data[data_name], Sequence):
                self.is_sequence_input[data_name] = True
                data[data_name] = np.array(data[data_name], dtype=np.float32)
            else:
                self.is_sequence_input[data_name] = False

        data = self.add_label_fields_to_data(data)

        for data_name in set(self.data_fields) & set(data.keys()):
            data[data_name] = self.check_and_convert(data[data_name], image_shape, direction="to")

    def check_and_convert(
        self,
        data: np.ndarray,
        image_shape: tuple[int, int],
        direction: Literal["to", "from"] = "to",
    ) -> np.ndarray:
        if self.params.format == "albumentations":
            self.check(data, image_shape)
            return data

        process_func = self.convert_to_albumentations if direction == "to" else self.convert_from_albumentations

        return process_func(data, image_shape)

    @abstractmethod
    def filter(self, data: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        pass

    @abstractmethod
    def check(self, data: np.ndarray, image_shape: tuple[int, int]) -> None:
        pass

    @abstractmethod
    def convert_to_albumentations(
        self,
        data: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        pass

    @abstractmethod
    def convert_from_albumentations(
        self,
        data: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        pass

    def add_label_fields_to_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            data_array = data[data_name]
            if not data_array.size:
                continue
            for label_field in self.params.label_fields:
                if len(data[data_name]) != len(data[label_field]):
                    raise ValueError(
                        f"The lengths of {data_name} and {label_field} do not match. Got {len(data[data_name])} "
                        f"and {len(data[label_field])} respectively.",
                    )

                # Encode labels
                encoder = LabelEncoder()
                encoded_labels = encoder.fit_transform(data[label_field])
                self.label_encoders[data_name][label_field] = encoder

                # Attach encoded labels as extra columns
                encoded_labels = encoded_labels.reshape(-1, 1)

                data_array = np.hstack((data_array, encoded_labels))

            data[data_name] = data_array
        return data

    def remove_label_fields_from_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            data_array = data[data_name]
            if not data_array.size:
                continue

            num_label_fields = len(self.params.label_fields)
            non_label_columns = data_array.shape[1] - num_label_fields

            for idx, label_field in enumerate(self.params.label_fields):
                encoded_labels = data_array[:, non_label_columns + idx]
                encoder = self.label_encoders.get(data_name, {}).get(label_field)
                if encoder:
                    decoded_labels = encoder.inverse_transform(encoded_labels.astype(int))
                    data[label_field] = decoded_labels.tolist()
                else:
                    raise ValueError(f"Label encoder for {label_field} not found")

            # Remove label columns from data
            data[data_name] = data_array[:, :non_label_columns]
        return data


def to_tuple(
    param: ScaleType,
    low: ScaleType | None = None,
    bias: ScalarType | None = None,
) -> tuple[int, int] | tuple[float, float]:
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
