from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast, overload

import numpy as np

from .serialization import Serializable
from .types import PAIR, ScalarType, ScaleFloatType, ScaleIntType, ScaleType

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

            if data_name == "keypoints" and len(data[data_name]) == 0:
                data[data_name] = np.array([], dtype=np.float32).reshape(0, len(self.params.format))

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

                del data[label_field]

            data[data_name] = data_array
        return data

    def remove_label_fields_from_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            data_array = data[data_name]
            if not data_array.size:
                for label_field in self.params.label_fields:
                    data[label_field] = []
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


def validate_args(low: ScaleType | None, bias: ScalarType | None) -> None:
    if low is not None and bias is not None:
        raise ValueError("Arguments 'low' and 'bias' cannot be used together.")


def process_sequence(param: Sequence[ScalarType]) -> tuple[ScalarType, ScalarType]:
    if len(param) != PAIR:
        raise ValueError("Sequence must contain exactly 2 elements.")
    return min(param), max(param)


def process_scalar(param: ScalarType, low: ScalarType | None) -> tuple[ScalarType, ScalarType]:
    if isinstance(low, Real):
        return (low, param) if low < param else (param, low)
    return -param, param


def apply_bias(min_val: ScalarType, max_val: ScalarType, bias: ScalarType) -> tuple[ScalarType, ScalarType]:
    return bias + min_val, bias + max_val


def ensure_int_output(
    min_val: ScalarType,
    max_val: ScalarType,
    param: ScalarType,
) -> tuple[int, int] | tuple[float, float]:
    return (int(min_val), int(max_val)) if isinstance(param, int) else (float(min_val), float(max_val))


@overload
def to_tuple(param: ScaleIntType, low: ScaleType | None = None, bias: ScalarType | None = None) -> tuple[int, int]: ...


@overload
def to_tuple(
    param: ScaleFloatType,
    low: ScaleType | None = None,
    bias: ScalarType | None = None,
) -> tuple[float, float]: ...


def to_tuple(
    param: ScaleType,
    low: ScaleType | None = None,
    bias: ScalarType | None = None,
) -> tuple[int, int] | tuple[float, float]:
    """Convert input argument to a min-max tuple.

    This function processes various input types and returns a tuple representing a range.
    It handles single values, sequences, and can apply optional low bounds or biases.

    Args:
        param (ScaleType): The primary input value. Can be:
            - A single int or float: Converted to a symmetric range around zero.
            - A tuple of two ints or two floats: Used directly as min and max values.

        low (ScaleType | None, optional): A lower bound value. Used when param is a single value.
            If provided, the result will be (low, param) or (param, low), depending on which is smaller.
            Cannot be used together with 'bias'. Defaults to None.

        bias (ScalarType | None, optional): A value to be added to both elements of the resulting tuple.
            Cannot be used together with 'low'. Defaults to None.

    Returns:
        tuple[int, int] | tuple[float, float]: A tuple representing the processed range.
            - If input is int-based, returns tuple[int, int]
            - If input is float-based, returns tuple[float, float]

    Raises:
        ValueError: If both 'low' and 'bias' are provided.
        TypeError: If 'param' is neither a scalar nor a sequence of 2 elements.

    Examples:
        >>> to_tuple(5)
        (-5, 5)
        >>> to_tuple(5.0)
        (-5.0, 5.0)
        >>> to_tuple((1, 10))
        (1, 10)
        >>> to_tuple(5, low=3)
        (3, 5)
        >>> to_tuple(5, bias=1)
        (-4, 6)

    Notes:
        - When 'param' is a single value and 'low' is not provided, the function creates a symmetric range around zero.
        - The function preserves the type (int or float) of the input in the output.
        - If a sequence is provided, it must contain exactly 2 elements.
    """
    validate_args(low, bias)

    if isinstance(param, Sequence):
        min_val, max_val = process_sequence(param)
    elif isinstance(param, Real):
        min_val, max_val = process_scalar(param, cast(ScalarType, low))
    else:
        raise TypeError("Argument 'param' must be either a scalar or a sequence of 2 elements.")

    if bias is not None:
        min_val, max_val = apply_bias(min_val, max_val, bias)

    return ensure_int_output(min_val, max_val, param if isinstance(param, (int, float)) else min_val)
