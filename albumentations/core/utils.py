from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np

from albumentations.core.label_manager import LabelManager

from .serialization import Serializable
from .type_definitions import PAIR, Number

if TYPE_CHECKING:
    import torch

ShapeType = dict[Literal["depth", "height", "width"], int]


def get_image_shape(img: np.ndarray | torch.Tensor) -> tuple[int, int]:
    if isinstance(img, np.ndarray):
        return img.shape[:2]  # HWC format
    try:
        import torch

        if torch.is_tensor(img):
            return img.shape[-2:]  # CHW format
    except ImportError:
        pass
    raise RuntimeError(f"Unsupported image type: {type(img)}")


def get_volume_shape(vol: np.ndarray | torch.Tensor) -> tuple[int, int, int]:
    if isinstance(vol, np.ndarray):
        return vol.shape[:3]  # DHWC format
    try:
        import torch

        if torch.is_tensor(vol):
            return vol.shape[-3:]  # CDHW format
    except ImportError:
        pass
    raise RuntimeError(f"Unsupported volume type: {type(vol)}")


def get_shape(data: dict[str, Any]) -> ShapeType:
    """Extract spatial dimensions from input data dictionary containing image or volume.

    Args:
        data: Dictionary containing image or volume data with one of:
            - 'volume': 3D array of shape (D, H, W, C) [numpy] or (C, D, H, W) [torch]
            - 'image': 2D array of shape (H, W, C) [numpy] or (C, H, W) [torch]
            - 'images': Batch of arrays of shape (N, H, W, C) [numpy] or (N, C, H, W) [torch]

    Returns:
        dict[Literal["depth", "height", "width"], int]: Dictionary containing spatial dimensions
    """
    if "volume" in data:
        depth, height, width = get_volume_shape(data["volume"])
        return {"depth": depth, "height": height, "width": width}
    if "image" in data:
        height, width = get_image_shape(data["image"])
        return {"height": height, "width": width}
    if "images" in data:
        height, width = get_image_shape(data["images"][0])
        return {"height": height, "width": width}
    raise ValueError("No image or volume found in data", data.keys())


def format_args(args_dict: dict[str, Any]) -> str:
    formatted_args = []
    for k, v in args_dict.items():
        v_formatted = f"'{v}'" if isinstance(v, str) else str(v)
        formatted_args.append(f"{k}={v_formatted}")
    return ", ".join(formatted_args)


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
        self.is_sequence_input: dict[str, bool] = {}
        self.label_manager = LabelManager()

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
        shape = get_shape(data)
        data = self._process_data_fields(data, shape)
        data = self.remove_label_fields_from_data(data)
        return self._convert_sequence_inputs(data)

    def _process_data_fields(self, data: dict[str, Any], shape: ShapeType) -> dict[str, Any]:
        for data_name in set(self.data_fields) & set(data.keys()):
            data[data_name] = self._process_single_field(data_name, data[data_name], shape)
        return data

    def _process_single_field(self, data_name: str, field_data: Any, shape: ShapeType) -> Any:
        field_data = self.filter(field_data, shape)

        if data_name == "keypoints" and len(field_data) == 0:
            field_data = self._create_empty_keypoints_array()

        return self.check_and_convert(field_data, shape, direction="from")

    def _create_empty_keypoints_array(self) -> np.ndarray:
        return np.array([], dtype=np.float32).reshape(0, len(self.params.format))

    def _convert_sequence_inputs(self, data: dict[str, Any]) -> dict[str, Any]:
        for data_name in set(self.data_fields) & set(data.keys()):
            if self.is_sequence_input.get(data_name, False):
                data[data_name] = data[data_name].tolist()
        return data

    def preprocess(self, data: dict[str, Any]) -> None:
        shape = get_shape(data)

        for data_name in set(self.data_fields) & set(data.keys()):  # Convert list of lists to numpy array if necessary
            if isinstance(data[data_name], Sequence):
                self.is_sequence_input[data_name] = True
                data[data_name] = np.array(data[data_name], dtype=np.float32)
            else:
                self.is_sequence_input[data_name] = False

        data = self.add_label_fields_to_data(data)
        for data_name in set(self.data_fields) & set(data.keys()):
            data[data_name] = self.check_and_convert(data[data_name], shape, direction="to")

    def check_and_convert(
        self,
        data: np.ndarray,
        shape: ShapeType,
        direction: Literal["to", "from"] = "to",
    ) -> np.ndarray:
        if self.params.format == "albumentations":
            self.check(data, shape)
            return data

        process_func = self.convert_to_albumentations if direction == "to" else self.convert_from_albumentations

        return process_func(data, shape)

    @abstractmethod
    def filter(self, data: np.ndarray, shape: ShapeType) -> np.ndarray:
        pass

    @abstractmethod
    def check(self, data: np.ndarray, shape: ShapeType) -> None:
        pass

    @abstractmethod
    def convert_to_albumentations(
        self,
        data: np.ndarray,
        shape: ShapeType,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def convert_from_albumentations(
        self,
        data: np.ndarray,
        shape: ShapeType,
    ) -> np.ndarray:
        pass

    def add_label_fields_to_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            if not data[data_name].size:
                continue
            data[data_name] = self._process_label_fields(data, data_name)

        return data

    def _process_label_fields(self, data: dict[str, Any], data_name: str) -> np.ndarray:
        data_array = data[data_name]
        if self.params.label_fields is not None:
            for label_field in self.params.label_fields:
                self._validate_label_field_length(data, data_name, label_field)
                encoded_labels = self.label_manager.process_field(data_name, label_field, data[label_field])
                data_array = np.hstack((data_array, encoded_labels))
                del data[label_field]
        return data_array

    def _validate_label_field_length(self, data: dict[str, Any], data_name: str, label_field: str) -> None:
        if len(data[data_name]) != len(data[label_field]):
            raise ValueError(
                f"The lengths of {data_name} and {label_field} do not match. "
                f"Got {len(data[data_name])} and {len(data[label_field])} respectively.",
            )

    def remove_label_fields_from_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            if not data[data_name].size:
                self._handle_empty_data_array(data)
                continue
            self._remove_label_fields(data, data_name)

        return data

    def _handle_empty_data_array(self, data: dict[str, Any]) -> None:
        if self.params.label_fields is not None:
            for label_field in self.params.label_fields:
                data[label_field] = self.label_manager.handle_empty_data()

    def _remove_label_fields(self, data: dict[str, Any], data_name: str) -> None:
        if self.params.label_fields is None:
            return

        data_array = data[data_name]
        num_label_fields = len(self.params.label_fields)
        non_label_columns = data_array.shape[1] - num_label_fields

        for idx, label_field in enumerate(self.params.label_fields):
            encoded_labels = data_array[:, non_label_columns + idx]
            data[label_field] = self.label_manager.restore_field(data_name, label_field, encoded_labels)

        data[data_name] = data_array[:, :non_label_columns]


def validate_args(
    low: float | Sequence[int] | Sequence[float] | None,
    bias: float | None,
) -> None:
    if low is not None and bias is not None:
        raise ValueError("Arguments 'low' and 'bias' cannot be used together.")


def process_sequence(param: Sequence[Number]) -> tuple[Number, Number]:
    if len(param) != PAIR:
        raise ValueError("Sequence must contain exactly 2 elements.")
    return min(param), max(param)


def process_scalar(param: Number, low: Number | None) -> tuple[Number, Number]:
    if isinstance(low, Real):
        return (low, param) if low < param else (param, low)
    return -param, param


def apply_bias(min_val: Number, max_val: Number, bias: Number) -> tuple[Number, Number]:
    return bias + min_val, bias + max_val


def ensure_int_output(
    min_val: Number,
    max_val: Number,
    param: Number,
) -> tuple[int, int] | tuple[float, float]:
    return (int(min_val), int(max_val)) if isinstance(param, int) else (float(min_val), float(max_val))


def ensure_contiguous_output(arg: np.ndarray | Sequence[np.ndarray]) -> np.ndarray | list[np.ndarray]:
    if isinstance(arg, np.ndarray):
        arg = np.ascontiguousarray(arg)
    elif isinstance(arg, Sequence):
        arg = list(map(ensure_contiguous_output, arg))
    return arg


@overload
def to_tuple(
    param: int | tuple[int, int],
    low: int | tuple[int, int] | None = None,
    bias: float | None = None,
) -> tuple[int, int]: ...


@overload
def to_tuple(
    param: float | tuple[float, float],
    low: float | tuple[float, float] | None = None,
    bias: float | None = None,
) -> tuple[float, float]: ...


def to_tuple(
    param: float | tuple[float, float] | tuple[int, int],
    low: float | tuple[float, float] | tuple[int, int] | None = None,
    bias: float | None = None,
) -> tuple[float, float] | tuple[int, int]:
    """Convert input argument to a min-max tuple.

    This function processes various input types and returns a tuple representing a range.
    It handles single values, sequences, and can apply optional low bounds or biases.

    Args:
        param (tuple[float, float] | float | tuple[int, int] | int): The primary input value. Can be:
            - A single int or float: Converted to a symmetric range around zero.
            - A tuple of two ints or two floats: Used directly as min and max values.

        low (tuple[float, float] | float | None, optional): A lower bound value. Used when param is a single value.
            If provided, the result will be (low, param) or (param, low), depending on which is smaller.
            Cannot be used together with 'bias'. Defaults to None.

        bias (float | int | None, optional): A value to be added to both elements of the resulting tuple.
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
        min_val, max_val = process_scalar(param, cast(Real, low))
    else:
        raise TypeError("Argument 'param' must be either a scalar or a sequence of 2 elements.")

    if bias is not None:
        min_val, max_val = apply_bias(min_val, max_val, bias)

    return ensure_int_output(min_val, max_val, param if isinstance(param, (int, float)) else min_val)
