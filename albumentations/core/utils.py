from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from functools import wraps
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, cast, overload

import numpy as np

from .serialization import Serializable
from .types import PAIR, Number, ScaleFloatType, ScaleIntType, ScaleType

if TYPE_CHECKING:
    import torch

F = TypeVar("F", bound=Callable[..., Any])


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


def custom_sort(item: Any) -> tuple[int, Real | str]:
    if isinstance(item, Real):
        return (0, item)  # Numerical items come first
    return (1, str(item))  # Non-numerical items come second, converted to strings


class LabelEncoder:
    def __init__(self) -> None:
        self.classes_: dict[str | Real, int] = {}
        self.inverse_classes_: dict[int, str | Real] = {}
        self.num_classes: int = 0
        self.is_numerical: bool = True

    def fit(self, y: Sequence[Any] | np.ndarray) -> LabelEncoder:
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        self.is_numerical = all(isinstance(label, Real) for label in y)

        if self.is_numerical:
            return self

        unique_labels = sorted(set(y), key=custom_sort)
        for label in unique_labels:
            if label not in self.classes_:
                self.classes_[label] = self.num_classes
                self.inverse_classes_[self.num_classes] = label
                self.num_classes += 1
        return self

    def transform(self, y: Sequence[Any] | np.ndarray) -> np.ndarray:
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        if self.is_numerical:
            return np.array(y)

        return np.array([self.classes_[label] for label in y])

    def fit_transform(self, y: Sequence[Any] | np.ndarray) -> np.ndarray:
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y: Sequence[Any] | np.ndarray) -> np.ndarray:
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        if self.is_numerical:
            return np.array(y)

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
        self.is_numerical_label: dict[str, dict[str, bool]] = defaultdict(dict)

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
        data = self._process_data_fields(data, image_shape)
        data = self.remove_label_fields_from_data(data)
        return self._convert_sequence_inputs(data)

    def _process_data_fields(self, data: dict[str, Any], image_shape: tuple[int, int]) -> dict[str, Any]:
        for data_name in set(self.data_fields) & set(data.keys()):
            data[data_name] = self._process_single_field(data_name, data[data_name], image_shape)
        return data

    def _process_single_field(self, data_name: str, field_data: Any, image_shape: tuple[int, int]) -> Any:
        field_data = self.filter(field_data, image_shape)

        if data_name == "keypoints" and len(field_data) == 0:
            field_data = self._create_empty_keypoints_array()

        return self.check_and_convert(field_data, image_shape, direction="from")

    def _create_empty_keypoints_array(self) -> np.ndarray:
        return np.array([], dtype=np.float32).reshape(0, len(self.params.format))

    def _convert_sequence_inputs(self, data: dict[str, Any]) -> dict[str, Any]:
        for data_name in set(self.data_fields) & set(data.keys()):
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
            if not data[data_name].size:
                continue
            data[data_name] = self._process_label_fields(data, data_name)

        return data

    def _process_label_fields(self, data: dict[str, Any], data_name: str) -> np.ndarray:
        data_array = data[data_name]
        if self.params.label_fields is not None:
            for label_field in self.params.label_fields:
                self._validate_label_field_length(data, data_name, label_field)
                encoded_labels = self._encode_label_field(data, data_name, label_field)
                data_array = np.hstack((data_array, encoded_labels))
                del data[label_field]
        return data_array

    def _validate_label_field_length(self, data: dict[str, Any], data_name: str, label_field: str) -> None:
        if len(data[data_name]) != len(data[label_field]):
            raise ValueError(
                f"The lengths of {data_name} and {label_field} do not match. "
                f"Got {len(data[data_name])} and {len(data[label_field])} respectively.",
            )

    def _encode_label_field(self, data: dict[str, Any], data_name: str, label_field: str) -> np.ndarray:
        is_numerical = all(isinstance(label, (int, float)) for label in data[label_field])
        self.is_numerical_label[data_name][label_field] = is_numerical

        if is_numerical:
            return np.array(data[label_field], dtype=np.float32).reshape(-1, 1)

        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(data[label_field]).reshape(-1, 1)
        self.label_encoders[data_name][label_field] = encoder
        return encoded_labels

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
                data[label_field] = []

    def _remove_label_fields(self, data: dict[str, Any], data_name: str) -> None:
        if self.params.label_fields is None:
            return

        data_array = data[data_name]
        num_label_fields = len(self.params.label_fields)
        non_label_columns = data_array.shape[1] - num_label_fields

        for idx, label_field in enumerate(self.params.label_fields):
            encoded_labels = data_array[:, non_label_columns + idx]
            decoded_labels = self._decode_label_field(data_name, label_field, encoded_labels)
            data[label_field] = decoded_labels.tolist()

        data[data_name] = data_array[:, :non_label_columns]

    def _decode_label_field(self, data_name: str, label_field: str, encoded_labels: np.ndarray) -> np.ndarray:
        if self.is_numerical_label[data_name][label_field]:
            return encoded_labels

        encoder = self.label_encoders.get(data_name, {}).get(label_field)
        if encoder:
            return encoder.inverse_transform(encoded_labels.astype(int))

        raise ValueError(f"Label encoder for {label_field} not found")


def validate_args(low: ScaleType | None, bias: Number | None) -> None:
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
        arg = np.require(arg, requirements=["C_CONTIGUOUS"])
    elif isinstance(arg, Sequence):
        arg = list(map(ensure_contiguous_output, arg))
    return arg


@overload
def to_tuple(param: ScaleIntType, low: ScaleType | None = None, bias: Number | None = None) -> tuple[int, int]: ...


@overload
def to_tuple(
    param: ScaleFloatType,
    low: ScaleType | None = None,
    bias: Number | None = None,
) -> tuple[float, float]: ...


def to_tuple(
    param: ScaleType,
    low: ScaleType | None = None,
    bias: Number | None = None,
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
        min_val, max_val = process_scalar(param, cast(Real, low))
    else:
        raise TypeError("Argument 'param' must be either a scalar or a sequence of 2 elements.")

    if bias is not None:
        min_val, max_val = apply_bias(min_val, max_val, bias)

    return ensure_int_output(min_val, max_val, param if isinstance(param, (int, float)) else min_val)


BatchTransformType = Literal["spatial", "channel", "full"]

ShapeType = Literal[
    "DHW",  # (D,H,W)
    "DHWC",  # (D,H,W,C)
    "NHW",  # (N,H,W)
    "NHWC",  # (N,H,W,C)
    "NDHW",  # (N,D,H,W)
    "NDHWC",  # (N,D,H,W,C)
]


def get_shape_type(
    shape: tuple[int, ...],
    has_batch_dim: bool,
    has_depth_dim: bool,
) -> ShapeType:
    """Determine the shape type based on dimensions and flags."""
    ndim = len(shape)

    if has_batch_dim and has_depth_dim:
        return "NDHWC" if ndim == 5 else "NDHW"
    if has_batch_dim:
        return "NHWC" if ndim == 4 else "NHW"
    if has_depth_dim:
        return "DHWC" if ndim == 4 else "DHW"
    raise ValueError("Either batch or depth dimension must be True")


def reshape_3d(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (D,H,W) or (D,H,W,C) for spatial transforms."""
    if data.ndim == 3:  # (D,H,W) => (H,W,D)
        depth, height, width = data.shape
        reshaped = np.ascontiguousarray(np.moveaxis(data, 0, -1))  # (H,W,D)
        return reshaped, data.shape
    if data.ndim == 4:  # (D,H,W,C) => (H,W,D*C)
        depth, height, width, channels = data.shape
        reshaped = np.moveaxis(data, 0, -2)  # (H,W,D,C)
        final = np.ascontiguousarray(reshaped.reshape(height, width, -1))  # (H,W,D*C)
        return final, data.shape
    if data.ndim == 5:  # (N,D,H,W,C) => (H,W,N*D*C)
        num_images, depth, height, width, channels = data.shape
        flat = data.reshape(-1, height, width, channels)  # (N*D,H,W,C)
        reshaped = np.moveaxis(flat, 0, -2)  # (H,W,N*D,C)
        final = np.ascontiguousarray(reshaped.reshape(height, width, -1))  # (H,W,N*D*C)
        return final, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_batch(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (N,H,W) or (N,H,W,C) for spatial transforms."""
    if data.ndim == 3:  # (N,H,W) => (H,W,N)
        num_images, height, width = data.shape
        reshaped = np.ascontiguousarray(np.moveaxis(data, 0, -1))  # (H,W,N)
        return reshaped, data.shape
    if data.ndim == 4:  # (N,H,W,C) => (H,W,N*C)
        num_images, height, width, channels = data.shape
        reshaped = np.moveaxis(data, 0, -2)  # (H,W,N,C)
        final = np.ascontiguousarray(reshaped.reshape(height, width, -1))  # (H,W,N*C)
        return final, data.shape
    if data.ndim == 5:  # (N,D,H,W,C) => (H,W,N*D*C)
        num_images, depth, height, width, channels = data.shape
        flat = data.reshape(-1, height, width, channels)  # (N*D,H,W,C)
        reshaped = np.moveaxis(flat, 0, -2)  # (H,W,N*D,C)
        final = np.ascontiguousarray(reshaped.reshape(height, width, -1))  # (H,W,N*D*C)
        return final, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_batch_3d(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (N,D,H,W) or (N,D,H,W,C) for spatial transforms."""
    if data.ndim == 4:  # (N,D,H,W) => (H,W,N*D)
        num_images, depth, height, width = data.shape
        flat = data.reshape(-1, height, width)  # (N*D,H,W)
        reshaped = np.ascontiguousarray(np.moveaxis(flat, 0, -1))  # (H,W,N*D)
        return reshaped, data.shape
    if data.ndim == 5:  # (N,D,H,W,C) => (H,W,N*D*C)
        num_images, depth, height, width, channels = data.shape
        flat = data.reshape(-1, height, width, channels)  # (N*D,H,W,C)
        reshaped = np.moveaxis(flat, 0, -2)  # (H,W,N*D,C)
        final = np.ascontiguousarray(reshaped.reshape(height, width, -1))  # (H,W,N*D*C)
        return final, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_batch_3d_keep_depth(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (N,D,H,W) or (N,D,H,W,C) preserving depth dimension."""
    if data.ndim == 4:  # (N,D,H,W) => (D,H,W,N)
        num_images, depth, height, width = data.shape
        reshaped = np.moveaxis(data, 0, -1)  # (D,H,W,N)
        return np.ascontiguousarray(reshaped), data.shape
    if data.ndim == 5:  # (N,D,H,W,C) => (D,H,W,N*C)
        num_images, depth, height, width, channels = data.shape
        reshaped = np.moveaxis(data, 0, -2)  # (D,H,W,N,C)
        final = np.ascontiguousarray(reshaped.reshape(depth, height, width, -1))  # (D,H,W,N*C)
        return final, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_3d_keep_depth(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (D,H,W) or (D,H,W,C) preserving depth dimension."""
    if data.ndim in {3, 4}:  # (D,H,W) or (D,H,W,C)
        return data, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def restore_3d(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (H,W,D) or (H,W,D*C) back to (D,H,W) or (D,H,W,C)."""
    height, width = data.shape[:2]

    if len(original_shape) == 3:  # (H',W',D) => (D,H',W')
        depth = original_shape[0]
        reshaped = data.reshape(height, width, depth)
        return np.ascontiguousarray(np.moveaxis(reshaped, -1, 0))
    # (H',W',D*C) => (D,H',W',C)
    depth, _, _, channels = original_shape
    reshaped = data.reshape(height, width, depth, channels)  # (H',W',D,C)
    return np.ascontiguousarray(np.moveaxis(reshaped, -2, 0))  # (D,H',W',C)


def restore_batch(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (H,W,N) or (H,W,N*C) back to (N,H,W) or (N,H,W,C)."""
    height, width = data.shape[:2]

    if len(original_shape) == 3:  # (H',W',N) => (N,H',W')
        num_images = original_shape[0]
        reshaped = data.reshape(height, width, num_images)
        return np.ascontiguousarray(np.moveaxis(reshaped, -1, 0))
    # (H',W',N*C) => (N,H',W',C)
    num_images, _, _, channels = original_shape
    reshaped = data.reshape(height, width, num_images, channels)  # (H',W',N,C)
    return np.ascontiguousarray(np.moveaxis(reshaped, -2, 0))  # (N,H',W',C)


def restore_batch_3d(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (H,W,N*D) or (H,W,N*D*C) back to (N,D,H,W) or (N,D,H,W,C)."""
    height, width = data.shape[:2]

    if len(original_shape) == 4:  # (H',W',N*D) => (N,D,H',W')
        num_images, depth = original_shape[:2]
        reshaped = np.moveaxis(data, -1, 0)  # (N*D,H',W')
        return np.ascontiguousarray(reshaped.reshape(num_images, depth, height, width))
    # (H',W',N*D*C) => (N,D,H',W',C)
    num_images, depth, _, _, channels = original_shape
    reshaped = data.reshape(height, width, -1, channels)  # (H',W',N*D,C)
    moved = np.moveaxis(reshaped, -2, 0)  # (N*D,H',W',C)
    return np.ascontiguousarray(moved.reshape(num_images, depth, height, width, channels))


def restore_batch_3d_keep_depth(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore data that kept its depth dimension.

    For data that was reshaped with keep_depth_dim=True:
    - (D',H',W',N) => (N,D',H',W')
    - (D',H',W',N*C) => (N,D',H',W',C)

    Note: D',H',W' can be different from original D,H,W after transforms like RandomCrop3D
    """
    # Use transformed D,H,W dimensions
    new_depth, new_height, new_width = data.shape[:3]

    if len(original_shape) == 4:  # (D',H',W',N) => (N,D',H',W')
        return np.moveaxis(data, -1, 0)
    # (D',H',W',N*C) => (N,D',H',W',C)
    num_images = original_shape[0]
    channels = original_shape[-1]
    # Use new_depth, new_height, new_width instead of original dimensions
    reshaped = data.reshape(new_depth, new_height, new_width, num_images, channels)
    return np.moveaxis(reshaped, -2, 0)


def restore_3d_keep_depth(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore data that kept its depth dimension.

    For data that was reshaped with keep_depth_dim=True:
    - (D,H,W) => (D,H,W)
    - (D,H,W,C) => (D,H,W,C)
    """
    # No reshape needed since we kept the original shape
    return data


def reshape_3d_channel(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (D,H,W) or (D,H,W,C) for channel transforms."""
    if data.ndim == 3:  # (D,H,W) => (D*H,W)
        depth, height, width = data.shape
        reshaped = np.ascontiguousarray(data.reshape(depth * height, width))
        return reshaped, data.shape
    if data.ndim == 4:  # (D,H,W,C) => (D*H,W,C)
        depth, height, width, channels = data.shape
        reshaped = np.ascontiguousarray(data.reshape(depth * height, width, channels))
        return reshaped, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_batch_channel(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (N,H,W) or (N,H,W,C) for channel transforms."""
    if data.ndim == 3:  # (N,H,W) => (N*H,W)
        num_images, height, width = data.shape
        reshaped = np.ascontiguousarray(data.reshape(num_images * height, width))
        return reshaped, data.shape
    if data.ndim == 4:  # (N,H,W,C) => (N*H,W,C)
        num_images, height, width, channels = data.shape
        reshaped = np.ascontiguousarray(data.reshape(num_images * height, width, channels))
        return reshaped, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def reshape_batch_3d_channel(data: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (N,D,H,W) or (N,D,H,W,C) for channel transforms."""
    if data.ndim == 4:  # (N,D,H,W) => (N*D*H,W)
        num_images, depth, height, width = data.shape
        # Flatten N,D,H together, keep W separate
        reshaped = np.ascontiguousarray(data.reshape(-1, width))
        return reshaped, data.shape
    if data.ndim == 5:  # (N,D,H,W,C) => (N*D*H,W,C)
        num_images, depth, height, width, channels = data.shape
        # Flatten N,D,H together, keep W and C separate
        reshaped = np.ascontiguousarray(data.reshape(-1, width, channels))
        return reshaped, data.shape
    raise ValueError(f"Unsupported number of dimensions: {data.ndim}")


def restore_3d_channel(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (D*H,W) or (D*H,W,C) back to (D,H,W) or (D,H,W,C).

    Args:
        data: Array of shape (D*H,W') or (D*H,W',C)
        original_shape: Original shape (D,H,W) or (D,H,W,C)
    """
    if len(original_shape) == 3:  # (D*H,W') => (D,H,W')
        depth, height, _ = original_shape
        return data.reshape(depth, height, data.shape[1])
    # (D*H,W',C) => (D,H,W',C)
    depth, height, _, channels = original_shape
    return data.reshape(depth, height, data.shape[1], channels)


def restore_batch_channel(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (N*H,W) or (N*H,W,C) back to (N,H,W) or (N,H,W,C).

    Args:
        data: Array of shape (N*H,W') or (N*H,W',C)
        original_shape: Original shape (N,H,W) or (N,H,W,C)
    """
    if len(original_shape) == 3:  # (N*H,W') => (N,H,W')
        num_images, height, _ = original_shape
        return data.reshape(num_images, height, data.shape[1])
    # (N*H,W',C) => (N,H,W',C)
    num_images, height, _, channels = original_shape
    return data.reshape(num_images, height, data.shape[1], channels)


def restore_batch_3d_channel(data: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Restore (N*D*H,W') or (N*D*H,W',C) back to (N,D,H,W',C).

    Note: W' can be different from original W after transforms.
    """
    if len(original_shape) == 4:  # (N*D*H,W') => (N,D,H,W')
        num_images, depth, height, _ = original_shape  # Don't use original width
        new_width = data.shape[1]  # Use transformed width
        return data.reshape(num_images, depth, height, new_width)

    # (N*D*H,W',C) => (N,D,H,W',C)
    num_images, depth, height, _, channels = original_shape  # Don't use original width
    new_width = data.shape[1]  # Use transformed width
    return data.reshape(num_images, depth, height, new_width, channels)


# Dictionary mapping shape types to channel reshape functions
CHANNEL_RESHAPE_FUNCS = {
    "DHW": reshape_3d_channel,
    "DHWC": reshape_3d_channel,
    "NHW": reshape_batch_channel,
    "NHWC": reshape_batch_channel,
    "NDHW": reshape_batch_3d_channel,
    "NDHWC": reshape_batch_3d_channel,
}

# Dictionary mapping shape types to channel restore functions
CHANNEL_RESTORE_FUNCS = {
    "DHW": restore_3d_channel,
    "DHWC": restore_3d_channel,
    "NHW": restore_batch_channel,
    "NHWC": restore_batch_channel,
    "NDHW": restore_batch_3d_channel,
    "NDHWC": restore_batch_3d_channel,
}


def reshape_for_channel(
    data: np.ndarray,
    has_batch_dim: bool,
    has_depth_dim: bool,
    keep_depth_dim: bool = False,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Choose appropriate reshape function based on data dimensions."""
    if data.size == 0:
        raise ValueError("Empty arrays are not supported")

    shape_type = get_shape_type(data.shape, has_batch_dim, has_depth_dim)
    reshape_func = CHANNEL_RESHAPE_FUNCS[shape_type]
    return reshape_func(data)


def restore_from_channel(
    data: np.ndarray,
    original_shape: tuple[int, ...],
    has_batch_dim: bool,
    has_depth_dim: bool,
    keep_depth_dim: bool = False,
) -> np.ndarray:
    """Choose appropriate restore function based on data dimensions."""
    shape_type = get_shape_type(original_shape, has_batch_dim, has_depth_dim)
    restore_func = CHANNEL_RESTORE_FUNCS[shape_type]
    result = restore_func(data, original_shape)
    return np.ascontiguousarray(result)


# Dictionary mapping shape types to reshape functions
SPATIAL_RESHAPE_FUNCS = {
    "DHW": reshape_3d,
    "DHWC": reshape_3d,
    "NHW": reshape_batch,
    "NHWC": reshape_batch,
    "NDHW": reshape_batch_3d,
    "NDHWC": reshape_batch_3d,
}

# Dictionary mapping shape types to restore functions
SPATIAL_RESTORE_FUNCS = {
    "DHW": restore_3d,
    "DHWC": restore_3d,
    "NHW": restore_batch,
    "NHWC": restore_batch,
    "NDHW": restore_batch_3d,
    "NDHWC": restore_batch_3d,
}


def reshape_for_spatial(
    data: np.ndarray,
    has_batch_dim: bool,
    has_depth_dim: bool,
    keep_depth_dim: bool = False,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Choose appropriate reshape function based on data dimensions."""
    if data.size == 0:
        raise ValueError("Empty arrays are not supported")

    shape_type = get_shape_type(data.shape, has_batch_dim, has_depth_dim)

    if keep_depth_dim:
        if shape_type in {"NDHW", "NDHWC"}:
            return reshape_batch_3d_keep_depth(data)
        if shape_type in {"DHW", "DHWC"}:
            return reshape_3d_keep_depth(data)

    reshape_func = SPATIAL_RESHAPE_FUNCS[shape_type]
    return reshape_func(data)


def restore_from_spatial(
    data: np.ndarray,
    original_shape: tuple[int, ...],
    has_batch_dim: bool,
    has_depth_dim: bool,
    keep_depth_dim: bool = False,
) -> np.ndarray:
    """Choose appropriate restore function based on data dimensions."""
    shape_type = get_shape_type(original_shape, has_batch_dim, has_depth_dim)

    if keep_depth_dim:
        if shape_type in {"NDHW", "NDHWC"}:
            result = restore_batch_3d_keep_depth(data, original_shape)
        elif shape_type in {"DHW", "DHWC"}:
            result = restore_3d_keep_depth(data, original_shape)
    else:
        restore_func = SPATIAL_RESTORE_FUNCS[shape_type]
        result = restore_func(data, original_shape)

    return np.ascontiguousarray(result)


def batch_transform(
    transform_type: BatchTransformType,
    has_batch_dim: bool = True,
    has_depth_dim: bool = False,
    keep_depth_dim: bool = False,
) -> Callable[[F], F]:
    """Decorator to handle batch transformations."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: Any, data: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)

            if transform_type == "full":
                return func(self, data, *args, **params)

            # Define the function mappings with proper types
            reshape_funcs: dict[str, Callable[..., tuple[np.ndarray, tuple[int, ...]]]] = {
                "spatial": reshape_for_spatial,
                "channel": reshape_for_channel,
            }

            restore_funcs: dict[str, Callable[..., np.ndarray]] = {
                "spatial": restore_from_spatial,
                "channel": restore_from_channel,
            }

            reshape_func = reshape_funcs[transform_type]
            restore_func = restore_funcs[transform_type]

            reshaped, original_shape = reshape_func(
                data,
                has_batch_dim,
                has_depth_dim,
                keep_depth_dim,
            )
            transformed = func(self, reshaped, *args, **params)
            return restore_func(
                transformed,
                original_shape,
                has_batch_dim,
                has_depth_dim,
                keep_depth_dim,
            )

        return cast(F, wrapper)

    return decorator
