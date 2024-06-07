from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from typing_extensions import Literal

from .serialization import Serializable
from .types import PAIR, BoxOrKeypointType, ScalarType, ScaleType, SizeType

if TYPE_CHECKING:
    import torch


def get_shape(img: Union["np.ndarray", "torch.Tensor"]) -> SizeType:
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


def format_args(args_dict: Dict[str, Any]) -> str:
    formatted_args = []
    for k, v in args_dict.items():
        v_formatted = f"'{v}'" if isinstance(v, str) else str(v)
        formatted_args.append(f"{k}={v_formatted}")
    return ", ".join(formatted_args)


class Params(Serializable, ABC):
    def __init__(self, format: str, label_fields: Optional[Sequence[str]] = None):
        self.format = format
        self.label_fields = label_fields

    def to_dict_private(self) -> Dict[str, Any]:
        return {"format": self.format, "label_fields": self.label_fields}


class DataProcessor(ABC):
    def __init__(self, params: Params, additional_targets: Optional[Dict[str, str]] = None):
        self.params = params
        self.data_fields = [self.default_data_name]
        if additional_targets is not None:
            self.add_targets(additional_targets)

    @property
    @abstractmethod
    def default_data_name(self) -> str:
        raise NotImplementedError

    def add_targets(self, additional_targets: Dict[str, str]) -> None:
        """Add targets to transform them the same way as one of existing targets"""
        for k, v in additional_targets.items():
            if v == self.default_data_name and k not in self.data_fields:
                self.data_fields.append(k)

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        pass

    def ensure_transforms_valid(self, transforms: Sequence[object]) -> None:
        pass

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rows, cols = get_shape(data["image"])

        for data_name in self.data_fields:
            if data_name in data:
                data[data_name] = self.filter(data[data_name], rows, cols)
                data[data_name] = self.check_and_convert(data[data_name], rows, cols, direction="from")

        return self.remove_label_fields_from_data(data)

    def preprocess(self, data: Dict[str, Any]) -> None:
        data = self.add_label_fields_to_data(data)

        rows, cols = data["image"].shape[:2]
        for data_name in self.data_fields:
            if data_name in data:
                data[data_name] = self.check_and_convert(data[data_name], rows, cols, direction="to")

    def check_and_convert(
        self,
        data: List[BoxOrKeypointType],
        rows: int,
        cols: int,
        direction: Literal["to", "from"] = "to",
    ) -> List[BoxOrKeypointType]:
        if self.params.format == "albumentations":
            self.check(data, rows, cols)
            return data

        if direction == "to":
            return self.convert_to_albumentations(data, rows, cols)

        if direction == "from":
            return self.convert_from_albumentations(data, rows, cols)

        raise ValueError(f"Invalid direction. Must be `to` or `from`. Got `{direction}`")

    @abstractmethod
    def filter(self, data: Sequence[BoxOrKeypointType], rows: int, cols: int) -> Sequence[BoxOrKeypointType]:
        pass

    @abstractmethod
    def check(self, data: List[BoxOrKeypointType], rows: int, cols: int) -> None:
        pass

    @abstractmethod
    def convert_to_albumentations(self, data: List[BoxOrKeypointType], rows: int, cols: int) -> List[BoxOrKeypointType]:
        pass

    @abstractmethod
    def convert_from_albumentations(
        self,
        data: List[BoxOrKeypointType],
        rows: int,
        cols: int,
    ) -> List[BoxOrKeypointType]:
        pass

    def add_label_fields_to_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.params.label_fields is None:
            return data
        for data_name in self.data_fields:
            if data_name in data:
                for field in self.params.label_fields:
                    if not len(data[data_name]) == len(data[field]):
                        raise ValueError(
                            f"The lengths of bboxes and labels do not match. Got {len(data[data_name])} "
                            f"and {len(data[field])} respectively.",
                        )

                    data_with_added_field = []
                    for d, field_value in zip(data[data_name], data[field]):
                        data_with_added_field.append([*list(d), field_value])
                    data[data_name] = data_with_added_field
        return data

    def remove_label_fields_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.params.label_fields:
            return data
        label_fields_len = len(self.params.label_fields)
        for data_name in self.data_fields:
            if data_name in data:
                for idx, field in enumerate(self.params.label_fields):
                    data[field] = [bbox[-label_fields_len + idx] for bbox in data[data_name]]
                data[data_name] = [d[:-label_fields_len] for d in data[data_name]]
        return data


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
