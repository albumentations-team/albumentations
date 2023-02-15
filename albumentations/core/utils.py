from __future__ import absolute_import

from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from .serialization import Serializable

InternalDtype = TypeVar("InternalDtype")


def ensure_internal_format(func: Callable) -> Callable:
    """Ensure data in inputs of the provided function is BatchInternalType,
    and ensure its data consistency.

    Args:
        func (Callable): a callable with the first argument being BatchInternalType.

    Returns:
        Callable, a callable with the first argument with type BatchInternalType.
    """
    from .transforms_interface import BatchInternalType

    @wraps(func)
    def wrapper(data, *args, **kwargs):  # noqa
        data = func(data, *args, **kwargs)
        if isinstance(data, BatchInternalType):
            data.check_consistency()
        else:
            raise TypeError(
                f"The return from {func.__name__} should be a `BatchInternalType`, "
                f"instead it returns a {type(data)}."
            )
        return data

    return wrapper


def get_shape(img: Any) -> Tuple[int, int]:
    if isinstance(img, np.ndarray):
        rows, cols = img.shape[:2]
        return rows, cols

    try:
        import torch

        if torch.is_tensor(img):
            rows, cols = img.shape[-2:]
            return rows, cols
    except ImportError:
        pass

    raise RuntimeError(
        f"Albumentations supports only numpy.ndarray and torch.Tensor data type for image. Got: {type(img)}"
    )


def format_args(args_dict: Dict):
    formatted_args = []
    for k, v in args_dict.items():
        if isinstance(v, str):
            v = f"'{v}'"
        formatted_args.append(f"{k}={v}")
    return ", ".join(formatted_args)


class Params(Serializable, ABC):
    def __init__(self, format: str, label_fields: Optional[Sequence[str]] = None):
        self.format = format
        self.label_fields = label_fields

    def _to_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "label_fields": self.label_fields}


class DataProcessor(ABC):
    def __init__(self, params: Params, additional_targets: Optional[Dict[str, str]] = None):
        self.params = params
        self.data_fields = [self.default_data_name]
        if additional_targets is not None:
            for k, v in additional_targets.items():
                if v == self.default_data_name:
                    self.data_fields.append(k)

    @property
    @abstractmethod
    def default_data_name(self) -> str:
        raise NotImplementedError

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        pass

    def ensure_transforms_valid(self, transforms: Sequence[object]) -> None:
        pass

    @abstractmethod
    def convert_to_internal_type(self, data):
        raise NotImplementedError

    @abstractmethod
    def convert_to_original_type(self, data):
        raise NotImplementedError

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rows, cols = get_shape(data["image"])

        for data_name in self.data_fields:
            _data = self.filter(data[data_name], rows, cols, data_name)
            _data = self.check_and_convert(_data, rows, cols, direction="from")
            data[data_name] = self.convert_to_original_type(_data)

        data = self.remove_label_fields_from_data(data)
        return data

    def preprocess(self, data: Dict[str, Any]) -> None:
        data = self.add_label_fields_to_data(data)

        rows, cols = data["image"].shape[:2]
        for data_name in self.data_fields:
            data[data_name] = self.convert_to_internal_type(data[data_name])
            data[data_name] = self.check_and_convert(data[data_name], rows, cols, direction="to")

    def check_and_convert(self, data: Union[Sequence, InternalDtype], rows: int, cols: int, direction: str = "to"):
        if self.params.format == "albumentations":
            self.check(data, rows, cols)
            return data

        if direction == "to":
            return self.convert_to_albumentations(data, rows, cols)
        elif direction == "from":
            return self.convert_from_albumentations(data, rows, cols)
        else:
            raise ValueError(f"Invalid direction. Must be `to` or `from`. Got `{direction}`")

    @abstractmethod
    def filter(
        self, data: Union[Sequence, InternalDtype], rows: int, cols: int, target_name: str
    ) -> Union[Sequence, InternalDtype]:
        pass

    @abstractmethod
    def check(self, data: Union[Sequence, InternalDtype], rows: int, cols: int) -> None:
        pass

    @abstractmethod
    def convert_to_albumentations(self, data: Any, rows: int, cols: int) -> InternalDtype:  # type: ignore[type-var]
        pass

    @abstractmethod
    def convert_from_albumentations(self, data: InternalDtype, rows: int, cols: int) -> Any:
        pass

    def add_label_fields_to_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.params.label_fields is None:
            return data
        for data_name in self.data_fields:
            for field in self.params.label_fields:
                assert len(data[data_name]) == len(data[field])
                data_with_added_field = []
                for d, field_value in zip(data[data_name], data[field]):
                    data_with_added_field.append(list(d) + [field_value])
                data[data_name] = data_with_added_field
        return data

    def remove_label_fields_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.params.label_fields is None:
            return data
        for data_name in self.data_fields:
            label_fields_len = len(self.params.label_fields)
            for idx, field in enumerate(self.params.label_fields):
                field_values = []
                for bbox in data[data_name]:
                    field_values.append(bbox[-label_fields_len + idx])
                data[field] = field_values
            if label_fields_len:
                data[data_name] = [d[:-label_fields_len] for d in data[data_name]]
        return data
