from __future__ import absolute_import

from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .serialization import Serializable


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

        self.label_buffer: Dict[str, Dict[str, Any]] = {field: {} for field in self.data_fields}
        self.addition_data: Dict[str, Any] = {field: {} for field in self.data_fields}

    def filter_labels(self, target_name: str, indices: Sequence[int]):
        if not len(indices):
            return
        for label_name, label_data in self.label_buffer[target_name].items():
            ret = list(itemgetter(*indices)(label_data)) if len(indices) > 1 else [itemgetter(*indices)(label_data)]
            self.label_buffer[target_name][label_name] = ret

        if self.addition_data[target_name]:
            ret = (
                list(itemgetter(*indices)(self.addition_data[target_name]))
                if len(indices) > 1
                else [itemgetter(*indices)(self.addition_data[target_name])]
            )
            self.addition_data[target_name] = ret

    @property
    @abstractmethod
    def default_data_name(self) -> str:
        raise NotImplementedError

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        pass

    def ensure_transforms_valid(self, transforms: Sequence[object]) -> None:
        pass

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rows, cols = get_shape(data["image"])

        for data_name in self.data_fields:
            data[data_name] = self.filter(data[data_name], rows, cols, data_name)
            data[data_name] = self.check_and_convert(data[data_name], rows, cols, direction="from")

        data = self.remove_label_fields_from_buffer(data)
        return data

    def preprocess(self, data: Dict[str, Any]) -> None:
        data = self.add_label_fields_to_buffer(data)

        rows, cols = data["image"].shape[:2]
        for data_name in self.data_fields:
            data[data_name] = self.check_and_convert(data[data_name], rows, cols, direction="to")

    def check_and_convert(self, data: np.ndarray, rows: int, cols: int, direction: str = "to") -> np.ndarray:
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
    def filter(self, data: np.ndarray, rows: int, cols: int, target_name: str) -> np.ndarray:
        pass

    @abstractmethod
    def check(self, data: np.ndarray, rows: int, cols: int) -> None:
        pass

    @abstractmethod
    def convert_to_albumentations(self, data: Union[Sequence, np.ndarray], rows: int, cols: int) -> np.ndarray:
        pass

    @abstractmethod
    def convert_from_albumentations(self, data: np.ndarray, rows: int, cols: int) -> np.ndarray:
        pass

    @abstractmethod
    def separate_label_from_data(self, data: Sequence) -> Tuple[Sequence, Sequence]:
        pass

    def add_label_fields_to_buffer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for data_name in self.data_fields:
            sep_data = self.separate_label_from_data(data[data_name])
            if sep_data is not None:
                data[data_name] = sep_data[0]
                self.addition_data[data_name] = sep_data[1]

            if self.params.label_fields is not None:
                for field in self.params.label_fields:
                    assert len(data[data_name]) == len(data[field])
                    self.label_buffer[data_name][field] = list(data[field])
        return data

    def remove_label_fields_from_buffer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for data_name in self.data_fields:
            ori_data = []
            for _data, add_data in zip(data[data_name], self.addition_data[data_name]):
                ori_data.append(tuple(_data) + tuple(add_data))
            data[data_name] = ori_data

            if self.params.label_fields is None:
                continue

            for field in self.params.label_fields:
                data[field] = self.label_buffer[data_name][field]
        return data
