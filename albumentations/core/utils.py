from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union

import numpy as np

from .serialization import Serializable
from .types import DataWithLabels, SizeType, TBBoxesOrKeypoints

if TYPE_CHECKING:
    import torch


DATA_DIM = 4
INSIDE_TARGET_LABELS_NAME = "_INSIDE_TARGET_LABELS"


def get_numpy_2d_array(data: Any) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if data.size == 0:
        data = data.reshape(0, 4)
    if data.ndim == 1:
        data = np.expand_dims(data, 1)
    elif data.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Expected 2d array. Got: {data.ndim}d")

    return data


def get_shape(img: Union["np.ndarray", "torch.Tensor"]) -> SizeType:
    if isinstance(img, np.ndarray):
        rows, cols = img.shape[:2]
        return rows, cols

    try:
        import torch

        if torch.is_tensor(img):
            return img.shape[-2:]
    except ImportError:
        pass

    raise RuntimeError(
        f"Albumentations supports only numpy.ndarray and torch.Tensor data type for image. Got: {type(img)}"
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
            for k, v in additional_targets.items():
                if v == self.default_data_name:
                    self.data_fields.append(k)

    @property
    def internal_type(self) -> Optional[Type[DataWithLabels]]:
        return None

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
            data[data_name] = self.filter(data[data_name], rows, cols)
            data[data_name] = self.check_and_convert(data[data_name], rows, cols, direction="from")

        return self.convert_from_internal_type(data)

    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.internal_type is not None:
            data = self.convert_to_internal_type(data)
        else:
            data = self.add_label_fields_to_data(data)

        rows, cols = data["image"].shape[:2]
        for data_name in self.data_fields:
            data[data_name] = self.check_and_convert(data[data_name], rows, cols, direction="to")

        return data

    def check_and_convert(
        self, data: TBBoxesOrKeypoints, rows: int, cols: int, direction: str = "to"
    ) -> TBBoxesOrKeypoints:
        if self.params.format == "albumentations":
            self.check(data, rows, cols)
            return data

        if direction == "to":
            return self.convert_to_albumentations(data, rows, cols)
        if direction == "from":
            return self.convert_from_albumentations(data, rows, cols)

        raise ValueError(f"Invalid direction. Must be `to` or `from`. Got `{direction}`")

    @abstractmethod
    def filter(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> TBBoxesOrKeypoints:
        pass

    @abstractmethod
    def check(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> None:
        pass

    @abstractmethod
    def convert_to_albumentations(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> TBBoxesOrKeypoints:
        pass

    @abstractmethod
    def convert_from_albumentations(self, data: TBBoxesOrKeypoints, rows: int, cols: int) -> TBBoxesOrKeypoints:
        pass

    def convert_to_internal_type(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.internal_type is None:
            return data

        result_data = {**data}

        def set_data_with_labels(data_name: str) -> None:
            current_data = result_data[data_name]
            values = current_data.data[:, :4]
            current_data.data = values
            current_data.labels[INSIDE_TARGET_LABELS_NAME] = [i[4:] for i in data[data_name]]  # to preserve type

        for data_name in self.data_fields:
            result_data[data_name] = self.internal_type(data=get_numpy_2d_array(data[data_name]))
            if self.params.label_fields:
                if result_data[data_name].data.shape[1] != DATA_DIM:
                    if INSIDE_TARGET_LABELS_NAME in data:
                        err_msg = (
                            "If labels field is set data must have 4 values"
                            f" or default label name `{INSIDE_TARGET_LABELS_NAME}` must be free."
                        )
                        raise ValueError(err_msg)
                    set_data_with_labels(data_name)

                for field in self.params.label_fields:
                    if len(data[data_name]) != len(data[field]):
                        err_msg = (
                            f"{data_name} and {field} has different length."
                            f" {field} can not be used as labels for {data_name}"
                        )
                        raise ValueError(err_msg)

                    result_data[data_name].labels[field] = data[field]
            else:
                set_data_with_labels(data_name)
        return result_data

    def add_label_fields_to_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.params.label_fields is None:
            return data
        for data_name in self.data_fields:
            for field in self.params.label_fields:
                if not len(data[data_name]) == len(data[field]):
                    raise ValueError

                data_with_added_field = []
                for d, field_value in zip(data[data_name], data[field]):
                    data_with_added_field.append([*list(d), field_value])
                data[data_name] = data_with_added_field
        return data

    def convert_from_internal_type(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result_data = {**data}
        for data_name in self.data_fields:
            current_data = data[data_name]
            label_fields_len = len(self.params.label_fields) if self.params.label_fields is not None else 0
            if isinstance(current_data, DataWithLabels):
                if label_fields_len:
                    for field in self.params.label_fields:  # type: ignore[union-attr]
                        result_data[field] = current_data.labels[field]

                if INSIDE_TARGET_LABELS_NAME in current_data.labels:
                    labels = current_data.labels[INSIDE_TARGET_LABELS_NAME]
                    if labels.size:
                        result_data[data_name] = np.concatenate(
                            [current_data.data, labels.reshape(len(current_data), -1)], axis=1
                        )
                        continue

                result_data[data_name] = current_data.data
            elif self.params.label_fields is not None:
                for idx, field in enumerate(self.params.label_fields):
                    result_data[field] = [bbox[-label_fields_len + idx] for bbox in data[data_name]]
                if label_fields_len:
                    result_data[data_name] = [d[:-label_fields_len] for d in data[data_name]]
        return result_data
