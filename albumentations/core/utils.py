from __future__ import absolute_import
from abc import ABCMeta, abstractmethod

from ..core.six import string_types


def format_args(args_dict):
    formatted_args = []
    for k, v in args_dict.items():
        if isinstance(v, string_types):
            v = "'{}'".format(v)
        formatted_args.append('{}={}'.format(k, v))
    return ', '.join(formatted_args)


class Params:
    __metaclass__ = ABCMeta

    def __init__(self, format, label_fields=None):
        self.format = format
        self.label_fields = label_fields or []

    def _to_dict(self):
        return {
            "format": self.format,
            "label_fields": self.label_fields
        }


class DataProcessor:
    __metaclass__ = ABCMeta

    def __init__(self, params, additional_targets=None):
        self.params = params
        self.data_fields = [self.default_data_name]
        if additional_targets is not None:
            for k, v in additional_targets.items():
                if v == self.default_data_name:
                    self.data_fields.append(k)
        self.data_length = 0

    @property
    @abstractmethod
    def default_data_name(self):
        raise NotImplementedError

    def ensure_data_valid(self, data):
        pass

    def ensure_transforms_valid(self, transforms):
        pass

    @abstractmethod
    def preprocess(self, data):
        pass

    @abstractmethod
    def postprocess(self, data):
        pass

    def add_label_fields_to_data(self, data):
        if self.params.label_fields is None:
            return data
        for data_name in self.data_fields:
            for field in self.params.label_fields:
                data_with_added_field = []
                for d, field_value in zip(data[data_name], data[field]):
                    self.data_length = len(list(d))
                    data_with_added_field.append(list(d) + [field_value])
                data[data_name] = data_with_added_field
        return data

    def remove_label_fields_from_data(self, data):
        if self.params.label_fields is None:
            return data
        for data_name in self.data_fields:
            for idx, field in enumerate(self.params.label_fields):
                field_values = []
                for bbox in data[data_name]:
                    field_values.append(bbox[self.data_length + idx])
                data[field] = field_values

            data[data_name] = [d[:self.data_length] for d in data[data_name]]
        return data
