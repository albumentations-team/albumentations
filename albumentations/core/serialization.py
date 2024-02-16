import json
import warnings
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Tuple, Type, Union, cast

try:
    import yaml

    yaml_available = True
except ImportError:
    yaml_available = False


from albumentations import __version__

__all__ = ["to_dict", "from_dict", "save", "load"]


SERIALIZABLE_REGISTRY: Dict[str, "SerializableMeta"] = {}
NON_SERIALIZABLE_REGISTRY: Dict[str, "SerializableMeta"] = {}


def shorten_class_name(class_fullname: str) -> str:
    splitted = class_fullname.split(".")
    if len(splitted) == 1:
        return class_fullname
    top_module, *_, class_name = splitted
    if top_module == "albumentations":
        return class_name
    return class_fullname


class SerializableMeta(ABCMeta):
    """
    A metaclass that is used to register classes in `SERIALIZABLE_REGISTRY` or `NON_SERIALIZABLE_REGISTRY`
    so they can be found later while deserializing transformation pipeline using classes full names.
    """

    def __new__(mcs, name: str, bases: Tuple[type, ...], *args: Any, **kwargs: Any) -> "SerializableMeta":
        cls_obj = super().__new__(mcs, name, bases, *args, **kwargs)
        if name != "Serializable" and ABC not in bases:
            if cls_obj.is_serializable():
                SERIALIZABLE_REGISTRY[cls_obj.get_class_fullname()] = cls_obj
            else:
                NON_SERIALIZABLE_REGISTRY[cls_obj.get_class_fullname()] = cls_obj
        return cls_obj

    @classmethod
    def is_serializable(mcs) -> bool:
        return False

    @classmethod
    def get_class_fullname(mcs) -> str:
        return get_shortest_class_fullname(mcs)

    @classmethod
    def _to_dict(mcs) -> Dict[str, Any]:
        return {}


class Serializable(metaclass=SerializableMeta):
    @classmethod
    @abstractmethod
    def is_serializable(cls) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_class_fullname(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def _to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def to_dict(self, on_not_implemented_error: str = "raise") -> Dict[str, Any]:
        """
        Take a transform pipeline and convert it to a serializable representation that uses only standard
        python data types: dictionaries, lists, strings, integers, and floats.

        Args:
            self: A transform that should be serialized. If the transform doesn't implement the `to_dict`
                method and `on_not_implemented_error` equals to 'raise' then `NotImplementedError` is raised.
                If `on_not_implemented_error` equals to 'warn' then `NotImplementedError` will be ignored
                but no transform parameters will be serialized.
            on_not_implemented_error (str): `raise` or `warn`.
        """
        if on_not_implemented_error not in {"raise", "warn"}:
            raise ValueError(
                "Unknown on_not_implemented_error value: {}. Supported values are: 'raise' and 'warn'".format(
                    on_not_implemented_error
                )
            )
        try:
            transform_dict = self._to_dict()
        except NotImplementedError as e:
            if on_not_implemented_error == "raise":
                raise e

            transform_dict = {}
            warnings.warn(
                "Got NotImplementedError while trying to serialize {obj}. Object arguments are not preserved. "
                "Implement either '{cls_name}.get_transform_init_args_names' or '{cls_name}.get_transform_init_args' "
                "method to make the transform serializable".format(obj=self, cls_name=self.__class__.__name__)
            )
        return {"__version__": __version__, "transform": transform_dict}


def to_dict(transform: Serializable, on_not_implemented_error: str = "raise") -> Dict[str, Any]:
    """
    Take a transform pipeline and convert it to a serializable representation that uses only standard
    python data types: dictionaries, lists, strings, integers, and floats.

    Args:
        transform: A transform that should be serialized. If the transform doesn't implement the `to_dict`
            method and `on_not_implemented_error` equals to 'raise' then `NotImplementedError` is raised.
            If `on_not_implemented_error` equals to 'warn' then `NotImplementedError` will be ignored
            but no transform parameters will be serialized.
        on_not_implemented_error (str): `raise` or `warn`.
    """
    return transform.to_dict(on_not_implemented_error)


def instantiate_nonserializable(
    transform: Dict[str, Any], nonserializable: Optional[Dict[str, Any]] = None
) -> Optional[Serializable]:
    if transform.get("__class_fullname__") in NON_SERIALIZABLE_REGISTRY:
        name = transform["__name__"]
        if nonserializable is None:
            raise ValueError(
                "To deserialize a non-serializable transform with name {name} you need to pass a dict with"
                "this transform as the `lambda_transforms` argument".format(name=name)
            )
        result_transform = nonserializable.get(name)
        if transform is None:
            raise ValueError(f"Non-serializable transform with {name} was not found in `nonserializable`")
        return result_transform
    return None


def from_dict(
    transform_dict: Dict[str, Any], nonserializable: Optional[Dict[str, Any]] = None
) -> Optional[Serializable]:
    """
    Args:
        transform_dict: A dictionary with serialized transform pipeline.
        nonserializable (dict): A dictionary that contains non-serializable transforms.
            This dictionary is required when you are restoring a pipeline that contains non-serializable transforms.
            Keys in that dictionary should be named same as `name` arguments in respective transforms from
            a serialized pipeline.
    """
    register_additional_transforms()
    transform = transform_dict["transform"]
    lmbd = instantiate_nonserializable(transform, nonserializable)
    if lmbd:
        return lmbd
    name = transform["__class_fullname__"]
    args = {k: v for k, v in transform.items() if k != "__class_fullname__"}
    cls = SERIALIZABLE_REGISTRY[shorten_class_name(name)]
    if "transforms" in args:
        args["transforms"] = [from_dict({"transform": t}, nonserializable=nonserializable) for t in args["transforms"]]
    return cls(**args)


def check_data_format(data_format: str) -> None:
    if data_format not in {"json", "yaml"}:
        raise ValueError(f"Unknown data_format {data_format}. Supported formats are: 'json' and 'yaml'")


def save(
    transform: "Serializable",
    filepath_or_buffer: Union[str, Path, TextIO],
    data_format: str = "json",
    on_not_implemented_error: str = "raise",
) -> None:
    """
    Serialize a transform pipeline and save it to either a file specified by a path or a file-like object
    in either JSON or YAML format.

    Args:
        transform (Serializable): The transform pipeline to serialize.
        filepath_or_buffer (Union[str, Path, TextIO]): The file path or file-like object to write the serialized
            data to.
            If a string is provided, it is interpreted as a path to a file. If a file-like object is provided,
            the serialized data will be written to it directly.
        data_format (str): The format to serialize the data in. Valid options are 'json' and 'yaml'.
            Defaults to 'json'.
        on_not_implemented_error (str): Determines the behavior if a transform does not implement the `to_dict` method.
            If set to 'raise', a `NotImplementedError` is raised. If set to 'warn', the exception is ignored, and
            no transform arguments are saved. Defaults to 'raise'.

    Raises:
        ValueError: If `data_format` is 'yaml' but PyYAML is not installed.
    """
    check_data_format(data_format)
    transform_dict = transform.to_dict(on_not_implemented_error=on_not_implemented_error)

    # Determine whether to write to a file or a file-like object
    if isinstance(filepath_or_buffer, (str, Path)):  # It's a filepath
        with open(filepath_or_buffer, "w") as f:
            if data_format == "yaml":
                if not yaml_available:
                    raise ValueError("You need to install PyYAML to save a pipeline in YAML format")
                yaml.safe_dump(transform_dict, f, default_flow_style=False)
            elif data_format == "json":
                json.dump(transform_dict, f)
    else:  # Assume it's a file-like object
        if data_format == "yaml":
            if not yaml_available:
                raise ValueError("You need to install PyYAML to save a pipeline in YAML format")
            yaml.safe_dump(transform_dict, filepath_or_buffer, default_flow_style=False)
        elif data_format == "json":
            json.dump(transform_dict, filepath_or_buffer)


def load(
    filepath_or_buffer: Union[str, Path, TextIO],
    data_format: str = "json",
    nonserializable: Optional[Dict[str, Any]] = None,
) -> object:
    """
    Load a serialized pipeline from a file or file-like object and construct a transform pipeline.

    Args:
        filepath_or_buffer (Union[str, Path, TextIO]): The file path or file-like object to read the serialized
            data from.
            If a string is provided, it is interpreted as a path to a file. If a file-like object is provided,
            the serialized data will be read from it directly.
        data_format (str): The format of the serialized data. Valid options are 'json' and 'yaml'.
            Defaults to 'json'.
        nonserializable (Optional[Dict[str, Any]]): A dictionary that contains non-serializable transforms.
            This dictionary is required when restoring a pipeline that contains non-serializable transforms.
            Keys in the dictionary should be named the same as the `name` arguments in respective transforms
            from the serialized pipeline. Defaults to None.

    Returns:
        object: The deserialized transform pipeline.

    Raises:
        ValueError: If `data_format` is 'yaml' but PyYAML is not installed.
    """
    check_data_format(data_format)

    if isinstance(filepath_or_buffer, (str, Path)):  # Assume it's a filepath
        with open(filepath_or_buffer) as f:
            if data_format == "json":
                transform_dict = json.load(f)
            else:
                if not yaml_available:
                    raise ValueError("You need to install PyYAML to load a pipeline in yaml format")
                transform_dict = yaml.safe_load(f)
    else:  # Assume it's a file-like object
        if data_format == "json":
            transform_dict = json.load(filepath_or_buffer)
        else:
            if not yaml_available:
                raise ValueError("You need to install PyYAML to load a pipeline in yaml format")
            transform_dict = yaml.safe_load(filepath_or_buffer)

    return from_dict(transform_dict, nonserializable=nonserializable)


def register_additional_transforms() -> None:
    """
    Register transforms that are not imported directly into the `albumentations` module.
    """
    try:
        # This import will result in ImportError if `torch` is not installed
        import albumentations.pytorch
    except ImportError:
        pass


def get_shortest_class_fullname(cls: Type[Any]) -> str:
    """
    The function `get_shortest_class_fullname` takes a class object as input and returns its shortened
    full name.

    :param cls: The parameter `cls` is of type `Type[BasicCompose]`, which means it expects a class that
    is a subclass of `BasicCompose`
    :type cls: Type[BasicCompose]
    :return: a string, which is the shortened version of the full class name.
    """
    class_fullname = "{cls.__module__}.{cls.__name__}".format(cls=cls)
    return shorten_class_name(class_fullname)
