import importlib.util
import json
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Tuple, Type, Union

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
    split = class_fullname.split(".")
    if len(split) == 1:
        return class_fullname
    top_module, *_, class_name = split
    if top_module == "albumentations":
        return class_name
    return class_fullname


class SerializableMeta(ABCMeta):
    """A metaclass that is used to register classes in `SERIALIZABLE_REGISTRY` or `NON_SERIALIZABLE_REGISTRY`
    so they can be found later while deserializing transformation pipeline using classes full names.
    """

    def __new__(cls, name: str, bases: Tuple[type, ...], *args: Any, **kwargs: Any) -> "SerializableMeta":
        cls_obj = super().__new__(cls, name, bases, *args, **kwargs)
        if name != "Serializable" and ABC not in bases:
            if cls_obj.is_serializable():
                SERIALIZABLE_REGISTRY[cls_obj.get_class_fullname()] = cls_obj
            else:
                NON_SERIALIZABLE_REGISTRY[cls_obj.get_class_fullname()] = cls_obj
        return cls_obj

    @classmethod
    def is_serializable(cls) -> bool:
        return False

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    @classmethod
    def _to_dict(cls) -> Dict[str, Any]:
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
    def to_dict_private(self) -> Dict[str, Any]:
        raise NotImplementedError

    def to_dict(self, on_not_implemented_error: str = "raise") -> Dict[str, Any]:
        """Take a transform pipeline and convert it to a serializable representation that uses only standard
        python data types: dictionaries, lists, strings, integers, and floats.

        Args:
            self: A transform that should be serialized. If the transform doesn't implement the `to_dict`
                method and `on_not_implemented_error` equals to 'raise' then `NotImplementedError` is raised.
                If `on_not_implemented_error` equals to 'warn' then `NotImplementedError` will be ignored
                but no transform parameters will be serialized.
            on_not_implemented_error (str): `raise` or `warn`.

        """
        if on_not_implemented_error not in {"raise", "warn"}:
            msg = f"Unknown on_not_implemented_error value: {on_not_implemented_error}. Supported values are: 'raise' "
            "and 'warn'"
            raise ValueError(msg)
        try:
            transform_dict = self.to_dict_private()
        except NotImplementedError:
            if on_not_implemented_error == "raise":
                raise

            transform_dict = {}
            warnings.warn(
                f"Got NotImplementedError while trying to serialize {self}. Object arguments are not preserved. "
                f"Implement either '{self.__class__.__name__}.get_transform_init_args_names' "
                f"or '{self.__class__.__name__}.get_transform_init_args' "
                "method to make the transform serializable",
            )
        return {"__version__": __version__, "transform": transform_dict}


def to_dict(transform: Serializable, on_not_implemented_error: str = "raise") -> Dict[str, Any]:
    """Take a transform pipeline and convert it to a serializable representation that uses only standard
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
    transform: Dict[str, Any],
    nonserializable: Optional[Dict[str, Any]] = None,
) -> Optional[Serializable]:
    if transform.get("__class_fullname__") in NON_SERIALIZABLE_REGISTRY:
        name = transform["__name__"]
        if nonserializable is None:
            msg = f"To deserialize a non-serializable transform with name {name} you need to pass a dict with"
            "this transform as the `lambda_transforms` argument"
            raise ValueError(msg)
        result_transform = nonserializable.get(name)
        if transform is None:
            raise ValueError(f"Non-serializable transform with {name} was not found in `nonserializable`")
        return result_transform
    return None


def from_dict(
    transform_dict: Dict[str, Any],
    nonserializable: Optional[Dict[str, Any]] = None,
) -> Optional[Serializable]:
    """Args:
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


def serialize_enum(obj: Any) -> Any:
    """Recursively search for Enum objects and convert them to their value.
    Also handle any Mapping or Sequence types.
    """
    if isinstance(obj, Mapping):
        return {k: serialize_enum(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, str):  # exclude strings since they're also sequences
        return [serialize_enum(v) for v in obj]
    return obj.value if isinstance(obj, Enum) else obj


def save(
    transform: "Serializable",
    filepath_or_buffer: Union[str, Path, TextIO],
    data_format: str = "json",
    on_not_implemented_error: str = "raise",
) -> None:
    """Serialize a transform pipeline and save it to either a file specified by a path or a file-like object
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
    transform_dict = serialize_enum(transform_dict)

    # Determine whether to write to a file or a file-like object
    if isinstance(filepath_or_buffer, (str, Path)):  # It's a filepath
        with open(filepath_or_buffer, "w") as f:
            if data_format == "yaml":
                if not yaml_available:
                    msg = "You need to install PyYAML to save a pipeline in YAML format"
                    raise ValueError(msg)
                yaml.safe_dump(transform_dict, f, default_flow_style=False)
            elif data_format == "json":
                json.dump(transform_dict, f)
    elif data_format == "yaml":
        if not yaml_available:
            msg = "You need to install PyYAML to save a pipeline in YAML format"
            raise ValueError(msg)
        yaml.safe_dump(transform_dict, filepath_or_buffer, default_flow_style=False)
    elif data_format == "json":
        json.dump(transform_dict, filepath_or_buffer)


def load(
    filepath_or_buffer: Union[str, Path, TextIO],
    data_format: str = "json",
    nonserializable: Optional[Dict[str, Any]] = None,
) -> object:
    """Load a serialized pipeline from a file or file-like object and construct a transform pipeline.

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
                    msg = "You need to install PyYAML to load a pipeline in yaml format"
                    raise ValueError(msg)
                transform_dict = yaml.safe_load(f)
    elif data_format == "json":
        transform_dict = json.load(filepath_or_buffer)
    else:
        if not yaml_available:
            msg = "You need to install PyYAML to load a pipeline in yaml format"
            raise ValueError(msg)
        transform_dict = yaml.safe_load(filepath_or_buffer)

    return from_dict(transform_dict, nonserializable=nonserializable)


def register_additional_transforms() -> None:
    """Register transforms that are not imported directly into the `albumentations` module by checking
    the availability of optional dependencies.
    """
    if importlib.util.find_spec("torch") is not None:
        try:
            # Import `albumentations.pytorch` only if `torch` is installed.
            import albumentations.pytorch

            # Use a dummy operation to acknowledge the use of the imported module and avoid linting errors.
            _ = albumentations.pytorch.ToTensorV2
        except ImportError:
            pass


def get_shortest_class_fullname(cls: Type[Any]) -> str:
    """The function `get_shortest_class_fullname` takes a class object as input and returns its shortened
    full name.

    :param cls: The parameter `cls` is of type `Type[BasicCompose]`, which means it expects a class that
    is a subclass of `BasicCompose`
    :type cls: Type[BasicCompose]
    :return: a string, which is the shortened version of the full class name.
    """
    class_fullname = f"{cls.__module__}.{cls.__name__}"
    return shorten_class_name(class_fullname)
