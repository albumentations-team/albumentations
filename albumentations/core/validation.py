"""Module containing validation mechanisms for transform parameters.

This module provides a metaclass that enables parameter validation for transforms using
Pydantic models. It intercepts the initialization of transform classes to validate their
parameters against schema definitions, raising appropriate errors for invalid values and
providing type conversion capabilities. This validation layer helps prevent runtime errors
by catching configuration issues at initialization time.
"""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Callable
from warnings import warn

from pydantic import BaseModel, ValidationError


class ValidatedTransformMeta(type):
    """Metaclass that validates transform parameters during instantiation.

    This metaclass enables automatic validation of transform parameters using Pydantic models,
    ensuring proper typing and constraints are enforced before object creation.

    Args:
        original_init (Callable[..., Any]): Original __init__ method of the class.
        args (tuple[Any, ...]): Positional arguments passed to the __init__ method.
        kwargs (dict[str, Any]): Keyword arguments passed to the __init__ method.

    """

    @staticmethod
    def _process_init_parameters(
        original_init: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str], bool]:
        init_params = signature(original_init).parameters
        param_names = list(init_params.keys())[1:]  # Exclude 'self'
        full_kwargs: dict[str, Any] = dict(zip(param_names, args)) | kwargs

        # Get strict value before validation
        strict = full_kwargs.pop("strict", False)

        # Add default values if not provided
        for parameter_name, parameter in init_params.items():
            if (
                parameter_name != "self"
                and parameter_name not in full_kwargs
                and parameter.default is not Parameter.empty
            ):
                full_kwargs[parameter_name] = parameter.default

        return full_kwargs, param_names, strict

    @staticmethod
    def _validate_parameters(
        schema_cls: type[BaseModel],
        full_kwargs: dict[str, Any],
        param_names: list[str],
        strict: bool,
    ) -> dict[str, Any]:
        try:
            # Include strict parameter for schema validation
            schema_kwargs = {k: v for k, v in full_kwargs.items() if k in param_names}
            schema_kwargs["strict"] = strict
            config = schema_cls(**schema_kwargs)
            validated_kwargs = config.model_dump()
            validated_kwargs.pop("strict", None)
        except ValidationError as e:
            raise ValueError(str(e)) from e
        except Exception as e:
            if strict:
                raise ValueError(str(e)) from e
            warn(str(e), stacklevel=2)
            return {}
        else:
            return validated_kwargs

    @staticmethod
    def _get_default_values(init_params: dict[str, Parameter]) -> dict[str, Any]:
        validated_kwargs = {}
        for param_name, param in init_params.items():
            if param_name in {"self", "strict"}:
                continue
            if param.default is not Parameter.empty:
                validated_kwargs[param_name] = param.default
        return validated_kwargs

    def __new__(cls: type[Any], name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> type[Any]:
        """This is a custom metaclass that validates the parameters of the class during instantiation.
        It is used to ensure that the parameters of the class are valid and that they are of the correct type.
        """
        if "InitSchema" in dct and issubclass(dct["InitSchema"], BaseModel):
            original_init: Callable[..., Any] | None = dct.get("__init__")
            if original_init is None:
                msg = "__init__ not found in class definition"
                raise ValueError(msg)

            original_sig = signature(original_init)

            def custom_init(self: Any, *args: Any, **kwargs: Any) -> None:
                full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)

                validated_kwargs = cls._validate_parameters(
                    dct["InitSchema"],
                    full_kwargs,
                    param_names,
                    strict,
                ) or cls._get_default_values(signature(original_init).parameters)

                # Store and check invalid args
                invalid_args = [name_arg for name_arg in kwargs if name_arg not in param_names and name_arg != "strict"]
                original_init(self, **validated_kwargs)
                self.invalid_args = invalid_args

                if invalid_args:
                    message = f"Argument(s) '{', '.join(invalid_args)}' are not valid for transform {name}"
                    if strict:
                        raise ValueError(message)
                    warn(message, stacklevel=2)

            # Preserve the original signature and docstring
            custom_init.__signature__ = original_sig  # type: ignore[attr-defined]
            custom_init.__doc__ = original_init.__doc__

            dct["__init__"] = custom_init

        return super().__new__(cls, name, bases, dct)
