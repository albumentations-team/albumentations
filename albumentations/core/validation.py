from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Callable
from warnings import warn

from pydantic import BaseModel


class ValidatedTransformMeta(type):
    def __new__(cls: type[Any], name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> type[Any]:
        if "InitSchema" in dct and issubclass(dct["InitSchema"], BaseModel):
            original_init: Callable[..., Any] | None = dct.get("__init__")
            if original_init is None:
                msg = "__init__ not found in class definition"
                raise ValueError(msg)

            original_sig = signature(original_init)

            def custom_init(self: Any, *args: Any, **kwargs: Any) -> None:
                init_params = signature(original_init).parameters
                param_names = list(init_params.keys())[1:]  # Exclude 'self'
                full_kwargs: dict[str, Any] = dict(zip(param_names, args))
                full_kwargs.update(kwargs)

                # Get strict value before validation
                strict = full_kwargs.pop("strict", False)  # Remove strict from kwargs

                for parameter_name, parameter in init_params.items():
                    if (
                        parameter_name != "self"
                        and parameter_name not in full_kwargs
                        and parameter.default is not Parameter.empty
                    ):
                        full_kwargs[parameter_name] = parameter.default

                # Configure model validation
                try:
                    config = dct["InitSchema"](**{k: v for k, v in full_kwargs.items() if k in param_names})
                    validated_kwargs = config.model_dump()
                    # Remove strict from validated kwargs to prevent it from being passed to __init__
                    validated_kwargs.pop("strict", None)
                except Exception as e:
                    if strict:
                        raise
                    warn(str(e), stacklevel=2)
                    # Use default values for invalid parameters
                    config = dct["InitSchema"]()
                    validated_kwargs = config.model_dump()
                    validated_kwargs.pop("strict", None)  # Also remove from default values

                # Store invalid args in the instance
                invalid_args = [
                    name_arg for name_arg in kwargs if name_arg not in validated_kwargs and name_arg != "strict"
                ]

                # Call original init with validated kwargs (strict removed)
                original_init(self, **validated_kwargs)

                # Store invalid args after initialization
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
