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

                for parameter_name, parameter in init_params.items():
                    if (
                        parameter_name != "self"
                        and parameter_name not in full_kwargs
                        and parameter.default is not Parameter.empty
                    ):
                        full_kwargs[parameter_name] = parameter.default

                # No try-except block needed as we want the exception to propagate naturally
                config = dct["InitSchema"](**full_kwargs)

                validated_kwargs = config.model_dump()
                for name_arg in kwargs:
                    if name_arg not in validated_kwargs:
                        warn(
                            f"Argument '{name_arg}' is not valid and will be ignored.",
                            stacklevel=2,
                        )

                original_init(self, **validated_kwargs)

            # Preserve the original signature and docstring
            custom_init.__signature__ = original_sig  # type: ignore[attr-defined]
            custom_init.__doc__ = original_init.__doc__

            # Rename __init__ to custom_init to avoid the N807 warning
            dct["__init__"] = custom_init

        return super().__new__(cls, name, bases, dct)
