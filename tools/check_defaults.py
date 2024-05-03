import sys
import inspect
import albumentations
from albumentations.core.transforms_interface import BasicTransform
from typing import Union, get_origin, get_args

def check_apply_methods(cls):
    """Check for issues in 'apply' methods related to default arguments and Optional type annotations."""
    issues = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith('apply'):
            signature = inspect.signature(method)
            for param in signature.parameters.values():
                # Check for default values
                if param.default is not inspect.Parameter.empty:
                    issues.append(f"Default argument found in {cls.__name__}.{name} for parameter {param.name} with default value {param.default}")
                # Check for Optional type annotations
                if check_optional_annotation(param.annotation):
                    issues.append(f"Optional type annotation found in {cls.__name__}.{name} for parameter {param.name}")
    return issues

def check_optional_annotation(annotation):
    """Check if the annotation is Optional[...]"""
    origin = get_origin(annotation)
    if origin is Union:
        return any(arg is type(None) for arg in get_args(annotation))
    return False

def is_subclass_of_basic_transform(cls):
    """Check if a given class is a subclass of BasicTransform, excluding BasicTransform itself."""
    return issubclass(cls, BasicTransform) and cls is not BasicTransform

def main():
    issues = []
    # Check all classes in the albumentations module
    for name, cls in inspect.getmembers(albumentations, predicate=inspect.isclass):
        if is_subclass_of_basic_transform(cls):
            issues.extend(check_apply_methods(cls))

    if issues:
        print("\n".join(issues))
        sys.exit(1)  # Exit with error status 1 if there are any issues

if __name__ == "__main__":
    main()
