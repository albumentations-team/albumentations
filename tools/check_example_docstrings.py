#!/usr/bin/env python3

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

from google_docstring_parser import parse_google_docstring

TARGET_PARENT_CLASSES = {"DualTransform", "ImageOnlyTransform", "Transform3D"}

# We'll check for both, but have different error messages
EXAMPLES_SECTION = "Examples"
EXAMPLE_SECTION = "Example"


def is_target_class(cls) -> bool:
    """Check if a class inherits from one of the target classes."""
    # Skip if the class itself is one of the target classes
    if cls.__name__ in TARGET_PARENT_CLASSES:
        return False

    # Get all base classes in the class's MRO (Method Resolution Order)
    try:
        bases = [base.__name__ for base in inspect.getmro(cls)]
        # Class should inherit from target classes but not be one itself
        return any(base in TARGET_PARENT_CLASSES for base in bases)
    except TypeError:
        # If we can't get the MRO, we'll use the AST-based approach
        # This will be handled in check_file
        return False


def build_inheritance_map(file_path: str) -> dict[str, list[str]]:
    """Build a map of class names to their direct parent class names."""
    with Path(file_path).open(encoding="utf-8") as f:
        tree = ast.parse(f.read())

    inheritance_map = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Get direct parent class names
            parent_names = [base.id for base in node.bases if isinstance(base, ast.Name)]
            inheritance_map[node.name] = parent_names

    return inheritance_map


def has_target_ancestor(
    class_name: str,
    inheritance_map: dict[str, list[str]],
    visited: set[str] | None = None,
) -> bool:
    """Recursively check if a class has any target class in its ancestry."""
    if visited is None:
        visited = set()

    # Avoid cycles in inheritance
    if class_name in visited:
        return False
    visited.add(class_name)

    # Base case: this is a target class
    if class_name in TARGET_PARENT_CLASSES:
        return True

    # Get direct parents
    parents = inheritance_map.get(class_name, [])

    return any(has_target_ancestor(parent, inheritance_map, visited) for parent in parents)


def check_docstring(docstring: str, class_name: str) -> list[tuple[str, str]]:
    """Check the docstring for a proper Examples section."""
    errors = []

    if not docstring:
        errors.append((class_name, "Missing docstring"))
        return errors

    try:
        parsed = parse_google_docstring(docstring)

        # First check if 'Example' is used instead of 'Examples'
        if EXAMPLE_SECTION in parsed and EXAMPLES_SECTION not in parsed:
            errors.append((class_name, f"Using '{EXAMPLE_SECTION}' instead of '{EXAMPLES_SECTION}' - use plural form"))
        # Then check if neither is present
        elif all(section not in parsed for section in [EXAMPLES_SECTION, EXAMPLE_SECTION]):
            errors.append((class_name, f"Missing '{EXAMPLES_SECTION}' section in docstring"))
    except (ValueError, AttributeError, TypeError) as e:
        errors.append((class_name, f"Error parsing docstring: {e!s}"))

    return errors


def check_file(file_path: str) -> list[tuple[str, str]]:
    """Check a file for classes that need examples in their docstrings."""
    errors = []

    try:
        # Try to import the module
        module_name = Path(file_path).stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            return []

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find all classes in the module
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__ and is_target_class(obj):
                docstring = inspect.getdoc(obj)
                errors.extend(check_docstring(docstring, obj.__name__))
    except (ImportError, AttributeError, ModuleNotFoundError, SyntaxError):
        # If module import fails, use AST to check
        with Path(file_path).open(encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        inheritance_map = build_inheritance_map(file_path)

        # Find all class definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Skip if class itself is a target class
                if node.name in TARGET_PARENT_CLASSES:
                    continue

                # Check if this class has a target class in its ancestry
                if has_target_ancestor(node.name, inheritance_map):
                    docstring = ast.get_docstring(node)
                    errors.extend(check_docstring(docstring, node.name))

    return errors


def main():
    """Main function for the pre-commit hook."""
    files = sys.argv[1:] if len(sys.argv) > 1 else []
    has_errors = False
    all_errors = []

    for file_path in files:
        if not file_path.endswith(".py"):
            continue

        errors = check_file(file_path)
        if errors:
            has_errors = True
            all_errors.append((file_path, errors))

    # Print all errors
    if all_errors:
        for file_path, errors in all_errors:
            file_rel_path = file_path.replace(str(Path.cwd()) + "/", "")
            print(f"\n{file_rel_path}:")
            for class_name, message in errors:
                print(f"  - {class_name}: {message}")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
