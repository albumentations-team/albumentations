#!/usr/bin/env python3
"""Pre-commit hook to check that classes inheriting from BaseModel (like InitSchema)
do not have default values in their field definitions.

This enforces the albumentations coding guideline:
"We do not have ANY default values in the InitSchema class"
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


class DefaultValueChecker(ast.NodeVisitor):
    def __init__(self):
        self.errors: list[tuple[str, int, str]] = []
        self.current_file = ""
        self.basemodel_classes: set[str] = set()
        self.class_inheritance: dict[str, list[str]] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        """Visit a class definition node to check for BaseModel inheritance."""
        # Track class inheritance
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                # Handle cases like pydantic.BaseModel
                base_names.append(ast.unparse(base))

        self.class_inheritance[node.name] = base_names

        # Check if this class inherits from BaseModel (directly or indirectly)
        if self._inherits_from_basemodel(node.name):
            self.basemodel_classes.add(node.name)
            self._check_class_fields(node)

        self.generic_visit(node)

    def _inherits_from_basemodel(self, class_name: str) -> bool:
        """Check if a class inherits from BaseModel directly or indirectly."""
        if class_name not in self.class_inheritance:
            return False

        bases = self.class_inheritance[class_name]

        # Direct inheritance
        for base in bases:
            if base in ("BaseModel", "pydantic.BaseModel", "BaseTransformInitSchema"):
                return True

        # Indirect inheritance (recursive check)
        return any(base in self.class_inheritance and self._inherits_from_basemodel(base) for base in bases)

    def _check_class_fields(self, node: ast.ClassDef) -> None:  # noqa: C901, PLR0912
        """Check for default values in class field annotations."""
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and item.value is not None:
                # This is an annotated assignment with a default value
                field_name = ast.unparse(item.target) if hasattr(ast, "unparse") else str(item.target)

                # Skip special fields that are allowed to have defaults
                if self._is_allowed_default_field(field_name):
                    continue

                # Skip discriminator fields (Literal types used for Pydantic discriminated unions)
                if self._is_discriminator_field(item):
                    continue

                # Check if it's a Field() call with default
                if isinstance(item.value, ast.Call):
                    if isinstance(item.value.func, ast.Name) and item.value.func.id == "Field":
                        # Check if Field() has a default parameter or positional arg
                        has_default = False

                        # Check positional arguments (first arg is default if present)
                        if item.value.args:
                            has_default = True

                        # Check keyword arguments for 'default'
                        for keyword in item.value.keywords:
                            if keyword.arg == "default":
                                has_default = True
                                break

                        if has_default:
                            self.errors.append(
                                (
                                    self.current_file,
                                    item.lineno,
                                    f"Field '{field_name}' in BaseModel class '{node.name}' has a default value",
                                ),
                            )
                else:
                    # Direct assignment (not Field())
                    self.errors.append(
                        (
                            self.current_file,
                            item.lineno,
                            f"Field '{field_name}' in BaseModel class '{node.name}' has a default value",
                        ),
                    )

            elif isinstance(item, ast.Assign):
                # Handle regular assignments (var = value)
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        field_name = target.id
                        if not self._is_allowed_default_field(field_name):
                            self.errors.append(
                                (
                                    self.current_file,
                                    item.lineno,
                                    f"Field '{field_name}' in BaseModel class '{node.name}' has a default value",
                                ),
                            )

    def _is_allowed_default_field(self, field_name: str) -> bool:
        """Check if a field is allowed to have default values."""
        # Allow private fields, class variables, and special methods
        if field_name.startswith("_"):
            return True

        # Allow specific field names that might legitimately have defaults
        allowed_fields = {
            "model_config",  # Pydantic config
            "strict",  # Core validation system field
            "__annotations__",
            "__module__",
            "__qualname__",
        }

        return field_name in allowed_fields

    def _is_discriminator_field(self, item: ast.AnnAssign) -> bool:
        """Check if this is a discriminator field for Pydantic discriminated unions."""
        if not item.annotation:
            return False

        # Check if the annotation is a Literal type
        annotation_str = ast.unparse(item.annotation) if hasattr(ast, "unparse") else str(item.annotation)

        # Look for Literal["some_value"] pattern
        if "Literal[" in annotation_str and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
            literal_value = item.value.value
            # Check if the literal value appears in the annotation
            if f'"{literal_value}"' in annotation_str or f"'{literal_value}'" in annotation_str:
                return True

        return False


def check_file(file_path: Path) -> list[tuple[str, int, str]]:
    """Check a single Python file for default values in BaseModel classes."""
    try:
        with file_path.open(encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        checker = DefaultValueChecker()
        checker.current_file = str(file_path)
        checker.visit(tree)

    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return []
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error processing {file_path}: {e}")
        return []
    else:
        return checker.errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that BaseModel classes don't have default values",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Python files to check",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Exclude files matching this pattern",
    )

    args = parser.parse_args()

    if not args.files:
        return 0

    all_errors = []

    for file_path in args.files:
        path = Path(file_path)

        # Skip non-Python files
        if path.suffix != ".py":
            continue

        # Skip excluded patterns
        skip = False
        for pattern in args.exclude_pattern:
            if pattern in str(path):
                skip = True
                break
        if skip:
            continue

        errors = check_file(path)
        all_errors.extend(errors)

    # Report errors
    if all_errors:
        print("❌ Found default values in BaseModel classes:")
        for file_path, line_no, message in all_errors:
            print(f"  {file_path}:{line_no}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
