"""Check for naming conflicts between modules and exported names.

This script detects conflicts between module names and defined function/class names
that could cause problems with frameworks like Hydra that rely on direct module paths.

Example usage:
    python -m tools.check_naming_conflicts
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


def get_module_names(base_dir: str) -> set[str]:
    """Find all submodule names (directories with __init__.py).

    Args:
        base_dir (str): Base directory to search within

    Returns:
        Set[str]: Set of module names (not paths, just the directory names)

    """
    module_names = set()
    for dirpath, _, filenames in os.walk(base_dir):
        if "__init__.py" in filenames:
            # We want the last component of the path as the module name
            module_name = Path(dirpath).name
            if dirpath != base_dir:  # Skip root dir
                module_names.add(module_name)
    return module_names


def _extract_names_from_node(node) -> set[str]:
    """Extract names from AST nodes."""
    names = set()
    # Classes and functions
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
        if not node.name.startswith("_"):
            names.add(node.name)
    # Variables
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                names.add(target.id)
    return names


def get_defined_names(base_dir: str) -> set[str]:
    """Find all top-level names defined in Python files that would be exported with *.

    Args:
        base_dir (str): Base directory to search within

    Returns:
        Set[str]: Set of defined names that could be exported via wildcard imports

    """
    defined_names = set()

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".py"):
                continue

            filepath = Path(root) / file

            try:
                with filepath.open(encoding="utf-8") as f:
                    file_content = f.read()

                tree = ast.parse(file_content, filepath)

                # Look for top-level definitions that don't start with underscore
                for node in ast.iter_child_nodes(tree):
                    defined_names.update(_extract_names_from_node(node))
            except (SyntaxError, UnicodeDecodeError, IsADirectoryError) as e:
                print(f"Error parsing {filepath}: {e}", file=sys.stderr)

    return defined_names


def find_conflicts(base_dir: str = "albumentations") -> tuple[set[str], set[str], set[str]]:
    """Find conflicts between module names and defined names.

    Args:
        base_dir (str): Base directory to check

    Returns:
        Tuple[Set[str], Set[str], Set[str]]: Tuple containing (module_names, defined_names, conflicts)

    """
    module_names = get_module_names(base_dir)
    defined_names = get_defined_names(base_dir)

    conflicts = module_names.intersection(defined_names)

    return module_names, defined_names, conflicts


def main():
    """Main entry point for the script."""
    base_dir = "albumentations"

    # Check if base directory exists
    if not Path(base_dir).is_dir():
        print(f"Error: Directory '{base_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    _, _, conflicts = find_conflicts(base_dir)

    if conflicts:
        print("⚠️ Naming conflicts detected between modules and defined names:", file=sys.stderr)
        for conflict in sorted(conflicts):
            print(f"  - '{conflict}' is both a module name and a function/class", file=sys.stderr)
        print("\nThese conflicts can cause problems with tools like Hydra that use direct module paths.")
        print("Consider renaming either the module or the function/class.")
        sys.exit(1)

    print("✅ No naming conflicts detected between modules and defined names.")
    sys.exit(0)


if __name__ == "__main__":
    main()
