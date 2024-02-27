import ast
import sys


def check_docstring_types(file_path):
    with open(file_path) as source:
        tree = ast.parse(source.read(), filename=file_path)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
                docstring = ast.get_docstring(node)
                if docstring:
                    # Example check: see if "Args:" is in docstring (for Google style)
                    if "Args:" not in docstring:
                        print(f"{file_path}: Function '{node.name}' is missing 'Args:' in its docstring")
                        return False
    return True


def main():
    success = True
    for file_path in sys.argv[1:]:
        if not check_docstring_types(file_path):
            success = False

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
