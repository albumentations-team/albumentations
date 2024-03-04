import re
import sys
from pathlib import Path

def check_docstrings_for_dashes(file_path):
    pattern = re.compile(r'["\']{3}[\s\S]+?["\']{3}')  # Regex to match docstrings
    dash_pattern = re.compile(r'---{2,}')  # Regex to match sequences of ---

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        matches = pattern.findall(content)
        for match in matches:
            if dash_pattern.search(match):
                return False  # Found forbidden sequence
    return True  # No forbidden sequences found

def main():
    exit_code = 0
    for file_path in sys.argv[1:]:
        if not check_docstrings_for_dashes(file_path):
            print(f"Error in {file_path}: According to Google Style docstrings, '---' should not be used to underline sections. "
                  "Please refer to https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings")
            exit_code = 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
