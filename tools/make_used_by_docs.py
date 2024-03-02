import argparse
import requests

def fetch_data(json_url: str) -> list:
    try:
        response = requests.get(json_url)
        response.raise_for_status()  # Raises an exception for 4XX/5XX responses
        return response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch JSON data: {e}")

def check_used_by_section(filepath: str, expected_html: list) -> None:
    try:
        with open(filepath, encoding="utf-8") as file:
            readme_content = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    combined_html = "\n".join(expected_html) + "\n"
    if combined_html not in readme_content:
        raise ValueError(
            "The 'Who is using Albumentations' section is outdated or incorrect. "
            "Please run 'python tools/make_used_by_docs.py update' to automatically update "
            "the README.md file in the ## Who is using Albumentations section."
        )

def update_used_by_section(filepath: str, new_html: list) -> None:
    combined_html = "\n".join(new_html) + "\n"
    start_marker = "## Who is using Albumentations\n"
    end_marker = "\n## "

    try:
        with open(filepath, encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    start_index = content.find(start_marker)
    if start_index != -1:
        start_index += len(start_marker)
        end_index = content.find(end_marker, start_index)
        end_index = end_index if end_index != -1 else len(content)

        new_content = content[:start_index] + "\n" + combined_html + content[end_index:]
        try:
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(new_content)
        except IOError as e:
            raise IOError(f"Failed to update the file {filepath}: {e}")
    else:
        raise ValueError("Could not find the 'Who is using Albumentations' section in the README.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate, check or update 'Who is using Albumentations' section.")
    parser.add_argument(
        "action",
        choices=["make", "check", "update"],
        help="Specify 'make' to generate HTML, 'check' to check the README.md file, or 'update' to update the README.md file."
    )
    parser.add_argument('files', nargs='*', help="Files to ignore (from pre-commit).")
    args = parser.parse_args()

    json_url = "https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/data/img_industry.json"
    base_img_url = "https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/"

    data = fetch_data(json_url)
    html_list = [f'<a href="{entry["url"]}" target="_blank"><img src="{base_img_url + entry["img_filename"]}" width="100"/></a>' for entry in data]

    if args.action == "make":
        for html in html_list:
            print(html)
    elif args.action == "check":
        check_used_by_section("README.md", html_list)
    elif args.action == "update":
        update_used_by_section("README.md", html_list)

if __name__ == "__main__":
    main()
