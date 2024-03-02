import argparse

import requests


def check_used_by_section(filepath: str, expected_html: list) -> None:
    """Check the 'Who is using Albumentations' section in the README."""
    with open(filepath, encoding="utf-8") as file:
        readme_content = file.read()

    # Combine all HTML strings into one for easier checking
    combined_html = "\n".join(expected_html) + "\n"

    # Check if the combined HTML strings are in the README content
    if combined_html not in readme_content:
        raise ValueError(
            "The 'Who is using Albumentations' section is outdated or incorrect. "
            "Please run 'python tools/make_used_by_docs.py make' to generate the latest section "
            "and update the README.md file in the ## Who is using Albumentations section."
        )


def update_used_by_section(filepath: str, new_html: list) -> None:
    """Update the 'Who is using Albumentations' section in the README."""
    # Combine all HTML strings into one
    combined_html = "\n".join(new_html) + "\n"

    with open(filepath, encoding="utf-8") as file:
        content = file.read()

    # Markers to identify the section in the README
    start_marker = "## Who is using Albumentations\n"
    end_marker = "\n## "  # Assuming each section starts with '## '

    # Use find method to locate the start and end of the section
    start_index = content.find(start_marker)
    if start_index != -1:
        start_index += len(start_marker)  # Move index to the end of the start marker to replace content after it
        end_index = content.find(end_marker, start_index)
        if end_index == -1:  # If no section follows, end at the end of the file
            end_index = len(content)

        # Update the section if found
        new_content = content[:start_index] + "\n" + combined_html + content[end_index:]
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(new_content)
    else:
        raise ValueError("Could not find the 'Who is using Albumentations' section in the README.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate, check or update 'Who is using Albumentations' section.")
    parser.add_argument(
        "action",
        choices=["make", "check", "update"],
        help="Specify 'make' to generate HTML, 'check' to check the README.md file, or 'update' to update the README.md file.",
    )
    parser.add_argument('files', nargs='*', help="Files to ignore (from pre-commit).")  # This line allows additional arguments
    args = parser.parse_args()

    # URLs for JSON data and base images
    json_url = "https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/data/img_industry.json"
    base_img_url = (
        "https://raw.githubusercontent.com/albumentations-team/albumentations.ai/main/html/assets/img/industry/"
    )

    # Fetch the JSON data
    response = requests.get(json_url)
    if response.status_code == 200:  # successful request
        data = response.json()  # parse JSON data

        # Initialize an empty list to store HTML strings
        html_list = []

        # Iterate over each entry in the JSON data
        for entry in data:
            # Construct the full image URL
            img_url = base_img_url + entry["img_filename"]

            # Create the HTML string for this entry
            html_str = f'<a href="{entry["url"]}" target="_blank"><img src="{img_url}" width="100"/></a>'

            # Append the HTML string to the list
            html_list.append(html_str)

        # Perform action based on the specified command
        if args.action == "make":
            for html in html_list:
                print(html)
        elif args.action == "check":
            check_used_by_section("README.md", html_list)
        elif args.action == "update":
            update_used_by_section("README.md", html_list)
    else:
        print(f"Failed to fetch JSON data: status code {response.status_code}")


if __name__ == "__main__":
    main()
