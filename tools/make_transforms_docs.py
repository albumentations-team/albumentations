from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
from pathlib import Path

sys.path.append("..")
import albumentations
from albumentations.core.type_definitions import Targets

IGNORED_CLASSES = {
    "BasicTransform",
    "DualTransform",
    "ImageOnlyTransform",
    "Transform3D",
}


def make_augmentation_docs_link(cls) -> str:
    return f"[{cls.__name__}](https://explore.albumentations.ai/transform/{cls.__name__})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Commands", dest="command")
    subparsers.add_parser("make")
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("filepath", type=str, help="Path to a file that should be checked")
    return parser.parse_args()


def make_separator(width: int, align_center: bool) -> str:
    if align_center:
        return ":" + "-" * (width - 2) + ":"
    return "-" * width


def is_deprecated(cls) -> bool:
    """Check if a given class is deprecated by looking for deprecation notices at the start of the docstring,
    not in the Args section.
    """
    if not cls.__doc__:
        return False

    # Split docstring into sections and look only at the first section (before Args:)
    main_desc = cls.__doc__.split("Args:")[0]

    # Check if there's a deprecation notice in the main description
    return any("deprecated" in line.lower() for line in main_desc.split("\n") if line.strip())


def get_image_only_transforms_info():
    image_only_info = {}
    members = inspect.getmembers(albumentations)
    for name, cls in members:
        if (
            inspect.isclass(cls)
            and issubclass(cls, albumentations.ImageOnlyTransform)
            and not issubclass(cls, albumentations.Transform3D)
            and name not in IGNORED_CLASSES
        ) and not is_deprecated(cls):
            image_only_info[name] = {
                "docs_link": make_augmentation_docs_link(cls),
            }
    return image_only_info


def get_dual_transforms_info():
    dual_transforms_info = {}
    members = inspect.getmembers(albumentations)
    for name, cls in members:
        if (
            inspect.isclass(cls)
            and issubclass(cls, albumentations.DualTransform)
            and not issubclass(cls, albumentations.Transform3D)  # Exclude 3D transforms
            and name not in IGNORED_CLASSES
        ) and not is_deprecated(cls):
            dual_transforms_info[name] = {
                "targets": cls._targets,
                "docs_link": make_augmentation_docs_link(cls),
            }
    return dual_transforms_info


def get_3d_transforms_info():
    transforms_3d_info = {}
    members = inspect.getmembers(albumentations)
    for name, cls in members:
        if (
            inspect.isclass(cls) and issubclass(cls, albumentations.Transform3D) and name not in IGNORED_CLASSES
        ) and not is_deprecated(cls):
            # Get targets from class or parent class if not defined
            targets = cls._targets if hasattr(cls, "_targets") else albumentations.Transform3D._targets

            transforms_3d_info[name] = {
                "targets": targets if isinstance(targets, tuple) else (targets,),
                "docs_link": make_augmentation_docs_link(cls),
            }
    return transforms_3d_info


def make_transforms_targets_table(transforms_info, header, targets_to_check=None):
    rows = [header]
    for transform, info in sorted(transforms_info.items(), key=lambda kv: kv[0]):
        transform_targets = []
        targets_iter = targets_to_check or Targets
        for target in targets_iter:
            mark = "✓" if target in info["targets"] else ""
            transform_targets.append(mark)
        row = [info["docs_link"] or transform, *transform_targets]
        rows.append(row)

    column_widths = [max(len(r) for r in column) for column in zip(*rows)]
    lines = [
        " | ".join(
            "{title: <{width}}".format(width=width, title=title) for width, title in zip(column_widths, rows[0])
        ),
        " | ".join(
            make_separator(width, align_center=column_index > 0) for column_index, width in enumerate(column_widths)
        ),
    ]
    lines.extend(
        " | ".join("{column: <{width}}".format(width=width, column=column) for width, column in zip(column_widths, row))
        for row in rows[1:]
    )
    return "\n".join(f"| {line} |" for line in lines)


def make_transforms_targets_links(transforms_info):
    return "\n".join("- " + info["docs_link"] for _, info in sorted(transforms_info.items(), key=lambda kv: kv[0]))


def check_docs(
    filepath: str,
    image_only_transforms_links: str,
    dual_transforms_table: str,
    transforms_3d_table: str,
) -> None:
    """Check if the documentation file is up to date with the current transforms.

    Args:
        filepath (str): Path to the documentation file (README.md)
        image_only_transforms_links (str): Generated links for pixel-level transforms
        dual_transforms_table (str): Generated table for spatial-level transforms
        transforms_3d_table (str): Generated table for 3D transforms

    Raises:
        ValueError: If any section is outdated with detailed information about missing lines

    """
    with Path(filepath).open(encoding="utf8") as f:
        text = f.read()

    # Find sections using regex
    sections = {
        "Pixel-level": {
            "pattern": r"### Pixel-level transforms\n\n(.*?)(?=###|\Z)",
            "generated": image_only_transforms_links,
            "lines_not_in_text": [],
        },
        "Spatial-level": {
            "pattern": r"### Spatial-level transforms\n\n(.*?)(?=###|\Z)",
            "generated": dual_transforms_table,
            "lines_not_in_text": [],
        },
        "3D": {
            "pattern": r"### 3D transforms\n\n(.*?)(?=###|\Z)",
            "generated": transforms_3d_table,
            "lines_not_in_text": [],
        },
    }

    outdated_docs = set()

    # Check each section
    for section_name, section_info in sections.items():
        # Find section content
        match = re.search(section_info["pattern"], text, re.DOTALL)
        if not match:
            outdated_docs.add(section_name)
            section_info["lines_not_in_text"].extend(
                section_info["generated"].split("\n"),
            )
            continue

        # Check if all generated lines are in the section
        section_content = match[1].strip()
        for line in section_info["generated"].split("\n"):
            if line.strip() and line not in section_content:
                outdated_docs.add(section_name)
                section_info["lines_not_in_text"].append(line)

    # If any sections are outdated, raise error with detailed information
    if outdated_docs:
        msg = (
            "Docs for the following transform types are outdated: {outdated_docs_headers}.\n"
            "Generate new docs by executing the `python -m tools.{py_file} make` command "
            "and paste them to {filename}.\n\n"
        ).format(
            outdated_docs_headers=", ".join(sorted(outdated_docs)),
            py_file=Path(os.path.realpath(__file__)).name,
            filename=Path(filepath).name,
        )

        # Add missing lines for each outdated section
        for section_name, section_info in sections.items():
            if section_name in outdated_docs:
                msg += (
                    f"# {section_name} transforms lines not in file:\n"
                    f"{chr(10).join(section_info['lines_not_in_text'])}\n\n"
                )

        raise ValueError(msg.strip())


def main() -> None:
    args = parse_args()
    command = args.command
    if command not in {"make", "check"}:
        raise ValueError(f"You should provide a valid command: {{make|check}}. Got {command} instead.")

    image_only_transforms = get_image_only_transforms_info()
    dual_transforms = get_dual_transforms_info()
    transforms_3d = get_3d_transforms_info()

    image_only_transforms_links = make_transforms_targets_links(image_only_transforms)
    dual_transforms_table = make_transforms_targets_table(
        dual_transforms,
        header=["Transform"] + [target.value for target in Targets],
    )

    transforms_3d_table = make_transforms_targets_table(
        transforms_3d,
        header=["Transform"] + [target.value for target in [Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS]],
        targets_to_check=[Targets.VOLUME, Targets.MASK3D, Targets.KEYPOINTS],
    )

    if command == "make":
        print("===== COPY THIS TABLE TO README.MD BELOW ### Pixel-level transforms =====")
        print(image_only_transforms_links)
        print("===== END OF COPY =====")
        print()
        print("===== COPY THIS TABLE TO README.MD BELOW ### Spatial-level transforms =====")
        print(dual_transforms_table)
        print("===== END OF COPY =====")
        print()
        print("===== COPY THIS TABLE TO README.MD BELOW ### 3D transforms =====")
        print(transforms_3d_table)
        print("===== END OF COPY =====")
    else:
        check_docs(
            args.filepath,
            image_only_transforms_links,
            dual_transforms_table,
            transforms_3d_table,
        )


if __name__ == "__main__":
    main()
