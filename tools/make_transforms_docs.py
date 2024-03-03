import argparse
import inspect
import os
import sys
from enum import Enum

sys.path.append("..")
import albumentations

IGNORED_CLASSES = {
    "BasicTransform",
    "DualTransform",
    "ImageOnlyTransform",
    "ReferenceBasedTransform"
}


def make_augmentation_docs_link(cls) -> str:
    module_parts = cls.__module__.split(".")
    module_page = "/".join(module_parts[1:])
    return (
        f"[{cls.__name__}](https://albumentations.ai/docs/api_reference/{module_page}/#{cls.__module__}.{cls.__name__})"
    )


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

import warnings

def is_deprecated(cls) -> bool:
    """
    Check if a given class is deprecated.
    """
    for warning in cls.__doc__.split('\n'):  # Assuming deprecation warnings are in the docstring
        if "deprecated" in warning.lower():
            return True  # The class itself is marked as deprecated
    return False

def get_targets(cls,):
    targets = {Targets.IMAGE, Targets.MASKS}

    has_bboxes_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(albumentations.ReferenceBasedTransform, attr, None)
        for attr in ["apply_to_bbox", "apply_to_bboxes"]
    )
    if has_bboxes_method:
        targets.add(Targets.BBOXES)

    has_keypoints_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(albumentations.ReferenceBasedTransform, attr, None)
        for attr in ["apply_to_keypoint", "apply_to_keypoints"]
    )
    if has_keypoints_method:
        targets.add(Targets.KEYPOINTS)

    has_global_label_method = any(
        hasattr(cls, attr) and getattr(cls, attr) is not getattr(albumentations.ReferenceBasedTransform, attr, None)
        for attr in ["apply_to_global_label", "apply_to_global_labels"]
    )
    if has_global_label_method:
        targets.add(Targets.GLOBAL_LABEL)

    return targets

def get_image_only_transforms_info():
    image_only_info = {}
    members = inspect.getmembers(albumentations)
    for name, cls in members:
        if inspect.isclass(cls) and issubclass(cls, albumentations.ImageOnlyTransform) and name not in IGNORED_CLASSES:
            if not is_deprecated(cls):
                image_only_info[name] = {
                    "docs_link": make_augmentation_docs_link(cls)
                }
    return image_only_info

def get_dual_transforms_info():
    dual_transforms_info = {}
    members = inspect.getmembers(albumentations)
    for name, cls in members:
        if inspect.isclass(cls) and issubclass(cls, albumentations.DualTransform) and not issubclass(cls, albumentations.ReferenceBasedTransform) and name not in IGNORED_CLASSES:
            if not is_deprecated(cls):
                dual_transforms_info[name] = {
                    "targets": get_targets(cls),
                    "docs_link": make_augmentation_docs_link(cls)
                }
    return dual_transforms_info


def get_mixing_transforms_info():
    mixing_transforms_info = {}
    members = inspect.getmembers(albumentations)
    for name, cls in members:
        if inspect.isclass(cls) and issubclass(cls, albumentations.ReferenceBasedTransform) and name not in IGNORED_CLASSES:
            if not is_deprecated(cls):
                mixing_transforms_info[name] = {
                    "targets": get_targets(cls),
                    "docs_link": make_augmentation_docs_link(cls)
                }
    return mixing_transforms_info

def get_reference_based_transforms_info():
    reference_based_transforms_info = {}
    members = inspect.getmembers(albumentations)
    for name, cls in members:
        if inspect.isclass(cls) and issubclass(cls, albumentations.ReferenceBasedTransform) and name not in IGNORED_CLASSES:
            if not is_deprecated(cls):
                targets = {Targets.IMAGE, Targets.MASKS, Targets.BBOXES, Targets.KEYPOINTS, Targets.GLOBAL_LABEL}
                reference_based_transforms_info[name] = {
                    "targets": targets,
                    "docs_link": make_augmentation_docs_link(cls)
                }
    return reference_based_transforms_info



def make_transforms_targets_table(transforms_info, header):
    rows = [header]
    for transform, info in sorted(transforms_info.items(), key=lambda kv: kv[0]):
        transform_targets = []
        for target in Targets:
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
    for row in rows[1:]:
        lines.append(
            " | ".join(
                "{column: <{width}}".format(width=width, column=column) for width, column in zip(column_widths, row)
            )
        )
    return "\n".join(f"| {line} |" for line in lines)


def make_transforms_targets_links(transforms_info):
    return "\n".join(
        "- " + info["docs_link"] for _, info in sorted(transforms_info.items(), key=lambda kv: kv[0])
    )


def check_docs(filepath, image_only_transforms_links, dual_transforms_table, mixing_transforms_table) -> None:
    with open(filepath, encoding="utf8") as f:
        text = f.read()

    outdated_docs = set()
    image_only_lines_not_in_text = []
    dual_lines_not_in_text = []
    mixing_lines_not_in_text = []

    for line in image_only_transforms_links.split("\n"):
        if line not in text:
            outdated_docs.update(["Pixel-level"])
            image_only_lines_not_in_text.append(line)

    for line in dual_transforms_table.split("\n"):
        if line not in text:
            dual_lines_not_in_text.append(line)
            outdated_docs.update(["Spatial-level"])

    for line in mixing_transforms_table.split("\n"):
        if line not in text:
            mixing_lines_not_in_text.append(line)
            outdated_docs.update(["Mixing-level"])

    if outdated_docs:
        msg = (
            "Docs for the following transform types are outdated: {outdated_docs_headers}. "
            "Generate new docs by executing the `python tools/{py_file} make` command "
            "and paste them to {filename}.\n"
            "# Pixel-level transforms lines not in file:\n"
            "{image_only_lines}\n"
            "# Spatial-level transforms lines not in file:\n"
            "{dual_lines}".format(
                outdated_docs_headers=", ".join(outdated_docs),
                py_file=os.path.basename(os.path.realpath(__file__)),
                filename=os.path.basename(filepath),
                image_only_lines="\n".join(image_only_lines_not_in_text),
                dual_lines="\n".join(dual_lines_not_in_text),
            )
        )
        raise ValueError(msg)

    if image_only_transforms_links not in text:
        msg = "Image only transforms links are outdated."
        raise ValueError(msg)
    if dual_transforms_table not in text:
        msg = "Dual transforms table are outdated."
        raise ValueError(msg)
    if mixing_transforms_table not in text:
        msg = "Mixing transforms table are outdated."
        raise ValueError(msg)



def main() -> None:
    args = parse_args()
    command = args.command
    if command not in {"make", "check"}:
        raise ValueError(f"You should provide a valid command: {{make|check}}. Got {command} instead.")

    image_only_transforms = get_image_only_transforms_info()
    dual_transforms = get_dual_transforms_info()
    mixing_transforms = get_mixing_transforms_info()

    image_only_transforms_links = make_transforms_targets_links(image_only_transforms)

    dual_transforms_table = make_transforms_targets_table(
        dual_transforms, header=["Transform"] + [target.value for target in Targets if target != Targets.GLOBAL_LABEL]
    )

    mixing_transforms_table = make_transforms_targets_table(
        mixing_transforms, header=["Transform"] + [target.value for target in Targets]
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
        print("===== COPY THIS TABLE TO README.MD BELOW ### Mixing transforms =====")
        print(mixing_transforms_table)
        print("===== END OF COPY =====")

    else:
        check_docs(args.filepath, image_only_transforms_links, dual_transforms_table, mixing_transforms_table)


if __name__ == "__main__":
    main()
