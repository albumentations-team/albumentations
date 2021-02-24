import inspect
import os
import sys
from enum import Enum
import argparse

sys.path.append("..")
import albumentations  # noqa: E402


IGNORED_CLASSES = {
    "BasicTransform",
    "BasicIAATransform",
    "DualIAATransform",
    "DualTransform",
    "ImageOnlyIAATransform",
    "ImageOnlyTransform",
}


def make_augmentation_docs_link(cls):
    module_parts = cls.__module__.split(".")
    module_page = "/".join(module_parts[1:])
    return (
        "[{cls.__name__}](https://albumentations.ai/docs/api_reference/{module_page}/#{cls.__module__}.{cls.__name__})"
    ).format(module_page=module_page, cls=cls)


class Targets(Enum):
    IMAGE = "Image"
    MASKS = "Masks"
    BBOXES = "BBoxes"
    KEYPOINTS = "Keypoints"


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Commands", dest="command")
    subparsers.add_parser("make")
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("filepath", type=str, help="Path to a file that should be checked")
    return parser.parse_args()


def make_separator(width, align_center):
    if align_center:
        return ":" + "-" * (width - 2) + ":"
    return "-" * width


def get_transforms_info():
    transforms_info = {}
    for name, cls in inspect.getmembers(albumentations):
        if inspect.isclass(cls) and issubclass(cls, albumentations.BasicTransform) and name not in IGNORED_CLASSES:
            if "DeprecationWarning" in inspect.getsource(cls) or "FutureWarning" in inspect.getsource(cls):
                continue

            targets = {Targets.IMAGE}
            if issubclass(cls, albumentations.DualTransform):
                targets.add(Targets.MASKS)

            if hasattr(cls, "apply_to_bbox") and cls.apply_to_bbox is not albumentations.DualTransform.apply_to_bbox:
                targets.add(Targets.BBOXES)

            if (
                hasattr(cls, "apply_to_keypoint")
                and cls.apply_to_keypoint is not albumentations.DualTransform.apply_to_keypoint
            ):
                targets.add(Targets.KEYPOINTS)

            if issubclass(cls, albumentations.DualIAATransform):
                targets.update({Targets.BBOXES, Targets.KEYPOINTS})

            if issubclass(cls, albumentations.Lambda):
                targets.add(Targets.MASKS)
                targets.add(Targets.BBOXES)
                targets.add(Targets.KEYPOINTS)

            transforms_info[name] = {
                "targets": targets,
                "docs_link": make_augmentation_docs_link(cls),
                "image_only": issubclass(cls, albumentations.ImageOnlyTransform),
            }
    return transforms_info


def make_transforms_targets_table(transforms_info, header):
    rows = [header]
    for transform, info in sorted(transforms_info.items(), key=lambda kv: kv[0]):
        transform_targets = []
        for target in Targets:
            mark = "✓" if target in info["targets"] else ""
            transform_targets.append(mark)
        row = [info["docs_link"] or transform] + transform_targets
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
    return "\n".join("| {line} |".format(line=line) for line in lines)


def make_transforms_targets_links(transforms_info):
    return "\n".join(
        "- " + info["docs_link"] for transform, info in sorted(transforms_info.items(), key=lambda kv: kv[0])
    )


def check_docs(filepath, image_only_transforms_links, dual_transforms_table):
    with open(filepath, "r", encoding="utf8") as f:
        text = f.read()
    outdated_docs = set()
    image_only_lines_not_in_text = []
    dual_lines_not_in_text = []
    for line in image_only_transforms_links.split("\n"):
        if line not in text:
            outdated_docs.update(["Pixel-level"])
            image_only_lines_not_in_text.append(line)
    for line in dual_transforms_table.split("\n"):
        if line not in text:
            dual_lines_not_in_text.append(line)
            outdated_docs.update(["Spatial-level"])
    if outdated_docs:
        raise ValueError(
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

    if image_only_transforms_links not in text:
        raise ValueError("Image only transforms links are outdated.")
    if dual_transforms_table not in text:
        raise ValueError("Dual transforms table are outdated.")


def main():
    args = parse_args()
    command = args.command
    if command not in {"make", "check"}:
        raise ValueError(
            "You should provide a valid command: {{make|check}}. Got {command} instead.".format(command=command)
        )
    transforms_info = get_transforms_info()
    image_only_transforms = {transform: info for transform, info in transforms_info.items() if info["image_only"]}
    dual_transforms = {transform: info for transform, info in transforms_info.items() if not info["image_only"]}
    image_only_transforms_links = make_transforms_targets_links(image_only_transforms)
    dual_transforms_table = make_transforms_targets_table(
        dual_transforms, header=["Transform"] + [target.value for target in Targets]
    )
    if command == "make":
        print(image_only_transforms_links)
        print(dual_transforms_table)
    else:
        check_docs(args.filepath, image_only_transforms_links, dual_transforms_table)


if __name__ == "__main__":
    main()
