import inspect
import os
import sys
from enum import Enum
import argparse

sys.path.append('..')
import albumentations  # noqa: E402


IGNORED_CLASSES = {
    'BasicTransform',
    'BasicIAATransform',
    'DualIAATransform',
    'DualTransform',
    'ImageOnlyIAATransform',
    'ImageOnlyTransform',
}


READTHEDOCS_TEMPLATE = '[{name}](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations'
TRANSFORM_NAME_WITH_LINK_TEMPLATE = READTHEDOCS_TEMPLATE + '.augmentations.transforms.{name})'
IMGAUG_TRANSFORM_NAME_WITH_LINK_TEMPLATE = READTHEDOCS_TEMPLATE + '.imgaug.transforms.{name})'


class Targets(Enum):
    IMAGE = 'Image'
    MASKS = 'Masks'
    BBOXES = 'BBoxes'
    KEYPOINTS = 'Keypoints'


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Commands', dest='command')
    subparsers.add_parser('make')
    check_parser = subparsers.add_parser('check')
    check_parser.add_argument('filepath', type=str, help='Path to a file that should be checked')
    return parser.parse_args()


def make_separator(width, align_center):
    if align_center:
        return ':' + '-' * (width - 2) + ':'
    return '-' * width


def get_transforms_info():
    transforms_info = {}
    for name, cls in inspect.getmembers(albumentations):
        if (
            inspect.isclass(cls) and
            issubclass(cls, albumentations.BasicTransform) and
            name not in IGNORED_CLASSES
        ):

            targets = {Targets.IMAGE}
            if issubclass(cls, albumentations.DualTransform):
                targets.add(Targets.MASKS)

            if hasattr(cls, 'apply_to_bbox') and cls.apply_to_bbox is not albumentations.DualTransform.apply_to_bbox:
                targets.add(Targets.BBOXES)

            if (
                hasattr(cls, 'apply_to_keypoint') and
                cls.apply_to_keypoint is not albumentations.DualTransform.apply_to_keypoint
            ):
                targets.add(Targets.KEYPOINTS)

            if issubclass(cls, albumentations.DualIAATransform):
                targets.update({Targets.BBOXES, Targets.KEYPOINTS})

            docs_link = None
            if cls.__module__ == 'albumentations.augmentations.transforms':
                docs_link = TRANSFORM_NAME_WITH_LINK_TEMPLATE.format(name=name)
            elif cls.__module__ == 'albumentations.imgaug.transforms':
                docs_link = IMGAUG_TRANSFORM_NAME_WITH_LINK_TEMPLATE.format(name=name)

            transforms_info[name] = {
                'targets': targets,
                'docs_link': docs_link,
                'image_only': issubclass(cls, albumentations.ImageOnlyTransform)
            }
    return transforms_info


def make_transforms_targets_table(transforms_info, header):
    rows = [header]
    for transform, info in sorted(transforms_info.items(), key=lambda kv: kv[0]):
        transform_targets = []
        for target in Targets:
            mark = '✓' if target in info['targets'] else ''
            transform_targets.append(mark)
        row = [info['docs_link'] or transform] + transform_targets
        rows.append(row)

    column_widths = [max([len(r) for r in column]) for column in zip(*rows)]
    lines = [
        ' | '.join(
            '{title: <{width}}'.format(width=width, title=title) for width, title in zip(column_widths, rows[0])
        ),
        ' | '.join(
            make_separator(width, align_center=column_index > 0) for column_index, width in enumerate(column_widths)
        )
    ]
    for row in rows[1:]:
        lines.append(' | '.join(
            '{column: <{width}}'.format(width=width, column=column)
            for width, column in zip(column_widths, row)
        ))
    return '\n'.join('| {line} |'.format(line=line) for line in lines)


def make_transforms_targets_links(transforms_info):
    return '\n'.join(
        '- ' + info['docs_link'] for transform, info in sorted(transforms_info.items(), key=lambda kv: kv[0])
    )


def check_docs(filepath, image_only_transforms_links, dual_transforms_table):
    with open(args.filepath) as f:
        text = f.read()
    outdated_docs = []
    if image_only_transforms_links not in text:
        outdated_docs.append('Pixel-level')
    if dual_transforms_table not in text:
        outdated_docs.append('Spatial-level')
    if not outdated_docs:
        return

    raise ValueError(
        'Docs for the following transform types are outdated: {outdated_docs_headers}. '
        'Generate new docs by executing the `python tools/{py_file} make` command '
        'and paste them to {filename}.'.format(
            outdated_docs_headers=', '.join(outdated_docs),
            py_file=os.path.basename(os.path.realpath(__file__)),
            filename=os.path.basename(filepath),
        )
    )


if __name__ == '__main__':
    args = parse_args()
    command = args.command
    if command not in {'make', 'check'}:
        raise ValueError(
            'You should provide a valid command: {{make|check}}. Got {command} instead.'.format(command=command)
        )
    transforms_info = get_transforms_info()
    image_only_transforms = {transform: info for transform, info in transforms_info.items() if info['image_only']}
    dual_transforms = {transform: info for transform, info in transforms_info.items() if not info['image_only']}
    image_only_transforms_links = make_transforms_targets_links(image_only_transforms)
    dual_transforms_table = make_transforms_targets_table(
        dual_transforms,
        header=['Transform'] + [target.value for target in Targets],
    )
    if command == 'make':
        print(image_only_transforms_links)
        print(dual_transforms_table)
    else:
        check_docs(args.filepath, image_only_transforms_links, dual_transforms_table)
