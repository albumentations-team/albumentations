import inspect
import sys
from enum import Enum

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


class Targets(Enum):
    IMAGE = 'image'
    MASKS = 'masks'
    BBOXES = 'bboxes'
    KEYPOINTS = 'keypoints'


READTHEDOCS_TEMPLATE = '[{name}](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations'
TRANSFORM_NAME_WITH_LINK_TEMPLATE = READTHEDOCS_TEMPLATE + '.augmentations.transforms.{name})'
IMGAUG_TRANSFORM_NAME_WITH_LINK_TEMPLATE = READTHEDOCS_TEMPLATE + '.imgaug.transforms.{name})'


def make_separator(width, align_center):
    if align_center:
        return ':' + '-' * (width - 2) + ':'
    return '-' * width


def make_transforms_targets_table():
    transforms_targets = {}
    transforms_docs_links = {}
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

            if cls.__module__ == 'albumentations.augmentations.transforms':
                transforms_docs_links[name] = TRANSFORM_NAME_WITH_LINK_TEMPLATE.format(name=name)
            elif cls.__module__ == 'albumentations.imgaug.transforms':
                transforms_docs_links[name] = IMGAUG_TRANSFORM_NAME_WITH_LINK_TEMPLATE.format(name=name)

            transforms_targets[name] = targets

    header = ['Transform'] + [target.name.capitalize() for target in Targets]
    rows = []
    for transform, targets in sorted(transforms_targets.items(), key=lambda kv: kv[0]):
        transform_targets = []
        for target in Targets:
            mark = '✓' if target in targets else ''
            transform_targets.append(mark)
        row = [transforms_docs_links.get(transform) or transform] + transform_targets
        rows.append(row)

    column_widths = [max([len(r) for r in column]) for column in zip(*rows)]
    lines = [
        ' | '.join(
            '{title: <{width}}'.format(width=width, title=title) for width, title in zip(column_widths, header)
        ),
        ' | '.join(
            make_separator(width, align_center=column_index > 0) for column_index, width in enumerate(column_widths)
        )
    ]
    for row in rows:
        lines.append(' | '.join(
            '{column: <{width}}'.format(width=width, column=column)
            for width, column in zip(column_widths, row)
        ))
    return '\n'.join('| {line} |'.format(line=line) for line in lines)


if __name__ == '__main__':
    table = make_transforms_targets_table()
    print(table)
