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
    IMAGE = 'Image'
    MASKS = 'Masks'
    BBOXES = 'BBoxes'
    KEYPOINTS = 'Keypoints'


READTHEDOCS_TEMPLATE = '[{name}](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations'
TRANSFORM_NAME_WITH_LINK_TEMPLATE = READTHEDOCS_TEMPLATE + '.augmentations.transforms.{name})'
IMGAUG_TRANSFORM_NAME_WITH_LINK_TEMPLATE = READTHEDOCS_TEMPLATE + '.imgaug.transforms.{name})'


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


if __name__ == '__main__':
    transforms_info = get_transforms_info()
    table = make_transforms_targets_table(
        transforms_info,
        header=['Transform'] + [target.value for target in Targets],

    )
    print(table)
