from albumentations import HorizontalFlip, OneOf, IAAPiecewiseAffine, IAAAffine, OpticalDistortion, GridDistortion
from albumentations.core.composition import find_dual_start_end
import pytest


def empty_aug1():
    return [
        HorizontalFlip(p=0.001),
        # IAAPiecewiseAffine(p=1.0),
        OneOf([
            # OpticalDistortion(p=0.1),
            # GridDistortion(p=0.1),
            # IAAPerspective(p=1.0),
            # IAAAffine(p=1.0),
            IAAPiecewiseAffine(p=1.0),
        ], p=0.0)
    ]


def empty_aug2():
    return [
        HorizontalFlip(p=0.001),
        IAAPiecewiseAffine(p=1.0),
        OneOf([
            # OpticalDistortion(p=0.1),
            # GridDistortion(p=0.1),
            # IAAPerspective(p=1.0),
            IAAAffine(p=1.0),
            # IAAPiecewiseAffine(p=1.0),
        ], p=0.0)
    ]


def empty_aug3():
    return [
        # HorizontalFlip(p=0.001),
        # IAAPiecewiseAffine(p=1.0),
        OneOf([
            OpticalDistortion(p=0.1),
            GridDistortion(p=0.1),
            # IAAPerspective(p=1.0),
            # IAAAffine(p=1.0),
            # IAAPiecewiseAffine(p=1.0),
        ], p=0.0)
    ]


@pytest.mark.parametrize(['aug', 'start_end'], [
    [empty_aug1, [0, 1]],
    [empty_aug2, [0, 2]],
    [empty_aug3, [0, 0]],
])
def test_strong_aug(aug, start_end):
    assert find_dual_start_end(aug()) == start_end
