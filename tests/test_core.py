from unittest import mock
from unittest.mock import Mock, MagicMock, call

import numpy as np
import pytest

from volumentations.core.transforms_interface import to_tuple, PointCloudsTransform
from volumentations.core.composition import (
    OneOrOther,
    Compose,
    OneOf,
    ReplayCompose,
)
from volumentations.augmentations.transforms import (
    Scale3d,
    RotateAroundAxis3d,
    Move3d,
    Center3d,
    RandomDropout3d,
    Flip3d,
)


def test_one_or_other():
    first = MagicMock()
    second = MagicMock()
    augmentation = OneOrOther(first, second, p=1)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert first.called != second.called


def test_compose():
    first = MagicMock()
    second = MagicMock()
    augmentation = Compose([first, second], p=1)
    image = np.ones((8, 8))
    augmentation(image=image)
    assert first.called
    assert second.called


def oneof_always_apply_crash():
    aug = Compose(
        [Scale3d(), RotateAroundAxis3d(), OneOf([Move3d(), Flip3d()], p=1)], p=1
    )
    points = np.ones((10, 3))
    data = aug(points=points)
    assert data


def test_always_apply():
    first = MagicMock(always_apply=True)
    second = MagicMock(always_apply=False)
    augmentation = Compose([first, second], p=0)
    points = np.ones((10, 3))
    augmentation(points=points)
    assert first.called
    assert not second.called


def test_one_of():
    transforms = [Mock(p=1) for _ in range(10)]
    augmentation = OneOf(transforms, p=1)
    points = np.ones((10, 3))
    augmentation(points=points)
    assert len([transform for transform in transforms if transform.called]) == 1


def test_to_tuple():
    assert to_tuple(10) == (-10, 10)
    assert to_tuple(0.5) == (-0.5, 0.5)
    assert to_tuple((-20, 20)) == (-20, 20)
    assert to_tuple([-20, 20]) == (-20, 20)
    assert to_tuple(100, low=30) == (30, 100)
    assert to_tuple(10, bias=1) == (-9, 11)
    assert to_tuple(100, bias=2) == (-98, 102)
