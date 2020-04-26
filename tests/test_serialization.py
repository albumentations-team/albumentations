import random
from unittest.mock import patch

import pytest
import numpy as np

import volumentations as V
from .utils import OpenMock

TEST_SEEDS = (0, 1, 42, 111, 9999)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [V.Scale3d, {"scale_limit": (0, 0, 0)}],
        [V.RotateAroundAxis3d, {"rotation_limit": 0}],
        [V.Move3d, {}],
        [V.Center3d, {}],
        [V.RandomDropout3d, {"dropout_ratio": 0}],
    ],
)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_serialization(
    augmentation_cls, params, p, seed, points, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = V.to_dict(aug)
    deserialized_aug = V.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(points=points)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(points=points)
    assert np.array_equal(aug_data["points"], deserialized_aug_data["points"])


AUGMENTATION_CLS_PARAMS = (
    [
        [V.Scale3d, {"scale_limit": (0.1, 0.1, 0.1)}],
        [V.RotateAroundAxis3d, {"axis": (0, 0, 1)}],
        [V.RotateAroundAxis3d, {"axis": (0, 1, 0), "rotation_limit": np.pi / 3}],
        [V.RotateAroundAxis3d, {"axis": (1, 0, 0), "rotation_limit": np.pi / 3}],
        [V.Move3d, {}],
        [V.Center3d, {}],
        [V.RandomDropout3d, {"dropout_ratio": 0}],
    ],
)


@pytest.mark.parametrize(["augmentation_cls", "params"], *AUGMENTATION_CLS_PARAMS)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
def test_augmentations_serialization_with_custom_parameters(
    augmentation_cls, params, p, seed, points, always_apply
):
    aug = augmentation_cls(p=p, always_apply=always_apply, **params)
    serialized_aug = V.to_dict(aug)
    deserialized_aug = V.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(points=points)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(points=points)
    assert np.array_equal(aug_data["points"], deserialized_aug_data["points"])


@pytest.mark.parametrize(["augmentation_cls", "params"], *AUGMENTATION_CLS_PARAMS)
@pytest.mark.parametrize("p", [0.5, 1])
@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("always_apply", (False, True))
@pytest.mark.parametrize("data_format", ("yaml",))
def test_augmentations_serialization_to_file_with_custom_parameters(
    augmentation_cls, params, p, seed, points, always_apply, data_format
):
    with patch("builtins.open", OpenMock()):
        aug = augmentation_cls(p=p, always_apply=always_apply, **params)
        filepath = "serialized.{}".format(data_format)
        V.save(aug, filepath, data_format=data_format)
        deserialized_aug = V.load(filepath, data_format=data_format)
        set_seed(seed)
        aug_data = aug(points=points)
        set_seed(seed)
        deserialized_aug_data = deserialized_aug(points=points)
        assert np.array_equal(aug_data["points"], deserialized_aug_data["points"])


@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_transform_pipeline_serialization(seed, points):
    aug = V.Compose(
        [
            V.OneOrOther(
                V.Compose(
                    [
                        V.RandomMove3d(),
                        V.OneOf(
                            [
                                V.Scale3d(scale_limit=(0.1, 0.1, 0)),
                                V.Scale3d(scale_limit=(0, 0, 0.05)),
                            ],
                        ),
                    ]
                ),
                V.Compose(
                    [
                        V.RotateAroundAxis3d(),
                        V.OneOf(
                            [
                                V.Scale3d(scale_limit=(1.0, 0.1, 0)),
                                V.Scale3d(scale_limit=(0, 1.0, 0.1)),
                            ],
                        ),
                    ]
                ),
            ),
            V.Flip3d(p=1),
        ]
    )
    serialized_aug = V.to_dict(aug)
    deserialized_aug = V.from_dict(serialized_aug)
    set_seed(seed)
    aug_data = aug(points=points)
    set_seed(seed)
    deserialized_aug_data = deserialized_aug(points=points)
    assert np.array_equal(aug_data["points"], deserialized_aug_data["points"])
