import numpy as np
import pytest

from volumentations import (
    Scale3d,
    RotateAroundAxis3d,
    Move3d,
    Center3d,
    RandomDropout3d,
    Flip3d,
)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [Scale3d, {"scale_limit": (0, 0, 0)}],
        [RotateAroundAxis3d, {"rotation_limit": 0}],
        [Move3d, {}],
        [RandomDropout3d, {"dropout_ratio": 0.0}],
    ],
)
def test_augmentations_wont_change_input(
    augmentation_cls, params, points, features, labels, normals, bboxes, cameras
):
    points_copy = points.copy()
    features_copy = features.copy()
    labels_copy = labels.copy()
    normals_copy = normals.copy()
    bboxes_copy = bboxes.copy()
    cameras_copy = cameras.copy()
    aug = augmentation_cls(p=1, **params)
    data = aug(
        points=points,
        features=features,
        labels=labels,
        normals=normals,
        bboxes=bboxes,
        cameras=cameras,
    )
    aug = augmentation_cls(p=1, **params)
    assert np.array_equal(data["points"], points_copy)
    assert np.array_equal(data["features"], features_copy)
    assert np.array_equal(data["labels"], labels_copy)
    assert np.array_equal(data["normals"], normals_copy)
    assert np.array_equal(data["bboxes"], bboxes_copy)
    assert np.array_equal(data["cameras"], cameras_copy)
