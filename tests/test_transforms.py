import numpy as np
import pytest

import volumentations as V
import volumentations.augmentations.functional as F


def __test_multiprocessing_support_proc(args):
    x, transform = args
    return transform(points=x)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
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
def test_multiprocessing_support(augmentation_cls, params, multiprocessing_context):
    """Checks whether we can use augmentations in multiprocessing environments"""
    aug = augmentation_cls(p=1, **params)
    points = np.empty((100, 3))

    pool = multiprocessing_context.Pool(8)
    pool.map(
        __test_multiprocessing_support_proc, map(lambda x: (x, aug), [points] * 100)
    )
    pool.close()
    pool.join()


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
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
def test_additional_targets_only(augmentation_cls, params):
    aug = V.Compose(
        [augmentation_cls(always_apply=True, **params)],
        additional_targets={"points2": "points"},
    )
    for _ in range(10):
        points1 = np.random.random((100, 3))
        points2 = points1.copy()
        res = aug(points=points1, points2=points2)
        aug1 = res["points"]
        aug2 = res["points2"]
        assert np.array_equal(aug1, aug2)
