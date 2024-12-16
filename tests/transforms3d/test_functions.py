import numpy as np
from albumentations.augmentations.transforms3d import functional as f3d

def test_uniqueness():
    # Test for 3D input (D,H,W)
    test_cube_3d = np.arange(27).reshape(3,3,3)
    transformations_3d = [f3d.transform_cube(test_cube_3d, i) for i in range(48)]
    unique_3d = set(str(t) for t in transformations_3d)
    print(f"Number of unique 3D transformations: {len(unique_3d)}")
    assert len(unique_3d) == 48, "Not all 3D transformations are unique!"

    # Test for 4D input (D,H,W,C)
    test_cube_4d = np.arange(54).reshape(3,3,3,2)
    transformations_4d = [f3d.transform_cube(test_cube_4d, i) for i in range(48)]
    unique_4d = set(str(t) for t in transformations_4d)
    print(f"Number of unique 4D transformations: {len(unique_4d)}")
    assert len(unique_4d) == 48, "Not all 4D transformations are unique!"

    # Check shapes are preserved
    for t in transformations_3d:
        assert t.shape == (3,3,3), f"Wrong shape for 3D output: {t.shape}"
    for t in transformations_4d:
        assert t.shape == (3,3,3,2), f"Wrong shape for 4D output: {t.shape}"
