import pytest
import numpy as np
from skimage.measure import label as ski_label
from albumentations.augmentations.dropout.functional import label as cv_label
from scipy import stats

from tests.utils import set_seed

@pytest.mark.parametrize("shape, dtype, connectivity", [
    ((8, 8), np.uint8, 1),
    ((10, 10), np.uint8, 2),
    ((12, 12), np.int32, 1),
    ((12, 12), np.int32, 2),
    ((14, 14), np.uint8, 1),
    ((35, 35), np.uint8, 2),
])
def test_label_function(shape, dtype, connectivity):
    set_seed(42)
    # Generate a random binary mask
    mask = np.random.randint(0, 2, shape).astype(dtype)

    # Compare results with scikit-image
    ski_result = ski_label(mask, connectivity=connectivity)
    cv_result = cv_label(mask, connectivity=connectivity)

    np.testing.assert_array_equal(cv_result, ski_result), "Label results do not match"

@pytest.mark.parametrize("shape, dtype, connectivity", [
    ((10, 10), np.uint8, 1),
    ((20, 20), np.int32, 2),
    ((30, 30), np.uint8, 1),
])
def test_label_function_return_num(shape, dtype, connectivity):
    mask = np.random.randint(0, 2, shape).astype(dtype)

    ski_result, ski_num = ski_label(mask, connectivity=connectivity, return_num=True)
    cv_result, cv_num = cv_label(mask, connectivity=connectivity, return_num=True)

    np.testing.assert_array_equal(cv_result, ski_result), "Label results do not match"
    assert ski_num == cv_num, "Number of labels do not match"

@pytest.mark.parametrize("shape, num_objects", [
    ((10, 10), 3),
    ((20, 20), 5),
    ((30, 30), 10),
])
def test_label_function_with_multiple_objects(shape, num_objects):
    set_seed(43)
    mask = np.zeros(shape, dtype=np.uint8)
    for i in range(1, num_objects + 1):
        x, y = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
        mask[x:x+3, y:y+3] = i

    ski_result, ski_num = ski_label(mask, return_num=True)
    cv_result, cv_num = cv_label(mask, return_num=True)

    # Check for one-to-one mapping
    combined = np.stack((ski_result, cv_result))
    unique_combinations = np.unique(combined.reshape(2, -1).T, axis=0)

    assert len(unique_combinations) == len(np.unique(ski_result)) == len(np.unique(cv_result)), \
        "Labels are not equal up to enumeration"

    assert ski_num == cv_num, "Number of labels do not match"
    assert cv_num == num_objects, f"Expected {num_objects} labels, got {cv_num}"

def test_label_function_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)

    ski_result, ski_num = ski_label(mask, return_num=True)
    cv_result, cv_num = cv_label(mask, return_num=True)

    np.testing.assert_array_equal(cv_result, ski_result), "Label results do not match for empty mask"
    assert ski_num == cv_num == 0, "Number of labels should be 0 for empty mask"

def test_label_function_full_mask():
    mask = np.ones((10, 10), dtype=np.uint8)

    ski_result, ski_num = ski_label(mask, return_num=True)
    cv_result, cv_num = cv_label(mask, return_num=True)

    np.testing.assert_array_equal(cv_result, ski_result), "Label results do not match for full mask"
    assert ski_num == cv_num, "Number of labels should be 2 for full mask (background + one object)"
