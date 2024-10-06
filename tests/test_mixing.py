from typing import Any, Dict, Tuple
import numpy as np
import pytest
import math

import albumentations as A
from tests.conftest import IMAGES, UINT8_IMAGES
from tests.utils import set_seed
from .test_functional_mixing import find_mix_coef
import random
from deepdiff import DeepDiff


def image_generator():
    yield {"image": np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)}

def complex_image_generator():
    height = 100
    width = 100
    yield {"image": (height, width)}

def complex_read_fn_image(x):
    return {"image": np.random.randint(0, 256, (x["image"][0], x["image"][1], 3), dtype=np.uint8)}


# Mock random.randint to produce consistent results
@pytest.fixture(autouse=True)
def mock_random(monkeypatch):
    def mock_randint(start, end):
        return start  # always return the start value for consistency in tests
    monkeypatch.setattr(random, "randint", mock_randint)


@pytest.mark.parametrize(
    "metadata, img_shape, expected_output",
    [
        (
            # Image + bbox without label + mask + mask_id + label_id + no offset
            {"image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
             "bbox": [0.3, 0.3, 0.5, 0.5],
             "mask": np.ones((20, 20), dtype=np.uint8) * 127,
             "mask_id": 1,
             "bbox_id": 99},
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((20, 20), dtype=np.uint8) * 127,
                "offset": (30, 30),
                "mask_id": 1,
                "bbox": [30, 30, 50, 50, 99],
            }
        ),
        # Image + bbox with label + mask_id + no mask
        (
            {"image": np.ones((20, 20, 3), dtype=np.uint8) * 255, "bbox": [0.3, 0.3, 0.5, 0.5, 99], "mask_id": 1},
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((20, 20), dtype=np.uint8),
                "offset": (30, 30),
                "mask_id": 1,
                "bbox": [30, 30, 50, 50, 99],
            }
        ),
        # Image + no bbox, no mask_id, no label_id, no_mask
        (
            {"image": np.ones((20, 20, 3), dtype=np.uint8) * 255},
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((20, 20, 3), dtype=np.uint8),
                "offset": (0, 0),
                "bbox": [0, 0, 20, 20],
            }
        ),
        # image + mask_id + label_id + no mask
        (
            {"image": np.ones((20, 20, 3), dtype=np.uint8) * 255, "mask_id": 1, "bbox_id": 99},
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((20, 20, 3), dtype=np.uint8),
                "offset": (0, 0),
                "mask_id": 1,
                "bbox": [0, 0, 20, 20, 99],
            }
        ),
        # Test case with triangular mask
        (
            {"image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
             "bbox": [0, 0, 0.2, 0.2],
             "mask": np.tri(20, 20, dtype=np.uint8) * 127,
             "mask_id": 2,
             "bbox_id": 100},
            (100, 100),
            {
                "overlay_image": np.ones((20, 20, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.tri(20, 20, dtype=np.uint8) * 127,
                "offset": (0, 0),
                "mask_id": 2,
                "bbox": [0, 0, 20, 20, 100],
            }
        ),
         # Test case with overlay_image having the same size as img_shape
        (
            {"image": np.ones((100, 100, 3), dtype=np.uint8) * 255,
             "bbox": [0, 0, 1, 1],
             "mask": np.ones((100, 100), dtype=np.uint8) * 127,
             "mask_id": 3,
             "bbox_id": 101},
            (100, 100),
            {
                "overlay_image": np.ones((100, 100, 3), dtype=np.uint8) * 255,
                "overlay_mask": np.ones((100, 100), dtype=np.uint8) * 127,
                "offset": (0, 0),
                "mask_id": 3,
                "bbox": [0, 0, 100, 100, 101],
            }
        ),
    ]
)
def test_preprocess_metadata(metadata: Dict[str, Any], img_shape: Tuple[int, int], expected_output: Dict[str, Any]):
    result = A.OverlayElements.preprocess_metadata(metadata, img_shape)

    assert DeepDiff(result, expected_output, ignore_type_in_groups=[(tuple, list)]) == {}


@pytest.mark.parametrize(
    "metadata, expected_output",
    [
        (
            {
                "image": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "bbox": [0.1, 0.2, 0.2, 0.3]
            },
            {
                "expected_overlay": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "expected_bbox": [10, 20, 20, 30]
            }
        ),
        (
            {
                "image": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "bbox": [0.3, 0.4, 0.4, 0.5],
                "label_id": 99
            },
            {
                "expected_overlay": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "expected_bbox": [30, 40, 40, 50, 99]
            }
        ),
        (
            {
                "image": np.ones((10, 10, 3), dtype=np.uint8) * 255
            },
            {
                "expected_overlay": np.ones((10, 10, 3), dtype=np.uint8) * 255,
                "expected_bbox": [0, 0, 10, 10]
            }
        ),
    ]
)
def test_end_to_end(metadata, expected_output):
    transform = A.Compose([A.OverlayElements(p=1)])

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    transformed = transform(image=img, overlay_metadata=metadata)

    expected_img = np.zeros((100, 100, 3), dtype=np.uint8)
    x_min, y_min, x_max, y_max = expected_output["expected_bbox"][:4]

    expected_img[y_min:y_max, x_min:x_max] = expected_output["expected_overlay"]

    np.testing.assert_array_equal(transformed["image"], expected_img)
