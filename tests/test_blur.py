from typing import Dict, List, Tuple

import numpy as np
import pytest

import albumentations as A
from albumentations.augmentations.blur.functional import gaussian_blur
from albumentations.core.transforms_interface import BasicTransform
from tests.conftest import UINT8_IMAGES
from tests.utils import set_seed


@pytest.mark.parametrize("aug", [A.Blur, A.MedianBlur, A.MotionBlur])
@pytest.mark.parametrize(
    "blur_limit_input, blur_limit_used", [[(3, 3), (3, 3)], [(13, 13), (13, 13)]]
)
@pytest.mark.parametrize("image", UINT8_IMAGES)
def test_blur_kernel_generation(
    image: np.ndarray,
    aug: BasicTransform,
    blur_limit_input: Tuple[int, int],
    blur_limit_used: Tuple[int, int],
) -> None:
    aug = aug(blur_limit=blur_limit_input, p=1)

    assert aug.blur_limit == blur_limit_used
    aug(image=image)["image"]


@pytest.mark.parametrize(["val_uint8"], [[0], [1], [128], [255]])
def test_glass_blur_float_uint8_diff_less_than_two(val_uint8: List[int]) -> None:
    x_uint8 = np.zeros((5, 5)).astype(np.uint8)
    x_uint8[2, 2] = val_uint8

    x_float32 = np.zeros((5, 5)).astype(np.float32)
    x_float32[2, 2] = val_uint8 / 255.0

    glassblur = A.GlassBlur(p=1, max_delta=1)

    set_seed(0)
    blur_uint8 = glassblur(image=x_uint8)["image"]

    set_seed(0)
    blur_float32 = glassblur(image=x_float32)["image"]

    # Before comparison, rescale the blur_float32 to [0, 255]
    diff = np.abs(blur_uint8 - blur_float32 * 255)

    # The difference between the results of float32 and uint8 will be at most 2.
    assert np.all(diff <= 2.0)


@pytest.mark.parametrize(["val_uint8"], [[0], [1], [128], [255]])
def test_advanced_blur_float_uint8_diff_less_than_two(val_uint8: List[int]) -> None:
    x_uint8 = np.zeros((5, 5)).astype(np.uint8)
    x_uint8[2, 2] = val_uint8

    x_float32 = np.zeros((5, 5)).astype(np.float32)
    x_float32[2, 2] = val_uint8 / 255.0

    adv_blur = A.AdvancedBlur(blur_limit=(3, 5), p=1)

    set_seed(0)
    adv_blur_uint8 = adv_blur(image=x_uint8)["image"]

    set_seed(0)
    adv_blur_float32 = adv_blur(image=x_float32)["image"]

    # Before comparison, rescale the adv_blur_float32 to [0, 255]
    diff = np.abs(adv_blur_uint8 - adv_blur_float32 * 255)

    # The difference between the results of float32 and uint8 will be at most 2.
    assert np.all(diff <= 2.0)


@pytest.mark.parametrize(
    ["params"],
    [
        [{"blur_limit": (2, 5)}],
        [{"blur_limit": (3, 6)}],
        [{"sigma_x_limit": (0.0, 1.0), "sigma_y_limit": (0.0, 1.0)}],
        [{"beta_limit": (0.1, 0.9)}],
        [{"beta_limit": (1.1, 8.0)}],
    ],
)
def test_advanced_blur_raises_on_incorrect_params(
    params: Dict[str, List[float]]
) -> None:
    with pytest.raises(ValueError):
        A.AdvancedBlur(**params)


@pytest.mark.parametrize(
    ["blur_limit", "sigma", "result_blur", "result_sigma"],
    [
        [[0, 0], [1, 1], 0, 1],
        [[1, 1], [0, 0], 1, 0],
        [[1, 1], [1, 1], 1, 1],
        [[0, 0], [0, 0], 3, 0],
        [[0, 3], [0, 0], 3, 0],
        [[0, 3], [0.1, 0.1], 3, 0.1],
    ],
)
def test_gaus_blur_limits(
    blur_limit: List[float], sigma: List[float], result_blur: int, result_sigma: float
) -> None:
    img = np.zeros([100, 100, 3], dtype=np.uint8)

    aug = A.Compose([A.GaussianBlur(blur_limit=blur_limit, sigma_limit=sigma, p=1)])

    res = aug(image=img)["image"]
    assert np.allclose(res, gaussian_blur(img, result_blur, result_sigma))
