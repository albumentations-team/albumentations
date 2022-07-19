import random
import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np

from albumentations.core.transforms_interface import ImageOnlyTransform

from .functional import cutout

__all__ = ["Cutout"]


class Cutout(ImageOnlyTransform):
    """CoarseDropout of the square regions in the image.

    Args:
        num_holes (int): number of regions to zero out
        max_h_size (int): maximum height of the hole
        max_w_size (int): maximum width of the hole
        fill_value (int, float, list of int, list of float): value for dropped pixels.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """

    def __init__(
        self,
        num_holes: int = 8,
        max_h_size: int = 8,
        max_w_size: int = 8,
        fill_value: Union[int, float] = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(Cutout, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        warnings.warn(
            f"{self.__class__.__name__} has been deprecated. Please use CoarseDropout",
            FutureWarning,
        )

    def apply(self, img: np.ndarray, fill_value: Union[int, float] = 0, holes=(), **params):
        return cutout(img, holes, fill_value)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("num_holes", "max_h_size", "max_w_size")
