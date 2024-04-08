from warnings import warn

import cv2
import numpy as np

from albumentations.augmentations.utils import clipped, preserve_shape


@clipped
@preserve_shape
def mix_arrays(array1: np.ndarray, array2: np.ndarray, mix_coef: float) -> np.ndarray:
    warn(
        "The function 'mix_arrays' is deprecated."
        "Please use 'add_weighted(array1, mix_coef, array_2, 1 - mix_coef)' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    array2 = array2.reshape(array1.shape).astype(array1.dtype)
    return cv2.addWeighted(array1, mix_coef, array2, 1 - mix_coef, 0)
