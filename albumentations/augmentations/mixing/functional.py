import cv2
import numpy as np

from albumentations.augmentations.utils import clipped, preserve_shape


@clipped
@preserve_shape
def mix_arrays(array1: np.ndarray, array2: np.ndarray, mix_coef: float) -> np.ndarray:
    array2 = array2.reshape(array1.shape).astype(array1.dtype)
    return cv2.addWeighted(array1, mix_coef, array2, 1 - mix_coef, 0)
