import cv2
import numpy as np

from albumentations.augmentations.utils import clipped


@clipped
def mix_arrays(array1: np.ndarray, array2: np.ndarray, mix_coef: float) -> np.ndarray:
    return cv2.addWeighted(array1, mix_coef, array2.reshape(array1.shape), 1 - mix_coef, 0)
