import numpy as np

from albumentations.augmentations.utils import clipped


@clipped
def mix_arrays(array1: np.ndarray, array2: np.ndarray, lam: float) -> np.ndarray:
    return lam * array1 + (1 - lam) * array2
