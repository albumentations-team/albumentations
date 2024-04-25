from warnings import warn

import numpy as np

from albumentations.augmentations.functional import add_weighted


def mix_arrays(array1: np.ndarray, array2: np.ndarray, mix_coef: float) -> np.ndarray:
    warn(
        "The function 'mix_arrays' is deprecated."
        "Please use 'add_weighted(array1, mix_coef, array_2, 1 - mix_coef)' instead",
        DeprecationWarning,
        stacklevel=2,
    )

    return add_weighted(array1, mix_coef, array2, 1 - mix_coef)
