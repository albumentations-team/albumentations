import numpy as np


def convert_2d_to_3d(array):
    # Converts a 2D numpy array with shape (H, W) into a 3D array with shape (H, W, 3)
    # by repeating the existing values along the new axis.
    return np.repeat(array[:, :, np.newaxis], repeats=3, axis=2)
