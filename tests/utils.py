import numpy as np


def convert_2d_to_3d(arrays, num_channels=3):
    # Converts a 2D numpy array with shape (H, W) into a 3D array with shape (H, W, num_channels)
    # by repeating the existing values along the new axis.
    arrays = tuple(np.repeat(array[:, :, np.newaxis], repeats=num_channels, axis=2) for array in arrays)
    if len(arrays) == 1:
        return arrays[0]
    return arrays


def convert_2d_to_target_format(arrays, target):
    if target == 'mask':
        return arrays[0] if len(arrays) == 1 else arrays
    elif target == 'image':
        return convert_2d_to_3d(arrays, num_channels=3)
    elif target == 'image_4_channels':
        return convert_2d_to_3d(arrays, num_channels=4)
    else:
        raise ValueError('Unknown target {}'.format(target))
