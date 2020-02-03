import torch
import random

import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(image, device="cpu"):
    if len(image.shape) == 2:
        image = np.expand_dims(image, -1)

    image = image.transpose([2, 0, 1])
    image = torch.from_numpy(image)

    return image.to(device)


def from_tensor(image):
    image = image.detach().cpu().numpy()
    image = image.transpose([1, 2, 0])
    return np.squeeze(image)


def convert_2d_to_3d(arrays, num_channels=3):
    # Converts a 2D numpy array with shape (H, W) into a 3D array with shape (H, W, num_channels)
    # by repeating the existing values along the new axis.
    arrays = tuple(np.repeat(array[:, :, np.newaxis], repeats=num_channels, axis=2) for array in arrays)
    if len(arrays) == 1:
        return arrays[0]
    return arrays


def convert_2d_to_target_format(arrays, target):
    if target == "mask":
        return arrays[0] if len(arrays) == 1 else arrays
    elif target == "image":
        return convert_2d_to_3d(arrays, num_channels=3)
    elif target == "image_4_channels":
        return convert_2d_to_3d(arrays, num_channels=4)
    else:
        raise ValueError("Unknown target {}".format(target))
