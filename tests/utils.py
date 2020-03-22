import numpy as np
from hypothesis.extra.numpy import arrays as h_array
from hypothesis.strategies import composite
from hypothesis.strategies import floats as h_float
from hypothesis.strategies import integers as h_int


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


@composite
def h_image(draw, width=100, height=100, num_channels=3, dtype=np.uint8):
    return draw(
        h_array(
            dtype=dtype,
            shape=(height, width, num_channels),
            elements=h_int(min_value=0, max_value=np.iinfo(dtype).max - 1),
        )
    )


@composite
def h_mask(draw, width=100, height=100, dtype=np.uint8):
    return draw(
        h_array(dtype=dtype, shape=(height, width), elements=h_int(min_value=0, max_value=np.iinfo(dtype).max - 1))
    )


@composite
def h_binary_mask(draw, width=100, height=100, dtype=np.uint8):
    return draw(h_array(dtype=dtype, shape=(height, width), elements=h_int(min_value=0, max_value=1)))


@composite
def h_float_image(draw, width=100, height=100, num_channels=3, dtype=np.float32):
    return draw(
        h_array(
            dtype=dtype,
            shape=(height, width, num_channels),
            elements=h_float(min_value=0, allow_nan=False, max_value=1, width=32),
        )
    )
