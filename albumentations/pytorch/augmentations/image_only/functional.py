import torch
import kornia as K


MAX_VALUES_BY_DTYPE = {torch.uint8: 255, torch.float32: 1.0}


def from_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(dtype)
            )
    return (img * max_value).type(dtype)


def to_float(img, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the max_value argument".format(img.dtype)
            )
    return img.type(torch.float32) / max_value


def clip(img, dtype, maxval):
    return torch.clamp(img, 0, maxval).type(dtype)


def cutout(img, holes, fill_value=0):
    # Make a copy of the input image since we don't want to modify it directly
    img = img.clone()
    for x1, y1, x2, y2 in holes:
        img[:, y1:y2, x1:x2] = fill_value
    return img


def add_snow(img, snow_point, brightness_coeff):
    """Bleaches out pixels, imitation snow.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        img (torch.Tensor): Image.
        snow_point: Number of show points.
        brightness_coeff: Brightness coefficient.

    Returns:
        numpy.ndarray: Image.

    """
    input_dtype = img.dtype
    needs_float = False

    snow_point *= 127.5  # = 255 / 2
    snow_point += 85  # = 255 / 3

    if input_dtype == torch.float32:
        img = from_float(img, dtype=torch.uint8)
        needs_float = True
    elif input_dtype not in (torch.uint8, torch.float32):
        raise ValueError("Unexpected dtype {} for RandomSnow augmentation".format(input_dtype))

    image_HLS = K.rgb_to_hls(img)
    image_HLS = image_HLS.type(torch.float32)

    image_HLS[1][image_HLS[1] < snow_point] *= brightness_coeff

    image_HLS[1] = clip(image_HLS[1], torch.uint8, 255)

    image_HLS = image_HLS.type(torch.uint8)

    image_RGB = K.hls_to_rgb(image_HLS)

    if needs_float:
        image_RGB = to_float(image_RGB, max_value=255)

    return image_RGB
