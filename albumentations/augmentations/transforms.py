from __future__ import absolute_import, division

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np

from ..core.transforms_interface import to_tuple, DualTransform, ImageOnlyTransform
from . import functional as F

__all__ = ['Blur', 'VerticalFlip', 'HorizontalFlip', 'Flip', 'Normalize', 'Transpose', 'RandomCrop', 'RandomGamma',
           'RandomRotate90', 'Rotate', 'ShiftScaleRotate', 'CenterCrop', 'OpticalDistortion', 'GridDistortion',
           'ElasticTransform', 'HueSaturationValue', 'PadIfNeeded', 'RGBShift', 'RandomBrightness', 'RandomContrast',
           'MotionBlur', 'MedianBlur', 'GaussNoise', 'CLAHE', 'ChannelShuffle', 'InvertImg', 'ToGray',
           'JpegCompression']


class PadIfNeeded(DualTransform):
    """Pads side of the image / max if side is less than desired number.

    Args:
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask

    TODO: add application to boxes
    """

    def __init__(self, min_height=1024, min_width=1024, p=1.0):
        super(PadIfNeeded, self).__init__(p)
        self.min_height = min_height
        self.min_width = min_width

    def apply(self, img, **params):
        return F.pad(img, min_height=self.min_height, min_width=self.min_width)

    def apply_to_bbox(self, bbox, **params):
        pass


class VerticalFlip(DualTransform):
    """Flips the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes
    """

    def apply(self, img, **params):
        return F.vflip(img)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_vflip(bbox, **params)


class HorizontalFlip(DualTransform):
    """Flips the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes
    """

    def apply(self, img, **params):
        return F.hflip(img)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_hflip(bbox, **params)


class Flip(DualTransform):
    """Flips the input either horizontally, vertically or both horizontally and vertically.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def apply(self, img, d=0, **params):
        """
        Args:
            d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """

        return F.random_flip(img, d)

    def get_params(self):
        # Random int in range [-1, 2)
        return {'d': np.random.randint(-1, 2)}


class Transpose(DualTransform):
    """Transposes the input by swapping rows and columns.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def apply(self, img, **params):
        return F.transpose(img)


class RandomRotate90(DualTransform):
    """Randomly rotates the input by 90 degrees zero or more times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def apply(self, img, factor=0, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """

        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        # Random int in range [0, 4)
        return {'factor': np.random.randint(0, 4)}


class Rotate(DualTransform):
    """Rotates the input by an angle selected randomly from the uniform distribution.

    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: 90
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=.5):
        super(Rotate, self).__init__(p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.rotate(img, angle, interpolation, self.border_mode)

    def get_params(self):
        return {'angle': np.random.uniform(self.limit[0], self.limit[1])}


class ShiftScaleRotate(DualTransform):
    """Randomly applies affine transforms: translates, scales and rotates the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [0, 1]. Default: 0.0625.
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Default: 0.1.
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: 45.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5):
        super(ShiftScaleRotate, self).__init__(p)
        self.shift_limit = to_tuple(shift_limit)
        self.scale_limit = to_tuple(scale_limit)
        self.rotate_limit = to_tuple(rotate_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode)

    def get_params(self):
        return {'angle': np.random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
                'scale': np.random.uniform(1 + self.scale_limit[0], 1 + self.scale_limit[1]),
                'dx': np.random.uniform(self.shift_limit[0], self.shift_limit[1]),
                'dy': np.random.uniform(self.shift_limit[0], self.shift_limit[1])}


class CenterCrop(DualTransform):
    """Crops the central part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask
    """

    def __init__(self, height, width, p=1.0):
        super(CenterCrop, self).__init__(p)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        return F.center_crop(img, self.height, self.width)


class RandomCrop(DualTransform):
    """Crops random part of the input.

        Args:
            height (int): height of the crop.
            width (int): width of the crop.
            p (float): probability of applying the transform. Default: 1.

        Targets:
            image, mask
        """

    def __init__(self, height, width, p=1.0):
        super(RandomCrop, self).__init__(p)
        self.height = height
        self.width = width

    def apply(self, img, h_start=0, w_start=0, **params):
        return F.random_crop(img, self.height, self.width, h_start, w_start)

    def get_params(self):
        return {'h_start': np.random.random(),
                'w_start': np.random.random()}


class OpticalDistortion(DualTransform):
    def __init__(self, distort_limit=0.05, shift_limit=0.05, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5):
        super(OpticalDistortion, self).__init__(p)
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, k=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.optical_distortion(img, k, dx, dy, interpolation, self.border_mode)

    def get_params(self):
        return {'k': np.random.uniform(self.distort_limit[0], self.distort_limit[1]),
                'dx': round(np.random.uniform(self.shift_limit[0], self.shift_limit[1])),
                'dy': round(np.random.uniform(self.shift_limit[0], self.shift_limit[1]))}


class GridDistortion(DualTransform):
    def __init__(self, num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5):
        super(GridDistortion, self).__init__(p)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, stepsx=[], stepsy=[], interpolation=cv2.INTER_LINEAR, **params):
        return F.grid_distortion(img, self.num_steps, stepsx, stepsy, interpolation, self.border_mode)

    def get_params(self):
        stepsx = [1 + np.random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in
                  range(self.num_steps + 1)]
        stepsy = [1 + np.random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in
                  range(self.num_steps + 1)]
        return {
            'stepsx': stepsx,
            'stepsy': stepsy
        }


class ElasticTransform(DualTransform):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, p=0.5):
        super(ElasticTransform, self).__init__(p)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, random_state=None, interpolation=cv2.INTER_LINEAR, **params):
        return F.elastic_transform_fast(img, self.alpha, self.sigma, self.alpha_affine, interpolation,
                                        self.border_mode, np.random.RandomState(random_state))

    def get_params(self):
        return {'random_state': np.random.randint(0, 10000)}


class Normalize(ImageOnlyTransform):
    """Divides pixel values by 255 = 2**8 - 1, subtracts mean per channel and divides by std per channel

        Args:
            mean (float, float, float) - mean values
            std  (float, float, float) - std values

        Targets:
            image
        """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0):
        super(Normalize, self).__init__(p)
        self.mean = mean
        self.std = std

    def apply(self, image, **params):
        return F.normalize(image, self.mean, self.std)


class JpegCompression(ImageOnlyTransform):
    """Decreases Jpeg compression of an image.
        Was essential part of the [IEEE's Signal Processing Society - Camera Model Identification Challenge]
            (https://www.kaggle.com/c/sp-society-camera-model-identification)


        Args:
            quality_lower (float) - lower bound on the jpeg quality. Should be in [0, 100] range
            quality_upper (float) - lower bound on the jpeg quality. Should be in [0, 100] range

        Targets:
            image
        """

    def __init__(self, quality_lower=99, quality_upper=100, p=0.5):
        super(JpegCompression, self).__init__(p)

        assert 0 <= quality_lower <= 100
        assert 0 <= quality_upper <= 100

        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def apply(self, image, quality=100, **params):
        return F.jpeg_compression(image, quality)

    def get_params(self):
        return {'quality': np.random.randint(self.quality_lower, self.quality_upper)}


class HueSaturationValue(ImageOnlyTransform):
    """Randomly changes hue, saturation and value of the input image.

    Args:
        hue_shift_limit ((int, int) or int): range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: 20.
        sat_shift_limit ((int, int) or int): range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: 30.
        val_shift_limit ((int, int) or int): range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: 20.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5):
        super(HueSaturationValue, self).__init__(p)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0, **params):
        assert image.dtype == np.uint8 or self.hue_shift_limit < 1.
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return {'hue_shift': np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
                'sat_shift': np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
                'val_shift': np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])}


class RGBShift(ImageOnlyTransform):
    """Randomly shifts values for each channel of the input RGB image.

    Args:
        r_shift_limit ((int, int) or int): range for changing values for the red channel. If r_shift_limit is a single
            int, the range will be (-r_shift_limit, r_shift_limit). Default: 20.
        g_shift_limit ((int, int) or int): range for changing values for the green channel. If g_shift_limit is a
            single int, the range  will be (-g_shift_limit, g_shift_limit). Default: 20.
        b_shift_limit ((int, int) or int): range for changing values for the blue channel. If b_shift_limit is a single
            int, the range will be (-b_shift_limit, b_shift_limit). Default: 20.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5):
        super(RGBShift, self).__init__(p)
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def apply(self, image, r_shift=0, g_shift=0, b_shift=0, **params):
        return F.shift_rgb(image, r_shift, g_shift, b_shift)

    def get_params(self):
        return {'r_shift': np.random.randint(self.r_shift_limit[0], self.r_shift_limit[1]),
                'g_shift': np.random.randint(self.g_shift_limit[0], self.g_shift_limit[1]),
                'b_shift': np.random.randint(self.b_shift_limit[0], self.b_shift_limit[1])}


class RandomBrightness(ImageOnlyTransform):
    """Randomly changes brightness of the input image.

    Args:
        limit ((float, float) or float): factor range for changing brightness. If limit is a single float, the range
            will be (-limit, limit). Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, limit=0.2, p=0.5):
        super(RandomBrightness, self).__init__(p)
        self.limit = to_tuple(limit)

    def apply(self, img, alpha=0.2, **params):
        return F.random_brightness(img, alpha)

    def get_params(self):
        return {'alpha': 1.0 + np.random.uniform(self.limit[0], self.limit[1])}


class RandomContrast(ImageOnlyTransform):
    """Randomly changes contrast of the input image.

    Args:
        limit ((float, float) or float): factor range for changing contrast. If limit is a single float, the range
            will be (-limit, limit). Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, limit=.2, p=.5):
        super(RandomContrast, self).__init__(p)
        self.limit = to_tuple(limit)

    def apply(self, img, alpha=0.2, **params):
        return F.random_contrast(img, alpha)

    def get_params(self):
        return {'alpha': 1.0 + np.random.uniform(self.limit[0], self.limit[1])}


class Blur(ImageOnlyTransform):
    """Blurs the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image. Default: 7.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, blur_limit=7, p=.5):
        super(Blur, self).__init__(p)
        self.blur_limit = to_tuple(blur_limit, 3)

    def apply(self, image, ksize=3, **params):
        return F.blur(image, ksize)

    def get_params(self):
        return {
            'ksize': np.random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        }


class MotionBlur(Blur):
    """Applies motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image. Default: 7.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def apply(self, img, ksize=9, **params):
        return F.motion_blur(img, ksize=ksize)


class MedianBlur(Blur):
    """Blurs the input image using using a median filter with a random aperture linear size.

    Args:
        blur_limit (int): maximum aperture linear size for blurring the input image. Default: 7.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def apply(self, image, ksize=3, **params):
        return F.median_blur(image, ksize)


class GaussNoise(ImageOnlyTransform):
    """Applies gaussian noise to the input image.

    Args:
        var_limit ((int, int) or int): variance range for noise. If var_limit is a single int, the range
            will be (-var_limit, var_limit). Default: (10, 50).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, var_limit=(10, 50), p=0.5):
        super(GaussNoise, self).__init__(p)
        self.var_limit = to_tuple(var_limit)

    def apply(self, img, var=30, **params):
        return F.gauss_noise(img, var=var)

    def get_params(self):
        return {
            'var': np.random.randint(self.var_limit[0], self.var_limit[1])
        }


class CLAHE(ImageOnlyTransform):
    """Applies Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float): upper threshold value for contrast limiting. Default: 4.0.
            tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), p=0.5):
        super(CLAHE, self).__init__(p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = tile_grid_size

    def apply(self, img, clip_limit=2, **params):
        return F.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self):
        return {'clip_limit': np.random.uniform(self.clip_limit[0], self.clip_limit[1])}


class ChannelShuffle(ImageOnlyTransform):
    """Randomly rearranges channels of the input RGB image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def apply(self, img, **params):
        return F.channel_shuffle(img)


class InvertImg(ImageOnlyTransform):
    """Inverts the input image by subtracting pixel values from 255.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def apply(self, img, **params):
        return F.invert(img)


class RandomGamma(ImageOnlyTransform):
    def __init__(self, gamma_limit=(80, 120), p=0.5):
        super(RandomGamma, self).__init__(p)
        self.gamma_limit = gamma_limit

    def apply(self, img, gamma=1, **params):
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self):
        return {
            'gamma': np.random.randint(self.gamma_limit[0], self.gamma_limit[1]) / 100.0
        }


class ToGray(ImageOnlyTransform):
    """Converts the input RGB image to grayscale. If the mean pixel value for the resulting image is greater
    than 127, inverts the resulting grayscale image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def apply(self, img, **params):
        return F.to_gray(img)
