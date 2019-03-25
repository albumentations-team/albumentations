from __future__ import absolute_import, division

import math
import random
import warnings

import cv2
import numpy as np

from . import functional as F
from .bbox_utils import union_of_bboxes, denormalize_bbox, normalize_bbox
from ..core.transforms_interface import to_tuple, DualTransform, ImageOnlyTransform

__all__ = ['Blur', 'VerticalFlip', 'HorizontalFlip', 'Flip', 'Normalize', 'Transpose', 'RandomCrop', 'RandomGamma',
           'RandomRotate90', 'Rotate', 'ShiftScaleRotate', 'CenterCrop', 'OpticalDistortion', 'GridDistortion',
           'ElasticTransform', 'HueSaturationValue', 'PadIfNeeded', 'RGBShift', 'RandomBrightness', 'RandomContrast',
           'MotionBlur', 'MedianBlur', 'GaussianBlur', 'GaussNoise', 'CLAHE', 'ChannelShuffle', 'InvertImg', 'ToGray',
           'JpegCompression', 'Cutout', 'ToFloat', 'FromFloat', 'Crop', 'RandomScale', 'LongestMaxSize',
           'SmallestMaxSize', 'Resize', 'RandomSizedCrop', 'RandomBrightnessContrast', 'RandomCropNearBBox',
           'RandomSizedBBoxSafeCrop', 'RandomSnow', 'RandomRain', 'RandomFog', 'RandomSunFlare',
           'RandomShadow']


class PadIfNeeded(DualTransform):
    """Pad side of the image / max if side is less than desired number.

    Args:
        p (float): probability of applying the transform. Default: 1.0.
        value (list of ints [r, g, b]): padding value if border_mode is cv2.BORDER_CONSTANT.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32

    """

    def __init__(self, min_height=1024, min_width=1024, border_mode=cv2.BORDER_REFLECT_101,
                 value=[0, 0, 0], always_apply=False, p=1.0):
        super(PadIfNeeded, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params['rows']
        cols = params['cols']

        if rows < self.min_height:
            h_pad_top = int((self.min_height - rows) / 2.0)
            h_pad_bottom = self.min_height - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < self.min_width:
            w_pad_left = int((self.min_width - cols) / 2.0)
            w_pad_right = self.min_width - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        params.update({'pad_top': h_pad_top,
                       'pad_bottom': h_pad_bottom,
                       'pad_left': w_pad_left,
                       'pad_right': w_pad_right})
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right,
                                 border_mode=self.border_mode, value=self.value)

    def apply_to_bbox(self, bbox, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        bbox = [x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top]
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, a, s = keypoint
        return [x + pad_left, y + pad_top, a, s]


class Crop(DualTransform):
    """Crop region from image.

    Args:
        x_min (int): minimum upper left x coordinate
        y_min (int): minimum upper left y coordinate
        x_max (int): maximum lower right x coordinate
        y_max (int): maximum lower right y coordinate

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, x_min=0, y_min=0, x_max=1024, y_max=1024, always_apply=False, p=1.0):
        super(Crop, self).__init__(always_apply, p)
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def apply(self, img, **params):
        return F.crop(img, x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_crop(bbox, x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max, **params)


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        return F.vflip(img)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_vflip(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_vflip(keypoint, **params)


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return F.hflip_cv2(img)
        else:
            return F.hflip(img)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_hflip(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_hflip(keypoint, **params)


class Flip(DualTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, d=0, **params):
        """Args:
        d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        return F.random_flip(img, d)

    def get_params(self):
        # Random int in the range [-1, 1]
        return {'d': random.randint(-1, 1)}

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_flip(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_flip(keypoint, **params)


class Transpose(DualTransform):
    """Transpose the input by swapping rows and columns.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        return F.transpose(img)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_transpose(bbox, 0, **params)


class LongestMaxSize(DualTransform):
    """Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        p (float): probability of applying the transform. Default: 1.
        max_size (int): maximum size of the image after the transformation

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, max_size=1024, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(LongestMaxSize, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        return F.longest_max_size(img, max_size=self.max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox


class SmallestMaxSize(DualTransform):
    """Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        p (float): probability of applying the transform. Default: 1.
        max_size (int): maximum size of smallest side of the image after the transformation

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, max_size=1024, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(SmallestMaxSize, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        return F.smallest_max_size(img, max_size=self.max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        return bbox


class Resize(DualTransform):
    """Resize the input to the given height and width.

    Args:
        p (float): probability of applying the transform. Default: 1.
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(Resize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        return F.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox


class RandomRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, factor=0, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        # Random int in the range [0, 3]
        return {'factor': random.randint(0, 3)}

    def apply_to_bbox(self, bbox, factor=0, **params):
        return F.bbox_rot90(bbox, factor, **params)

    def apply_to_keypoint(self, keypoint, factor=0, **params):
        return F.keypoint_rot90(keypoint, factor, **params)


class Rotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.

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
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101,
                 always_apply=False, p=.5):
        super(Rotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.rotate(img, angle, interpolation, self.border_mode)

    def get_params(self):
        return {'angle': random.uniform(self.limit[0], self.limit[1])}

    def apply_to_bbox(self, bbox, angle=0, **params):
        return F.bbox_rotate(bbox, angle, **params)

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        return F.keypoint_rotate(keypoint, angle, **params)


class RandomScale(DualTransform):
    """Randomly resize the input. Output image size is different from the input image size.

    Args:
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (1 - scale_limit, 1 + scale_limit). Default: 0.1.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, scale_limit=0.1, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5):
        super(RandomScale, self).__init__(always_apply, p)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.interpolation = interpolation

    def get_params(self):
        return {'scale': random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, img, scale=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.scale(img, scale, interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, scale=0, **params):
        return F.keypoint_scale(keypoint, scale, scale)


class ShiftScaleRotate(DualTransform):
    """Randomly apply affine transforms: translate, scale and rotate the input.

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
        image, mask, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5):
        super(ShiftScaleRotate, self).__init__(always_apply, p)
        self.shift_limit = to_tuple(shift_limit)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.rotate_limit = to_tuple(rotate_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode)

    def apply_to_keypoint(self, keypoint, angle=0, scale=0, dx=0, dy=0, rows=0, cols=0, interpolation=cv2.INTER_LINEAR,
                          **params):
        return F.keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols)

    def get_params(self):
        return {'angle': random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
                'scale': random.uniform(self.scale_limit[0], self.scale_limit[1]),
                'dx': random.uniform(self.shift_limit[0], self.shift_limit[1]),
                'dy': random.uniform(self.shift_limit[0], self.shift_limit[1])}

    def apply_to_bbox(self, bbox, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, **params):
        return F.bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, **params)


class CenterCrop(DualTransform):
    """Crop the central part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        It is recommended to use uint8 images as input.
        Otherwise the operation will require internal conversion
        float32 -> uint8 -> float32 that causes worse performance.
    """

    def __init__(self, height, width, always_apply=False, p=1.0):
        super(CenterCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        return F.center_crop(img, self.height, self.width)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_center_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_center_crop(keypoint, self.height, self.width, **params)


class RandomCrop(DualTransform):
    """Crop a random part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, always_apply=False, p=1.0):
        super(RandomCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, h_start=0, w_start=0, **params):
        return F.random_crop(img, self.height, self.width, h_start, w_start)

    def get_params(self):
        return {'h_start': random.random(),
                'w_start': random.random()}

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_random_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_random_crop(keypoint, self.height, self.width, **params)


class RandomCropNearBBox(DualTransform):
    """Crop bbox from image with random shift by x,y coordinates

    Args:
        max_part_shift (float): float value in (0.0, 1.0) range. Default 0.3
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, max_part_shift=0.3, always_apply=False, p=1.0):
        super(RandomCropNearBBox, self).__init__(always_apply, p)
        self.max_part_shift = max_part_shift

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return F.clamping_crop(img, x_min, y_min, x_max, y_max)

    def get_params_dependent_on_targets(self, params):
        bbox = params['cropping_bbox']
        h_max_shift = int((bbox[3] - bbox[1]) * self.max_part_shift)
        w_max_shift = int((bbox[2] - bbox[0]) * self.max_part_shift)

        x_min = bbox[0] - random.randint(-w_max_shift, w_max_shift)
        x_max = bbox[2] + random.randint(-w_max_shift, w_max_shift)

        y_min = bbox[1] - random.randint(-h_max_shift, h_max_shift)
        y_max = bbox[3] + random.randint(-h_max_shift, h_max_shift)

        return {'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
                }

    def apply_to_bbox(self, bbox, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        h_start = y_min
        w_start = x_min
        return F.bbox_crop(bbox, y_max - y_min, x_max - x_min, h_start, w_start, **params)

    @property
    def targets_as_params(self):
        return ['cropping_bbox']


class RandomSizedCrop(DualTransform):
    """Crop a random part of the input and rescale it to some size.

    Args:
        min_max_height ((int, int)): crop size limits.
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        w2h_ratio (float): aspect ratio of crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, min_max_height, height, width, w2h_ratio=1., interpolation=cv2.INTER_LINEAR,
                 always_apply=False, p=1.0):
        super(RandomSizedCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return F.resize(crop, self.height, self.width, interpolation)

    def get_params(self):
        crop_height = random.randint(self.min_max_height[0], self.min_max_height[1])
        return {'h_start': random.random(),
                'w_start': random.random(),
                'crop_height': crop_height,
                'crop_width': int(crop_height * self.w2h_ratio)}

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    def apply_to_keypoint(self, keypoint, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        keypoint = F.keypoint_random_crop(keypoint, crop_height, crop_width, h_start, w_start, rows, cols)
        scale_x = self.width / crop_height
        scale_y = self.height / crop_height
        keypoint = F.keypoint_scale(keypoint, scale_x, scale_y)
        return keypoint


class RandomSizedBBoxSafeCrop(DualTransform):
    """Crop a random part of the input and rescale it to some size without loss of bboxes.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, erosion_rate=0.0, interpolation=cv2.INTER_LINEAR,
                 always_apply=False, p=1.0):
        super(RandomSizedBBoxSafeCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.erosion_rate = erosion_rate

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return F.resize(crop, self.height, self.width, interpolation)

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params['image'].shape[:2]
        if 'bboxes' not in params:  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            return {'h_start': random.random(),
                    'w_start': random.random(),
                    'crop_height': crop_height,
                    'crop_width': int(crop_height * img_w / img_h)}
        # get union of all bboxes
        x, y, x2, y2 = union_of_bboxes(width=img_w, height=img_h, bboxes=params['bboxes'],
                                       erosion_rate=self.erosion_rate)
        # find bigger region
        bx, by = x * random.random(), y * random.random()
        bx2, by2 = x2 + (1 - x2) * random.random(), y2 + (1 - y2) * random.random()
        bw, bh = bx2 - bx, by2 - by
        crop_height, crop_width = int(img_h * bh), int(img_w * bw)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        return {'h_start': h_start,
                'w_start': w_start,
                'crop_height': crop_height,
                'crop_width': crop_width}

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    @property
    def targets_as_params(self):
        return ['image']


class OpticalDistortion(DualTransform):
    """
    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(self, distort_limit=0.05, shift_limit=0.05, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5):
        super(OpticalDistortion, self).__init__(always_apply, p)
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, k=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.optical_distortion(img, k, dx, dy, interpolation, self.border_mode)

    def get_params(self):
        return {'k': random.uniform(self.distort_limit[0], self.distort_limit[1]),
                'dx': round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
                'dy': round(random.uniform(self.shift_limit[0], self.shift_limit[1]))}


class GridDistortion(DualTransform):
    """
    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(self, num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5):
        super(GridDistortion, self).__init__(always_apply, p)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def apply(self, img, stepsx=[], stepsy=[], interpolation=cv2.INTER_LINEAR, **params):
        return F.grid_distortion(img, self.num_steps, stepsx, stepsy, interpolation, self.border_mode)

    def get_params(self):
        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in
                  range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in
                  range(self.num_steps + 1)]
        return {
            'stepsx': stepsx,
            'stepsy': stepsy
        }


class ElasticTransform(DualTransform):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Args:
        approximate (boolean): Whether to smooth displacement map with fixed kernel size.
                               Enabling this option gives ~2X speedup on large images.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(self, alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, always_apply=False, approximate=False, p=0.5):
        super(ElasticTransform, self).__init__(always_apply, p)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.approximate = approximate

    def apply(self, img, random_state=None, interpolation=cv2.INTER_LINEAR, **params):
        return F.elastic_transform(img, self.alpha, self.sigma, self.alpha_affine, interpolation,
                                   self.border_mode, np.random.RandomState(random_state),
                                   self.approximate)

    def get_params(self):
        return {'random_state': random.randint(0, 10000)}


class Normalize(ImageOnlyTransform):
    """Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.

    Args:
        mean (float, float, float): mean values
        std  (float, float, float): std values
        max_pixel_value (float): maximum possible pixel value

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
                 always_apply=False, p=1.0):
        super(Normalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, image, **params):
        return F.normalize(image, self.mean, self.std, self.max_pixel_value)


class Cutout(ImageOnlyTransform):
    """CoarseDropout of the square regions in the image.

    Args:
        num_holes (int): number of regions to zero out
        max_h_size (int): maximum height of the hole
        max_w_size (int): maximum width of the hole

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py

    """

    def __init__(self, num_holes=8, max_h_size=8, max_w_size=8, always_apply=False, p=0.5):
        super(Cutout, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def apply(self, image, holes=[], **params):
        return F.cutout(image, holes)

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]

        holes = []
        for n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y + self.max_h_size // 2, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x + self.max_w_size // 2, 0, width)
            holes.append((x1, y1, x2, y2))

        return {'holes': holes}

    @property
    def targets_as_params(self):
        return ['image']


class JpegCompression(ImageOnlyTransform):
    """Decrease Jpeg compression of an image.

    Args:
        quality_lower (float): lower bound on the jpeg quality. Should be in [0, 100] range
        quality_upper (float): upper bound on the jpeg quality. Should be in [0, 100] range

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, quality_lower=99, quality_upper=100, always_apply=False, p=0.5):
        super(JpegCompression, self).__init__(always_apply, p)

        assert 0 <= quality_lower <= 100
        assert 0 <= quality_upper <= 100

        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def apply(self, image, quality=100, **params):
        return F.jpeg_compression(image, quality)

    def get_params(self):
        return {'quality': random.randint(self.quality_lower, self.quality_upper)}


class RandomSnow(ImageOnlyTransform):
    """Bleach out some pixel values simulating snow.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        snow_point_lower (float): lower_bond of the amount of snow. Should be in [0, 1] range
        snow_point_upper (float): upper_bond of the amount of snow. Should be in [0, 1] range
        brightness_coeff (float): larger number will lead to a more snow on the image. Should be >= 0

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=0.5):
        super(RandomSnow, self).__init__(always_apply, p)

        assert 0 <= snow_point_lower <= snow_point_upper <= 1
        assert 0 <= brightness_coeff

        self.snow_point_lower = snow_point_lower
        self.snow_point_upper = snow_point_upper
        self.brightness_coeff = brightness_coeff

    def apply(self, image, snow_point=0.1, **params):
        return F.add_snow(image, snow_point, self.brightness_coeff)

    def get_params(self):
        return {'snow_point': random.uniform(self.snow_point_lower, self.snow_point_upper)}


class RandomRain(ImageOnlyTransform):
    """Adds rain effects.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        slant_lower:
        slant_upper:
        drop_length:
        drop_width:
        drop_color:
        blur_value (int): rainy view are blurry
        brightness_coefficient (float): rainy days are usually shady
        rain_type: [None, "drizzle", "heavy", "torrestial"]


    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self,
                 slant_lower=-10,
                 slant_upper=10,
                 drop_length=20,
                 drop_width=1,
                 drop_color=(200, 200, 200),
                 blur_value=7,
                 brightness_coefficient=0.7,
                 rain_type=None,
                 always_apply=False, p=0.5):
        super(RandomRain, self).__init__(always_apply, p)

        assert rain_type in ['drizzle', 'heavy', 'torrential', None]

        assert -20 <= slant_lower <= slant_upper <= 20
        assert 1 <= drop_width <= 5
        assert 0 <= drop_length <= 100
        assert 0 <= brightness_coefficient <= 1

        self.slant_lower = slant_lower
        self.slant_upper = slant_upper

        self.drop_length = drop_length
        self.drop_width = drop_width
        self.drop_color = drop_color
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient
        self.rain_type = rain_type

    def apply(self, image, slant=10, drop_length=20, rain_drops=[], **params):
        return F.add_rain(image,
                          slant,
                          drop_length,
                          self.drop_width,
                          self.drop_color,
                          self.blur_value,
                          self.brightness_coefficient,
                          rain_drops)

    @property
    def targets_as_params(self):
        return ['image']

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        slant = int(random.uniform(self.slant_lower, self.slant_upper))

        height, width = img.shape[:2]
        area = height * width

        if self.rain_type == 'drizzle':
            num_drops = area // 770
            drop_length = 10
        elif self.rain_type == 'heavy':
            num_drops = width * height // 600
            drop_length = 30
        elif self.rain_type == 'torrential':
            num_drops = area // 500
            drop_length = 60
        else:
            drop_length = self.drop_length
            num_drops = area // 600

        rain_drops = []

        for i in range(num_drops):  # If You want heavy rain, try increasing this
            if slant < 0:
                x = random.randint(slant, width)
            else:
                x = random.randint(0, width - slant)

            y = random.randint(0, height - drop_length)

            rain_drops.append((x, y))

        return {'drop_length': drop_length,
                'rain_drops': rain_drops}


class RandomFog(ImageOnlyTransform):
    """Simulates fog for the image

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        fog_coef_lower (float): lower limit for fog intensity coefficient. Should be in [0, 1] range.
        fog_coef_upper (float): upper limit for fog intensity coefficient. Should be in [0, 1] range.
        alpha_coef (float): transparence of the fog circles. Should be in [0, 1] range.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.5):
        super(RandomFog, self).__init__(always_apply, p)

        assert 0 <= fog_coef_lower <= fog_coef_upper <= 1
        assert 0 <= alpha_coef <= 1

        self.fog_coef_lower = fog_coef_lower
        self.fog_coef_upper = fog_coef_upper
        self.alpha_coef = alpha_coef

    def apply(self, image, fog_coef=0.1, haze_list=[], **params):
        return F.add_fog(image, fog_coef, self.alpha_coef, haze_list)

    @property
    def targets_as_params(self):
        return ['image']

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        fog_coef = random.uniform(self.fog_coef_lower, self.fog_coef_upper)

        height, width = imshape = img.shape[:2]

        hw = int(width // 3 * fog_coef)

        haze_list = []
        midx = width // 2 - 2 * hw
        midy = height // 2 - hw
        index = 1

        while midx > -hw or midy > - hw:
            for i in range(hw // 10 * index):
                x = random.randint(midx, width - midx - hw)
                y = random.randint(midy, height - midy - hw)
                haze_list.append((x, y))

            midx -= 3 * hw * width // sum(imshape)
            midy -= 3 * hw * height // sum(imshape)
            index += 1

        return {'haze_list': haze_list,
                'fog_coef': fog_coef}


class RandomSunFlare(ImageOnlyTransform):
    """Simulates Sun Flare for the image

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        flare_roi (float, float, float, float): region of the image where flare will
                                                    appear (x_min, y_min, x_max, y_max)
        angle_lower (float):
        angle_upper (float):
        num_flare_circles_lower (int): lower limit for the number of flare circles.
        num_flare_circles_upper (int): upper limit for the number of flare circles.
        src_radius (int):
        src_color (int, int, int): color of the flare

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self,
                 flare_roi=(0, 0, 1, 0.5),
                 angle_lower=0,
                 angle_upper=1,
                 num_flare_circles_lower=6,
                 num_flare_circles_upper=10,
                 src_radius=400,
                 src_color=(255, 255, 255),
                 always_apply=False,
                 p=0.5):
        super(RandomSunFlare, self).__init__(always_apply, p)

        (flare_center_lower_x, flare_center_lower_y, flare_center_upper_x, flare_center_upper_y) = flare_roi

        assert 0 <= flare_center_lower_x < flare_center_upper_x <= 1
        assert 0 <= flare_center_lower_y < flare_center_upper_y <= 1
        assert 0 <= angle_lower < angle_upper <= 1
        assert 0 <= num_flare_circles_lower < num_flare_circles_upper

        self.flare_center_lower_x = flare_center_lower_x
        self.flare_center_upper_x = flare_center_upper_x

        self.flare_center_lower_y = flare_center_lower_y
        self.flare_center_upper_y = flare_center_upper_y

        self.angle_lower = angle_lower
        self.angle_upper = angle_upper
        self.num_flare_circles_lower = num_flare_circles_lower
        self.num_flare_circles_upper = num_flare_circles_upper

        self.src_radius = src_radius
        self.src_color = src_color

    def apply(self,
              image,
              flare_center_x=0.5,
              flare_center_y=0.5,
              circles=[], **params):

        return F.add_sun_flare(image, flare_center_x, flare_center_y,
                               self.src_radius,
                               self.src_color,
                               circles)

    @property
    def targets_as_params(self):
        return ['image']

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]

        angle = 2 * math.pi * random.uniform(self.angle_lower, self.angle_upper)

        flare_center_x = random.uniform(self.flare_center_lower_x, self.flare_center_upper_x)
        flare_center_y = random.uniform(self.flare_center_lower_y, self.flare_center_upper_y)

        flare_center_x = int(width * flare_center_x)
        flare_center_y = int(height * flare_center_y)

        num_circles = random.randint(self.num_flare_circles_lower, self.num_flare_circles_upper)

        circles = []

        x = []
        y = []

        for rand_x in range(0, width, 10):
            rand_y = math.tan(angle) * (rand_x - flare_center_x) + flare_center_y
            x.append(rand_x)
            y.append(2 * flare_center_y - rand_y)

        for i in range(num_circles):
            alpha = random.uniform(0.05, 0.2)
            r = random.randint(0, len(x) - 1)
            rad = random.randint(1, max(height // 100 - 2, 2))

            r_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])
            g_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])
            b_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])

            circles += [(alpha, (int(x[r]), int(y[r])), pow(rad, 3), (r_color,
                                                                      g_color,
                                                                      b_color))]

        return {'circles': circles,
                'flare_center_x': flare_center_x,
                'flare_center_y': flare_center_y}


class RandomShadow(ImageOnlyTransform):
    """Simulates shadows for the image

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        shadow_roi (float, float, float, float): region of the image where shadows
            will appear (x_min, y_min, x_max, y_max)
        num_shadows_lower (int): Lower limit for the possible number of shadows.
        num_shadows_upper (int): Lower limit for the possible number of shadows.
        shadow_dimension (int): number of edges in the shadow polygons

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self,
                 shadow_roi=(0, 0.5, 1, 1),
                 num_shadows_lower=1,
                 num_shadows_upper=2,
                 shadow_dimension=5,
                 always_apply=False,
                 p=0.5):
        super(RandomShadow, self).__init__(always_apply, p)

        (shadow_lower_x, shadow_lower_y, shadow_upper_x, shadow_upper_y) = shadow_roi

        assert 0 <= shadow_lower_x <= shadow_upper_x <= 1
        assert 0 <= shadow_lower_y <= shadow_upper_y <= 1
        assert 0 <= num_shadows_lower <= num_shadows_upper

        self.shadow_roi = shadow_roi

        self.num_shadows_lower = num_shadows_lower
        self.num_shadows_upper = num_shadows_upper

        self.shadow_dimension = shadow_dimension

    def apply(self, image, vertices_list=[], **params):
        return F.add_shadow(image, vertices_list)

    @property
    def targets_as_params(self):
        return ['image']

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]

        num_shadows = random.randint(self.num_shadows_lower, self.num_shadows_upper)

        x_min, y_min, x_max, y_max = self.shadow_roi

        x_min = int(x_min * width)
        x_max = int(x_max * width)
        y_min = int(y_min * height)
        y_max = int(y_max * height)

        vertices_list = []

        for index in range(num_shadows):
            vertex = []
            for dimensions in range(self.shadow_dimension):
                vertex.append((random.randint(x_min, x_max), random.randint(y_min, y_max)))

            vertices = np.array([vertex], dtype=np.int32)
            vertices_list.append(vertices)

        return {'vertices_list': vertices_list}


class HueSaturationValue(ImageOnlyTransform):
    """Randomly change hue, saturation and value of the input image.

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

    Image types:
        uint8, float32
    """

    def __init__(self, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5):
        super(HueSaturationValue, self).__init__(always_apply, p)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0, **params):
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return {'hue_shift': random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
                'sat_shift': random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
                'val_shift': random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])}


class RGBShift(ImageOnlyTransform):
    """Randomly shift values for each channel of the input RGB image.

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

    Image types:
        uint8, float32
    """

    def __init__(self, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5):
        super(RGBShift, self).__init__(always_apply, p)
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def apply(self, image, r_shift=0, g_shift=0, b_shift=0, **params):
        return F.shift_rgb(image, r_shift, g_shift, b_shift)

    def get_params(self):
        return {'r_shift': random.uniform(self.r_shift_limit[0], self.r_shift_limit[1]),
                'g_shift': random.uniform(self.g_shift_limit[0], self.g_shift_limit[1]),
                'b_shift': random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])}


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: 0.2.
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5):
        super(RandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)

    def apply(self, img, alpha=1., beta=0., **params):
        return F.brightness_contrast_adjust(img, alpha, beta)

    def get_params(self):
        return {
            'alpha': 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            'beta': 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])
        }


class RandomBrightness(RandomBrightnessContrast):
    def __init__(self, limit=0.2, always_apply=False, p=0.5):
        super(RandomBrightness, self).__init__(brightness_limit=limit, contrast_limit=0,
                                               always_apply=always_apply, p=p)
        warnings.warn("This class has been deprecated. Please use RandomBrightnessContrast", DeprecationWarning)


class RandomContrast(RandomBrightnessContrast):
    def __init__(self, limit=0.2, always_apply=False, p=0.5):
        super(RandomContrast, self).__init__(brightness_limit=0, contrast_limit=limit, always_apply=always_apply, p=p)
        warnings.warn("This class has been deprecated. Please use RandomBrightnessContrast", DeprecationWarning)


class Blur(ImageOnlyTransform):
    """Blur the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image. Default: 7.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=7, always_apply=False, p=.5):
        super(Blur, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)

    def apply(self, image, ksize=3, **params):
        return F.blur(image, ksize)

    def get_params(self):
        return {
            'ksize': random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        }


class MotionBlur(Blur):
    """Apply motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image. Default: 7.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def apply(self, img, kernel=None, **params):
        return F.motion_blur(img, kernel=kernel)

    def get_params(self):
        ksize = random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        assert ksize > 2
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        xs, xe = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        if xs == xe:
            ys, ye = random.sample(range(ksize), 2)
        else:
            ys, ye = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
        return {
            'kernel': kernel
        }


class MedianBlur(Blur):
    """Blur the input image using using a median filter with a random aperture linear size.

    Args:
        blur_limit (int): maximum aperture linear size for blurring the input image. Default: 7.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=7, always_apply=False, p=0.5):
        super(MedianBlur, self).__init__(blur_limit, always_apply, p)

    def apply(self, image, ksize=3, **params):
        return F.median_blur(image, ksize)


class GaussianBlur(Blur):
    """Blur the input image using using a Gaussian filter with a random kernel size.

    Args:
        blur_limit (int): maximum Gaussian kernel size for blurring the input image. Default: 7.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=7, always_apply=False, p=0.5):
        super(GaussianBlur, self).__init__(blur_limit, always_apply, p)

    def apply(self, image, ksize=3, **params):
        return F.gaussian_blur(image, ksize)


class GaussNoise(ImageOnlyTransform):
    """Apply gaussian noise to the input image.

    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (-var_limit, var_limit). Default: (10., 50.).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self, var_limit=(10., 50.), always_apply=False, p=0.5):
        super(GaussNoise, self).__init__(always_apply, p)
        self.var_limit = to_tuple(var_limit)

    def apply(self, img, gauss=None, **params):
        return F.gauss_noise(img, gauss=gauss)

    def get_params_dependent_on_targets(self, params):
        image = params['image']
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        mean = var
        sigma = var ** 0.5
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        gauss = random_state.normal(mean, sigma, image.shape)
        gauss = gauss - np.min(gauss)
        return {
            'gauss': gauss
        }

    @property
    def targets_as_params(self):
        return ['image']


class CLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float): upper threshold value for contrast limiting. Default: 4.0.
            tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5):
        super(CLAHE, self).__init__(always_apply, p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = tile_grid_size

    def apply(self, img, clip_limit=2, **params):
        return F.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self):
        return {'clip_limit': random.uniform(self.clip_limit[0], self.clip_limit[1])}


class ChannelShuffle(ImageOnlyTransform):
    """Randomly rearrange channels of the input RGB image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def apply(self, img, channels_shuffled=[0, 1, 2], **params):
        return F.channel_shuffle(img, channels_shuffled)

    def get_params(self):
        ch_arr = [0, 1, 2]
        random.shuffle(ch_arr)
        return {'channels_shuffled': ch_arr}


class InvertImg(ImageOnlyTransform):
    """Invert the input image by subtracting pixel values from 255.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def apply(self, img, **params):
        return F.invert(img)


class RandomGamma(ImageOnlyTransform):
    """
    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, gamma_limit=(80, 120), always_apply=False, p=0.5):
        super(RandomGamma, self).__init__(always_apply, p)
        self.gamma_limit = gamma_limit

    def apply(self, img, gamma=1, **params):
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self):
        return {
            'gamma': random.randint(self.gamma_limit[0], self.gamma_limit[1]) / 100.0
        }


class ToGray(ImageOnlyTransform):
    """Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater
    than 127, invert the resulting grayscale image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        return F.to_gray(img)


class ToFloat(ImageOnlyTransform):
    """Divide pixel values by `max_value` to get a float32 output array where all values lie in the range [0, 1.0].
    If `max_value` is None the transform will try to infer the maximum value by inspecting the data type of the input
    image.

    See Also:
        :class:`~albumentations.augmentations.transforms.FromFloat`

    Args:
        max_value (float): maximum possible input value. Default: None.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        any type

    """

    def __init__(self, max_value=None, always_apply=False, p=1.0):
        super(ToFloat, self).__init__(always_apply, p)
        self.max_value = max_value

    def apply(self, img, **params):
        return F.to_float(img, self.max_value)


class FromFloat(ImageOnlyTransform):
    """Take an input array where all values should lie in the range [0, 1.0], multiply them by `max_value` and then
    cast the resulted value to a type specified by `dtype`. If `max_value` is None the transform will try to infer
    the maximum value for the data type from the `dtype` argument.

    This is the inverse transform for :class:`~albumentations.augmentations.transforms.ToFloat`.

    Args:
        max_value (float): maximum possible input value. Default: None.
        dtype (string or numpy data type): data type of the output. See the `'Data types' page from the NumPy docs`_.
            Default: 'uint16'.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        float32

    .. _'Data types' page from the NumPy docs:
       https://docs.scipy.org/doc/numpy/user/basics.types.html
    """

    def __init__(self, dtype='uint16', max_value=None, always_apply=False, p=1.0):
        super(FromFloat, self).__init__(always_apply, p)
        self.dtype = np.dtype(dtype)
        self.max_value = max_value

    def apply(self, img, **params):
        return F.from_float(img, self.dtype, self.max_value)
