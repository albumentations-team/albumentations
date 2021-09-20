from __future__ import absolute_import, division

import math
import random
import numbers
import warnings
from enum import IntEnum, Enum
from types import LambdaType
from typing import Optional, Union, Sequence, Tuple

import cv2
import numpy as np
from skimage.measure import label

from . import functional as F
from .bbox_utils import denormalize_bbox, normalize_bbox
from ..core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    NoOp,
    to_tuple,
)
from ..core.utils import format_args

__all__ = [
    "Blur",
    "VerticalFlip",
    "HorizontalFlip",
    "Flip",
    "Normalize",
    "Transpose",
    "RandomGamma",
    "OpticalDistortion",
    "GridDistortion",
    "RandomGridShuffle",
    "HueSaturationValue",
    "PadIfNeeded",
    "RGBShift",
    "RandomBrightness",
    "RandomContrast",
    "MotionBlur",
    "MedianBlur",
    "GaussianBlur",
    "GaussNoise",
    "GlassBlur",
    "CLAHE",
    "ChannelShuffle",
    "InvertImg",
    "ToGray",
    "ToSepia",
    "JpegCompression",
    "ImageCompression",
    "Cutout",
    "CoarseDropout",
    "ToFloat",
    "FromFloat",
    "RandomBrightnessContrast",
    "RandomSnow",
    "RandomRain",
    "RandomFog",
    "RandomSunFlare",
    "RandomShadow",
    "RandomToneCurve",
    "Lambda",
    "ChannelDropout",
    "ISONoise",
    "Solarize",
    "Equalize",
    "Posterize",
    "Downscale",
    "MultiplicativeNoise",
    "FancyPCA",
    "MaskDropout",
    "GridDropout",
    "ColorJitter",
    "Sharpen",
    "Emboss",
    "Superpixels",
    "TemplateTransform",
]


class PadIfNeeded(DualTransform):
    """Pad side of the image / max if side is less than desired number.

    Args:
        min_height (int): minimal result image height.
        min_width (int): minimal result image width.
        pad_height_divisor (int): if not None, ensures image height is dividable by value of this argument.
        pad_width_divisor (int): if not None, ensures image width is dividable by value of this argument.
        position (Union[str, PositionType]): Position of the image. should be PositionType.CENTER or
            PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT.
            Default: PositionType.CENTER.
        border_mode (OpenCV flag): OpenCV border mode.
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32
    """

    class PositionType(Enum):
        CENTER = "center"
        TOP_LEFT = "top_left"
        TOP_RIGHT = "top_right"
        BOTTOM_LEFT = "bottom_left"
        BOTTOM_RIGHT = "bottom_right"

    def __init__(
        self,
        min_height: Optional[int] = 1024,
        min_width: Optional[int] = 1024,
        pad_height_divisor: Optional[int] = None,
        pad_width_divisor: Optional[int] = None,
        position: Union[PositionType, str] = PositionType.CENTER,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=1.0,
    ):
        if (min_height is None) == (pad_height_divisor is None):
            raise ValueError("Only one of 'min_height' and 'pad_height_divisor' parameters must be set")

        if (min_width is None) == (pad_width_divisor is None):
            raise ValueError("Only one of 'min_width' and 'pad_width_divisor' parameters must be set")

        super(PadIfNeeded, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.pad_width_divisor = pad_width_divisor
        self.pad_height_divisor = pad_height_divisor
        self.position = PadIfNeeded.PositionType(position)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if self.min_height is not None:
            if rows < self.min_height:
                h_pad_top = int((self.min_height - rows) / 2.0)
                h_pad_bottom = self.min_height - rows - h_pad_top
            else:
                h_pad_top = 0
                h_pad_bottom = 0
        else:
            pad_remained = rows % self.pad_height_divisor
            pad_rows = self.pad_height_divisor - pad_remained if pad_remained > 0 else 0

            h_pad_top = pad_rows // 2
            h_pad_bottom = pad_rows - h_pad_top

        if self.min_width is not None:
            if cols < self.min_width:
                w_pad_left = int((self.min_width - cols) / 2.0)
                w_pad_right = self.min_width - cols - w_pad_left
            else:
                w_pad_left = 0
                w_pad_right = 0
        else:
            pad_remainder = cols % self.pad_width_divisor
            pad_cols = self.pad_width_divisor - pad_remainder if pad_remainder > 0 else 0

            w_pad_left = pad_cols // 2
            w_pad_right = pad_cols - w_pad_left

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = self.__update_position_params(
            h_top=h_pad_top, h_bottom=h_pad_bottom, w_left=w_pad_left, w_right=w_pad_right
        )

        params.update(
            {
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
            }
        )
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.mask_value,
        )

    def apply_to_bbox(self, bbox, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    # skipcq: PYL-W0613
    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, angle, scale = keypoint
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return (
            "min_height",
            "min_width",
            "pad_height_divisor",
            "pad_width_divisor",
            "border_mode",
            "value",
            "mask_value",
        )

    def __update_position_params(
        self, h_top: int, h_bottom: int, w_left: int, w_right: int
    ) -> Tuple[int, int, int, int]:
        if self.position == PadIfNeeded.PositionType.TOP_LEFT:
            h_bottom += h_top
            w_right += w_left
            h_top = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.TOP_RIGHT:
            h_bottom += h_top
            w_left += w_right
            h_top = 0
            w_right = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_LEFT:
            h_top += h_bottom
            w_right += w_left
            h_bottom = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_RIGHT:
            h_top += h_bottom
            w_left += w_right
            h_bottom = 0
            w_right = 0

        return h_top, h_bottom, w_left, w_right


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

    def get_transform_init_args_names(self):
        return ()


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

        return F.hflip(img)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_hflip(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_hflip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


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
        return {"d": random.randint(-1, 1)}

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_flip(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_flip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class Transpose(DualTransform):
    """Transpose the input by swapping rows and columns.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        return F.transpose(img)

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_transpose(bbox, 0, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_transpose(keypoint)

    def get_transform_init_args_names(self):
        return ()


class OpticalDistortion(DualTransform):
    """
    Args:
        distort_limit (float, (float, float)): If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: (-0.05, 0.05).
        shift_limit (float, (float, float))): If shift_limit is a single float, the range
            will be (-shift_limit, shift_limit). Default: (-0.05, 0.05).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        distort_limit=0.05,
        shift_limit=0.05,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(OpticalDistortion, self).__init__(always_apply, p)
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, k=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.optical_distortion(img, k, dx, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, k=0, dx=0, dy=0, **params):
        return F.optical_distortion(img, k, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def get_params(self):
        return {
            "k": random.uniform(self.distort_limit[0], self.distort_limit[1]),
            "dx": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
            "dy": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
        }

    def get_transform_init_args_names(self):
        return (
            "distort_limit",
            "shift_limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
        )


class GridDistortion(DualTransform):
    """
    Args:
        num_steps (int): count of grid cells on each side.
        distort_limit (float, (float, float)): If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: (-0.03, 0.03).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        num_steps=5,
        distort_limit=0.3,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(GridDistortion, self).__init__(always_apply, p)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, stepsx=(), stepsy=(), interpolation=cv2.INTER_LINEAR, **params):
        return F.grid_distortion(
            img,
            self.num_steps,
            stepsx,
            stepsy,
            interpolation,
            self.border_mode,
            self.value,
        )

    def apply_to_mask(self, img, stepsx=(), stepsy=(), **params):
        return F.grid_distortion(
            img,
            self.num_steps,
            stepsx,
            stepsy,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
        )

    def get_params(self):
        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in range(self.num_steps + 1)]
        return {"stepsx": stepsx, "stepsy": stepsy}

    def get_transform_init_args_names(self):
        return (
            "num_steps",
            "distort_limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
        )


class RandomGridShuffle(DualTransform):
    """
    Random shuffle grid's cells on image.

    Args:
        grid ((int, int)): size of grid for splitting image.

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(self, grid=(3, 3), always_apply=False, p=0.5):
        super(RandomGridShuffle, self).__init__(always_apply, p)
        self.grid = grid

    def apply(self, img, tiles=None, **params):
        if tiles is None:
            tiles = []

        return F.swap_tiles_on_image(img, tiles)

    def apply_to_mask(self, img, tiles=None, **params):
        if tiles is None:
            tiles = []

        return F.swap_tiles_on_image(img, tiles)

    def get_params_dependent_on_targets(self, params):
        height, width = params["image"].shape[:2]
        n, m = self.grid

        if n <= 0 or m <= 0:
            raise ValueError("Grid's values must be positive. Current grid [%s, %s]" % (n, m))

        if n > height // 2 or m > width // 2:
            raise ValueError("Incorrect size cell of grid. Just shuffle pixels of image")

        random_state = np.random.RandomState(random.randint(0, 10000))

        height_split = np.linspace(0, height, n + 1, dtype=np.int)
        width_split = np.linspace(0, width, m + 1, dtype=np.int)

        height_matrix, width_matrix = np.meshgrid(height_split, width_split, indexing="ij")

        index_height_matrix = height_matrix[:-1, :-1]
        index_width_matrix = width_matrix[:-1, :-1]

        shifted_index_height_matrix = height_matrix[1:, 1:]
        shifted_index_width_matrix = width_matrix[1:, 1:]

        height_tile_sizes = shifted_index_height_matrix - index_height_matrix
        width_tile_sizes = shifted_index_width_matrix - index_width_matrix

        tiles_sizes = np.stack((height_tile_sizes, width_tile_sizes), axis=2)

        index_matrix = np.indices((n, m))
        new_index_matrix = np.stack(index_matrix, axis=2)

        for bbox_size in np.unique(tiles_sizes.reshape(-1, 2), axis=0):
            eq_mat = np.all(tiles_sizes == bbox_size, axis=2)
            new_index_matrix[eq_mat] = random_state.permutation(new_index_matrix[eq_mat])

        new_index_matrix = np.split(new_index_matrix, 2, axis=2)

        old_x = index_height_matrix[new_index_matrix[0], new_index_matrix[1]].reshape(-1)
        old_y = index_width_matrix[new_index_matrix[0], new_index_matrix[1]].reshape(-1)

        shift_x = height_tile_sizes.reshape(-1)
        shift_y = width_tile_sizes.reshape(-1)

        curr_x = index_height_matrix.reshape(-1)
        curr_y = index_width_matrix.reshape(-1)

        tiles = np.stack([curr_x, curr_y, old_x, old_y, shift_x, shift_y], axis=1)

        return {"tiles": tiles}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("grid",)


class Normalize(ImageOnlyTransform):
    """Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`

    Args:
        mean (float, list of float): mean values
        std  (float, list of float): std values
        max_pixel_value (float): maximum possible pixel value

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
        always_apply=False,
        p=1.0,
    ):
        super(Normalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, image, **params):
        return F.normalize(image, self.mean, self.std, self.max_pixel_value)

    def get_transform_init_args_names(self):
        return ("mean", "std", "max_pixel_value")


class Cutout(ImageOnlyTransform):
    """CoarseDropout of the square regions in the image.

    Args:
        num_holes (int): number of regions to zero out
        max_h_size (int): maximum height of the hole
        max_w_size (int): maximum width of the hole
        fill_value (int, float, list of int, list of float): value for dropped pixels.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """

    def __init__(
        self,
        num_holes=8,
        max_h_size=8,
        max_w_size=8,
        fill_value=0,
        always_apply=False,
        p=0.5,
    ):
        super(Cutout, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
        warnings.warn(
            "This class has been deprecated. Please use CoarseDropout",
            FutureWarning,
        )

    def apply(self, image, fill_value=0, holes=(), **params):
        return F.cutout(image, holes, fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")


class CoarseDropout(DualTransform):
    """CoarseDropout of the rectangular regions in the image.

    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int, float): Maximum height of the hole.
        If float, it is calculated as a fraction of the image height.
        max_width (int, float): Maximum width of the hole.
        If float, it is calculated as a fraction of the image width.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int, float): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
            If float, it is calculated as a fraction of the image height.
        min_width (int, float): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
            If float, it is calculated as a fraction of the image width.

        fill_value (int, float, list of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, list of int, list of float): fill value for dropped pixels
            in mask. If `None` - mask is not affected. Default: `None`.

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """

    def __init__(
        self,
        max_holes=8,
        max_height=8,
        max_width=8,
        min_holes=None,
        min_height=None,
        min_width=None,
        fill_value=0,
        mask_fill_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(CoarseDropout, self).__init__(always_apply, p)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_holes, max_holes]))

        self.check_range(self.max_height)
        self.check_range(self.min_height)
        self.check_range(self.max_width)
        self.check_range(self.min_width)

        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height and max_height. Got: {}".format([min_height, max_height])
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError("Invalid combination of min_width and max_width. Got: {}".format([min_width, max_width]))

    def check_range(self, dimension):
        if isinstance(dimension, float) and not 0 <= dimension < 1.0:
            raise ValueError(
                "Invalid value {}. If using floats, the value should be in the range [0.0, 1.0)".format(dimension)
            )

    def apply(self, image, fill_value=0, holes=(), **params):
        return F.cutout(image, holes, fill_value)

    def apply_to_mask(self, image, mask_fill_value=0, holes=(), **params):
        if mask_fill_value is None:
            return image
        return F.cutout(image, holes, mask_fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(random.randint(self.min_holes, self.max_holes)):
            if all(
                [
                    isinstance(self.min_height, int),
                    isinstance(self.min_width, int),
                    isinstance(self.max_height, int),
                    isinstance(self.max_width, int),
                ]
            ):
                hole_height = random.randint(self.min_height, self.max_height)
                hole_width = random.randint(self.min_width, self.max_width)
            elif all(
                [
                    isinstance(self.min_height, float),
                    isinstance(self.min_width, float),
                    isinstance(self.max_height, float),
                    isinstance(self.max_width, float),
                ]
            ):
                hole_height = int(height * random.uniform(self.min_height, self.max_height))
                hole_width = int(width * random.uniform(self.min_width, self.max_width))
            else:
                raise ValueError(
                    "Min width, max width, \
                    min height and max height \
                    should all either be ints or floats. \
                    Got: {} respectively".format(
                        [
                            type(self.min_width),
                            type(self.max_width),
                            type(self.min_height),
                            type(self.max_height),
                        ]
                    )
                )

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "max_holes",
            "max_height",
            "max_width",
            "min_holes",
            "min_height",
            "min_width",
            "fill_value",
            "mask_fill_value",
        )


class ImageCompression(ImageOnlyTransform):
    """Decrease Jpeg, WebP compression of an image.

    Args:
        quality_lower (float): lower bound on the image quality.
                               Should be in [0, 100] range for jpeg and [1, 100] for webp.
        quality_upper (float): upper bound on the image quality.
                               Should be in [0, 100] range for jpeg and [1, 100] for webp.
        compression_type (ImageCompressionType): should be ImageCompressionType.JPEG or ImageCompressionType.WEBP.
            Default: ImageCompressionType.JPEG

    Targets:
        image

    Image types:
        uint8, float32
    """

    class ImageCompressionType(IntEnum):
        JPEG = 0
        WEBP = 1

    def __init__(
        self,
        quality_lower=99,
        quality_upper=100,
        compression_type=ImageCompressionType.JPEG,
        always_apply=False,
        p=0.5,
    ):
        super(ImageCompression, self).__init__(always_apply, p)

        self.compression_type = ImageCompression.ImageCompressionType(compression_type)
        low_thresh_quality_assert = 0

        if self.compression_type == ImageCompression.ImageCompressionType.WEBP:
            low_thresh_quality_assert = 1

        if not low_thresh_quality_assert <= quality_lower <= 100:
            raise ValueError("Invalid quality_lower. Got: {}".format(quality_lower))
        if not low_thresh_quality_assert <= quality_upper <= 100:
            raise ValueError("Invalid quality_upper. Got: {}".format(quality_upper))

        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def apply(self, image, quality=100, image_type=".jpg", **params):
        if not image.ndim == 2 and image.shape[-1] not in (1, 3, 4):
            raise TypeError("ImageCompression transformation expects 1, 3 or 4 channel images.")
        return F.image_compression(image, quality, image_type)

    def get_params(self):
        image_type = ".jpg"

        if self.compression_type == ImageCompression.ImageCompressionType.WEBP:
            image_type = ".webp"

        return {
            "quality": random.randint(self.quality_lower, self.quality_upper),
            "image_type": image_type,
        }

    def get_transform_init_args(self):
        return {
            "quality_lower": self.quality_lower,
            "quality_upper": self.quality_upper,
            "compression_type": self.compression_type.value,
        }


class JpegCompression(ImageCompression):
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
        super(JpegCompression, self).__init__(
            quality_lower=quality_lower,
            quality_upper=quality_upper,
            compression_type=ImageCompression.ImageCompressionType.JPEG,
            always_apply=always_apply,
            p=p,
        )
        warnings.warn(
            "This class has been deprecated. Please use ImageCompression",
            FutureWarning,
        )

    def get_transform_init_args(self):
        return {
            "quality_lower": self.quality_lower,
            "quality_upper": self.quality_upper,
        }


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

    def __init__(
        self,
        snow_point_lower=0.1,
        snow_point_upper=0.3,
        brightness_coeff=2.5,
        always_apply=False,
        p=0.5,
    ):
        super(RandomSnow, self).__init__(always_apply, p)

        if not 0 <= snow_point_lower <= snow_point_upper <= 1:
            raise ValueError(
                "Invalid combination of snow_point_lower and snow_point_upper. Got: {}".format(
                    (snow_point_lower, snow_point_upper)
                )
            )
        if brightness_coeff < 0:
            raise ValueError("brightness_coeff must be greater than 0. Got: {}".format(brightness_coeff))

        self.snow_point_lower = snow_point_lower
        self.snow_point_upper = snow_point_upper
        self.brightness_coeff = brightness_coeff

    def apply(self, image, snow_point=0.1, **params):
        return F.add_snow(image, snow_point, self.brightness_coeff)

    def get_params(self):
        return {"snow_point": random.uniform(self.snow_point_lower, self.snow_point_upper)}

    def get_transform_init_args_names(self):
        return ("snow_point_lower", "snow_point_upper", "brightness_coeff")


class RandomRain(ImageOnlyTransform):
    """Adds rain effects.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        slant_lower: should be in range [-20, 20].
        slant_upper: should be in range [-20, 20].
        drop_length: should be in range [0, 100].
        drop_width: should be in range [1, 5].
        drop_color (list of (r, g, b)): rain lines color.
        blur_value (int): rainy view are blurry
        brightness_coefficient (float): rainy days are usually shady. Should be in range [0, 1].
        rain_type: One of [None, "drizzle", "heavy", "torrestial"]

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        slant_lower=-10,
        slant_upper=10,
        drop_length=20,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=7,
        brightness_coefficient=0.7,
        rain_type=None,
        always_apply=False,
        p=0.5,
    ):
        super(RandomRain, self).__init__(always_apply, p)

        if rain_type not in ["drizzle", "heavy", "torrential", None]:
            raise ValueError(
                "raint_type must be one of ({}). Got: {}".format(["drizzle", "heavy", "torrential", None], rain_type)
            )
        if not -20 <= slant_lower <= slant_upper <= 20:
            raise ValueError(
                "Invalid combination of slant_lower and slant_upper. Got: {}".format((slant_lower, slant_upper))
            )
        if not 1 <= drop_width <= 5:
            raise ValueError("drop_width must be in range [1, 5]. Got: {}".format(drop_width))
        if not 0 <= drop_length <= 100:
            raise ValueError("drop_length must be in range [0, 100]. Got: {}".format(drop_length))
        if not 0 <= brightness_coefficient <= 1:
            raise ValueError("brightness_coefficient must be in range [0, 1]. Got: {}".format(brightness_coefficient))

        self.slant_lower = slant_lower
        self.slant_upper = slant_upper

        self.drop_length = drop_length
        self.drop_width = drop_width
        self.drop_color = drop_color
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient
        self.rain_type = rain_type

    def apply(self, image, slant=10, drop_length=20, rain_drops=(), **params):
        return F.add_rain(
            image,
            slant,
            drop_length,
            self.drop_width,
            self.drop_color,
            self.blur_value,
            self.brightness_coefficient,
            rain_drops,
        )

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        slant = int(random.uniform(self.slant_lower, self.slant_upper))

        height, width = img.shape[:2]
        area = height * width

        if self.rain_type == "drizzle":
            num_drops = area // 770
            drop_length = 10
        elif self.rain_type == "heavy":
            num_drops = width * height // 600
            drop_length = 30
        elif self.rain_type == "torrential":
            num_drops = area // 500
            drop_length = 60
        else:
            drop_length = self.drop_length
            num_drops = area // 600

        rain_drops = []

        for _i in range(num_drops):  # If You want heavy rain, try increasing this
            if slant < 0:
                x = random.randint(slant, width)
            else:
                x = random.randint(0, width - slant)

            y = random.randint(0, height - drop_length)

            rain_drops.append((x, y))

        return {"drop_length": drop_length, "rain_drops": rain_drops}

    def get_transform_init_args_names(self):
        return (
            "slant_lower",
            "slant_upper",
            "drop_length",
            "drop_width",
            "drop_color",
            "blur_value",
            "brightness_coefficient",
            "rain_type",
        )


class RandomFog(ImageOnlyTransform):
    """Simulates fog for the image

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        fog_coef_lower (float): lower limit for fog intensity coefficient. Should be in [0, 1] range.
        fog_coef_upper (float): upper limit for fog intensity coefficient. Should be in [0, 1] range.
        alpha_coef (float): transparency of the fog circles. Should be in [0, 1] range.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        fog_coef_lower=0.3,
        fog_coef_upper=1,
        alpha_coef=0.08,
        always_apply=False,
        p=0.5,
    ):
        super(RandomFog, self).__init__(always_apply, p)

        if not 0 <= fog_coef_lower <= fog_coef_upper <= 1:
            raise ValueError(
                "Invalid combination if fog_coef_lower and fog_coef_upper. Got: {}".format(
                    (fog_coef_lower, fog_coef_upper)
                )
            )
        if not 0 <= alpha_coef <= 1:
            raise ValueError("alpha_coef must be in range [0, 1]. Got: {}".format(alpha_coef))

        self.fog_coef_lower = fog_coef_lower
        self.fog_coef_upper = fog_coef_upper
        self.alpha_coef = alpha_coef

    def apply(self, image, fog_coef=0.1, haze_list=(), **params):
        return F.add_fog(image, fog_coef, self.alpha_coef, haze_list)

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        fog_coef = random.uniform(self.fog_coef_lower, self.fog_coef_upper)

        height, width = imshape = img.shape[:2]

        hw = max(1, int(width // 3 * fog_coef))

        haze_list = []
        midx = width // 2 - 2 * hw
        midy = height // 2 - hw
        index = 1

        while midx > -hw or midy > -hw:
            for _i in range(hw // 10 * index):
                x = random.randint(midx, width - midx - hw)
                y = random.randint(midy, height - midy - hw)
                haze_list.append((x, y))

            midx -= 3 * hw * width // sum(imshape)
            midy -= 3 * hw * height // sum(imshape)
            index += 1

        return {"haze_list": haze_list, "fog_coef": fog_coef}

    def get_transform_init_args_names(self):
        return ("fog_coef_lower", "fog_coef_upper", "alpha_coef")


class RandomSunFlare(ImageOnlyTransform):
    """Simulates Sun Flare for the image

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        flare_roi (float, float, float, float): region of the image where flare will
            appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].
        angle_lower (float): should be in range [0, `angle_upper`].
        angle_upper (float): should be in range [`angle_lower`, 1].
        num_flare_circles_lower (int): lower limit for the number of flare circles.
            Should be in range [0, `num_flare_circles_upper`].
        num_flare_circles_upper (int): upper limit for the number of flare circles.
            Should be in range [`num_flare_circles_lower`, inf].
        src_radius (int):
        src_color ((int, int, int)): color of the flare

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        flare_roi=(0, 0, 1, 0.5),
        angle_lower=0,
        angle_upper=1,
        num_flare_circles_lower=6,
        num_flare_circles_upper=10,
        src_radius=400,
        src_color=(255, 255, 255),
        always_apply=False,
        p=0.5,
    ):
        super(RandomSunFlare, self).__init__(always_apply, p)

        (
            flare_center_lower_x,
            flare_center_lower_y,
            flare_center_upper_x,
            flare_center_upper_y,
        ) = flare_roi

        if (
            not 0 <= flare_center_lower_x < flare_center_upper_x <= 1
            or not 0 <= flare_center_lower_y < flare_center_upper_y <= 1
        ):
            raise ValueError("Invalid flare_roi. Got: {}".format(flare_roi))
        if not 0 <= angle_lower < angle_upper <= 1:
            raise ValueError(
                "Invalid combination of angle_lower nad angle_upper. Got: {}".format((angle_lower, angle_upper))
            )
        if not 0 <= num_flare_circles_lower < num_flare_circles_upper:
            raise ValueError(
                "Invalid combination of num_flare_circles_lower nad num_flare_circles_upper. Got: {}".format(
                    (num_flare_circles_lower, num_flare_circles_upper)
                )
            )

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

    def apply(self, image, flare_center_x=0.5, flare_center_y=0.5, circles=(), **params):
        return F.add_sun_flare(
            image,
            flare_center_x,
            flare_center_y,
            self.src_radius,
            self.src_color,
            circles,
        )

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
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

        for _i in range(num_circles):
            alpha = random.uniform(0.05, 0.2)
            r = random.randint(0, len(x) - 1)
            rad = random.randint(1, max(height // 100 - 2, 2))

            r_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])
            g_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])
            b_color = random.randint(max(self.src_color[0] - 50, 0), self.src_color[0])

            circles += [
                (
                    alpha,
                    (int(x[r]), int(y[r])),
                    pow(rad, 3),
                    (r_color, g_color, b_color),
                )
            ]

        return {
            "circles": circles,
            "flare_center_x": flare_center_x,
            "flare_center_y": flare_center_y,
        }

    def get_transform_init_args(self):
        return {
            "flare_roi": (
                self.flare_center_lower_x,
                self.flare_center_lower_y,
                self.flare_center_upper_x,
                self.flare_center_upper_y,
            ),
            "angle_lower": self.angle_lower,
            "angle_upper": self.angle_upper,
            "num_flare_circles_lower": self.num_flare_circles_lower,
            "num_flare_circles_upper": self.num_flare_circles_upper,
            "src_radius": self.src_radius,
            "src_color": self.src_color,
        }


class RandomShadow(ImageOnlyTransform):
    """Simulates shadows for the image

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        shadow_roi (float, float, float, float): region of the image where shadows
            will appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].
        num_shadows_lower (int): Lower limit for the possible number of shadows.
            Should be in range [0, `num_shadows_upper`].
        num_shadows_upper (int): Lower limit for the possible number of shadows.
            Should be in range [`num_shadows_lower`, inf].
        shadow_dimension (int): number of edges in the shadow polygons

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        shadow_roi=(0, 0.5, 1, 1),
        num_shadows_lower=1,
        num_shadows_upper=2,
        shadow_dimension=5,
        always_apply=False,
        p=0.5,
    ):
        super(RandomShadow, self).__init__(always_apply, p)

        (shadow_lower_x, shadow_lower_y, shadow_upper_x, shadow_upper_y) = shadow_roi

        if not 0 <= shadow_lower_x <= shadow_upper_x <= 1 or not 0 <= shadow_lower_y <= shadow_upper_y <= 1:
            raise ValueError("Invalid shadow_roi. Got: {}".format(shadow_roi))
        if not 0 <= num_shadows_lower <= num_shadows_upper:
            raise ValueError(
                "Invalid combination of num_shadows_lower nad num_shadows_upper. Got: {}".format(
                    (num_shadows_lower, num_shadows_upper)
                )
            )

        self.shadow_roi = shadow_roi

        self.num_shadows_lower = num_shadows_lower
        self.num_shadows_upper = num_shadows_upper

        self.shadow_dimension = shadow_dimension

    def apply(self, image, vertices_list=(), **params):
        return F.add_shadow(image, vertices_list)

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        num_shadows = random.randint(self.num_shadows_lower, self.num_shadows_upper)

        x_min, y_min, x_max, y_max = self.shadow_roi

        x_min = int(x_min * width)
        x_max = int(x_max * width)
        y_min = int(y_min * height)
        y_max = int(y_max * height)

        vertices_list = []

        for _index in range(num_shadows):
            vertex = []
            for _dimension in range(self.shadow_dimension):
                vertex.append((random.randint(x_min, x_max), random.randint(y_min, y_max)))

            vertices = np.array([vertex], dtype=np.int32)
            vertices_list.append(vertices)

        return {"vertices_list": vertices_list}

    def get_transform_init_args_names(self):
        return (
            "shadow_roi",
            "num_shadows_lower",
            "num_shadows_upper",
            "shadow_dimension",
        )


class RandomToneCurve(ImageOnlyTransform):
    """Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.

    Args:
        scale (float): standard deviation of the normal distribution.
            Used to sample random distances to move two control points that modify the image's curve.
            Values should be in range [0, 1]. Default: 0.1


    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self,
        scale=0.1,
        always_apply=False,
        p=0.5,
    ):
        super(RandomToneCurve, self).__init__(always_apply, p)
        self.scale = scale

    def apply(self, image, low_y, high_y, **params):
        return F.move_tone_curve(image, low_y, high_y)

    def get_params(self):
        return {
            "low_y": np.clip(np.random.normal(loc=0.25, scale=self.scale), 0, 1),
            "high_y": np.clip(np.random.normal(loc=0.75, scale=self.scale), 0, 1),
        }

    def get_transform_init_args_names(self):
        return ("scale",)


class HueSaturationValue(ImageOnlyTransform):
    """Randomly change hue, saturation and value of the input image.

    Args:
        hue_shift_limit ((int, int) or int): range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
        sat_shift_limit ((int, int) or int): range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
        val_shift_limit ((int, int) or int): range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        always_apply=False,
        p=0.5,
    ):
        super(HueSaturationValue, self).__init__(always_apply, p)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0, **params):
        if not F.is_rgb_image(image) and not F.is_grayscale_image(image):
            raise TypeError("HueSaturationValue transformation expects 1-channel or 3-channel images.")
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return {
            "hue_shift": random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
            "sat_shift": random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
            "val_shift": random.uniform(self.val_shift_limit[0], self.val_shift_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("hue_shift_limit", "sat_shift_limit", "val_shift_limit")


class Solarize(ImageOnlyTransform):
    """Invert all pixel values above a threshold.

    Args:
        threshold ((int, int) or int, or (float, float) or float): range for solarizing threshold.
        If threshold is a single value, the range will be [threshold, threshold]. Default: 128.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        any
    """

    def __init__(self, threshold=128, always_apply=False, p=0.5):
        super(Solarize, self).__init__(always_apply, p)

        if isinstance(threshold, (int, float)):
            self.threshold = to_tuple(threshold, low=threshold)
        else:
            self.threshold = to_tuple(threshold, low=0)

    def apply(self, image, threshold=0, **params):
        return F.solarize(image, threshold)

    def get_params(self):
        return {"threshold": random.uniform(self.threshold[0], self.threshold[1])}

    def get_transform_init_args_names(self):
        return ("threshold",)


class Posterize(ImageOnlyTransform):
    """Reduce the number of bits for each color channel.

    Args:
        num_bits ((int, int) or int,
                  or list of ints [r, g, b],
                  or list of ints [[r1, r1], [g1, g2], [b1, b2]]): number of high bits.
            If num_bits is a single value, the range will be [num_bits, num_bits].
            Must be in range [0, 8]. Default: 4.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
    image

    Image types:
        uint8
    """

    def __init__(self, num_bits=4, always_apply=False, p=0.5):
        super(Posterize, self).__init__(always_apply, p)

        if isinstance(num_bits, (list, tuple)):
            if len(num_bits) == 3:
                self.num_bits = [to_tuple(i, 0) for i in num_bits]
            else:
                self.num_bits = to_tuple(num_bits, 0)
        else:
            self.num_bits = to_tuple(num_bits, num_bits)

    def apply(self, image, num_bits=1, **params):
        return F.posterize(image, num_bits)

    def get_params(self):
        if len(self.num_bits) == 3:
            return {"num_bits": [random.randint(i[0], i[1]) for i in self.num_bits]}
        return {"num_bits": random.randint(self.num_bits[0], self.num_bits[1])}

    def get_transform_init_args_names(self):
        return ("num_bits",)


class Equalize(ImageOnlyTransform):
    """Equalize the image histogram.

    Args:
        mode (str): {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
        by_channels (bool): If True, use equalization by channels separately,
            else convert image to YCbCr representation and use equalization by `Y` channel.
        mask (np.ndarray, callable): If given, only the pixels selected by
            the mask are included in the analysis. Maybe 1 channel or 3 channel array or callable.
            Function signature must include `image` argument.
        mask_params (list of str): Params for mask function.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self,
        mode="cv",
        by_channels=True,
        mask=None,
        mask_params=(),
        always_apply=False,
        p=0.5,
    ):
        modes = ["cv", "pil"]
        if mode not in modes:
            raise ValueError("Unsupported equalization mode. Supports: {}. " "Got: {}".format(modes, mode))

        super(Equalize, self).__init__(always_apply, p)
        self.mode = mode
        self.by_channels = by_channels
        self.mask = mask
        self.mask_params = mask_params

    def apply(self, image, mask=None, **params):
        return F.equalize(image, mode=self.mode, by_channels=self.by_channels, mask=mask)

    def get_params_dependent_on_targets(self, params):
        if not callable(self.mask):
            return {"mask": self.mask}

        return {"mask": self.mask(**params)}

    @property
    def targets_as_params(self):
        return ["image"] + list(self.mask_params)

    def get_transform_init_args_names(self):
        return ("mode", "by_channels")


class RGBShift(ImageOnlyTransform):
    """Randomly shift values for each channel of the input RGB image.

    Args:
        r_shift_limit ((int, int) or int): range for changing values for the red channel. If r_shift_limit is a single
            int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).
        g_shift_limit ((int, int) or int): range for changing values for the green channel. If g_shift_limit is a
            single int, the range  will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).
        b_shift_limit ((int, int) or int): range for changing values for the blue channel. If b_shift_limit is a single
            int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        r_shift_limit=20,
        g_shift_limit=20,
        b_shift_limit=20,
        always_apply=False,
        p=0.5,
    ):
        super(RGBShift, self).__init__(always_apply, p)
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def apply(self, image, r_shift=0, g_shift=0, b_shift=0, **params):
        if not F.is_rgb_image(image):
            raise TypeError("RGBShift transformation expects 3-channel images.")
        return F.shift_rgb(image, r_shift, g_shift, b_shift)

    def get_params(self):
        return {
            "r_shift": random.uniform(self.r_shift_limit[0], self.r_shift_limit[1]),
            "g_shift": random.uniform(self.g_shift_limit[0], self.g_shift_limit[1]),
            "b_shift": random.uniform(self.b_shift_limit[0], self.b_shift_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("r_shift_limit", "g_shift_limit", "b_shift_limit")


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        brightness_limit=0.2,
        contrast_limit=0.2,
        brightness_by_max=True,
        always_apply=False,
        p=0.5,
    ):
        super(RandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def apply(self, img, alpha=1.0, beta=0.0, **params):
        return F.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)

    def get_params(self):
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit", "brightness_by_max")


class RandomBrightness(RandomBrightnessContrast):
    """Randomly change brightness of the input image.

    Args:
        limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, limit=0.2, always_apply=False, p=0.5):
        super(RandomBrightness, self).__init__(
            brightness_limit=limit, contrast_limit=0, always_apply=always_apply, p=p
        )
        warnings.warn(
            "This class has been deprecated. Please use RandomBrightnessContrast",
            FutureWarning,
        )

    def get_transform_init_args(self):
        return {"limit": self.brightness_limit}


class RandomContrast(RandomBrightnessContrast):
    """Randomly change contrast of the input image.

    Args:
        limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, limit=0.2, always_apply=False, p=0.5):
        super(RandomContrast, self).__init__(brightness_limit=0, contrast_limit=limit, always_apply=always_apply, p=p)
        warnings.warn(
            "This class has been deprecated. Please use RandomBrightnessContrast",
            FutureWarning,
        )

    def get_transform_init_args(self):
        return {"limit": self.contrast_limit}


class Blur(ImageOnlyTransform):
    """Blur the input image using a random-sized kernel.

    Args:
        blur_limit (int, (int, int)): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=7, always_apply=False, p=0.5):
        super(Blur, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)

    def apply(self, image, ksize=3, **params):
        return F.blur(image, ksize)

    def get_params(self):
        return {"ksize": int(random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2)))}

    def get_transform_init_args_names(self):
        return ("blur_limit",)


class MotionBlur(Blur):
    """Apply motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def apply(self, img, kernel=None, **params):
        return F.convolve(img, kernel=kernel)

    def get_params(self):
        ksize = random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        if ksize <= 2:
            raise ValueError("ksize must be > 2. Got: {}".format(ksize))
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        xs, xe = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        if xs == xe:
            ys, ye = random.sample(range(ksize), 2)
        else:
            ys, ye = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        return {"kernel": kernel}


class MedianBlur(Blur):
    """Blur the input image using a median filter with a random aperture linear size.

    Args:
        blur_limit (int): maximum aperture linear size for blurring the input image.
            Must be odd and in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=7, always_apply=False, p=0.5):
        super(MedianBlur, self).__init__(blur_limit, always_apply, p)

        if self.blur_limit[0] % 2 != 1 or self.blur_limit[1] % 2 != 1:
            raise ValueError("MedianBlur supports only odd blur limits.")

    def apply(self, image, ksize=3, **params):
        return F.median_blur(image, ksize)


class GaussianBlur(ImageOnlyTransform):
    """Blur the input image using a Gaussian filter with a random kernel size.

    Args:
        blur_limit (int, (int, int)): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be greater in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5):
        super(GaussianBlur, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 0)
        self.sigma_limit = to_tuple(sigma_limit if sigma_limit is not None else 0, 0)

        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            warnings.warn(
                "blur_limit and sigma_limit minimum value can not be both equal to 0. "
                "blur_limit minimum value changed to 3."
            )

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("GaussianBlur supports only odd blur limits.")

    def apply(self, image, ksize=3, sigma=0, **params):
        return F.gaussian_blur(image, ksize, sigma=sigma)

    def get_params(self):
        ksize = np.random.randint(self.blur_limit[0], self.blur_limit[1] + 1)
        if ksize != 0 and ksize % 2 != 1:
            ksize = (ksize + 1) % (self.blur_limit[1] + 1)

        return {"ksize": ksize, "sigma": random.uniform(*self.sigma_limit)}

    def get_transform_init_args_names(self):
        return ("blur_limit", "sigma_limit")


class GaussNoise(ImageOnlyTransform):
    """Apply gaussian noise to the input image.

    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        per_channel (bool): if set to True, noise will be sampled for each channel independently.
            Otherwise, the noise will be sampled once for all channels. Default: True
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5):
        super(GaussNoise, self).__init__(always_apply, p)
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
            )

        self.mean = mean
        self.per_channel = per_channel

    def apply(self, img, gauss=None, **params):
        return F.gauss_noise(img, gauss=gauss)

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

        if self.per_channel:
            gauss = random_state.normal(self.mean, sigma, image.shape)
        else:
            gauss = random_state.normal(self.mean, sigma, image.shape[:2])
            if len(image.shape) == 3:
                gauss = np.expand_dims(gauss, -1)

        return {"gauss": gauss}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("var_limit", "per_channel", "mean")


class ISONoise(ImageOnlyTransform):
    """
    Apply camera sensor noise.

    Args:
        color_shift (float, float): variance range for color hue change.
            Measured as a fraction of 360 degree Hue angle in HLS colorspace.
        intensity ((float, float): Multiplicative factor that control strength
            of color and luminace noise.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(self, color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5):
        super(ISONoise, self).__init__(always_apply, p)
        self.intensity = intensity
        self.color_shift = color_shift

    def apply(self, img, color_shift=0.05, intensity=1.0, random_state=None, **params):
        return F.iso_noise(img, color_shift, intensity, np.random.RandomState(random_state))

    def get_params(self):
        return {
            "color_shift": random.uniform(self.color_shift[0], self.color_shift[1]),
            "intensity": random.uniform(self.intensity[0], self.intensity[1]),
            "random_state": random.randint(0, 65536),
        }

    def get_transform_init_args_names(self):
        return ("intensity", "color_shift")


class CLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
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
        self.tile_grid_size = tuple(tile_grid_size)

    def apply(self, img, clip_limit=2, **params):
        if not F.is_rgb_image(img) and not F.is_grayscale_image(img):
            raise TypeError("CLAHE transformation expects 1-channel or 3-channel images.")

        return F.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self):
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self):
        return ("clip_limit", "tile_grid_size")


class ChannelDropout(ImageOnlyTransform):
    """Randomly Drop Channels in the input Image.

    Args:
        channel_drop_range (int, int): range from which we choose the number of channels to drop.
        fill_value (int, float): pixel value for the dropped channel.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, unit32, float32
    """

    def __init__(self, channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5):
        super(ChannelDropout, self).__init__(always_apply, p)

        self.channel_drop_range = channel_drop_range

        self.min_channels = channel_drop_range[0]
        self.max_channels = channel_drop_range[1]

        if not 1 <= self.min_channels <= self.max_channels:
            raise ValueError("Invalid channel_drop_range. Got: {}".format(channel_drop_range))

        self.fill_value = fill_value

    def apply(self, img, channels_to_drop=(0,), **params):
        return F.channel_dropout(img, channels_to_drop, self.fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]

        num_channels = img.shape[-1]

        if len(img.shape) == 2 or num_channels == 1:
            raise NotImplementedError("Images has one channel. ChannelDropout is not defined.")

        if self.max_channels >= num_channels:
            raise ValueError("Can not drop all channels in ChannelDropout.")

        num_drop_channels = random.randint(self.min_channels, self.max_channels)

        channels_to_drop = random.sample(range(num_channels), k=num_drop_channels)

        return {"channels_to_drop": channels_to_drop}

    def get_transform_init_args_names(self):
        return ("channel_drop_range", "fill_value")

    @property
    def targets_as_params(self):
        return ["image"]


class ChannelShuffle(ImageOnlyTransform):
    """Randomly rearrange channels of the input RGB image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    @property
    def targets_as_params(self):
        return ["image"]

    def apply(self, img, channels_shuffled=(0, 1, 2), **params):
        return F.channel_shuffle(img, channels_shuffled)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        ch_arr = list(range(img.shape[2]))
        random.shuffle(ch_arr)
        return {"channels_shuffled": ch_arr}

    def get_transform_init_args_names(self):
        return ()


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

    def get_transform_init_args_names(self):
        return ()


class RandomGamma(ImageOnlyTransform):
    """
    Args:
        gamma_limit (float or (float, float)): If gamma_limit is a single float value,
            the range will be (-gamma_limit, gamma_limit). Default: (80, 120).
        eps: Deprecated.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5):
        super(RandomGamma, self).__init__(always_apply, p)
        self.gamma_limit = to_tuple(gamma_limit)
        self.eps = eps

    def apply(self, img, gamma=1, **params):
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self):
        return {"gamma": random.randint(self.gamma_limit[0], self.gamma_limit[1]) / 100.0}

    def get_transform_init_args_names(self):
        return ("gamma_limit", "eps")


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
        if F.is_grayscale_image(img):
            warnings.warn("The image is already gray.")
            return img
        if not F.is_rgb_image(img):
            raise TypeError("ToGray transformation expects 3-channel images.")

        return F.to_gray(img)

    def get_transform_init_args_names(self):
        return ()


class ToSepia(ImageOnlyTransform):
    """Applies sepia filter to the input RGB image

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=0.5):
        super(ToSepia, self).__init__(always_apply, p)
        self.sepia_transformation_matrix = np.matrix(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
        )

    def apply(self, image, **params):
        if not F.is_rgb_image(image):
            raise TypeError("ToSepia transformation expects 3-channel images.")
        return F.linear_transformation_rgb(image, self.sepia_transformation_matrix)

    def get_transform_init_args_names(self):
        return ()


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

    def get_transform_init_args_names(self):
        return ("max_value",)


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

    def __init__(self, dtype="uint16", max_value=None, always_apply=False, p=1.0):
        super(FromFloat, self).__init__(always_apply, p)
        self.dtype = np.dtype(dtype)
        self.max_value = max_value

    def apply(self, img, **params):
        return F.from_float(img, self.dtype, self.max_value)

    def get_transform_init_args(self):
        return {"dtype": self.dtype.name, "max_value": self.max_value}


class Downscale(ImageOnlyTransform):
    """Decreases image quality by downscaling and upscaling back.

    Args:
        scale_min (float): lower bound on the image scale. Should be < 1.
        scale_max (float):  lower bound on the image scale. Should be .
        interpolation: cv2 interpolation method. cv2.INTER_NEAREST by default

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale_min=0.25,
        scale_max=0.25,
        interpolation=cv2.INTER_NEAREST,
        always_apply=False,
        p=0.5,
    ):
        super(Downscale, self).__init__(always_apply, p)
        if scale_min > scale_max:
            raise ValueError("Expected scale_min be less or equal scale_max, got {} {}".format(scale_min, scale_max))
        if scale_max >= 1:
            raise ValueError("Expected scale_max to be less than 1, got {}".format(scale_max))
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.interpolation = interpolation

    def apply(self, image, scale, interpolation, **params):
        return F.downscale(image, scale=scale, interpolation=interpolation)

    def get_params(self):
        return {
            "scale": np.random.uniform(self.scale_min, self.scale_max),
            "interpolation": self.interpolation,
        }

    def get_transform_init_args_names(self):
        return "scale_min", "scale_max", "interpolation"


class Lambda(NoOp):
    """A flexible transformation class for using user-defined transformation functions per targets.
    Function signature must include **kwargs to accept optinal arguments like interpolation method, image size, etc:

    Args:
        image (callable): Image transformation function.
        mask (callable): Mask transformation function.
        keypoint (callable): Keypoint transformation function.
        bbox (callable): BBox transformation function.
        always_apply (bool): Indicates whether this transformation should be always applied.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        Any
    """

    def __init__(
        self,
        image=None,
        mask=None,
        keypoint=None,
        bbox=None,
        name=None,
        always_apply=False,
        p=1.0,
    ):
        super(Lambda, self).__init__(always_apply, p)

        self.name = name
        self.custom_apply_fns = {target_name: F.noop for target_name in ("image", "mask", "keypoint", "bbox")}
        for target_name, custom_apply_fn in {
            "image": image,
            "mask": mask,
            "keypoint": keypoint,
            "bbox": bbox,
        }.items():
            if custom_apply_fn is not None:
                if isinstance(custom_apply_fn, LambdaType) and custom_apply_fn.__name__ == "<lambda>":
                    warnings.warn(
                        "Using lambda is incompatible with multiprocessing. "
                        "Consider using regular functions or partial()."
                    )

                self.custom_apply_fns[target_name] = custom_apply_fn

    def apply(self, img, **params):
        fn = self.custom_apply_fns["image"]
        return fn(img, **params)

    def apply_to_mask(self, mask, **params):
        fn = self.custom_apply_fns["mask"]
        return fn(mask, **params)

    def apply_to_bbox(self, bbox, **params):
        fn = self.custom_apply_fns["bbox"]
        return fn(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        fn = self.custom_apply_fns["keypoint"]
        return fn(keypoint, **params)

    @classmethod
    def is_serializable(cls):
        return False

    def _to_dict(self):
        if self.name is None:
            raise ValueError(
                "To make a Lambda transform serializable you should provide the `name` argument, "
                "e.g. `Lambda(name='my_transform', image=<some func>, ...)`."
            )
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}

    def __repr__(self):
        state = {"name": self.name}
        state.update(self.custom_apply_fns.items())
        state.update(self.get_base_init_args())
        return "{name}({args})".format(name=self.__class__.__name__, args=format_args(state))


class MultiplicativeNoise(ImageOnlyTransform):
    """Multiply image to random number or array of numbers.

    Args:
        multiplier (float or tuple of floats): If single float image will be multiplied to this number.
            If tuple of float multiplier will be in range `[multiplier[0], multiplier[1])`. Default: (0.9, 1.1).
        per_channel (bool): If `False`, same values for all channels will be used.
            If `True` use sample values for each channels. Default False.
        elementwise (bool): If `False` multiply multiply all pixels in an image with a random value sampled once.
            If `True` Multiply image pixels with values that are pixelwise randomly sampled. Defaule: False.

    Targets:
        image

    Image types:
        Any
    """

    def __init__(
        self,
        multiplier=(0.9, 1.1),
        per_channel=False,
        elementwise=False,
        always_apply=False,
        p=0.5,
    ):
        super(MultiplicativeNoise, self).__init__(always_apply, p)
        self.multiplier = to_tuple(multiplier, multiplier)
        self.per_channel = per_channel
        self.elementwise = elementwise

    def apply(self, img, multiplier=np.array([1]), **kwargs):
        return F.multiply(img, multiplier)

    def get_params_dependent_on_targets(self, params):
        if self.multiplier[0] == self.multiplier[1]:
            return {"multiplier": np.array([self.multiplier[0]])}

        img = params["image"]

        h, w = img.shape[:2]

        if self.per_channel:
            c = 1 if F.is_grayscale_image(img) else img.shape[-1]
        else:
            c = 1

        if self.elementwise:
            shape = [h, w, c]
        else:
            shape = [c]

        multiplier = np.random.uniform(self.multiplier[0], self.multiplier[1], shape)
        if F.is_grayscale_image(img) and img.ndim == 2:
            multiplier = np.squeeze(multiplier)

        return {"multiplier": multiplier}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "multiplier", "per_channel", "elementwise"


class FancyPCA(ImageOnlyTransform):
    """Augment RGB image using FancyPCA from Krizhevsky's paper
    "ImageNet Classification with Deep Convolutional Neural Networks"

    Args:
        alpha (float):  how much to perturb/scale the eigen vecs and vals.
            scale is samples from gaussian distribution (mu=0, sigma=alpha)

    Targets:
        image

    Image types:
        3-channel uint8 images only

    Credit:
        http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
        https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
    """

    def __init__(self, alpha=0.1, always_apply=False, p=0.5):
        super(FancyPCA, self).__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, img, alpha=0.1, **params):
        img = F.fancy_pca(img, alpha)
        return img

    def get_params(self):
        return {"alpha": random.gauss(0, self.alpha)}

    def get_transform_init_args_names(self):
        return ("alpha",)


class MaskDropout(DualTransform):
    """
    Image & mask augmentation that zero out mask and image regions corresponding
    to randomly chosen object instance from mask.

    Mask must be single-channel image, zero values treated as background.
    Image can be any number of channels.

    Inspired by https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114254
    """

    def __init__(
        self,
        max_objects=1,
        image_fill_value=0,
        mask_fill_value=0,
        always_apply=False,
        p=0.5,
    ):
        """
        Args:
            max_objects: Maximum number of labels that can be zeroed out. Can be tuple, in this case it's [min, max]
            image_fill_value: Fill value to use when filling image.
                Can be 'inpaint' to apply inpaining (works only  for 3-chahnel images)
            mask_fill_value: Fill value to use when filling mask.

        Targets:
            image, mask

        Image types:
            uint8, float32
        """
        super(MaskDropout, self).__init__(always_apply, p)
        self.max_objects = to_tuple(max_objects, 1)
        self.image_fill_value = image_fill_value
        self.mask_fill_value = mask_fill_value

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]

        label_image, num_labels = label(mask, return_num=True)

        if num_labels == 0:
            dropout_mask = None
        else:
            objects_to_drop = random.randint(self.max_objects[0], self.max_objects[1])
            objects_to_drop = min(num_labels, objects_to_drop)

            if objects_to_drop == num_labels:
                dropout_mask = mask > 0
            else:
                labels_index = random.sample(range(1, num_labels + 1), objects_to_drop)
                dropout_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.bool)
                for label_index in labels_index:
                    dropout_mask |= label_image == label_index

        params.update({"dropout_mask": dropout_mask})
        return params

    def apply(self, img, dropout_mask=None, **params):
        if dropout_mask is None:
            return img

        if self.image_fill_value == "inpaint":
            dropout_mask = dropout_mask.astype(np.uint8)
            _, _, w, h = cv2.boundingRect(dropout_mask)
            radius = min(3, max(w, h) // 2)
            img = cv2.inpaint(img, dropout_mask, radius, cv2.INPAINT_NS)
        else:
            img = img.copy()
            img[dropout_mask] = self.image_fill_value

        return img

    def apply_to_mask(self, img, dropout_mask=None, **params):
        if dropout_mask is None:
            return img

        img = img.copy()
        img[dropout_mask] = self.mask_fill_value
        return img

    def get_transform_init_args_names(self):
        return ("max_objects", "image_fill_value", "mask_fill_value")


class GlassBlur(Blur):
    """Apply glass noise to the input image.

    Args:
        sigma (float): standard deviation for Gaussian kernel.
        max_delta (int): max distance between pixels which are swapped.
        iterations (int): number of repeats.
            Should be in range [1, inf). Default: (2).
        mode (str): mode of computation: fast or exact. Default: "fast".
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1903.12261
    |  https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
    """

    def __init__(
        self,
        sigma=0.7,
        max_delta=4,
        iterations=2,
        always_apply=False,
        mode="fast",
        p=0.5,
    ):
        super(GlassBlur, self).__init__(always_apply=always_apply, p=p)
        if iterations < 1:
            raise ValueError("Iterations should be more or equal to 1, but we got {}".format(iterations))

        if mode not in ["fast", "exact"]:
            raise ValueError("Mode should be 'fast' or 'exact', but we got {}".format(iterations))

        self.sigma = sigma
        self.max_delta = max_delta
        self.iterations = iterations
        self.mode = mode

    def apply(self, img, dxy=0, **params):
        return F.glass_blur(img, self.sigma, self.max_delta, self.iterations, dxy, self.mode)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]

        # generate array containing all necessary values for transformations
        width_pixels = img.shape[0] - self.max_delta * 2
        height_pixels = img.shape[1] - self.max_delta * 2
        total_pixels = width_pixels * height_pixels
        dxy = np.random.randint(-self.max_delta, self.max_delta, size=(total_pixels, self.iterations, 2))

        return {"dxy": dxy}

    def get_transform_init_args_names(self):
        return ("sigma", "max_delta", "iterations")

    @property
    def targets_as_params(self):
        return ["image"]


class GridDropout(DualTransform):
    """GridDropout, drops out rectangular regions of an image and the corresponding mask in a grid fashion.

    Args:
        ratio (float): the ratio of the mask holes to the unit_size (same for horizontal and vertical directions).
            Must be between 0 and 1. Default: 0.5.
        unit_size_min (int): minimum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        unit_size_max (int): maximum size of the grid unit. Must be between 2 and the image shorter edge.
            If 'None', holes_number_x and holes_number_y are used to setup the grid. Default: `None`.
        holes_number_x (int): the number of grid units in x direction. Must be between 1 and image width//2.
            If 'None', grid unit width is set as image_width//10. Default: `None`.
        holes_number_y (int): the number of grid units in y direction. Must be between 1 and image height//2.
            If `None`, grid unit height is set equal to the grid unit width or image height, whatever is smaller.
        shift_x (int): offsets of the grid start in x direction from (0,0) coordinate.
            Clipped between 0 and grid unit_width - hole_width. Default: 0.
        shift_y (int): offsets of the grid start in y direction from (0,0) coordinate.
            Clipped between 0 and grid unit height - hole_height. Default: 0.
        random_offset (boolean): weather to offset the grid randomly between 0 and grid unit size - hole size
            If 'True', entered shift_x, shift_y are ignored and set randomly. Default: `False`.
        fill_value (int): value for the dropped pixels. Default = 0
        mask_fill_value (int): value for the dropped pixels in mask.
            If `None`, transformation is not applied to the mask. Default: `None`.

    Targets:
        image, mask

    Image types:
        uint8, float32

    References:
        https://arxiv.org/abs/2001.04086

    """

    def __init__(
        self,
        ratio: float = 0.5,
        unit_size_min: int = None,
        unit_size_max: int = None,
        holes_number_x: int = None,
        holes_number_y: int = None,
        shift_x: int = 0,
        shift_y: int = 0,
        random_offset: bool = False,
        fill_value: int = 0,
        mask_fill_value: int = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(GridDropout, self).__init__(always_apply, p)
        self.ratio = ratio
        self.unit_size_min = unit_size_min
        self.unit_size_max = unit_size_max
        self.holes_number_x = holes_number_x
        self.holes_number_y = holes_number_y
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.random_offset = random_offset
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.ratio <= 1:
            raise ValueError("ratio must be between 0 and 1.")

    def apply(self, image, holes=(), **params):
        return F.cutout(image, holes, self.fill_value)

    def apply_to_mask(self, image, holes=(), **params):
        if self.mask_fill_value is None:
            return image

        return F.cutout(image, holes, self.mask_fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]
        # set grid using unit size limits
        if self.unit_size_min and self.unit_size_max:
            if not 2 <= self.unit_size_min <= self.unit_size_max:
                raise ValueError("Max unit size should be >= min size, both at least 2 pixels.")
            if self.unit_size_max > min(height, width):
                raise ValueError("Grid size limits must be within the shortest image edge.")
            unit_width = random.randint(self.unit_size_min, self.unit_size_max + 1)
            unit_height = unit_width
        else:
            # set grid using holes numbers
            if self.holes_number_x is None:
                unit_width = max(2, width // 10)
            else:
                if not 1 <= self.holes_number_x <= width // 2:
                    raise ValueError("The hole_number_x must be between 1 and image width//2.")
                unit_width = width // self.holes_number_x
            if self.holes_number_y is None:
                unit_height = max(min(unit_width, height), 2)
            else:
                if not 1 <= self.holes_number_y <= height // 2:
                    raise ValueError("The hole_number_y must be between 1 and image height//2.")
                unit_height = height // self.holes_number_y

        hole_width = int(unit_width * self.ratio)
        hole_height = int(unit_height * self.ratio)
        # min 1 pixel and max unit length - 1
        hole_width = min(max(hole_width, 1), unit_width - 1)
        hole_height = min(max(hole_height, 1), unit_height - 1)
        # set offset of the grid
        if self.shift_x is None:
            shift_x = 0
        else:
            shift_x = min(max(0, self.shift_x), unit_width - hole_width)
        if self.shift_y is None:
            shift_y = 0
        else:
            shift_y = min(max(0, self.shift_y), unit_height - hole_height)
        if self.random_offset:
            shift_x = random.randint(0, unit_width - hole_width)
            shift_y = random.randint(0, unit_height - hole_height)
        holes = []
        for i in range(width // unit_width + 1):
            for j in range(height // unit_height + 1):
                x1 = min(shift_x + unit_width * i, width)
                y1 = min(shift_y + unit_height * j, height)
                x2 = min(x1 + hole_width, width)
                y2 = min(y1 + hole_height, height)
                holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "ratio",
            "unit_size_min",
            "unit_size_max",
            "holes_number_x",
            "holes_number_y",
            "shift_x",
            "shift_y",
            "mask_fill_value",
            "random_offset",
        )


class ColorJitter(ImageOnlyTransform):
    """Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision,
    this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in
    Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8
    overflow, but we use value saturation.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(
        self,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2,
        always_apply=False,
        p=0.5,
    ):
        super(ColorJitter, self).__init__(always_apply=always_apply, p=p)

        self.brightness = self.__check_values(brightness, "brightness")
        self.contrast = self.__check_values(contrast, "contrast")
        self.saturation = self.__check_values(saturation, "saturation")
        self.hue = self.__check_values(hue, "hue", offset=0, bounds=[-0.5, 0.5], clip=False)

    @staticmethod
    def __check_values(value, name, offset=1, bounds=(0, float("inf")), clip=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [offset - value, offset + value]
            if clip:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError("{} values should be between {}".format(name, bounds))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        return value

    def get_params(self):
        brightness = random.uniform(self.brightness[0], self.brightness[1])
        contrast = random.uniform(self.contrast[0], self.contrast[1])
        saturation = random.uniform(self.saturation[0], self.saturation[1])
        hue = random.uniform(self.hue[0], self.hue[1])

        transforms = [
            lambda x: F.adjust_brightness_torchvision(x, brightness),
            lambda x: F.adjust_contrast_torchvision(x, contrast),
            lambda x: F.adjust_saturation_torchvision(x, saturation),
            lambda x: F.adjust_hue_torchvision(x, hue),
        ]
        random.shuffle(transforms)

        return {"transforms": transforms}

    def apply(self, img, transforms=(), **params):
        if not F.is_rgb_image(img) and not F.is_grayscale_image(img):
            raise TypeError("ColorJitter transformation expects 1-channel or 3-channel images.")

        for transform in transforms:
            img = transform(img)
        return img

    def get_transform_init_args_names(self):
        return ("brightness", "contrast", "saturation", "hue")


class Sharpen(ImageOnlyTransform):
    """Sharpen the input image and overlays the result with the original image.

    Args:
        alpha ((float, float)): range to choose the visibility of the sharpened image. At 0, only the original image is
            visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).
        lightness ((float, float)): range to choose the lightness of the sharpened image. Default: (0.5, 1.0).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5):
        super(Sharpen, self).__init__(always_apply, p)
        self.alpha = self.__check_values(to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0))
        self.lightness = self.__check_values(to_tuple(lightness, 0.0), name="lightness")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError("{} values should be between {}".format(name, bounds))
        return value

    @staticmethod
    def __generate_sharpening_matrix(alpha_sample, lightness_sample):
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [[-1, -1, -1], [-1, 8 + lightness_sample, -1], [-1, -1, -1]],
            dtype=np.float32,
        )

        matrix = (1 - alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return matrix

    def get_params(self):
        alpha = random.uniform(*self.alpha)
        lightness = random.uniform(*self.lightness)
        sharpening_matrix = self.__generate_sharpening_matrix(alpha_sample=alpha, lightness_sample=lightness)
        return {"sharpening_matrix": sharpening_matrix}

    def apply(self, img, sharpening_matrix=None, **params):
        return F.convolve(img, sharpening_matrix)

    def get_transform_init_args_names(self):
        return ("alpha", "lightness")


class Emboss(ImageOnlyTransform):
    """Emboss the input image and overlays the result with the original image.

    Args:
        alpha ((float, float)): range to choose the visibility of the embossed image. At 0, only the original image is
            visible,at 1.0 only its embossed version is visible. Default: (0.2, 0.5).
        strength ((float, float)): strength range of the embossing. Default: (0.2, 0.7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5):
        super(Emboss, self).__init__(always_apply, p)
        self.alpha = self.__check_values(to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0))
        self.strength = self.__check_values(to_tuple(strength, 0.0), name="strength")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError("{} values should be between {}".format(name, bounds))
        return value

    @staticmethod
    def __generate_emboss_matrix(alpha_sample, strength_sample):
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [
                [-1 - strength_sample, 0 - strength_sample, 0],
                [0 - strength_sample, 1, 0 + strength_sample],
                [0, 0 + strength_sample, 1 + strength_sample],
            ],
            dtype=np.float32,
        )
        matrix = (1 - alpha_sample) * matrix_nochange + alpha_sample * matrix_effect
        return matrix

    def get_params(self):
        alpha = random.uniform(*self.alpha)
        strength = random.uniform(*self.strength)
        emboss_matrix = self.__generate_emboss_matrix(alpha_sample=alpha, strength_sample=strength)
        return {"emboss_matrix": emboss_matrix}

    def apply(self, img, emboss_matrix=None, **params):
        return F.convolve(img, emboss_matrix)

    def get_transform_init_args_names(self):
        return ("alpha", "strength")


class Superpixels(ImageOnlyTransform):
    """Transform images partially/completely to their superpixel representation.
    This implementation uses skimage's version of the SLIC algorithm.

    Args:
        p_replace (float or tuple of float): Defines for any segment the probability that the pixels within that
            segment are replaced by their average color (otherwise, the pixels are not changed).
            Examples:
                * A probability of ``0.0`` would mean, that the pixels in no
                  segment are replaced by their average color (image is not
                  changed at all).
                * A probability of ``0.5`` would mean, that around half of all
                  segments are replaced by their average color.
                * A probability of ``1.0`` would mean, that all segments are
                  replaced by their average color (resulting in a voronoi
                  image).
            Behaviour based on chosen data types for this parameter:
                * If a ``float``, then that ``flat`` will always be used.
                * If ``tuple`` ``(a, b)``, then a random probability will be
                  sampled from the interval ``[a, b]`` per image.
        n_segments (int, or tuple of int): Rough target number of how many superpixels to generate (the algorithm
            may deviate from this number). Lower value will lead to coarser superpixels.
            Higher values are computationally more intensive and will hence lead to a slowdown
            * If a single ``int``, then that value will always be used as the
              number of segments.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
        max_size (int or None): Maximum image size at which the augmentation is performed.
            If the width or height of an image exceeds this value, it will be
            downscaled before the augmentation so that the longest side matches `max_size`.
            This is done to speed up the process. The final output image has the same size as the input image.
            Note that in case `p_replace` is below ``1.0``,
            the down-/upscaling will affect the not-replaced pixels too.
            Use ``None`` to apply no down-/upscaling.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(
        self,
        p_replace: Union[float, Sequence[float]] = 0.1,
        n_segments: Union[int, Sequence[int]] = 100,
        max_size: Optional[int] = 128,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.p_replace = to_tuple(p_replace, p_replace)
        self.n_segments = to_tuple(n_segments, n_segments)
        self.max_size = max_size
        self.interpolation = interpolation

        if min(self.n_segments) < 1:
            raise ValueError(f"n_segments must be >= 1. Got: {n_segments}")

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return ("p_replace", "n_segments", "max_size", "interpolation")

    def get_params(self) -> dict:
        n_segments = random.randint(*self.n_segments)
        p = random.uniform(*self.p_replace)
        return {"replace_samples": np.random.random(n_segments) < p, "n_segments": n_segments}

    def apply(self, img: np.ndarray, replace_samples: Sequence[bool] = (False,), n_segments: int = 1, **kwargs):
        return F.superpixels(img, n_segments, replace_samples, self.max_size, self.interpolation)


class TemplateTransform(ImageOnlyTransform):
    """
    Apply blending of input image with specified templates
    Args:
        templates (numpy array or list of numpy arrays): Images as template for transform.
        img_weight ((float, float) or float): If single float will be used as weight for input image.
            If tuple of float img_weight will be in range `[img_weight[0], img_weight[1])`. Default: 0.5.
        template_weight ((float, float) or float): If single float will be used as weight for template.
            If tuple of float template_weight will be in range `[template_weight[0], template_weight[1])`.
            Default: 0.5.
        template_transform: transformation object which could be applied to template,
            must produce template the same size as input image.
        name (string): (Optional) Name of transform, used only for deserialization.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        templates,
        img_weight=0.5,
        template_weight=0.5,
        template_transform=None,
        name=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)

        self.templates = templates if isinstance(templates, (list, tuple)) else [templates]
        self.img_weight = to_tuple(img_weight, img_weight)
        self.template_weight = to_tuple(template_weight, template_weight)
        self.template_transform = template_transform
        self.name = name

    def apply(self, img, template=None, img_weight=0.5, template_weight=0.5, **params):
        return F.add_weighted(img, img_weight, template, template_weight)

    def get_params(self):
        return {
            "img_weight": random.uniform(self.img_weight[0], self.img_weight[1]),
            "template_weight": random.uniform(self.template_weight[0], self.template_weight[1]),
        }

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        template = random.choice(self.templates)

        if self.template_transform is not None:
            template = self.template_transform(image=template)["image"]

        if F.get_num_channels(template) not in [1, F.get_num_channels(img)]:
            raise ValueError(
                "Template must be a single channel or "
                "has the same number of channels as input image ({}), got {}".format(
                    F.get_num_channels(img), F.get_num_channels(template)
                )
            )

        if template.dtype != img.dtype:
            raise ValueError("Image and template must be the same image type")

        if img.shape[:2] != template.shape[:2]:
            raise ValueError(
                "Image and template must be the same size, got {} and {}".format(img.shape[:2], template.shape[:2])
            )

        if F.get_num_channels(template) == 1 and F.get_num_channels(img) > 1:
            template = np.stack((template,) * F.get_num_channels(img), axis=-1)

        # in order to support grayscale image with dummy dim
        template = template.reshape(img.shape)

        return {"template": template}

    @classmethod
    def is_serializable(cls):
        return False

    @property
    def targets_as_params(self):
        return ["image"]

    def _to_dict(self):
        if self.name is None:
            raise ValueError(
                "To make a TemplateTransform serializable you should provide the `name` argument, "
                "e.g. `TemplateTransform(name='my_transform', ...)`."
            )
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}
