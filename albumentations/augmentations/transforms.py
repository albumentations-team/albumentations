from __future__ import absolute_import, division

import math
import numbers
import pathlib
import random
import warnings
from enum import IntEnum
from types import LambdaType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy import special
from scipy.ndimage.filters import gaussian_filter

from albumentations import random_utils
from albumentations.augmentations.blur.functional import blur
from albumentations.augmentations.geometric.functional import hflip, rotate_bound, vflip
from albumentations.augmentations.utils import (
    MAX_VALUES_BY_DTYPE,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
)

from ..core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    NoOp,
    ScaleFloatType,
    to_tuple,
)
from ..core.utils import format_args
from . import functional as F

__all__ = [
    "Normalize",
    "RandomGamma",
    "RandomGridShuffle",
    "HueSaturationValue",
    "RGBShift",
    "RandomBrightness",
    "RandomContrast",
    "GaussNoise",
    "CLAHE",
    "ChannelShuffle",
    "InvertImg",
    "ToGray",
    "ToSepia",
    "JpegCompression",
    "ImageCompression",
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
    "ISONoise",
    "Solarize",
    "Equalize",
    "Posterize",
    "Downscale",
    "MultiplicativeNoise",
    "FancyPCA",
    "ColorJitter",
    "Sharpen",
    "Emboss",
    "Superpixels",
    "TemplateTransform",
    "RingingOvershoot",
    "UnsharpMask",
    "PixelDropout",
    "Spatter",
    "CutAndPaste",
]


class RandomGridShuffle(DualTransform):
    """
    Random shuffle grid's cells on image.

    Args:
        grid ((int, int)): size of grid for splitting image.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, grid: Tuple[int, int] = (3, 3), always_apply: bool = False, p: float = 0.5):
        super(RandomGridShuffle, self).__init__(always_apply, p)
        self.grid = grid

    def apply(self, img: np.ndarray, tiles: np.ndarray = None, **params):
        if tiles is not None:
            img = F.swap_tiles_on_image(img, tiles)
        return img

    def apply_to_mask(self, img: np.ndarray, tiles: np.ndarray = None, **params):
        if tiles is not None:
            img = F.swap_tiles_on_image(img, tiles)
        return img

    def apply_to_keypoint(
        self, keypoint: Tuple[float, ...], tiles: np.ndarray = None, rows: int = 0, cols: int = 0, **params
    ):
        if tiles is None:
            return keypoint

        for (
            current_left_up_corner_row,
            current_left_up_corner_col,
            old_left_up_corner_row,
            old_left_up_corner_col,
            height_tile,
            width_tile,
        ) in tiles:
            x, y = keypoint[:2]

            if (old_left_up_corner_row <= y < (old_left_up_corner_row + height_tile)) and (
                old_left_up_corner_col <= x < (old_left_up_corner_col + width_tile)
            ):
                x = x - old_left_up_corner_col + current_left_up_corner_col
                y = y - old_left_up_corner_row + current_left_up_corner_row
                keypoint = (x, y) + tuple(keypoint[2:])
                break

        return keypoint

    def get_params_dependent_on_targets(self, params):
        height, width = params["image"].shape[:2]
        n, m = self.grid

        if n <= 0 or m <= 0:
            raise ValueError("Grid's values must be positive. Current grid [%s, %s]" % (n, m))

        if n > height // 2 or m > width // 2:
            raise ValueError("Incorrect size cell of grid. Just shuffle pixels of image")

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
            new_index_matrix[eq_mat] = random_utils.permutation(new_index_matrix[eq_mat])

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
            f"{self.__class__.__name__} has been deprecated. Please use ImageCompression",
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
        rain_type: One of [None, "drizzle", "heavy", "torrential"]

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

        return {"drop_length": drop_length, "slant": slant, "rain_drops": rain_drops}

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
            g_color = random.randint(max(self.src_color[1] - 50, 0), self.src_color[1])
            b_color = random.randint(max(self.src_color[2] - 50, 0), self.src_color[2])

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
            "low_y": np.clip(random_utils.normal(loc=0.25, scale=self.scale), 0, 1),
            "high_y": np.clip(random_utils.normal(loc=0.75, scale=self.scale), 0, 1),
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
        if not is_rgb_image(image) and not is_grayscale_image(image):
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
        if not is_rgb_image(image):
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
        super(RandomBrightness, self).__init__(brightness_limit=limit, contrast_limit=0, always_apply=always_apply, p=p)
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
            f"{self.__class__.__name__} has been deprecated. Please use RandomBrightnessContrast",
            FutureWarning,
        )

    def get_transform_init_args(self):
        return {"limit": self.contrast_limit}


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
        sigma = var**0.5

        if self.per_channel:
            gauss = random_utils.normal(self.mean, sigma, image.shape)
        else:
            gauss = random_utils.normal(self.mean, sigma, image.shape[:2])
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
        if not is_rgb_image(img) and not is_grayscale_image(img):
            raise TypeError("CLAHE transformation expects 1-channel or 3-channel images.")

        return F.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self):
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self):
        return ("clip_limit", "tile_grid_size")


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
        uint8, float32
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
        return {"gamma": random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0}

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
        if is_grayscale_image(img):
            warnings.warn("The image is already gray.")
            return img
        if not is_rgb_image(img):
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
        if not is_rgb_image(image):
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
        interpolation: cv2 interpolation method. Could be:
            - single cv2 interpolation flag - selected method will be used for downscale and upscale.
            - dict(downscale=flag, upscale=flag)
            - Downscale.Interpolation(downscale=flag, upscale=flag) -
            Default: Interpolation(downscale=cv2.INTER_NEAREST, upscale=cv2.INTER_NEAREST)

    Targets:
        image

    Image types:
        uint8, float32
    """

    class Interpolation:
        def __init__(self, *, downscale: int = cv2.INTER_NEAREST, upscale: int = cv2.INTER_NEAREST):
            self.downscale = downscale
            self.upscale = upscale

    def __init__(
        self,
        scale_min: float = 0.25,
        scale_max: float = 0.25,
        interpolation: Optional[Union[int, Interpolation, Dict[str, int]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(Downscale, self).__init__(always_apply, p)
        if interpolation is None:
            self.interpolation = self.Interpolation(downscale=cv2.INTER_NEAREST, upscale=cv2.INTER_NEAREST)
            warnings.warn(
                "Using default interpolation INTER_NEAREST, which is sub-optimal."
                "Please specify interpolation mode for downscale and upscale explicitly."
                "For additional information see this PR https://github.com/albumentations-team/albumentations/pull/584"
            )
        elif isinstance(interpolation, int):
            self.interpolation = self.Interpolation(downscale=interpolation, upscale=interpolation)
        elif isinstance(interpolation, self.Interpolation):
            self.interpolation = interpolation
        elif isinstance(interpolation, dict):
            self.interpolation = self.Interpolation(**interpolation)
        else:
            raise ValueError(
                "Wrong interpolation data type. Supported types: `Optional[Union[int, Interpolation, Dict[str, int]]]`."
                f" Got: {type(interpolation)}"
            )

        if scale_min > scale_max:
            raise ValueError("Expected scale_min be less or equal scale_max, got {} {}".format(scale_min, scale_max))
        if scale_max >= 1:
            raise ValueError("Expected scale_max to be less than 1, got {}".format(scale_max))
        self.scale_min = scale_min
        self.scale_max = scale_max

    def apply(self, img: np.ndarray, scale: Optional[float] = None, **params) -> np.ndarray:
        return F.downscale(
            img,
            scale=scale,
            down_interpolation=self.interpolation.downscale,
            up_interpolation=self.interpolation.upscale,
        )

    def get_params(self) -> Dict[str, Any]:
        return {"scale": random.uniform(self.scale_min, self.scale_max)}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return "scale_min", "scale_max"

    def _to_dict(self) -> Dict[str, Any]:
        result = super()._to_dict()
        result["interpolation"] = {"upscale": self.interpolation.upscale, "downscale": self.interpolation.downscale}
        return result


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
            c = 1 if is_grayscale_image(img) else img.shape[-1]
        else:
            c = 1

        if self.elementwise:
            shape = [h, w, c]
        else:
            shape = [c]

        multiplier = random_utils.uniform(self.multiplier[0], self.multiplier[1], shape)
        if is_grayscale_image(img) and img.ndim == 2:
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

        self.transforms = [
            F.adjust_brightness_torchvision,
            F.adjust_contrast_torchvision,
            F.adjust_saturation_torchvision,
            F.adjust_hue_torchvision,
        ]

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

        order = [0, 1, 2, 3]
        random.shuffle(order)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "order": order,
        }

    def apply(self, img, brightness=1.0, contrast=1.0, saturation=1.0, hue=0, order=[0, 1, 2, 3], **params):
        if not is_rgb_image(img) and not is_grayscale_image(img):
            raise TypeError("ColorJitter transformation expects 1-channel or 3-channel images.")
        params = [brightness, contrast, saturation, hue]
        for i in order:
            img = self.transforms[i](img, params[i])
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
        return {"replace_samples": random_utils.random(n_segments) < p, "n_segments": n_segments}

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

        if get_num_channels(template) not in [1, get_num_channels(img)]:
            raise ValueError(
                "Template must be a single channel or "
                "has the same number of channels as input image ({}), got {}".format(
                    get_num_channels(img), get_num_channels(template)
                )
            )

        if template.dtype != img.dtype:
            raise ValueError("Image and template must be the same image type")

        if img.shape[:2] != template.shape[:2]:
            raise ValueError(
                "Image and template must be the same size, got {} and {}".format(img.shape[:2], template.shape[:2])
            )

        if get_num_channels(template) == 1 and get_num_channels(img) > 1:
            template = np.stack((template,) * get_num_channels(img), axis=-1)

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


class RingingOvershoot(ImageOnlyTransform):
    """Create ringing or overshoot artefacts by conlvolving image with 2D sinc filter.

    Args:
        blur_limit (int, (int, int)): maximum kernel size for sinc filter.
            Should be in range [3, inf). Default: (7, 15).
        cutoff (float, (float, float)): range to choose the cutoff frequency in radians.
            Should be in range (0, np.pi)
            Default: (np.pi / 4, np.pi / 2).
        p (float): probability of applying the transform. Default: 0.5.

    Reference:
        dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
        https://arxiv.org/abs/2107.10833

    Targets:
        image
    """

    def __init__(
        self,
        blur_limit: Union[int, Sequence[int]] = (7, 15),
        cutoff: Union[float, Sequence[float]] = (np.pi / 4, np.pi / 2),
        always_apply=False,
        p=0.5,
    ):
        super(RingingOvershoot, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)
        self.cutoff = self.__check_values(to_tuple(cutoff, np.pi / 2), name="cutoff", bounds=(0, np.pi))

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def get_params(self):
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        if ksize % 2 == 0:
            raise ValueError(f"Kernel size must be odd. Got: {ksize}")

        cutoff = random.uniform(*self.cutoff)

        # From dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = np.fromfunction(
                lambda x, y: cutoff
                * special.j1(cutoff * np.sqrt((x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2))
                / (2 * np.pi * np.sqrt((x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2)),
                [ksize, ksize],
            )
        kernel[(ksize - 1) // 2, (ksize - 1) // 2] = cutoff**2 / (4 * np.pi)

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)

        return {"kernel": kernel}

    def apply(self, img, kernel=None, **params):
        return F.convolve(img, kernel)

    def get_transform_init_args_names(self):
        return ("blur_limit", "cutoff")


class UnsharpMask(ImageOnlyTransform):
    """
    Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.

    Args:
        blur_limit (int, (int, int)): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        alpha (float, (float, float)): range to choose the visibility of the sharpened image.
            At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
            Default: (0.2, 0.5).
        threshold (int): Value to limit sharpening only for areas with high pixel difference between original image
            and it's smoothed version. Higher threshold means less sharpening on flat areas.
            Must be in range [0, 255]. Default: 10.
        p (float): probability of applying the transform. Default: 0.5.

    Reference:
        arxiv.org/pdf/2107.10833.pdf

    Targets:
        image
    """

    def __init__(
        self,
        blur_limit: Union[int, Sequence[int]] = (3, 7),
        sigma_limit: Union[float, Sequence[float]] = 0.0,
        alpha: Union[float, Sequence[float]] = (0.2, 0.5),
        threshold: int = 10,
        always_apply=False,
        p=0.5,
    ):
        super(UnsharpMask, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)
        self.sigma_limit = self.__check_values(to_tuple(sigma_limit, 0.0), name="sigma_limit")
        self.alpha = self.__check_values(to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0))
        self.threshold = threshold

        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            raise ValueError("blur_limit and sigma_limit minimum value can not be both equal to 0.")

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("UnsharpMask supports only odd blur limits.")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def get_params(self):
        return {
            "ksize": random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2),
            "sigma": random.uniform(*self.sigma_limit),
            "alpha": random.uniform(*self.alpha),
        }

    def apply(self, img, ksize=3, sigma=0, alpha=0.2, **params):
        return F.unsharp_mask(img, ksize, sigma=sigma, alpha=alpha, threshold=self.threshold)

    def get_transform_init_args_names(self):
        return ("blur_limit", "sigma_limit", "alpha", "threshold")


class PixelDropout(DualTransform):
    """Set pixels to 0 with some probability.

    Args:
        dropout_prob (float): pixel drop probability. Default: 0.01
        per_channel (bool): if set to `True` drop mask will be sampled fo each channel,
            otherwise the same mask will be sampled for all channels. Default: False
        drop_value (number or sequence of numbers or None): Value that will be set in dropped place.
            If set to None value will be sampled randomly, default ranges will be used:
                - uint8 - [0, 255]
                - uint16 - [0, 65535]
                - uint32 - [0, 4294967295]
                - float, double - [0, 1]
            Default: 0
        mask_drop_value (number or sequence of numbers or None): Value that will be set in dropped place in masks.
            If set to None masks will be unchanged. Default: 0
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    Image types:
        any
    """

    def __init__(
        self,
        dropout_prob: float = 0.01,
        per_channel: bool = False,
        drop_value: Optional[Union[float, Sequence[float]]] = 0,
        mask_drop_value: Optional[Union[float, Sequence[float]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.drop_value = drop_value
        self.mask_drop_value = mask_drop_value

        if self.mask_drop_value is not None and self.per_channel:
            raise ValueError("PixelDropout supports mask only with per_channel=False")

    def apply(
        self, img: np.ndarray, drop_mask: np.ndarray = None, drop_value: Union[float, Sequence[float]] = (), **params
    ) -> np.ndarray:
        assert drop_mask is not None
        return F.pixel_dropout(img, drop_mask, drop_value)

    def apply_to_mask(self, img: np.ndarray, drop_mask: np.ndarray = np.array([]), **params) -> np.ndarray:
        if self.mask_drop_value is None:
            return img

        if img.ndim == 2:
            drop_mask = np.squeeze(drop_mask)

        return F.pixel_dropout(img, drop_mask, self.mask_drop_value)

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        shape = img.shape if self.per_channel else img.shape[:2]

        rnd = np.random.RandomState(random.randint(0, 1 << 31))
        # Use choice to create boolean matrix, if we will use binomial after that we will need type conversion
        drop_mask = rnd.choice([True, False], shape, p=[self.dropout_prob, 1 - self.dropout_prob])

        drop_value: Union[float, Sequence[float], np.ndarray]
        if drop_mask.ndim != img.ndim:
            drop_mask = np.expand_dims(drop_mask, -1)
        if self.drop_value is None:
            drop_shape = 1 if is_grayscale_image(img) else int(img.shape[-1])

            if img.dtype in (np.uint8, np.uint16, np.uint32):
                drop_value = rnd.randint(0, int(F.MAX_VALUES_BY_DTYPE[img.dtype]), drop_shape, img.dtype)
            elif img.dtype in [np.float32, np.double]:
                drop_value = rnd.uniform(0, 1, drop_shape).astype(img.dtpye)
            else:
                raise ValueError(f"Unsupported dtype: {img.dtype}")
        else:
            drop_value = self.drop_value

        return {"drop_mask": drop_mask, "drop_value": drop_value}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return ("dropout_prob", "per_channel", "drop_value", "mask_drop_value")


class Spatter(ImageOnlyTransform):
    """
    Apply spatter transform. It simulates corruption which can occlude a lens in the form of rain or mud.

    Args:
        mean (float, or tuple of floats): Mean value of normal distribution for generating liquid layer.
            If single float it will be used as mean.
            If tuple of float mean will be sampled from range `[mean[0], mean[1])`. Default: (0.65).
        std (float, or tuple of floats): Standard deviation value of normal distribution for generating liquid layer.
            If single float it will be used as std.
            If tuple of float std will be sampled from range `[std[0], std[1])`. Default: (0.3).
        gauss_sigma (float, or tuple of floats): Sigma value for gaussian filtering of liquid layer.
            If single float it will be used as gauss_sigma.
            If tuple of float gauss_sigma will be sampled from range `[sigma[0], sigma[1])`. Default: (2).
        cutout_threshold (float, or tuple of floats): Threshold for filtering liqued layer
            (determines number of drops). If single float it will used as cutout_threshold.
            If tuple of float cutout_threshold will be sampled from range `[cutout_threshold[0], cutout_threshold[1])`.
            Default: (0.68).
        intensity (float, or tuple of floats): Intensity of corruption.
            If single float it will be used as intensity.
            If tuple of float intensity will be sampled from range `[intensity[0], intensity[1])`. Default: (0.6).
        mode (string, or list of strings): Type of corruption. Currently, supported options are 'rain' and 'mud'.
             If list is provided type of corruption will be sampled list. Default: ("rain").
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/pdf/1903.12261.pdf
    |  https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
    """

    def __init__(
        self,
        mean: ScaleFloatType = 0.65,
        std: ScaleFloatType = 0.3,
        gauss_sigma: ScaleFloatType = 2,
        cutout_threshold: ScaleFloatType = 0.68,
        intensity: ScaleFloatType = 0.6,
        mode: Union[str, Sequence[str]] = "rain",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        self.mean = to_tuple(mean, mean)
        self.std = to_tuple(std, std)
        self.gauss_sigma = to_tuple(gauss_sigma, gauss_sigma)
        self.intensity = to_tuple(intensity, intensity)
        self.cutout_threshold = to_tuple(cutout_threshold, cutout_threshold)
        self.mode = mode if isinstance(mode, (list, tuple)) else [mode]
        for i in self.mode:
            if i not in ["rain", "mud"]:
                raise ValueError(f"Unsupported color mode: {mode}. Transform supports only `rain` and `mud` mods.")

    def apply(
        self,
        img: np.ndarray,
        non_mud: Optional[np.ndarray] = None,
        mud: Optional[np.ndarray] = None,
        drops: Optional[np.ndarray] = None,
        mode: str = "",
        **params
    ) -> np.ndarray:
        return F.spatter(img, non_mud, mud, drops, mode)

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        h, w = params["image"].shape[:2]

        mean = random.uniform(self.mean[0], self.mean[1])
        std = random.uniform(self.std[0], self.std[1])
        cutout_threshold = random.uniform(self.cutout_threshold[0], self.cutout_threshold[1])
        sigma = random.uniform(self.gauss_sigma[0], self.gauss_sigma[1])
        mode = random.choice(self.mode)
        intensity = random.uniform(self.intensity[0], self.intensity[1])

        liquid_layer = random_utils.normal(size=(h, w), loc=mean, scale=std)
        liquid_layer = gaussian_filter(liquid_layer, sigma=sigma, mode="nearest")
        liquid_layer[liquid_layer < cutout_threshold] = 0

        if mode == "rain":
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = blur(dist, 3).astype(np.uint8)
            dist = F.equalize(dist)

            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = F.convolve(dist, ker)
            dist = blur(dist, 3).astype(np.float32)

            m = liquid_layer * dist
            m *= 1 / np.max(m, axis=(0, 1))

            drops = m[:, :, None] * np.array([238 / 255.0, 238 / 255.0, 175 / 255.0]) * intensity
            mud = None
            non_mud = None
        else:
            m = np.where(liquid_layer > cutout_threshold, 1, 0)
            m = gaussian_filter(m.astype(np.float32), sigma=sigma, mode="nearest")
            m[m < 1.2 * cutout_threshold] = 0
            m = m[..., np.newaxis]
            mud = m * np.array([20 / 255.0, 42 / 255.0, 63 / 255.0])
            non_mud = 1 - m
            drops = None

        return {
            "non_mud": non_mud,
            "mud": mud,
            "drops": drops,
            "mode": mode,
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str, str, str]:
        return "mean", "std", "gauss_sigma", "intensity", "cutout_threshold", "mode"


class CutAndPaste(DualTransform):
    """
    Args:
        paste_image_dir (str | pathlib.Path):
            directory path where objects to be pasted exist.
        paste_image_glob (str):
            pattern string to match object files. Default is "*.png".
        blend_method (str):
            keyword to select a blending method. Should be one of:
            ["GAUSSIAN", "NORMAL_CLONE", "MIXED_CLONE", "MONOCHROME_TRANSFER"].
            Lower cases will be capitalized. Default is "GAUSSIAN".
        sigma (float):
            standard deviation for Gaussian kernel used in the "GAUSSIAN" blending method.
        num_object_limit ((int, int) or int):
            range of the number of objects to be pasted. The values should be between 0 and 254. Default is (1, 1).
        scale_limit ((float, float) or float):
            scaling factor range of pasted object. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            If the scaled object size will be larger than the base image, it is automatically limited to
            the base image size.
            Default: (-0.1, 0.1).
        rotate_limit ((int, int) or int):
            rotation range of pasted object. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        hflip (bool)):
             flag to enable horisontal random flip of pasted object. Default True.
        vflip (bool):
             flag to enable vertical random flip of pasted object. Default True.
        obj_min_visibility (float):
            when the pasted image covers the existing objects, and the ratios of the remained area
            are smaller than the obj_min_visibility, the objects are filtered out.
            The `obj_min_visibility` should be between 0 and 1. Default is 0.1
        obj_min_area (float):
            when the pasted image covers the existing objects, and the value of the remained area
            are smaller than the obj_min_area, the objects are filtered out.
            Default is 0.
        p (float):
            probability of applying the transform. Default: 0.5.
    Targets:
        image, bboxes, masks
    Image types:
        uint8, float32
    Reference:
        Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, Barret Zoph:
        "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
        Patrick Perez, Michel Gangnet, Andrew Blake: "Poisson Image Editing"
    """

    def __init__(
        self,
        paste_image_dir: Union[str, pathlib.Path],
        paste_image_glob: str = "*.png",
        blend_method: str = "GAUSSIAN",
        sigma: float = 1.0,  # for gaussian blend
        num_object_limit: Tuple[int, int] = (1, 1),
        scale_limit: Tuple[float, float] = (-0.1, 0.1),
        rotate_limit: Tuple[float, float] = (-45, 45),
        hflip: bool = True,
        vflip: bool = True,
        obj_min_visibility: float = 0.1,
        obj_min_area: int = 0,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        if isinstance(paste_image_dir, str):
            paste_image_dir = pathlib.Path(paste_image_dir)
        self.paste_image_dir = paste_image_dir
        self.paste_image_glob = paste_image_glob
        # Sort the file list to be reproducible
        self.paste_image_paths = sorted(list(self.paste_image_dir.glob(self.paste_image_glob)))
        if len(self.paste_image_paths) == 0:
            raise ValueError(f"No png file is found in {self.paste_image_dir}")

        self.blend_method = blend_method.upper()
        if self.blend_method not in F.BLEND_METHODS:
            raise ValueError(f"blend_method should be one of {F.BLEND_METHODS}, but got {blend_method}")

        self.sigma = sigma
        self.num_object_limit = to_tuple(num_object_limit)
        self.scale_limit = to_tuple(scale_limit)
        self.rotate_limit = to_tuple(rotate_limit)
        self.hflip = hflip
        self.vflip = vflip
        self.obj_min_visibility = obj_min_visibility
        self.obj_min_area = obj_min_area

    def apply(
        self, img, paste_image_paths, obj_toplefts, obj_shapes, blend_method, sigma, angles, vflips, hflips, **params
    ):
        n_obj = len(paste_image_paths)
        for i in range(n_obj):
            obj_img, obj_mask, _ = self.load_paste_image(paste_image_paths[i])
            if vflips[i]:
                obj_img = vflip(obj_img)
                obj_mask = vflip(obj_mask)
            if hflips[i]:
                obj_img = hflip(obj_img)
                obj_mask = hflip(obj_mask)
            angle = angles[i]
            obj_img = rotate_bound(obj_img, angle)
            obj_mask = rotate_bound(obj_mask, angle)

            obj_h, obj_w = obj_shapes[i]

            obj_img = self.resize(obj_img, w=obj_w, h=obj_h)
            obj_mask = self.resize(obj_mask, w=obj_w, h=obj_h)
            obj_top, obj_left = obj_toplefts[i]
            img = F.paste(img, obj_img, obj_mask, obj_top, obj_left, sigma=sigma, blend_method=blend_method)
        return img

    def apply_to_bboxes(self, bboxes, new_masks, new_labels, rows, cols, **params):
        # re-create bboxes from modified masks
        new_bboxes = []
        n_obj = len(new_labels)
        for i in range(n_obj):
            mask = new_masks[i]
            x, y, w, h = self.mask2bbox(mask)
            label = new_labels[i]
            new_bboxes.append([x / cols, y / rows, (x + w) / cols, (y + h) / rows, label])
        return new_bboxes

    def apply_to_bbox(self, **params):
        raise NotImplementedError("bbox target is not supported, use bboxes target")

    def apply_to_masks(self, masks, new_masks, **params):
        return new_masks

    def apply_to_mask(self, **params):
        raise NotImplementedError("mask target is not supported, use masks target")

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("paste_image_dir", "blend_method", "sigma", "obj_min_visibility", "obj_min_area")

    def apply_to_keypoints(self, **params):
        return

    @property
    def targets_as_params(self):
        return ["image", "masks", "bboxes"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        h, w = img.shape[:2]
        n_obj = random.randint(*self.num_object_limit)
        paste_image_paths = random.choices(self.paste_image_paths, k=n_obj)
        scales = np.random.uniform(self.scale_limit[0] + 1, self.scale_limit[1] + 1, n_obj)
        angles = np.random.uniform(self.rotate_limit[0], self.rotate_limit[1], n_obj)
        if self.hflip:
            hflips = random.choices([True, False], k=n_obj)
        else:
            hflips = [False] * n_obj
        if self.vflip:
            vflips = random.choices([True, False], k=n_obj)
        else:
            vflips = [False] * n_obj

        # collect paste image props
        obj_masks = []
        obj_labels = []
        obj_toplefts = []
        obj_shapes = []

        for i in range(n_obj):
            paste_image_path = paste_image_paths[i]
            _, obj_mask, obj_label = self.load_paste_image(paste_image_path)

            if vflips[i]:
                obj_mask = vflip(obj_mask)
            if hflips[i]:
                obj_mask = hflip(obj_mask)

            angle = angles[i]
            obj_mask = rotate_bound(obj_mask, angle)

            scale = scales[i]
            obj_h, obj_w = obj_mask.shape[:2]
            # Get resized shape
            obj_h, obj_w = int(scale * obj_h), int(scale * obj_w)
            if obj_h > h or obj_w > w:
                # If object size is larger than base image, resize object
                r = min(h / obj_h, w / obj_w)
                obj_h = int(r * obj_h)
                obj_w = int(r * obj_w)
            obj_mask = self.resize(obj_mask, w=obj_w, h=obj_h, interpolation=cv2.INTER_NEAREST)

            obj_top = int((h - obj_h) * random.uniform(0, 1))
            obj_left = int((w - obj_w) * random.uniform(0, 1))
            obj_masks.append(obj_mask)
            obj_labels.append(obj_label)
            obj_toplefts.append((obj_top, obj_left))
            obj_shapes.append((obj_h, obj_w))

        # get masks and labels of base image
        masks = params["masks"]
        labels = [bbox[4] for bbox in params["bboxes"]]

        # pre-compute mask augmentation because bbox depend on the augmented masks
        new_masks, new_labels = self._apply_to_masks(
            masks, labels, obj_masks, obj_labels, obj_toplefts, obj_shapes, h, w
        )
        return {
            "obj_toplefts": obj_toplefts,
            "obj_shapes": obj_shapes,
            "paste_image_paths": paste_image_paths,
            "blend_method": self.blend_method,
            "sigma": self.sigma,
            "new_masks": new_masks,
            "new_labels": new_labels,
            "scales": scales,
            "angles": angles,
            "vflips": vflips,
            "hflips": hflips,
        }

    def load_paste_image(self, image_path):
        if isinstance(image_path, pathlib.Path):
            image_path = str(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] != 4:
            raise ValueError(f"RGBA image is expected, but the {image_path} has the shape: {img.shape}")
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        label = self.get_label_from_path(image_path)
        mask = np.where(img[:, :, 3] > 0, 1, 0).astype(np.uint8)[:, :, None]
        img = img[:, :, :3]
        return img, mask, label

    def get_label_from_path(self, image_path):
        if not isinstance(image_path, pathlib.Path):
            image_path = pathlib.Path(image_path)
        label = image_path.stem.split("_")[-1]
        if label.isdigit():
            label = int(label)
        return label

    def mask2bbox(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(contour)
        return [x, y, w, h]

    def resize(self, img, w, h, interpolation=cv2.INTER_LINEAR):
        return cv2.resize(img, (w, h), interpolation=interpolation)

    def _apply_to_masks(self, base_masks, base_labels, obj_masks, obj_labels, obj_toplefts, obj_shapes, rows, cols):
        # To make the obj_masks and base_masks mutually exclusive, we assign a depth for each binary mask and
        # select the shallowest mask's pixels as visible ones.
        # The pixels hidden by the visible pixels are reset to zero.
        n_paste = len(obj_masks)
        n_base = len(base_masks)
        n_obj = n_paste + n_base

        if n_paste == 0:
            return base_masks, base_labels

        # Background is set as the deepest
        bg = MAX_VALUES_BY_DTYPE[np.dtype(np.uint8)]
        masks = np.zeros((rows, cols, n_obj), np.uint8)

        original_mask_areas = []

        # The depth is assigned in an index order while all paste images are shallower than base images.
        for i in range(n_paste):
            # get binary mask
            obj_mask = obj_masks[i] > 0
            original_mask_areas.append(obj_mask.sum())
            obj_h, obj_w = obj_shapes[i]
            obj_top, obj_left = obj_toplefts[i]
            depth = i + 1
            masks[obj_top : obj_top + obj_h, obj_left : obj_left + obj_w, i] = obj_mask * depth

        for i in range(n_base):
            # Get binary mask
            base_mask = base_masks[i] > 0
            original_mask_areas.append(base_mask.sum())
            depth = i + 1 + n_paste
            masks[:, :, i + n_paste] = base_mask * depth

        # Set the deepest value to the background area
        masks = np.where(masks == 0, bg, masks)

        # Select minimum (shallowest) pixels along the depth axis
        merged = masks.min(axis=2)
        self.merged = merged
        # Expand to n_obj binary masks from the aggregated mask
        masks = merged[:, :, None] == np.arange(1, n_obj + 1).reshape((1, 1, n_obj))

        # Filter the hidden objects
        mask_areas = np.array([masks[:, :, i].sum() for i in range(n_obj)])
        original_mask_areas = np.array(original_mask_areas)
        is_visibles = np.logical_and(
            mask_areas > self.obj_min_area, original_mask_areas / mask_areas > self.obj_min_visibility
        )
        indices = np.where(is_visibles)[0]
        masks = [masks[:, :, i].astype(np.uint8) for i in indices]
        labels = obj_labels + base_labels
        labels = [labels[i] for i in indices]

        return masks, labels
