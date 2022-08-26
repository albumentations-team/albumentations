from __future__ import absolute_import, division

import math
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
from albumentations.augmentations.color.functional import equalize
from albumentations.augmentations.utils import get_num_channels, is_grayscale_image

from ..core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    NoOp,
    ScaleFloatType,
    ScaleIntType,
    to_tuple,
)
from ..core.utils import format_args
from . import functional as F

__all__ = [
    "Blur",
    "RandomGridShuffle",
    "MotionBlur",
    "MedianBlur",
    "GaussianBlur",
    "GaussNoise",
    "GlassBlur",
    "JpegCompression",
    "ImageCompression",
    "RandomSnow",
    "RandomRain",
    "RandomFog",
    "RandomSunFlare",
    "RandomShadow",
    "RandomToneCurve",
    "Lambda",
    "ISONoise",
    "Downscale",
    "MultiplicativeNoise",
    "Sharpen",
    "Emboss",
    "Superpixels",
    "TemplateTransform",
    "RingingOvershoot",
    "UnsharpMask",
    "AdvancedBlur",
    "PixelDropout",
    "Spatter",
    "Defocus",
    "ZoomBlur",
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
            "low_y": np.clip(random_utils.normal(loc=0.25, scale=self.scale), 0, 1),
            "high_y": np.clip(random_utils.normal(loc=0.75, scale=self.scale), 0, 1),
        }

    def get_transform_init_args_names(self):
        return ("scale",)


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
        allow_shifted (bool): if set to true creates non shifted kernels only,
            otherwise creates randomly shifted kernels. Default: True.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: Union[int, Sequence[int]] = 7,
        allow_shifted: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(blur_limit=blur_limit, always_apply=always_apply, p=p)
        self.allow_shifted = allow_shifted

        if not allow_shifted and self.blur_limit[0] % 2 != 1 or self.blur_limit[1] % 2 != 1:
            raise ValueError(f"Blur limit must be odd when centered=True. Got: {self.blur_limit}")

    def get_transform_init_args_names(self):
        return super().get_transform_init_args_names() + ("allow_shifted",)

    def apply(self, img, kernel=None, **params):
        return F.convolve(img, kernel=kernel)

    def get_params(self):
        ksize = random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        if ksize <= 2:
            raise ValueError("ksize must be > 2. Got: {}".format(ksize))
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        x1, x2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        if x1 == x2:
            y1, y2 = random.sample(range(ksize), 2)
        else:
            y1, y2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)

        def make_odd_val(v1, v2):
            len_v = abs(v1 - v2) + 1
            if len_v % 2 != 1:
                if v2 > v1:
                    v2 -= 1
                else:
                    v1 -= 1
            return v1, v2

        if not self.allow_shifted:
            x1, x2 = make_odd_val(x1, x2)
            y1, y2 = make_odd_val(y1, y2)

            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2

            center = ksize / 2 - 0.5
            dx = xc - center
            dy = yc - center
            x1, x2 = [int(i - dx) for i in [x1, x2]]
            y1, y2 = [int(i - dy) for i in [y1, y2]]

        cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)

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
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
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
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1)
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
        dxy = random_utils.randint(-self.max_delta, self.max_delta, size=(total_pixels, self.iterations, 2))

        return {"dxy": dxy}

    def get_transform_init_args_names(self):
        return ("sigma", "max_delta", "iterations")

    @property
    def targets_as_params(self):
        return ["image"]


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


class AdvancedBlur(ImageOnlyTransform):
    """Blur the input image using a Generalized Normal filter with a randomly selected parameters.
        This transform also adds multiplicative noise to generated kernel before convolution.

    Args:
        blur_limit: maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigmaX_limit: Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigmaX_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        sigmaY_limit: Same as `sigmaY_limit` for another dimension.
        rotate_limit: Range from which a random angle used to rotate Gaussian kernel is picked.
            If limit is a single int an angle is picked from (-rotate_limit, rotate_limit). Default: (-90, 90).
        beta_limit: Distribution shape parameter, 1 is the normal distribution. Values below 1.0 make distribution
            tails heavier than normal, values above 1.0 make it lighter than normal. Default: (0.5, 8.0).
        noise_limit: Multiplicative factor that control strength of kernel noise. Must be positive and preferably
            centered around 1.0. If set single value `noise_limit` will be in range (0, noise_limit).
            Default: (0.75, 1.25).
        p (float): probability of applying the transform. Default: 0.5.

    Reference:
        https://arxiv.org/abs/2107.10833

    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: Union[int, Sequence[int]] = (3, 7),
        sigmaX_limit: Union[float, Sequence[float]] = (0.2, 1.0),
        sigmaY_limit: Union[float, Sequence[float]] = (0.2, 1.0),
        rotate_limit: Union[int, Sequence[int]] = 90,
        beta_limit: Union[float, Sequence[float]] = (0.5, 8.0),
        noise_limit: Union[float, Sequence[float]] = (0.9, 1.1),
        always_apply=False,
        p=0.5,
    ):
        super(AdvancedBlur, self).__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)
        self.sigmaX_limit = self.__check_values(to_tuple(sigmaX_limit, 0.0), name="sigmaX_limit")
        self.sigmaY_limit = self.__check_values(to_tuple(sigmaY_limit, 0.0), name="sigmaY_limit")
        self.rotate_limit = to_tuple(rotate_limit)
        self.beta_limit = to_tuple(beta_limit, low=0.0)
        self.noise_limit = self.__check_values(to_tuple(noise_limit, 0.0), name="noise_limit")

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("AdvancedBlur supports only odd blur limits.")

        if self.sigmaX_limit[0] == 0 and self.sigmaY_limit[0] == 0:
            raise ValueError("sigmaX_limit and sigmaY_limit minimum value can not be both equal to 0.")

        if not (self.beta_limit[0] < 1.0 < self.beta_limit[1]):
            raise ValueError("Beta limit is expected to include 1.0")

    @staticmethod
    def __check_values(value, name, bounds=(0, float("inf"))):
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def apply(self, img, kernel=None, **params):
        return F.convolve(img, kernel=kernel)

    def get_params(self):
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        sigmaX = random.uniform(*self.sigmaX_limit)
        sigmaY = random.uniform(*self.sigmaY_limit)
        angle = np.deg2rad(random.uniform(*self.rotate_limit))

        # Split into 2 cases to avoid selection of narrow kernels (beta > 1) too often.
        if random.random() < 0.5:
            beta = random.uniform(self.beta_limit[0], 1)
        else:
            beta = random.uniform(1, self.beta_limit[1])

        random_state = np.random.RandomState(random.randint(0, 65536))
        noise_matrix = random_state.uniform(*self.noise_limit, size=[ksize, ksize])

        # Generate mesh grid centered at zero.
        ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
        # Shape (ksize, ksize, 2)
        grid = np.stack(np.meshgrid(ax, ax), axis=-1)

        # Calculate rotated sigma matrix
        d_matrix = np.array([[sigmaX**2, 0], [0, sigmaY**2]])
        u_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        sigma_matrix = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

        inverse_sigma = np.linalg.inv(sigma_matrix)
        # Described in "Parameter Estimation For Multivariate Generalized Gaussian Distributions"
        kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
        # Add noise
        kernel = kernel * noise_matrix

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        return {"kernel": kernel}

    def get_transform_init_args_names(self):
        return (
            "blur_limit",
            "sigmaX_limit",
            "sigmaY_limit",
            "rotate_limit",
            "beta_limit",
            "noise_limit",
        )


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
            dist = F.blur(dist, 3).astype(np.uint8)
            dist = equalize(dist)

            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = F.convolve(dist, ker)
            dist = F.blur(dist, 3).astype(np.float32)

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


class Defocus(ImageOnlyTransform):
    """
    Apply defocus transform. See https://arxiv.org/abs/1903.12261.

    Args:
        radius ((int, int) or int): range for radius of defocusing.
            If limit is a single int, the range will be [1, limit]. Default: (3, 10).
        alias_blur ((float, float) or float): range for alias_blur of defocusing (sigma of gaussian blur).
            If limit is a single float, the range will be (0, limit). Default: (0.1, 0.5).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        Any
    """

    def __init__(
        self,
        radius: ScaleIntType = (3, 10),
        alias_blur: ScaleFloatType = (0.1, 0.5),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.radius = to_tuple(radius, low=1)
        self.alias_blur = to_tuple(alias_blur, low=0)

        if self.radius[0] <= 0:
            raise ValueError("Parameter radius must be positive")

        if self.alias_blur[0] < 0:
            raise ValueError("Parameter alias_blur must be non-negative")

    def apply(self, img: np.ndarray, radius: int = 3, alias_blur: float = 0.5, **params) -> np.ndarray:
        return F.defocus(img, radius, alias_blur)

    def get_params(self) -> Dict[str, Any]:
        return {
            "radius": random_utils.randint(self.radius[0], self.radius[1] + 1),
            "alias_blur": random_utils.uniform(self.alias_blur[0], self.alias_blur[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("radius", "alias_blur")


class ZoomBlur(ImageOnlyTransform):
    """
    Apply zoom blur transform. See https://arxiv.org/abs/1903.12261.

    Args:
        max_factor ((float, float) or float): range for max factor for blurring.
            If max_factor is a single float, the range will be (1, limit). Default: (1, 1.31).
            All max_factor values should be larger than 1.
        step_factor ((float, float) or float): If single float will be used as step parameter for np.arange.
            If tuple of float step_factor will be in range `[step_factor[0], step_factor[1])`. Default: (0.01, 0.03).
            All step_factor values should be positive.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        Any
    """

    def __init__(
        self,
        max_factor: ScaleFloatType = 1.31,
        step_factor: ScaleFloatType = (0.01, 0.03),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.max_factor = to_tuple(max_factor, low=1.0)
        self.step_factor = to_tuple(step_factor, step_factor)

        if self.max_factor[0] < 1:
            raise ValueError("Max factor must be larger or equal 1")
        if self.step_factor[0] <= 0:
            raise ValueError("Step factor must be positive")

    def apply(self, img: np.ndarray, zoom_factors: np.ndarray = None, **params) -> np.ndarray:
        assert zoom_factors is not None
        return F.zoom_blur(img, zoom_factors)

    def get_params(self) -> Dict[str, Any]:
        max_factor = random.uniform(self.max_factor[0], self.max_factor[1])
        step_factor = random.uniform(self.step_factor[0], self.step_factor[1])
        return {"zoom_factors": np.arange(1.0, max_factor, step_factor)}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("max_factor", "step_factor")
