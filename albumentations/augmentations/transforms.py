import math
import numbers
import random
import warnings
from enum import IntEnum
from types import LambdaType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import cv2
import numpy as np
from scipy import special
from scipy.ndimage import gaussian_filter

from albumentations import random_utils
from albumentations.augmentations.blur.functional import blur
from albumentations.augmentations.utils import (
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
)
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform, Interpolation, NoOp, to_tuple
from albumentations.core.types import (
    BoxInternalType,
    ChromaticAberrationMode,
    ImageMode,
    KeypointInternalType,
    MorphologyMode,
    ScaleFloatType,
    ScaleIntType,
    ScaleType,
    SpatterMode,
    Targets,
    image_modes,
)
from albumentations.core.utils import format_args

from . import functional as F

__all__ = [
    "Normalize",
    "RandomGamma",
    "RandomGridShuffle",
    "HueSaturationValue",
    "RGBShift",
    "GaussNoise",
    "CLAHE",
    "ChannelShuffle",
    "InvertImg",
    "ToGray",
    "ToRGB",
    "ToSepia",
    "ImageCompression",
    "ToFloat",
    "FromFloat",
    "RandomBrightnessContrast",
    "RandomSnow",
    "RandomGravel",
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
    "ChromaticAberration",
    "Morphological",
]

MAX_JPEG_QUALITY = 100
TWENTY = 20
FIVE = 5
THREE = 3
TWO = 2


class RandomGridShuffle(DualTransform):
    """Random shuffle grid's cells on image.

    Args:
        grid ((int, int)): size of grid for splitting image.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32

    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS)

    def __init__(self, grid: Tuple[int, int] = (2, 2), always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

        n, m = grid

        if not all(isinstance(dim, int) and dim > 0 for dim in [n, m]):
            raise ValueError(f"Grid dimensions must be positive integers. Current grid dimensions: [{n}, {m}]")

        self.grid = grid

    def apply(
        self, img: np.ndarray, tiles: Optional[np.ndarray] = None, mapping: Optional[List[int]] = None, **params: Any
    ) -> np.ndarray:
        return F.swap_tiles_on_image(img, tiles, mapping)

    def apply_to_mask(
        self, mask: np.ndarray, tiles: Optional[np.ndarray] = None, mapping: Optional[List[int]] = None, **params: Any
    ) -> np.ndarray:
        return F.swap_tiles_on_image(mask, tiles, mapping)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        tiles: np.ndarray,
        mapping: List[int],
        **params: Any,
    ) -> KeypointInternalType:
        x, y = keypoint[:2]

        # Find which original tile the keypoint belongs to
        for original_index, new_index in enumerate(mapping):
            start_y, start_x, end_y, end_x = tiles[original_index]
            # check if the keypoint is in this tile
            if start_y <= y < end_y and start_x <= x < end_x:
                # Get the new tile's coordinates
                new_start_y, new_start_x = tiles[new_index][:2]

                # Map the keypoint to the new tile's position
                new_x = (x - start_x) + new_start_x
                new_y = (y - start_y) + new_start_y

                return (new_x, new_y, *keypoint[2:])

        # If the keypoint wasn't in any tile (shouldn't happen), log a warning for debugging purposes
        warn(
            "Keypoint not in any tile, returning it unchanged. This is unexpected and should be investigated.",
            RuntimeWarning,
        )
        return keypoint

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        height, weight = params["image"].shape[:2]
        # check if image size is divisible by grid
        # if not, warn and return empty dict -> no changes will be applied
        if height % self.grid[0] != 0 or weight % self.grid[1] != 0:
            warn("Image size must be divisible by grid size")
            return {"tiles": np.array([])}
        # Generate the original grid
        original_tiles = F.split_uniform_grid((height, weight), self.grid)
        # Shuffle order of tiles
        mapping = random_utils.shuffle(list(range(len(original_tiles))))
        return {"tiles": original_tiles, "mapping": mapping}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("grid",)


class Normalize(ImageOnlyTransform):
    """Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`

    Args:
        mean: mean values
        std: std values
        max_pixel_value: maximum possible pixel value

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        mean: Union[float, Sequence[float]] = (0.485, 0.456, 0.406),
        std: Union[float, Sequence[float]] = (0.229, 0.224, 0.225),
        max_pixel_value: float = 255.0,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return F.normalize(img, self.mean, self.std, self.max_pixel_value)

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("mean", "std", "max_pixel_value")


class ImageCompression(ImageOnlyTransform):
    """Decreases image quality by Jpeg, WebP compression of an image.

    Args:
        quality_lower: lower bound on the image quality. Should be in [0, 100] range for jpeg and [1, 100] for webp.
        quality_upper: upper bound on the image quality. Should be in [0, 100] range for jpeg and [1, 100] for webp.
        compression_type (ImageCompressionType): should be ImageCompressionType.JPEG or ImageCompressionType.WEBP.
            Default: ImageCompressionType.JPEG

    Targets:
        image

    Image types:
        uint8, float32

    """

    class ImageCompressionType(IntEnum):
        """Defines the types of image compression.

        This Enum class is used to specify the image compression format.

        Attributes:
            JPEG (int): Represents the JPEG image compression format.
            WEBP (int): Represents the WEBP image compression format.

        """

        JPEG = 0
        WEBP = 1

    def __init__(
        self,
        quality_lower: int = 99,
        quality_upper: int = 100,
        compression_type: ImageCompressionType = ImageCompressionType.JPEG,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        self.compression_type = ImageCompression.ImageCompressionType(compression_type)
        low_thresh_quality_assert = 0

        if self.compression_type == ImageCompression.ImageCompressionType.WEBP:
            low_thresh_quality_assert = 1

        if not low_thresh_quality_assert <= quality_lower <= MAX_JPEG_QUALITY:
            raise ValueError(f"Invalid quality_lower. Got: {quality_lower}")
        if not low_thresh_quality_assert <= quality_upper <= MAX_JPEG_QUALITY:
            raise ValueError(f"Invalid quality_upper. Got: {quality_upper}")

        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def apply(self, img: np.ndarray, quality: int = 100, image_type: str = ".jpg", **params: Any) -> np.ndarray:
        if img.ndim != TWO and img.shape[-1] not in (1, 3, 4):
            msg = "ImageCompression transformation expects 1, 3 or 4 channel images."
            raise TypeError(msg)
        return F.image_compression(img, quality, image_type)

    def get_params(self) -> Dict[str, Any]:
        image_type = ".jpg"

        if self.compression_type == ImageCompression.ImageCompressionType.WEBP:
            image_type = ".webp"

        return {
            "quality": random.randint(self.quality_lower, self.quality_upper),
            "image_type": image_type,
        }

    def get_transform_init_args(self) -> Dict[str, Any]:
        return {
            "quality_lower": self.quality_lower,
            "quality_upper": self.quality_upper,
            "compression_type": self.compression_type.value,
        }


class RandomSnow(ImageOnlyTransform):
    """Bleach out some pixel values simulating snow.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        snow_point_lower: lower_bond of the amount of snow. Should be in [0, 1] range
        snow_point_upper: upper_bond of the amount of snow. Should be in [0, 1] range
        brightness_coeff: larger number will lead to a more snow on the image. Should be >= 0

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        snow_point_lower: float = 0.1,
        snow_point_upper: float = 0.3,
        brightness_coeff: float = 2.5,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        if not 0 <= snow_point_lower <= snow_point_upper <= 1:
            msg = (
                "Invalid combination of snow_point_lower and snow_point_upper. "
                f"Got: {(snow_point_lower, snow_point_upper)}"
            )
            raise ValueError(msg)
        if brightness_coeff < 0:
            raise ValueError(f"brightness_coeff must be greater than 0. Got: {brightness_coeff}")

        self.snow_point_lower = snow_point_lower
        self.snow_point_upper = snow_point_upper
        self.brightness_coeff = brightness_coeff

    def apply(self, img: np.ndarray, snow_point: float = 0.1, **params: Any) -> np.ndarray:
        return F.add_snow(img, snow_point, self.brightness_coeff)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"snow_point": random.uniform(self.snow_point_lower, self.snow_point_upper)}

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("snow_point_lower", "snow_point_upper", "brightness_coeff")


class RandomGravel(ImageOnlyTransform):
    """Add gravels.

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        gravel_roi: (top-left x, top-left y,
            bottom-right x, bottom right y). Should be in [0, 1] range
        number_of_patches: no. of gravel patches required

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        gravel_roi: Tuple[float, float, float, float] = (0.1, 0.4, 0.9, 0.9),
        number_of_patches: int = 2,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        (gravel_lower_x, gravel_lower_y, gravel_upper_x, gravel_upper_y) = gravel_roi

        if not 0 <= gravel_lower_x < gravel_upper_x <= 1 or not 0 <= gravel_lower_y < gravel_upper_y <= 1:
            raise ValueError(f"Invalid gravel_roi. Got: {gravel_roi}.")
        if number_of_patches < 1:
            raise ValueError(f"Invalid gravel number_of_patches. Got: {number_of_patches}.")

        self.gravel_roi = gravel_roi
        self.number_of_patches = number_of_patches

    def generate_gravel_patch(self, rectangular_roi: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = rectangular_roi
        area = abs((x2 - x1) * (y2 - y1))
        count = area // 10
        gravels = np.empty([count, 2], dtype=np.int64)
        gravels[:, 0] = random_utils.randint(x1, x2, count)
        gravels[:, 1] = random_utils.randint(y1, y2, count)
        return gravels

    def apply(self, img: np.ndarray, gravels_infos: Optional[List[Any]] = None, **params: Any) -> np.ndarray:
        if gravels_infos is None:
            gravels_infos = []
        return F.add_gravel(img, gravels_infos)

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        img = params["image"]
        height, width = img.shape[:2]

        x_min, y_min, x_max, y_max = self.gravel_roi
        x_min = int(x_min * width)
        x_max = int(x_max * width)
        y_min = int(y_min * height)
        y_max = int(y_max * height)

        max_height = 200
        max_width = 30

        rectangular_rois = np.zeros([self.number_of_patches, 4], dtype=np.int64)
        xx1 = random_utils.randint(x_min + 1, x_max, self.number_of_patches)  # xmax
        xx2 = random_utils.randint(x_min, xx1)  # xmin
        yy1 = random_utils.randint(y_min + 1, y_max, self.number_of_patches)  # ymax
        yy2 = random_utils.randint(y_min, yy1)  # ymin

        rectangular_rois[:, 0] = xx2
        rectangular_rois[:, 1] = yy2
        rectangular_rois[:, 2] = [min(tup) for tup in zip(xx1, xx2 + max_height)]
        rectangular_rois[:, 3] = [min(tup) for tup in zip(yy1, yy2 + max_width)]

        minx = []
        maxx = []
        miny = []
        maxy = []
        val = []
        for roi in rectangular_rois:
            gravels = self.generate_gravel_patch(roi)
            x = gravels[:, 0]
            y = gravels[:, 1]
            r = random_utils.randint(1, 4, len(gravels))
            sat = random_utils.randint(0, 255, len(gravels))
            miny.append(np.maximum(y - r, 0))
            maxy.append(np.minimum(y + r, y))
            minx.append(np.maximum(x - r, 0))
            maxx.append(np.minimum(x + r, x))
            val.append(sat)

        return {
            "gravels_infos": np.stack(
                [
                    np.concatenate(miny),
                    np.concatenate(maxy),
                    np.concatenate(minx),
                    np.concatenate(maxx),
                    np.concatenate(val),
                ],
                1,
            )
        }

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return "gravel_roi", "number_of_patches"


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
        slant_lower: int = -10,
        slant_upper: int = 10,
        drop_length: int = 20,
        drop_width: int = 1,
        drop_color: Tuple[int, int, int] = (200, 200, 200),
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        rain_type: Optional[str] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        if rain_type not in ["drizzle", "heavy", "torrential", None]:
            msg = "raint_type must be one of ({}). Got: {}".format(["drizzle", "heavy", "torrential", None], rain_type)
            raise ValueError(msg)
        if not -TWENTY <= slant_lower <= slant_upper <= TWENTY:
            raise ValueError(f"Invalid combination of slant_lower and slant_upper. Got: {(slant_lower, slant_upper)}")
        if not 1 <= drop_width <= FIVE:
            raise ValueError(f"drop_width must be in range [1, 5]. Got: {drop_width}")
        if not 0 <= drop_length <= MAX_JPEG_QUALITY:
            raise ValueError(f"drop_length must be in range [0, 100]. Got: {drop_length}")
        if not 0 <= brightness_coefficient <= 1:
            raise ValueError(f"brightness_coefficient must be in range [0, 1]. Got: {brightness_coefficient}")

        self.slant_lower = slant_lower
        self.slant_upper = slant_upper

        self.drop_length = drop_length
        self.drop_width = drop_width
        self.drop_color = drop_color
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient
        self.rain_type = rain_type

    def apply(
        self,
        img: np.ndarray,
        slant: int = 10,
        drop_length: int = 20,
        rain_drops: Optional[List[Tuple[int, int]]] = None,
        **params: Any,
    ) -> np.ndarray:
        if rain_drops is None:
            rain_drops = []
        return F.add_rain(
            img,
            slant,
            drop_length,
            self.drop_width,
            self.drop_color,
            self.blur_value,
            self.brightness_coefficient,
            rain_drops,
        )

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
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

        for _ in range(num_drops):  # If You want heavy rain, try increasing this
            x = random.randint(slant, width) if slant < 0 else random.randint(0, width - slant)

            y = random.randint(0, height - drop_length)

            rain_drops.append((x, y))

        return {"drop_length": drop_length, "slant": slant, "rain_drops": rain_drops}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
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
        fog_coef_lower: lower limit for fog intensity coefficient. Should be in [0, 1] range.
        fog_coef_upper: upper limit for fog intensity coefficient. Should be in [0, 1] range.
        alpha_coef: transparency of the fog circles. Should be in [0, 1] range.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        fog_coef_lower: float = 0.3,
        fog_coef_upper: float = 1,
        alpha_coef: float = 0.08,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        if not 0 <= fog_coef_lower <= fog_coef_upper <= 1:
            raise ValueError(
                f"Invalid combination if fog_coef_lower and fog_coef_upper. Got: {(fog_coef_lower, fog_coef_upper)}"
            )
        if not 0 <= alpha_coef <= 1:
            raise ValueError(f"alpha_coef must be in range [0, 1]. Got: {alpha_coef}")

        self.fog_coef_lower = fog_coef_lower
        self.fog_coef_upper = fog_coef_upper
        self.alpha_coef = alpha_coef

    def apply(
        self,
        img: np.ndarray,
        fog_coef: np.ndarray = 0.1,
        haze_list: Optional[List[Tuple[int, int]]] = None,
        **params: Any,
    ) -> np.ndarray:
        if haze_list is None:
            haze_list = []
        return F.add_fog(img, fog_coef, self.alpha_coef, haze_list)

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        fog_coef = random.uniform(self.fog_coef_lower, self.fog_coef_upper)

        height, width = imshape = img.shape[:2]

        hw = max(1, int(width // 3 * fog_coef))

        haze_list = []
        midx = width // 2 - 2 * hw
        midy = height // 2 - hw
        index = 1

        while midx > -hw or midy > -hw:
            for _ in range(hw // 10 * index):
                x = random.randint(midx, width - midx - hw)
                y = random.randint(midy, height - midy - hw)
                haze_list.append((x, y))

            midx -= 3 * hw * width // sum(imshape)
            midy -= 3 * hw * height // sum(imshape)
            index += 1

        return {"haze_list": haze_list, "fog_coef": fog_coef}

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("fog_coef_lower", "fog_coef_upper", "alpha_coef")


class RandomSunFlare(ImageOnlyTransform):
    """Simulates Sun Flare for the image

    From https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    Args:
        flare_roi: region of the image where flare will appear (x_min, y_min, x_max, y_max).
            All values should be in range [0, 1].
        angle_lower: should be in range [0, `angle_upper`].
        angle_upper: should be in range [`angle_lower`, 1].
        num_flare_circles_lower: lower limit for the number of flare circles.
            Should be in range [0, `num_flare_circles_upper`].
        num_flare_circles_upper: upper limit for the number of flare circles.
            Should be in range [`num_flare_circles_lower`, inf].
        src_radius:
        src_color: color of the flare

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        flare_roi: Tuple[float, float, float, float] = (0, 0, 1, 0.5),
        angle_lower: float = 0,
        angle_upper: float = 1,
        num_flare_circles_lower: int = 6,
        num_flare_circles_upper: int = 10,
        src_radius: int = 400,
        src_color: Tuple[int, int, int] = (255, 255, 255),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

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
            raise ValueError(f"Invalid flare_roi. Got: {flare_roi}")
        if not 0 <= angle_lower < angle_upper <= 1:
            raise ValueError(f"Invalid combination of angle_lower nad angle_upper. Got: {(angle_lower, angle_upper)}")
        if not 0 <= num_flare_circles_lower < num_flare_circles_upper:
            msg = (
                "Invalid combination of num_flare_circles_lower and num_flare_circles_upper. "
                f"Got: {(num_flare_circles_lower, num_flare_circles_upper)}"
            )
            raise ValueError(msg)

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

    def apply(
        self,
        img: np.ndarray,
        flare_center_x: float = 0.5,
        flare_center_y: float = 0.5,
        circles: Optional[List[Any]] = None,
        **params: Any,
    ) -> np.ndarray:
        if circles is None:
            circles = []
        return F.add_sun_flare(
            img,
            flare_center_x,
            flare_center_y,
            self.src_radius,
            self.src_color,
            circles,
        )

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
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

        def line(t: float) -> Tuple[float, float]:
            return (flare_center_x + t * math.cos(angle), flare_center_y + t * math.sin(angle))

        for t_val in range(-flare_center_x, width - flare_center_x, 10):
            rand_x, rand_y = line(t_val)
            x.append(rand_x)
            y.append(rand_y)

        for _ in range(num_circles):
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

    def get_transform_init_args(self) -> Dict[str, Any]:
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
        shadow_roi: region of the image where shadows
            will appear. All values should be in range [0, 1].
        num_shadows_limit: Lower and upper limits for the possible number of shadows.
        num_shadows_lower: Deprecated: Lower limit for the possible number of shadows.
            Should be in range [0, `num_shadows_upper`].
        num_shadows_upper: Deprecated: Lower limit for the possible number of shadows.
            Should be in range [`num_shadows_lower`, inf].
        shadow_dimension: number of edges in the shadow polygons

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        shadow_roi: Tuple[float, float, float, float] = (0, 0.5, 1, 1),
        num_shadows_limit: Tuple[int, int] = (1, 2),
        num_shadows_lower: Optional[int] = None,
        num_shadows_upper: Optional[int] = None,
        shadow_dimension: int = 5,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        (shadow_lower_x, shadow_lower_y, shadow_upper_x, shadow_upper_y) = shadow_roi

        if not 0 <= shadow_lower_x <= shadow_upper_x <= 1 or not 0 <= shadow_lower_y <= shadow_upper_y <= 1:
            raise ValueError(f"Invalid shadow_roi. Got: {shadow_roi}")

        if num_shadows_lower is not None or num_shadows_upper is not None:
            warn(
                "`num_shadows_lower` and `num_shadows_upper` are deprecated. "
                "Use `num_shadows_limit` as tuple (num_shadows_lower, num_shadows_upper) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        num_shadows_lower = num_shadows_lower or num_shadows_limit[0]
        num_shadows_upper = num_shadows_upper or num_shadows_limit[1]

        if not 0 <= num_shadows_lower <= num_shadows_upper:
            msg = "Invalid combination of num_shadows_lower nad num_shadows_upper. "
            f"Got: {(num_shadows_lower, num_shadows_upper)}"
            raise ValueError(msg)

        self.shadow_roi = shadow_roi

        self.num_shadows_lower = num_shadows_lower
        self.num_shadows_upper = num_shadows_upper

        self.shadow_dimension = shadow_dimension

    def apply(
        self, img: np.ndarray, vertices_list: Optional[List[List[Tuple[int, int]]]] = None, **params: Any
    ) -> np.ndarray:
        if vertices_list is None:
            vertices_list = []
        return F.add_shadow(img, vertices_list)

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
        img = params["image"]
        height, width = img.shape[:2]

        num_shadows = random_utils.randint(self.num_shadows_lower, self.num_shadows_upper)

        x_min, y_min, x_max, y_max = self.shadow_roi

        x_min = int(x_min * width)
        x_max = int(x_max * width)
        y_min = int(y_min * height)
        y_max = int(y_max * height)

        vertices_list = [
            np.stack(
                [
                    random_utils.randint(x_min, x_max, size=5),
                    random_utils.randint(y_min, y_max, size=5),
                ],
                axis=1,
            )
            for _ in range(num_shadows)
        ]

        return {"vertices_list": vertices_list}

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return (
            "shadow_roi",
            "num_shadows_lower",
            "num_shadows_upper",
            "shadow_dimension",
        )


class RandomToneCurve(ImageOnlyTransform):
    """Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.

    Args:
        scale: standard deviation of the normal distribution.
            Used to sample random distances to move two control points that modify the image's curve.
            Values should be in range [0, 1]. Default: 0.1


    Targets:
        image

    Image types:
        uint8

    """

    def __init__(
        self,
        scale: float = 0.1,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.scale = scale

    def apply(self, img: np.ndarray, low_y: float, high_y: float, **params: Any) -> np.ndarray:
        return F.move_tone_curve(img, low_y, high_y)

    def get_params(self) -> Dict[str, float]:
        return {
            "low_y": np.clip(random_utils.normal(loc=0.25, scale=self.scale), 0, 1),
            "high_y": np.clip(random_utils.normal(loc=0.75, scale=self.scale), 0, 1),
        }

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("scale",)


class HueSaturationValue(ImageOnlyTransform):
    """Randomly change hue, saturation and value of the input image.

    Args:
        hue_shift_limit: range for changing hue. If hue_shift_limit is a single int, the range
            will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
        sat_shift_limit: range for changing saturation. If sat_shift_limit is a single int,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
        val_shift_limit: range for changing value. If val_shift_limit is a single int, the range
            will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        hue_shift_limit: ScaleIntType = 20,
        sat_shift_limit: ScaleIntType = 30,
        val_shift_limit: ScaleIntType = 20,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(
        self, img: np.ndarray, hue_shift: int = 0, sat_shift: int = 0, val_shift: int = 0, **params: Any
    ) -> np.ndarray:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "HueSaturationValue transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)
        return F.shift_hsv(img, hue_shift, sat_shift, val_shift)

    def get_params(self) -> Dict[str, float]:
        return {
            "hue_shift": random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
            "sat_shift": random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
            "val_shift": random.uniform(self.val_shift_limit[0], self.val_shift_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("hue_shift_limit", "sat_shift_limit", "val_shift_limit")


class Solarize(ImageOnlyTransform):
    """Invert all pixel values above a threshold.

    Args:
        threshold: range for solarizing threshold.
            If threshold is a single value, the range will be [threshold, threshold]. Default: 128.
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        any

    """

    def __init__(self, threshold: ScaleType = 128, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

        if isinstance(threshold, (int, float)):
            self.threshold = to_tuple(threshold, low=threshold)
        else:
            self.threshold = to_tuple(threshold, low=0)

        self.threshold = self.threshold

    def apply(self, img: np.ndarray, threshold: int = 0, **params: Any) -> np.ndarray:
        return F.solarize(img, threshold)

    def get_params(self) -> Dict[str, float]:
        return {"threshold": random.uniform(self.threshold[0], self.threshold[1])}

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("threshold",)


class Posterize(ImageOnlyTransform):
    """Reduce the number of bits for each color channel.

    Args:
        num_bits ((int, int) or int,
                  or list of ints [r, g, b],
                  or list of ints [[r1, r1], [g1, g2], [b1, b2]]): number of high bits.
            If num_bits is a single value, the range will be [num_bits, num_bits].
            Must be in range [0, 8]. Default: 4.
        p: probability of applying the transform. Default: 0.5.

    Targets:
    image

    Image types:
        uint8

    """

    def __init__(
        self,
        num_bits: Union[int, Tuple[int, int], Tuple[int, int, int]] = 4,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        if isinstance(num_bits, int):
            self.num_bits = to_tuple(num_bits, num_bits)
        elif isinstance(num_bits, Sequence) and len(num_bits) == THREE:
            self.num_bits = [to_tuple(i, 0) for i in num_bits]  # type: ignore[assignment]
        else:
            self.num_bits = to_tuple(num_bits, 0)  # type: ignore[arg-type]

    def apply(self, img: np.ndarray, num_bits: int = 1, **params: Any) -> np.ndarray:
        return F.posterize(img, num_bits)

    def get_params(self) -> Dict[str, Any]:
        if len(self.num_bits) == THREE:
            return {"num_bits": [random.randint(int(i[0]), int(i[1])) for i in self.num_bits]}  # type: ignore[index]
        num_bits = self.num_bits
        return {"num_bits": random.randint(int(num_bits[0]), int(num_bits[1]))}

    def get_transform_init_args_names(self) -> Tuple[str]:
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
        mode: ImageMode = "cv",
        by_channels: bool = True,
        mask: Optional[np.ndarray] = None,
        mask_params: Tuple[()] = (),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        if mode not in image_modes:
            raise ValueError(f"Unsupported equalization mode. Supports: {image_modes}. " f"Got: {mode}")

        super().__init__(always_apply, p)
        self.mode = mode
        self.by_channels = by_channels
        self.mask = mask
        self.mask_params = mask_params

    def apply(self, img: np.ndarray, mask: Optional[np.ndarray] = None, **params: Any) -> np.ndarray:
        return F.equalize(img, mode=self.mode, by_channels=self.by_channels, mask=mask)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not callable(self.mask):
            return {"mask": self.mask}

        return {"mask": self.mask(**params)}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image", *list(self.mask_params)]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("mode", "by_channels", "mask", "mask_params")


class RGBShift(ImageOnlyTransform):
    """Randomly shift values for each channel of the input RGB image.

    Args:
        r_shift_limit: range for changing values for the red channel. If r_shift_limit is a single
            int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).
        g_shift_limit: range for changing values for the green channel. If g_shift_limit is a
            single int, the range  will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).
        b_shift_limit: range for changing values for the blue channel. If b_shift_limit is a single
            int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        r_shift_limit: ScaleIntType = 20,
        g_shift_limit: ScaleIntType = 20,
        b_shift_limit: ScaleIntType = 20,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def apply(self, img: np.ndarray, r_shift: int = 0, g_shift: int = 0, b_shift: int = 0, **params: Any) -> np.ndarray:
        if not is_rgb_image(img):
            msg = "RGBShift transformation expects 3-channel images."
            raise TypeError(msg)
        return F.shift_rgb(img, r_shift, g_shift, b_shift)

    def get_params(self) -> Dict[str, Any]:
        return {
            "r_shift": random.uniform(self.r_shift_limit[0], self.r_shift_limit[1]),
            "g_shift": random.uniform(self.g_shift_limit[0], self.g_shift_limit[1]),
            "b_shift": random.uniform(self.b_shift_limit[0], self.b_shift_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("r_shift_limit", "g_shift_limit", "b_shift_limit")


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit: factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit: factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max: If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        brightness_limit: ScaleFloatType = 0.2,
        contrast_limit: ScaleFloatType = 0.2,
        brightness_by_max: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def apply(self, img: np.ndarray, alpha: float = 1.0, beta: float = 0.0, **params: Any) -> np.ndarray:
        return F.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)

    def get_params(self) -> Dict[str, float]:
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("brightness_limit", "contrast_limit", "brightness_by_max")


class GaussNoise(ImageOnlyTransform):
    """Apply gaussian noise to the input image.

    Args:
        var_limit: variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean: mean of the noise. Default: 0
        per_channel: if set to True, noise will be sampled for each channel independently.
            Otherwise, the noise will be sampled once for all channels. Default: True
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        var_limit: ScaleFloatType = (10.0, 50.0),
        mean: float = 0,
        per_channel: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                msg = "Lower var_limit should be non negative."
                raise ValueError(msg)
            if var_limit[1] < 0:
                msg = "Upper var_limit should be non negative."
                raise ValueError(msg)
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                msg = "var_limit should be non negative."
                raise ValueError(msg)

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(f"Expected var_limit type to be one of (int, float, tuple, list), got {type(var_limit)}")

        self.mean = mean
        self.per_channel = per_channel

    def apply(self, img: np.ndarray, gauss: Optional[float] = None, **params: Any) -> np.ndarray:
        return F.gauss_noise(img, gauss=gauss)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, float]:
        image = params["image"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var**0.5

        if self.per_channel:
            gauss = random_utils.normal(self.mean, sigma, image.shape)
        else:
            gauss = random_utils.normal(self.mean, sigma, image.shape[:2])
            if len(image.shape) == THREE:
                gauss = np.expand_dims(gauss, -1)

        return {"gauss": gauss}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("var_limit", "per_channel", "mean")


class ISONoise(ImageOnlyTransform):
    """Apply camera sensor noise.

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

    def __init__(
        self,
        color_shift: Tuple[float, float] = (0.01, 0.05),
        intensity: Tuple[float, float] = (0.1, 0.5),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.intensity = intensity
        self.color_shift = color_shift

    def apply(
        self,
        img: np.ndarray,
        color_shift: float = 0.05,
        intensity: float = 1.0,
        random_state: Optional[int] = None,
        **params: Any,
    ) -> np.ndarray:
        return F.iso_noise(img, color_shift, intensity, np.random.RandomState(random_state))

    def get_params(self) -> Dict[str, Any]:
        return {
            "color_shift": random.uniform(self.color_shift[0], self.color_shift[1]),
            "intensity": random.uniform(self.intensity[0], self.intensity[1]),
            "random_state": random.randint(0, 65536),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("intensity", "color_shift")


class CLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit: upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size: size of grid for histogram equalization. Default: (8, 8).
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8

    """

    def __init__(
        self,
        clip_limit: ScaleFloatType = 4.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = cast(Tuple[int, int], tuple(tile_grid_size))

    def apply(self, img: np.ndarray, clip_limit: float = 2, **params: Any) -> np.ndarray:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "CLAHE transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)

        return F.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self) -> Dict[str, float]:
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("clip_limit", "tile_grid_size")


class ChannelShuffle(ImageOnlyTransform):
    """Randomly rearrange channels of the input RGB image.

    Args:
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def apply(self, img: np.ndarray, channels_shuffled: Tuple[int, int, int] = (0, 1, 2), **params: Any) -> np.ndarray:
        return F.channel_shuffle(img, channels_shuffled)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        ch_arr = list(range(img.shape[2]))
        random.shuffle(ch_arr)
        return {"channels_shuffled": ch_arr}

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()


class InvertImg(ImageOnlyTransform):
    """Invert the input image by subtracting pixel values from max values of the image types,
    i.e., 255 for uint8 and 1.0 for float32.

    Args:
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return F.invert(img)

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()


class RandomGamma(ImageOnlyTransform):
    """Applies random gamma correction to an image as a form of data augmentation.

    This class adjusts the luminance of an image by applying gamma correction with a randomly
    selected gamma value from a specified range. Gamma correction can simulate various lighting
    conditions, potentially enhancing model generalization. For more details on gamma correction,
    see: https://en.wikipedia.org/wiki/Gamma_correction

    Attributes:
        gamma_limit (Union[int, Tuple[int, int]]): The range for gamma adjustment. If `gamma_limit` is a single
            int, the range will be interpreted as (-gamma_limit, gamma_limit), defining how much
            to adjust the image's gamma. Default is (80, 120).
        always_apply (bool): If `True`, the transform will always be applied, regardless of `p`.
            Default is `False`.
        p (float): The probability that the transform will be applied. Default is 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        gamma_limit: ScaleIntType = (80, 120),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.gamma_limit = to_tuple(gamma_limit)

    def apply(self, img: np.ndarray, gamma: float = 1, **params: Any) -> np.ndarray:
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self) -> Dict[str, float]:
        return {"gamma": random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("gamma_limit",)


class ToGray(ImageOnlyTransform):
    """Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater
    than 127, invert the resulting grayscale image.

    Args:
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if is_grayscale_image(img):
            warnings.warn("The image is already gray.")
            return img
        if not is_rgb_image(img):
            msg = "ToGray transformation expects 3-channel images."
            raise TypeError(msg)

        return F.to_gray(img)

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()


class ToRGB(ImageOnlyTransform):
    """Convert the input grayscale image to RGB.

    Args:
        p: probability of applying the transform. Default: 1.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if is_rgb_image(img):
            warnings.warn("The image is already an RGB.")
            return img
        if not is_grayscale_image(img):
            msg = "ToRGB transformation expects 2-dim images or 3-dim with the last dimension equal to 1."
            raise TypeError(msg)

        return F.gray_to_rgb(img)

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()


class ToSepia(ImageOnlyTransform):
    """Applies sepia filter to the input RGB image

    Args:
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.sepia_transformation_matrix = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
        )

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if not is_rgb_image(img):
            msg = "ToSepia transformation expects 3-channel images."
            raise TypeError(msg)
        return F.linear_transformation_rgb(img, self.sepia_transformation_matrix)

    def get_transform_init_args_names(self) -> Tuple[()]:
        return ()


class ToFloat(ImageOnlyTransform):
    """Divide pixel values by `max_value` to get a float32 output array where all values lie in the range [0, 1.0].
    If `max_value` is None the transform will try to infer the maximum value by inspecting the data type of the input
    image.

    See Also:
        :class:`~albumentations.augmentations.transforms.FromFloat`

    Args:
        max_value: maximum possible input value. Default: None.
        p: probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        any type

    """

    def __init__(self, max_value: Optional[float] = None, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return F.to_float(img, self.max_value)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("max_value",)


class FromFloat(ImageOnlyTransform):
    """Take an input array where all values should lie in the range [0, 1.0], multiply them by `max_value` and then
    cast the resulted value to a type specified by `dtype`. If `max_value` is None the transform will try to infer
    the maximum value for the data type from the `dtype` argument.

    This is the inverse transform for :class:`~albumentations.augmentations.transforms.ToFloat`.

    Args:
        max_value: maximum possible input value. Default: None.
        dtype: data type of the output. See the `'Data types' page from the NumPy docs`_.
            Default: 'uint16'.
        p: probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        float32

    .. _'Data types' page from the NumPy docs:
       https://docs.scipy.org/doc/numpy/user/basics.types.html

    """

    def __init__(
        self, dtype: str = "uint16", max_value: Optional[float] = None, always_apply: bool = False, p: float = 1.0
    ):
        super().__init__(always_apply, p)
        self.dtype = np.dtype(dtype)
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return F.from_float(img, self.dtype, self.max_value)

    def get_transform_init_args(self) -> Dict[str, Any]:
        return {"dtype": self.dtype.name, "max_value": self.max_value}


class Downscale(ImageOnlyTransform):
    """Decreases image quality by downscaling and upscaling back.

    Args:
        scale_min: lower bound on the image scale. Should be <= scale_max.
        scale_max: upper bound on the image scale. Should be < 1.
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

    def __init__(
        self,
        scale_min: float = 0.25,
        scale_max: float = 0.25,
        interpolation: Optional[Union[int, Interpolation, Dict[str, int]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        if interpolation is None:
            self.interpolation = Interpolation(downscale=cv2.INTER_NEAREST, upscale=cv2.INTER_NEAREST)
            warnings.warn(
                "Using default interpolation INTER_NEAREST, which is sub-optimal."
                "Please specify interpolation mode for downscale and upscale explicitly."
                "For additional information see this PR https://github.com/albumentations-team/albumentations/pull/584"
            )
        elif isinstance(interpolation, int):
            self.interpolation = Interpolation(downscale=interpolation, upscale=interpolation)
        elif isinstance(interpolation, Interpolation):
            self.interpolation = interpolation
        elif isinstance(interpolation, dict):
            self.interpolation = Interpolation(**interpolation)
        else:
            raise ValueError(
                "Wrong interpolation data type. Supported types: `Optional[Union[int, Interpolation, Dict[str, int]]]`."
                f" Got: {type(interpolation)}"
            )

        if scale_min > scale_max:
            raise ValueError(f"Expected scale_min be less or equal scale_max, got {scale_min} {scale_max}")
        if scale_max >= 1:
            raise ValueError(f"Expected scale_max to be less than 1, got {scale_max}")
        self.scale_min = scale_min
        self.scale_max = scale_max

    def apply(self, img: np.ndarray, scale: float, **params: Any) -> np.ndarray:
        if isinstance(self.interpolation, int):
            msg = "Should not be here, added for typing purposes. Please report this issue."
            raise TypeError(msg)
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

    def to_dict_private(self) -> Dict[str, Any]:
        if isinstance(self.interpolation, int):
            msg = "Should not be here, added for typing purposes. Please report this issue."
            raise TypeError(msg)
        result = super().to_dict_private()
        result["interpolation"] = {"upscale": self.interpolation.upscale, "downscale": self.interpolation.downscale}
        return result


class Lambda(NoOp):
    """A flexible transformation class for using user-defined transformation functions per targets.
    Function signature must include **kwargs to accept optional arguments like interpolation method, image size, etc:

    Args:
        image: Image transformation function.
        mask: Mask transformation function.
        keypoint: Keypoint transformation function.
        bbox: BBox transformation function.
        global_label: Global label transformation function.
        always_apply: Indicates whether this transformation should be always applied.
        p: probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints, global_label

    Image types:
        Any

    """

    def __init__(
        self,
        image: Optional[Callable[..., Any]] = None,
        mask: Optional[Callable[..., Any]] = None,
        keypoint: Optional[Callable[..., Any]] = None,
        bbox: Optional[Callable[..., Any]] = None,
        global_label: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)

        self.name = name
        self.custom_apply_fns = {
            target_name: F.noop for target_name in ("image", "mask", "keypoint", "bbox", "global_label")
        }
        for target_name, custom_apply_fn in {
            "image": image,
            "mask": mask,
            "keypoint": keypoint,
            "bbox": bbox,
            "global_label": global_label,
        }.items():
            if custom_apply_fn is not None:
                if isinstance(custom_apply_fn, LambdaType) and custom_apply_fn.__name__ == "<lambda>":
                    warnings.warn(
                        "Using lambda is incompatible with multiprocessing. "
                        "Consider using regular functions or partial()."
                    )

                self.custom_apply_fns[target_name] = custom_apply_fn

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        fn = self.custom_apply_fns["image"]
        return fn(img, **params)

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        fn = self.custom_apply_fns["mask"]
        return fn(mask, **params)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        fn = self.custom_apply_fns["bbox"]
        return fn(bbox, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
        fn = self.custom_apply_fns["keypoint"]
        return fn(keypoint, **params)

    def apply_to_global_label(self, label: np.ndarray, **params: Any) -> np.ndarray:
        fn = self.custom_apply_fns["global_label"]
        return fn(label, **params)

    @classmethod
    def is_serializable(cls) -> bool:
        return False

    def to_dict_private(self) -> Dict[str, Any]:
        if self.name is None:
            msg = (
                "To make a Lambda transform serializable you should provide the `name` argument, "
                "e.g. `Lambda(name='my_transform', image=<some func>, ...)`."
            )
            raise ValueError(msg)
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}

    def __repr__(self) -> str:
        state = {"name": self.name}
        state.update(self.custom_apply_fns.items())  # type: ignore[arg-type]
        state.update(self.get_base_init_args())
        return f"{self.__class__.__name__}({format_args(state)})"


class MultiplicativeNoise(ImageOnlyTransform):
    """Multiply image to random number or array of numbers.

    Args:
        multiplier: If single float image will be multiplied to this number.
            If tuple of float multiplier will be in range `[multiplier[0], multiplier[1])`. Default: (0.9, 1.1).
        per_channel: If `False`, same values for all channels will be used.
            If `True` use sample values for each channels. Default False.
        elementwise: If `False` multiply multiply all pixels in an image with a random value sampled once.
            If `True` Multiply image pixels with values that are pixelwise randomly sampled. Default: False.

    Targets:
        image

    Image types:
        Any

    """

    def __init__(
        self,
        multiplier: ScaleFloatType = (0.9, 1.1),
        per_channel: bool = False,
        elementwise: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.multiplier = to_tuple(multiplier, multiplier)
        self.per_channel = per_channel
        self.elementwise = elementwise

    def apply(self, img: np.ndarray, multiplier: float = np.array([1]), **kwargs: Any) -> np.ndarray:
        return F.multiply(img, multiplier)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.multiplier[0] == self.multiplier[1]:
            return {"multiplier": np.array([self.multiplier[0]])}

        img = params["image"]

        height, width = img.shape[:2]

        num_channels = (1 if is_grayscale_image(img) else img.shape[-1]) if self.per_channel else 1

        shape = [height, width, num_channels] if self.elementwise else [num_channels]

        multiplier = random_utils.uniform(self.multiplier[0], self.multiplier[1], tuple(shape))
        if is_grayscale_image(img) and img.ndim == TWO:
            multiplier = np.squeeze(multiplier)

        return {"multiplier": multiplier}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return "multiplier", "per_channel", "elementwise"


class FancyPCA(ImageOnlyTransform):
    """Augment RGB image using FancyPCA from Krizhevsky's paper
    "ImageNet Classification with Deep Convolutional Neural Networks"

    Args:
        alpha:  how much to perturb/scale the eigen vecs and vals.
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

    def __init__(self, alpha: float = 0.1, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, img: np.ndarray, alpha: float = 0.1, **params: Any) -> np.ndarray:
        return F.fancy_pca(img, alpha)

    def get_params(self) -> Dict[str, float]:
        return {"alpha": random.gauss(0, self.alpha)}

    def get_transform_init_args_names(self) -> Tuple[str]:
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
        brightness: ScaleFloatType = 0.2,
        contrast: ScaleFloatType = 0.2,
        saturation: ScaleFloatType = 0.2,
        hue: ScaleFloatType = 0.2,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        self.brightness = self.__check_values(brightness, "brightness")
        self.contrast = self.__check_values(contrast, "contrast")
        self.saturation = self.__check_values(saturation, "saturation")
        self.hue = self.__check_values(hue, "hue", offset=0, bounds=(-0.5, 0.5), clip=False)

        self.transforms = [
            F.adjust_brightness_torchvision,
            F.adjust_contrast_torchvision,
            F.adjust_saturation_torchvision,
            F.adjust_hue_torchvision,
        ]

    @staticmethod
    def __check_values(
        value: ScaleFloatType,
        name: str,
        offset: float = 1,
        bounds: Tuple[float, float] = (0, float("inf")),
        clip: bool = True,
    ) -> Tuple[float, float]:
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [offset - value, offset + value]
            if clip:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == TWO:
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError(f"{name} values should be between {bounds}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        return value

    def get_params(self) -> Dict[str, Any]:
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

    def apply(
        self,
        img: np.ndarray,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        hue: float = 0,
        order: Optional[List[int]] = None,
        **params: Any,
    ) -> np.ndarray:
        if order is None:
            order = [0, 1, 2, 3]
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "ColorJitter transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)
        color_transforms = [brightness, contrast, saturation, hue]
        for i in order:
            img = self.transforms[i](img, color_transforms[i])  # type: ignore[operator]
        return img

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return ("brightness", "contrast", "saturation", "hue")


class Sharpen(ImageOnlyTransform):
    """Sharpen the input image and overlays the result with the original image.

    Args:
        alpha: range to choose the visibility of the sharpened image. At 0, only the original image is
            visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).
        lightness: range to choose the lightness of the sharpened image. Default: (0.5, 1.0).
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    """

    def __init__(
        self,
        alpha: Tuple[float, float] = (0.2, 0.5),
        lightness: Tuple[float, float] = (0.5, 1.0),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.alpha = self.__check_values(to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0))
        self.lightness = self.__check_values(to_tuple(lightness, 0.0), name="lightness")

    @staticmethod
    def __check_values(
        value: Tuple[float, float], name: str, bounds: Tuple[float, float] = (0, float("inf"))
    ) -> Tuple[float, float]:
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    @staticmethod
    def __generate_sharpening_matrix(alpha_sample: np.ndarray, lightness_sample: np.ndarray) -> np.ndarray:
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [[-1, -1, -1], [-1, 8 + lightness_sample, -1], [-1, -1, -1]],
            dtype=np.float32,
        )

        return (1 - alpha_sample) * matrix_nochange + alpha_sample * matrix_effect

    def get_params(self) -> Dict[str, np.ndarray]:
        alpha = random.uniform(*self.alpha)
        lightness = random.uniform(*self.lightness)
        sharpening_matrix = self.__generate_sharpening_matrix(alpha_sample=alpha, lightness_sample=lightness)
        return {"sharpening_matrix": sharpening_matrix}

    def apply(self, img: np.ndarray, sharpening_matrix: Optional[np.ndarray] = None, **params: Any) -> np.ndarray:
        return F.convolve(img, sharpening_matrix)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("alpha", "lightness")


class Emboss(ImageOnlyTransform):
    """Emboss the input image and overlays the result with the original image.

    Args:
        alpha: range to choose the visibility of the embossed image. At 0, only the original image is
            visible,at 1.0 only its embossed version is visible. Default: (0.2, 0.5).
        strength: strength range of the embossing. Default: (0.2, 0.7).
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    """

    def __init__(
        self,
        alpha: Tuple[float, float] = (0.2, 0.5),
        strength: Tuple[float, float] = (0.2, 0.7),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.alpha = self.__check_values(to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0))
        self.strength = self.__check_values(to_tuple(strength, 0.0), name="strength")

    @staticmethod
    def __check_values(
        value: Tuple[float, float], name: str, bounds: Tuple[float, float] = (0, float("inf"))
    ) -> Tuple[float, float]:
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    @staticmethod
    def __generate_emboss_matrix(alpha_sample: np.ndarray, strength_sample: np.ndarray) -> np.ndarray:
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [
                [-1 - strength_sample, 0 - strength_sample, 0],
                [0 - strength_sample, 1, 0 + strength_sample],
                [0, 0 + strength_sample, 1 + strength_sample],
            ],
            dtype=np.float32,
        )
        return (1 - alpha_sample) * matrix_nochange + alpha_sample * matrix_effect

    def get_params(self) -> Dict[str, np.ndarray]:
        alpha = random.uniform(*self.alpha)
        strength = random.uniform(*self.strength)
        emboss_matrix = self.__generate_emboss_matrix(alpha_sample=alpha, strength_sample=strength)
        return {"emboss_matrix": emboss_matrix}

    def apply(self, img: np.ndarray, emboss_matrix: Optional[np.ndarray] = None, **params: Any) -> np.ndarray:
        return F.convolve(img, emboss_matrix)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
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
        p_replace: ScaleFloatType = 0.1,
        n_segments: ScaleIntType = 100,
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

    def get_params(self) -> Dict[str, Any]:
        n_segments = random.randint(*self.n_segments)
        p = random.uniform(*self.p_replace)
        return {"replace_samples": random_utils.random(n_segments) < p, "n_segments": n_segments}

    def apply(
        self, img: np.ndarray, replace_samples: Sequence[bool] = (False,), n_segments: int = 1, **kwargs: Any
    ) -> np.ndarray:
        return F.superpixels(img, n_segments, replace_samples, self.max_size, cast(int, self.interpolation))


class TemplateTransform(ImageOnlyTransform):
    """Apply blending of input image with specified templates
    Args:
        templates (numpy array or list of numpy arrays): Images as template for transform.
        img_weight: If single float will be used as weight for input image.
            If tuple of float img_weight will be in range `[img_weight[0], img_weight[1])`. Default: 0.5.
        template_weight: If single float will be used as weight for template.
            If tuple of float template_weight will be in range `[template_weight[0], template_weight[1])`.
            Default: 0.5.
        template_transform: transformation object which could be applied to template,
            must produce template the same size as input image.
        name: (Optional) Name of transform, used only for deserialization.
        p: probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        templates: Union[np.ndarray, List[np.ndarray]],
        img_weight: ScaleFloatType = 0.5,
        template_weight: ScaleFloatType = 0.5,
        template_transform: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        self.templates = templates if isinstance(templates, (list, tuple)) else [templates]
        self.img_weight = to_tuple(img_weight, img_weight)
        self.template_weight = to_tuple(template_weight, template_weight)
        self.template_transform = template_transform
        self.name = name

    def apply(
        self,
        img: np.ndarray,
        template: Optional[np.ndarray] = None,
        img_weight: float = 0.5,
        template_weight: float = 0.5,
        **params: Any,
    ) -> np.ndarray:
        return F.add_weighted(img, img_weight, template, template_weight)

    def get_params(self) -> Dict[str, float]:
        return {
            "img_weight": random.uniform(self.img_weight[0], self.img_weight[1]),
            "template_weight": random.uniform(self.template_weight[0], self.template_weight[1]),
        }

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        img = params["image"]
        template = random.choice(self.templates)

        if self.template_transform is not None:
            template = self.template_transform(image=template)["image"]

        if get_num_channels(template) not in [1, get_num_channels(img)]:
            msg = (
                "Template must be a single channel or "
                "has the same number of channels as input "
                f"image ({get_num_channels(img)}), got {get_num_channels(template)}"
            )
            raise ValueError(msg)

        if template.dtype != img.dtype:
            msg = "Image and template must be the same image type"
            raise ValueError(msg)

        if img.shape[:2] != template.shape[:2]:
            raise ValueError(f"Image and template must be the same size, got {img.shape[:2]} and {template.shape[:2]}")

        if get_num_channels(template) == 1 and get_num_channels(img) > 1:
            template = np.stack((template,) * get_num_channels(img), axis=-1)

        # in order to support grayscale image with dummy dim
        template = template.reshape(img.shape)

        return {"template": template}

    @classmethod
    def is_serializable(cls) -> bool:
        return False

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def to_dict_private(self) -> Dict[str, Any]:
        if self.name is None:
            msg = (
                "To make a TemplateTransform serializable you should provide the `name` argument, "
                "e.g. `TemplateTransform(name='my_transform', ...)`."
            )
            raise ValueError(msg)
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}


class RingingOvershoot(ImageOnlyTransform):
    """Create ringing or overshoot artefacts by conlvolving image with 2D sinc filter.

    Args:
        blur_limit: maximum kernel size for sinc filter.
            Should be in range [3, inf). Default: (7, 15).
        cutoff: range to choose the cutoff frequency in radians.
            Should be in range (0, np.pi)
            Default: (np.pi / 4, np.pi / 2).
        p: probability of applying the transform. Default: 0.5.

    Reference:
        dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
        https://arxiv.org/abs/2107.10833

    Targets:
        image

    """

    def __init__(
        self,
        blur_limit: ScaleIntType = (7, 15),
        cutoff: ScaleFloatType = (np.pi / 4, np.pi / 2),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.blur_limit = cast(Tuple[int, int], to_tuple(blur_limit, 3))
        self.cutoff = self.__check_values(to_tuple(cutoff, np.pi / 2), name="cutoff", bounds=(0, np.pi))

    @staticmethod
    def __check_values(
        value: Tuple[float, float], name: str, bounds: Tuple[float, float] = (0, float("inf"))
    ) -> Tuple[float, float]:
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def get_params(self) -> Dict[str, np.ndarray]:
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

    def apply(self, img: np.ndarray, kernel: Optional[int] = None, **params: Any) -> np.ndarray:
        return F.convolve(img, kernel)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("blur_limit", "cutoff")


class UnsharpMask(ImageOnlyTransform):
    """Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.

    Args:
        blur_limit: maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit: Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        alpha: range to choose the visibility of the sharpened image.
            At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
            Default: (0.2, 0.5).
        threshold: Value to limit sharpening only for areas with high pixel difference between original image
            and it's smoothed version. Higher threshold means less sharpening on flat areas.
            Must be in range [0, 255]. Default: 10.
        p: probability of applying the transform. Default: 0.5.

    Reference:
        arxiv.org/pdf/2107.10833.pdf

    Targets:
        image

    """

    def __init__(
        self,
        blur_limit: ScaleIntType = (3, 7),
        sigma_limit: ScaleFloatType = 0.0,
        alpha: ScaleFloatType = (0.2, 0.5),
        threshold: int = 10,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.blur_limit = cast(Tuple[int, int], to_tuple(blur_limit, 3))
        self.sigma_limit = self.__check_values(to_tuple(sigma_limit, 0.0), name="sigma_limit")
        self.alpha = self.__check_values(to_tuple(alpha, 0.0), name="alpha", bounds=(0.0, 1.0))
        self.threshold = threshold

        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            msg = "blur_limit and sigma_limit minimum value can not be both equal to 0."
            raise ValueError(msg)

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            msg = "UnsharpMask supports only odd blur limits."
            raise ValueError(msg)

    @staticmethod
    def __check_values(
        value: Union[Tuple[int, int], Tuple[float, float]], name: str, bounds: Tuple[float, float] = (0, float("inf"))
    ) -> Tuple[float, float]:
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def get_params(self) -> Dict[str, Any]:
        return {
            "ksize": random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2),
            "sigma": random.uniform(*self.sigma_limit),
            "alpha": random.uniform(*self.alpha),
        }

    def apply(self, img: np.ndarray, ksize: int = 3, sigma: int = 0, alpha: float = 0.2, **params: Any) -> np.ndarray:
        return F.unsharp_mask(img, ksize, sigma=sigma, alpha=alpha, threshold=self.threshold)

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return "blur_limit", "sigma_limit", "alpha", "threshold"


class PixelDropout(DualTransform):
    """Set pixels to 0 with some probability.

    Args:
        dropout_prob (float): pixel drop probability. Default: 0.01
        per_channel (bool): if set to `True` drop mask will be sampled for each channel,
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

    _targets = (Targets.IMAGE, Targets.MASK)

    def __init__(
        self,
        dropout_prob: float = 0.01,
        per_channel: bool = False,
        drop_value: Optional[ScaleFloatType] = 0,
        mask_drop_value: Optional[ScaleFloatType] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.drop_value = drop_value
        self.mask_drop_value = mask_drop_value

        if self.mask_drop_value is not None and self.per_channel:
            msg = "PixelDropout supports mask only with per_channel=False"
            raise ValueError(msg)

    def apply(
        self,
        img: np.ndarray,
        drop_mask: Optional[np.ndarray] = None,
        drop_value: Union[float, Sequence[float]] = (),
        **params: Any,
    ) -> np.ndarray:
        return F.pixel_dropout(img, drop_mask, drop_value)

    def apply_to_mask(self, mask: np.ndarray, drop_mask: Optional[np.ndarray] = None, **params: Any) -> np.ndarray:
        if self.mask_drop_value is None:
            return mask

        if mask.ndim == TWO:
            drop_mask = np.squeeze(drop_mask)

        return F.pixel_dropout(mask, drop_mask, self.mask_drop_value)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        return bbox

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
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
                drop_value = rnd.uniform(0, 1, drop_shape).astype(img.dtype)
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
    """Apply spatter transform. It simulates corruption which can occlude a lens in the form of rain or mud.

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
        color (list of (r, g, b) or dict or None): Corruption elements color.
            If list uses provided list as color for specified mode.
            If dict uses provided color for specified mode. Color for each specified mode should be provided in dict.
            If None uses default colors (rain: (238, 238, 175), mud: (20, 42, 63)).
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
        mode: Union[SpatterMode, Sequence[SpatterMode]] = "rain",
        color: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        self.mean = to_tuple(mean, mean)
        self.std = to_tuple(std, std)
        self.gauss_sigma = to_tuple(gauss_sigma, gauss_sigma)
        self.intensity = to_tuple(intensity, intensity)
        self.cutout_threshold = to_tuple(cutout_threshold, cutout_threshold)
        self.color = (
            color
            if color is not None
            else {
                "rain": [238, 238, 175],
                "mud": [20, 42, 63],
            }
        )
        self.mode = mode if isinstance(mode, (list, tuple)) else [mode]

        if len(set(self.mode)) > 1 and not isinstance(self.color, dict):
            raise ValueError(f"Unsupported color: {self.color}. Please specify color for each mode (use dict for it).")

        for i in self.mode:
            if i not in ["rain", "mud"]:
                raise ValueError(f"Unsupported color mode: {mode}. Transform supports only `rain` and `mud` mods.")
            if isinstance(self.color, dict):
                if i not in self.color:
                    raise ValueError(f"Wrong color definition: {self.color}. Color for mode: {i} not specified.")
                if len(self.color[i]) != THREE:
                    raise ValueError(
                        f"Unsupported color: {self.color[i]} for mode {i}. Color should be presented in RGB format."
                    )

        if isinstance(self.color, (list, tuple)):
            if len(self.color) != THREE:
                raise ValueError(f"Unsupported color: {self.color}. Color should be presented in RGB format.")
            self.color = {self.mode[0]: self.color}

    def apply(
        self,
        img: np.ndarray,
        non_mud: Optional[np.ndarray] = None,
        mud: Optional[np.ndarray] = None,
        drops: Optional[np.ndarray] = None,
        mode: SpatterMode = "mud",
        **params: Dict[str, Any],
    ) -> np.ndarray:
        return F.spatter(img, non_mud, mud, drops, mode)

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        height, width = params["image"].shape[:2]

        mean = random.uniform(self.mean[0], self.mean[1])
        std = random.uniform(self.std[0], self.std[1])
        cutout_threshold = random.uniform(self.cutout_threshold[0], self.cutout_threshold[1])
        sigma = random.uniform(self.gauss_sigma[0], self.gauss_sigma[1])
        mode = random.choice(self.mode)
        intensity = random.uniform(self.intensity[0], self.intensity[1])
        color = np.array(self.color[mode]) / 255.0

        liquid_layer = random_utils.normal(size=(height, width), loc=mean, scale=std)
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

            drops = m[:, :, None] * color * intensity
            mud = None
            non_mud = None
        else:
            m = np.where(liquid_layer > cutout_threshold, 1, 0)
            m = gaussian_filter(m.astype(np.float32), sigma=sigma, mode="nearest")
            m[m < 1.2 * cutout_threshold] = 0
            m = m[..., np.newaxis]

            mud = m * color
            non_mud = 1 - m
            drops = None

        return {
            "non_mud": non_mud,
            "mud": mud,
            "drops": drops,
            "mode": mode,
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str, str, str, str]:
        return "mean", "std", "gauss_sigma", "intensity", "cutout_threshold", "mode", "color"


class ChromaticAberration(ImageOnlyTransform):
    """Add lateral chromatic aberration by distorting the red and blue channels of the input image.

    Args:
        primary_distortion_limit: range of the primary radial distortion coefficient.
            If primary_distortion_limit is a single float value, the range will be
            (-primary_distortion_limit, primary_distortion_limit).
            Controls the distortion in the center of the image (positive values result in pincushion distortion,
            negative values result in barrel distortion).
            Default: 0.02.
        secondary_distortion_limit: range of the secondary radial distortion coefficient.
            If secondary_distortion_limit is a single float value, the range will be
            (-secondary_distortion_limit, secondary_distortion_limit).
            Controls the distortion in the corners of the image (positive values result in pincushion distortion,
            negative values result in barrel distortion).
            Default: 0.05.
        mode: type of color fringing.
            Supported modes are 'green_purple', 'red_blue' and 'random'.
            'random' will choose one of the modes 'green_purple' or 'red_blue' randomly.
            Default: 'green_purple'.
        interpolation: flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p: probability of applying the transform.
            Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        primary_distortion_limit: ScaleFloatType = 0.02,
        secondary_distortion_limit: ScaleFloatType = 0.05,
        mode: ChromaticAberrationMode = "green_purple",
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.primary_distortion_limit = to_tuple(primary_distortion_limit)
        self.secondary_distortion_limit = to_tuple(secondary_distortion_limit)
        self.mode = self._validate_mode(mode)
        self.interpolation = interpolation

    @staticmethod
    def _validate_mode(
        mode: ChromaticAberrationMode,
    ) -> ChromaticAberrationMode:
        valid_modes = ["green_purple", "red_blue", "random"]
        if mode not in valid_modes:
            msg = f"Unsupported mode: {mode}. Supported modes are 'green_purple', 'red_blue', 'random'."
            raise ValueError(msg)
        return mode

    def apply(
        self,
        img: np.ndarray,
        primary_distortion_red: float = -0.02,
        secondary_distortion_red: float = -0.05,
        primary_distortion_blue: float = -0.02,
        secondary_distortion_blue: float = -0.05,
        **params: Any,
    ) -> np.ndarray:
        return F.chromatic_aberration(
            img,
            primary_distortion_red,
            secondary_distortion_red,
            primary_distortion_blue,
            secondary_distortion_blue,
            cast(int, self.interpolation),
        )

    def get_params(self) -> Dict[str, float]:
        primary_distortion_red = random_utils.uniform(*self.primary_distortion_limit)
        secondary_distortion_red = random_utils.uniform(*self.secondary_distortion_limit)
        primary_distortion_blue = random_utils.uniform(*self.primary_distortion_limit)
        secondary_distortion_blue = random_utils.uniform(*self.secondary_distortion_limit)

        secondary_distortion_red = self._match_sign(primary_distortion_red, secondary_distortion_red)
        secondary_distortion_blue = self._match_sign(primary_distortion_blue, secondary_distortion_blue)

        if self.mode == "green_purple":
            # distortion coefficients of the red and blue channels have the same sign
            primary_distortion_blue = self._match_sign(primary_distortion_red, primary_distortion_blue)
            secondary_distortion_blue = self._match_sign(secondary_distortion_red, secondary_distortion_blue)
        if self.mode == "red_blue":
            # distortion coefficients of the red and blue channels have the opposite sign
            primary_distortion_blue = self._unmatch_sign(primary_distortion_red, primary_distortion_blue)
            secondary_distortion_blue = self._unmatch_sign(secondary_distortion_red, secondary_distortion_blue)

        return {
            "primary_distortion_red": primary_distortion_red,
            "secondary_distortion_red": secondary_distortion_red,
            "primary_distortion_blue": primary_distortion_blue,
            "secondary_distortion_blue": secondary_distortion_blue,
        }

    @staticmethod
    def _match_sign(a: float, b: float) -> float:
        # Match the sign of b to a
        if (a < 0 < b) or (a > 0 > b):
            b = -b
        return b

    @staticmethod
    def _unmatch_sign(a: float, b: float) -> float:
        # Unmatch the sign of b to a
        if (a < 0 and b < 0) or (a > 0 and b > 0):
            b = -b
        return b

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return "primary_distortion_limit", "secondary_distortion_limit", "mode", "interpolation"


class Morphological(DualTransform):
    """Apply a morphological operation (dilation or erosion) to an image,
    with particular value for enhancing document scans.

    Morphological operations modify the structure of the image.
    Dilation expands the white (foreground) regions in a binary or grayscale image, while erosion shrinks them.
    These operations are beneficial in document processing, for example:
    - Dilation helps in closing up gaps within text or making thin lines thicker,
        enhancing legibility for OCR (Optical Character Recognition).
    - Erosion can remove small white noise and detach connected objects,
        making the structure of larger objects more pronounced.

    Args:
        scale (int or tuple/list of int): Specifies the size of the structuring element (kernel) used for the operation.
            - If an integer is provided, a square kernel of that size will be used.
            - If a tuple or list is provided, it should contain two integers representing the minimum
                and maximum sizes for the dilation kernel.
        operation (str, optional): The morphological operation to apply. Options are 'dilation' or 'erosion'.
            Default is 'dilation'.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
        https://github.com/facebookresearch/nougat

    Example:
        >>> import albumentations as A
        >>> transform = A.Compose([
        >>>     A.Morphological(scale=(2, 3), operation='dilation', p=0.5)
        >>> ])
        >>> image = transform(image=image)["image"]
    """

    _targets = (Targets.IMAGE, Targets.MASK)

    def __init__(
        self,
        scale: ScaleIntType = (2, 3),
        operation: MorphologyMode = "dilation",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.scale = to_tuple(scale, scale)
        self.operation = operation

    def apply(self, img: np.ndarray, kernel: Tuple[int, int], **params: Any) -> np.ndarray:
        return F.morphology(img, kernel, self.operation)

    def apply_to_mask(self, mask: np.ndarray, kernel: Tuple[int, int], **params: Any) -> np.ndarray:
        return F.morphology(mask, kernel, self.operation)

    def get_params(self) -> Dict[str, float]:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(random_utils.randint(self.scale[0], self.scale[1], 2))
        )
        return {
            "kernel": kernel,
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("scale", "operation")
