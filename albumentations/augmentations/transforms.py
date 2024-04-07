import math
import numbers
import random
import warnings
from types import LambdaType
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import cv2
import numpy as np
from pydantic import Field, ValidationInfo, field_validator, model_validator
from scipy import special
from scipy.ndimage import gaussian_filter
from typing_extensions import Annotated, Self

from albumentations import random_utils
from albumentations.augmentations.blur.functional import blur
from albumentations.augmentations.blur.transforms import BlurInitSchema, process_blur_limit
from albumentations.augmentations.functional import split_uniform_grid
from albumentations.augmentations.utils import (
    check_range,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
)
from albumentations.core.pydantic import (
    InterpolationType,
    OnePlusRangeType,
    ProbabilityType,
    RangeNonNegativeType,
    RangeSymmetricType,
    ZeroOneRangeType,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
    ImageOnlyTransform,
    Interpolation,
    NoOp,
)
from albumentations.core.types import (
    BoxInternalType,
    ChromaticAberrationMode,
    ColorType,
    ImageCompressionType,
    ImageMode,
    KeypointInternalType,
    RainMode,
    ScaleFloatType,
    ScaleIntType,
    ScaleType,
    SpatterMode,
    Targets,
)
from albumentations.core.utils import format_args, to_tuple

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
]

NUM_BITS_ARRAY_LENGTH = 3
GRAYSCALE_SHAPE_LEN = 2
NUM_RGB_CHANNELS = 3
MAX_JPEG_QUALITY = 100
TWENTY = 20
PAIR = 2


class RandomGridShuffle(DualTransform):
    """Random shuffle grid's cells on image.

    Args:
        grid ((int, int)): size of grid for splitting image.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32

    """

    class InitSchema(BaseTransformInitSchema):
        grid: OnePlusRangeType = (3, 3)

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS)

    def __init__(self, grid: Tuple[int, int] = (3, 3), always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.grid = grid

    def apply(self, img: np.ndarray, tiles: Optional[np.ndarray] = None, **params: Any) -> np.ndarray:
        return F.swap_tiles_on_image(img, tiles)

    def apply_to_mask(self, mask: np.ndarray, tiles: Optional[np.ndarray] = None, **params: Any) -> np.ndarray:
        return F.swap_tiles_on_image(mask, tiles)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        tiles: np.ndarray,
        mapping: Dict[int, int],
        **params: Any,
    ) -> KeypointInternalType:
        x, y = keypoint[:2]

        # Find which original tile the keypoint belongs to
        for original_index, (start_y, start_x, end_y, end_x) in enumerate(tiles):
            if start_y <= y < end_y and start_x <= x < end_x:
                # Find this tile's new index after shuffling
                new_index = mapping[original_index]
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
        # Generate the original grid
        original_tiles = split_uniform_grid(params["image"].shape[:2], self.grid)

        # Copy the original grid to keep track of the initial positions
        indexed_tiles = np.array(list(enumerate(original_tiles)), dtype=object)

        # Shuffle the tiles while keeping track of original indices
        random_utils.shuffle(indexed_tiles)

        # Create a mapping from original positions to new positions
        mapping = {original_index: i for i, (original_index, tile) in enumerate(indexed_tiles)}

        # Extract the shuffled tiles without indices
        shuffled_tiles = np.array([tile for _, tile in indexed_tiles])

        return {"tiles": shuffled_tiles, "mapping": mapping}

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

    class InitSchema(BaseTransformInitSchema):
        mean: ColorType = Field(default=(0.485, 0.456, 0.406), description="Mean values for normalization")
        std: ColorType = Field(default=(0.229, 0.224, 0.225), description="Standard deviation values for normalization")
        max_pixel_value: float = Field(default=255.0, description="Maximum possible pixel value")
        p: ProbabilityType = 1

    def __init__(
        self,
        mean: ColorType = (0.485, 0.456, 0.406),
        std: ColorType = (0.229, 0.224, 0.225),
        max_pixel_value: float = 255.0,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
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

    class InitSchema(BaseTransformInitSchema):
        quality_lower: int = Field(default=99, description="Lower bound on the image quality", ge=1, le=100)
        quality_upper: int = Field(default=100, description="Upper bound on the image quality", ge=1, le=100)
        compression_type: ImageCompressionType = Field(
            default=ImageCompressionType.JPEG, description="Image compression format"
        )

        @model_validator(mode="after")
        def validate_quality(self) -> Self:
            if self.quality_lower >= self.quality_upper:
                msg = "quality_lower must be less than quality_upper"
                raise ValueError(msg)
            return self

    def __init__(
        self,
        quality_lower: int = 99,
        quality_upper: int = 100,
        compression_type: ImageCompressionType = ImageCompressionType.JPEG,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.compression_type = compression_type

    def apply(self, img: np.ndarray, quality: int = 100, image_type: str = ".jpg", **params: Any) -> np.ndarray:
        if img.ndim != GRAYSCALE_SHAPE_LEN and img.shape[-1] not in (1, 3, 4):
            msg = "ImageCompression transformation expects 1, 3 or 4 channel images."
            raise TypeError(msg)
        return F.image_compression(img, quality, image_type)

    def get_params(self) -> Dict[str, Any]:
        image_type = ".jpg"

        if self.compression_type == ImageCompressionType.WEBP:
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

    Args:
        snow_point_lower: lower_bond of the amount of snow. Should be in [0, 1] range
        snow_point_upper: upper_bond of the amount of snow. Should be in [0, 1] range
        brightness_coeff: larger number will lead to a more snow on the image. Should be >= 0

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """

    class InitSchema(BaseTransformInitSchema):
        snow_point_lower: float = Field(default=0.1, description="Lower bound of the amount of snow", ge=0, le=1)
        snow_point_upper: float = Field(default=0.3, description="Upper bound of the amount of snow", ge=0, le=1)
        brightness_coeff: float = Field(default=2.5, description="Brightness coefficient, must be >= 0", ge=0)

        @model_validator(mode="after")
        def validate_snow_points(self) -> Self:
            if self.snow_point_lower > self.snow_point_upper:
                msg = "snow_point_lower must be less than or equal to snow_point_upper."
                raise ValueError(msg)
            return self

    def __init__(
        self,
        snow_point_lower: float = 0.1,
        snow_point_upper: float = 0.3,
        brightness_coeff: float = 2.5,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)

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

    Args:
        gravel_roi: (top-left x, top-left y,
            bottom-right x, bottom right y). Should be in [0, 1] range
        number_of_patches: no. of gravel patches required

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """

    class InitSchema(BaseTransformInitSchema):
        gravel_roi: Tuple[float, float, float, float] = Field(
            default=(0.1, 0.4, 0.9, 0.9), description="Region of interest for gravel placement"
        )
        number_of_patches: int = Field(default=2, description="Number of gravel patches", ge=1)

        @model_validator(mode="after")
        def validate_gravel_roi(self) -> Self:
            gravel_lower_x, gravel_lower_y, gravel_upper_x, gravel_upper_y = self.gravel_roi
            if not 0 <= gravel_lower_x < gravel_upper_x <= 1 or not 0 <= gravel_lower_y < gravel_upper_y <= 1:
                raise ValueError(f"Invalid gravel_roi. Got: {self.gravel_roi}.")
            return self

    def __init__(
        self,
        gravel_roi: Tuple[float, float, float, float] = (0.1, 0.4, 0.9, 0.9),
        number_of_patches: int = 2,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
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

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """

    class InitSchema(BaseTransformInitSchema):
        slant_lower: int = Field(default=-10, description="Lower bound for rain slant angle", ge=-20, le=20)
        slant_upper: int = Field(default=10, description="Upper bound for rain slant angle", ge=-20, le=20)
        drop_length: int = Field(default=20, description="Length of raindrops", ge=0, le=100)
        drop_width: int = Field(default=1, description="Width of raindrops", ge=1, le=5)
        drop_color: Tuple[int, int, int] = Field(default=(200, 200, 200), description="Color of raindrops")
        blur_value: int = Field(default=7, description="Blur value for simulating rain effect", ge=0)
        brightness_coefficient: float = Field(
            default=0.7, description="Brightness coefficient for rainy effect", ge=0, le=1
        )
        rain_type: Optional[RainMode] = Field(default=None, description="Type of rain to simulate")

        @model_validator(mode="after")
        def validate_slant_range_and_rain_type(self) -> Self:
            if self.slant_lower >= self.slant_upper:
                msg = "slant_upper must be greater than or equal to slant_lower."
                raise ValueError(msg)
            if self.rain_type not in ["drizzle", "heavy", "torrential", None]:
                raise ValueError(
                    f"rain_type must be one of ['drizzle', 'heavy', 'torrential', None]. Got: {self.rain_type}"
                )
            return self

    def __init__(
        self,
        slant_lower: int = -10,
        slant_upper: int = 10,
        drop_length: int = 20,
        drop_width: int = 1,
        drop_color: Tuple[int, int, int] = (200, 200, 200),
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        rain_type: Optional[RainMode] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
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

    Args:
        fog_coef_lower: lower limit for fog intensity coefficient. Should be in [0, 1] range.
        fog_coef_upper: upper limit for fog intensity coefficient. Should be in [0, 1] range.
        alpha_coef: transparency of the fog circles. Should be in [0, 1] range.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

    """

    class InitSchema(BaseTransformInitSchema):
        fog_coef_lower: float = Field(default=0.3, description="Lower limit for fog intensity coefficient", ge=0, le=1)
        fog_coef_upper: float = Field(default=1, description="Upper limit for fog intensity coefficient", ge=0, le=1)
        alpha_coef: float = Field(default=0.08, description="Transparency of the fog circles", ge=0, le=1)

        @model_validator(mode="after")
        def validate_fog_coefficients(self) -> Self:
            if self.fog_coef_lower > self.fog_coef_upper:
                msg = "fog_coef_upper must be greater than or equal to fog_coef_lower."
                raise ValueError(msg)
            return self

    def __init__(
        self,
        fog_coef_lower: float = 0.3,
        fog_coef_upper: float = 1,
        alpha_coef: float = 0.08,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
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

    class InitSchema(BaseTransformInitSchema):
        flare_roi: Tuple[float, float, float, float] = Field(
            default=(0, 0, 1, 0.5), description="Region of the image where flare will appear"
        )
        angle_lower: float = Field(default=0, description="Lower bound for the angle", ge=0, le=1)
        angle_upper: float = Field(default=1, description="Upper bound for the angle", ge=0, le=1)
        num_flare_circles_lower: int = Field(default=6, description="Lower limit for the number of flare circles", ge=0)
        num_flare_circles_upper: int = Field(
            default=10, description="Upper limit for the number of flare circles", gt=0
        )
        src_radius: int = Field(default=400, description="Source radius for the flare")
        src_color: Tuple[int, int, int] = Field(default=(255, 255, 255), description="Color of the flare")

        @model_validator(mode="after")
        def validate_parameters(self) -> Self:
            flare_center_lower_x, flare_center_lower_y, flare_center_upper_x, flare_center_upper_y = self.flare_roi
            if (
                not 0 <= flare_center_lower_x < flare_center_upper_x <= 1
                or not 0 <= flare_center_lower_y < flare_center_upper_y <= 1
            ):
                raise ValueError(f"Invalid flare_roi. Got: {self.flare_roi}")
            if self.angle_lower >= self.angle_upper:
                raise ValueError(
                    f"angle_upper must be greater than angle_lower. Got: {self.angle_lower}, {self.angle_upper}"
                )
            if self.num_flare_circles_lower >= self.num_flare_circles_upper:
                msg = "num_flare_circles_upper must be greater than num_flare_circles_lower."
                raise ValueError(msg)
            return self

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
        super().__init__(always_apply=always_apply, p=p)

        self.angle_lower = angle_lower
        self.angle_upper = angle_upper
        self.num_flare_circles_lower = num_flare_circles_lower
        self.num_flare_circles_upper = num_flare_circles_upper
        self.src_radius = src_radius
        self.src_color = src_color
        self.flare_roi = flare_roi

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

        (flare_center_lower_x, flare_center_lower_y, flare_center_upper_x, flare_center_upper_y) = self.flare_roi

        flare_center_x = random.uniform(flare_center_lower_x, flare_center_upper_x)
        flare_center_y = random.uniform(flare_center_lower_y, flare_center_upper_y)

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
            "flare_roi": self.flare_roi,
            "angle_lower": self.angle_lower,
            "angle_upper": self.angle_upper,
            "num_flare_circles_lower": self.num_flare_circles_lower,
            "num_flare_circles_upper": self.num_flare_circles_upper,
            "src_radius": self.src_radius,
            "src_color": self.src_color,
        }


class RandomShadow(ImageOnlyTransform):
    """Simulates shadows for the image

    Args:
        shadow_roi: region of the image where shadows
            will appear. All values should be in range [0, 1].
        num_shadows_lower: Lower limit for the possible number of shadows.
            Should be in range [0, `num_shadows_upper`].
        num_shadows_upper: Lower limit for the possible number of shadows.
            Should be in range [`num_shadows_lower`, inf].
        shadow_dimension: number of edges in the shadow polygons

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
    """

    class InitSchema(BaseTransformInitSchema):
        shadow_roi: Tuple[float, float, float, float] = Field(
            default=(0, 0.5, 1, 1), description="Region of the image where shadows will appear"
        )
        num_shadows_lower: int = Field(default=1, description="Lower limit for the possible number of shadows", ge=0)
        num_shadows_upper: int = Field(default=2, description="Upper limit for the possible number of shadows", ge=0)
        shadow_dimension: int = Field(default=5, description="Number of edges in the shadow polygons", gt=0)

        @model_validator(mode="after")
        def validate_shadows(self) -> Self:
            shadow_lower_x, shadow_lower_y, shadow_upper_x, shadow_upper_y = self.shadow_roi
            if not 0 <= shadow_lower_x <= shadow_upper_x <= 1 or not 0 <= shadow_lower_y <= shadow_upper_y <= 1:
                raise ValueError(f"Invalid shadow_roi. Got: {self.shadow_roi}")
            if self.num_shadows_lower > self.num_shadows_upper:
                msg = "num_shadows_upper must be greater than or equal to num_shadows_lower."
                raise ValueError(msg)
            return self

    def __init__(
        self,
        shadow_roi: Tuple[float, float, float, float] = (0, 0.5, 1, 1),
        num_shadows_lower: int = 1,
        num_shadows_upper: int = 2,
        shadow_dimension: int = 5,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
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

        num_shadows = random.randint(self.num_shadows_lower, self.num_shadows_upper)

        x_min, y_min, x_max, y_max = self.shadow_roi

        x_min = int(x_min * width)
        x_max = int(x_max * width)
        y_min = int(y_min * height)
        y_max = int(y_max * height)

        vertices_list = []

        for _ in range(num_shadows):
            vertex = [
                (random.randint(x_min, x_max), random.randint(y_min, y_max)) for _ in range(self.shadow_dimension)
            ]

            vertices = np.array([vertex], dtype=np.int32)
            vertices_list.append(vertices)

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

    class InitSchema(BaseTransformInitSchema):
        scale: float = Field(
            default=0.1,
            description="Standard deviation of the normal distribution used to sample random distances",
            ge=0,
            le=1,
        )

    def __init__(
        self,
        scale: float = 0.1,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
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

    class InitSchema(BaseTransformInitSchema):
        hue_shift_limit: RangeSymmetricType = (-20, 20)
        sat_shift_limit: RangeSymmetricType = (-30, 30)
        val_shift_limit: RangeSymmetricType = (-20, 20)

    def __init__(
        self,
        hue_shift_limit: ScaleFloatType = 20,
        sat_shift_limit: ScaleFloatType = 30,
        val_shift_limit: ScaleFloatType = 20,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.hue_shift_limit = cast(Tuple[float, float], hue_shift_limit)
        self.sat_shift_limit = cast(Tuple[float, float], sat_shift_limit)
        self.val_shift_limit = cast(Tuple[float, float], val_shift_limit)

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

    class InitSchema(BaseTransformInitSchema):
        threshold: Annotated[ScaleType, Field(default=(128, 128), description="Range for solarizing threshold.")]

        @field_validator("threshold")
        @classmethod
        def convert_to_tuple(cls, threshold: Any) -> Union[Tuple[int, int], Tuple[float, float]]:
            if isinstance(threshold, (int, float)):
                return to_tuple(threshold, low=threshold)

            return to_tuple(threshold)

    def __init__(self, threshold: ScaleType = 128, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.threshold = cast(Tuple[float, float], threshold)

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

    class InitSchema(BaseTransformInitSchema):
        num_bits: Annotated[
            Union[int, Tuple[int, int], Tuple[int, int, int]], Field(default=4, description="Number of high bits")
        ]

        @field_validator("num_bits")
        @classmethod
        def validate_num_bits(cls, num_bits: Any) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
            if isinstance(num_bits, int):
                return cast(Tuple[int, int], to_tuple(num_bits, num_bits))
            if isinstance(num_bits, Sequence) and len(num_bits) == NUM_BITS_ARRAY_LENGTH:
                return [cast(Tuple[int, int], to_tuple(i, 0)) for i in num_bits]
            return cast(Tuple[int, int], to_tuple(num_bits, 0))

    def __init__(
        self,
        num_bits: Union[int, Tuple[int, int], Tuple[int, int, int]] = 4,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.num_bits = cast(Union[Tuple[int, ...], List[Tuple[int, ...]]], num_bits)

    def apply(self, img: np.ndarray, num_bits: int = 1, **params: Any) -> np.ndarray:
        return F.posterize(img, num_bits)

    def get_params(self) -> Dict[str, Any]:
        if len(self.num_bits) == NUM_BITS_ARRAY_LENGTH:
            return {"num_bits": [random.randint(int(i[0]), int(i[1])) for i in self.num_bits]}  # type: ignore[index]
        num_bits = self.num_bits
        return {"num_bits": random.randint(int(num_bits[0]), int(num_bits[1]))}  # type: ignore[arg-type]

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

    class InitSchema(BaseTransformInitSchema):
        mode: ImageMode = "cv"
        by_channels: Annotated[bool, Field(default=True, description="Equalize channels separately if True")]
        mask: Annotated[
            Optional[Union[np.ndarray, Callable[..., Any]]],
            Field(default=None, description="Mask to apply for equalization"),
        ]
        mask_params: Annotated[Sequence[str], Field(default=[], description="Parameters for mask function")]

    def __init__(
        self,
        mode: ImageMode = "cv",
        by_channels: bool = True,
        mask: Optional[Union[np.ndarray, Callable[..., Any]]] = None,
        mask_params: Sequence[str] = (),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

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

    class InitSchema(BaseTransformInitSchema):
        r_shift_limit: RangeSymmetricType = (-20, 20)
        g_shift_limit: RangeSymmetricType = (-20, 20)
        b_shift_limit: RangeSymmetricType = (-20, 20)

    def __init__(
        self,
        r_shift_limit: ScaleType = (-20, 20),
        g_shift_limit: ScaleType = (-20, 20),
        b_shift_limit: ScaleType = (-20, 20),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.r_shift_limit = cast(Tuple[float, float], r_shift_limit)
        self.g_shift_limit = cast(Tuple[float, float], g_shift_limit)
        self.b_shift_limit = cast(Tuple[float, float], b_shift_limit)

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

    class InitSchema(BaseTransformInitSchema):
        brightness_limit: RangeSymmetricType = (-0.2, 0.2)
        contrast_limit: RangeSymmetricType = (-0.2, 0.2)
        brightness_by_max: bool = Field(default=True, description="Adjust brightness by image dtype maximum if True.")

    def __init__(
        self,
        brightness_limit: ScaleFloatType = (-0.2, 0.2),
        contrast_limit: ScaleFloatType = (-0.2, 0.2),
        brightness_by_max: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.brightness_limit = cast(Tuple[float, float], brightness_limit)
        self.contrast_limit = cast(Tuple[float, float], contrast_limit)
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

    class InitSchema(BaseTransformInitSchema):
        var_limit: RangeNonNegativeType = (10.0, 50.0)
        mean: float = Field(default=0, description="Mean of the noise.")
        per_channel: bool = Field(default=True, description="Apply noise per channel.")

    def __init__(
        self,
        var_limit: ScaleType = (10.0, 50.0),
        mean: float = 0,
        per_channel: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.var_limit = cast(Tuple[float, float], var_limit)
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
            if len(image.shape) > GRAYSCALE_SHAPE_LEN:
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

    class InitSchema(BaseTransformInitSchema):
        color_shift: Tuple[float, float] = Field(
            default=(0.01, 0.05),
            description=(
                "Variance range for color hue change. Measured as a fraction of 360 degree Hue angle in HLS colorspace."
            ),
        )
        intensity: Tuple[float, float] = Field(
            default=(0.1, 0.5), description="Multiplicative factor that control strength of color and luminance noise."
        )

    def __init__(
        self,
        color_shift: Tuple[float, float] = (0.01, 0.05),
        intensity: Tuple[float, float] = (0.1, 0.5),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
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

    class InitSchema(BaseTransformInitSchema):
        clip_limit: OnePlusRangeType = (1.0, 4.0)
        tile_grid_size: OnePlusRangeType = (8, 8)

    def __init__(
        self,
        clip_limit: ScaleFloatType = (1, 4),
        tile_grid_size: Tuple[int, int] = (8, 8),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.clip_limit = cast(Tuple[float, float], clip_limit)
        self.tile_grid_size = tile_grid_size

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
    conditions, potentially enhancing model generalization.

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

    Reference:
         https://en.wikipedia.org/wiki/Gamma_correction

    """

    class InitSchema(BaseTransformInitSchema):
        gamma_limit: RangeSymmetricType = (80, 120)

    def __init__(
        self,
        gamma_limit: ScaleType = (80, 120),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.gamma_limit = cast(Tuple[float, float], gamma_limit)

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
        super().__init__(always_apply=always_apply, p=p)
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

    class InitSchema(BaseTransformInitSchema):
        max_value: Optional[float] = Field(default=None, description="Maximum possible input value.")
        p: ProbabilityType = 1

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

    class InitSchema(BaseTransformInitSchema):
        dtype: Literal["uint8", "uint16", "float32", "float64"] = "uint8"
        max_value: Optional[float] = Field(default=None, description="Maximum possible input value.")
        p: ProbabilityType = 1

    def __init__(
        self,
        dtype: Literal["uint8", "uint16", "float32", "float64"] = "uint16",
        max_value: Optional[float] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
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

    class InitSchema(BaseTransformInitSchema):
        scale_min: float = Field(default=0.25, ge=0, le=1, description="Lower bound on the image scale.")
        scale_max: float = Field(default=0.25, ge=0, lt=1, description="Upper bound on the image scale.")
        interpolation: Optional[Union[int, Interpolation, Dict[str, int]]] = Field(
            default_factory=lambda: Interpolation(downscale=cv2.INTER_NEAREST, upscale=cv2.INTER_NEAREST),
            description="CV2 interpolation method or a dictionary specifying downscale and upscale methods.",
        )

        @model_validator(mode="after")
        def validate_scale(self) -> Self:
            if self.scale_min > self.scale_max:
                msg = "scale_min must be less than or equal to scale_max"
                raise ValueError(msg)
            return self

        @field_validator("interpolation")
        @classmethod
        def set_interpolation(cls, v: Any) -> Interpolation:
            if isinstance(v, dict):
                return Interpolation(**v)
            if isinstance(v, int):
                return Interpolation(downscale=v, upscale=v)
            if isinstance(v, Interpolation):
                return v
            if v is None:
                return Interpolation(downscale=cv2.INTER_NEAREST, upscale=cv2.INTER_NEAREST)

            msg = (
                "Interpolation must be an int, Interpolation instance, "
                "or dict specifying downscale and upscale methods."
            )
            raise ValueError(msg)

    def __init__(
        self,
        scale_min: float = 0.25,
        scale_max: float = 0.25,
        interpolation: Optional[Union[int, Interpolation, Dict[str, int]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.interpolation = cast(Interpolation, interpolation)

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
        uint8, float32

    """

    class InitSchema(BaseTransformInitSchema):
        multiplier: RangeNonNegativeType = (0.9, 1.1)
        per_channel: bool = Field(default=False, description="Apply multiplier per channel.")
        elementwise: bool = Field(default=False, description="Apply multiplier element-wise to pixels.")

    def __init__(
        self,
        multiplier: ScaleFloatType = (0.9, 1.1),
        per_channel: bool = False,
        elementwise: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.multiplier = cast(Tuple[float, float], multiplier)
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
        if is_grayscale_image(img) and img.ndim == GRAYSCALE_SHAPE_LEN:
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

    Reference:
        http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
        https://pixelatedbrian.github.io/2018-04-29-fancy_pca/

    """

    class InitSchema(BaseTransformInitSchema):
        alpha: float = Field(default=0.1, description="Scale for perturbing the eigen vectors and values", ge=0)

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

    class InitSchema(BaseTransformInitSchema):
        brightness: Annotated[ScaleFloatType, Field(default=(0.8, 1.2), description="Range for jittering brightness.")]
        contrast: Annotated[ScaleFloatType, Field(default=(0.8, 1.2), description="Range for jittering contrast.")]
        saturation: Annotated[ScaleFloatType, Field(default=(0.8, 1.2), description="Range for jittering saturation.")]
        hue: Annotated[ScaleFloatType, Field(default=(-0.2, 0.2), description="Range for jittering hue.")]

        @field_validator("brightness", "contrast", "saturation", "hue")
        @classmethod
        def check_ranges(cls, value: ScaleFloatType, info: ValidationInfo) -> Tuple[float, float]:
            if info.field_name == "hue":
                bounds = -0.5, 0.5
                bias = 0
                clip = False
            elif info.field_name in ["brightness", "contrast", "saturation"]:
                bounds = 0, float("inf")
                bias = 1
                clip = True

            if isinstance(value, numbers.Number):
                if value < 0:
                    raise ValueError(f"If {info.field_name} is a single number, it must be non negative.")
                value = [bias - value, bias + value]
                if clip:
                    value[0] = max(value[0], 0)
            elif isinstance(value, (tuple, list)) and len(value) == PAIR:
                check_range(value, *bounds, info.field_name)

            return cast(Tuple[float, float], value)

    def __init__(
        self,
        brightness: ScaleFloatType = (0.8, 1.2),
        contrast: ScaleFloatType = (0.8, 1.2),
        saturation: ScaleFloatType = (0.8, 1.2),
        hue: ScaleFloatType = (-0.2, 0.2),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        self.brightness = cast(Tuple[float, float], brightness)
        self.contrast = cast(Tuple[float, float], contrast)
        self.saturation = cast(Tuple[float, float], saturation)
        self.hue = cast(Tuple[float, float], hue)

        self.transforms = [
            F.adjust_brightness_torchvision,
            F.adjust_contrast_torchvision,
            F.adjust_saturation_torchvision,
            F.adjust_hue_torchvision,
        ]

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

    class InitSchema(BaseTransformInitSchema):
        alpha: ZeroOneRangeType = (0.2, 0.5)
        lightness: RangeNonNegativeType = (0.5, 1.0)

    def __init__(
        self,
        alpha: Tuple[float, float] = (0.2, 0.5),
        lightness: Tuple[float, float] = (0.5, 1.0),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha
        self.lightness = lightness

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

    class InitSchema(BaseTransformInitSchema):
        alpha: ZeroOneRangeType = (0.2, 0.5)
        strength: RangeNonNegativeType = (0.2, 0.7)

    def __init__(
        self,
        alpha: Tuple[float, float] = (0.2, 0.5),
        strength: Tuple[float, float] = (0.2, 0.7),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha
        self.strength = strength

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
            Behavior based on chosen data types for this parameter:
                * If a ``float``, then that ``float`` will always be used.
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

    class InitSchema(BaseTransformInitSchema):
        p_replace: ZeroOneRangeType = (0.1, 0.1)
        n_segments: OnePlusRangeType = (100, 100)
        max_size: Optional[int] = Field(default=128, ge=1, description="Maximum image size for the transformation.")
        interpolation: InterpolationType

    def __init__(
        self,
        p_replace: ScaleFloatType = (0.1, 0.1),
        n_segments: ScaleIntType = (100, 100),
        max_size: Optional[int] = 128,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.p_replace = cast(Tuple[float, float], p_replace)
        self.n_segments = cast(Tuple[int, int], n_segments)
        self.max_size = max_size
        self.interpolation = interpolation

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

    class InitSchema(BaseTransformInitSchema):
        templates: Union[np.ndarray, Sequence[np.ndarray]] = Field(..., description="Images as template for transform.")
        img_weight: ZeroOneRangeType = (0.5, 0.5)
        template_weight: ZeroOneRangeType = (0.5, 0.5)
        template_transform: Optional[Callable[..., Any]] = Field(
            default=None, description="Transformation object applied to template."
        )
        name: Optional[str] = Field(default=None, description="Name of transform, used only for deserialization.")

        @field_validator("templates")
        @classmethod
        def validate_templates(cls, v: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
            if isinstance(v, np.ndarray):
                return [v]
            if isinstance(v, list):
                if not all(isinstance(item, np.ndarray) for item in v):
                    msg = "All templates must be numpy arrays."
                    raise ValueError(msg)
                return v
            msg = "Templates must be a numpy array or a list of numpy arrays."
            raise TypeError(msg)

    def __init__(
        self,
        templates: Union[np.ndarray, Sequence[np.ndarray]],
        img_weight: ScaleFloatType = (0.5, 0.5),
        template_weight: ScaleFloatType = (0.5, 0.5),
        template_transform: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.templates = templates
        self.img_weight = cast(Tuple[float, float], img_weight)
        self.template_weight = cast(Tuple[float, float], template_weight)
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
    """Create ringing or overshoot artifacts by conlvolving image with 2D sinc filter.

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

    class InitSchema(BlurInitSchema):
        blur_limit: ScaleIntType = Field(default=(7, 15), description="Maximum kernel size for sinc filter.")
        cutoff: ScaleFloatType = Field(default=(np.pi / 4, np.pi / 2), description="Cutoff frequency range in radians.")

        @field_validator("cutoff")
        @classmethod
        def check_cutoff(cls, v: ScaleFloatType, info: ValidationInfo) -> Tuple[float, float]:
            bounds = 0, np.pi
            result = to_tuple(v, v)
            check_range(result, *bounds, info.field_name)
            return result

    def __init__(
        self,
        blur_limit: ScaleIntType = (7, 15),
        cutoff: ScaleFloatType = (np.pi / 4, np.pi / 2),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.blur_limit = cast(Tuple[int, int], blur_limit)
        self.cutoff = cast(Tuple[float, float], cutoff)

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
        https://arxiv.org/abs/2107.10833

    Targets:
        image

    """

    class InitSchema(BaseTransformInitSchema):
        sigma_limit: RangeNonNegativeType = 0.0
        alpha: ZeroOneRangeType = (0.2, 0.5)
        threshold: int = Field(default=10, ge=0, le=255, description="Threshold for limiting sharpening.")

        blur_limit: ScaleIntType = Field(
            default=(3, 7), description="Maximum kernel size for blurring the input image."
        )

        @field_validator("blur_limit")
        @classmethod
        def process_blur(cls, value: ScaleIntType, info: ValidationInfo) -> Tuple[int, int]:
            return process_blur_limit(value, info, min_value=3)

    def __init__(
        self,
        blur_limit: ScaleIntType = (3, 7),
        sigma_limit: ScaleFloatType = 0.0,
        alpha: ScaleFloatType = (0.2, 0.5),
        threshold: int = 10,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.blur_limit = cast(Tuple[int, int], blur_limit)
        self.sigma_limit = cast(Tuple[float, float], sigma_limit)
        self.alpha = cast(Tuple[float, float], alpha)
        self.threshold = threshold

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
        unit8, float32

    """

    class InitSchema(BaseTransformInitSchema):
        dropout_prob: ProbabilityType = 0.01
        per_channel: bool = Field(default=False, description="Sample drop mask per channel.")
        drop_value: Optional[ColorType] = Field(
            default=0, description="Value to set in dropped pixels. None for random sampling."
        )
        mask_drop_value: Optional[ColorType] = Field(
            default=None, description="Value to set in dropped pixels in masks. None to leave masks unchanged."
        )

        @model_validator(mode="after")
        def validate_mask_drop_value(self) -> Self:
            if self.mask_drop_value is not None and self.per_channel:
                msg = "PixelDropout supports mask only with per_channel=False."
                raise ValueError(msg)
            return self

    _targets = (Targets.IMAGE, Targets.MASK)

    def __init__(
        self,
        dropout_prob: float = 0.01,
        per_channel: bool = False,
        drop_value: Optional[ColorType] = 0,
        mask_drop_value: Optional[ColorType] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.drop_value = drop_value
        self.mask_drop_value = mask_drop_value

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

        if mask.ndim == GRAYSCALE_SHAPE_LEN:
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
        https://arxiv.org/abs/1903.12261
        https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py

    """

    class InitSchema(BaseTransformInitSchema):
        mean: ZeroOneRangeType = (0.65, 0.65)
        std: ZeroOneRangeType = (0.3, 0.3)
        gauss_sigma: RangeNonNegativeType = (2, 2)
        cutout_threshold: ZeroOneRangeType = (0.68, 0.68)
        intensity: ZeroOneRangeType = (0.6, 0.6)
        mode: Union[SpatterMode, Sequence[SpatterMode]] = "rain"
        color: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None

        @field_validator("mode")
        @classmethod
        def check_mode(cls, mode: Union[SpatterMode, Sequence[SpatterMode]]) -> Sequence[SpatterMode]:
            if isinstance(mode, str):
                return [mode]
            return mode

        @model_validator(mode="after")
        def check_color(self) -> Self:
            if self.color is None:
                self.color = {"rain": [238, 238, 175], "mud": [20, 42, 63]}

            elif isinstance(self.color, (list, tuple)) and len(self.mode) == 1:
                if len(self.color) != NUM_RGB_CHANNELS:
                    msg = "Color must be a list of three integers for RGB format."
                    raise ValueError(msg)
                self.color = {self.mode[0]: self.color}
            elif isinstance(self.color, dict):
                result = {}
                for mode in self.mode:
                    if mode not in self.color:
                        raise ValueError(f"Color for mode {mode} is not specified.")
                    if len(self.color[mode]) != NUM_RGB_CHANNELS:
                        raise ValueError(f"Color for mode {mode} must be in RGB format.")
                    result[mode] = self.color[mode]
            else:
                msg = "Color must be a list of RGB values or a dict mapping mode to RGB values."
                raise ValueError(msg)
            return self

    def __init__(
        self,
        mean: ScaleFloatType = (0.65, 0.65),
        std: ScaleFloatType = (0.3, 0.3),
        gauss_sigma: ScaleFloatType = (2, 2),
        cutout_threshold: ScaleFloatType = (0.68, 0.68),
        intensity: ScaleFloatType = (0.6, 0.6),
        mode: Union[SpatterMode, Sequence[SpatterMode]] = "rain",
        color: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.mean = cast(Tuple[float, float], mean)
        self.std = cast(Tuple[float, float], std)
        self.gauss_sigma = cast(Tuple[float, float], gauss_sigma)
        self.cutout_threshold = cast(Tuple[float, float], cutout_threshold)
        self.intensity = cast(Tuple[float, float], intensity)
        self.mode = mode
        self.color = cast(Dict[str, Sequence[int]], color)

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

    class InitSchema(BaseTransformInitSchema):
        primary_distortion_limit: RangeSymmetricType = (-0.02, 0.02)
        secondary_distortion_limit: RangeSymmetricType = (-0.05, 0.05)
        mode: ChromaticAberrationMode = "green_purple"
        interpolation: InterpolationType = cv2.INTER_LINEAR

    def __init__(
        self,
        primary_distortion_limit: ScaleFloatType = (-0.02, 0.02),
        secondary_distortion_limit: ScaleFloatType = (0.05, 0.05),
        mode: ChromaticAberrationMode = "green_purple",
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.primary_distortion_limit = cast(Tuple[float, float], primary_distortion_limit)
        self.secondary_distortion_limit = cast(Tuple[float, float], secondary_distortion_limit)
        self.mode = mode
        self.interpolation = interpolation

    def apply(
        self,
        img: np.ndarray,
        primary_distortion_red: float,
        secondary_distortion_red: float,
        primary_distortion_blue: float,
        secondary_distortion_blue: float,
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
            return -b
        return b

    @staticmethod
    def _unmatch_sign(a: float, b: float) -> float:
        # Unmatch the sign of b to a
        if (a < 0 and b < 0) or (a > 0 and b > 0):
            return -b
        return b

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return "primary_distortion_limit", "secondary_distortion_limit", "mode", "interpolation"
