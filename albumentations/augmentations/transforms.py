from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Sequence
from types import LambdaType
from typing import Annotated, Any, Callable, Union, cast
from warnings import warn

import albucore
import cv2
import numpy as np
from albucore import (
    MAX_VALUES_BY_DTYPE,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    clip,
    from_float,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    multiply,
    normalize,
    normalize_per_image,
    to_float,
)
from pydantic import AfterValidator, BaseModel, Field, ValidationInfo, field_validator, model_validator
from scipy import special
from scipy.ndimage import gaussian_filter
from typing_extensions import Literal, Self, TypedDict

import albumentations.augmentations.dropout.functional as fdropout
import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.blur import functional as fblur
from albumentations.augmentations.blur.transforms import BlurInitSchema
from albumentations.augmentations.utils import check_range, non_rgb_error
from albumentations.core.bbox_utils import BboxProcessor, denormalize_bboxes, normalize_bboxes
from albumentations.core.keypoints_utils import KeypointsProcessor
from albumentations.core.pydantic import (
    InterpolationType,
    NonNegativeFloatRangeType,
    OnePlusFloatRangeType,
    OnePlusIntRangeType,
    ProbabilityType,
    SymmetricRangeType,
    ZeroOneRangeType,
    check_0plus,
    check_01,
    check_1plus,
    nondecreasing,
    repeat_if_scalar,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
    ImageOnlyTransform,
    Interpolation,
    NoOp,
)
from albumentations.core.types import (
    MAX_RAIN_ANGLE,
    MONO_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
    PAIR,
    ChromaticAberrationMode,
    ColorType,
    ImageMode,
    MorphologyMode,
    RainMode,
    ScaleFloatType,
    ScaleIntType,
    SpatterMode,
    Targets,
)
from albumentations.core.utils import format_args, to_tuple

from . import functional as fmain

__all__ = [
    "Normalize",
    "RandomGamma",
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
    "RingingOvershoot",
    "UnsharpMask",
    "PixelDropout",
    "Spatter",
    "ChromaticAberration",
    "Morphological",
    "PlanckianJitter",
    "ShotNoise",
]

NUM_BITS_ARRAY_LENGTH = 3
MAX_JPEG_QUALITY = 100
TWENTY = 20


class Normalize(ImageOnlyTransform):
    """Applies various normalization techniques to an image. The specific normalization technique can be selected
        with the `normalization` parameter.

    Standard normalization is applied using the formula:
        `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`.
        Other normalization techniques adjust the image based on global or per-channel statistics,
        or scale pixel values to a specified range.

    Args:
        mean (ColorType | None): Mean values for standard normalization.
            For "standard" normalization, the default values are ImageNet mean values: (0.485, 0.456, 0.406).
        std (ColorType | None): Standard deviation values for standard normalization.
            For "standard" normalization, the default values are ImageNet standard deviation :(0.229, 0.224, 0.225).
        max_pixel_value (float | None): Maximum possible pixel value, used for scaling in standard normalization.
            Defaults to 255.0.
        normalization (Literal["standard", "image", "image_per_channel", "min_max", "min_max_per_channel"])
            Specifies the normalization technique to apply. Defaults to "standard".
            - "standard": Applies the formula `(img - mean * max_pixel_value) / (std * max_pixel_value)`.
                The default mean and std are based on ImageNet. You can use mean and std values of (0.5, 0.5, 0.5)
                for inception normalization. And mean values of (0, 0, 0) and std values of (1, 1, 1) for YOLO.
            - "image": Normalizes the whole image based on its global mean and standard deviation.
            - "image_per_channel": Normalizes the image per channel based on each channel's mean and standard deviation.
            - "min_max": Scales the image pixel values to a [0, 1] range based on the global
                minimum and maximum pixel values.
            - "min_max_per_channel": Scales each channel of the image pixel values to a [0, 1]
                range based on the per-channel minimum and maximum pixel values.

        p (float): Probability of applying the transform. Defaults to 1.0.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - For "standard" normalization, `mean`, `std`, and `max_pixel_value` must be provided.
        - For other normalization types, these parameters are ignored.
        - For inception normalization, use mean values of (0.5, 0.5, 0.5).
        - For YOLO normalization, use mean values of (0.5, 0.5, 0.5) and std values of (0, 0, 0).
        - This transform is often used as a final step in image preprocessing pipelines to
          prepare images for neural network input.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> # Standard ImageNet normalization
        >>> transform = A.Normalize(
        ...     mean=(0.485, 0.456, 0.406),
        ...     std=(0.229, 0.224, 0.225),
        ...     max_pixel_value=255.0,
        ...     p=1.0
        ... )
        >>> normalized_image = transform(image=image)["image"]
        >>>
        >>> # Min-max normalization
        >>> transform_minmax = A.Normalize(normalization="min_max", p=1.0)
        >>> normalized_image_minmax = transform_minmax(image=image)["image"]

    References:
        - ImageNet mean and std: https://pytorch.org/vision/stable/models.html
        - Inception preprocessing: https://keras.io/api/applications/inceptionv3/
    """

    class InitSchema(BaseTransformInitSchema):
        mean: ColorType | None
        std: ColorType | None
        max_pixel_value: float | None
        normalization: Literal[
            "standard",
            "image",
            "image_per_channel",
            "min_max",
            "min_max_per_channel",
        ]

        @model_validator(mode="after")
        def validate_normalization(self) -> Self:
            if (
                self.mean is None
                or self.std is None
                or self.max_pixel_value is None
                and self.normalization == "standard"
            ):
                raise ValueError("mean, std, and max_pixel_value must be provided for standard normalization.")
            return self

    def __init__(
        self,
        mean: ColorType | None = (0.485, 0.456, 0.406),
        std: ColorType | None = (0.229, 0.224, 0.225),
        max_pixel_value: float | None = 255.0,
        normalization: Literal["standard", "image", "image_per_channel", "min_max", "min_max_per_channel"] = "standard",
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.mean = mean
        self.mean_np = np.array(mean, dtype=np.float32) * max_pixel_value
        self.std = std
        self.denominator = np.reciprocal(np.array(std, dtype=np.float32) * max_pixel_value)
        self.max_pixel_value = max_pixel_value
        self.normalization = normalization

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if self.normalization == "standard":
            return normalize(
                img,
                self.mean_np,
                self.denominator,
            )
        return normalize_per_image(img, self.normalization)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "mean", "std", "max_pixel_value", "normalization"


class ImageCompression(ImageOnlyTransform):
    """Decrease image quality by applying JPEG or WebP compression.

    This transform simulates the effect of saving an image with lower quality settings,
    which can introduce compression artifacts. It's useful for data augmentation and
    for testing model robustness against varying image qualities.

    Args:
        quality_range (tuple[int, int]): Range for the compression quality.
            The values should be in [1, 100] range, where:
            - 1 is the lowest quality (maximum compression)
            - 100 is the highest quality (minimum compression)
            Default: (99, 100)

        compression_type (Literal["jpeg", "webp"]): Type of compression to apply.
            - "jpeg": JPEG compression
            - "webp": WebP compression
            Default: "jpeg"

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This transform expects images with 1, 3, or 4 channels.
        - For JPEG compression, alpha channels (4th channel) will be ignored.
        - WebP compression supports transparency (4 channels).
        - The actual file is not saved to disk; the compression is simulated in memory.
        - Lower quality values result in smaller file sizes but may introduce visible artifacts.
        - This transform can be useful for:
          * Data augmentation to improve model robustness
          * Testing how models perform on images of varying quality
          * Simulating images transmitted over low-bandwidth connections

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ImageCompression(quality_range=(50, 90), compression_type=0, p=1.0)
        >>> result = transform(image=image)
        >>> compressed_image = result["image"]

    References:
        - JPEG compression: https://en.wikipedia.org/wiki/JPEG
        - WebP compression: https://developers.google.com/speed/webp
    """

    class InitSchema(BaseTransformInitSchema):
        quality_range: Annotated[tuple[int, int], AfterValidator(check_1plus), AfterValidator(nondecreasing)]

        quality_lower: int | None = Field(
            ge=1,
            le=100,
        )
        quality_upper: int | None = Field(
            ge=1,
            le=100,
        )
        compression_type: Literal["jpeg", "webp"]

        @model_validator(mode="after")
        def validate_ranges(self) -> Self:
            # Update the quality_range based on the non-None values of quality_lower and quality_upper
            if self.quality_lower is not None or self.quality_upper is not None:
                if self.quality_lower is not None:
                    warn(
                        "`quality_lower` is deprecated. Use `quality_range` as tuple"
                        " (quality_lower, quality_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if self.quality_upper is not None:
                    warn(
                        "`quality_upper` is deprecated. Use `quality_range` as tuple"
                        " (quality_lower, quality_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                lower = self.quality_lower if self.quality_lower is not None else self.quality_range[0]
                upper = self.quality_upper if self.quality_upper is not None else self.quality_range[1]
                self.quality_range = (lower, upper)
                # Clear the deprecated individual quality settings
                self.quality_lower = None
                self.quality_upper = None

            # Validate the quality_range
            if not (1 <= self.quality_range[0] <= MAX_JPEG_QUALITY and 1 <= self.quality_range[1] <= MAX_JPEG_QUALITY):
                raise ValueError(f"Quality range values should be within [1, {MAX_JPEG_QUALITY}] range.")

            return self

    def __init__(
        self,
        quality_lower: int | None = None,
        quality_upper: int | None = None,
        compression_type: Literal["jpeg", "webp"] = "jpeg",
        quality_range: tuple[int, int] = (99, 100),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.quality_range = quality_range
        self.compression_type = compression_type

    def apply(self, img: np.ndarray, quality: int, image_type: Literal[".jpg", ".webp"], **params: Any) -> np.ndarray:
        return fmain.image_compression(img, quality, image_type)

    def get_params(self) -> dict[str, int | str]:
        if self.compression_type == "jpeg":
            image_type = ".jpg"
        elif self.compression_type == "webp":
            image_type = ".webp"
        else:
            raise ValueError(f"Unknown image compression type: {self.compression_type}")

        return {
            "quality": self.py_random.randint(*self.quality_range),
            "image_type": image_type,
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "quality_range", "compression_type"


class RandomSnow(ImageOnlyTransform):
    """Applies a random snow effect to the input image.

    This transform simulates snowfall by either bleaching out some pixel values or
    adding a snow texture to the image, depending on the chosen method.

    Args:
        snow_point_range (tuple[float, float]): Range for the snow point threshold.
            Both values should be in the (0, 1) range. Default: (0.1, 0.3).
        brightness_coeff (float): Coefficient applied to increase the brightness of pixels
            below the snow_point threshold. Larger values lead to more pronounced snow effects.
            Should be > 0. Default: 2.5.
        method (Literal["bleach", "texture"]): The snow simulation method to use. Options are:
            - "bleach": Uses a simple pixel value thresholding technique.
            - "texture": Applies a more realistic snow texture overlay.
            Default: "texture".
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - The "bleach" method increases the brightness of pixels above a certain threshold,
          creating a simple snow effect. This method is faster but may look less realistic.
        - The "texture" method creates a more realistic snow effect through the following steps:
          1. Converts the image to HSV color space for better control over brightness.
          2. Increases overall image brightness to simulate the reflective nature of snow.
          3. Generates a snow texture using Gaussian noise, which is then smoothed with a Gaussian filter.
          4. Applies a depth effect to the snow texture, making it more prominent at the top of the image.
          5. Blends the snow texture with the original image using alpha compositing.
          6. Adds a slight blue tint to simulate the cool color of snow.
          7. Adds random sparkle effects to simulate light reflecting off snow crystals.
          This method produces a more realistic result but is computationally more expensive.

    Mathematical Formulation:
        For the "bleach" method:
        Let L be the lightness channel in HLS color space.
        For each pixel (i, j):
        If L[i, j] > snow_point:
            L[i, j] = L[i, j] * brightness_coeff

        For the "texture" method:
        1. Brightness adjustment: V_new = V * (1 + brightness_coeff * snow_point)
        2. Snow texture generation: T = GaussianFilter(GaussianNoise(μ=0.5, sigma=0.3))
        3. Depth effect: D = LinearGradient(1.0 to 0.2)
        4. Final pixel value: P = (1 - alpha) * original_pixel + alpha * (T * D * 255)
           where alpha is the snow intensity factor derived from snow_point.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage (bleach method)
        >>> transform = A.RandomSnow(p=1.0)
        >>> snowy_image = transform(image=image)["image"]

        # Using texture method with custom parameters
        >>> transform = A.RandomSnow(
        ...     snow_point_range=(0.2, 0.4),
        ...     brightness_coeff=2.0,
        ...     method="texture",
        ...     p=1.0
        ... )
        >>> snowy_image = transform(image=image)["image"]

    References:
        - Bleach method: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library
        - Texture method: Inspired by computer graphics techniques for snow rendering
          and atmospheric scattering simulations.
    """

    class InitSchema(BaseTransformInitSchema):
        snow_point_range: Annotated[tuple[float, float], AfterValidator(check_01), AfterValidator(nondecreasing)]

        snow_point_lower: float | None = Field(
            gt=0,
            lt=1,
        )
        snow_point_upper: float | None = Field(
            gt=0,
            lt=1,
        )
        brightness_coeff: float = Field(gt=0)
        method: Literal["bleach", "texture"]

        @model_validator(mode="after")
        def validate_ranges(self) -> Self:
            if self.snow_point_lower is not None or self.snow_point_upper is not None:
                if self.snow_point_lower is not None:
                    warn(
                        "`snow_point_lower` deprecated. Use `snow_point_range` as tuple"
                        " (snow_point_lower, snow_point_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if self.snow_point_upper is not None:
                    warn(
                        "`snow_point_upper` deprecated. Use `snow_point_range` as tuple"
                        "(snow_point_lower, snow_point_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                lower = self.snow_point_lower if self.snow_point_lower is not None else self.snow_point_range[0]
                upper = self.snow_point_upper if self.snow_point_upper is not None else self.snow_point_range[1]
                self.snow_point_range = (lower, upper)
                self.snow_point_lower = None
                self.snow_point_upper = None

            # Validate the snow_point_range
            if not (0 < self.snow_point_range[0] <= self.snow_point_range[1] < 1):
                raise ValueError("snow_point_range values should be increasing within (0, 1) range.")

            return self

    def __init__(
        self,
        snow_point_lower: float | None = None,
        snow_point_upper: float | None = None,
        brightness_coeff: float = 2.5,
        snow_point_range: tuple[float, float] = (0.1, 0.3),
        method: Literal["bleach", "texture"] = "bleach",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.snow_point_range = snow_point_range
        self.brightness_coeff = brightness_coeff
        self.method = method

    def apply(
        self,
        img: np.ndarray,
        snow_point: float,
        snow_texture: np.ndarray,
        sparkle_mask: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        non_rgb_error(img)

        if self.method == "bleach":
            return fmain.add_snow_bleach(img, snow_point, self.brightness_coeff)
        if self.method == "texture":
            return fmain.add_snow_texture(img, snow_point, self.brightness_coeff, snow_texture, sparkle_mask)

        raise ValueError(f"Unknown snow method: {self.method}")

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, np.ndarray | None]:
        image_shape = params["shape"][:2]
        result = {
            "snow_point": self.py_random.uniform(*self.snow_point_range),
            "snow_texture": None,
            "sparkle_mask": None,
        }

        if self.method == "texture":
            snow_texture, sparkle_mask = fmain.generate_snow_textures(
                img_shape=image_shape,
                random_generator=self.random_generator,
            )
            result["snow_texture"] = snow_texture
            result["sparkle_mask"] = sparkle_mask

        return result

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "snow_point_range", "brightness_coeff", "method"


class RandomGravel(ImageOnlyTransform):
    """Adds gravel-like artifacts to the input image.

    This transform simulates the appearance of gravel or small stones scattered across
    specific regions of an image. It's particularly useful for augmenting datasets of
    road or terrain images, adding realistic texture variations.

    Args:
        gravel_roi (tuple[float, float, float, float]): Region of interest where gravel
            will be added, specified as (x_min, y_min, x_max, y_max) in relative coordinates
            [0, 1]. Default: (0.1, 0.4, 0.9, 0.9).
        number_of_patches (int): Number of gravel patch regions to generate within the ROI.
            Each patch will contain multiple gravel particles. Default: 2.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - The gravel effect is created by modifying the saturation channel in the HLS color space.
        - Gravel particles are distributed within randomly generated patches inside the specified ROI.
        - This transform is particularly useful for:
          * Augmenting datasets for road condition analysis
          * Simulating variations in terrain for computer vision tasks
          * Adding realistic texture to synthetic images of outdoor scenes

    Mathematical Formulation:
        For each gravel patch:
        1. A rectangular region is randomly generated within the specified ROI.
        2. Within this region, multiple gravel particles are placed.
        3. For each particle:
           - Random (x, y) coordinates are generated within the patch.
           - A random radius (r) between 1 and 3 pixels is assigned.
           - A random saturation value (sat) between 0 and 255 is assigned.
        4. The saturation channel of the image is modified for each particle:
           image_hls[y-r:y+r, x-r:x+r, 1] = sat

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomGravel(p=1.0)
        >>> augmented_image = transform(image=image)["image"]

        # Custom ROI and number of patches
        >>> transform = A.RandomGravel(
        ...     gravel_roi=(0.2, 0.2, 0.8, 0.8),
        ...     number_of_patches=5,
        ...     p=1.0
        ... )
        >>> augmented_image = transform(image=image)["image"]

        # Combining with other transforms
        >>> transform = A.Compose([
        ...     A.RandomGravel(p=0.7),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> augmented_image = transform(image=image)["image"]

    References:
        - Road surface textures: https://en.wikipedia.org/wiki/Road_surface
        - HLS color space: https://en.wikipedia.org/wiki/HSL_and_HSV
    """

    class InitSchema(BaseTransformInitSchema):
        gravel_roi: tuple[float, float, float, float]
        number_of_patches: int = Field(ge=1)

        @model_validator(mode="after")
        def validate_gravel_roi(self) -> Self:
            gravel_lower_x, gravel_lower_y, gravel_upper_x, gravel_upper_y = self.gravel_roi
            if not 0 <= gravel_lower_x < gravel_upper_x <= 1 or not 0 <= gravel_lower_y < gravel_upper_y <= 1:
                raise ValueError(f"Invalid gravel_roi. Got: {self.gravel_roi}.")
            return self

    def __init__(
        self,
        gravel_roi: tuple[float, float, float, float] = (0.1, 0.4, 0.9, 0.9),
        number_of_patches: int = 2,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)
        self.gravel_roi = gravel_roi
        self.number_of_patches = number_of_patches

    def generate_gravel_patch(self, rectangular_roi: tuple[int, int, int, int]) -> np.ndarray:
        x_min, y_min, x_max, y_max = rectangular_roi
        area = abs((x_max - x_min) * (y_max - y_min))
        count = area // 10
        gravels = np.empty([count, 2], dtype=np.int64)
        gravels[:, 0] = self.random_generator.integers(x_min, x_max, count)
        gravels[:, 1] = self.random_generator.integers(y_min, y_max, count)
        return gravels

    def apply(self, img: np.ndarray, gravels_infos: list[Any], **params: Any) -> np.ndarray:
        return fmain.add_gravel(img, gravels_infos)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
        height, width = params["shape"][:2]

        # Calculate ROI in pixels
        x_min, y_min, x_max, y_max = (
            int(coord * dim) for coord, dim in zip(self.gravel_roi, [width, height, width, height])
        )

        roi_width = x_max - x_min
        roi_height = y_max - y_min

        gravels_info = []

        for _ in range(self.number_of_patches):
            # Generate a random rectangular region within the ROI
            patch_width = self.py_random.randint(roi_width // 10, roi_width // 5)
            patch_height = self.py_random.randint(roi_height // 10, roi_height // 5)

            patch_x = self.py_random.randint(x_min, x_max - patch_width)
            patch_y = self.py_random.randint(y_min, y_max - patch_height)

            # Generate gravel particles within this patch
            num_particles = (patch_width * patch_height) // 100  # Adjust this divisor to control density

            for _ in range(num_particles):
                x = self.py_random.randint(patch_x, patch_x + patch_width)
                y = self.py_random.randint(patch_y, patch_y + patch_height)
                r = self.py_random.randint(1, 3)
                sat = self.py_random.randint(0, 255)

                gravels_info.append(
                    [
                        max(y - r, 0),  # min_y
                        min(y + r, height - 1),  # max_y
                        max(x - r, 0),  # min_x
                        min(x + r, width - 1),  # max_x
                        sat,  # saturation
                    ],
                )

        return {"gravels_infos": np.array(gravels_info, dtype=np.int64)}

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "gravel_roi", "number_of_patches"


class RandomRain(ImageOnlyTransform):
    """Adds rain effects to an image.

    This transform simulates rainfall by overlaying semi-transparent streaks onto the image,
    creating a realistic rain effect. It can be used to augment datasets for computer vision
    tasks that need to perform well in rainy conditions.

    Args:
        slant_range (tuple[int, int]): Range for the rain slant angle in degrees.
            Negative values slant to the left, positive to the right. Default: (-10, 10).
        drop_length (int): Length of the rain drops in pixels. Default: 20.
        drop_width (int): Width of the rain drops in pixels. Default: 1.
        drop_color (tuple[int, int, int]): Color of the rain drops in RGB format. Default: (200, 200, 200).
        blur_value (int): Blur value for simulating rain effect. Rainy views are typically blurry. Default: 7.
        brightness_coefficient (float): Coefficient to adjust the brightness of the image.
            Rainy scenes are usually darker. Should be in the range (0, 1]. Default: 0.7.
        rain_type (Literal["drizzle", "heavy", "torrential", "default"]): Type of rain to simulate.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - The rain effect is created by drawing semi-transparent lines on the image.
        - The slant of the rain can be controlled to simulate wind effects.
        - Different rain types (drizzle, heavy, torrential) adjust the density and appearance of the rain.
        - The transform also adjusts image brightness and applies a blur to simulate the visual effects of rain.
        - This transform is particularly useful for:
          * Augmenting datasets for autonomous driving in rainy conditions
          * Testing the robustness of computer vision models to weather effects
          * Creating realistic rainy scenes for image editing or film production

    Mathematical Formulation:
        For each raindrop:
        1. Start position (x1, y1) is randomly generated within the image.
        2. End position (x2, y2) is calculated based on drop_length and slant:
           x2 = x1 + drop_length * sin(slant)
           y2 = y1 + drop_length * cos(slant)
        3. A line is drawn from (x1, y1) to (x2, y2) with the specified drop_color and drop_width.
        4. The image is then blurred and its brightness is adjusted.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomRain(p=1.0)
        >>> rainy_image = transform(image=image)["image"]

        # Custom rain parameters
        >>> transform = A.RandomRain(
        ...     slant_range=(-15, 15),
        ...     drop_length=30,
        ...     drop_width=2,
        ...     drop_color=(180, 180, 180),
        ...     blur_value=5,
        ...     brightness_coefficient=0.8,
        ...     p=1.0
        ... )
        >>> rainy_image = transform(image=image)["image"]

        # Simulating heavy rain
        >>> transform = A.RandomRain(rain_type="heavy", p=1.0)
        >>> heavy_rain_image = transform(image=image)["image"]

    References:
        - Rain visualization techniques: https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-real-time-rain-rendering
        - Weather effects in computer vision: https://www.sciencedirect.com/science/article/pii/S1077314220300692
    """

    class InitSchema(BaseTransformInitSchema):
        slant_lower: int | None = Field(default=None)
        slant_upper: int | None = Field(default=None)
        slant_range: Annotated[tuple[float, float], AfterValidator(nondecreasing)]
        drop_length: int = Field(ge=1)
        drop_width: int = Field(ge=1)
        drop_color: tuple[int, int, int]
        blur_value: int = Field(ge=1)
        brightness_coefficient: float = Field(gt=0, le=1)
        rain_type: RainMode

        @model_validator(mode="after")
        def validate_ranges(self) -> Self:
            if self.slant_lower is not None or self.slant_upper is not None:
                if self.slant_lower is not None:
                    warn(
                        "`slant_lower` deprecated. Use `slant_range` as tuple (slant_lower, slant_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if self.slant_upper is not None:
                    warn(
                        "`slant_upper` deprecated. Use `slant_range` as tuple (slant_lower, slant_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                lower = self.slant_lower if self.slant_lower is not None else self.slant_range[0]
                upper = self.slant_upper if self.slant_upper is not None else self.slant_range[1]
                self.slant_range = (lower, upper)
                self.slant_lower = None
                self.slant_upper = None

            # Validate the slant_range
            if not (-MAX_RAIN_ANGLE <= self.slant_range[0] <= self.slant_range[1] <= MAX_RAIN_ANGLE):
                raise ValueError(
                    f"slant_range values should be increasing within [-{MAX_RAIN_ANGLE}, {MAX_RAIN_ANGLE}] range.",
                )
            return self

    def __init__(
        self,
        slant_lower: int | None = None,
        slant_upper: int | None = None,
        slant_range: tuple[int, int] = (-10, 10),
        drop_length: int = 20,
        drop_width: int = 1,
        drop_color: tuple[int, int, int] = (200, 200, 200),
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        rain_type: RainMode = "default",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.slant_range = slant_range
        self.drop_length = drop_length
        self.drop_width = drop_width
        self.drop_color = drop_color
        self.blur_value = blur_value
        self.brightness_coefficient = brightness_coefficient
        self.rain_type = rain_type

    def apply(
        self,
        img: np.ndarray,
        slant: int,
        drop_length: int,
        rain_drops: list[tuple[int, int]],
        **params: Any,
    ) -> np.ndarray:
        non_rgb_error(img)

        return fmain.add_rain(
            img,
            slant,
            drop_length,
            self.drop_width,
            self.drop_color,
            self.blur_value,
            self.brightness_coefficient,
            rain_drops,
        )

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        slant = int(self.py_random.uniform(*self.slant_range))

        height, width = params["shape"][:2]
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
            x = self.py_random.randint(slant, width) if slant < 0 else self.py_random.randint(0, max(width - slant, 0))
            y = self.py_random.randint(0, max(height - drop_length, 0))

            rain_drops.append((x, y))

        return {"drop_length": drop_length, "slant": slant, "rain_drops": rain_drops}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "slant_range",
            "drop_length",
            "drop_width",
            "drop_color",
            "blur_value",
            "brightness_coefficient",
            "rain_type",
        )


class RandomFog(ImageOnlyTransform):
    """Simulates fog for the image by adding random fog-like artifacts.

    This transform creates a fog effect by generating semi-transparent overlays
    that mimic the visual characteristics of fog. The fog intensity and distribution
    can be controlled to create various fog-like conditions.

    Args:
        fog_coef_range (tuple[float, float]): Range for fog intensity coefficient. Should be in [0, 1] range.
        alpha_coef (float): Transparency of the fog circles. Should be in [0, 1] range. Default: 0.08.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The fog effect is created by overlaying semi-transparent circles on the image.
        - Higher fog coefficient values result in denser fog effects.
        - The fog is typically denser in the center of the image and gradually decreases towards the edges.
        - This transform is useful for:
          * Simulating various weather conditions in outdoor scenes
          * Data augmentation for improving model robustness to foggy conditions
          * Creating atmospheric effects in image editing

    Mathematical Formulation:
        For each fog particle:
        1. A position (x, y) is randomly generated within the image.
        2. A circle with random radius is drawn at this position.
        3. The circle's alpha (transparency) is determined by the alpha_coef.
        4. These circles are overlaid on the original image to create the fog effect.

        The final pixel value is calculated as:
        output = (1 - alpha) * original_pixel + alpha * fog_color

        where alpha is influenced by the fog_coef and alpha_coef parameters.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomFog(p=1.0)
        >>> foggy_image = transform(image=image)["image"]

        # Custom fog intensity range
        >>> transform = A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.8, p=1.0)
        >>> foggy_image = transform(image=image)["image"]

        # Adjust fog transparency
        >>> transform = A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.1, p=1.0)
        >>> foggy_image = transform(image=image)["image"]

    References:
        - Fog: https://en.wikipedia.org/wiki/Fog
        - Atmospheric perspective: https://en.wikipedia.org/wiki/Aerial_perspective
    """

    class InitSchema(BaseTransformInitSchema):
        fog_coef_lower: float | None = Field(
            ge=0,
            le=1,
        )
        fog_coef_upper: float | None = Field(
            ge=0,
            le=1,
        )
        fog_coef_range: Annotated[tuple[float, float], AfterValidator(check_01), AfterValidator(nondecreasing)]

        alpha_coef: float = Field(ge=0, le=1)

        @model_validator(mode="after")
        def validate_fog_coefficients(self) -> Self:
            if self.fog_coef_lower is not None:
                warn("`fog_coef_lower` is deprecated, use `fog_coef_range` instead.", DeprecationWarning, stacklevel=2)
            if self.fog_coef_upper is not None:
                warn("`fog_coef_upper` is deprecated, use `fog_coef_range` instead.", DeprecationWarning, stacklevel=2)

            lower = self.fog_coef_lower if self.fog_coef_lower is not None else self.fog_coef_range[0]
            upper = self.fog_coef_upper if self.fog_coef_upper is not None else self.fog_coef_range[1]
            self.fog_coef_range = (lower, upper)

            self.fog_coef_lower = None
            self.fog_coef_upper = None

            return self

    def __init__(
        self,
        fog_coef_lower: float | None = None,
        fog_coef_upper: float | None = None,
        alpha_coef: float = 0.08,
        fog_coef_range: tuple[float, float] = (0.3, 1),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.fog_coef_range = fog_coef_range
        self.alpha_coef = alpha_coef

    def apply(
        self,
        img: np.ndarray,
        particle_positions: list[tuple[int, int]],
        radiuses: list[int],
        intensity: float,
        **params: Any,
    ) -> np.ndarray:
        return fmain.add_fog(
            img,
            intensity,
            self.alpha_coef,
            particle_positions,
            radiuses,
        )

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        # Select a random fog intensity within the specified range
        intensity = self.py_random.uniform(*self.fog_coef_range)

        image_shape = params["shape"][:2]

        image_height, image_width = image_shape

        # Calculate the size of the fog effect region based on image width and fog intensity
        fog_region_size = max(1, int(image_width // 3 * intensity))

        particle_positions = []

        # Initialize the central region where fog will be most dense
        center_x, center_y = (int(x) for x in fgeometric.center(image_shape))

        # Define the initial size of the foggy area
        current_width = image_width
        current_height = image_height

        # Define shrink factor for reducing the foggy area each iteration
        shrink_factor = 0.1

        max_iterations = 10  # Prevent infinite loop
        iteration = 0

        while current_width > fog_region_size and current_height > fog_region_size and iteration < max_iterations:
            # Calculate the number of particles for this region
            area = current_width * current_height
            particles_in_region = int(area / (fog_region_size * fog_region_size) * intensity * 10)

            for _ in range(particles_in_region):
                # Generate random positions within the current region
                x = self.py_random.randint(center_x - current_width // 2, center_x + current_width // 2)
                y = self.py_random.randint(center_y - current_height // 2, center_y + current_height // 2)
                particle_positions.append((x, y))

            # Shrink the region for the next iteration
            current_width = int(current_width * (1 - shrink_factor))
            current_height = int(current_height * (1 - shrink_factor))

            iteration += 1

        radiuses = fmain.get_fog_particle_radiuses(
            image_shape,
            len(particle_positions),
            intensity,
            self.random_generator,
        )

        return {
            "particle_positions": particle_positions,
            "intensity": intensity,
            "radiuses": radiuses,
        }

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "fog_coef_range", "alpha_coef"


class RandomSunFlare(ImageOnlyTransform):
    """Simulates a sun flare effect on the image by adding circles of light.

    This transform creates a sun flare effect by overlaying multiple semi-transparent
    circles of varying sizes and intensities along a line originating from a "sun" point.
    It offers two methods: a simple overlay technique and a more complex physics-based approach.

    Args:
        flare_roi (tuple[float, float, float, float]): Region of interest where the sun flare
            can appear. Values are in the range [0, 1] and represent (x_min, y_min, x_max, y_max)
            in relative coordinates. Default: (0, 0, 1, 0.5).
        angle_range (tuple[float, float]): Range of angles (in radians) for the flare direction.
            Values should be in the range [0, 1], where 0 represents 0 radians and 1 represents 2π radians.
            Default: (0, 1).
        num_flare_circles_range (tuple[int, int]): Range for the number of flare circles to generate.
            Default: (6, 10).
        src_radius (int): Radius of the sun circle in pixels. Default: 400.
        src_color (tuple[int, int, int]): Color of the sun in RGB format. Default: (255, 255, 255).
        method (Literal["overlay", "physics_based"]): Method to use for generating the sun flare.
            "overlay" uses a simple alpha blending technique, while "physics_based" simulates
            more realistic optical phenomena. Default: "physics_based".

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        - overlay: Any
        - physics_based: RGB

    Note:
        The transform offers two methods for generating sun flares:

        1. Overlay Method ("overlay"):
           - Creates a simple sun flare effect using basic alpha blending.
           - Steps:
             a. Generate the main sun circle with a radial gradient.
             b. Create smaller flare circles along the flare line.
             c. Blend these elements with the original image using alpha compositing.
           - Characteristics:
             * Faster computation
             * Less realistic appearance
             * Suitable for basic augmentation or when performance is a priority

        2. Physics-based Method ("physics_based"):
           - Simulates more realistic optical phenomena observed in actual lens flares.
           - Steps:
             a. Create a separate flare layer for complex manipulations.
             b. Add the main sun circle and diffraction spikes to simulate light diffraction.
             c. Generate and add multiple flare circles with varying properties.
             d. Apply Gaussian blur to create a soft, glowing effect.
             e. Create and apply a radial gradient mask for natural fading from the center.
             f. Simulate chromatic aberration by applying different blurs to color channels.
             g. Blend the flare with the original image using screen blending mode.
           - Characteristics:
             * More computationally intensive
             * Produces more realistic and visually appealing results
             * Includes effects like diffraction spikes and chromatic aberration
             * Suitable for high-quality augmentation or realistic image synthesis

    Mathematical Formulation:
        For both methods:
        1. Sun position (x_s, y_s) is randomly chosen within the specified ROI.
        2. Flare angle θ is randomly chosen from the angle_range.
        3. For each flare circle i:
           - Position (x_i, y_i) = (x_s + t_i * cos(θ), y_s + t_i * sin(θ))
             where t_i is a random distance along the flare line.
           - Radius r_i is randomly chosen, with larger circles closer to the sun.
           - Alpha (transparency) alpha_i is randomly chosen in the range [0.05, 0.2].
           - Color (R_i, G_i, B_i) is randomly chosen close to src_color.

        Overlay method blending:
        new_pixel = (1 - alpha_i) * original_pixel + alpha_i * flare_color_i

        Physics-based method blending:
        new_pixel = 255 - ((255 - original_pixel) * (255 - flare_pixel) / 255)

        4. Each flare circle is blended with the image using alpha compositing:
           new_pixel = (1 - alpha_i) * original_pixel + alpha_i * flare_color_i

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [1000, 1000, 3], dtype=np.uint8)

        # Default sun flare (overlay method)
        >>> transform = A.RandomSunFlare(p=1.0)
        >>> flared_image = transform(image=image)["image"]

        # Physics-based sun flare with custom parameters

        # Default sun flare
        >>> transform = A.RandomSunFlare(p=1.0)
        >>> flared_image = transform(image=image)["image"]

        # Custom sun flare parameters

        >>> transform = A.RandomSunFlare(
        ...     flare_roi=(0.1, 0, 0.9, 0.3),
        ...     angle_range=(0.25, 0.75),
        ...     num_flare_circles_range=(5, 15),
        ...     src_radius=200,
        ...     src_color=(255, 200, 100),
        ...     method="physics_based",
        ...     p=1.0
        ... )
        >>> flared_image = transform(image=image)["image"]

    References:
        - Lens flare: https://en.wikipedia.org/wiki/Lens_flare
        - Alpha compositing: https://en.wikipedia.org/wiki/Alpha_compositing
        - Diffraction: https://en.wikipedia.org/wiki/Diffraction
        - Chromatic aberration: https://en.wikipedia.org/wiki/Chromatic_aberration
        - Screen blending: https://en.wikipedia.org/wiki/Blend_modes#Screen
    """

    class InitSchema(BaseTransformInitSchema):
        flare_roi: tuple[float, float, float, float]
        angle_lower: float | None = Field(ge=0, le=1)
        angle_upper: float | None = Field(ge=0, le=1)

        num_flare_circles_lower: int | None = Field(
            ge=0,
        )
        num_flare_circles_upper: int | None = Field(
            gt=0,
        )
        src_radius: int = Field(gt=1)
        src_color: tuple[int, ...]

        angle_range: Annotated[tuple[float, float], AfterValidator(check_01), AfterValidator(nondecreasing)]

        num_flare_circles_range: Annotated[
            tuple[int, int],
            AfterValidator(check_1plus),
            AfterValidator(nondecreasing),
        ]
        method: Literal["overlay", "physics_based"]

        @model_validator(mode="after")
        def validate_parameters(self) -> Self:
            flare_center_lower_x, flare_center_lower_y, flare_center_upper_x, flare_center_upper_y = self.flare_roi
            if (
                not 0 <= flare_center_lower_x < flare_center_upper_x <= 1
                or not 0 <= flare_center_lower_y < flare_center_upper_y <= 1
            ):
                raise ValueError(f"Invalid flare_roi. Got: {self.flare_roi}")

            if self.angle_lower is not None or self.angle_upper is not None:
                if self.angle_lower is not None:
                    warn(
                        "`angle_lower` deprecated. Use `angle_range` as tuple (angle_lower, angle_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if self.angle_upper is not None:
                    warn(
                        "`angle_upper` deprecated. Use `angle_range` as tuple(angle_lower, angle_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                lower = self.angle_lower if self.angle_lower is not None else self.angle_range[0]
                upper = self.angle_upper if self.angle_upper is not None else self.angle_range[1]
                self.angle_range = (lower, upper)

            if self.num_flare_circles_lower is not None or self.num_flare_circles_upper is not None:
                if self.num_flare_circles_lower is not None:
                    warn(
                        "`num_flare_circles_lower` deprecated. Use `num_flare_circles_range` as tuple"
                        " (num_flare_circles_lower, num_flare_circles_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if self.num_flare_circles_upper is not None:
                    warn(
                        "`num_flare_circles_upper` deprecated. Use `num_flare_circles_range` as tuple"
                        " (num_flare_circles_lower, num_flare_circles_upper) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                lower = (
                    self.num_flare_circles_lower
                    if self.num_flare_circles_lower is not None
                    else self.num_flare_circles_range[0]
                )
                upper = (
                    self.num_flare_circles_upper
                    if self.num_flare_circles_upper is not None
                    else self.num_flare_circles_range[1]
                )
                self.num_flare_circles_range = (lower, upper)

            return self

    def __init__(
        self,
        flare_roi: tuple[float, float, float, float] = (0, 0, 1, 0.5),
        angle_lower: float | None = None,
        angle_upper: float | None = None,
        num_flare_circles_lower: int | None = None,
        num_flare_circles_upper: int | None = None,
        src_radius: int = 400,
        src_color: tuple[int, ...] = (255, 255, 255),
        angle_range: tuple[float, float] = (0, 1),
        num_flare_circles_range: tuple[int, int] = (6, 10),
        method: Literal["overlay", "physics_based"] = "overlay",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.angle_range = angle_range
        self.num_flare_circles_range = num_flare_circles_range

        self.src_radius = src_radius
        self.src_color = src_color
        self.flare_roi = flare_roi
        self.method = method

    def apply(
        self,
        img: np.ndarray,
        flare_center: tuple[float, float],
        circles: list[Any],
        **params: Any,
    ) -> np.ndarray:
        if self.method == "overlay":
            return fmain.add_sun_flare_overlay(
                img,
                flare_center,
                self.src_radius,
                self.src_color,
                circles,
            )
        if self.method == "physics_based":
            non_rgb_error(img)
            return fmain.add_sun_flare_physics_based(
                img,
                flare_center,
                self.src_radius,
                self.src_color,
                circles,
            )

        raise ValueError(f"Invalid method: {self.method}")

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]
        diagonal = math.sqrt(height**2 + width**2)

        angle = 2 * math.pi * self.py_random.uniform(*self.angle_range)

        # Calculate flare center in pixel coordinates
        x_min, y_min, x_max, y_max = self.flare_roi
        flare_center_x = int(width * self.py_random.uniform(x_min, x_max))
        flare_center_y = int(height * self.py_random.uniform(y_min, y_max))

        num_circles = self.py_random.randint(*self.num_flare_circles_range)

        # Calculate parameters relative to image size
        step_size = max(1, int(diagonal * 0.01))  # 1% of diagonal, minimum 1 pixel
        max_radius = max(2, int(height * 0.01))  # 1% of height, minimum 2 pixels
        color_range = int(max(self.src_color) * 0.2)  # 20% of max color value

        def line(t: float) -> tuple[float, float]:
            return (flare_center_x + t * math.cos(angle), flare_center_y + t * math.sin(angle))

        # Generate points along the flare line
        t_range = range(-flare_center_x, width - flare_center_x, step_size)
        points = [line(t) for t in t_range]

        circles = []
        for _ in range(num_circles):
            alpha = self.py_random.uniform(0.05, 0.2)
            point = self.py_random.choice(points)
            rad = self.py_random.randint(1, max_radius)

            # Generate colors relative to src_color
            colors = [self.py_random.randint(max(c - color_range, 0), c) for c in self.src_color]

            circles.append(
                (
                    alpha,
                    (int(point[0]), int(point[1])),
                    pow(rad, 3),
                    tuple(colors),
                ),
            )

        return {
            "circles": circles,
            "flare_center": (flare_center_x, flare_center_y),
        }

    def get_transform_init_args(self) -> dict[str, Any]:
        return {
            "flare_roi": self.flare_roi,
            "angle_range": self.angle_range,
            "num_flare_circles_range": self.num_flare_circles_range,
            "src_radius": self.src_radius,
            "src_color": self.src_color,
        }


class RandomShadow(ImageOnlyTransform):
    """Simulates shadows for the image by reducing the brightness of the image in shadow regions.

    This transform adds realistic shadow effects to images, which can be useful for augmenting
    datasets for outdoor scene analysis, autonomous driving, or any computer vision task where
    shadows may be present.

    Args:
        shadow_roi (tuple[float, float, float, float]): Region of the image where shadows
            will appear (x_min, y_min, x_max, y_max). All values should be in range [0, 1].
            Default: (0, 0.5, 1, 1).
        num_shadows_limit (tuple[int, int]): Lower and upper limits for the possible number of shadows.
            Default: (1, 2).
        shadow_dimension (int): Number of edges in the shadow polygons. Default: 5.
        shadow_intensity_range (tuple[float, float]): Range for the shadow intensity. Larger value
            means darker shadow. Should be two float values between 0 and 1. Default: (0.5, 0.5).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Shadows are created by generating random polygons within the specified ROI and
          reducing the brightness of the image in these areas.
        - The number of shadows, their shapes, and intensities can be randomized for variety.
        - This transform is particularly useful for:
          * Augmenting datasets for outdoor scene understanding
          * Improving robustness of object detection models to shadowed conditions
          * Simulating different lighting conditions in synthetic datasets

    Mathematical Formulation:
        For each shadow:
        1. A polygon with `shadow_dimension` vertices is generated within the shadow ROI.
        2. The shadow intensity a is randomly chosen from `shadow_intensity_range`.
        3. For each pixel (x, y) within the polygon:
           new_pixel_value = original_pixel_value * (1 - a)

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomShadow(p=1.0)
        >>> shadowed_image = transform(image=image)["image"]

        # Custom shadow parameters
        >>> transform = A.RandomShadow(
        ...     shadow_roi=(0.2, 0.2, 0.8, 0.8),
        ...     num_shadows_limit=(2, 4),
        ...     shadow_dimension=8,
        ...     shadow_intensity_range=(0.3, 0.7),
        ...     p=1.0
        ... )
        >>> shadowed_image = transform(image=image)["image"]

        # Combining with other transforms
        >>> transform = A.Compose([
        ...     A.RandomShadow(p=0.5),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> augmented_image = transform(image=image)["image"]

    References:
        - Shadow detection and removal: https://www.sciencedirect.com/science/article/pii/S1047320315002035
        - Shadows in computer vision: https://en.wikipedia.org/wiki/Shadow_detection
    """

    class InitSchema(BaseTransformInitSchema):
        shadow_roi: tuple[float, float, float, float]
        num_shadows_limit: Annotated[tuple[int, int], AfterValidator(check_1plus), AfterValidator(nondecreasing)]
        num_shadows_lower: int | None
        num_shadows_upper: int | None
        shadow_dimension: int = Field(ge=3)

        shadow_intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_01),
            AfterValidator(nondecreasing),
        ]

        @model_validator(mode="after")
        def validate_shadows(self) -> Self:
            if self.num_shadows_lower is not None:
                warn(
                    "`num_shadows_lower` is deprecated. Use `num_shadows_limit` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if self.num_shadows_upper is not None:
                warn(
                    "`num_shadows_upper` is deprecated. Use `num_shadows_limit` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            if self.num_shadows_lower is not None or self.num_shadows_upper is not None:
                num_shadows_lower = (
                    self.num_shadows_lower if self.num_shadows_lower is not None else self.num_shadows_limit[0]
                )
                num_shadows_upper = (
                    self.num_shadows_upper if self.num_shadows_upper is not None else self.num_shadows_limit[1]
                )

                self.num_shadows_limit = (num_shadows_lower, num_shadows_upper)
                self.num_shadows_lower = None
                self.num_shadows_upper = None

            shadow_lower_x, shadow_lower_y, shadow_upper_x, shadow_upper_y = self.shadow_roi

            if not 0 <= shadow_lower_x <= shadow_upper_x <= 1 or not 0 <= shadow_lower_y <= shadow_upper_y <= 1:
                raise ValueError(f"Invalid shadow_roi. Got: {self.shadow_roi}")

            if isinstance(self.shadow_intensity_range, float):
                if not (0 <= self.shadow_intensity_range <= 1):
                    raise ValueError(
                        f"shadow_intensity_range value should be within [0, 1] range. "
                        f"Got: {self.shadow_intensity_range}",
                    )
            elif isinstance(self.shadow_intensity_range, tuple):
                if not (0 <= self.shadow_intensity_range[0] <= self.shadow_intensity_range[1] <= 1):
                    raise ValueError(
                        f"shadow_intensity_range values should be within [0, 1] range and increasing. "
                        f"Got: {self.shadow_intensity_range}",
                    )
            else:
                raise TypeError("shadow_intensity_range should be an float or a tuple of floats.")

            return self

    def __init__(
        self,
        shadow_roi: tuple[float, float, float, float] = (0, 0.5, 1, 1),
        num_shadows_limit: tuple[int, int] = (1, 2),
        num_shadows_lower: int | None = None,
        num_shadows_upper: int | None = None,
        shadow_dimension: int = 5,
        shadow_intensity_range: tuple[float, float] = (0.5, 0.5),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.shadow_roi = shadow_roi
        self.shadow_dimension = shadow_dimension
        self.num_shadows_limit = num_shadows_limit
        self.shadow_intensity_range = shadow_intensity_range

    def apply(
        self,
        img: np.ndarray,
        vertices_list: list[np.ndarray],
        intensities: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        return fmain.add_shadow(img, vertices_list, intensities)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, list[np.ndarray]]:
        height, width = params["shape"][:2]

        num_shadows = self.py_random.randint(*self.num_shadows_limit)

        x_min, y_min, x_max, y_max = self.shadow_roi

        x_min = int(x_min * width)
        x_max = int(x_max * width)
        y_min = int(y_min * height)
        y_max = int(y_max * height)

        vertices_list = [
            np.stack(
                [
                    self.random_generator.integers(x_min, x_max, size=self.shadow_dimension),
                    self.random_generator.integers(y_min, y_max, size=self.shadow_dimension),
                ],
                axis=1,
            )
            for _ in range(num_shadows)
        ]

        # Sample shadow intensity for each shadow
        intensities = self.random_generator.uniform(
            *self.shadow_intensity_range,
            size=num_shadows,
        )

        return {"vertices_list": vertices_list, "intensities": intensities}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "shadow_roi",
            "num_shadows_limit",
            "shadow_dimension",
        )


class RandomToneCurve(ImageOnlyTransform):
    """Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.

    This transform applies a random S-curve to the image's tone curve, adjusting the brightness and contrast
    in a non-linear manner. It can be applied to the entire image or to each channel separately.

    Args:
        scale (float): Standard deviation of the normal distribution used to sample random distances
            to move two control points that modify the image's curve. Values should be in range [0, 1].
            Higher values will result in more dramatic changes to the image. Default: 0.1
        per_channel (bool): If True, the tone curve will be applied to each channel of the input image separately,
            which can lead to color distortion. If False, the same curve is applied to all channels,
            preserving the original color relationships. Default: False
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This transform modifies the image's histogram by applying a smooth, S-shaped curve to it.
        - The S-curve is defined by moving two control points of a quadratic Bézier curve.
        - When per_channel is False, the same curve is applied to all channels, maintaining color balance.
        - When per_channel is True, different curves are applied to each channel, which can create color shifts.
        - This transform can be used to adjust image contrast and brightness in a more natural way than linear
            transforms.
        - The effect can range from subtle contrast adjustments to more dramatic "vintage" or "faded" looks.

    Mathematical Formulation:
        1. Two control points are randomly moved from their default positions (0.25, 0.25) and (0.75, 0.75).
        2. The new positions are sampled from a normal distribution: N(μ, σ²), where μ is the original position
        and alpha is the scale parameter.
        3. These points, along with fixed points at (0, 0) and (1, 1), define a quadratic Bézier curve.
        4. The curve is applied as a lookup table to the image intensities:
           new_intensity = curve(original_intensity)

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Apply a random tone curve to all channels together
        >>> transform = A.RandomToneCurve(scale=0.1, per_channel=False, p=1.0)
        >>> augmented_image = transform(image=image)['image']

        # Apply random tone curves to each channel separately
        >>> transform = A.RandomToneCurve(scale=0.2, per_channel=True, p=1.0)
        >>> augmented_image = transform(image=image)['image']

    References:
        - "What Else Can Fool Deep Learning? Addressing Color Constancy Errors on Deep Neural Network Performance"
          by Mahmoud Afifi and Michael S. Brown, ICCV 2019.
        - Bézier curve: https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Quadratic_B%C3%A9zier_curves
        - Tone mapping: https://en.wikipedia.org/wiki/Tone_mapping
    """

    class InitSchema(BaseTransformInitSchema):
        scale: float = Field(
            ge=0,
            le=1,
        )
        per_channel: bool

    def __init__(
        self,
        scale: float = 0.1,
        per_channel: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.scale = scale
        self.per_channel = per_channel

    def apply(
        self,
        img: np.ndarray,
        low_y: float | np.ndarray,
        high_y: float | np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        return fmain.move_tone_curve(img, low_y, high_y)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"] if "image" in data else data["images"][0]

        num_channels = get_num_channels(image)

        if self.per_channel and num_channels != 1:
            return {
                "low_y": np.clip(self.random_generator.normal(loc=0.25, scale=self.scale, size=(num_channels,)), 0, 1),
                "high_y": np.clip(self.random_generator.normal(loc=0.75, scale=self.scale, size=(num_channels,)), 0, 1),
            }
        # Same values for all channels
        low_y = np.clip(self.random_generator.normal(loc=0.25, scale=self.scale), 0, 1)
        high_y = np.clip(self.random_generator.normal(loc=0.75, scale=self.scale), 0, 1)

        return {"low_y": low_y, "high_y": high_y}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "scale", "per_channel"


class HueSaturationValue(ImageOnlyTransform):
    """Randomly change hue, saturation and value of the input image.

    This transform adjusts the HSV (Hue, Saturation, Value) channels of an input RGB image.
    It allows for independent control over each channel, providing a wide range of color
    and brightness modifications.

    Args:
        hue_shift_limit (float | tuple[float, float]): Range for changing hue.
            If a single float value is provided, the range will be (-hue_shift_limit, hue_shift_limit).
            Values should be in the range [-180, 180]. Default: (-20, 20).

        sat_shift_limit (float | tuple[float, float]): Range for changing saturation.
            If a single float value is provided, the range will be (-sat_shift_limit, sat_shift_limit).
            Values should be in the range [-255, 255]. Default: (-30, 30).

        val_shift_limit (float | tuple[float, float]): Range for changing value (brightness).
            If a single float value is provided, the range will be (-val_shift_limit, val_shift_limit).
            Values should be in the range [-255, 255]. Default: (-20, 20).

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - The transform first converts the input RGB image to the HSV color space.
        - Each channel (Hue, Saturation, Value) is adjusted independently.
        - Hue is circular, so it wraps around at 180 degrees.
        - For float32 images, the shift values are applied as percentages of the full range.
        - This transform is particularly useful for color augmentation and simulating
          different lighting conditions.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.HueSaturationValue(
        ...     hue_shift_limit=20,
        ...     sat_shift_limit=30,
        ...     val_shift_limit=20,
        ...     p=0.7
        ... )
        >>> result = transform(image=image)
        >>> augmented_image = result["image"]

    References:
        - HSV color space: https://en.wikipedia.org/wiki/HSL_and_HSV
    """

    class InitSchema(BaseTransformInitSchema):
        hue_shift_limit: SymmetricRangeType
        sat_shift_limit: SymmetricRangeType
        val_shift_limit: SymmetricRangeType

    def __init__(
        self,
        hue_shift_limit: ScaleFloatType = (-20, 20),
        sat_shift_limit: ScaleFloatType = (-30, 30),
        val_shift_limit: ScaleFloatType = (-20, 20),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.hue_shift_limit = cast(tuple[float, float], hue_shift_limit)
        self.sat_shift_limit = cast(tuple[float, float], sat_shift_limit)
        self.val_shift_limit = cast(tuple[float, float], val_shift_limit)

    def apply(
        self,
        img: np.ndarray,
        hue_shift: int,
        sat_shift: int,
        val_shift: int,
        **params: Any,
    ) -> np.ndarray:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "HueSaturationValue transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)
        return fmain.shift_hsv(img, hue_shift, sat_shift, val_shift)

    def get_params(self) -> dict[str, float]:
        return {
            "hue_shift": self.py_random.uniform(*self.hue_shift_limit),
            "sat_shift": self.py_random.uniform(*self.sat_shift_limit),
            "val_shift": self.py_random.uniform(*self.val_shift_limit),
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "hue_shift_limit", "sat_shift_limit", "val_shift_limit"


class Solarize(ImageOnlyTransform):
    """Invert all pixel values above a threshold.

    This transform applies a solarization effect to the input image. Solarization is a phenomenon in
    photography in which the image recorded on a negative or on a photographic print is wholly or
    partially reversed in tone. Dark areas appear light or light areas appear dark.

    In this implementation, all pixel values above a threshold are inverted.

    Args:
        threshold (float | tuple[float, float]): Range for solarizing threshold.
            If threshold is a single int, the range will be [threshold, threshold].
            If it's a tuple of (min, max), the range will be [min, max].
            The threshold should be in the range [0, 255] for uint8 images or [0, 1.0] for float images.
            Default: 128.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - For uint8 images, pixel values above the threshold are inverted as: 255 - pixel_value
        - For float32 images, pixel values above the threshold are inverted as: 1.0 - pixel_value
        - The threshold is applied to each channel independently
        - This transform can create interesting artistic effects or be used for data augmentation

    Raises:
        TypeError: If the input image data type is not supported.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        # Solarize uint8 image with fixed threshold
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Solarize(threshold=128, p=1.0)
        >>> solarized_image = transform(image=image)['image']
        >>>
        # Solarize uint8 image with random threshold
        >>> transform = A.Solarize(threshold=(100, 200), p=1.0)
        >>> solarized_image = transform(image=image)['image']
        >>>
        # Solarize float32 image
        >>> image = np.random.rand(100, 100, 3).astype(np.float32)
        >>> transform = A.Solarize(threshold=0.5, p=1.0)
        >>> solarized_image = transform(image=image)['image']

    Mathematical Formulation:
        For each pixel value p and threshold t:
        if p > t:
            p_new = max_value - p
        else:
            p_new = p

        Where max_value is 255 for uint8 images and 1.0 for float32 images.

    See Also:
        Invert: For inverting all pixel values regardless of a threshold.
    """

    class InitSchema(BaseTransformInitSchema):
        threshold: Annotated[ScaleFloatType, AfterValidator(repeat_if_scalar), AfterValidator(check_0plus)]

    def __init__(self, threshold: ScaleFloatType = (128, 128), p: float = 0.5, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        self.threshold = cast(tuple[float, float], threshold)

    def apply(self, img: np.ndarray, threshold: int, **params: Any) -> np.ndarray:
        return fmain.solarize(img, threshold)

    def get_params(self) -> dict[str, float]:
        return {"threshold": self.py_random.uniform(*self.threshold)}

    def get_transform_init_args_names(self) -> tuple[str]:
        return ("threshold",)


class Posterize(ImageOnlyTransform):
    """Reduces the number of bits for each color channel in the image.

    This transform applies color posterization, a technique that reduces the number of distinct
    colors used in an image. It works by lowering the number of bits used to represent each
    color channel, effectively creating a "poster-like" effect with fewer color gradations.

    Args:
        num_bits (int | tuple[int, int] | list[int] | list[tuple[int, int]]):
            Defines the number of bits to keep for each color channel. Can be specified in several ways:
            - Single int: Same number of bits for all channels. Range: [0, 8].
            - tuple of two ints: (min_bits, max_bits) to randomly choose from. Range for each: [0, 8].
            - list of three ints: Specific number of bits for each channel [r_bits, g_bits, b_bits].
            - list of three tuples: Ranges for each channel [(r_min, r_max), (g_min, g_max), (b_min, b_max)].
            Default: 4

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The effect becomes more pronounced as the number of bits is reduced.
        - Using 0 bits for a channel will reduce it to a single color (usually black).
        - Using 8 bits leaves the channel unchanged.
        - This transform can create interesting artistic effects or be used for image compression simulation.
        - Posterization is particularly useful for:
          * Creating stylized or retro-looking images
          * Reducing the color palette for specific artistic effects
          * Simulating the look of older or lower-quality digital images
          * Data augmentation in scenarios where color depth might vary

    Mathematical Background:
        For an 8-bit color channel, posterization to n bits can be expressed as:
        new_value = (old_value >> (8 - n)) << (8 - n)
        This operation keeps the n most significant bits and sets the rest to zero.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Posterize all channels to 3 bits
        >>> transform = A.Posterize(num_bits=3, p=1.0)
        >>> posterized_image = transform(image=image)["image"]

        # Randomly posterize between 2 and 5 bits
        >>> transform = A.Posterize(num_bits=(2, 5), p=1.0)
        >>> posterized_image = transform(image=image)["image"]

        # Different bits for each channel
        >>> transform = A.Posterize(num_bits=[3, 5, 2], p=1.0)
        >>> posterized_image = transform(image=image)["image"]

        # Range of bits for each channel
        >>> transform = A.Posterize(num_bits=[(1, 3), (3, 5), (2, 4)], p=1.0)
        >>> posterized_image = transform(image=image)["image"]

    References:
        - Color Quantization: https://en.wikipedia.org/wiki/Color_quantization
        - Posterization: https://en.wikipedia.org/wiki/Posterization
    """

    class InitSchema(BaseTransformInitSchema):
        num_bits: Annotated[
            int | tuple[int, int] | list[tuple[int, int]],
            Field(default=4, description="Number of high bits"),
        ]

        @field_validator("num_bits")
        @classmethod
        def validate_num_bits(cls, num_bits: Any) -> tuple[int, int] | list[tuple[int, int]]:
            if isinstance(num_bits, int):
                return to_tuple(num_bits, num_bits)
            if isinstance(num_bits, Sequence) and len(num_bits) == NUM_BITS_ARRAY_LENGTH:
                return [to_tuple(i, 0) for i in num_bits]
            return cast(tuple[int, int], to_tuple(num_bits, 0))

    def __init__(
        self,
        num_bits: int | tuple[int, int] | list[tuple[int, int]] = 4,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.num_bits = cast(Union[tuple[int, ...], list[tuple[int, ...]]], num_bits)

    def apply(self, img: np.ndarray, num_bits: int, **params: Any) -> np.ndarray:
        return fmain.posterize(img, num_bits)

    def get_params(self) -> dict[str, Any]:
        if len(self.num_bits) == NUM_BITS_ARRAY_LENGTH:
            return {"num_bits": [self.py_random.randint(int(i[0]), int(i[1])) for i in self.num_bits]}  # type: ignore[index]
        num_bits = self.num_bits
        return {"num_bits": self.py_random.randint(int(num_bits[0]), int(num_bits[1]))}  # type: ignore[arg-type]

    def get_transform_init_args_names(self) -> tuple[str]:
        return ("num_bits",)


class Equalize(ImageOnlyTransform):
    """Equalize the image histogram.

    This transform applies histogram equalization to the input image. Histogram equalization
    is a method in image processing of contrast adjustment using the image's histogram.

    Args:
        mode (Literal['cv', 'pil']): Use OpenCV or Pillow equalization method.
            Default: 'cv'
        by_channels (bool): If True, use equalization by channels separately,
            else convert image to YCbCr representation and use equalization by `Y` channel.
            Default: True
        mask (np.ndarray, callable): If given, only the pixels selected by
            the mask are included in the analysis. Can be:
            - A 1-channel or 3-channel numpy array of the same size as the input image.
            - A callable (function) that generates a mask. The function should accept 'image'
              as its first argument, and can accept additional arguments specified in mask_params.
            Default: None
        mask_params (list[str]): Additional parameters to pass to the mask function.
            These parameters will be taken from the data dict passed to __call__.
            Default: ()
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - When mode='cv', OpenCV's equalizeHist() function is used.
        - When mode='pil', Pillow's equalize() function is used.
        - The 'by_channels' parameter determines whether equalization is applied to each color channel
          independently (True) or to the luminance channel only (False).
        - If a mask is provided as a numpy array, it should have the same height and width as the input image.
        - If a mask is provided as a function, it allows for dynamic mask generation based on the input image
          and additional parameters. This is useful for scenarios where the mask depends on the image content
          or external data (e.g., bounding boxes, segmentation masks).

    Mask Function:
        When mask is a callable, it should have the following signature:
        mask_func(image, *args) -> np.ndarray

        - image: The input image (numpy array)
        - *args: Additional arguments as specified in mask_params

        The function should return a numpy array of the same height and width as the input image,
        where non-zero pixels indicate areas to be equalized.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        >>> # Using a static mask
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> transform = A.Equalize(mask=mask, p=1.0)
        >>> result = transform(image=image)
        >>>
        >>> # Using a dynamic mask function
        >>> def mask_func(image, bboxes):
        ...     mask = np.ones_like(image[:, :, 0], dtype=np.uint8)
        ...     for bbox in bboxes:
        ...         x1, y1, x2, y2 = map(int, bbox)
        ...         mask[y1:y2, x1:x2] = 0  # Exclude areas inside bounding boxes
        ...     return mask
        >>>
        >>> transform = A.Equalize(mask=mask_func, mask_params=['bboxes'], p=1.0)
        >>> bboxes = [(10, 10, 50, 50), (60, 60, 90, 90)]  # Example bounding boxes
        >>> result = transform(image=image, bboxes=bboxes)

    References:
        - OpenCV equalizeHist: https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga7e54091f0c937d49bf84152a16f76d6e
        - Pillow ImageOps.equalize: https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.equalize
        - Histogram Equalization: https://en.wikipedia.org/wiki/Histogram_equalization
    """

    class InitSchema(BaseTransformInitSchema):
        mode: ImageMode
        by_channels: bool
        mask: np.ndarray | Callable[..., Any] | None
        mask_params: Sequence[str]

    def __init__(
        self,
        mode: ImageMode = "cv",
        by_channels: bool = True,
        mask: np.ndarray | Callable[..., Any] | None = None,
        mask_params: Sequence[str] = (),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.mode = mode
        self.by_channels = by_channels
        self.mask = mask
        self.mask_params = mask_params

    def apply(self, img: np.ndarray, mask: np.ndarray, **params: Any) -> np.ndarray:
        return fmain.equalize(img, mode=self.mode, by_channels=self.by_channels, mask=mask)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        if not callable(self.mask):
            return {"mask": self.mask}

        mask_params = {"image": data["image"]}
        for key in self.mask_params:
            if key not in data:
                raise KeyError(f"Required parameter '{key}' for mask function is missing in data.")
            mask_params[key] = data[key]

        return {"mask": self.mask(**mask_params)}

    @property
    def targets_as_params(self) -> list[str]:
        return ["image", *list(self.mask_params)]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "mode", "by_channels", "mask", "mask_params"


class RGBShift(ImageOnlyTransform):
    """Randomly shift values for each channel of the input RGB image.

    Args:
        r_shift_limit ((int, int) or int): range for changing values for the red channel. If r_shift_limit is a
            single int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).
        g_shift_limit ((int, int) or int): range for changing values for the green channel. If g_shift_limit is a
            single int, the range will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).
        b_shift_limit ((int, int) or int): range for changing values for the blue channel. If b_shift_limit is a
            single int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - For uint8 images, the shift values represent absolute pixel values in the range [0, 255].
          For example, a shift of 20 for a uint8 image would add 20 to the corresponding channel.
        - For float32 images, the shift values represent fractions of the full value range [0, 1].
          For example, a shift of 0.1 for a float32 image would add 0.1 to the corresponding channel.
        - The shift values are applied independently to each channel.
        - After applying the shift, values are clipped to the valid range for the image dtype:
          [0, 255] for uint8 and [0, 1] for float32.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        # Shift RGB channels for uint8 image
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1.0)
        >>> shifted_image = transform(image=image)['image']
        >>>
        # Shift RGB channels for float32 image
        >>> image = np.random.rand(100, 100, 3).astype(np.float32)
        >>> transform = A.RGBShift(r_shift_limit=0.1, g_shift_limit=0.1, b_shift_limit=0.1, p=1.0)
        >>> shifted_image = transform(image=image)['image']

    """

    class InitSchema(BaseTransformInitSchema):
        r_shift_limit: SymmetricRangeType
        g_shift_limit: SymmetricRangeType
        b_shift_limit: SymmetricRangeType

    def __init__(
        self,
        r_shift_limit: ScaleFloatType = (-20, 20),
        g_shift_limit: ScaleFloatType = (-20, 20),
        b_shift_limit: ScaleFloatType = (-20, 20),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.r_shift_limit = cast(tuple[float, float], r_shift_limit)
        self.g_shift_limit = cast(tuple[float, float], g_shift_limit)
        self.b_shift_limit = cast(tuple[float, float], b_shift_limit)

    def apply(self, img: np.ndarray, shift: np.ndarray, **params: Any) -> np.ndarray:
        non_rgb_error(img)
        return albucore.add_vector(img, shift, inplace=False)

    def get_params(self) -> dict[str, Any]:
        return {
            "shift": np.array(
                [
                    self.py_random.uniform(*self.r_shift_limit),
                    self.py_random.uniform(*self.g_shift_limit),
                    self.py_random.uniform(*self.b_shift_limit),
                ],
            ),
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "r_shift_limit", "g_shift_limit", "b_shift_limit"


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly changes the brightness and contrast of the input image.

    This transform adjusts the brightness and contrast of an image simultaneously, allowing for
    a wide range of lighting and contrast variations. It's particularly useful for data augmentation
    in computer vision tasks, helping models become more robust to different lighting conditions.

    Args:
        brightness_limit (float | tuple[float, float]): Factor range for changing brightness.
            If a single float value is provided, the range will be (-brightness_limit, brightness_limit).
            Values should typically be in the range [-1.0, 1.0], where 0 means no change,
            1.0 means maximum brightness, and -1.0 means minimum brightness.
            Default: (-0.2, 0.2).

        contrast_limit (float | tuple[float, float]): Factor range for changing contrast.
            If a single float value is provided, the range will be (-contrast_limit, contrast_limit).
            Values should typically be in the range [-1.0, 1.0], where 0 means no change,
            1.0 means maximum increase in contrast, and -1.0 means maximum decrease in contrast.
            Default: (-0.2, 0.2).

        brightness_by_max (bool): If True, adjusts brightness by scaling pixel values up to the
            maximum value of the image's dtype. If False, uses the mean pixel value for adjustment.
            Default: True.

        ensure_safe_range (bool): If True, adjusts alpha and beta to prevent overflow/underflow.
            This ensures output values stay within the valid range for the image dtype without clipping.
            Default: False.

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The order of operation is: contrast adjustment, then brightness adjustment.
        - For uint8 images, the output is clipped to [0, 255] range.
        - For float32 images, the output is clipped to [0, 1] range.
        - The `brightness_by_max` parameter affects how brightness is adjusted:
          * If True, brightness adjustment is more pronounced and can lead to more saturated results.
          * If False, brightness adjustment is more subtle and preserves the overall lighting better.
        - This transform is useful for:
          * Simulating different lighting conditions
          * Enhancing low-light or overexposed images
          * Data augmentation to improve model robustness

    Mathematical Formulation:
        Let a be the contrast adjustment factor and β be the brightness adjustment factor.
        For each pixel value x:
        1. Contrast adjustment: x' = clip((x - mean) * (1 + a) + mean)
        2. Brightness adjustment:
           If brightness_by_max is True:  x'' = clip(x' * (1 + β))
           If brightness_by_max is False: x'' = clip(x' + β * max_value)
        Where clip() ensures values stay within the valid range for the image dtype.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomBrightnessContrast(p=1.0)
        >>> augmented_image = transform(image=image)["image"]

        # Custom brightness and contrast limits
        >>> transform = A.RandomBrightnessContrast(
        ...     brightness_limit=0.3,
        ...     contrast_limit=0.3,
        ...     p=1.0
        ... )
        >>> augmented_image = transform(image=image)["image"]

        # Adjust brightness based on mean value
        >>> transform = A.RandomBrightnessContrast(
        ...     brightness_limit=0.2,
        ...     contrast_limit=0.2,
        ...     brightness_by_max=False,
        ...     p=1.0
        ... )
        >>> augmented_image = transform(image=image)["image"]

    References:
        - Brightness: https://en.wikipedia.org/wiki/Brightness
        - Contrast: https://en.wikipedia.org/wiki/Contrast_(vision)
    """

    class InitSchema(BaseTransformInitSchema):
        brightness_limit: SymmetricRangeType
        contrast_limit: SymmetricRangeType
        brightness_by_max: bool
        ensure_safe_range: bool

    def __init__(
        self,
        brightness_limit: ScaleFloatType = (-0.2, 0.2),
        contrast_limit: ScaleFloatType = (-0.2, 0.2),
        brightness_by_max: bool = True,
        ensure_safe_range: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.brightness_limit = cast(tuple[float, float], brightness_limit)
        self.contrast_limit = cast(tuple[float, float], contrast_limit)
        self.brightness_by_max = brightness_by_max
        self.ensure_safe_range = ensure_safe_range

    def apply(self, img: np.ndarray, alpha: float, beta: float, **params: Any) -> np.ndarray:
        return albucore.multiply_add(img, alpha, beta, inplace=False)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, float]:
        image = data["image"] if "image" in data else data["images"][0]

        # Sample initial values
        alpha = 1.0 + self.py_random.uniform(*self.contrast_limit)
        beta = self.py_random.uniform(*self.brightness_limit)

        max_value = MAX_VALUES_BY_DTYPE[image.dtype]
        # Scale beta according to brightness_by_max setting
        beta = beta * max_value if self.brightness_by_max else beta * np.mean(image)

        # Clip values to safe ranges if needed
        if self.ensure_safe_range:
            alpha, beta = fmain.get_safe_brightness_contrast_params(alpha, beta, max_value)

        return {
            "alpha": alpha,
            "beta": beta,
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "brightness_limit", "contrast_limit", "brightness_by_max", "ensure_safe_range"


class GaussNoise(ImageOnlyTransform):
    """Apply Gaussian noise to the input image.

    Args:
        var_limit (tuple[float, float] | float): Variance range for noise. If var_limit is a single float value,
            the range will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): Mean of the noise. Default: 0.
        per_channel (bool): If True, noise will be sampled for each channel independently.
            Otherwise, the noise will be sampled once for all channels. Default: True.
        noise_scale_factor (float): Scaling factor for noise generation. Value should be in the range (0, 1].
            When set to 1, noise is sampled for each pixel independently. If less, noise is sampled for a smaller size
            and resized to fit the shape of the image. Smaller values make the transform faster. Default: 1.0.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Returns:
        numpy.ndarray: Image with applied Gaussian noise.

    Note:
        - The noise is generated in the same range as the input image.
        - For uint8 input images, the noise is generated in the range [0, 255].
        - For float32 input images, the noise is generated in the range [0, 1].
        - The resulting image is clipped to keep its values in the input range.
        - Setting per_channel=False is faster but applies the same noise to all channels.
        - The noise_scale_factor parameter allows for a trade-off between transform speed and noise granularity.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>>
        >>> # Apply Gaussian noise with default parameters
        >>> transform = A.GaussNoise(p=1.0)
        >>> noisy_image = transform(image=image)['image']
        >>>
        >>> # Apply Gaussian noise with custom variance range and mean
        >>> transform = A.GaussNoise(var_limit=(50.0, 100.0), mean=10, p=1.0)
        >>> noisy_image = transform(image=image)['image']
        >>>
        >>> # Apply the same noise to all channels
        >>> transform = A.GaussNoise(per_channel=False, p=1.0)
        >>> noisy_image = transform(image=image)['image']
        >>>
        >>> # Apply noise with reduced granularity for faster processing
        >>> transform = A.GaussNoise(noise_scale_factor=0.5, p=1.0)
        >>> noisy_image = transform(image=image)['image']

    """

    class InitSchema(BaseTransformInitSchema):
        var_limit: NonNegativeFloatRangeType
        mean: float
        per_channel: bool
        noise_scale_factor: float = Field(gt=0, le=1)

    def __init__(
        self,
        var_limit: ScaleFloatType = (10.0, 50.0),
        mean: float = 0,
        per_channel: bool = True,
        noise_scale_factor: float = 1,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.var_limit = cast(tuple[float, float], var_limit)
        self.mean = mean
        self.per_channel = per_channel
        self.noise_scale_factor = noise_scale_factor

    def apply(self, img: np.ndarray, gauss: np.ndarray, **params: Any) -> np.ndarray:
        return fmain.add_noise(img, gauss)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, float]:
        image = data["image"] if "image" in data else data["images"][0]
        var = self.py_random.uniform(*self.var_limit)
        sigma = math.sqrt(var)

        if self.per_channel:
            target_shape = image.shape
            if self.noise_scale_factor == 1:
                gauss = self.random_generator.normal(self.mean, sigma, target_shape)
            else:
                gauss = fmain.generate_approx_gaussian_noise(
                    target_shape,
                    self.mean,
                    sigma,
                    self.noise_scale_factor,
                    self.random_generator,
                )
        else:
            target_shape = image.shape[:2]
            if self.noise_scale_factor == 1:
                gauss = self.random_generator.normal(self.mean, sigma, target_shape)
            else:
                gauss = fmain.generate_approx_gaussian_noise(
                    target_shape,
                    self.mean,
                    sigma,
                    self.noise_scale_factor,
                    self.random_generator,
                )

            if image.ndim > MONO_CHANNEL_DIMENSIONS:
                gauss = np.expand_dims(gauss, -1)

        return {"gauss": gauss}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "var_limit", "per_channel", "mean", "noise_scale_factor"


class ISONoise(ImageOnlyTransform):
    """Applies camera sensor noise to the input image, simulating high ISO settings.

    This transform adds random noise to an image, mimicking the effect of using high ISO settings
    in digital photography. It simulates two main components of ISO noise:
    1. Color noise: random shifts in color hue
    2. Luminance noise: random variations in pixel intensity

    Args:
        color_shift (tuple[float, float]): Range for changing color hue.
            Values should be in the range [0, 1], where 1 represents a full 360° hue rotation.
            Default: (0.01, 0.05)

        intensity (tuple[float, float]): Range for the noise intensity.
            Higher values increase the strength of both color and luminance noise.
            Default: (0.1, 0.5)

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - This transform only works with RGB images. It will raise a TypeError if applied to
          non-RGB images.
        - The color shift is applied in the HSV color space, affecting the hue channel.
        - Luminance noise is added to all channels independently.
        - This transform can be useful for data augmentation in low-light scenarios or when
          training models to be robust against noisy inputs.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)
        >>> result = transform(image=image)
        >>> noisy_image = result["image"]

    References:
        - ISO noise in digital photography:
          https://en.wikipedia.org/wiki/Image_noise#In_digital_cameras
    """

    class InitSchema(BaseTransformInitSchema):
        color_shift: Annotated[tuple[float, float], AfterValidator(check_01), AfterValidator(nondecreasing)]
        intensity: Annotated[tuple[float, float], AfterValidator(check_0plus), AfterValidator(nondecreasing)]

    def __init__(
        self,
        color_shift: tuple[float, float] = (0.01, 0.05),
        intensity: tuple[float, float] = (0.1, 0.5),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.intensity = intensity
        self.color_shift = color_shift

    def apply(
        self,
        img: np.ndarray,
        color_shift: float,
        intensity: float,
        random_seed: int,
        **params: Any,
    ) -> np.ndarray:
        non_rgb_error(img)
        return fmain.iso_noise(img, color_shift, intensity, np.random.default_rng(random_seed))

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        random_seed = self.random_generator.integers(0, 2**32 - 1)
        return {
            "color_shift": self.py_random.uniform(*self.color_shift),
            "intensity": self.py_random.uniform(*self.intensity),
            "random_seed": random_seed,
        }

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "intensity", "color_shift"


class CLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.

    CLAHE is an advanced method of improving the contrast in an image. Unlike regular histogram
    equalization, which operates on the entire image, CLAHE operates on small regions (tiles)
    in the image. This results in a more balanced equalization, preventing over-amplification
    of contrast in areas with initially low contrast.

    Args:
        clip_limit (tuple[float, float] | float): Controls the contrast enhancement limit.
            - If a single float is provided, the range will be (1, clip_limit).
            - If a tuple of two floats is provided, it defines the range for random selection.
            Higher values allow for more contrast enhancement, but may also increase noise.
            Default: (1, 4)

        tile_grid_size (tuple[int, int]): Defines the number of tiles in the row and column directions.
            Format is (rows, columns). Smaller tile sizes can lead to more localized enhancements,
            while larger sizes give results closer to global histogram equalization.
            Default: (8, 8)

        p (float): Probability of applying the transform. Default: 0.5

    Notes:
        - Supports only RGB or grayscale images.
        - For color images, CLAHE is applied to the L channel in the LAB color space.
        - The clip limit determines the maximum slope of the cumulative histogram. A lower
          clip limit will result in more contrast limiting.
        - Tile grid size affects the adaptiveness of the method. More tiles increase local
          adaptiveness but can lead to an unnatural look if set too high.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        1, 3

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0)
        >>> result = transform(image=image)
        >>> clahe_image = result["image"]

    References:
        - https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
        - Zuiderveld, Karel. "Contrast Limited Adaptive Histogram Equalization."
          Graphic Gems IV. Academic Press Professional, Inc., 1994.
    """

    class InitSchema(BaseTransformInitSchema):
        clip_limit: OnePlusFloatRangeType
        tile_grid_size: Annotated[tuple[int, int], AfterValidator(check_1plus)]

    def __init__(
        self,
        clip_limit: ScaleFloatType = 4.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.clip_limit = cast(tuple[float, float], clip_limit)
        self.tile_grid_size = tile_grid_size

    def apply(self, img: np.ndarray, clip_limit: float, **params: Any) -> np.ndarray:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "CLAHE transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)

        return fmain.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self) -> dict[str, float]:
        return {"clip_limit": self.py_random.uniform(*self.clip_limit)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("clip_limit", "tile_grid_size")


class ChannelShuffle(ImageOnlyTransform):
    """Randomly rearrange channels of the image.

    Args:
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    def apply(self, img: np.ndarray, channels_shuffled: tuple[int, ...], **params: Any) -> np.ndarray:
        return fmain.channel_shuffle(img, channels_shuffled)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        ch_arr = list(range(params["shape"][2]))
        self.random_generator.shuffle(ch_arr)
        return {"channels_shuffled": ch_arr}

    def get_transform_init_args_names(self) -> tuple[()]:
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

    Number of channels:
        Any

    """

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return fmain.invert(img)

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()


class RandomGamma(ImageOnlyTransform):
    """Applies random gamma correction to the input image.

    Gamma correction, or simply gamma, is a nonlinear operation used to encode and decode luminance
    or tristimulus values in imaging systems. This transform can adjust the brightness of an image
    while preserving the relative differences between darker and lighter areas, making it useful
    for simulating different lighting conditions or correcting for display characteristics.

    Args:
        gamma_limit (float | tuple[float, float]): If gamma_limit is a single float value, the range
            will be (1, gamma_limit). If it's a tuple of two floats, they will serve as
            the lower and upper bounds for gamma adjustment. Values are in terms of percentage change,
            e.g., (80, 120) means the gamma will be between 80% and 120% of the original.
            Default: (80, 120).
        eps: A small value added to the gamma to avoid division by zero or log of zero errors.
            Default: 1e-7.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The gamma correction is applied using the formula: output = input^gamma
        - Gamma values > 1 will make the image darker, while values < 1 will make it brighter
        - This transform is particularly useful for:
          * Simulating different lighting conditions
          * Correcting for non-linear display characteristics
          * Enhancing contrast in certain regions of the image
          * Data augmentation in computer vision tasks

    Mathematical Formulation:
        Let I be the input image and G (gamma) be the correction factor.
        The gamma correction is applied as follows:
        1. Normalize the image to [0, 1] range: I_norm = I / 255 (for uint8 images)
        2. Apply gamma correction: I_corrected = I_norm ^ (1 / G)
        3. Scale back to original range: output = I_corrected * 255 (for uint8 images)

        The actual gamma value used is calculated as:
        G = 1 + (random_value / 100), where random_value is sampled from gamma_limit range.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Default usage
        >>> transform = A.RandomGamma(p=1.0)
        >>> augmented_image = transform(image=image)["image"]

        # Custom gamma range
        >>> transform = A.RandomGamma(gamma_limit=(50, 150), p=1.0)
        >>> augmented_image = transform(image=image)["image"]

        # Applying with other transforms
        >>> transform = A.Compose([
        ...     A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> augmented_image = transform(image=image)["image"]

    References:
        - Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction
        - Power law (Gamma) encoding: https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
    """

    class InitSchema(BaseTransformInitSchema):
        gamma_limit: OnePlusFloatRangeType

    def __init__(
        self,
        gamma_limit: ScaleFloatType = (80, 120),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)
        self.gamma_limit = cast(tuple[float, float], gamma_limit)

    def apply(self, img: np.ndarray, gamma: float, **params: Any) -> np.ndarray:
        return fmain.gamma_transform(img, gamma=gamma)

    def get_params(self) -> dict[str, float]:
        return {"gamma": self.py_random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("gamma_limit",)


class ToGray(ImageOnlyTransform):
    """Convert an image to grayscale and optionally replicate the grayscale channel.

    This transform first converts a color image to a single-channel grayscale image using various methods,
    then replicates the grayscale channel if num_output_channels is greater than 1.

    Args:
        num_output_channels (int): The number of channels in the output image. If greater than 1,
            the grayscale channel will be replicated. Default: 3.
        method (Literal["weighted_average", "from_lab", "desaturation", "average", "max", "pca"]):
            The method used for grayscale conversion:
            - "weighted_average": Uses a weighted sum of RGB channels (0.299R + 0.587G + 0.114B).
              Works only with 3-channel images. Provides realistic results based on human perception.
            - "from_lab": Extracts the L channel from the LAB color space.
              Works only with 3-channel images. Gives perceptually uniform results.
            - "desaturation": Averages the maximum and minimum values across channels.
              Works with any number of channels. Fast but may not preserve perceived brightness well.
            - "average": Simple average of all channels.
              Works with any number of channels. Fast but may not give realistic results.
            - "max": Takes the maximum value across all channels.
              Works with any number of channels. Tends to produce brighter results.
            - "pca": Applies Principal Component Analysis to reduce channels.
              Works with any number of channels. Can preserve more information but is computationally intensive.
        p (float): Probability of applying the transform. Default: 0.5.

    Raises:
        TypeError: If the input image doesn't have 3 channels for methods that require it.

    Note:
        - The transform first converts the input image to single-channel grayscale, then replicates
          this channel if num_output_channels > 1.
        - "weighted_average" and "from_lab" are typically used in image processing and computer vision
          applications where accurate representation of human perception is important.
        - "desaturation" and "average" are often used in simple image manipulation tools or when
          computational speed is a priority.
        - "max" method can be useful in scenarios where preserving bright features is important,
          such as in some medical imaging applications.
        - "pca" might be used in advanced image analysis tasks or when dealing with hyperspectral images.

    Image types:
        uint8, float32

    Returns:
        np.ndarray: Grayscale image with the specified number of channels.
    """

    class InitSchema(BaseTransformInitSchema):
        num_output_channels: int = Field(default=3, description="The number of output channels.", ge=1)
        method: Literal["weighted_average", "from_lab", "desaturation", "average", "max", "pca"]

    def __init__(
        self,
        num_output_channels: int = 3,
        method: Literal["weighted_average", "from_lab", "desaturation", "average", "max", "pca"] = "weighted_average",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.num_output_channels = num_output_channels
        self.method = method

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if is_grayscale_image(img):
            warnings.warn("The image is already gray.", stacklevel=2)
            return img

        num_channels = get_num_channels(img)

        if num_channels != NUM_RGB_CHANNELS and self.method not in {"desaturation", "average", "max", "pca"}:
            msg = "ToGray transformation expects 3-channel images."
            raise TypeError(msg)

        return fmain.to_gray(img, self.num_output_channels, self.method)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "num_output_channels", "method"


class ToRGB(ImageOnlyTransform):
    """Convert an input image from grayscale to RGB format.

    Args:
        num_output_channels (int): The number of channels in the output image. Default: 3.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        1

    Note:
        - For single-channel (grayscale) images, the channel is replicated to create an RGB image.
        - If the input is already a 3-channel RGB image, it is returned unchanged.
        - This transform does not change the data type of the image (e.g., uint8 remains uint8).

    Raises:
        TypeError: If the input image has more than 1 channel.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Convert a grayscale image to RGB
        >>> transform = A.Compose([A.ToRGB(p=1.0)])
        >>> grayscale_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> rgb_image = transform(image=grayscale_image)['image']
        >>> assert rgb_image.shape == (100, 100, 3)
    """

    class InitSchema(BaseTransformInitSchema):
        num_output_channels: int = Field(ge=1)

    def __init__(self, num_output_channels: int = 3, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)

        self.num_output_channels = num_output_channels

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        if is_rgb_image(img):
            warnings.warn("The image is already an RGB.", stacklevel=2)
            return np.ascontiguousarray(img)
        if not is_grayscale_image(img):
            msg = "ToRGB transformation expects 2-dim images or 3-dim with the last dimension equal to 1."
            raise TypeError(msg)

        return fmain.grayscale_to_multichannel(img, num_output_channels=self.num_output_channels)

    def get_transform_init_args_names(self) -> tuple[str]:
        return ("num_output_channels",)


class ToSepia(ImageOnlyTransform):
    """Apply a sepia filter to the input image.

    This transform converts a color image to a sepia tone, giving it a warm, brownish tint
    that is reminiscent of old photographs. The sepia effect is achieved by applying a
    specific color transformation matrix to the RGB channels of the input image.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - This transform only works with RGB images (3 channels).
        - The sepia effect is created using a fixed color transformation matrix:
          [[0.393, 0.769, 0.189],
           [0.349, 0.686, 0.168],
           [0.272, 0.534, 0.131]]
        - The output image will have the same data type as the input image.
        - For float32 images, ensure the input values are in the range [0, 1].

    Raises:
        TypeError: If the input image is not a 3-channel RGB image.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        # Apply sepia effect to a uint8 image
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ToSepia(p=1.0)
        >>> sepia_image = transform(image=image)['image']
        >>> assert sepia_image.shape == image.shape
        >>> assert sepia_image.dtype == np.uint8
        >>>
        # Apply sepia effect to a float32 image
        >>> image = np.random.rand(100, 100, 3).astype(np.float32)
        >>> transform = A.ToSepia(p=1.0)
        >>> sepia_image = transform(image=image)['image']
        >>> assert sepia_image.shape == image.shape
        >>> assert sepia_image.dtype == np.float32
        >>> assert 0 <= sepia_image.min() <= sepia_image.max() <= 1.0

    Mathematical Formulation:
        Given an input pixel [R, G, B], the sepia tone is calculated as:
        R_sepia = 0.393*R + 0.769*G + 0.189*B
        G_sepia = 0.349*R + 0.686*G + 0.168*B
        B_sepia = 0.272*R + 0.534*G + 0.131*B

        The output values are then clipped to the valid range for the image's data type.

    See Also:
        ToGray: For converting images to grayscale instead of sepia.
    """

    def __init__(self, p: float = 0.5, always_apply: bool | None = None):
        super().__init__(p, always_apply)
        self.sepia_transformation_matrix = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        )

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        non_rgb_error(img)
        return fmain.linear_transformation_rgb(img, self.sepia_transformation_matrix)

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()


class ToFloat(ImageOnlyTransform):
    """Convert the input image to a floating-point representation.

    This transform divides pixel values by `max_value` to get a float32 output array
    where all values lie in the range [0, 1.0]. It's useful for normalizing image data
    before feeding it into neural networks or other algorithms that expect float input.

    Args:
        max_value (float | None): The maximum possible input value. If None, the transform
            will try to infer the maximum value by inspecting the data type of the input image:
            - uint8: 255
            - uint16: 65535
            - uint32: 4294967295
            - float32: 1.0
            Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, uint16, uint32, float32

    Returns:
        np.ndarray: Image in floating point representation, with values in range [0, 1.0].

    Note:
        - If the input image is already float32 with values in [0, 1], it will be returned unchanged.
        - For integer types (uint8, uint16, uint32), the function will scale the values to [0, 1] range.
        - The output will always be float32, regardless of the input type.
        - This transform is often used as a preprocessing step before applying other transformations
          or feeding the image into a neural network.

    Raises:
        TypeError: If the input image data type is not supported.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        # Convert uint8 image to float
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ToFloat(max_value=None)
        >>> float_image = transform(image=image)['image']
        >>> assert float_image.dtype == np.float32
        >>> assert 0 <= float_image.min() <= float_image.max() <= 1.0
        >>>
        # Convert uint16 image to float with custom max_value
        >>> image = np.random.randint(0, 4096, (100, 100, 3), dtype=np.uint16)
        >>> transform = A.ToFloat(max_value=4095)
        >>> float_image = transform(image=image)['image']
        >>> assert float_image.dtype == np.float32
        >>> assert 0 <= float_image.min() <= float_image.max() <= 1.0

    See Also:
        FromFloat: The inverse operation, converting from float back to the original data type.
    """

    class InitSchema(BaseTransformInitSchema):
        max_value: float | None

    def __init__(self, max_value: float | None = None, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p, always_apply)
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return to_float(img, self.max_value)

    def get_transform_init_args_names(self) -> tuple[str]:
        return ("max_value",)


class FromFloat(ImageOnlyTransform):
    """Convert an image from floating point representation to the specified data type.

    This transform is designed to convert images from a normalized floating-point representation
    (typically with values in the range [0, 1]) to other data types, scaling the values appropriately.

    Args:
        dtype (str): The desired output data type. Supported types include 'uint8', 'uint16',
                     'uint32'. Default: 'uint8'.
        max_value (float | None): The maximum value for the output dtype. If None, the transform
                                  will attempt to infer the maximum value based on the dtype.
                                  Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        float32, float64

    Note:
        - This is the inverse transform for ToFloat.
        - Input images are expected to be in floating point format with values in the range [0, 1].
        - For integer output types (uint8, uint16, uint32), the function will scale the values
          to the appropriate range (e.g., 0-255 for uint8).
        - For float output types (float32, float64), the values will remain in the [0, 1] range.
        - The transform uses the `from_float` function internally, which ensures output values
          are within the valid range for the specified dtype.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> transform = A.FromFloat(dtype='uint8', max_value=None, p=1.0)
        >>> image = np.random.rand(100, 100, 3).astype(np.float32)  # Float image in [0, 1] range
        >>> result = transform(image=image)
        >>> uint8_image = result['image']
        >>> assert uint8_image.dtype == np.uint8
        >>> assert uint8_image.min() >= 0 and uint8_image.max() <= 255

    """

    class InitSchema(BaseTransformInitSchema):
        dtype: Literal["uint8", "uint16", "float32", "float64"]
        max_value: float | None

    def __init__(
        self,
        dtype: Literal["uint8", "uint16", "float32", "float64"] = "uint8",
        max_value: float | None = None,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.dtype = np.dtype(dtype)
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return from_float(img, self.dtype, self.max_value)

    def get_transform_init_args(self) -> dict[str, Any]:
        return {"dtype": self.dtype.name, "max_value": self.max_value}


class InterpolationDict(TypedDict):
    upscale: int
    downscale: int


class InterpolationPydantic(BaseModel):
    upscale: InterpolationType
    downscale: InterpolationType


class Downscale(ImageOnlyTransform):
    """Decrease image quality by downscaling and upscaling back.

    This transform simulates the effect of a low-resolution image by first downscaling
    the image to a lower resolution and then upscaling it back to its original size.
    This process introduces loss of detail and can be used to simulate low-quality
    images or to test the robustness of models to different image resolutions.

    Args:
        scale_range (tuple[float, float]): Range for the downscaling factor.
            Should be two float values between 0 and 1, where the first value is less than or equal to the second.
            The actual downscaling factor will be randomly chosen from this range for each image.
            Lower values result in more aggressive downscaling.
            Default: (0.25, 0.25)

        interpolation_pair (InterpolationDict): A dictionary specifying the interpolation methods to use for
            downscaling and upscaling. Should contain two keys:
            - 'downscale': Interpolation method for downscaling
            - 'upscale': Interpolation method for upscaling
            Values should be OpenCV interpolation flags (e.g., cv2.INTER_NEAREST, cv2.INTER_LINEAR, etc.)
            Default: {'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_NEAREST}

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - The actual downscaling factor is randomly chosen for each image from the range
          specified in scale_range.
        - Using different interpolation methods for downscaling and upscaling can produce
          various effects. For example, using INTER_NEAREST for both can create a pixelated look,
          while using INTER_LINEAR or INTER_CUBIC can produce smoother results.
        - This transform can be useful for data augmentation, especially when training models
          that need to be robust to variations in image quality or resolution.

    Example:
        >>> import albumentations as A
        >>> import cv2
        >>> transform = A.Downscale(
        ...     scale_range=(0.5, 0.75),
        ...     interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR},
        ...     p=0.5
        ... )
        >>> transformed = transform(image=image)
        >>> downscaled_image = transformed['image']
    """

    class InitSchema(BaseTransformInitSchema):
        scale_min: float | None
        scale_max: float | None

        interpolation: int | Interpolation | InterpolationDict | None = Field(
            default_factory=lambda: Interpolation(downscale=cv2.INTER_NEAREST, upscale=cv2.INTER_NEAREST),
        )
        interpolation_pair: InterpolationPydantic

        scale_range: Annotated[tuple[float, float], AfterValidator(check_01), AfterValidator(nondecreasing)]

        @model_validator(mode="after")
        def validate_params(self) -> Self:
            if self.scale_min is not None and self.scale_max is not None:
                warn(
                    "scale_min and scale_max are deprecated. Use scale_range instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

                self.scale_range = (self.scale_min, self.scale_max)
                self.scale_min = None
                self.scale_max = None

            if self.interpolation is not None:
                warn(
                    "Downscale.interpolation is deprecated. Use Downscale.interpolation_pair instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

                if isinstance(self.interpolation, dict):
                    self.interpolation_pair = InterpolationPydantic(**self.interpolation)
                elif isinstance(self.interpolation, int):
                    self.interpolation_pair = InterpolationPydantic(
                        upscale=self.interpolation,
                        downscale=self.interpolation,
                    )
                elif isinstance(self.interpolation, Interpolation):
                    self.interpolation_pair = InterpolationPydantic(
                        upscale=self.interpolation.upscale,
                        downscale=self.interpolation.downscale,
                    )
                self.interpolation = None

            return self

    def __init__(
        self,
        scale_min: float | None = None,
        scale_max: float | None = None,
        interpolation: int | Interpolation | InterpolationDict | None = None,
        scale_range: tuple[float, float] = (0.25, 0.25),
        interpolation_pair: InterpolationDict = InterpolationDict(
            {"upscale": cv2.INTER_NEAREST, "downscale": cv2.INTER_NEAREST},
        ),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.scale_range = scale_range
        self.interpolation_pair = interpolation_pair

    def apply(self, img: np.ndarray, scale: float, **params: Any) -> np.ndarray:
        return fmain.downscale(
            img,
            scale=scale,
            down_interpolation=self.interpolation_pair["downscale"],
            up_interpolation=self.interpolation_pair["upscale"],
        )

    def get_params(self) -> dict[str, Any]:
        return {"scale": self.py_random.uniform(*self.scale_range)}

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "scale_range", "interpolation_pair"


class Lambda(NoOp):
    """A flexible transformation class for using user-defined transformation functions per targets.
    Function signature must include **kwargs to accept optional arguments like interpolation method, image size, etc:

    Args:
        image: Image transformation function.
        mask: Mask transformation function.
        keypoint: Keypoint transformation function.
        bbox: BBox transformation function.
        p: probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Number of channels:
        Any

    """

    def __init__(
        self,
        image: Callable[..., Any] | None = None,
        mask: Callable[..., Any] | None = None,
        keypoints: Callable[..., Any] | None = None,
        bboxes: Callable[..., Any] | None = None,
        name: str | None = None,
        always_apply: bool | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.name = name
        self.custom_apply_fns = {
            target_name: fmain.noop for target_name in ("image", "mask", "keypoints", "bboxes", "global_label")
        }
        for target_name, custom_apply_fn in {
            "image": image,
            "mask": mask,
            "keypoints": keypoints,
            "bboxes": bboxes,
        }.items():
            if custom_apply_fn is not None:
                if isinstance(custom_apply_fn, LambdaType) and custom_apply_fn.__name__ == "<lambda>":
                    warnings.warn(
                        "Using lambda is incompatible with multiprocessing. "
                        "Consider using regular functions or partial().",
                        stacklevel=2,
                    )

                self.custom_apply_fns[target_name] = custom_apply_fn

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        fn = self.custom_apply_fns["image"]
        return fn(img, **params)

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        fn = self.custom_apply_fns["mask"]
        return fn(mask, **params)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        is_ndarray = True

        if not isinstance(bboxes, np.ndarray):
            is_ndarray = False
            bboxes = np.array(bboxes, dtype=np.float32)

        fn = self.custom_apply_fns["bboxes"]
        result = fn(bboxes, **params)

        if not is_ndarray:
            return result.tolist()

        return result

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        is_ndarray = True
        if not isinstance(keypoints, np.ndarray):
            is_ndarray = False
            keypoints = np.array(keypoints, dtype=np.float32)

        fn = self.custom_apply_fns["keypoints"]
        result = fn(keypoints, **params)

        if not is_ndarray:
            return result.tolist()

        return result

    @classmethod
    def is_serializable(cls) -> bool:
        return False

    def to_dict_private(self) -> dict[str, Any]:
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
    """Apply multiplicative noise to the input image.

    This transform multiplies each pixel in the image by a random value or array of values,
    effectively creating a noise pattern that scales with the image intensity.

    Args:
        multiplier (tuple[float, float]): The range for the random multiplier.
            Defines the range from which the multiplier is sampled.
            Default: (0.9, 1.1)

        per_channel (bool): If True, use a different random multiplier for each channel.
            If False, use the same multiplier for all channels.
            Setting this to False is slightly faster.
            Default: False

        elementwise (bool): If True, generates a unique multiplier for each pixel.
            If False, generates a single multiplier (or one per channel if per_channel=True).
            Default: False

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - When elementwise=False and per_channel=False, a single multiplier is applied to the entire image.
        - When elementwise=False and per_channel=True, each channel gets a different multiplier.
        - When elementwise=True and per_channel=False, each pixel gets the same multiplier across all channels.
        - When elementwise=True and per_channel=True, each pixel in each channel gets a unique multiplier.
        - Setting per_channel=False is slightly faster, especially for larger images.
        - This transform can be used to simulate various lighting conditions or to create noise that
          scales with image intensity.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1.0)
        >>> result = transform(image=image)
        >>> noisy_image = result["image"]

    References:
        - Multiplicative noise: https://en.wikipedia.org/wiki/Multiplicative_noise
    """

    class InitSchema(BaseTransformInitSchema):
        multiplier: Annotated[tuple[float, float], AfterValidator(check_0plus), AfterValidator(nondecreasing)]
        per_channel: bool
        elementwise: bool

    def __init__(
        self,
        multiplier: ScaleFloatType = (0.9, 1.1),
        per_channel: bool = False,
        elementwise: bool = False,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.multiplier = cast(tuple[float, float], multiplier)
        self.elementwise = elementwise
        self.per_channel = per_channel

    def apply(
        self,
        img: np.ndarray,
        multiplier: float | np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        return multiply(img, multiplier)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"] if "image" in data else data["images"][0]

        num_channels = get_num_channels(image)

        if self.elementwise:
            shape = image.shape if self.per_channel else (*image.shape[:2], 1)
        else:
            shape = (num_channels,) if self.per_channel else (1,)

        multiplier = self.random_generator.uniform(self.multiplier[0], self.multiplier[1], shape).astype(np.float32)

        if not self.per_channel and num_channels > 1:
            # Replicate the multiplier for all channels if not per_channel
            multiplier = np.repeat(multiplier, num_channels, axis=-1)

        if not self.elementwise and self.per_channel:
            # Reshape to broadcast correctly when not elementwise but per_channel
            multiplier = multiplier.reshape(1, 1, -1)

        if multiplier.shape != image.shape:
            multiplier = multiplier.squeeze()

        return {"multiplier": multiplier}

    def get_transform_init_args_names(self) -> tuple[str, str, str]:
        return "multiplier", "elementwise", "per_channel"


class FancyPCA(ImageOnlyTransform):
    """Apply Fancy PCA augmentation to the input image.

    This augmentation technique applies PCA (Principal Component Analysis) to the image's color channels,
    then adds multiples of the principal components to the image, with magnitudes proportional to the
    corresponding eigenvalues times a random variable drawn from a Gaussian with mean 0 and standard
    deviation 'alpha'.

    Args:
        alpha (tuple[float, float] | float): Standard deviation of the Gaussian distribution used to generate
            random noise for each principal component. If a single float is provided, it will be used for
            all channels. If a tuple of two floats (min, max) is provided, the standard deviation will be
            uniformly sampled from this range for each run. Default: 0.1.
        always_apply (bool): If True, the transform will always be applied. Default: False.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        any

    Note:
        - This augmentation is particularly effective for RGB images but can work with any number of channels.
        - For grayscale images, it applies a simplified version of the augmentation.
        - The transform preserves the mean of the image while adjusting the color/intensity variation.
        - This implementation is based on the paper by Krizhevsky et al. and is similar to the one used
          in the original AlexNet paper.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.FancyPCA(alpha=0.1, p=1.0)
        >>> result = transform(image=image)
        >>> augmented_image = result["image"]

    References:
        - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep
          convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
        - https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    """

    class InitSchema(BaseTransformInitSchema):
        alpha: float = Field(ge=0)

    def __init__(self, alpha: float = 0.1, p: float = 0.5, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        self.alpha = alpha

    def apply(self, img: np.ndarray, alpha_vector: np.ndarray, **params: Any) -> np.ndarray:
        return fmain.fancy_pca(img, alpha_vector)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        shape = params["shape"]
        num_channels = shape[-1] if len(shape) == NUM_MULTI_CHANNEL_DIMENSIONS else 1
        alpha_vector = self.random_generator.normal(0, self.alpha, num_channels).astype(np.float32)
        return {"alpha_vector": alpha_vector}

    def get_transform_init_args_names(self) -> tuple[str]:
        return ("alpha",)


class ColorJitter(ImageOnlyTransform):
    """Randomly changes the brightness, contrast, saturation, and hue of an image.

    This transform is similar to torchvision's ColorJitter but with some differences due to the use of OpenCV
    instead of Pillow. The main differences are:
    1. OpenCV and Pillow use different formulas to convert images to HSV format.
    2. This implementation uses value saturation instead of uint8 overflow as in Pillow.

    These differences may result in slightly different output compared to torchvision's ColorJitter.

    Args:
        brightness (tuple[float, float] | float): How much to jitter brightness.
            If float:
                The brightness factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
            If tuple:
                The brightness factor is sampled from the range specified.
            Should be non-negative numbers.
            Default: (0.8, 1.2)

        contrast (tuple[float, float] | float): How much to jitter contrast.
            If float:
                The contrast factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
            If tuple:
                The contrast factor is sampled from the range specified.
            Should be non-negative numbers.
            Default: (0.8, 1.2)

        saturation (tuple[float, float] | float): How much to jitter saturation.
            If float:
                The saturation factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
            If tuple:
                The saturation factor is sampled from the range specified.
            Should be non-negative numbers.
            Default: (0.8, 1.2)

        hue (float or tuple of float (min, max)): How much to jitter hue.
            If float:
                The hue factor is chosen uniformly from [-hue, hue]. Should have 0 <= hue <= 0.5.
            If tuple:
                The hue factor is sampled from the range specified. Values should be in range [-0.5, 0.5].
            Default: (-0.5, 0.5)

         p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5


    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        1, 3

    Note:
        - The order of application for these color transformations is random for each image.
        - The ranges for brightness, contrast, and saturation are applied as multiplicative factors.
        - The range for hue is applied as an additive factor.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        >>> result = transform(image=image)
        >>> jittered_image = result['image']

    References:
        - https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ColorJitter
        - https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    """

    class InitSchema(BaseTransformInitSchema):
        brightness: ScaleFloatType
        contrast: ScaleFloatType
        saturation: ScaleFloatType
        hue: ScaleFloatType

        @field_validator("brightness", "contrast", "saturation", "hue")
        @classmethod
        def check_ranges(cls, value: ScaleFloatType, info: ValidationInfo) -> tuple[float, float]:
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
                left = bias - value
                if clip:
                    left = max(left, 0)
                value = (left, bias + value)
            elif isinstance(value, tuple) and len(value) == PAIR:
                check_range(value, *bounds, info.field_name)

            return cast(tuple[float, float], value)

    def __init__(
        self,
        brightness: ScaleFloatType = (0.8, 1.2),
        contrast: ScaleFloatType = (0.8, 1.2),
        saturation: ScaleFloatType = (0.8, 1.2),
        hue: ScaleFloatType = (-0.5, 0.5),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.brightness = cast(tuple[float, float], brightness)
        self.contrast = cast(tuple[float, float], contrast)
        self.saturation = cast(tuple[float, float], saturation)
        self.hue = cast(tuple[float, float], hue)

        self.transforms = [
            fmain.adjust_brightness_torchvision,
            fmain.adjust_contrast_torchvision,
            fmain.adjust_saturation_torchvision,
            fmain.adjust_hue_torchvision,
        ]

    def get_params(self) -> dict[str, Any]:
        brightness = self.py_random.uniform(*self.brightness)
        contrast = self.py_random.uniform(*self.contrast)
        saturation = self.py_random.uniform(*self.saturation)
        hue = self.py_random.uniform(*self.hue)

        order = [0, 1, 2, 3]
        self.random_generator.shuffle(order)

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
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        order: list[int],
        **params: Any,
    ) -> np.ndarray:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "ColorJitter transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)
        color_transforms = [brightness, contrast, saturation, hue]
        for i in order:
            img = self.transforms[i](img, color_transforms[i])
        return img

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "brightness", "contrast", "saturation", "hue"


class Sharpen(ImageOnlyTransform):
    """Sharpen the input image and overlays the result with the original image.

    This transform applies a sharpening filter to the input image and then blends
    the sharpened image with the original using a specified alpha value.

    Args:
        alpha (tuple[float, float]): Range to choose the visibility of the sharpened image.
            At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
            Values should be in the range [0, 1].
            Default: (0.2, 0.5).

        lightness (tuple of float): Range to choose the lightness of the sharpened image.
            Larger values will create images with higher contrast.
            Values should be greater than 0.
            Default: (0.5, 1.0).

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The sharpening effect is achieved using a 3x3 sharpening kernel.
        - The kernel is dynamically generated based on the 'alpha' and 'lightness' parameters.
        - Higher 'alpha' values will result in a more pronounced sharpening effect.
        - Higher 'lightness' values will increase the contrast of the sharpened areas.
        - This transform can be useful for:
          * Enhancing edge details in images
          * Improving the perceived quality of slightly blurred images
          * Creating a more crisp appearance in photographs

    Mathematical Formulation:
        The sharpening kernel K is defined as:

        K = (1 - alpha) * I + alpha * L

        where:
        - alpha is the alpha value (from the 'alpha' parameter)
        - I is the identity kernel [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        - L is the Laplacian kernel [[-1, -1, -1], [-1, 8+l, -1], [-1, -1, -1]]
          (l is the lightness value from the 'lightness' parameter)

        The sharpened image S is obtained by convolving the input image I with the kernel K:

        S = I * K

        The final output O is a blend of the original and sharpened images:

        O = (1 - alpha) * I + alpha * S

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Apply sharpening with default parameters
        >>> transform = A.Sharpen(p=1.0)
        >>> sharpened_image = transform(image=image)['image']

        # Apply sharpening with custom parameters
        >>> transform = A.Sharpen(alpha=(0.4, 0.7), lightness=(0.8, 1.2), p=1.0)
        >>> sharpened_image = transform(image=image)['image']

    References:
        - Image sharpening: https://en.wikipedia.org/wiki/Unsharp_masking
        - Laplacian operator: https://en.wikipedia.org/wiki/Laplace_operator
        - "Digital Image Processing" by Rafael C. Gonzalez and Richard E. Woods, 4th Edition
    """

    class InitSchema(BaseTransformInitSchema):
        alpha: Annotated[tuple[float, float], AfterValidator(check_01)]
        lightness: Annotated[tuple[float, float], AfterValidator(check_0plus)]

    def __init__(
        self,
        alpha: tuple[float, float] = (0.2, 0.5),
        lightness: tuple[float, float] = (0.5, 1.0),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
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

    def get_params(self) -> dict[str, np.ndarray]:
        alpha = self.py_random.uniform(*self.alpha)
        lightness = self.py_random.uniform(*self.lightness)
        sharpening_matrix = self.__generate_sharpening_matrix(alpha_sample=alpha, lightness_sample=lightness)
        return {"sharpening_matrix": sharpening_matrix}

    def apply(self, img: np.ndarray, sharpening_matrix: np.ndarray, **params: Any) -> np.ndarray:
        return fmain.convolve(img, sharpening_matrix)

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return ("alpha", "lightness")


class Emboss(ImageOnlyTransform):
    """Apply embossing effect to the input image.

    This transform creates an emboss effect by highlighting edges and creating a 3D-like texture
    in the image. It works by applying a specific convolution kernel to the image that emphasizes
    differences in adjacent pixel values.

    Args:
        alpha (tuple[float, float]): Range to choose the visibility of the embossed image.
            At 0, only the original image is visible, at 1.0 only its embossed version is visible.
            Values should be in the range [0, 1].
            Alpha will be randomly selected from this range for each image.
            Default: (0.2, 0.5)

        strength (tuple[float, float]): Range to choose the strength of the embossing effect.
            Higher values create a more pronounced 3D effect.
            Values should be non-negative.
            Strength will be randomly selected from this range for each image.
            Default: (0.2, 0.7)

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - The emboss effect is created using a 3x3 convolution kernel.
        - The 'alpha' parameter controls the blend between the original image and the embossed version.
          A higher alpha value will result in a more pronounced emboss effect.
        - The 'strength' parameter affects the intensity of the embossing. Higher strength values
          will create more contrast in the embossed areas, resulting in a stronger 3D-like effect.
        - This transform can be useful for creating artistic effects or for data augmentation
          in tasks where edge information is important.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5)
        >>> result = transform(image=image)
        >>> embossed_image = result['image']

    References:
        - https://en.wikipedia.org/wiki/Image_embossing
        - https://www.researchgate.net/publication/303412455_Application_of_Emboss_Filtering_in_Image_Processing
    """

    class InitSchema(BaseTransformInitSchema):
        alpha: Annotated[tuple[float, float], AfterValidator(check_01)]
        strength: Annotated[tuple[float, float], AfterValidator(check_0plus)]

    def __init__(
        self,
        alpha: tuple[float, float] = (0.2, 0.5),
        strength: tuple[float, float] = (0.2, 0.7),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
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

    def get_params(self) -> dict[str, np.ndarray]:
        alpha = self.py_random.uniform(*self.alpha)
        strength = self.py_random.uniform(*self.strength)
        emboss_matrix = self.__generate_emboss_matrix(alpha_sample=alpha, strength_sample=strength)
        return {"emboss_matrix": emboss_matrix}

    def apply(self, img: np.ndarray, emboss_matrix: np.ndarray, **params: Any) -> np.ndarray:
        return fmain.convolve(img, emboss_matrix)

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return ("alpha", "strength")


class Superpixels(ImageOnlyTransform):
    """Transform images partially/completely to their superpixel representation.

    Args:
        p_replace (tuple[float, float] | float): Defines for any segment the probability that the pixels within that
            segment are replaced by their average color (otherwise, the pixels are not changed).


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
            Default: (0.1, 0.3)

        n_segments (tuple[int, int] | int): Rough target number of how many superpixels to generate.
            The algorithm may deviate from this number.
            Lower value will lead to coarser superpixels.
            Higher values are computationally more intensive and will hence lead to a slowdown.
            If tuple ``(a, b)``, then a value from the discrete interval ``[a..b]`` will be sampled per image.
            Default: (15, 120)

        max_size (int | None): Maximum image size at which the augmentation is performed.
            If the width or height of an image exceeds this value, it will be
            downscaled before the augmentation so that the longest side matches `max_size`.
            This is done to speed up the process. The final output image has the same size as the input image.
            Note that in case `p_replace` is below ``1.0``,
            the down-/upscaling will affect the not-replaced pixels too.
            Use ``None`` to apply no down-/upscaling.
            Default: 128

        interpolation (OpenCV flag): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This transform can significantly change the visual appearance of the image.
        - The transform makes use of a superpixel algorithm, which tends to be slow.
        If performance is a concern, consider using `max_size` to limit the image size.
        - The effect of this transform can vary greatly depending on the `p_replace` and `n_segments` parameters.
        - When `p_replace` is high, the image can become highly abstracted, resembling a voronoi diagram.
        - The transform preserves the original image type (uint8 or float32).

    Mathematical Formulation:
        1. The image is segmented into approximately `n_segments` superpixels using the SLIC algorithm.
        2. For each superpixel:
        - With probability `p_replace`, all pixels in the superpixel are replaced with their mean color.
        - With probability `1 - p_replace`, the superpixel is left unchanged.
        3. If the image was resized due to `max_size`, it is resized back to its original dimensions.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Apply superpixels with default parameters
        >>> transform = A.Superpixels(p=1.0)
        >>> augmented_image = transform(image=image)['image']

        # Apply superpixels with custom parameters
        >>> transform = A.Superpixels(
        ...     p_replace=(0.5, 0.7),
        ...     n_segments=(50, 100),
        ...     max_size=None,
        ...     interpolation=cv2.INTER_NEAREST,
        ...     p=1.0
        ... )
        >>> augmented_image = transform(image=image)['image']
    """

    class InitSchema(BaseTransformInitSchema):
        p_replace: ZeroOneRangeType
        n_segments: OnePlusIntRangeType
        max_size: int | None = Field(ge=1)
        interpolation: InterpolationType

    def __init__(
        self,
        p_replace: ScaleFloatType = (0, 0.1),
        n_segments: ScaleIntType = (100, 100),
        max_size: int | None = 128,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.p_replace = cast(tuple[float, float], p_replace)
        self.n_segments = cast(tuple[int, int], n_segments)
        self.max_size = max_size
        self.interpolation = interpolation

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "p_replace", "n_segments", "max_size", "interpolation"

    def get_params(self) -> dict[str, Any]:
        n_segments = self.py_random.randint(*self.n_segments)
        p = self.py_random.uniform(*self.p_replace)
        return {"replace_samples": self.random_generator.random(n_segments) < p, "n_segments": n_segments}

    def apply(
        self,
        img: np.ndarray,
        replace_samples: Sequence[bool],
        n_segments: int,
        **kwargs: Any,
    ) -> np.ndarray:
        return fmain.superpixels(img, n_segments, replace_samples, self.max_size, self.interpolation)


class RingingOvershoot(ImageOnlyTransform):
    """Create ringing or overshoot artifacts by convolving the image with a 2D sinc filter.

    This transform simulates the ringing artifacts that can occur in digital image processing,
    particularly after sharpening or edge enhancement operations. It creates oscillations
    or overshoots near sharp transitions in the image.

    Args:
        blur_limit (tuple[int, int] | int): Maximum kernel size for the sinc filter.
            Must be an odd number in the range [3, inf).
            If a single int is provided, the kernel size will be randomly chosen
            from the range (3, blur_limit). If a tuple (min, max) is provided,
            the kernel size will be randomly chosen from the range (min, max).
            Default: (7, 15).
        cutoff (tuple[float, float]): Range to choose the cutoff frequency in radians.
            Values should be in the range (0, π). A lower cutoff frequency will
            result in more pronounced ringing effects.
            Default: (π/4, π/2).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - Ringing artifacts are oscillations of the image intensity function in the neighborhood
          of sharp transitions, such as edges or object boundaries.
        - This transform uses a 2D sinc filter (also known as a 2D cardinal sine function)
          to introduce these artifacts.
        - The severity of the ringing effect is controlled by both the kernel size (blur_limit)
          and the cutoff frequency.
        - Larger kernel sizes and lower cutoff frequencies will generally produce more
          noticeable ringing effects.
        - This transform can be useful for:
          * Simulating imperfections in image processing or transmission systems
          * Testing the robustness of computer vision models to ringing artifacts
          * Creating artistic effects that emphasize edges and transitions in images

    Mathematical Formulation:
        The 2D sinc filter kernel is defined as:

        K(x, y) = cutoff * J₁(cutoff * √(x² + y²)) / (2π * √(x² + y²))

        where:
        - J₁ is the Bessel function of the first kind of order 1
        - cutoff is the chosen cutoff frequency
        - x and y are the distances from the kernel center

        The filtered image I' is obtained by convolving the input image I with the kernel K:

        I'(x, y) = ∑∑ I(x-u, y-v) * K(u, v)

        The convolution operation introduces the ringing artifacts near sharp transitions.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)

        # Apply ringing effect with default parameters
        >>> transform = A.RingingOvershoot(p=1.0)
        >>> ringing_image = transform(image=image)['image']

        # Apply ringing effect with custom parameters
        >>> transform = A.RingingOvershoot(
        ...     blur_limit=(9, 17),
        ...     cutoff=(np.pi/6, np.pi/3),
        ...     p=1.0
        ... )
        >>> ringing_image = transform(image=image)['image']

    References:
        - Ringing artifacts: https://en.wikipedia.org/wiki/Ringing_artifacts
        - Sinc filter: https://en.wikipedia.org/wiki/Sinc_filter
        - "The Importance of Ringing Artifacts in Image Processing" by Jae S. Lim, 1981
        - "Digital Image Processing" by Rafael C. Gonzalez and Richard E. Woods, 4th Edition
    """

    class InitSchema(BlurInitSchema):
        blur_limit: ScaleIntType
        cutoff: Annotated[tuple[float, float], nondecreasing]

        @field_validator("cutoff")
        @classmethod
        def check_cutoff(cls, v: tuple[float, float], info: ValidationInfo) -> tuple[float, float]:
            bounds = 0, np.pi
            check_range(v, *bounds, info.field_name)
            return v

    def __init__(
        self,
        blur_limit: ScaleIntType = (7, 15),
        cutoff: tuple[float, float] = (np.pi / 4, np.pi / 2),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.blur_limit = cast(tuple[int, int], blur_limit)
        self.cutoff = cutoff

    def get_params(self) -> dict[str, np.ndarray]:
        ksize = self.py_random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        if ksize % 2 == 0:
            raise ValueError(f"Kernel size must be odd. Got: {ksize}")

        cutoff = self.py_random.uniform(*self.cutoff)

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

    def apply(self, img: np.ndarray, kernel: int, **params: Any) -> np.ndarray:
        return fmain.convolve(img, kernel)

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return ("blur_limit", "cutoff")


class UnsharpMask(ImageOnlyTransform):
    """Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.

    Unsharp masking is a technique that enhances edge contrast in an image, creating the illusion of increased
        sharpness.
    This transform applies Gaussian blur to create a blurred version of the image, then uses this to create a mask
    which is combined with the original image to enhance edges and fine details.

    Args:
        blur_limit (tuple[int, int] | int): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (tuple[float, float] | float): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        alpha (tuple[float, float]): range to choose the visibility of the sharpened image.
            At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
            Default: (0.2, 0.5).
        threshold (int): Value to limit sharpening only for areas with high pixel difference between original image
            and it's smoothed version. Higher threshold means less sharpening on flat areas.
            Must be in range [0, 255]. Default: 10.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - The algorithm creates a mask M = (I - G) * alpha, where I is the original image and G is the Gaussian
            blurred version.
        - The final image is computed as: output = I + M if |I - G| > threshold, else I.
        - Higher alpha values increase the strength of the sharpening effect.
        - Higher threshold values limit the sharpening effect to areas with more significant edges or details.
        - The blur_limit and sigma_limit parameters control the Gaussian blur used to create the mask.

    References:
        - https://en.wikipedia.org/wiki/Unsharp_masking
        - https://arxiv.org/pdf/2107.10833.pdf

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>>
        # Apply UnsharpMask with default parameters
        >>> transform = A.UnsharpMask(p=1.0)
        >>> sharpened_image = transform(image=image)['image']
        >>>
        # Apply UnsharpMask with custom parameters
        >>> transform = A.UnsharpMask(
        ...     blur_limit=(3, 7),
        ...     sigma_limit=(0.1, 0.5),
        ...     alpha=(0.2, 0.7),
        ...     threshold=15,
        ...     p=1.0
        ... )
        >>> sharpened_image = transform(image=image)['image']
    """

    class InitSchema(BaseTransformInitSchema):
        sigma_limit: NonNegativeFloatRangeType
        alpha: ZeroOneRangeType
        threshold: int = Field(ge=0, le=255)
        blur_limit: ScaleIntType

        @field_validator("blur_limit")
        @classmethod
        def process_blur(cls, value: ScaleIntType, info: ValidationInfo) -> tuple[int, int]:
            return fblur.process_blur_limit(value, info, min_value=3)

    def __init__(
        self,
        blur_limit: ScaleIntType = (3, 7),
        sigma_limit: ScaleFloatType = 0.0,
        alpha: ScaleFloatType = (0.2, 0.5),
        threshold: int = 10,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.blur_limit = cast(tuple[int, int], blur_limit)
        self.sigma_limit = cast(tuple[float, float], sigma_limit)
        self.alpha = cast(tuple[float, float], alpha)
        self.threshold = threshold

    def get_params(self) -> dict[str, Any]:
        return {
            "ksize": self.py_random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2),
            "sigma": self.py_random.uniform(*self.sigma_limit),
            "alpha": self.py_random.uniform(*self.alpha),
        }

    def apply(self, img: np.ndarray, ksize: int, sigma: int, alpha: float, **params: Any) -> np.ndarray:
        return fmain.unsharp_mask(img, ksize, sigma=sigma, alpha=alpha, threshold=self.threshold)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "blur_limit", "sigma_limit", "alpha", "threshold"


class PixelDropout(DualTransform):
    """Drops random pixels from the image.

    This transform randomly sets pixels in the image to a specified value, effectively "dropping out" those pixels.
    It can be applied to both the image and its corresponding mask.

    Args:
        dropout_prob (float): Probability of dropping out each pixel. Should be in the range [0, 1].
            Default: 0.01

        per_channel (bool): If True, the dropout mask will be generated independently for each channel.
            If False, the same dropout mask will be applied to all channels.
            Default: False

        drop_value (float | Sequence[float] | None): Value to assign to the dropped pixels.
            If None, the value will be randomly sampled for each application:
                - For uint8 images: Random integer in [0, 255]
                - For float32 images: Random float in [0, 1]
            If a single number, that value will be used for all dropped pixels.
            If a sequence, it should contain one value per channel.
            Default: 0

        mask_drop_value (float | Sequence[float] | None): Value to assign to dropped pixels in the mask.
            If None, the mask will remain unchanged.
            If a single number, that value will be used for all dropped pixels in the mask.
            If a sequence, it should contain one value per channel of the mask.
            Note: Only applicable when per_channel=False.
            Default: None

        always_apply (bool): If True, the transform will always be applied.
            Default: False

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        - When applied to bounding boxes, this transform may cause some boxes to have zero area
          if all pixels within the box are dropped. Such boxes will be removed.
        - When applied to keypoints, keypoints that fall on dropped pixels will be removed if
          the keypoint processor is configured to remove invisible keypoints.
        - The 'per_channel' option is not supported for mask dropout. If you need to drop pixels
          in a multi-channel mask independently, consider applying this transform multiple times
          with per_channel=False.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
        >>> transform = A.PixelDropout(dropout_prob=0.1, per_channel=True, p=1.0)
        >>> result = transform(image=image, mask=mask)
        >>> dropped_image, dropped_mask = result['image'], result['mask']
    """

    class InitSchema(BaseTransformInitSchema):
        dropout_prob: ProbabilityType
        per_channel: bool
        drop_value: ScaleFloatType | None
        mask_drop_value: ScaleFloatType | None

        @model_validator(mode="after")
        def validate_mask_drop_value(self) -> Self:
            if self.mask_drop_value is not None and self.per_channel:
                msg = "PixelDropout supports mask only with per_channel=False."
                raise ValueError(msg)
            return self

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    def __init__(
        self,
        dropout_prob: float = 0.01,
        per_channel: bool = False,
        drop_value: ScaleFloatType | None = 0,
        mask_drop_value: ScaleFloatType | None = None,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.drop_value = drop_value
        self.mask_drop_value = mask_drop_value

    def apply(
        self,
        img: np.ndarray,
        drop_mask: np.ndarray,
        drop_value: float | Sequence[float],
        **params: Any,
    ) -> np.ndarray:
        return fmain.pixel_dropout(img, drop_mask, drop_value)

    def apply_to_mask(self, mask: np.ndarray, drop_mask: np.ndarray, **params: Any) -> np.ndarray:
        if self.mask_drop_value is None:
            return mask

        if mask.ndim == MONO_CHANNEL_DIMENSIONS:
            drop_mask = np.squeeze(drop_mask)

        return fmain.pixel_dropout(mask, drop_mask, self.mask_drop_value)

    def apply_to_bboxes(self, bboxes: np.ndarray, drop_mask: np.ndarray | None, **params: Any) -> np.ndarray:
        if drop_mask is None or self.per_channel:
            return bboxes

        processor = cast(BboxProcessor, self.get_processor("bboxes"))
        if processor is None:
            return bboxes

        image_shape = params["shape"][:2]

        denormalized_bboxes = denormalize_bboxes(bboxes, image_shape)

        result = fdropout.mask_dropout_bboxes(
            denormalized_bboxes,
            drop_mask,
            image_shape,
            processor.params.min_area,
            processor.params.min_visibility,
        )

        return normalize_bboxes(result, image_shape)

    def apply_to_keypoints(self, keypoints: np.ndarray, drop_mask: np.ndarray | None, **params: Any) -> np.ndarray:
        if drop_mask is None or self.per_channel:
            return keypoints

        processor = cast(KeypointsProcessor, self.get_processor("keypoints"))

        if processor is None or not processor.params.remove_invisible:
            return keypoints

        return fdropout.mask_dropout_keypoints(keypoints, drop_mask)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"] if "image" in data else data["images"][0]

        shape = image.shape if self.per_channel else image.shape[:2]

        # Use choice to create boolean matrix, if we will use binomial after that we will need type conversion
        drop_mask = self.random_generator.choice([True, False], shape, p=[self.dropout_prob, 1 - self.dropout_prob])

        drop_value: float | Sequence[float] | np.ndarray

        if drop_mask.ndim != image.ndim:
            drop_mask = np.expand_dims(drop_mask, -1)
        if self.drop_value is None:
            drop_shape = 1 if is_grayscale_image(image) else int(image.shape[-1])

            if image.dtype == np.uint8:
                drop_value = self.random_generator.integers(
                    0,
                    int(MAX_VALUES_BY_DTYPE[image.dtype]),
                    size=drop_shape,
                    dtype=image.dtype,
                )
            elif image.dtype == np.float32:
                drop_value = self.random_generator.uniform(0, 1, size=drop_shape).astype(image.dtype)
            else:
                raise ValueError(f"Unsupported dtype: {image.dtype}")
        else:
            drop_value = self.drop_value

        return {"drop_mask": drop_mask, "drop_value": drop_value}

    def get_transform_init_args_names(self) -> tuple[str, str, str, str]:
        return ("dropout_prob", "per_channel", "drop_value", "mask_drop_value")


class Spatter(ImageOnlyTransform):
    """Apply spatter transform. It simulates corruption which can occlude a lens in the form of rain or mud.

    Args:
        mean (tuple[float, float] | float): Mean value of normal distribution for generating liquid layer.
            If single float mean will be sampled from `(0, mean)`
            If tuple of float mean will be sampled from range `(mean[0], mean[1])`.
            If you want constant value use (mean, mean).
            Default (0.65, 0.65)
        std (tuple[float, float] | float): Standard deviation value of normal distribution for generating liquid layer.
            If single float the number will be sampled from `(0, std)`.
            If tuple of float std will be sampled from range `(std[0], std[1])`.
            If you want constant value use (std, std).
            Default: (0.3, 0.3).
        gauss_sigma (tuple[float, float] | floats): Sigma value for gaussian filtering of liquid layer.
            If single float the number will be sampled from `(0, gauss_sigma)`.
            If tuple of float gauss_sigma will be sampled from range `(gauss_sigma[0], gauss_sigma[1])`.
            If you want constant value use (gauss_sigma, gauss_sigma).
            Default: (2, 3).
        cutout_threshold (tuple[float, float] | floats): Threshold for filtering liqued layer
            (determines number of drops). If single float it will used as cutout_threshold.
            If single float the number will be sampled from `(0, cutout_threshold)`.
            If tuple of float cutout_threshold will be sampled from range `(cutout_threshold[0], cutout_threshold[1])`.
            If you want constant value use `(cutout_threshold, cutout_threshold)`.
            Default: (0.68, 0.68).
        intensity (tuple[float, float] | floats): Intensity of corruption.
            If single float the number will be sampled from `(0, intensity)`.
            If tuple of float intensity will be sampled from range `(intensity[0], intensity[1])`.
            If you want constant value use `(intensity, intensity)`.
            Default: (0.6, 0.6).
        mode (str, or list[str]): Type of corruption. Currently, supported options are 'rain' and 'mud'.
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
        gauss_sigma: NonNegativeFloatRangeType = (2, 2)
        cutout_threshold: ZeroOneRangeType = (0.68, 0.68)
        intensity: ZeroOneRangeType = (0.6, 0.6)
        mode: SpatterMode | Sequence[SpatterMode] = Field(
            default="rain",
            description="Type of corruption ('rain', 'mud').",
        )
        color: Sequence[int] | dict[str, Sequence[int]] | None = None

        @field_validator("mode")
        @classmethod
        def check_mode(cls, mode: SpatterMode | Sequence[SpatterMode]) -> Sequence[SpatterMode]:
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
        mode: SpatterMode | Sequence[SpatterMode] = "rain",
        color: Sequence[int] | dict[str, Sequence[int]] | None = None,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.mean = cast(tuple[float, float], mean)
        self.std = cast(tuple[float, float], std)
        self.gauss_sigma = cast(tuple[float, float], gauss_sigma)
        self.cutout_threshold = cast(tuple[float, float], cutout_threshold)
        self.intensity = cast(tuple[float, float], intensity)
        self.mode = mode
        self.color = cast(dict[str, Sequence[int]], color)

    def apply(
        self,
        img: np.ndarray,
        non_mud: np.ndarray,
        mud: np.ndarray,
        drops: np.ndarray,
        mode: SpatterMode,
        **params: dict[str, Any],
    ) -> np.ndarray:
        non_rgb_error(img)
        return fmain.spatter(img, non_mud, mud, drops, mode)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]

        mean = self.py_random.uniform(*self.mean)
        std = self.py_random.uniform(*self.std)
        cutout_threshold = self.py_random.uniform(*self.cutout_threshold)
        sigma = self.py_random.uniform(*self.gauss_sigma)
        mode = self.py_random.choice(self.mode)
        intensity = self.py_random.uniform(*self.intensity)
        color = np.array(self.color[mode]) / 255.0

        liquid_layer = self.random_generator.normal(size=(height, width), loc=mean, scale=std)
        liquid_layer = gaussian_filter(liquid_layer, sigma=sigma, mode="nearest")
        liquid_layer[liquid_layer < cutout_threshold] = 0

        if mode == "rain":
            liquid_layer = clip(liquid_layer * 255, np.uint8, inplace=False)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = clip(fblur.blur(dist, 3), np.uint8, inplace=True)
            dist = fmain.equalize(dist)

            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = fmain.convolve(dist, ker)
            dist = fblur.blur(dist, 3).astype(np.float32)

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

    def get_transform_init_args_names(self) -> tuple[str, str, str, str, str, str, str]:
        return "mean", "std", "gauss_sigma", "intensity", "cutout_threshold", "mode", "color"


class ChromaticAberration(ImageOnlyTransform):
    """Add lateral chromatic aberration by distorting the red and blue channels of the input image.

    Chromatic aberration is an optical effect that occurs when a lens fails to focus all colors to the same point.
    This transform simulates this effect by applying different radial distortions to the red and blue channels
    of the image, while leaving the green channel unchanged.

    Args:
        primary_distortion_limit (tuple[float, float] | float): Range of the primary radial distortion coefficient.
            If a single float value is provided, the range
            will be (-primary_distortion_limit, primary_distortion_limit).
            This parameter controls the distortion in the center of the image:
            - Positive values result in pincushion distortion (edges bend inward)
            - Negative values result in barrel distortion (edges bend outward)
            Default: (-0.02, 0.02).

        secondary_distortion_limit (tuple[float, float] | float): Range of the secondary radial distortion coefficient.
            If a single float value is provided, the range
            will be (-secondary_distortion_limit, secondary_distortion_limit).
            This parameter controls the distortion in the corners of the image:
            - Positive values enhance pincushion distortion
            - Negative values enhance barrel distortion
            Default: (-0.05, 0.05).

        mode (Literal["green_purple", "red_blue", "random"]): Type of color fringing to apply. Options are:
            - 'green_purple': Distorts red and blue channels in opposite directions, creating green-purple fringing.
            - 'red_blue': Distorts red and blue channels in the same direction, creating red-blue fringing.
            - 'random': Randomly chooses between 'green_purple' and 'red_blue' modes for each application.
            Default: 'green_purple'.

        interpolation (InterpolationType): Flag specifying the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - This transform only affects RGB images. Grayscale images will raise an error.
        - The strength of the effect depends on both primary and secondary distortion limits.
        - Higher absolute values for distortion limits will result in more pronounced chromatic aberration.
        - The 'green_purple' mode tends to produce more noticeable effects than 'red_blue'.

    Example:
        >>> import albumentations as A
        >>> import cv2
        >>> transform = A.ChromaticAberration(
        ...     primary_distortion_limit=0.05,
        ...     secondary_distortion_limit=0.1,
        ...     mode='green_purple',
        ...     interpolation=cv2.INTER_LINEAR,
        ...     p=1.0
        ... )
        >>> transformed = transform(image=image)
        >>> aberrated_image = transformed['image']

    References:
        - https://en.wikipedia.org/wiki/Chromatic_aberration
        - https://www.researchgate.net/publication/320691320_Chromatic_Aberration_in_Digital_Images
    """

    class InitSchema(BaseTransformInitSchema):
        primary_distortion_limit: SymmetricRangeType
        secondary_distortion_limit: SymmetricRangeType
        mode: ChromaticAberrationMode
        interpolation: InterpolationType

    def __init__(
        self,
        primary_distortion_limit: ScaleFloatType = (-0.02, 0.02),
        secondary_distortion_limit: ScaleFloatType = (-0.05, 0.05),
        mode: ChromaticAberrationMode = "green_purple",
        interpolation: InterpolationType = cv2.INTER_LINEAR,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.primary_distortion_limit = cast(tuple[float, float], primary_distortion_limit)
        self.secondary_distortion_limit = cast(tuple[float, float], secondary_distortion_limit)
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
        non_rgb_error(img)
        return fmain.chromatic_aberration(
            img,
            primary_distortion_red,
            secondary_distortion_red,
            primary_distortion_blue,
            secondary_distortion_blue,
            self.interpolation,
        )

    def get_params(self) -> dict[str, float]:
        primary_distortion_red = self.py_random.uniform(*self.primary_distortion_limit)
        secondary_distortion_red = self.py_random.uniform(*self.secondary_distortion_limit)
        primary_distortion_blue = self.py_random.uniform(*self.primary_distortion_limit)
        secondary_distortion_blue = self.py_random.uniform(*self.secondary_distortion_limit)

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

    def get_transform_init_args_names(self) -> tuple[str, str, str, str]:
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
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Targets:
        image, mask, keypoints, bboxes

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

    _targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)

    class InitSchema(BaseTransformInitSchema):
        scale: OnePlusIntRangeType
        operation: MorphologyMode

    def __init__(
        self,
        scale: ScaleIntType = (2, 3),
        operation: MorphologyMode = "dilation",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.scale = cast(tuple[int, int], scale)
        self.operation = operation

    def apply(self, img: np.ndarray, kernel: tuple[int, int], **params: Any) -> np.ndarray:
        return fmain.morphology(img, kernel, self.operation)

    def apply_to_bboxes(self, bboxes: np.ndarray, kernel: tuple[int, int], **params: Any) -> np.ndarray:
        image_shape = params["shape"]

        denormalized_boxes = denormalize_bboxes(bboxes, image_shape)

        result = fmain.bboxes_morphology(denormalized_boxes, kernel, self.operation, image_shape)

        return normalize_bboxes(result, image_shape)

    def apply_to_keypoints(self, keypoints: np.ndarray, kernel: tuple[int, int], **params: Any) -> np.ndarray:
        return keypoints

    def get_params(self) -> dict[str, float]:
        return {
            "kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.scale),
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("scale", "operation")


PLANKIAN_JITTER_CONST = {
    "MAX_TEMP": max(*fmain.PLANCKIAN_COEFFS["blackbody"].keys(), *fmain.PLANCKIAN_COEFFS["cied"].keys()),
    "MIN_BLACKBODY_TEMP": min(fmain.PLANCKIAN_COEFFS["blackbody"].keys()),
    "MIN_CIED_TEMP": min(fmain.PLANCKIAN_COEFFS["cied"].keys()),
    "WHITE_TEMP": 6_000,
    "SAMPLING_TEMP_PROB": 0.4,
}


class PlanckianJitter(ImageOnlyTransform):
    """Applies Planckian Jitter to the input image, simulating color temperature variations in illumination.

    This transform adjusts the color of an image to mimic the effect of different color temperatures
    of light sources, based on Planck's law of black body radiation. It can simulate the appearance
    of an image under various lighting conditions, from warm (reddish) to cool (bluish) color casts.

    PlanckianJitter vs. ColorJitter:
    PlanckianJitter is fundamentally different from ColorJitter in its approach and use cases:
    1. Physics-based: PlanckianJitter is grounded in the physics of light, simulating real-world
       color temperature changes. ColorJitter applies arbitrary color adjustments.
    2. Natural effects: This transform produces color shifts that correspond to natural lighting
       variations, making it ideal for outdoor scene simulation or color constancy problems.
    3. Single parameter: Color changes are controlled by a single, physically meaningful parameter
       (color temperature), unlike ColorJitter's multiple abstract parameters.
    4. Correlated changes: Color shifts are correlated across channels in a way that mimics natural
       light, whereas ColorJitter can make independent channel adjustments.

    When to use PlanckianJitter:
    - Simulating different times of day or lighting conditions in outdoor scenes
    - Augmenting data for computer vision tasks that need to be robust to natural lighting changes
    - Preparing synthetic data to better match real-world lighting variations
    - Color constancy research or applications
    - When you need physically plausible color variations rather than arbitrary color changes

    The logic behind PlanckianJitter:
    As the color temperature increases:
    1. Lower temperatures (around 3000K) produce warm, reddish tones, simulating sunset or incandescent lighting.
    2. Mid-range temperatures (around 5500K) correspond to daylight.
    3. Higher temperatures (above 7000K) result in cool, bluish tones, similar to overcast sky or shade.
    This progression mimics the natural variation of sunlight throughout the day and in different weather conditions.

    Args:
        mode (Literal["blackbody", "cied"]): The mode of the transformation.
            - "blackbody": Simulates blackbody radiation color changes.
            - "cied": Uses the CIE D illuminant series for color temperature simulation.
            Default: "blackbody"

        temperature_limit (tuple[int, int] | None): The range of color temperatures (in Kelvin) to sample from.
            - For "blackbody" mode: Should be within [3000K, 15000K]. Default: (3000, 15000)
            - For "cied" mode: Should be within [4000K, 15000K]. Default: (4000, 15000)
            If None, the default ranges will be used based on the selected mode.
            Higher temperatures produce cooler (bluish) images, lower temperatures produce warmer (reddish) images.

        sampling_method (Literal["uniform", "gaussian"]): Method to sample the temperature.
            - "uniform": Samples uniformly across the specified range.
            - "gaussian": Samples from a Gaussian distribution centered at 6500K (approximate daylight).
            Default: "uniform"

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The transform preserves the overall brightness of the image while shifting its color.
        - The "blackbody" mode provides a wider range of color shifts, especially in the lower (warmer) temperatures.
        - The "cied" mode is based on standard illuminants and may provide more realistic daylight variations.
        - The Gaussian sampling method tends to produce more subtle variations, as it's centered around daylight.
        - Unlike ColorJitter, this transform ensures that color changes are physically plausible and correlated
          across channels, maintaining the natural appearance of the scene under different lighting conditions.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> transform = A.PlanckianJitter(mode="blackbody",
        ...                               temperature_range=(3000, 9000),
        ...                               sampling_method="uniform",
        ...                               p=1.0)
        >>> result = transform(image=image)
        >>> jittered_image = result["image"]

    References:
        - Planck's law: https://en.wikipedia.org/wiki/Planck%27s_law
        - CIE Standard Illuminants: https://en.wikipedia.org/wiki/Standard_illuminant
        - Color temperature: https://en.wikipedia.org/wiki/Color_temperature
        - Implementation inspired by: https://github.com/TheZino/PlanckianJitter
    """

    class InitSchema(BaseTransformInitSchema):
        mode: Literal["blackbody", "cied"]
        temperature_limit: Annotated[tuple[int, int], AfterValidator(nondecreasing)] | None
        sampling_method: Literal["uniform", "gaussian"]

        @model_validator(mode="after")
        def validate_temperature(self) -> Self:
            max_temp = int(PLANKIAN_JITTER_CONST["MAX_TEMP"])

            if self.temperature_limit is None:
                if self.mode == "blackbody":
                    self.temperature_limit = int(PLANKIAN_JITTER_CONST["MIN_BLACKBODY_TEMP"]), max_temp
                elif self.mode == "cied":
                    self.temperature_limit = int(PLANKIAN_JITTER_CONST["MIN_CIED_TEMP"]), max_temp
            else:
                if self.mode == "blackbody" and (
                    min(self.temperature_limit) < PLANKIAN_JITTER_CONST["MIN_BLACKBODY_TEMP"]
                    or max(self.temperature_limit) > max_temp
                ):
                    raise ValueError("Temperature limits for blackbody should be in [3000, 15000] range")
                if self.mode == "cied" and (
                    min(self.temperature_limit) < PLANKIAN_JITTER_CONST["MIN_CIED_TEMP"]
                    or max(self.temperature_limit) > max_temp
                ):
                    raise ValueError("Temperature limits for CIED should be in [4000, 15000] range")

                if not self.temperature_limit[0] <= PLANKIAN_JITTER_CONST["WHITE_TEMP"] <= self.temperature_limit[1]:
                    raise ValueError("White temperature should be within the temperature limits")

            return self

    def __init__(
        self,
        mode: Literal["blackbody", "cied"] = "blackbody",
        temperature_limit: tuple[int, int] | None = None,
        sampling_method: Literal["uniform", "gaussian"] = "uniform",
        always_apply: bool | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p, always_apply=always_apply)

        self.mode = mode
        self.temperature_limit = cast(tuple[int, int], temperature_limit)
        self.sampling_method = sampling_method

    def apply(self, img: np.ndarray, temperature: int, **params: Any) -> np.ndarray:
        return fmain.planckian_jitter(img, temperature, mode=self.mode)

    def get_params(self) -> dict[str, Any]:
        sampling_prob_boundary = PLANKIAN_JITTER_CONST["SAMPLING_TEMP_PROB"]
        sampling_temp_boundary = PLANKIAN_JITTER_CONST["WHITE_TEMP"]

        if self.sampling_method == "uniform":
            # Split into 2 cases to avoid selecting cold temperatures (>6000) too often
            if self.py_random.random() < sampling_prob_boundary:
                temperature = self.py_random.uniform(
                    self.temperature_limit[0],
                    sampling_temp_boundary,
                )
            else:
                temperature = self.py_random.uniform(
                    sampling_temp_boundary,
                    self.temperature_limit[1],
                )
        elif self.sampling_method == "gaussian":
            # Sample values from asymmetric gaussian distribution
            if self.py_random.random() < sampling_prob_boundary:
                # Left side
                shift = np.abs(
                    self.py_random.gauss(
                        0,
                        np.abs(sampling_temp_boundary - self.temperature_limit[0]) / 3,
                    ),
                )
                temperature = sampling_temp_boundary - shift
            else:
                # Right side
                shift = np.abs(
                    self.py_random.gauss(
                        0,
                        np.abs(self.temperature_limit[1] - sampling_temp_boundary) / 3,
                    ),
                )
                temperature = sampling_temp_boundary + shift
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        # Ensure temperature is within the valid range
        temperature = np.clip(temperature, self.temperature_limit[0], self.temperature_limit[1])

        return {"temperature": int(temperature)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "mode", "temperature_limit", "sampling_method"


class ShotNoise(ImageOnlyTransform):
    """Apply shot noise to the image by modeling photon counting as a Poisson process.

    Shot noise (also known as Poisson noise) occurs in imaging due to the quantum nature of light.
    When photons hit an imaging sensor, they arrive at random times following Poisson statistics.
    This transform simulates this physical process in linear light space by:
    1. Converting to linear space (removing gamma)
    2. Treating each pixel value as an expected photon count
    3. Sampling actual photon counts from a Poisson distribution
    4. Converting back to display space (reapplying gamma)

    The noise characteristics follow real camera behavior:
    - Noise variance equals signal mean in linear space (Poisson statistics)
    - Brighter regions have more absolute noise but less relative noise
    - Darker regions have less absolute noise but more relative noise
    - Noise is generated independently for each pixel and color channel

    Args:
        scale_range (tuple[float, float]): Range for sampling the noise scale factor.
            Represents the reciprocal of the expected photon count per unit intensity.
            Higher values mean more noise:
            - scale = 0.1: ~100 photons per unit intensity (low noise)
            - scale = 1.0: ~1 photon per unit intensity (moderate noise)
            - scale = 10.0: ~0.1 photons per unit intensity (high noise)
            Default: (0.1, 0.3)
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - Performs calculations in linear light space (gamma = 2.2)
        - Preserves the image's mean intensity
        - Memory efficient with in-place operations
        - Thread-safe with independent random seeds

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> # Generate synthetic image
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> # Apply moderate shot noise
        >>> transform = A.ShotNoise(scale_range=(0.1, 1.0), p=1.0)
        >>> noisy_image = transform(image=image)["image"]

    References:
        - Shot noise: https://en.wikipedia.org/wiki/Shot_noise
        - Original paper: https://doi.org/10.1002/andp.19183622304 (Schottky, 1918)
        - Poisson process: https://en.wikipedia.org/wiki/Poisson_point_process
        - Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction
    """

    class InitSchema(BaseTransformInitSchema):
        scale_range: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_0plus)]

    def __init__(self, scale_range: tuple[float, float] = (0.1, 0.3), always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p, always_apply=always_apply)
        self.scale_range = scale_range

    def apply(self, img: np.ndarray, scale: float, random_seed: int, **params: Any) -> np.ndarray:
        return fmain.shot_noise(img, scale, np.random.default_rng(random_seed))

    def get_params(self) -> dict[str, Any]:
        return {
            "scale": self.py_random.uniform(*self.scale_range),
            "random_seed": self.random_generator.integers(0, 2**32 - 1),
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("scale_range",)
