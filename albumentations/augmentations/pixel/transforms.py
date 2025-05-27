"""Pixel-level transformations for image augmentation.

This module contains transforms that modify pixel values without changing the geometry of the image.
Includes transforms for adjusting color, brightness, contrast, adding noise, simulating weather effects,
and other pixel-level manipulations.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Sequence
from typing import Annotated, Any, Callable, Union, cast

import albucore
import cv2
import numpy as np
from albucore import (
    MAX_VALUES_BY_DTYPE,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    batch_transform,
    get_num_channels,
    is_grayscale_image,
    is_rgb_image,
    multiply,
    normalize,
    normalize_per_image,
)
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from scipy import special
from typing_extensions import Literal, Self

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.blur import functional as fblur
from albumentations.augmentations.blur.transforms import BlurInitSchema
from albumentations.augmentations.pixel import functional as fpixel
from albumentations.augmentations.utils import check_range, non_rgb_error
from albumentations.core.pydantic import (
    NonNegativeFloatRangeType,
    OnePlusFloatRangeType,
    OnePlusIntRangeType,
    SymmetricRangeType,
    ZeroOneRangeType,
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)
from albumentations.core.type_definitions import (
    MAX_RAIN_ANGLE,
    NUM_RGB_CHANNELS,
    PAIR,
    SEVEN,
)
from albumentations.core.utils import to_tuple

__all__ = [
    "CLAHE",
    "AdditiveNoise",
    "AutoContrast",
    "ChannelShuffle",
    "ChromaticAberration",
    "ColorJitter",
    "Downscale",
    "Emboss",
    "Equalize",
    "FancyPCA",
    "GaussNoise",
    "HEStain",
    "HueSaturationValue",
    "ISONoise",
    "Illumination",
    "ImageCompression",
    "InvertImg",
    "MultiplicativeNoise",
    "Normalize",
    "PlanckianJitter",
    "PlasmaBrightnessContrast",
    "PlasmaShadow",
    "Posterize",
    "RGBShift",
    "RandomBrightnessContrast",
    "RandomFog",
    "RandomGamma",
    "RandomGravel",
    "RandomRain",
    "RandomShadow",
    "RandomSnow",
    "RandomSunFlare",
    "RandomToneCurve",
    "RingingOvershoot",
    "SaltAndPepper",
    "Sharpen",
    "ShotNoise",
    "Solarize",
    "Spatter",
    "Superpixels",
    "ToGray",
    "ToRGB",
    "ToSepia",
    "UnsharpMask",
]

NUM_BITS_ARRAY_LENGTH = 3
TWENTY = 20


class Normalize(ImageOnlyTransform):
    """Applies various normalization techniques to an image. The specific normalization technique can be selected
        with the `normalization` parameter.

    Standard normalization is applied using the formula:
        `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`.
        Other normalization techniques adjust the image based on global or per-channel statistics,
        or scale pixel values to a specified range.

    Args:
        mean (tuple[float, float] | float | None): Mean values for standard normalization.
            For "standard" normalization, the default values are ImageNet mean values: (0.485, 0.456, 0.406).
        std (tuple[float, float] | float | None): Standard deviation values for standard normalization.
            For "standard" normalization, the default values are ImageNet standard deviation :(0.229, 0.224, 0.225).
        max_pixel_value (float | None): Maximum possible pixel value, used for scaling in standard normalization.
            Defaults to 255.0.
        normalization (Literal["standard", "image", "image_per_channel", "min_max", "min_max_per_channel"]):
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
        - For YOLO normalization, use mean values of (0, 0, 0) and std values of (1, 1, 1).
        - This transform is often used as a final step in image preprocessing pipelines to
          prepare images for neural network input.

    Examples:
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
        mean: tuple[float, ...] | float | None
        std: tuple[float, ...] | float | None
        max_pixel_value: float | None
        normalization: Literal[
            "standard",
            "image",
            "image_per_channel",
            "min_max",
            "min_max_per_channel",
        ]

        @model_validator(mode="after")
        def _validate_normalization(self) -> Self:
            if (
                self.mean is None
                or self.std is None
                or (self.max_pixel_value is None and self.normalization == "standard")
            ):
                raise ValueError(
                    "mean, std, and max_pixel_value must be provided for standard normalization.",
                )
            return self

    def __init__(
        self,
        mean: tuple[float, ...] | float | None = (0.485, 0.456, 0.406),
        std: tuple[float, ...] | float | None = (0.229, 0.224, 0.225),
        max_pixel_value: float | None = 255.0,
        normalization: Literal[
            "standard",
            "image",
            "image_per_channel",
            "min_max",
            "min_max_per_channel",
        ] = "standard",
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.mean = mean
        self.mean_np = np.array(mean, dtype=np.float32) * max_pixel_value
        self.std = std
        self.denominator = np.reciprocal(
            np.array(std, dtype=np.float32) * max_pixel_value,
        )
        self.max_pixel_value = max_pixel_value
        self.normalization = normalization

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply normalization to the input image.

        Args:
            img (np.ndarray): The input image to normalize.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The normalized image.

        """
        if self.normalization == "standard":
            return normalize(
                img,
                self.mean_np,
                self.denominator,
            )
        return normalize_per_image(img, self.normalization)

    @batch_transform("channel", has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply normalization to a batch of images.

        Args:
            images (np.ndarray): Batch of images to normalize with shape (batch, height, width, channels).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Normalized batch of images.

        """
        return self.apply(images, **params)

    @batch_transform("channel", has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply normalization to a 3D volume.

        Args:
            volume (np.ndarray): 3D volume to normalize with shape (depth, height, width, channels).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Normalized 3D volume.

        """
        return self.apply(volume, **params)

    @batch_transform("channel", has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply normalization to a batch of 3D volumes.

        Args:
            volumes (np.ndarray): Batch of 3D volumes to normalize with shape (batch, depth, height, width, channels).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Normalized batch of 3D volumes.

        """
        return self.apply(volumes, **params)


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

    Examples:
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
        quality_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, 100)),
            AfterValidator(nondecreasing),
        ]

        compression_type: Literal["jpeg", "webp"]

    def __init__(
        self,
        compression_type: Literal["jpeg", "webp"] = "jpeg",
        quality_range: tuple[int, int] = (99, 100),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.quality_range = quality_range
        self.compression_type = compression_type

    def apply(
        self,
        img: np.ndarray,
        quality: int,
        image_type: Literal[".jpg", ".webp"],
        **params: Any,
    ) -> np.ndarray:
        """Apply compression to the input image.

        Args:
            img (np.ndarray): The input image to be compressed.
            quality (int): Compression quality level (1-100).
            image_type (Literal[".jpg", ".webp"]): File extension indicating compression format.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The compressed image.

        """
        return fpixel.image_compression(img, quality, image_type)

    def get_params(self) -> dict[str, int | str]:
        """Generate random parameters for the transform.

        Returns:
            dict[str, int | str]: Dictionary with the following keys:
                - "quality" (int): Random quality value within the specified range.
                - "image_type" (str): File extension for the chosen compression type.

        """
        image_type = ".jpg" if self.compression_type == "jpeg" else ".webp"

        return {
            "quality": self.py_random.randint(*self.quality_range),
            "image_type": image_type,
        }


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
        snow_point_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

        brightness_coeff: float = Field(gt=0)
        method: Literal["bleach", "texture"]

    def __init__(
        self,
        brightness_coeff: float = 2.5,
        snow_point_range: tuple[float, float] = (0.1, 0.3),
        method: Literal["bleach", "texture"] = "bleach",
        p: float = 0.5,
    ):
        super().__init__(p=p)

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
        """Apply the snow effect to the input image.

        Args:
            img (np.ndarray): The input image to apply the snow effect to.
            snow_point (float): The snow point threshold.
            snow_texture (np.ndarray): The snow texture overlay.
            sparkle_mask (np.ndarray): The sparkle mask for the snow effect.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied snow effect.

        """
        non_rgb_error(img)

        if self.method == "bleach":
            return fpixel.add_snow_bleach(img, snow_point, self.brightness_coeff)
        if self.method == "texture":
            return fpixel.add_snow_texture(
                img,
                snow_point,
                self.brightness_coeff,
                snow_texture,
                sparkle_mask,
            )

        raise ValueError(f"Unknown snow method: {self.method}")

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, np.ndarray | None]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, np.ndarray | None]: Dictionary with the following keys:
                - "snow_point" (np.ndarray | None): The snow point threshold.
                - "snow_texture" (np.ndarray | None): The snow texture overlay.
                - "sparkle_mask" (np.ndarray | None): The sparkle mask for the snow effect.

        """
        image_shape = params["shape"][:2]
        result = {
            "snow_point": self.py_random.uniform(*self.snow_point_range),
            "snow_texture": None,
            "sparkle_mask": None,
        }

        if self.method == "texture":
            snow_texture, sparkle_mask = fpixel.generate_snow_textures(
                img_shape=image_shape,
                random_generator=self.random_generator,
            )
            result["snow_texture"] = snow_texture
            result["sparkle_mask"] = sparkle_mask

        return result


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
        def _validate_gravel_roi(self) -> Self:
            gravel_lower_x, gravel_lower_y, gravel_upper_x, gravel_upper_y = self.gravel_roi
            if not 0 <= gravel_lower_x < gravel_upper_x <= 1 or not 0 <= gravel_lower_y < gravel_upper_y <= 1:
                raise ValueError(f"Invalid gravel_roi. Got: {self.gravel_roi}.")
            return self

    def __init__(
        self,
        gravel_roi: tuple[float, float, float, float] = (0.1, 0.4, 0.9, 0.9),
        number_of_patches: int = 2,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.gravel_roi = gravel_roi
        self.number_of_patches = number_of_patches

    def generate_gravel_patch(
        self,
        rectangular_roi: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Generate gravel particles within a specified rectangular region.

        Args:
            rectangular_roi (tuple[int, int, int, int]): The rectangular region where gravel
                particles will be generated, specified as (x_min, y_min, x_max, y_max) in pixel coordinates.

        Returns:
            np.ndarray: An array of gravel particles with shape (count, 2), where count is the number of particles.
            Each row contains the (x, y) coordinates of a gravel particle.

        """
        x_min, y_min, x_max, y_max = rectangular_roi
        area = abs((x_max - x_min) * (y_max - y_min))
        count = area // 10
        gravels = np.empty([count, 2], dtype=np.int64)
        gravels[:, 0] = self.random_generator.integers(x_min, x_max, count)
        gravels[:, 1] = self.random_generator.integers(y_min, y_max, count)
        return gravels

    def apply(
        self,
        img: np.ndarray,
        gravels_infos: list[Any],
        **params: Any,
    ) -> np.ndarray:
        """Apply the gravel effect to the input image.

        Args:
            img (np.ndarray): The input image to apply the gravel effect to.
            gravels_infos (list[Any]): Information about the gravel particles.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied gravel effect.

        """
        return fpixel.add_gravel(img, gravels_infos)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, np.ndarray]: Dictionary with the following keys:
                - "gravels_infos" (np.ndarray): Information about the gravel particles.

        """
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


class RandomRain(ImageOnlyTransform):
    """Adds rain effects to an image.

    This transform simulates rainfall by overlaying semi-transparent streaks onto the image,
    creating a realistic rain effect. It can be used to augment datasets for computer vision
    tasks that need to perform well in rainy conditions.

    Args:
        slant_range (tuple[float, float]): Range for the rain slant angle in degrees.
            Negative values slant to the left, positive to the right. Default: (-10, 10).
        drop_length (int | None): Length of the rain drops in pixels.
            If None, drop length will be automatically calculated as height // 8.
            This allows the rain effect to scale with the image size.
            Default: None
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
        slant_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-MAX_RAIN_ANGLE, MAX_RAIN_ANGLE)),
        ]
        drop_length: int | None
        drop_width: int = Field(ge=1)
        drop_color: tuple[int, int, int]
        blur_value: int = Field(ge=1)
        brightness_coefficient: float = Field(gt=0, le=1)
        rain_type: Literal["drizzle", "heavy", "torrential", "default"]

    def __init__(
        self,
        slant_range: tuple[float, float] = (-10, 10),
        drop_length: int | None = None,
        drop_width: int = 1,
        drop_color: tuple[int, int, int] = (200, 200, 200),
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        rain_type: Literal["drizzle", "heavy", "torrential", "default"] = "default",
        p: float = 0.5,
    ):
        super().__init__(p=p)
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
        slant: float,
        drop_length: int,
        rain_drops: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the rain effect to the input image.

        Args:
            img (np.ndarray): The input image to apply the rain effect to.
            slant (float): The slant angle of the rain.
            drop_length (int): The length of the rain drops.
            rain_drops (np.ndarray): The coordinates of the rain drops.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied rain effect.

        """
        non_rgb_error(img)

        return fpixel.add_rain(
            img,
            slant,
            drop_length,
            self.drop_width,
            self.drop_color,
            self.blur_value,
            self.brightness_coefficient,
            rain_drops,
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, Any]: Dictionary with the following keys:
                - "drop_length" (int): The length of the rain drops.
                - "slant" (float): The slant angle of the rain.
                - "rain_drops" (np.ndarray): The coordinates of the rain drops.

        """
        height, width = params["shape"][:2]

        # Simpler calculations, directly following Kornia
        if self.rain_type == "drizzle":
            num_drops = height // 4
        elif self.rain_type == "heavy":
            num_drops = height
        elif self.rain_type == "torrential":
            num_drops = height * 2
        else:
            num_drops = height // 3

        drop_length = max(1, height // 8) if self.drop_length is None else self.drop_length

        # Simplified slant calculation
        slant = self.py_random.uniform(*self.slant_range)

        # Single random call for all coordinates
        if num_drops > 0:
            # Generate all coordinates in one call
            coords = self.random_generator.integers(
                low=[0, 0],
                high=[width, height - drop_length],
                size=(num_drops, 2),
                dtype=np.int32,
            )
            rain_drops = coords
        else:
            rain_drops = np.empty((0, 2), dtype=np.int32)

        return {"drop_length": drop_length, "slant": slant, "rain_drops": rain_drops}


class RandomFog(ImageOnlyTransform):
    """Simulates fog for the image by adding random fog-like artifacts.

    This transform creates a fog effect by generating semi-transparent overlays
    that mimic the visual characteristics of fog. The fog intensity and distribution
    can be controlled to create various fog-like conditions. An image size dependent
    Gaussian blur is applied to the resulting image

    Args:
        fog_coef_range (tuple[float, float]): Range for fog intensity coefficient. Should be in [0, 1] range.
        alpha_coef (float): Transparency of the fog circles. Should be in [0, 1] range. Default: 0.08.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        - The fog effect is created by overlaying semi-transparent circles on the image.
        - Higher fog coefficient values result in denser fog effects.
        - The fog is typically denser in the center of the image and gradually decreases towards the edges.
        - Image is blurred to decrease the sharpness
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
        5. A Gaussian blur dependent on the shorter dimension is applied

        The final pixel value is calculated as:
        output = blur((1 - alpha) * original_pixel + alpha * fog_color)

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
        fog_coef_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

        alpha_coef: float = Field(ge=0, le=1)

    def __init__(
        self,
        alpha_coef: float = 0.08,
        fog_coef_range: tuple[float, float] = (0.3, 1),
        p: float = 0.5,
    ):
        super().__init__(p=p)
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
        """Apply the fog effect to the input image.

        Args:
            img (np.ndarray): The input image to apply the fog effect to.
            particle_positions (list[tuple[int, int]]): The coordinates of the fog particles.
            radiuses (list[int]): The radii of the fog particles.
            intensity (float): The intensity of the fog.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied fog effect.

        """
        non_rgb_error(img)
        return fpixel.add_fog(
            img,
            intensity,
            self.alpha_coef,
            particle_positions,
            radiuses,
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, Any]: Dictionary with the following keys:
                - "intensity" (float): The intensity of the fog.
                - "particle_positions" (list[tuple[int, int]]): The coordinates of the fog particles.
                - "radiuses" (list[int]): The radii of the fog particles.

        """
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
            particles_in_region = int(
                area / (fog_region_size * fog_region_size) * intensity * 10,
            )

            for _ in range(particles_in_region):
                # Generate random positions within the current region
                x = self.py_random.randint(
                    center_x - current_width // 2,
                    center_x + current_width // 2,
                )
                y = self.py_random.randint(
                    center_y - current_height // 2,
                    center_y + current_height // 2,
                )
                particle_positions.append((x, y))

            # Shrink the region for the next iteration
            current_width = int(current_width * (1 - shrink_factor))
            current_height = int(current_height * (1 - shrink_factor))

            iteration += 1

        radiuses = fpixel.get_fog_particle_radiuses(
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
            more realistic optical phenomena. Default: "overlay".

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        3

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
        src_radius: int = Field(gt=1)
        src_color: tuple[int, ...]

        angle_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

        num_flare_circles_range: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        method: Literal["overlay", "physics_based"]

        @model_validator(mode="after")
        def _validate_parameters(self) -> Self:
            (
                flare_center_lower_x,
                flare_center_lower_y,
                flare_center_upper_x,
                flare_center_upper_y,
            ) = self.flare_roi
            if (
                not 0 <= flare_center_lower_x < flare_center_upper_x <= 1
                or not 0 <= flare_center_lower_y < flare_center_upper_y <= 1
            ):
                raise ValueError(f"Invalid flare_roi. Got: {self.flare_roi}")

            return self

    def __init__(
        self,
        flare_roi: tuple[float, float, float, float] = (0, 0, 1, 0.5),
        src_radius: int = 400,
        src_color: tuple[int, ...] = (255, 255, 255),
        angle_range: tuple[float, float] = (0, 1),
        num_flare_circles_range: tuple[int, int] = (6, 10),
        method: Literal["overlay", "physics_based"] = "overlay",
        p: float = 0.5,
    ):
        super().__init__(p=p)

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
        """Apply the sun flare effect to the input image.

        Args:
            img (np.ndarray): The input image to apply the sun flare effect to.
            flare_center (tuple[float, float]): The center of the sun.
            circles (list[Any]): The circles to apply the sun flare effect to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied sun flare effect.

        """
        non_rgb_error(img)
        if self.method == "overlay":
            return fpixel.add_sun_flare_overlay(
                img,
                flare_center,
                self.src_radius,
                self.src_color,
                circles,
            )
        if self.method == "physics_based":
            return fpixel.add_sun_flare_physics_based(
                img,
                flare_center,
                self.src_radius,
                self.src_color,
                circles,
            )

        raise ValueError(f"Invalid method: {self.method}")

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, Any]: Dictionary with the following keys:
                - "circles" (list[Any]): The circles to apply the sun flare effect to.
                - "flare_center" (tuple[float, float]): The center of the sun.

        """
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
            return (
                flare_center_x + t * math.cos(angle),
                flare_center_y + t * math.sin(angle),
            )

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
        num_shadows_limit: Annotated[
            tuple[int, int],
            AfterValidator(check_range_bounds(1, None)),
            AfterValidator(nondecreasing),
        ]
        shadow_dimension: int = Field(ge=3)

        shadow_intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

        @model_validator(mode="after")
        def _validate_shadows(self) -> Self:
            shadow_lower_x, shadow_lower_y, shadow_upper_x, shadow_upper_y = self.shadow_roi

            if not 0 <= shadow_lower_x <= shadow_upper_x <= 1 or not 0 <= shadow_lower_y <= shadow_upper_y <= 1:
                raise ValueError(f"Invalid shadow_roi. Got: {self.shadow_roi}")

            return self

    def __init__(
        self,
        shadow_roi: tuple[float, float, float, float] = (0, 0.5, 1, 1),
        num_shadows_limit: tuple[int, int] = (1, 2),
        shadow_dimension: int = 5,
        shadow_intensity_range: tuple[float, float] = (0.5, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)

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
        """Apply the shadow effect to the input image.

        Args:
            img (np.ndarray): The input image to apply the shadow effect to.
            vertices_list (list[np.ndarray]): The vertices of the shadow polygons.
            intensities (np.ndarray): The intensities of the shadows.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied shadow effect.

        """
        return fpixel.add_shadow(img, vertices_list, intensities)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, list[np.ndarray]]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, list[np.ndarray]]: Dictionary with the following keys:
                - "vertices_list" (list[np.ndarray]): The vertices of the shadow polygons.
                - "intensities" (np.ndarray): The intensities of the shadows.

        """
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
                    self.random_generator.integers(
                        x_min,
                        x_max,
                        size=self.shadow_dimension,
                    ),
                    self.random_generator.integers(
                        y_min,
                        y_max,
                        size=self.shadow_dimension,
                    ),
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
        - "What Else Can Fool Deep Learning? Addressing Color Constancy Errors on Deep Neural Network Performance":
          https://arxiv.org/abs/1912.06960
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
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.scale = scale
        self.per_channel = per_channel

    def apply(
        self,
        img: np.ndarray,
        low_y: float | np.ndarray,
        high_y: float | np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the tone curve to the input image.

        Args:
            img (np.ndarray): The input image to apply the tone curve to.
            low_y (float | np.ndarray): The lower control point of the tone curve.
            high_y (float | np.ndarray): The upper control point of the tone curve.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied tone curve.

        """
        return fpixel.move_tone_curve(img, low_y, high_y)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, Any]: Dictionary with the following keys:
                - "low_y" (float | np.ndarray): The lower control point of the tone curve.
                - "high_y" (float | np.ndarray): The upper control point of the tone curve.

        """
        image = data["image"] if "image" in data else data["images"][0]

        num_channels = get_num_channels(image)

        if self.per_channel and num_channels != 1:
            return {
                "low_y": np.clip(
                    self.random_generator.normal(
                        loc=0.25,
                        scale=self.scale,
                        size=(num_channels,),
                    ),
                    0,
                    1,
                ),
                "high_y": np.clip(
                    self.random_generator.normal(
                        loc=0.75,
                        scale=self.scale,
                        size=(num_channels,),
                    ),
                    0,
                    1,
                ),
            }
        # Same values for all channels
        low_y = np.clip(self.random_generator.normal(loc=0.25, scale=self.scale), 0, 1)
        high_y = np.clip(self.random_generator.normal(loc=0.75, scale=self.scale), 0, 1)

        return {"low_y": low_y, "high_y": high_y}


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

    Examples:
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
        HSV color space: https://en.wikipedia.org/wiki/HSL_and_HSV

    """

    class InitSchema(BaseTransformInitSchema):
        hue_shift_limit: SymmetricRangeType
        sat_shift_limit: SymmetricRangeType
        val_shift_limit: SymmetricRangeType

    def __init__(
        self,
        hue_shift_limit: tuple[float, float] | float = (-20, 20),
        sat_shift_limit: tuple[float, float] | float = (-30, 30),
        val_shift_limit: tuple[float, float] | float = (-20, 20),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.hue_shift_limit = cast("tuple[float, float]", hue_shift_limit)
        self.sat_shift_limit = cast("tuple[float, float]", sat_shift_limit)
        self.val_shift_limit = cast("tuple[float, float]", val_shift_limit)

    def apply(
        self,
        img: np.ndarray,
        hue_shift: int,
        sat_shift: int,
        val_shift: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the hue, saturation, and value shifts to the input image.

        Args:
            img (np.ndarray): The input image to apply the hue, saturation, and value shifts to.
            hue_shift (int): The hue shift value.
            sat_shift (int): The saturation shift value.
            val_shift (int): The value (brightness) shift value.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied hue, saturation, and value shifts.

        """
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "HueSaturationValue transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)
        return fpixel.shift_hsv(img, hue_shift, sat_shift, val_shift)

    def get_params(self) -> dict[str, float]:
        """Generate parameters dependent on the input data.

        Returns:
            dict[str, float]: Dictionary with the following keys:
                - "hue_shift" (float): The hue shift value.
                - "sat_shift" (float): The saturation shift value.
                - "val_shift" (float): The value (brightness) shift value.

        """
        return {
            "hue_shift": self.py_random.uniform(*self.hue_shift_limit),
            "sat_shift": self.py_random.uniform(*self.sat_shift_limit),
            "val_shift": self.py_random.uniform(*self.val_shift_limit),
        }


class Solarize(ImageOnlyTransform):
    """Invert all pixel values above a threshold.

    This transform applies a solarization effect to the input image. Solarization is a phenomenon in
    photography in which the image recorded on a negative or on a photographic print is wholly or
    partially reversed in tone. Dark areas appear light or light areas appear dark.

    In this implementation, all pixel values above a threshold are inverted.

    Args:
        threshold_range (tuple[float, float]): Range for solarizing threshold as a fraction
            of maximum value. The threshold_range should be in the range [0, 1] and will be multiplied by the
            maximum value of the image type (255 for uint8 images or 1.0 for float images).
            Default: (0.5, 0.5) (corresponds to 127.5 for uint8 and 0.5 for float32).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - For uint8 images, pixel values above the threshold are inverted as: 255 - pixel_value
        - For float32 images, pixel values above the threshold are inverted as: 1.0 - pixel_value
        - The threshold is applied to each channel independently
        - The threshold is calculated in two steps:
          1. Sample a value from threshold_range
          2. Multiply by the image's maximum value:
             * For uint8: threshold = sampled_value * 255
             * For float32: threshold = sampled_value * 1.0
        - This transform can create interesting artistic effects or be used for data augmentation

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        # Solarize uint8 image with fixed threshold at 50% of max value (127.5)
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Solarize(threshold_range=(0.5, 0.5), p=1.0)
        >>> solarized_image = transform(image=image)['image']
        >>>
        # Solarize uint8 image with random threshold between 40-60% of max value (102-153)
        >>> transform = A.Solarize(threshold_range=(0.4, 0.6), p=1.0)
        >>> solarized_image = transform(image=image)['image']
        >>>
        # Solarize float32 image at 50% of max value (0.5)
        >>> image = np.random.rand(100, 100, 3).astype(np.float32)
        >>> transform = A.Solarize(threshold_range=(0.5, 0.5), p=1.0)
        >>> solarized_image = transform(image=image)['image']

    Mathematical Formulation:
        Let f be a value sampled from threshold_range (min, max).
        For each pixel value p:
        threshold = f * max_value
        if p > threshold:
            p_new = max_value - p
        else:
            p_new = p

        Where max_value is 255 for uint8 images and 1.0 for float32 images.

    See Also:
        Invert: For inverting all pixel values regardless of a threshold.

    """

    class InitSchema(BaseTransformInitSchema):
        threshold_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        threshold_range: tuple[float, float] = (0.5, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.threshold_range = threshold_range

    def apply(self, img: np.ndarray, threshold: float, **params: Any) -> np.ndarray:
        """Apply the solarize effect to the input image.

        Args:
            img (np.ndarray): The input image to apply the solarize effect to.
            threshold (float): The threshold value.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied solarize effect.

        """
        return fpixel.solarize(img, threshold)

    def get_params(self) -> dict[str, float]:
        """Generate parameters dependent on the input data.

        Returns:
            dict[str, float]: Dictionary with the following key:
                - "threshold" (float): The threshold value.

        """
        return {"threshold": self.py_random.uniform(*self.threshold_range)}


class Posterize(ImageOnlyTransform):
    """Reduces the number of bits for each color channel in the image.

    This transform applies color posterization, a technique that reduces the number of distinct
    colors used in an image. It works by lowering the number of bits used to represent each
    color channel, effectively creating a "poster-like" effect with fewer color gradations.

    Args:
        num_bits (int | tuple[int, int] | list[int] | list[tuple[int, int]]):
            Defines the number of bits to keep for each color channel. Can be specified in several ways:
            - Single int: Same number of bits for all channels. Range: [1, 7].
            - tuple of two ints: (min_bits, max_bits) to randomly choose from. Range for each: [1, 7].
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
        num_bits: int | tuple[int, int] | list[tuple[int, int]]

        @field_validator("num_bits")
        @classmethod
        def _validate_num_bits(
            cls,
            num_bits: Any,
        ) -> tuple[int, int] | list[tuple[int, int]]:
            if isinstance(num_bits, int):
                if num_bits < 1 or num_bits > SEVEN:
                    raise ValueError("num_bits must be in the range [1, 7]")
                return (num_bits, num_bits)
            if isinstance(num_bits, Sequence) and len(num_bits) > PAIR:
                return [to_tuple(i, i) for i in num_bits]
            return to_tuple(num_bits, num_bits)

    def __init__(
        self,
        num_bits: int | tuple[int, int] | list[tuple[int, int]] = 4,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_bits = cast("Union[tuple[int, int], list[tuple[int, int]]]", num_bits)

    def apply(
        self,
        img: np.ndarray,
        num_bits: Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]],
        **params: Any,
    ) -> np.ndarray:
        """Apply the posterize effect to the input image.

        Args:
            img (np.ndarray): The input image to apply the posterize effect to.
            num_bits (Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]]):
                The number of bits to keep for each color channel.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied posterize effect.

        """
        return fpixel.posterize(img, num_bits)

    def get_params(self) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Returns:
            dict[str, Any]: Dictionary with the following key:
                - "num_bits" (Literal[1, 2, 3, 4, 5, 6, 7] | list[Literal[1, 2, 3, 4, 5, 6, 7]]):
                    The number of bits to keep for each color channel.

        """
        if isinstance(self.num_bits, list):
            num_bits = [self.py_random.randint(*i) for i in self.num_bits]
            return {"num_bits": num_bits}
        return {"num_bits": self.py_random.randint(*self.num_bits)}


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
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1,3

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

    Examples:
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
        mode: Literal["cv", "pil"]
        by_channels: bool
        mask: np.ndarray | Callable[..., Any] | None
        mask_params: Sequence[str]

    def __init__(
        self,
        mode: Literal["cv", "pil"] = "cv",
        by_channels: bool = True,
        mask: np.ndarray | Callable[..., Any] | None = None,
        mask_params: Sequence[str] = (),
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.mode = mode
        self.by_channels = by_channels
        self.mask = mask
        self.mask_params = mask_params

    def apply(self, img: np.ndarray, mask: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the equalization effect to the input image.

        Args:
            img (np.ndarray): The input image to apply the equalization effect to.
            mask (np.ndarray): The mask to apply the equalization effect to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied equalization effect.

        """
        if not is_rgb_image(img) and not is_grayscale_image(img):
            raise ValueError("Equalize transform is only supported for RGB and grayscale images.")
        return fpixel.equalize(
            img,
            mode=self.mode,
            by_channels=self.by_channels,
            mask=mask,
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, Any]: Dictionary with the following key:
                - "mask" (np.ndarray): The mask to apply the equalization effect to.

        """
        if not callable(self.mask):
            return {"mask": self.mask}

        mask_params = {"image": data["image"]}
        for key in self.mask_params:
            if key not in data:
                raise KeyError(
                    f"Required parameter '{key}' for mask function is missing in data.",
                )
            mask_params[key] = data[key]

        return {"mask": self.mask(**mask_params)}

    @property
    def targets_as_params(self) -> list[str]:
        """Return the list of parameters that are used for generating the mask.

        Returns:
            list[str]: List of parameter names.

        """
        return [*list(self.mask_params)]


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
        image, volume

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
        brightness_limit: tuple[float, float] | float = (-0.2, 0.2),
        contrast_limit: tuple[float, float] | float = (-0.2, 0.2),
        brightness_by_max: bool = True,
        ensure_safe_range: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.brightness_limit = cast("tuple[float, float]", brightness_limit)
        self.contrast_limit = cast("tuple[float, float]", contrast_limit)
        self.brightness_by_max = brightness_by_max
        self.ensure_safe_range = ensure_safe_range

    def apply(
        self,
        img: np.ndarray,
        alpha: float,
        beta: float,
        **params: Any,
    ) -> np.ndarray:
        """Apply the brightness and contrast adjustment to the input image.

        Args:
            img (np.ndarray): The input image to apply the brightness and contrast adjustment to.
            alpha (float): The contrast adjustment factor.
            beta (float): The brightness adjustment factor.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied brightness and contrast adjustment.

        """
        return albucore.multiply_add(img, alpha, beta, inplace=False)

    def apply_to_images(self, images: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply the brightness and contrast adjustment to a batch of images.

        Args:
            images (np.ndarray): The batch of images to apply the brightness and contrast adjustment to.
            *args (Any): Additional arguments.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The batch of images with the applied brightness and contrast adjustment.

        """
        return self.apply(images, *args, **params)

    def apply_to_volumes(self, volumes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply the brightness and contrast adjustment to a batch of volumes.

        Args:
            volumes (np.ndarray): The batch of volumes to apply the brightness and contrast adjustment to.
            *args (Any): Additional arguments.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The batch of volumes with the applied brightness and contrast adjustment.

        """
        return self.apply(volumes, *args, **params)

    def apply_to_volume(self, volume: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply the brightness and contrast adjustment to a single volume.

        Args:
            volume (np.ndarray): The volume to apply the brightness and contrast adjustment to.
            *args (Any): Additional arguments.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The volume with the applied brightness and contrast adjustment.

        """
        return self.apply(volume, *args, **params)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, float]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, float]: Dictionary with the following keys:
                - "alpha" (float): The contrast adjustment factor.
                - "beta" (float): The brightness adjustment factor.

        """
        image = data["image"] if "image" in data else data["images"][0]

        # Sample initial values
        alpha = 1.0 + self.py_random.uniform(*self.contrast_limit)
        beta = self.py_random.uniform(*self.brightness_limit)

        max_value = MAX_VALUES_BY_DTYPE[image.dtype]
        # Scale beta according to brightness_by_max setting
        beta = beta * max_value if self.brightness_by_max else beta * np.mean(image)

        # Clip values to safe ranges if needed
        if self.ensure_safe_range:
            alpha, beta = fpixel.get_safe_brightness_contrast_params(
                alpha,
                beta,
                max_value,
            )

        return {
            "alpha": alpha,
            "beta": beta,
        }


class GaussNoise(ImageOnlyTransform):
    """Apply Gaussian noise to the input image.

    Args:
        std_range (tuple[float, float]): Range for noise standard deviation as a fraction
            of the maximum value (255 for uint8 images or 1.0 for float images).
            Values should be in range [0, 1]. Default: (0.2, 0.44).
        mean_range (tuple[float, float]): Range for noise mean as a fraction
            of the maximum value (255 for uint8 images or 1.0 for float images).
            Values should be in range [-1, 1]. Default: (0.0, 0.0).
        per_channel (bool): If True, noise will be sampled for each channel independently.
            Otherwise, the noise will be sampled once for all channels. Default: True.
        noise_scale_factor (float): Scaling factor for noise generation. Value should be in the range (0, 1].
            When set to 1, noise is sampled for each pixel independently. If less, noise is sampled for a smaller size
            and resized to fit the shape of the image. Smaller values make the transform faster. Default: 1.0.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The noise parameters (std_range and mean_range) are normalized to [0, 1] range:
          * For uint8 images, they are multiplied by 255
          * For float32 images, they are used directly
        - Setting per_channel=False is faster but applies the same noise to all channels
        - The noise_scale_factor parameter allows for a trade-off between transform speed and noise granularity

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        >>>
        >>> # Apply Gaussian noise with normalized std_range
        >>> transform = A.GaussNoise(std_range=(0.1, 0.2), p=1.0)  # 10-20% of max value
        >>> noisy_image = transform(image=image)['image']

    """

    class InitSchema(BaseTransformInitSchema):
        std_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        mean_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(-1, 1)),
            AfterValidator(nondecreasing),
        ]
        per_channel: bool
        noise_scale_factor: float = Field(gt=0, le=1)

    def __init__(
        self,
        std_range: tuple[float, float] = (0.2, 0.44),  # sqrt(10 / 255), sqrt(50 / 255)
        mean_range: tuple[float, float] = (0.0, 0.0),
        per_channel: bool = True,
        noise_scale_factor: float = 1,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.std_range = std_range
        self.mean_range = mean_range
        self.per_channel = per_channel
        self.noise_scale_factor = noise_scale_factor

    def apply(
        self,
        img: np.ndarray,
        noise_map: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Gaussian noise to the input image.

        Args:
            img (np.ndarray): The input image to apply the Gaussian noise to.
            noise_map (np.ndarray): The noise map to apply to the image.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied Gaussian noise.

        """
        return fpixel.add_noise(img, noise_map)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, float]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, float]: Dictionary with the following key:
                - "noise_map" (np.ndarray): The noise map to apply to the image.

        """
        image = data["image"] if "image" in data else data["images"][0]
        max_value = MAX_VALUES_BY_DTYPE[image.dtype]

        sigma = self.py_random.uniform(*self.std_range)

        mean = self.py_random.uniform(*self.mean_range)

        noise_map = fpixel.generate_noise(
            noise_type="gaussian",
            spatial_mode="per_pixel" if self.per_channel else "shared",
            shape=image.shape,
            params={"mean_range": (mean, mean), "std_range": (sigma, sigma)},
            max_value=max_value,
            approximation=self.noise_scale_factor,
            random_generator=self.random_generator,
        )
        return {"noise_map": noise_map}


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
        image, volume

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)
        >>> result = transform(image=image)
        >>> noisy_image = result["image"]

    References:
        ISO noise in digital photography: https://en.wikipedia.org/wiki/Image_noise#In_digital_cameras

    """

    class InitSchema(BaseTransformInitSchema):
        color_shift: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]
        intensity: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        color_shift: tuple[float, float] = (0.01, 0.05),
        intensity: tuple[float, float] = (0.1, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
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
        """Apply the ISONoise transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the ISONoise transform to.
            color_shift (float): The color shift value.
            intensity (float): The intensity value.
            random_seed (int): The random seed.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied ISONoise transform.

        """
        non_rgb_error(img)
        return fpixel.iso_noise(
            img,
            color_shift,
            intensity,
            np.random.default_rng(random_seed),
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, Any]: Dictionary with the following keys:
                - "color_shift" (float): The color shift value.
                - "intensity" (float): The intensity value.
                - "random_seed" (int): The random seed.

        """
        random_seed = self.random_generator.integers(0, 2**32 - 1)
        return {
            "color_shift": self.py_random.uniform(*self.color_shift),
            "intensity": self.py_random.uniform(*self.intensity),
            "random_seed": random_seed,
        }


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
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1, 3

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0)
        >>> result = transform(image=image)
        >>> clahe_image = result["image"]

    References:
        - Tutorial: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
        - "Contrast Limited Adaptive Histogram Equalization.": https://ieeexplore.ieee.org/document/109340

    """

    class InitSchema(BaseTransformInitSchema):
        clip_limit: OnePlusFloatRangeType
        tile_grid_size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]

    def __init__(
        self,
        clip_limit: tuple[float, float] | float = 4.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.clip_limit = cast("tuple[float, float]", clip_limit)
        self.tile_grid_size = tile_grid_size

    def apply(self, img: np.ndarray, clip_limit: float, **params: Any) -> np.ndarray:
        """Apply the CLAHE transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the CLAHE transform to.
            clip_limit (float): The contrast enhancement limit.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied CLAHE transform.

        """
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "CLAHE transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)

        return fpixel.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self) -> dict[str, float]:
        """Generate parameters dependent on the input data.

        Returns:
            dict[str, float]: Dictionary with the following key:
                - "clip_limit" (float): The contrast enhancement limit.

        """
        return {"clip_limit": self.py_random.uniform(*self.clip_limit)}


class ChannelShuffle(ImageOnlyTransform):
    """Randomly rearrange channels of the image.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Number of channels:
        Any

    Image types:
        uint8, float32

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        >>> # Create a sample image with distinct RGB channels
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> # Red channel (first channel)
        >>> image[:, :, 0] = np.linspace(0, 255, 100, dtype=np.uint8).reshape(1, 100)
        >>> # Green channel (second channel)
        >>> image[:, :, 1] = np.linspace(0, 255, 100, dtype=np.uint8).reshape(100, 1)
        >>> # Blue channel (third channel) - constant value
        >>> image[:, :, 2] = 128
        >>>
        >>> # Apply channel shuffle transform
        >>> transform = A.ChannelShuffle(p=1.0)
        >>> result = transform(image=image)
        >>> shuffled_image = result['image']
        >>>
        >>> # The channels have been randomly rearranged
        >>> # For example, the original order [R, G, B] might become [G, B, R] or [B, R, G]
        >>> # This results in a color shift while preserving all the original image data
        >>> # Note: For images with more than 3 channels, all channels are shuffled similarly

    """

    def apply(
        self,
        img: np.ndarray,
        channels_shuffled: list[int] | None,
        **params: Any,
    ) -> np.ndarray:
        """Apply the ChannelShuffle transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the ChannelShuffle transform to.
            channels_shuffled (list[int] | None): The channels to shuffle.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied ChannelShuffle transform.

        """
        if channels_shuffled is None:
            return img
        return fpixel.channel_shuffle(img, channels_shuffled)

    def apply_to_images(self, images: np.ndarray, channels_shuffled: list[int] | None, **params: Any) -> np.ndarray:
        """Apply the ChannelShuffle transform to the input images.

        Args:
            images (np.ndarray): The input images to apply the ChannelShuffle transform to.
            channels_shuffled (list[int] | None): The channels to shuffle.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The images with the applied ChannelShuffle transform.

        """
        if channels_shuffled is None:
            return images
        return fpixel.volume_channel_shuffle(images, channels_shuffled)

    def apply_to_volumes(self, volumes: np.ndarray, channels_shuffled: list[int] | None, **params: Any) -> np.ndarray:
        """Apply the ChannelShuffle transform to the input volumes.

        Args:
            volumes (np.ndarray): The input volumes to apply the ChannelShuffle transform to.
            channels_shuffled (list[int] | None): The channels to shuffle.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The volumes with the applied ChannelShuffle transform.

        """
        if channels_shuffled is None:
            return volumes
        return fpixel.volumes_channel_shuffle(volumes, channels_shuffled)

    def apply_to_volume(self, volume: np.ndarray, channels_shuffled: list[int] | None, **params: Any) -> np.ndarray:
        """Apply the ChannelShuffle transform to the input volume.

        Args:
            volume (np.ndarray): The input volume to apply the ChannelShuffle transform to.
            channels_shuffled (list[int] | None): The channels to shuffle.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The volume with the applied ChannelShuffle transform.

        """
        return self.apply_to_images(volume, channels_shuffled, **params)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, Any]: Dictionary with the following key:
                - "channels_shuffled" (tuple[int, ...] | None): The channels to shuffle.

        """
        shape = params["shape"]
        if len(shape) == 2 or shape[-1] == 1:
            return {"channels_shuffled": None}
        ch_arr = list(range(shape[-1]))
        self.py_random.shuffle(ch_arr)
        return {"channels_shuffled": ch_arr}


class InvertImg(ImageOnlyTransform):
    """Invert the input image by subtracting pixel values from max values of the image types,
    i.e., 255 for uint8 and 1.0 for float32.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        Any

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image with different elements
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> cv2.circle(image, (30, 30), 20, (255, 255, 255), -1)  # White circle
        >>> cv2.rectangle(image, (60, 60), (90, 90), (128, 128, 128), -1)  # Gray rectangle
        >>>
        >>> # Apply InvertImg transform
        >>> transform = A.InvertImg(p=1.0)
        >>> result = transform(image=image)
        >>> inverted_image = result['image']
        >>>
        >>> # Result:
        >>> # - Black background becomes white (0 → 255)
        >>> # - White circle becomes black (255 → 0)
        >>> # - Gray rectangle is inverted (128 → 127)
        >>> # The same approach works for float32 images (0-1 range) and grayscale images

    """

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the InvertImg transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the InvertImg transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied InvertImg transform.

        """
        return fpixel.invert(img)

    def apply_to_images(self, images: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply the InvertImg transform to the input images.

        Args:
            images (np.ndarray): The input images to apply the InvertImg transform to.
            *args (Any): Additional arguments (not used in this transform).
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The images with the applied InvertImg transform.

        """
        return self.apply(images, *args, **params)

    def apply_to_volumes(self, volumes: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply the InvertImg transform to the input volumes.

        Args:
            volumes (np.ndarray): The input volumes to apply the InvertImg transform to.
            *args (Any): Additional arguments (not used in this transform).
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The volumes with the applied InvertImg transform.

        """
        return self.apply(volumes, *args, **params)

    def apply_to_volume(self, volume: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        """Apply the InvertImg transform to the input volume.

        Args:
            volume (np.ndarray): The input volume to apply the InvertImg transform to.
            *args (Any): Additional arguments (not used in this transform).
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The volume with the applied InvertImg transform.

        """
        return self.apply(volume, *args, **params)


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
        eps (float): A small value added to the gamma to avoid division by zero or log of zero errors.
            Default: 1e-7.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

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
        gamma_limit: tuple[float, float] | float = (80, 120),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.gamma_limit = cast("tuple[float, float]", gamma_limit)

    def apply(self, img: np.ndarray, gamma: float, **params: Any) -> np.ndarray:
        """Apply the RandomGamma transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the RandomGamma transform to.
            gamma (float): The gamma value.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied RandomGamma transform.

        """
        return fpixel.gamma_transform(img, gamma=gamma)

    def apply_to_volume(self, volume: np.ndarray, gamma: float, **params: Any) -> np.ndarray:
        """Apply the RandomGamma transform to the input volume.

        Args:
            volume (np.ndarray): The input volume to apply the RandomGamma transform to.
            gamma (float): The gamma value.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The volume with the applied RandomGamma transform.

        """
        return self.apply(volume, gamma=gamma)

    def apply_to_volumes(self, volumes: np.ndarray, gamma: float, **params: Any) -> np.ndarray:
        """Apply the RandomGamma transform to the input volumes.

        Args:
            volumes (np.ndarray): The input volumes to apply the RandomGamma transform to.
            gamma (float): The gamma value.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The volumes with the applied RandomGamma transform.

        """
        return self.apply(volumes, gamma=gamma)

    def apply_to_images(self, images: np.ndarray, gamma: float, **params: Any) -> np.ndarray:
        """Apply the RandomGamma transform to the input images.

        Args:
            images (np.ndarray): The input images to apply the RandomGamma transform to.
            gamma (float): The gamma value.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The images with the applied RandomGamma transform.

        """
        return self.apply(images, gamma=gamma)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, Any]: Dictionary with the following key:
                - "gamma" (float): The gamma value.

        """
        return {
            "gamma": self.py_random.uniform(*self.gamma_limit) / 100.0,
        }


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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample color image with distinct RGB values
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> # Red square in top-left
        >>> image[10:40, 10:40, 0] = 200
        >>> # Green square in top-right
        >>> image[10:40, 60:90, 1] = 200
        >>> # Blue square in bottom-left
        >>> image[60:90, 10:40, 2] = 200
        >>> # Yellow square in bottom-right (Red + Green)
        >>> image[60:90, 60:90, 0] = 200
        >>> image[60:90, 60:90, 1] = 200
        >>>
        >>> # Example 1: Default conversion (weighted average, 3 channels)
        >>> transform = A.ToGray(p=1.0)
        >>> result = transform(image=image)
        >>> gray_image = result['image']
        >>> # Output has 3 duplicate channels with values based on RGB perception weights
        >>> # R=0.299, G=0.587, B=0.114
        >>> assert gray_image.shape == (100, 100, 3)
        >>> assert np.allclose(gray_image[:, :, 0], gray_image[:, :, 1])
        >>> assert np.allclose(gray_image[:, :, 1], gray_image[:, :, 2])
        >>>
        >>> # Example 2: Single-channel output
        >>> transform = A.ToGray(num_output_channels=1, p=1.0)
        >>> result = transform(image=image)
        >>> gray_image = result['image']
        >>> assert gray_image.shape == (100, 100, 1)
        >>>
        >>> # Example 3: Using different conversion methods
        >>> # "desaturation" method (min+max)/2
        >>> transform_desaturate = A.ToGray(
        ...     method="desaturation",
        ...     p=1.0
        ... )
        >>> result = transform_desaturate(image=image)
        >>> gray_desaturate = result['image']
        >>>
        >>> # "from_lab" method (using L channel from LAB colorspace)
        >>> transform_lab = A.ToGray(
        ...     method="from_lab",
        ...     p=1.0
        >>> )
        >>> result = transform_lab(image=image)
        >>> gray_lab = result['image']
        >>>
        >>> # "average" method (simple average of channels)
        >>> transform_avg = A.ToGray(
        ...     method="average",
        ...     p=1.0
        >>> )
        >>> result = transform_avg(image=image)
        >>> gray_avg = result['image']
        >>>
        >>> # "max" method (takes max value across channels)
        >>> transform_max = A.ToGray(
        ...     method="max",
        ...     p=1.0
        >>> )
        >>> result = transform_max(image=image)
        >>> gray_max = result['image']
        >>>
        >>> # Example 4: Using grayscale in an augmentation pipeline
        >>> pipeline = A.Compose([
        ...     A.ToGray(p=0.5),           # 50% chance of grayscale conversion
        ...     A.RandomBrightnessContrast(p=1.0)  # Always apply brightness/contrast
        ... ])
        >>> result = pipeline(image=image)
        >>> augmented_image = result['image']  # May be grayscale or color
        >>>
        >>> # Example 5: Converting float32 image
        >>> float_image = image.astype(np.float32) / 255.0  # Range [0, 1]
        >>> transform = A.ToGray(p=1.0)
        >>> result = transform(image=float_image)
        >>> gray_float_image = result['image']
        >>> assert gray_float_image.dtype == np.float32
        >>> assert gray_float_image.max() <= 1.0

    """

    class InitSchema(BaseTransformInitSchema):
        num_output_channels: int = Field(
            description="The number of output channels.",
            ge=1,
        )
        method: Literal[
            "weighted_average",
            "from_lab",
            "desaturation",
            "average",
            "max",
            "pca",
        ]

    def __init__(
        self,
        num_output_channels: int = 3,
        method: Literal[
            "weighted_average",
            "from_lab",
            "desaturation",
            "average",
            "max",
            "pca",
        ] = "weighted_average",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_output_channels = num_output_channels
        self.method = method

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the ToGray transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the ToGray transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied ToGray transform.

        """
        if is_grayscale_image(img):
            warnings.warn("The image is already gray.", stacklevel=2)
            return img

        num_channels = get_num_channels(img)

        if num_channels != NUM_RGB_CHANNELS and self.method not in {
            "desaturation",
            "average",
            "max",
            "pca",
        }:
            msg = "ToGray transformation expects 3-channel images."
            raise TypeError(msg)

        return fpixel.to_gray(img, self.num_output_channels, self.method)


class ToRGB(ImageOnlyTransform):
    """Convert an input image from grayscale to RGB format.

    Args:
        num_output_channels (int): The number of channels in the output image. Default: 3.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, volume

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

    def __init__(
        self,
        num_output_channels: int = 3,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.num_output_channels = num_output_channels

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the ToRGB transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the ToRGB transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied ToRGB transform.

        """
        if is_rgb_image(img):
            warnings.warn("The image is already an RGB.", stacklevel=2)
            return np.ascontiguousarray(img)
        if not is_grayscale_image(img):
            msg = "ToRGB transformation expects 2-dim images or 3-dim with the last dimension equal to 1."
            raise TypeError(msg)

        return fpixel.grayscale_to_multichannel(
            img,
            num_output_channels=self.num_output_channels,
        )


class ToSepia(ImageOnlyTransform):
    """Apply a sepia filter to the input image.

    This transform converts a color image to a sepia tone, giving it a warm, brownish tint
    that is reminiscent of old photographs. The sepia effect is achieved by applying a
    specific color transformation matrix to the RGB channels of the input image.
    For grayscale images, the transform is a no-op and returns the original image.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1,3

    Note:
        - The sepia effect only works with RGB images (3 channels). For grayscale images,
          the original image is returned unchanged since the sepia transformation would
          have no visible effect when R=G=B.
        - The sepia effect is created using a fixed color transformation matrix:
          [[0.393, 0.769, 0.189],
           [0.349, 0.686, 0.168],
           [0.272, 0.534, 0.131]]
        - The output image will have the same data type as the input image.
        - For float32 images, ensure the input values are in the range [0, 1].

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        # Apply sepia effect to a uint8 RGB image
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ToSepia(p=1.0)
        >>> sepia_image = transform(image=image)['image']
        >>> assert sepia_image.shape == image.shape
        >>> assert sepia_image.dtype == np.uint8
        >>>
        # Apply sepia effect to a float32 RGB image
        >>> image = np.random.rand(100, 100, 3).astype(np.float32)
        >>> transform = A.ToSepia(p=1.0)
        >>> sepia_image = transform(image=image)['image']
        >>> assert sepia_image.shape == image.shape
        >>> assert sepia_image.dtype == np.float32
        >>> assert 0 <= sepia_image.min() <= sepia_image.max() <= 1.0
        >>>
        # No effect on grayscale images
        >>> gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> transform = A.ToSepia(p=1.0)
        >>> result = transform(image=gray_image)['image']
        >>> assert np.array_equal(result, gray_image)

    Mathematical Formulation:
        Given an input pixel [R, G, B], the sepia tone is calculated as:
        R_sepia = 0.393*R + 0.769*G + 0.189*B
        G_sepia = 0.349*R + 0.686*G + 0.168*B
        B_sepia = 0.272*R + 0.534*G + 0.131*B

        For grayscale images where R=G=B, this transformation would result in a simple
        scaling of the original value, so we skip it.

        The output values are clipped to the valid range for the image's data type.

    See Also:
        ToGray: For converting images to grayscale instead of sepia.

    """

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)
        self.sepia_transformation_matrix = np.array(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
        )

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the ToSepia transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the ToSepia transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied ToSepia transform.

        """
        if is_grayscale_image(img):
            return img

        if not is_rgb_image(img):
            msg = "ToSepia transformation expects 1 or 3-channel images."
            raise TypeError(msg)
        return fpixel.linear_transformation_rgb(img, self.sepia_transformation_matrix)


class InterpolationPydantic(BaseModel):
    upscale: Literal[
        cv2.INTER_NEAREST,
        cv2.INTER_NEAREST_EXACT,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR_EXACT,
    ]

    downscale: Literal[
        cv2.INTER_NEAREST,
        cv2.INTER_NEAREST_EXACT,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR_EXACT,
    ]


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

        interpolation_pair (dict[Literal["downscale", "upscale"], int]): A dictionary specifying
            the interpolation methods to use for downscaling and upscaling.
            Should contain two keys:
            - 'downscale': Interpolation method for downscaling
            - 'upscale': Interpolation method for upscaling
            Values should be OpenCV interpolation flags (e.g., cv2.INTER_NEAREST, cv2.INTER_LINEAR, etc.)
            Default: {'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_NEAREST}

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image, volume

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

    Examples:
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
        interpolation_pair: dict[
            Literal["downscale", "upscale"],
            Literal[
                cv2.INTER_NEAREST,
                cv2.INTER_NEAREST_EXACT,
                cv2.INTER_LINEAR,
                cv2.INTER_CUBIC,
                cv2.INTER_AREA,
                cv2.INTER_LANCZOS4,
                cv2.INTER_LINEAR_EXACT,
            ],
        ]
        scale_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.25, 0.25),
        interpolation_pair: dict[
            Literal["downscale", "upscale"],
            Literal[
                cv2.INTER_NEAREST,
                cv2.INTER_NEAREST_EXACT,
                cv2.INTER_LINEAR,
                cv2.INTER_CUBIC,
                cv2.INTER_AREA,
                cv2.INTER_LANCZOS4,
                cv2.INTER_LINEAR_EXACT,
            ],
        ] = {"upscale": cv2.INTER_NEAREST, "downscale": cv2.INTER_NEAREST},
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.scale_range = scale_range
        self.interpolation_pair = interpolation_pair

    def apply(self, img: np.ndarray, scale: float, **params: Any) -> np.ndarray:
        """Apply the Downscale transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the Downscale transform to.
            scale (float): The downscaling factor.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied Downscale transform.

        """
        return fpixel.downscale(
            img,
            scale=scale,
            down_interpolation=self.interpolation_pair["downscale"],
            up_interpolation=self.interpolation_pair["upscale"],
        )

    def get_params(self) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Returns:
            dict[str, Any]: Dictionary with the following key:
                - "scale" (float): The downscaling factor.

        """
        return {"scale": self.py_random.uniform(*self.scale_range)}


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
        image, volume

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1.0)
        >>> result = transform(image=image)
        >>> noisy_image = result["image"]

    References:
        Multiplicative noise: https://en.wikipedia.org/wiki/Multiplicative_noise

    """

    class InitSchema(BaseTransformInitSchema):
        multiplier: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        per_channel: bool
        elementwise: bool

    def __init__(
        self,
        multiplier: tuple[float, float] | float = (0.9, 1.1),
        per_channel: bool = False,
        elementwise: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.multiplier = cast("tuple[float, float]", multiplier)
        self.elementwise = elementwise
        self.per_channel = per_channel

    def apply(
        self,
        img: np.ndarray,
        multiplier: float | np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply the MultiplicativeNoise transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the MultiplicativeNoise transform to.
            multiplier (float | np.ndarray): The random multiplier.
            **kwargs (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied MultiplicativeNoise transform.

        """
        return multiply(img, multiplier)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data to apply the transform to.

        Returns:
            dict[str, Any]: The parameters of the transform.

        """
        image = data["image"] if "image" in data else data["images"][0]

        num_channels = get_num_channels(image)

        if self.elementwise:
            shape = image.shape if self.per_channel else (*image.shape[:2], 1)
        else:
            shape = (num_channels,) if self.per_channel else (1,)

        multiplier = self.random_generator.uniform(
            self.multiplier[0],
            self.multiplier[1],
            shape,
        ).astype(np.float32)

        if not self.per_channel and num_channels > 1:
            # Replicate the multiplier for all channels if not per_channel
            multiplier = np.repeat(multiplier, num_channels, axis=-1)

        if not self.elementwise and self.per_channel:
            # Reshape to broadcast correctly when not elementwise but per_channel
            multiplier = multiplier.reshape(1, 1, -1)

        if multiplier.shape != image.shape:
            multiplier = multiplier.squeeze()

        return {"multiplier": multiplier}


class FancyPCA(ImageOnlyTransform):
    """Apply Fancy PCA augmentation to the input image.

    This augmentation technique applies PCA (Principal Component Analysis) to the image's color channels,
    then adds multiples of the principal components to the image, with magnitudes proportional to the
    corresponding eigenvalues times a random variable drawn from a Gaussian with mean 0 and standard
    deviation 'alpha'.

    Args:
        alpha (float): Standard deviation of the Gaussian distribution used to generate
            random noise for each principal component. Default: 0.1.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image, volume

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.FancyPCA(alpha=0.1, p=1.0)
        >>> result = transform(image=image)
        >>> augmented_image = result["image"]

    References:
        ImageNet Classification with Deep Convolutional Neural Networks: In Advances in Neural Information
        Processing Systems (Vol. 25). Curran Associates, Inc.

    """

    class InitSchema(BaseTransformInitSchema):
        alpha: float = Field(ge=0)

    def __init__(
        self,
        alpha: float = 0.1,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.alpha = alpha

    def apply(
        self,
        img: np.ndarray,
        alpha_vector: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the FancyPCA transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the FancyPCA transform to.
            alpha_vector (np.ndarray): The random noise for each principal component.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied FancyPCA transform.

        """
        return fpixel.fancy_pca(img, alpha_vector)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters dependent on the input data.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data to apply the transform to.

        Returns:
            dict[str, Any]: The parameters of the transform.

        """
        shape = params["shape"]
        num_channels = shape[-1] if len(shape) == NUM_MULTI_CHANNEL_DIMENSIONS else 1
        alpha_vector = self.random_generator.normal(0, self.alpha, num_channels).astype(
            np.float32,
        )
        return {"alpha_vector": alpha_vector}


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
        image, volume

    Image types:
        uint8, float32

    Number of channels:
        1, 3

    Note:
        - The order of application for these color transformations is random for each image.
        - The ranges for brightness, contrast, and saturation are applied as multiplicative factors.
        - The range for hue is applied as an additive factor.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
        >>> result = transform(image=image)
        >>> jittered_image = result['image']

    References:
        - ColorJitter: https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html
        - Color Conversions: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

    """

    class InitSchema(BaseTransformInitSchema):
        brightness: tuple[float, float] | float
        contrast: tuple[float, float] | float
        saturation: tuple[float, float] | float
        hue: tuple[float, float] | float

        @field_validator("brightness", "contrast", "saturation", "hue")
        @classmethod
        def _check_ranges(
            cls,
            value: tuple[float, float] | float,
            info: ValidationInfo,
        ) -> tuple[float, float]:
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
                    raise ValueError(
                        f"If {info.field_name} is a single number, it must be non negative.",
                    )
                left = bias - value
                if clip:
                    left = max(left, 0)
                value = (left, bias + value)
            elif isinstance(value, tuple) and len(value) == PAIR:
                check_range(value, *bounds, info.field_name)

            return cast("tuple[float, float]", value)

    def __init__(
        self,
        brightness: tuple[float, float] | float = (0.8, 1.2),
        contrast: tuple[float, float] | float = (0.8, 1.2),
        saturation: tuple[float, float] | float = (0.8, 1.2),
        hue: tuple[float, float] | float = (-0.5, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.brightness = cast("tuple[float, float]", brightness)
        self.contrast = cast("tuple[float, float]", contrast)
        self.saturation = cast("tuple[float, float]", saturation)
        self.hue = cast("tuple[float, float]", hue)

        self.transforms = [
            fpixel.adjust_brightness_torchvision,
            fpixel.adjust_contrast_torchvision,
            fpixel.adjust_saturation_torchvision,
            fpixel.adjust_hue_torchvision,
        ]

    def get_params(self) -> dict[str, Any]:
        """Generate parameters for the ColorJitter transform.

        Returns:
            dict[str, Any]: The parameters of the transform.

        """
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
        """Apply the ColorJitter transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the ColorJitter transform to.
            brightness (float): The brightness factor.
            contrast (float): The contrast factor.
            saturation (float): The saturation factor.
            hue (float): The hue factor.
            order (list[int]): The order of application for the color transformations.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied ColorJitter transform.

        """
        if not is_rgb_image(img) and not is_grayscale_image(img):
            msg = "ColorJitter transformation expects 1-channel or 3-channel images."
            raise TypeError(msg)

        color_transforms = [brightness, contrast, saturation, hue]
        for i in order:
            img = self.transforms[i](img, color_transforms[i])
        return img


class Sharpen(ImageOnlyTransform):
    """Sharpen the input image using either kernel-based or Gaussian interpolation method.

    Implements two different approaches to image sharpening:
    1. Traditional kernel-based method using Laplacian operator
    2. Gaussian interpolation method (similar to Kornia's approach)

    Args:
        alpha (tuple[float, float]): Range for the visibility of sharpening effect.
            At 0, only the original image is visible, at 1.0 only its processed version is visible.
            Values should be in the range [0, 1].
            Used in both methods. Default: (0.2, 0.5).

        lightness (tuple[float, float]): Range for the lightness of the sharpened image.
            Only used in 'kernel' method. Larger values create higher contrast.
            Values should be greater than 0. Default: (0.5, 1.0).

        method (Literal['kernel', 'gaussian']): Sharpening algorithm to use:
            - 'kernel': Traditional kernel-based sharpening using Laplacian operator
            - 'gaussian': Interpolation between Gaussian blurred and original image
            Default: 'kernel'

        kernel_size (int): Size of the Gaussian blur kernel for 'gaussian' method.
            Must be odd. Default: 5

        sigma (float): Standard deviation for Gaussian kernel in 'gaussian' method.
            Default: 1.0

        p (float): Probability of applying the transform. Default: 0.5.

    Image types:
        uint8, float32

    Number of channels:
        Any

    Mathematical Formulation:
        1. Kernel Method:
           The sharpening operation is based on the Laplacian operator L:
           L = [[-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]]

           The final kernel K is a weighted sum:
           K = (1 - a)I + a(L + λI)

           where:
           - a is the alpha value
           - λ is the lightness value
           - I is the identity kernel

           The output image O is computed as:
           O = K * I  (convolution)

        2. Gaussian Method:
           Based on the unsharp mask principle:
           O = aI + (1-a)G

           where:
           - I is the input image
           - G is the Gaussian blurred version of I
           - a is the alpha value (sharpness)

           The Gaussian kernel G(x,y) is defined as:
           G(x,y) = (1/(2πs²))exp(-(x²+y²)/(2s²))

    Note:
        - Kernel sizes must be odd to maintain spatial alignment
        - Methods produce different visual results:
          * Kernel method: More pronounced edges, possible artifacts
          * Gaussian method: More natural look, limited to original sharpness

    Examples:
        >>> import albumentations as A
        >>> import numpy as np

        # Traditional kernel sharpening
        >>> transform = A.Sharpen(
        ...     alpha=(0.2, 0.5),
        ...     lightness=(0.5, 1.0),
        ...     method='kernel',
        ...     p=1.0
        ... )

        # Gaussian interpolation sharpening
        >>> transform = A.Sharpen(
        ...     alpha=(0.5, 1.0),
        ...     method='gaussian',
        ...     kernel_size=5,
        ...     sigma=1.0,
        ...     p=1.0
        ... )

    References:
        - R. C. Gonzalez and R. E. Woods, "Digital Image Processing (4th Edition),": Chapter 3:
            Intensity Transformations and Spatial Filtering.
        - J. C. Russ, "The Image Processing Handbook (7th Edition),": Chapter 4: Image Enhancement.
        - T. Acharya and A. K. Ray, "Image Processing: Principles and Applications,": Chapter 5: Image Enhancement.
        - Unsharp masking: https://en.wikipedia.org/wiki/Unsharp_masking
        - Laplacian operator: https://en.wikipedia.org/wiki/Laplace_operator
        - Gaussian blur: https://en.wikipedia.org/wiki/Gaussian_blur

    See Also:
        - Blur: For Gaussian blurring
        - UnsharpMask: Alternative sharpening method
        - RandomBrightnessContrast: For adjusting image contrast

    """

    class InitSchema(BaseTransformInitSchema):
        alpha: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        lightness: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, None))]
        method: Literal["kernel", "gaussian"]
        kernel_size: int = Field(ge=3)
        sigma: float = Field(gt=0)

    @field_validator("kernel_size")
    @classmethod
    def _check_kernel_size(cls, value: int) -> int:
        return value + 1 if value % 2 == 0 else value

    def __init__(
        self,
        alpha: tuple[float, float] = (0.2, 0.5),
        lightness: tuple[float, float] = (0.5, 1.0),
        method: Literal["kernel", "gaussian"] = "kernel",
        kernel_size: int = 5,
        sigma: float = 1.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.alpha = alpha
        self.lightness = lightness
        self.method = method
        self.kernel_size = kernel_size
        self.sigma = sigma

    @staticmethod
    def __generate_sharpening_matrix(
        alpha: np.ndarray,
        lightness: np.ndarray,
    ) -> np.ndarray:
        matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        matrix_effect = np.array(
            [[-1, -1, -1], [-1, 8 + lightness, -1], [-1, -1, -1]],
            dtype=np.float32,
        )

        return (1 - alpha) * matrix_nochange + alpha * matrix_effect

    def get_params(self) -> dict[str, Any]:
        """Generate parameters for the Sharpen transform.

        Returns:
            dict[str, Any]: The parameters of the transform.

        """
        alpha = self.py_random.uniform(*self.alpha)

        if self.method == "kernel":
            lightness = self.py_random.uniform(*self.lightness)
            return {
                "alpha": alpha,
                "sharpening_matrix": self.__generate_sharpening_matrix(
                    alpha,
                    lightness,
                ),
            }

        return {"alpha": alpha, "sharpening_matrix": None}

    def apply(
        self,
        img: np.ndarray,
        alpha: float,
        sharpening_matrix: np.ndarray | None,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Sharpen transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the Sharpen transform to.
            alpha (float): The alpha value.
            sharpening_matrix (np.ndarray | None): The sharpening matrix.
            **params (Any): Additional parameters for the transform.

        """
        if self.method == "kernel":
            return fpixel.convolve(img, sharpening_matrix)
        return fpixel.sharpen_gaussian(img, alpha, self.kernel_size, self.sigma)


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
        image, volume

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5)
        >>> result = transform(image=image)
        >>> embossed_image = result['image']

    References:
        - Image Embossing: https://en.wikipedia.org/wiki/Image_embossing
        - Application of Emboss Filtering in Image Processing: https://www.researchgate.net/publication/303412455_Application_of_Emboss_Filtering_in_Image_Processing

    """

    class InitSchema(BaseTransformInitSchema):
        alpha: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        strength: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, None))]

    def __init__(
        self,
        alpha: tuple[float, float] = (0.2, 0.5),
        strength: tuple[float, float] = (0.2, 0.7),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.alpha = alpha
        self.strength = strength

    @staticmethod
    def __generate_emboss_matrix(
        alpha_sample: np.ndarray,
        strength_sample: np.ndarray,
    ) -> np.ndarray:
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
        """Generate parameters for the Emboss transform.

        Returns:
            dict[str, np.ndarray]: The parameters of the transform.

        """
        alpha = self.py_random.uniform(*self.alpha)
        strength = self.py_random.uniform(*self.strength)
        emboss_matrix = self.__generate_emboss_matrix(
            alpha_sample=alpha,
            strength_sample=strength,
        )
        return {"emboss_matrix": emboss_matrix}

    def apply(
        self,
        img: np.ndarray,
        emboss_matrix: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the Emboss transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the Emboss transform to.
            emboss_matrix (np.ndarray): The emboss matrix.
            **params (Any): Additional parameters for the transform.

        """
        return fpixel.convolve(img, emboss_matrix)


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
        image, volume

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
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]

    def __init__(
        self,
        p_replace: tuple[float, float] | float = (0, 0.1),
        n_segments: tuple[int, int] | int = (100, 100),
        max_size: int | None = 128,
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.p_replace = cast("tuple[float, float]", p_replace)
        self.n_segments = cast("tuple[int, int]", n_segments)
        self.max_size = max_size
        self.interpolation = interpolation

    def get_params(self) -> dict[str, Any]:
        """Generate parameters for the Superpixels transform.

        Returns:
            dict[str, Any]: The parameters of the transform.

        """
        n_segments = self.py_random.randint(*self.n_segments)
        p = self.py_random.uniform(*self.p_replace)
        return {
            "replace_samples": self.random_generator.random(n_segments) < p,
            "n_segments": n_segments,
        }

    def apply(
        self,
        img: np.ndarray,
        replace_samples: Sequence[bool],
        n_segments: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Apply the Superpixels transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the Superpixels transform to.
            replace_samples (Sequence[bool]): Whether to replace pixels in segments.
            n_segments (int): Number of superpixels.
            **kwargs (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied Superpixels transform.

        """
        return fpixel.superpixels(
            img,
            n_segments,
            replace_samples,
            self.max_size,
            self.interpolation,
        )


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
        image, volume

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
        - Digital Image Processing: Rafael C. Gonzalez and Richard E. Woods, 4th Edition

    """

    class InitSchema(BlurInitSchema):
        blur_limit: tuple[int, int] | int
        cutoff: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, np.pi)),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        blur_limit: tuple[int, int] | int = (7, 15),
        cutoff: tuple[float, float] = (np.pi / 4, np.pi / 2),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.blur_limit = cast("tuple[int, int]", blur_limit)
        self.cutoff = cutoff

    def get_params(self) -> dict[str, np.ndarray]:
        """Generate parameters for the RingingOvershoot transform.

        Returns:
            dict[str, np.ndarray]: The parameters of the transform.

        """
        ksize = self.py_random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        if ksize % 2 == 0:
            ksize += 1

        cutoff = self.py_random.uniform(*self.cutoff)

        # From dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = np.fromfunction(
                lambda x, y: cutoff
                * special.j1(
                    cutoff * np.sqrt((x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2),
                )
                / (2 * np.pi * np.sqrt((x - (ksize - 1) / 2) ** 2 + (y - (ksize - 1) / 2) ** 2)),
                [ksize, ksize],
            )
        kernel[(ksize - 1) // 2, (ksize - 1) // 2] = cutoff**2 / (4 * np.pi)

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)

        return {"kernel": kernel}

    def apply(self, img: np.ndarray, kernel: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the RingingOvershoot transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the RingingOvershoot transform to.
            kernel (np.ndarray): The kernel for the convolution.
            **params (Any): Additional parameters (not used in this transform).

        """
        return fpixel.convolve(img, kernel)


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
        sigma_limit (tuple[float, float] | float): Gaussian kernel standard deviation. Must be more or equal to 0.
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
        image, volume

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
        Unsharp Masking: https://en.wikipedia.org/wiki/Unsharp_masking

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
        blur_limit: tuple[int, int] | int

        @field_validator("blur_limit")
        @classmethod
        def _process_blur(
            cls,
            value: tuple[int, int] | int,
            info: ValidationInfo,
        ) -> tuple[int, int]:
            return fblur.process_blur_limit(value, info, min_value=3)

    def __init__(
        self,
        blur_limit: tuple[int, int] | int = (3, 7),
        sigma_limit: tuple[float, float] | float = 0.0,
        alpha: tuple[float, float] | float = (0.2, 0.5),
        threshold: int = 10,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.blur_limit = cast("tuple[int, int]", blur_limit)
        self.sigma_limit = cast("tuple[float, float]", sigma_limit)
        self.alpha = cast("tuple[float, float]", alpha)
        self.threshold = threshold

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters for the UnsharpMask transform.

        Returns:
            dict[str, Any]: The parameters of the transform.

        """
        return {
            "ksize": self.py_random.randrange(
                self.blur_limit[0],
                self.blur_limit[1] + 1,
                2,
            ),
            "sigma": self.py_random.uniform(*self.sigma_limit),
            "alpha": self.py_random.uniform(*self.alpha),
        }

    def apply(
        self,
        img: np.ndarray,
        ksize: int,
        sigma: int,
        alpha: float,
        **params: Any,
    ) -> np.ndarray:
        """Apply the UnsharpMask transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the UnsharpMask transform to.
            ksize (int): The kernel size for the convolution.
            sigma (int): The standard deviation for the Gaussian blur.
            alpha (float): The visibility of the sharpened image.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied UnsharpMask transform.

        """
        return fpixel.unsharp_mask(
            img,
            ksize,
            sigma=sigma,
            alpha=alpha,
            threshold=self.threshold,
        )


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
        cutout_threshold (tuple[float, float] | floats): Threshold for filtering liquid layer
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
        mode (Literal["rain", "mud"]): Type of corruption. Default: "rain".
        color (tuple[int, ...] | None): Corruption elements color.
            If list uses provided list as color for the effect.
            If None uses default colors based on mode (rain: (238, 238, 175), mud: (20, 42, 63)).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    References:
        Benchmarking Neural Network Robustness to Common Corruptions and Perturbations: https://arxiv.org/abs/1903.12261

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image
        >>> image = np.ones((300, 300, 3), dtype=np.uint8) * 200  # Light gray background
        >>> # Add some gradient to make effects more visible
        >>> for i in range(300):
        ...     image[i, :, :] = np.clip(image[i, :, :] - i // 3, 0, 255)
        >>>
        >>> # Example 1: Rain effect with default parameters
        >>> rain_transform = A.Spatter(
        ...     mode="rain",
        ...     p=1.0
        ... )
        >>> rain_result = rain_transform(image=image)
        >>> rain_image = rain_result['image']  # Image with rain drops
        >>>
        >>> # Example 2: Heavy rain with custom parameters
        >>> heavy_rain = A.Spatter(
        ...     mode="rain",
        ...     mean=(0.7, 0.7),             # Higher mean = more coverage
        ...     std=(0.2, 0.2),              # Lower std = more uniform effect
        ...     cutout_threshold=(0.65, 0.65),  # Lower threshold = more drops
        ...     intensity=(0.8, 0.8),        # Higher intensity = more visible effect
        ...     color=(200, 200, 255),       # Blueish rain drops
        ...     p=1.0
        ... )
        >>> heavy_rain_result = heavy_rain(image=image)
        >>> heavy_rain_image = heavy_rain_result['image']
        >>>
        >>> # Example 3: Mud effect
        >>> mud_transform = A.Spatter(
        ...     mode="mud",
        ...     mean=(0.6, 0.6),
        ...     std=(0.3, 0.3),
        ...     cutout_threshold=(0.62, 0.62),
        ...     intensity=(0.7, 0.7),
        ...     p=1.0
        ... )
        >>> mud_result = mud_transform(image=image)
        >>> mud_image = mud_result['image']  # Image with mud splatters
        >>>
        >>> # Example 4: Custom colored mud
        >>> red_mud = A.Spatter(
        ...     mode="mud",
        ...     mean=(0.55, 0.55),
        ...     std=(0.25, 0.25),
        ...     cutout_threshold=(0.7, 0.7),
        ...     intensity=(0.6, 0.6),
        ...     color=(120, 40, 40),  # Reddish-brown mud
        ...     p=1.0
        ... )
        >>> red_mud_result = red_mud(image=image)
        >>> red_mud_image = red_mud_result['image']
        >>>
        >>> # Example 5: Random effect (50% chance of applying)
        >>> random_spatter = A.Compose([
        ...     A.Spatter(
        ...         mode="rain" if np.random.random() < 0.5 else "mud",
        ...         p=0.5
        ...     )
        ... ])
        >>> random_result = random_spatter(image=image)
        >>> result_image = random_result['image']  # May or may not have spatter effect

    """

    class InitSchema(BaseTransformInitSchema):
        mean: ZeroOneRangeType
        std: ZeroOneRangeType
        gauss_sigma: NonNegativeFloatRangeType
        cutout_threshold: ZeroOneRangeType
        intensity: ZeroOneRangeType
        mode: Literal["rain", "mud"]
        color: Sequence[int] | None

        @model_validator(mode="after")
        def _check_color(self) -> Self:
            # Default colors for each mode
            default_colors = {"rain": [238, 238, 175], "mud": [20, 42, 63]}

            if self.color is None:
                # Use default color for the selected mode
                self.color = default_colors[self.mode]
            # Validate the provided color
            elif len(self.color) != NUM_RGB_CHANNELS:
                msg = "Color must be a list of three integers for RGB format."
                raise ValueError(msg)
            return self

    def __init__(
        self,
        mean: tuple[float, float] | float = (0.65, 0.65),
        std: tuple[float, float] | float = (0.3, 0.3),
        gauss_sigma: tuple[float, float] | float = (2, 2),
        cutout_threshold: tuple[float, float] | float = (0.68, 0.68),
        intensity: tuple[float, float] | float = (0.6, 0.6),
        mode: Literal["rain", "mud"] = "rain",
        color: tuple[int, ...] | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.mean = cast("tuple[float, float]", mean)
        self.std = cast("tuple[float, float]", std)
        self.gauss_sigma = cast("tuple[float, float]", gauss_sigma)
        self.cutout_threshold = cast("tuple[float, float]", cutout_threshold)
        self.intensity = cast("tuple[float, float]", intensity)
        self.mode = mode
        self.color = cast("tuple[int, ...]", color)

    def apply(
        self,
        img: np.ndarray,
        **params: dict[str, Any],
    ) -> np.ndarray:
        """Apply the Spatter transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the Spatter transform to.
            **params (dict[str, Any]): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied Spatter transform.

        """
        non_rgb_error(img)

        if params["mode"] == "rain":
            return fpixel.spatter_rain(img, params["drops"])

        return fpixel.spatter_mud(img, params["non_mud"], params["mud"])

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters for the Spatter transform.

        Returns:
            dict[str, Any]: The parameters of the transform.

        """
        height, width = params["shape"][:2]

        mean = self.py_random.uniform(*self.mean)
        std = self.py_random.uniform(*self.std)
        cutout_threshold = self.py_random.uniform(*self.cutout_threshold)
        sigma = self.py_random.uniform(*self.gauss_sigma)
        mode = self.mode
        intensity = self.py_random.uniform(*self.intensity)
        color = np.array(self.color) / 255.0

        liquid_layer = self.random_generator.normal(
            size=(height, width),
            loc=mean,
            scale=std,
        )
        # Convert sigma to kernel size (must be odd)
        ksize = int(2 * round(3 * sigma) + 1)  # 3 sigma rule, rounded to nearest odd
        cv2.GaussianBlur(
            src=liquid_layer,
            dst=liquid_layer,  # in-place operation
            ksize=(ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE,
        )

        # Important line, without it the rain effect looses drops
        liquid_layer[liquid_layer < cutout_threshold] = 0

        if mode == "rain":
            return {
                "mode": "rain",
                **fpixel.get_rain_params(liquid_layer=liquid_layer, color=color, intensity=intensity),
            }

        return {
            "mode": "mud",
            **fpixel.get_mud_params(
                liquid_layer=liquid_layer,
                color=color,
                cutout_threshold=cutout_threshold,
                sigma=sigma,
                intensity=intensity,
                random_generator=self.random_generator,
            ),
        }


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

    Examples:
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
        Chromatic Aberration: https://en.wikipedia.org/wiki/Chromatic_aberration

    """

    class InitSchema(BaseTransformInitSchema):
        primary_distortion_limit: SymmetricRangeType
        secondary_distortion_limit: SymmetricRangeType
        mode: Literal["green_purple", "red_blue", "random"]
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]

    def __init__(
        self,
        primary_distortion_limit: tuple[float, float] | float = (-0.02, 0.02),
        secondary_distortion_limit: tuple[float, float] | float = (-0.05, 0.05),
        mode: Literal["green_purple", "red_blue", "random"] = "green_purple",
        interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.primary_distortion_limit = cast(
            "tuple[float, float]",
            primary_distortion_limit,
        )
        self.secondary_distortion_limit = cast(
            "tuple[float, float]",
            secondary_distortion_limit,
        )
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
        """Apply the ChromaticAberration transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the ChromaticAberration transform to.
            primary_distortion_red (float): The primary distortion coefficient for the red channel.
            secondary_distortion_red (float): The secondary distortion coefficient for the red channel.
            primary_distortion_blue (float): The primary distortion coefficient for the blue channel.
            secondary_distortion_blue (float): The secondary distortion coefficient for the blue channel.
            **params (dict[str, Any]): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied ChromaticAberration transform.

        """
        non_rgb_error(img)
        return fpixel.chromatic_aberration(
            img,
            primary_distortion_red,
            secondary_distortion_red,
            primary_distortion_blue,
            secondary_distortion_blue,
            self.interpolation,
        )

    def get_params(self) -> dict[str, float]:
        """Generate parameters for the ChromaticAberration transform.

        Returns:
            dict[str, float]: The parameters of the transform.

        """
        primary_distortion_red = self.py_random.uniform(*self.primary_distortion_limit)
        secondary_distortion_red = self.py_random.uniform(
            *self.secondary_distortion_limit,
        )
        primary_distortion_blue = self.py_random.uniform(*self.primary_distortion_limit)
        secondary_distortion_blue = self.py_random.uniform(
            *self.secondary_distortion_limit,
        )

        secondary_distortion_red = self._match_sign(
            primary_distortion_red,
            secondary_distortion_red,
        )
        secondary_distortion_blue = self._match_sign(
            primary_distortion_blue,
            secondary_distortion_blue,
        )

        if self.mode == "green_purple":
            # distortion coefficients of the red and blue channels have the same sign
            primary_distortion_blue = self._match_sign(
                primary_distortion_red,
                primary_distortion_blue,
            )
            secondary_distortion_blue = self._match_sign(
                secondary_distortion_red,
                secondary_distortion_blue,
            )
        if self.mode == "red_blue":
            # distortion coefficients of the red and blue channels have the opposite sign
            primary_distortion_blue = self._unmatch_sign(
                primary_distortion_red,
                primary_distortion_blue,
            )
            secondary_distortion_blue = self._unmatch_sign(
                secondary_distortion_red,
                secondary_distortion_blue,
            )

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


PLANKIAN_JITTER_CONST = {
    "MAX_TEMP": max(
        *fpixel.PLANCKIAN_COEFFS["blackbody"].keys(),
        *fpixel.PLANCKIAN_COEFFS["cied"].keys(),
    ),
    "MIN_BLACKBODY_TEMP": min(fpixel.PLANCKIAN_COEFFS["blackbody"].keys()),
    "MIN_CIED_TEMP": min(fpixel.PLANCKIAN_COEFFS["cied"].keys()),
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
        3

    Note:
        - The transform preserves the overall brightness of the image while shifting its color.
        - The "blackbody" mode provides a wider range of color shifts, especially in the lower (warmer) temperatures.
        - The "cied" mode is based on standard illuminants and may provide more realistic daylight variations.
        - The Gaussian sampling method tends to produce more subtle variations, as it's centered around daylight.
        - Unlike ColorJitter, this transform ensures that color changes are physically plausible and correlated
          across channels, maintaining the natural appearance of the scene under different lighting conditions.

    Examples:
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
        def _validate_temperature(self) -> Self:
            max_temp = int(PLANKIAN_JITTER_CONST["MAX_TEMP"])

            if self.temperature_limit is None:
                if self.mode == "blackbody":
                    self.temperature_limit = (
                        int(PLANKIAN_JITTER_CONST["MIN_BLACKBODY_TEMP"]),
                        max_temp,
                    )
                elif self.mode == "cied":
                    self.temperature_limit = (
                        int(PLANKIAN_JITTER_CONST["MIN_CIED_TEMP"]),
                        max_temp,
                    )
            else:
                if self.mode == "blackbody" and (
                    min(self.temperature_limit) < PLANKIAN_JITTER_CONST["MIN_BLACKBODY_TEMP"]
                    or max(self.temperature_limit) > max_temp
                ):
                    raise ValueError(
                        "Temperature limits for blackbody should be in [3000, 15000] range",
                    )
                if self.mode == "cied" and (
                    min(self.temperature_limit) < PLANKIAN_JITTER_CONST["MIN_CIED_TEMP"]
                    or max(self.temperature_limit) > max_temp
                ):
                    raise ValueError(
                        "Temperature limits for CIED should be in [4000, 15000] range",
                    )

                if not self.temperature_limit[0] <= PLANKIAN_JITTER_CONST["WHITE_TEMP"] <= self.temperature_limit[1]:
                    raise ValueError(
                        "White temperature should be within the temperature limits",
                    )

            return self

    def __init__(
        self,
        mode: Literal["blackbody", "cied"] = "blackbody",
        temperature_limit: tuple[int, int] | None = None,
        sampling_method: Literal["uniform", "gaussian"] = "uniform",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)

        self.mode = mode
        self.temperature_limit = cast("tuple[int, int]", temperature_limit)
        self.sampling_method = sampling_method

    def apply(self, img: np.ndarray, temperature: int, **params: Any) -> np.ndarray:
        """Apply the PlanckianJitter transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the PlanckianJitter transform to.
            temperature (int): The temperature to apply to the image.
            **params (Any): Additional parameters for the transform.

        """
        non_rgb_error(img)
        return fpixel.planckian_jitter(img, temperature, mode=self.mode)

    def get_params(self) -> dict[str, Any]:
        """Generate parameters for the PlanckianJitter transform.

        Returns:
            dict[str, Any]: The parameters of the transform.

        """
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
        temperature = np.clip(
            temperature,
            self.temperature_limit[0],
            self.temperature_limit[1],
        )

        return {"temperature": int(temperature)}


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

    Examples:
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
        scale_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, None)),
        ]

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.1, 0.3),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.scale_range = scale_range

    def apply(
        self,
        img: np.ndarray,
        scale: float,
        random_seed: int,
        **params: Any,
    ) -> np.ndarray:
        """Apply the ShotNoise transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the ShotNoise transform to.
            scale (float): The scale factor for the noise.
            random_seed (int): The random seed for the noise.
            **params (Any): Additional parameters for the transform.

        """
        return fpixel.shot_noise(img, scale, np.random.default_rng(random_seed))

    def get_params(self) -> dict[str, Any]:
        """Generate parameters for the ShotNoise transform.

        Returns:
            dict[str, Any]: The parameters of the transform.

        """
        return {
            "scale": self.py_random.uniform(*self.scale_range),
            "random_seed": self.random_generator.integers(0, 2**32 - 1),
        }


class NoiseParamsBase(BaseModel):
    """Base class for all noise parameter models."""

    model_config = ConfigDict(extra="forbid")
    noise_type: str


class UniformParams(NoiseParamsBase):
    noise_type: Literal["uniform"] = "uniform"
    ranges: list[Sequence[float]] = Field(min_length=1)

    @field_validator("ranges", mode="after")
    @classmethod
    def validate_ranges(cls, v: list[Sequence[float]]) -> list[tuple[float, float]]:
        result = []
        for range_values in v:
            if len(range_values) != PAIR:
                raise ValueError("Each range must have exactly 2 values")
            min_val, max_val = range_values
            if not (-1 <= min_val <= max_val <= 1):
                raise ValueError("Range values must be in [-1, 1] and min <= max")
            result.append((float(min_val), float(max_val)))
        return result


class GaussianParams(NoiseParamsBase):
    noise_type: Literal["gaussian"] = "gaussian"
    mean_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=-1, max_val=1)),
    ]
    std_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=0, max_val=1)),
    ]


class LaplaceParams(NoiseParamsBase):
    noise_type: Literal["laplace"] = "laplace"
    mean_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=-1, max_val=1)),
    ]
    scale_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=0, max_val=1)),
    ]


class BetaParams(NoiseParamsBase):
    noise_type: Literal["beta"] = "beta"
    alpha_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=0)),
    ]
    beta_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=0)),
    ]
    scale_range: Annotated[
        Sequence[float],
        AfterValidator(check_range_bounds(min_val=0, max_val=1)),
    ]


NoiseParams = Annotated[
    Union[UniformParams, GaussianParams, LaplaceParams, BetaParams],
    Field(discriminator="noise_type"),
]


class AdditiveNoise(ImageOnlyTransform):
    """Apply random noise to image channels using various noise distributions.

    This transform generates noise using different probability distributions and applies it
    to image channels. The noise can be generated in three spatial modes and supports
    multiple noise distributions, each with configurable parameters.

    Args:
        noise_type(Literal["uniform", "gaussian", "laplace", "beta"]): Type of noise distribution to use. Options:
            - "uniform": Uniform distribution, good for simple random perturbations
            - "gaussian": Normal distribution, models natural random processes
            - "laplace": Similar to Gaussian but with heavier tails, good for outliers
            - "beta": Flexible bounded distribution, can be symmetric or skewed

        spatial_mode(Literal["constant", "per_pixel", "shared"]): How to generate and apply the noise. Options:
            - "constant": One noise value per channel, fastest
            - "per_pixel": Independent noise value for each pixel and channel, slowest
            - "shared": One noise map shared across all channels, medium speed

        approximation(float): float in [0, 1], default=1.0
            Controls noise generation speed vs quality tradeoff.
            - 1.0: Generate full resolution noise (slowest, highest quality)
            - 0.5: Generate noise at half resolution and upsample
            - 0.25: Generate noise at quarter resolution and upsample
            Only affects 'per_pixel' and 'shared' spatial modes.

        noise_params(dict[str, Any] | None): Parameters for the chosen noise distribution.
            Must match the noise_type:

            uniform:
                ranges: list[tuple[float, float]]
                    List of (min, max) ranges for each channel.
                    Each range must be in [-1, 1].
                    If only one range is provided, it will be used for all channels.

                    [(-0.2, 0.2)]  # Same range for all channels
                    [(-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)]  # Different ranges for RGB

            gaussian:
                mean_range: tuple[float, float], default (0.0, 0.0)
                    Range for sampling mean value, in [-1, 1]
                std_range: tuple[float, float], default (0.1, 0.1)
                    Range for sampling standard deviation, in [0, 1]

            laplace:
                mean_range: tuple[float, float], default (0.0, 0.0)
                    Range for sampling location parameter, in [-1, 1]
                scale_range: tuple[float, float], default (0.1, 0.1)
                    Range for sampling scale parameter, in [0, 1]

            beta:
                alpha_range: tuple[float, float], default (0.5, 1.5)
                    Value < 1 = U-shaped, Value > 1 = Bell-shaped
                    Range for sampling first shape parameter, in (0, inf)
                beta_range: tuple[float, float], default (0.5, 1.5)
                    Value < 1 = U-shaped, Value > 1 = Bell-shaped
                    Range for sampling second shape parameter, in (0, inf)
                scale_range: tuple[float, float], default (0.1, 0.3)
                    Smaller scale for subtler noise
                    Range for sampling output scale, in [0, 1]

    Examples:
        >>> # Constant RGB shift with different ranges per channel:
        >>> transform = AdditiveNoise(
        ...     noise_type="uniform",
        ...     spatial_mode="constant",
        ...     noise_params={"ranges": [(-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1)]}
        ... )

        Gaussian noise shared across channels:
        >>> transform = AdditiveNoise(
        ...     noise_type="gaussian",
        ...     spatial_mode="shared",
        ...     noise_params={"mean_range": (0.0, 0.0), "std_range": (0.05, 0.15)}
        ... )

    Note:
        Performance considerations:
            - "constant" mode is fastest as it generates only C values (C = number of channels)
            - "shared" mode generates HxW values and reuses them for all channels
            - "per_pixel" mode generates HxWxC values, slowest but most flexible

        Distribution characteristics:
            - uniform: Equal probability within range, good for simple perturbations
            - gaussian: Bell-shaped, symmetric, good for natural noise
            - laplace: Like gaussian but with heavier tails, good for outliers
            - beta: Very flexible shape, can be uniform, bell-shaped, or U-shaped

        Implementation details:
            - All noise is generated in normalized range and scaled by image max value
            - For uint8 images, final noise range is [-255, 255]
            - For float images, final noise range is [-1, 1]

    """

    class InitSchema(BaseTransformInitSchema):
        noise_type: Literal["uniform", "gaussian", "laplace", "beta"]
        spatial_mode: Literal["constant", "per_pixel", "shared"]
        noise_params: dict[str, Any] | None
        approximation: float = Field(ge=0.0, le=1.0)

        @model_validator(mode="after")
        def _validate_noise_params(self) -> Self:
            # Default parameters for each noise type
            default_params = {
                "uniform": {
                    "ranges": [(-0.1, 0.1)],  # Single channel by default
                },
                "gaussian": {"mean_range": (0.0, 0.0), "std_range": (0.05, 0.15)},
                "laplace": {"mean_range": (0.0, 0.0), "scale_range": (0.05, 0.15)},
                "beta": {
                    "alpha_range": (0.5, 1.5),
                    "beta_range": (0.5, 1.5),
                    "scale_range": (0.1, 0.3),
                },
            }

            # Use default params if none provided
            params_dict = self.noise_params if self.noise_params is not None else default_params[self.noise_type]

            # Add noise_type to params if not present
            params_dict = {**params_dict, "noise_type": self.noise_type}  # type: ignore[dict-item]

            # Convert dict to appropriate NoiseParams object and validate
            params_class = {
                "uniform": UniformParams,
                "gaussian": GaussianParams,
                "laplace": LaplaceParams,
                "beta": BetaParams,
            }[self.noise_type]

            # Validate using the appropriate NoiseParams class
            validated_params = params_class(**params_dict)

            # Store the validated parameters as a dict
            self.noise_params = validated_params.model_dump()

            return self

    def __init__(
        self,
        noise_type: Literal["uniform", "gaussian", "laplace", "beta"] = "uniform",
        spatial_mode: Literal["constant", "per_pixel", "shared"] = "constant",
        noise_params: dict[str, Any] | None = None,
        approximation: float = 1.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.noise_type = noise_type
        self.spatial_mode = spatial_mode
        self.noise_params = noise_params
        self.approximation = approximation

    def apply(
        self,
        img: np.ndarray,
        noise_map: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the AdditiveNoise transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the AdditiveNoise transform to.
            noise_map (np.ndarray): The noise map to apply to the image.
            **params (Any): Additional parameters for the transform.

        """
        return fpixel.add_noise(img, noise_map)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters for the AdditiveNoise transform.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data to apply the transform to.

        """
        image = data["image"] if "image" in data else data["images"][0]

        max_value = MAX_VALUES_BY_DTYPE[image.dtype]

        noise_map = fpixel.generate_noise(
            noise_type=self.noise_type,
            spatial_mode=self.spatial_mode,
            shape=image.shape,
            params=self.noise_params,
            max_value=max_value,
            approximation=self.approximation,
            random_generator=self.random_generator,
        )
        return {"noise_map": noise_map}


class RGBShift(AdditiveNoise):
    """Randomly shift values for each channel of the input RGB image.

    A specialized version of AdditiveNoise that applies constant uniform shifts to RGB channels.
    Each channel (R,G,B) can have its own shift range specified.

    Args:
        r_shift_limit ((int, int) or int): Range for shifting the red channel. Options:
            - If tuple (min, max): Sample shift value from this range
            - If int: Sample shift value from (-r_shift_limit, r_shift_limit)
            - For uint8 images: Values represent absolute shifts in [0, 255]
            - For float images: Values represent relative shifts in [0, 1]
            Default: (-20, 20)

        g_shift_limit ((int, int) or int): Range for shifting the green channel. Options:
            - If tuple (min, max): Sample shift value from this range
            - If int: Sample shift value from (-g_shift_limit, g_shift_limit)
            - For uint8 images: Values represent absolute shifts in [0, 255]
            - For float images: Values represent relative shifts in [0, 1]
            Default: (-20, 20)

        b_shift_limit ((int, int) or int): Range for shifting the blue channel. Options:
            - If tuple (min, max): Sample shift value from this range
            - If int: Sample shift value from (-b_shift_limit, b_shift_limit)
            - For uint8 images: Values represent absolute shifts in [0, 255]
            - For float images: Values represent relative shifts in [0, 1]
            Default: (-20, 20)

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - Values are shifted independently for each channel
        - For uint8 images:
            * Input ranges like (-20, 20) represent pixel value shifts
            * A shift of 20 means adding 20 to that channel
            * Final values are clipped to [0, 255]
        - For float32 images:
            * Input ranges like (-0.1, 0.1) represent relative shifts
            * A shift of 0.1 means adding 0.1 to that channel
            * Final values are clipped to [0, 1]

    Examples:
        >>> import numpy as np
        >>> import albumentations as A

        # Shift RGB channels of uint8 image
        >>> transform = A.RGBShift(
        ...     r_shift_limit=30,  # Will sample red shift from [-30, 30]
        ...     g_shift_limit=(-20, 20),  # Will sample green shift from [-20, 20]
        ...     b_shift_limit=(-10, 10),  # Will sample blue shift from [-10, 10]
        ...     p=1.0
        ... )
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> shifted = transform(image=image)["image"]

        # Same effect using AdditiveNoise
        >>> transform = A.AdditiveNoise(
        ...     noise_type="uniform",
        ...     spatial_mode="constant",  # One value per channel
        ...     noise_params={
        ...         "ranges": [(-30/255, 30/255), (-20/255, 20/255), (-10/255, 10/255)]
        ...     },
        ...     p=1.0
        ... )

    See Also:
        - AdditiveNoise: More general noise transform with various options:
            * Different noise distributions (uniform, gaussian, laplace, beta)
            * Spatial modes (constant, per-pixel, shared)
            * Approximation for faster computation
        - RandomToneCurve: For non-linear color transformations
        - RandomBrightnessContrast: For combined brightness and contrast adjustments
        - PlankianJitter: For color temperature adjustments
        - HueSaturationValue: For HSV color space adjustments
        - ColorJitter: For combined brightness, contrast, saturation adjustments

    """

    class InitSchema(BaseTransformInitSchema):
        r_shift_limit: SymmetricRangeType
        g_shift_limit: SymmetricRangeType
        b_shift_limit: SymmetricRangeType

    def __init__(
        self,
        r_shift_limit: tuple[float, float] | float = (-20, 20),
        g_shift_limit: tuple[float, float] | float = (-20, 20),
        b_shift_limit: tuple[float, float] | float = (-20, 20),
        p: float = 0.5,
    ):
        # Convert RGB shift limits to normalized ranges if needed
        def normalize_range(limit: tuple[float, float]) -> tuple[float, float]:
            # If any value is > 1, assume uint8 range and normalize
            if abs(limit[0]) > 1 or abs(limit[1]) > 1:
                return (limit[0] / 255.0, limit[1] / 255.0)
            return limit

        ranges = [
            normalize_range(cast("tuple[float, float]", r_shift_limit)),
            normalize_range(cast("tuple[float, float]", g_shift_limit)),
            normalize_range(cast("tuple[float, float]", b_shift_limit)),
        ]

        # Initialize with fixed noise type and spatial mode
        super().__init__(
            noise_type="uniform",
            spatial_mode="constant",
            noise_params={"ranges": ranges},
            approximation=1.0,
            p=p,
        )

        # Store original limits for get_transform_init_args
        self.r_shift_limit = cast("tuple[float, float]", r_shift_limit)
        self.g_shift_limit = cast("tuple[float, float]", g_shift_limit)
        self.b_shift_limit = cast("tuple[float, float]", b_shift_limit)


class SaltAndPepper(ImageOnlyTransform):
    """Apply salt and pepper noise to the input image.

    Salt and pepper noise is a form of impulse noise that randomly sets pixels to either maximum value (salt)
    or minimum value (pepper). The amount and proportion of salt vs pepper noise can be controlled.
    The same noise mask is applied to all channels of the image to preserve color consistency.

    Args:
        amount ((float, float)): Range for total amount of noise (both salt and pepper).
            Values between 0 and 1. For example:
            - 0.05 means 5% of all pixels will be replaced with noise
            - (0.01, 0.06) will sample amount uniformly from 1% to 6%
            Default: (0.01, 0.06)

        salt_vs_pepper ((float, float)): Range for ratio of salt (white) vs pepper (black) noise.
            Values between 0 and 1. For example:
            - 0.5 means equal amounts of salt and pepper
            - 0.7 means 70% of noisy pixels will be salt, 30% pepper
            - (0.4, 0.6) will sample ratio uniformly from 40% to 60%
            Default: (0.4, 0.6)

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - Salt noise sets pixels to maximum value (255 for uint8, 1.0 for float32)
        - Pepper noise sets pixels to 0
        - The noise mask is generated once and applied to all channels to maintain
          color consistency (i.e., if a pixel is set to salt, all its color channels
          will be set to maximum value)
        - The exact number of affected pixels matches the specified amount as masks
          are generated without overlap

    Mathematical Formulation:
        For an input image I, the output O is:
        O[c,x,y] = max_value,  if salt_mask[x,y] = True
        O[c,x,y] = 0,         if pepper_mask[x,y] = True
        O[c,x,y] = I[c,x,y],  otherwise

        where:
        - c is the channel index
        - salt_mask and pepper_mask are 2D boolean arrays applied to all channels
        - Number of True values in salt_mask = floor(H*W * amount * salt_ratio)
        - Number of True values in pepper_mask = floor(H*W * amount * (1 - salt_ratio))
        - amount ∈ [amount_min, amount_max]
        - salt_ratio ∈ [salt_vs_pepper_min, salt_vs_pepper_max]

    Examples:
        >>> import albumentations as A
        >>> import numpy as np

        # Apply salt and pepper noise with default parameters
        >>> transform = A.SaltAndPepper(p=1.0)
        >>> noisy_image = transform(image=image)["image"]

        # Heavy noise with more salt than pepper
        >>> transform = A.SaltAndPepper(
        ...     amount=(0.1, 0.2),       # 10-20% of pixels will be noisy
        ...     salt_vs_pepper=(0.7, 0.9),  # 70-90% of noise will be salt
        ...     p=1.0
        ... )
        >>> noisy_image = transform(image=image)["image"]

    References:
        - Digital Image Processing: Rafael C. Gonzalez and Richard E. Woods, 4th Edition,
            Chapter 5: Image Restoration and Reconstruction.
        - Fundamentals of Digital Image Processing: A. K. Jain, Chapter 7: Image Degradation and Restoration.
        - Salt and pepper noise: https://en.wikipedia.org/wiki/Salt-and-pepper_noise

    See Also:
        - GaussNoise: For additive Gaussian noise
        - MultiplicativeNoise: For multiplicative noise
        - ISONoise: For camera sensor noise simulation

    """

    class InitSchema(BaseTransformInitSchema):
        amount: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        salt_vs_pepper: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]

    def __init__(
        self,
        amount: tuple[float, float] = (0.01, 0.06),
        salt_vs_pepper: tuple[float, float] = (0.4, 0.6),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters for the SaltAndPepper transform.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data to apply the transform to.

        """
        image = data["image"] if "image" in data else data["images"][0]
        height, width = image.shape[:2]

        total_amount = self.py_random.uniform(*self.amount)
        salt_ratio = self.py_random.uniform(*self.salt_vs_pepper)

        area = height * width

        num_pixels = int(area * total_amount)
        num_salt = int(num_pixels * salt_ratio)

        # Generate all positions at once
        noise_positions = self.random_generator.choice(area, size=num_pixels, replace=False)

        # Create masks
        salt_mask = np.zeros(area, dtype=bool)
        pepper_mask = np.zeros(area, dtype=bool)

        # Set salt and pepper positions
        salt_mask[noise_positions[:num_salt]] = True
        pepper_mask[noise_positions[num_salt:]] = True

        # Reshape to 2D
        salt_mask = salt_mask.reshape(height, width)
        pepper_mask = pepper_mask.reshape(height, width)

        return {
            "salt_mask": salt_mask,
            "pepper_mask": pepper_mask,
        }

    def apply(
        self,
        img: np.ndarray,
        salt_mask: np.ndarray,
        pepper_mask: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the SaltAndPepper transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the SaltAndPepper transform to.
            salt_mask (np.ndarray): The salt mask to apply to the image.
            pepper_mask (np.ndarray): The pepper mask to apply to the image.
            **params (Any): Additional parameters for the transform.

        """
        return fpixel.apply_salt_and_pepper(img, salt_mask, pepper_mask)


class PlasmaBrightnessContrast(ImageOnlyTransform):
    """Apply plasma fractal pattern to modify image brightness and contrast.

    Uses Diamond-Square algorithm to generate organic-looking fractal patterns
    that create spatially-varying brightness and contrast adjustments.

    Args:
        brightness_range ((float, float)): Range for brightness adjustment strength.
            Values between -1 and 1:
            - Positive values increase brightness
            - Negative values decrease brightness
            - 0 means no brightness change
            Default: (-0.3, 0.3)

        contrast_range ((float, float)): Range for contrast adjustment strength.
            Values between -1 and 1:
            - Positive values increase contrast
            - Negative values decrease contrast
            - 0 means no contrast change
            Default: (-0.3, 0.3)

        plasma_size (int): Size of the initial plasma pattern grid.
            Larger values create more detailed patterns but are slower to compute.
            The pattern will be resized to match the input image dimensions.
            Default: 256

        roughness (float): Controls how quickly the noise amplitude increases at each iteration.
            Must be greater than 0:
            - Low values (< 1.0): Smoother, more gradual pattern
            - Medium values (~2.0): Natural-looking pattern
            - High values (> 3.0): Very rough, noisy pattern
            Default: 3.0

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - Works with any number of channels (grayscale, RGB, multispectral)
        - The same plasma pattern is applied to all channels
        - Operations are performed in float32 precision
        - Final values are clipped to valid range [0, max_value]

    Mathematical Formulation:
        1. Plasma Pattern Generation (Diamond-Square Algorithm):
           Starting with a 3x3 grid of random values in [-1, 1], iteratively:
           a) Diamond Step: For each 2x2 cell, compute center using diamond kernel:
              [[0.25, 0.0, 0.25],
               [0.0,  0.0, 0.0 ],
               [0.25, 0.0, 0.25]]

           b) Square Step: Fill remaining points using square kernel:
              [[0.0,  0.25, 0.0 ],
               [0.25, 0.0,  0.25],
               [0.0,  0.25, 0.0 ]]

           c) Add random noise scaled by roughness^iteration

           d) Normalize final pattern P to [0,1] range using min-max normalization

        2. Brightness Adjustment:
           For each pixel (x,y):
           O(x,y) = I(x,y) + b·P(x,y)
           where:
           - I is the input image
           - b is the brightness factor
           - P is the normalized plasma pattern

        3. Contrast Adjustment:
           For each pixel (x,y):
           O(x,y) = I(x,y)·(1 + c·P(x,y)) + μ·(1 - (1 + c·P(x,y)))
           where:
           - I is the input image
           - c is the contrast factor
           - P is the normalized plasma pattern
           - μ is the mean pixel value

    Examples:
        >>> import albumentations as A
        >>> import numpy as np

        # Default parameters
        >>> transform = A.PlasmaBrightnessContrast(p=1.0)

        # Custom adjustments
        >>> transform = A.PlasmaBrightnessContrast(
        ...     brightness_range=(-0.5, 0.5),
        ...     contrast_range=(-0.3, 0.3),
        ...     plasma_size=512,    # More detailed pattern
        ...     roughness=0.7,      # Smoother transitions
        ...     p=1.0
        ... )

    References:
        - Fournier, Fussell, and Carpenter, "Computer rendering of stochastic models,": Communications of
            the ACM, 1982. Paper introducing the Diamond-Square algorithm.
        - Diamond-Square algorithm: https://en.wikipedia.org/wiki/Diamond-square_algorithm

    See Also:
        - RandomBrightnessContrast: For uniform brightness/contrast adjustments
        - CLAHE: For contrast limited adaptive histogram equalization
        - FancyPCA: For color-based contrast enhancement
        - HistogramMatching: For reference-based contrast adjustment

    """

    class InitSchema(BaseTransformInitSchema):
        brightness_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(-1, 1)),
        ]
        contrast_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(-1, 1)),
        ]
        plasma_size: int = Field(ge=1)
        roughness: float = Field(gt=0)

    def __init__(
        self,
        brightness_range: tuple[float, float] = (-0.3, 0.3),
        contrast_range: tuple[float, float] = (-0.3, 0.3),
        plasma_size: int = 256,
        roughness: float = 3.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.plasma_size = plasma_size
        self.roughness = roughness

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters for the PlasmaBrightnessContrast transform.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data to apply the transform to.

        """
        shape = params["shape"]

        # Sample adjustment strengths
        brightness = self.py_random.uniform(*self.brightness_range)
        contrast = self.py_random.uniform(*self.contrast_range)

        # Generate plasma pattern
        plasma = fpixel.generate_plasma_pattern(
            target_shape=shape[:2],
            roughness=self.roughness,
            random_generator=self.random_generator,
        )

        return {
            "brightness_factor": brightness,
            "contrast_factor": contrast,
            "plasma_pattern": plasma,
        }

    def apply(
        self,
        img: np.ndarray,
        brightness_factor: float,
        contrast_factor: float,
        plasma_pattern: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the PlasmaBrightnessContrast transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the PlasmaBrightnessContrast transform to.
            brightness_factor (float): The brightness factor to apply to the image.
            contrast_factor (float): The contrast factor to apply to the image.
            plasma_pattern (np.ndarray): The plasma pattern to apply to the image.
            **params (Any): Additional parameters for the transform.

        """
        return fpixel.apply_plasma_brightness_contrast(
            img,
            brightness_factor,
            contrast_factor,
            plasma_pattern,
        )

    @batch_transform("spatial", keep_depth_dim=False, has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the PlasmaBrightnessContrast transform to a batch of images.

        Args:
            images (np.ndarray): The input images to apply the PlasmaBrightnessContrast transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(images, **params)

    @batch_transform("spatial", keep_depth_dim=True, has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the PlasmaBrightnessContrast transform to a volume.

        Args:
            volume (np.ndarray): The input volume to apply the PlasmaBrightnessContrast transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(volume, **params)

    @batch_transform("spatial", keep_depth_dim=True, has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the PlasmaBrightnessContrast transform to a batch of volumes.

        Args:
            volumes (np.ndarray): The input volumes to apply the PlasmaBrightnessContrast transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(volumes, **params)


class PlasmaShadow(ImageOnlyTransform):
    """Apply plasma-based shadow effect to the image using Diamond-Square algorithm.

    Creates organic-looking shadows using plasma fractal noise pattern.
    The shadow intensity varies smoothly across the image, creating natural-looking
    darkening effects that can simulate shadows, shading, or lighting variations.

    Args:
        shadow_intensity_range (tuple[float, float]): Range for shadow intensity.
            Values between 0 and 1:
            - 0 means no shadow (original image)
            - 1 means maximum darkening (black)
            - Values between create partial shadows
            Default: (0.3, 0.7)

        roughness (float): Controls how quickly the noise amplitude increases at each iteration.
            Must be greater than 0:
            - Low values (< 1.0): Smoother, more gradual shadows
            - Medium values (~2.0): Natural-looking shadows
            - High values (> 3.0): Very rough, noisy shadows
            Default: 3.0

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - The transform darkens the image using a plasma pattern
        - Works with any number of channels (grayscale, RGB, multispectral)
        - Shadow pattern is generated using Diamond-Square algorithm with specific kernels
        - The same shadow pattern is applied to all channels
        - Final values are clipped to valid range [0, max_value]

    Mathematical Formulation:
        1. Plasma Pattern Generation (Diamond-Square Algorithm):
           Starting with a 3x3 grid of random values in [-1, 1], iteratively:
           a) Diamond Step: For each 2x2 cell, compute center using diamond kernel:
              [[0.25, 0.0, 0.25],
               [0.0,  0.0, 0.0 ],
               [0.25, 0.0, 0.25]]

           b) Square Step: Fill remaining points using square kernel:
              [[0.0,  0.25, 0.0 ],
               [0.25, 0.0,  0.25],
               [0.0,  0.25, 0.0 ]]

           c) Add random noise scaled by roughness^iteration

           d) Normalize final pattern P to [0,1] range using min-max normalization

        2. Shadow Application:
           For each pixel (x,y):
           O(x,y) = I(x,y) * (1 - i*P(x,y))
           where:
           - I is the input image
           - P is the normalized plasma pattern
           - i is the sampled shadow intensity
           - O is the output image

    Examples:
        >>> import albumentations as A
        >>> import numpy as np

        # Default parameters for natural shadows
        >>> transform = A.PlasmaShadow(p=1.0)

        # Subtle, smooth shadows
        >>> transform = A.PlasmaShadow(
        ...     shadow_intensity_range=(0.1, 0.3),
        ...     roughness=0.7,
        ...     p=1.0
        ... )

        # Dramatic, detailed shadows
        >>> transform = A.PlasmaShadow(
        ...     shadow_intensity_range=(0.5, 0.9),
        ...     roughness=0.3,
        ...     p=1.0
        ... )

    References:
        - Fournier, Fussell, and Carpenter, "Computer rendering of stochastic models,": Communications of
            the ACM, 1982. Paper introducing the Diamond-Square algorithm.
        - Diamond-Square algorithm: https://en.wikipedia.org/wiki/Diamond-square_algorithm

    See Also:
        - PlasmaBrightnessContrast: For brightness/contrast adjustments using plasma patterns
        - RandomShadow: For geometric shadow effects
        - RandomToneCurve: For global lighting adjustments
        - PlasmaBrightnessContrast: For brightness/contrast adjustments using plasma patterns

    """

    class InitSchema(BaseTransformInitSchema):
        shadow_intensity_range: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1))]
        roughness: float = Field(gt=0)

    def __init__(
        self,
        shadow_intensity_range: tuple[float, float] = (0.3, 0.7),
        plasma_size: int = 256,
        roughness: float = 3.0,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.shadow_intensity_range = shadow_intensity_range
        self.plasma_size = plasma_size
        self.roughness = roughness

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate parameters for the PlasmaShadow transform.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data to apply the transform to.

        """
        shape = params["shape"]

        # Sample shadow intensity
        intensity = self.py_random.uniform(*self.shadow_intensity_range)

        # Generate plasma pattern
        plasma = fpixel.generate_plasma_pattern(
            target_shape=shape[:2],
            roughness=self.roughness,
            random_generator=self.random_generator,
        )

        return {
            "intensity": intensity,
            "plasma_pattern": plasma,
        }

    def apply(
        self,
        img: np.ndarray,
        intensity: float,
        plasma_pattern: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the PlasmaShadow transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the PlasmaShadow transform to.
            intensity (float): The intensity of the shadow to apply to the image.
            plasma_pattern (np.ndarray): The plasma pattern to apply to the image.
            **params (Any): Additional parameters for the transform.

        """
        return fpixel.apply_plasma_shadow(img, intensity, plasma_pattern)

    @batch_transform("spatial", keep_depth_dim=False, has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the PlasmaShadow transform to a batch of images.

        Args:
            images (np.ndarray): The input images to apply the PlasmaShadow transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(images, **params)

    @batch_transform("spatial", keep_depth_dim=True, has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the PlasmaShadow transform to a batch of volume.

        Args:
            volume (np.ndarray): The input volume to apply the PlasmaShadow transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(volume, **params)

    @batch_transform("spatial", keep_depth_dim=True, has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the PlasmaShadow transform to a batch of volumes.

        Args:
            volumes (np.ndarray): The input volumes to apply the PlasmaShadow transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(volumes, **params)


class Illumination(ImageOnlyTransform):
    """Apply various illumination effects to the image.

    This transform simulates different lighting conditions by applying controlled
    illumination patterns. It can create effects like:
    - Directional lighting (linear mode)
    - Corner shadows/highlights (corner mode)
    - Spotlights or local lighting (gaussian mode)

    These effects can be used to:
    - Simulate natural lighting variations
    - Add dramatic lighting effects
    - Create synthetic shadows or highlights
    - Augment training data with different lighting conditions

    Args:
        mode (Literal["linear", "corner", "gaussian"]): Type of illumination pattern:
            - 'linear': Creates a smooth gradient across the image,
                       simulating directional lighting like sunlight
                       through a window
            - 'corner': Applies gradient from any corner,
                       simulating light source from a corner
            - 'gaussian': Creates a circular spotlight effect,
                         simulating local light sources
            Default: 'linear'

        intensity_range (tuple[float, float]): Range for effect strength.
            Values between 0.01 and 0.2:
            - 0.01-0.05: Subtle lighting changes
            - 0.05-0.1: Moderate lighting effects
            - 0.1-0.2: Strong lighting effects
            Default: (0.01, 0.2)

        effect_type (str): Type of lighting change:
            - 'brighten': Only adds light (like a spotlight)
            - 'darken': Only removes light (like a shadow)
            - 'both': Randomly chooses between brightening and darkening
            Default: 'both'

        angle_range (tuple[float, float]): Range for gradient angle in degrees.
            Controls direction of linear gradient:
            - 0°: Left to right
            - 90°: Top to bottom
            - 180°: Right to left
            - 270°: Bottom to top
            Only used for 'linear' mode.
            Default: (0, 360)

        center_range (tuple[float, float]): Range for spotlight position.
            Values between 0 and 1 representing relative position:
            - (0, 0): Top-left corner
            - (1, 1): Bottom-right corner
            - (0.5, 0.5): Center of image
            Only used for 'gaussian' mode.
            Default: (0.1, 0.9)

        sigma_range (tuple[float, float]): Range for spotlight size.
            Values between 0.2 and 1.0:
            - 0.2: Small, focused spotlight
            - 0.5: Medium-sized light area
            - 1.0: Broad, soft lighting
            Only used for 'gaussian' mode.
            Default: (0.2, 1.0)

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Examples:
        >>> import albumentations as A
        >>> # Simulate sunlight through window
        >>> transform = A.Illumination(
        ...     mode='linear',
        ...     intensity_range=(0.05, 0.1),
        ...     effect_type='brighten',
        ...     angle_range=(30, 60)
        ... )
        >>>
        >>> # Create dramatic corner shadow
        >>> transform = A.Illumination(
        ...     mode='corner',
        ...     intensity_range=(0.1, 0.2),
        ...     effect_type='darken'
        ... )
        >>>
        >>> # Add multiple spotlights
        >>> transform1 = A.Illumination(
        ...     mode='gaussian',
        ...     intensity_range=(0.05, 0.15),
        ...     effect_type='brighten',
        ...     center_range=(0.2, 0.4),
        ...     sigma_range=(0.2, 0.3)
        ... )
        >>> transform2 = A.Illumination(
        ...     mode='gaussian',
        ...     intensity_range=(0.05, 0.15),
        ...     effect_type='darken',
        ...     center_range=(0.6, 0.8),
        ...     sigma_range=(0.3, 0.5)
        ... )
        >>> transforms = A.Compose([transform1, transform2])

    References:
        - Lighting in Computer Vision:
          https://en.wikipedia.org/wiki/Lighting_in_computer_vision

        - Image-based lighting:
          https://en.wikipedia.org/wiki/Image-based_lighting

        - Similar implementation in Kornia:
          https://kornia.readthedocs.io/en/latest/augmentation.html#randomlinearillumination

        - Research on lighting augmentation:
          "Learning Deep Representations of Fine-grained Visual Descriptions"
          https://arxiv.org/abs/1605.05395

        - Photography lighting patterns:
          https://en.wikipedia.org/wiki/Lighting_pattern

    Note:
        - The transform preserves image range and dtype
        - Effects are applied multiplicatively to preserve texture
        - Can be combined with other transforms for complex lighting scenarios
        - Useful for training models to be robust to lighting variations

    """

    class InitSchema(BaseTransformInitSchema):
        mode: Literal["linear", "corner", "gaussian"]
        intensity_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0.01, 0.2)),
        ]
        effect_type: Literal["brighten", "darken", "both"]
        angle_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 360)),
        ]
        center_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, 1)),
        ]
        sigma_range: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0.2, 1.0)),
        ]

    def __init__(
        self,
        mode: Literal["linear", "corner", "gaussian"] = "linear",
        intensity_range: tuple[float, float] = (0.01, 0.2),
        effect_type: Literal["brighten", "darken", "both"] = "both",
        angle_range: tuple[float, float] = (0, 360),
        center_range: tuple[float, float] = (0.1, 0.9),
        sigma_range: tuple[float, float] = (0.2, 1.0),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.mode = mode
        self.intensity_range = intensity_range
        self.effect_type = effect_type
        self.angle_range = angle_range
        self.center_range = center_range
        self.sigma_range = sigma_range

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Generate parameters for the Illumination transform.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data to apply the transform to.

        """
        intensity = self.py_random.uniform(*self.intensity_range)

        # Determine if brightening or darkening
        sign = 1  # brighten
        if self.effect_type == "both":
            sign = 1 if self.py_random.random() > 0.5 else -1
        elif self.effect_type == "darken":
            sign = -1

        intensity *= sign

        if self.mode == "linear":
            angle = self.py_random.uniform(*self.angle_range)
            return {
                "intensity": intensity,
                "angle": angle,
            }
        if self.mode == "corner":
            corner = self.py_random.randint(0, 3)  # Choose random corner
            return {
                "intensity": intensity,
                "corner": corner,
            }

        x = self.py_random.uniform(*self.center_range)
        y = self.py_random.uniform(*self.center_range)
        sigma = self.py_random.uniform(*self.sigma_range)
        return {
            "intensity": intensity,
            "center": (x, y),
            "sigma": sigma,
        }

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the Illumination transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the Illumination transform to.
            **params (Any): Additional parameters for the transform.

        """
        if self.mode == "linear":
            return fpixel.apply_linear_illumination(
                img,
                intensity=params["intensity"],
                angle=params["angle"],
            )
        if self.mode == "corner":
            return fpixel.apply_corner_illumination(
                img,
                intensity=params["intensity"],
                corner=params["corner"],
            )

        return fpixel.apply_gaussian_illumination(
            img,
            intensity=params["intensity"],
            center=params["center"],
            sigma=params["sigma"],
        )


class AutoContrast(ImageOnlyTransform):
    """Automatically adjust image contrast by stretching the intensity range.

    This transform provides two methods for contrast enhancement:
    1. CDF method (default): Uses cumulative distribution function for more gradual adjustment
    2. PIL method: Uses linear scaling like PIL.ImageOps.autocontrast

    The transform can optionally exclude extreme values from both ends of the
    intensity range and preserve specific intensity values (e.g., alpha channel).

    Args:
        cutoff (float): Percentage of pixels to exclude from both ends of the histogram.
            Range: [0, 100]. Default: 0 (use full intensity range)
            - 0 means use the minimum and maximum intensity values found
            - 20 means exclude darkest and brightest 20% of pixels
        ignore (int, optional): Intensity value to preserve (e.g., alpha channel).
            Range: [0, 255]. Default: None
            - If specified, this intensity value will not be modified
            - Useful for images with alpha channel or special marker values
        method (Literal["cdf", "pil"]): Algorithm to use for contrast enhancement.
            Default: "cdf"
            - "cdf": Uses cumulative distribution for smoother adjustment
            - "pil": Uses linear scaling like PIL.ImageOps.autocontrast
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - The transform processes each color channel independently
        - For grayscale images, only one channel is processed
        - The output maintains the same dtype as input
        - Empty or single-color channels remain unchanged

    Examples:
        >>> import albumentations as A
        >>> # Basic usage
        >>> transform = A.AutoContrast(p=1.0)
        >>>
        >>> # Exclude extreme values
        >>> transform = A.AutoContrast(cutoff=20, p=1.0)
        >>>
        >>> # Preserve alpha channel
        >>> transform = A.AutoContrast(ignore=255, p=1.0)
        >>>
        >>> # Use PIL-like contrast enhancement
        >>> transform = A.AutoContrast(method="pil", p=1.0)

    """

    class InitSchema(BaseTransformInitSchema):
        cutoff: float = Field(ge=0, le=100)
        ignore: int | None = Field(ge=0, le=255)
        method: Literal["cdf", "pil"]

    def __init__(
        self,
        cutoff: float = 0,
        ignore: int | None = None,
        method: Literal["cdf", "pil"] = "cdf",
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.cutoff = cutoff
        self.ignore = ignore
        self.method = method

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the AutoContrast transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the AutoContrast transform to.
            **params (Any): Additional parameters for the transform.

        """
        return fpixel.auto_contrast(img, self.cutoff, self.ignore, self.method)

    @batch_transform("channel", has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the AutoContrast transform to a batch of images.

        Args:
            images (np.ndarray): The input images to apply the AutoContrast transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(images, **params)

    @batch_transform("channel", has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the AutoContrast transform to a batch of volumes.

        Args:
            volume (np.ndarray): The input volume to apply the AutoContrast transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(volume, **params)

    @batch_transform("channel", has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the AutoContrast transform to a batch of volumes.

        Args:
            volumes (np.ndarray): The input volumes to apply the AutoContrast transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(volumes, **params)


class HEStain(ImageOnlyTransform):
    """Applies H&E (Hematoxylin and Eosin) stain augmentation to histopathology images.

    This transform simulates different H&E staining conditions using either:
    1. Predefined stain matrices (8 standard references)
    2. Vahadane method for stain extraction
    3. Macenko method for stain extraction
    4. Custom stain matrices

    Args:
        method(Literal["preset", "random_preset", "vahadane", "macenko"]): Method to use for stain augmentation:
            - "preset": Use predefined stain matrices
            - "random_preset": Randomly select a preset matrix each time
            - "vahadane": Extract using Vahadane method
            - "macenko": Extract using Macenko method
            Default: "preset"

        preset(str | None): Preset stain matrix to use when method="preset":
            - "ruifrok": Standard reference from Ruifrok & Johnston
            - "macenko": Reference from Macenko's method
            - "standard": Typical bright-field microscopy
            - "high_contrast": Enhanced contrast
            - "h_heavy": Hematoxylin dominant
            - "e_heavy": Eosin dominant
            - "dark": Darker staining
            - "light": Lighter staining
            Default: "standard"

        intensity_scale_range(tuple[float, float]): Range for multiplicative stain intensity variation.
            Values are multipliers between 0.5 and 1.5. For example:
            - (0.7, 1.3) means stain intensities will vary from 70% to 130%
            - (0.9, 1.1) gives subtle variations
            - (0.5, 1.5) gives dramatic variations
            Default: (0.7, 1.3)

        intensity_shift_range(tuple[float, float]): Range for additive stain intensity variation.
            Values between -0.3 and 0.3. For example:
            - (-0.2, 0.2) means intensities will be shifted by -20% to +20%
            - (-0.1, 0.1) gives subtle shifts
            - (-0.3, 0.3) gives dramatic shifts
            Default: (-0.2, 0.2)

        augment_background(bool): Whether to apply augmentation to background regions.
            Default: False

    Targets:
        image

    Number of channels:
        3

    Image types:
        uint8, float32

    References:
        - A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical": Analytical and quantitative
            cytology and histology, 2001.
        - M. Macenko et al., "A method for normalizing histology slides for: 2009 IEEE International Symposium on
            quantitative analysis," 2009 IEEE International Symposium on Biomedical Imaging, 2009.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample H&E stained histopathology image
        >>> # For real use cases, load an actual H&E stained image
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Simulate tissue regions with different staining patterns
        >>> image[50:150, 50:150] = np.array([120, 140, 180], dtype=np.uint8)  # Hematoxylin-rich region
        >>> image[150:250, 150:250] = np.array([140, 160, 120], dtype=np.uint8)  # Eosin-rich region
        >>>
        >>> # Example 1: Using a specific preset stain matrix
        >>> transform = A.HEStain(
        ...     method="preset",
        ...     preset="standard",
        ...     intensity_scale_range=(0.8, 1.2),
        ...     intensity_shift_range=(-0.1, 0.1),
        ...     augment_background=False,
        ...     p=1.0
        ... )
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        >>>
        >>> # Example 2: Using random preset selection
        >>> transform = A.HEStain(
        ...     method="random_preset",
        ...     intensity_scale_range=(0.7, 1.3),
        ...     intensity_shift_range=(-0.15, 0.15),
        ...     p=1.0
        ... )
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        >>>
        >>> # Example 3: Using Vahadane method (requires H&E stained input)
        >>> transform = A.HEStain(
        ...     method="vahadane",
        ...     intensity_scale_range=(0.7, 1.3),
        ...     p=1.0
        ... )
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        >>>
        >>> # Example 4: Using Macenko method (requires H&E stained input)
        >>> transform = A.HEStain(
        ...     method="macenko",
        ...     intensity_scale_range=(0.7, 1.3),
        ...     intensity_shift_range=(-0.2, 0.2),
        ...     p=1.0
        ... )
        >>> result = transform(image=image)
        >>> transformed_image = result['image']
        >>>
        >>> # Example 5: Combining with other transforms in a pipeline
        >>> transform = A.Compose([
        ...     A.HEStain(method="preset", preset="high_contrast", p=1.0),
        ...     A.RandomBrightnessContrast(p=0.5),
        ... ])
        >>> result = transform(image=image)
        >>> transformed_image = result['image']

    """

    class InitSchema(BaseTransformInitSchema):
        method: Literal["preset", "random_preset", "vahadane", "macenko"]
        preset: (
            Literal[
                "ruifrok",
                "macenko",
                "standard",
                "high_contrast",
                "h_heavy",
                "e_heavy",
                "dark",
                "light",
            ]
            | None
        )
        intensity_scale_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, None)),
        ]
        intensity_shift_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(-1, 1)),
        ]
        augment_background: bool

        @model_validator(mode="after")
        def _validate_matrix_selection(self) -> Self:
            if self.method == "preset" and self.preset is None:
                self.preset = "standard"
            elif self.method == "random_preset" and self.preset is not None:
                raise ValueError("preset should not be specified when method='random_preset'")
            return self

    def __init__(
        self,
        method: Literal["preset", "random_preset", "vahadane", "macenko"] = "random_preset",
        preset: Literal[
            "ruifrok",
            "macenko",
            "standard",
            "high_contrast",
            "h_heavy",
            "e_heavy",
            "dark",
            "light",
        ]
        | None = None,
        intensity_scale_range: tuple[float, float] = (0.7, 1.3),
        intensity_shift_range: tuple[float, float] = (-0.2, 0.2),
        augment_background: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.method = method
        self.preset = preset
        self.intensity_scale_range = intensity_scale_range
        self.intensity_shift_range = intensity_shift_range
        self.augment_background = augment_background
        self.stain_normalizer = None

        # Initialize stain extractor here if needed
        if method in ["vahadane", "macenko"]:
            self.stain_extractor = fpixel.get_normalizer(
                cast("Literal['vahadane', 'macenko']", method),
            )

        self.preset_names = [
            "ruifrok",
            "macenko",
            "standard",
            "high_contrast",
            "h_heavy",
            "e_heavy",
            "dark",
            "light",
        ]

    def _get_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        """Get stain matrix based on selected method."""
        if self.method == "preset" and self.preset is not None:
            return fpixel.STAIN_MATRICES[self.preset]
        if self.method == "random_preset":
            random_preset = self.py_random.choice(self.preset_names)
            return fpixel.STAIN_MATRICES[random_preset]
        # vahadane or macenko
        self.stain_extractor.fit(img)
        return self.stain_extractor.stain_matrix_target

    def apply(
        self,
        img: np.ndarray,
        stain_matrix: np.ndarray,
        scale_factors: np.ndarray,
        shift_values: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply the HEStain transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the HEStain transform to.
            stain_matrix (np.ndarray): The stain matrix to use for the transform.
            scale_factors (np.ndarray): The scale factors to use for the transform.
            shift_values (np.ndarray): The shift values to use for the transform.
            **params (Any): Additional parameters for the transform.

        """
        non_rgb_error(img)
        return fpixel.apply_he_stain_augmentation(
            img=img,
            stain_matrix=stain_matrix,
            scale_factors=scale_factors,
            shift_values=shift_values,
            augment_background=self.augment_background,
        )

    @batch_transform("channel", has_batch_dim=True, has_depth_dim=False)
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the HEStain transform to a batch of images.

        Args:
            images (np.ndarray): The input images to apply the HEStain transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(images, **params)

    @batch_transform("channel", has_batch_dim=False, has_depth_dim=True)
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the HEStain transform to a batch of volumes.

        Args:
            volume (np.ndarray): The input volumes to apply the HEStain transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(volume, **params)

    @batch_transform("channel", has_batch_dim=True, has_depth_dim=True)
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the HEStain transform to a batch of volumes.

        Args:
            volumes (np.ndarray): The input volumes to apply the HEStain transform to.
            **params (Any): Additional parameters for the transform.

        """
        return self.apply(volumes, **params)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Generate parameters for the HEStain transform.

        Args:
            params (dict[str, Any]): The parameters of the transform.
            data (dict[str, Any]): The data to apply the transform to.

        """
        # Get stain matrix
        image = data["image"] if "image" in data else data["images"][0]

        stain_matrix = self._get_stain_matrix(image)

        # Generate random scaling and shift parameters for both H&E channels
        scale_factors = np.array(
            [
                self.py_random.uniform(*self.intensity_scale_range),
                self.py_random.uniform(*self.intensity_scale_range),
            ],
        )
        shift_values = np.array(
            [
                self.py_random.uniform(*self.intensity_shift_range),
                self.py_random.uniform(*self.intensity_shift_range),
            ],
        )

        return {
            "stain_matrix": stain_matrix,
            "scale_factors": scale_factors,
            "shift_values": shift_values,
        }
