from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Annotated, Self

from albumentations.augmentations.utils import MAX_VALUES_BY_DTYPE
from albumentations.core.transforms_interface import Interpolation, to_tuple
from albumentations.core.types import (
    ImageCompressionType,
    ImageMode,
    RainMode,
    ScaleFloatType,
    ScaleIntType,
    ScaleType,
    image_modes,
)

MAX_JPEG_QUALITY = 100

NUM_BITS_ARRAY_LENGTH = 3

BIG_INTEGER = MAX_VALUES_BY_DTYPE[np.uint32]


def check_and_convert_range(
    value: ScaleType, lower_bound: float, upper_bound: float, name: str, bias: float = 0
) -> Union[Tuple[float, float], Tuple[int, int]]:
    """Checks if the given value is within the specified bounds and converts single values to tuples.

    Args:
        value: The value to check and convert. Can be a single float or a tuple of floats.
        lower_bound: The lower bound for the range check.
        upper_bound: The upper bound for the range check.
        name: The name of the parameter being checked. Used for error messages.
        bias: The bias to use when converting single values to tuples.

    Returns:
        A tuple of two floats that is guaranteed to be within the specified bounds.

    Raises:
        ValueError: If the value is outside the bounds or if the tuple values are not ordered correctly.
    """
    if isinstance(value, (float, int)):
        if not (lower_bound <= value <= upper_bound):
            raise ValueError(f"{name} must be within [{lower_bound}, {upper_bound}] for single value inputs.")
        return to_tuple(value, bias=bias)
    if isinstance(value, (tuple, list)):
        if not all(lower_bound <= x <= upper_bound for x in value):
            raise ValueError(f"All values in {name} must be within [{lower_bound}, {upper_bound}] for tuple inputs.")
        if value[0] > value[1]:
            raise ValueError(f"{name} tuple values must be ordered as (min, max). Got: {value}")
        return value

    raise TypeError(f"{name} must be a float or a tuple of floats.")


class BaseTransformConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    always_apply: bool = Field(default=False, description="Always apply the transform")
    p: Annotated[float, Field(default=0.5, description="Probability of applying the transform", ge=0, le=1)]


class RandomGridShuffleConfig(BaseTransformConfig):
    grid: Tuple[int, int] = Field(default=(3, 3), description="Size of grid for splitting image")

    @field_validator("grid")
    @classmethod
    def check_grid_dimensions(cls, value: Tuple[int, int]) -> Tuple[int, int]:
        if not all(isinstance(dim, int) and dim > 0 for dim in value):
            raise ValueError(f"Grid dimensions must be positive integers. Got {value}")
        return value


class NormalizeConfig(BaseTransformConfig):
    mean: Union[float, Sequence[float]] = Field(
        default=(0.485, 0.456, 0.406), description="Mean values for normalization"
    )
    std: Union[float, Sequence[float]] = Field(
        default=(0.229, 0.224, 0.225), description="Standard deviation values for normalization"
    )
    max_pixel_value: float = Field(default=255.0, description="Maximum possible pixel value")

    @field_validator("mean", "std")
    @classmethod
    def validate_sequences(cls, v: Union[float, Tuple[float, ...]]) -> Union[float, Tuple[float, ...]]:
        if isinstance(v, float):
            return (v,)
        if not isinstance(v, Sequence) or not all(isinstance(x, (float, int)) for x in v):
            msg = "Mean and std must be either a float or a sequence of floats."
            raise ValueError(msg)
        return v


class ImageCompressionConfig(BaseTransformConfig):
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


class RandomSnowConfig(BaseTransformConfig):
    snow_point_lower: float = Field(default=0.1, description="Lower bound of the amount of snow", ge=0, le=1)
    snow_point_upper: float = Field(default=0.3, description="Upper bound of the amount of snow", ge=0, le=1)
    brightness_coeff: float = Field(default=2.5, description="Brightness coefficient, must be >= 0", ge=0)

    @model_validator(mode="after")
    def validate_snow_points(self) -> Self:
        if self.snow_point_lower > self.snow_point_upper:
            msg = "snow_point_lower must be less than or equal to snow_point_upper."
            raise ValueError(msg)
        return self


class RandomGravelConfig(BaseTransformConfig):
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


class RandomRainConfig(BaseTransformConfig):
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


class RandomFogConfig(BaseTransformConfig):
    fog_coef_lower: float = Field(default=0.3, description="Lower limit for fog intensity coefficient", ge=0, le=1)
    fog_coef_upper: float = Field(default=1, description="Upper limit for fog intensity coefficient", ge=0, le=1)
    alpha_coef: float = Field(default=0.08, description="Transparency of the fog circles", ge=0, le=1)

    @model_validator(mode="after")
    def validate_fog_coefficients(self) -> Self:
        if self.fog_coef_lower > self.fog_coef_upper:
            msg = "fog_coef_upper must be greater than or equal to fog_coef_lower."
            raise ValueError(msg)
        return self


class RandomSunFlareConfig(BaseTransformConfig):
    flare_roi: Tuple[float, float, float, float] = Field(
        default=(0, 0, 1, 0.5), description="Region of the image where flare will appear"
    )
    angle_lower: float = Field(default=0, description="Lower bound for the angle", ge=0, le=1)
    angle_upper: float = Field(default=1, description="Upper bound for the angle", ge=0, le=1)
    num_flare_circles_lower: int = Field(default=6, description="Lower limit for the number of flare circles", ge=0)
    num_flare_circles_upper: int = Field(default=10, description="Upper limit for the number of flare circles", gt=0)
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


class RandomShadowConfig(BaseTransformConfig):
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


class RandomToneCurveConfig(BaseTransformConfig):
    scale: float = Field(
        default=0.1,
        description="Standard deviation of the normal distribution used to sample random distances",
        ge=0,
        le=1,
    )

    @model_validator(mode="after")
    def validate_scale(self) -> Self:
        if not (0 <= self.scale <= 1):
            raise ValueError(f"Scale must be in range [0, 1]. Got: {self.scale}")
        return self


class HueSaturationValueConfig(BaseTransformConfig):
    hue_shift_limit: ScaleType = Field(default=(-20, 20), description="Range for changing hue.")
    sat_shift_limit: ScaleType = Field(default=(-30, 30), description="Range for changing saturation.")
    val_shift_limit: ScaleType = Field(default=(-20, 20), description="Range for changing value.")

    @field_validator("hue_shift_limit", "sat_shift_limit", "val_shift_limit")
    @classmethod
    def convert_to_tuple(cls, v: Any) -> Union[Tuple[int, int], Tuple[float, float]]:
        return to_tuple(v)


class SolarizeConfig(BaseTransformConfig):
    threshold: Annotated[ScaleType, Field(default=(128, 128), description="Range for solarizing threshold.")]

    @field_validator("threshold")
    @classmethod
    def convert_to_tuple(cls, threshold: Any) -> Union[Tuple[int, int], Tuple[float, float]]:
        if isinstance(threshold, (int, float)):
            return to_tuple(threshold, low=threshold)

        return to_tuple(threshold, low=0)


class PosterizeConfig(BaseTransformConfig):
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


class EqualizeConfig(BaseTransformConfig):
    mode: Annotated[ImageMode, Field(default="cv", description="Equalization mode, 'cv' or 'pil'")]
    by_channels: Annotated[bool, Field(default=True, description="Equalize channels separately if True")]
    mask: Annotated[
        Optional[Union[np.ndarray, Callable[..., Any]]],
        Field(default=None, description="Mask to apply for equalization"),
    ]
    mask_params: Annotated[Sequence[str], Field(default=[], description="Parameters for mask function")]

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        if value not in image_modes:
            raise ValueError(f"Unsupported equalization mode. Supports: ['cv', 'pil']. Got: {value}")
        return value


class RGBShiftConfig(BaseTransformConfig):
    r_shift_limit: Annotated[ScaleType, Field(default=20, description="Range for changing values for the red channel.")]
    g_shift_limit: Annotated[
        ScaleType, Field(default=20, description="Range for changing values for the green channel.")
    ]
    b_shift_limit: Annotated[
        ScaleType, Field(default=20, description="Range for changing values for the blue channel.")
    ]

    @field_validator("r_shift_limit", "g_shift_limit", "b_shift_limit")
    @classmethod
    def convert_to_tuple(cls, v: ScaleType) -> Tuple[float, float]:
        return cast(Tuple[float, float], to_tuple(v))


class RandomBrightnessContrastConfig(BaseTransformConfig):
    brightness_limit: ScaleFloatType = Field(default=0.2, description="Factor range for changing brightness.")
    contrast_limit: ScaleFloatType = Field(default=0.2, description="Factor range for changing contrast.")
    brightness_by_max: bool = Field(default=True, description="Adjust brightness by image dtype maximum if True.")

    @field_validator("brightness_limit", "contrast_limit")
    @classmethod
    def validate_and_convert(cls, v: Any) -> Tuple[float, float]:
        return to_tuple(v)


class GaussNoiseConfig(BaseTransformConfig):
    var_limit: ScaleType = Field(default=(10.0, 50.0), description="Variance range for noise.")
    mean: float = Field(default=0, description="Mean of the noise.")
    per_channel: bool = Field(default=True, description="Apply noise per channel.")

    @field_validator("var_limit")
    @classmethod
    def validate_var_limit(cls, var_limit: ScaleType) -> Tuple[float, float]:
        if isinstance(var_limit, (int, float)):
            if var_limit < 0:
                msg = "var_limit should be non negative."
                raise ValueError(msg)
            return (0, var_limit)
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0 or var_limit[1] < 0:
                msg = "var_limit values should be non negative."
                raise ValueError(msg)
            return cast(Tuple[float, float], tuple(var_limit))

        msg = f"Expected var_limit type to be one of (int, float, tuple, list), got {type(var_limit)}"
        raise TypeError(msg)


class ISONoiseConfig(BaseTransformConfig):
    color_shift: Tuple[float, float] = Field(
        default=(0.01, 0.05),
        description=(
            "Variance range for color hue change. " "Measured as a fraction of 360 degree Hue angle in HLS colorspace."
        ),
    )
    intensity: Tuple[float, float] = Field(
        default=(0.1, 0.5), description="Multiplicative factor that control strength of color and luminance noise."
    )


class CLAHEConfig(BaseTransformConfig):
    clip_limit: ScaleFloatType = Field(default=(1.0, 4.0), description="Upper threshold value for contrast limiting.")
    tile_grid_size: Tuple[int, int] = Field(default=(8, 8), description="Size of grid for histogram equalization.")

    @field_validator("clip_limit")
    @classmethod
    def validate_clip_limit(cls, v: Any) -> Tuple[float, float]:
        return to_tuple(v, low=1)


class FromFloatConfig(BaseTransformConfig):
    dtype: str = Field(default="uint16", description="Data type of the output.")
    max_value: Optional[float] = Field(default=None, description="Maximum possible input value.")
    p: float = Field(default=1.0, description="Probability of applying the transform", ge=0, le=1)

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        supported_dtypes = ["uint8", "uint16", "float32", "float64"]  # Add other supported dtypes as necessary
        if value not in supported_dtypes:
            raise ValueError(f"Unsupported dtype. Supports: {supported_dtypes}. Got: {value}")
        return value


class DownscaleConfig(BaseTransformConfig):
    scale_min: float = Field(default=0.25, ge=0, le=1, description="Lower bound on the image scale.")
    scale_max: float = Field(default=0.25, ge=0, lt=1, description="Upper bound on the image scale.")
    interpolation: Optional[Union[int, Interpolation, Dict[str, int]]] = Field(
        default_factory=lambda: Interpolation(downscale=cv2.INTER_NEAREST, upscale=cv2.INTER_NEAREST),
        description="CV2 interpolation method or a dictionary specifying downscale and upscale methods.",
    )
    always_apply: bool = Field(default=False, description="Always apply the transform.")
    p: float = Field(default=0.5, ge=0, le=1, description="Probability of applying the transform.")

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

        msg = "Interpolation must be an int, Interpolation instance, or dict specifying downscale and upscale methods."
        raise ValueError(msg)


class MultiplicativeNoiseConfig(BaseTransformConfig):
    multiplier: ScaleFloatType = Field(default=(0.9, 1.1), description="Multiplier for image values.")
    per_channel: bool = Field(default=False, description="Apply multiplier per channel.")
    elementwise: bool = Field(default=False, description="Apply multiplier element-wise to pixels.")

    @model_validator(mode="after")
    def check_validity(self) -> Self:
        if isinstance(self.multiplier, (list, tuple)) and self.multiplier[0] >= self.multiplier[1]:
            msg = "multiplier[0] should be less than multiplier[1]"
            raise ValueError(msg)
        if isinstance(self.multiplier, (float, int)):
            self.multiplier = to_tuple(self.multiplier, self.multiplier)
        return self


class FancyPCAConfig(BaseTransformConfig):
    alpha: float = Field(default=0.1, description="Scale for perturbing the eigen vectors and values", ge=0)


class ColorJitterConfig(BaseTransformConfig):
    brightness: Annotated[ScaleFloatType, Field(default=0.2, description="Range for jittering brightness.")]
    contrast: Annotated[ScaleFloatType, Field(default=0.2, description="Range for jittering contrast.")]
    saturation: Annotated[ScaleFloatType, Field(default=0.2, description="Range for jittering saturation.")]
    hue: Annotated[ScaleFloatType, Field(default=0.2, description="Range for jittering hue.")]

    @field_validator("brightness", "contrast", "saturation", "hue")
    @classmethod
    def check_ranges(cls, value: ScaleFloatType, info: ValidationInfo) -> Tuple[float, float]:
        if info.field_name == "hue":
            bounds = -0.5, 0.5
            bias = 0
        elif info.field_name in ["brightness", "contrast", "saturation"]:
            bounds = 0, float("inf")
            bias = 1
        else:
            raise ValueError(f"Unsupported field name: {info.field_name}")

        return check_and_convert_range(value, *bounds, str(info.field_name), bias=bias)


class SharpenConfig(BaseTransformConfig):
    alpha: Tuple[float, float] = Field(default=(0.2, 0.5), description="Visibility of the sharpened image.")
    lightness: Tuple[float, float] = Field(default=(0.5, 1.0), description="Lightness of the sharpened image.")

    @field_validator("alpha", "lightness")
    @classmethod
    def check_ranges(cls, value: Tuple[float, float], info: ValidationInfo) -> Tuple[float, float]:
        if info.field_name == "alpha":
            bounds = 0.0, 1.0
        elif info.field_name == "lightness":
            bounds = 0.0, float("inf")

        return check_and_convert_range(value, *bounds, str(info.field_name))


class EmbossConfig(BaseTransformConfig):
    alpha: Tuple[float, float] = Field(default=(0.2, 0.5), description="Visibility of the embossed image.")
    strength: Tuple[float, float] = Field(default=(0.2, 0.7), description="Strength of the embossing.")

    @field_validator("alpha", "strength")
    @classmethod
    def check_ranges(cls, value: Tuple[float, float], info: ValidationInfo) -> Tuple[float, float]:
        if info.field_name == "alpha":
            bounds = 0.0, 1.0
        elif info.field_name == "strength":
            bounds = 0.0, float("inf")

        return check_and_convert_range(value, *bounds, str(info.field_name))


class SuperpixelsConfig(BaseTransformConfig):
    p_replace: ScaleFloatType = Field(
        default=0.1,
        description="Probability of replacing segment pixels with their average color.",
    )
    n_segments: Annotated[ScaleIntType, Field(default=100, description="Target number of superpixels to generate.")]
    max_size: Optional[int] = Field(default=128, ge=1, description="Maximum image size for the transformation.")
    interpolation: int = Field(default=cv2.INTER_LINEAR, description="Interpolation algorithm used during scaling.")

    @field_validator("p_replace")
    @classmethod
    def check_p_replace(cls, v: ScaleFloatType, info: ValidationInfo) -> Tuple[float, float]:
        bounds = 0, 1
        result = to_tuple(v, v)
        return check_and_convert_range(result, *bounds, str(info.field_name))

    @field_validator("n_segments")
    @classmethod
    def check_n_segments(cls, v: ScaleIntType, info: ValidationInfo) -> Tuple[int, int]:
        bounds = 1, BIG_INTEGER
        result = to_tuple(v, v)
        return cast(Tuple[int, int], check_and_convert_range(result, *bounds, str(info.field_name)))


class TemplateTransformConfig(BaseTransformConfig):
    templates: Union[np.ndarray, Sequence[np.ndarray]] = Field(..., description="Images as template for transform.")
    img_weight: ScaleFloatType = Field(default=0.5, description="Weight for input image.")
    template_weight: ScaleFloatType = Field(default=0.5, description="Weight for template.")
    template_transform: Optional[Callable[..., Any]] = Field(
        default=None, description="Transformation object applied to template."
    )
    name: Optional[str] = Field(default=None, description="Name of transform, used only for deserialization.")

    @field_validator("img_weight", "template_weight")
    @classmethod
    def check_weight(cls, v: ScaleFloatType, info: ValidationInfo) -> Tuple[float, float]:
        bounds = 0, 1
        result = to_tuple(v, v)
        return check_and_convert_range(result, *bounds, str(info.field_name))

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


class RingingOvershootConfig(BaseTransformConfig):
    blur_limit: ScaleIntType = Field(default=(7, 15), description="Maximum kernel size for sinc filter.")
    cutoff: ScaleFloatType = Field(default=(np.pi / 4, np.pi / 2), description="Cutoff frequency range in radians.")

    @field_validator("blur_limit")
    @classmethod
    def check_blur_limit(cls, v: ScaleIntType, info: ValidationInfo) -> Tuple[int, int]:
        bounds = 3, BIG_INTEGER
        return cast(Tuple[int, int], check_and_convert_range(v, *bounds, str(info.field_name)))

    @field_validator("cutoff")
    @classmethod
    def check_cutoff(cls, v: ScaleFloatType, info: ValidationInfo) -> Tuple[float, float]:
        bounds = 0, np.pi
        return check_and_convert_range(v, *bounds, str(info.field_name))


class UnsharpMaskConfig(BaseTransformConfig):
    blur_limit: ScaleIntType = Field(default=(3, 7), description="Maximum Gaussian kernel size for blurring.")
    sigma_limit: ScaleFloatType = Field(description="Gaussian kernel standard deviation.")
    alpha: ScaleFloatType = Field(default=(0.2, 0.5), description="Visibility of the sharpened image.")
    threshold: int = Field(default=10, ge=0, le=255, description="Threshold for limiting sharpening.")

    @field_validator("blur_limit")
    @classmethod
    def check_blur_limit(cls, v: ScaleIntType, info: ValidationInfo) -> Tuple[int, int]:
        # Convert and check for odd values if not 0
        bounds = 0, float("inf")
        result = to_tuple(v, low=3)
        if (result[0] != 0 and result[0] % 2 != 1) or (result[1] != 0 and result[1] % 2 != 1):
            msg = "UnsharpMask supports only odd blur limits or 0."
            raise ValueError(msg)
        return cast(Tuple[int, int], check_and_convert_range(result, *bounds, str(info.field_name)))

    @field_validator("sigma_limit")
    @classmethod
    def check_sigma_limit(cls, v: ScaleIntType, info: ValidationInfo) -> Tuple[float, float]:
        bounds = 0, float("inf")
        result = to_tuple(v, 0)
        return check_and_convert_range(result, *bounds, str(info.field_name))

    @field_validator("alpha")
    @classmethod
    def check_alpha(cls, v: ScaleIntType, info: ValidationInfo) -> Tuple[float, float]:
        bounds = 0, 1
        result = to_tuple(v, 0)
        return check_and_convert_range(result, *bounds, str(info.field_name))
