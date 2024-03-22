from typing import Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Annotated, Self

from albumentations.core.types import ImageCompressionType

MAX_JPEG_QUALITY = 100


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
