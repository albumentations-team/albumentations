import cv2
from pydantic import Field
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

valid_interpolations = [
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_CUBIC,
    cv2.INTER_AREA,
    cv2.INTER_LANCZOS4,
    cv2.INTER_LINEAR_EXACT,
    cv2.INTER_MAX,
]


def check_valid_interpolation(value: int) -> int:
    if value not in valid_interpolations:
        raise ValueError(f"Interpolation should be one of {valid_interpolations}, got {value} instead")
    return value


InterpolationType = Annotated[int, Field(description="Interpolation"), AfterValidator(check_valid_interpolation)]

valid_border_modes = [
    cv2.BORDER_CONSTANT,  # Adds a constant colored border. The value should be given as next argument.
    cv2.BORDER_REPLICATE,  # Replicates the last element.
    cv2.BORDER_REFLECT,  # Mirrors the image border without repeating the last element.
    cv2.BORDER_WRAP,  # Wraps the image around like a tiled texture.
    cv2.BORDER_REFLECT_101,  # Also known as cv2.BORDER_DEFAULT. Mirrors the image border, repeating the last element.
    cv2.BORDER_TRANSPARENT,  # Makes the border transparent
]


def check_valid_border_modes(value: int) -> int:
    if value not in valid_interpolations:
        raise ValueError(f"Border mode should be one of {valid_border_modes}, got {value} instead")
    return value


BorderModeType = Annotated[int, Field(description="Border Mode"), AfterValidator(check_valid_border_modes)]

ProbabilityType = Annotated[float, Field(description="Probability of applying the transform", ge=0, le=1)]
