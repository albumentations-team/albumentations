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


InterpolationType = Annotated[
    int, Field(default=cv2.INTER_LINEAR, description="Interpolation"), AfterValidator(check_valid_interpolation)
]
