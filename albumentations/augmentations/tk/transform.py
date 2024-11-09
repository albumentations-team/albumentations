from typing import Annotated
from warnings import warn

from pydantic import AfterValidator

from albumentations.augmentations.transforms import ImageCompression
from albumentations.core.pydantic import check_0plus, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema

__all__ = ["RandomJPEG"]


class RandomJPEG(ImageCompression):
    """Apply random JPEG compression to an image.

    This transform applies JPEG compression with randomly sampled quality factor.
    Lower quality values result in higher compression and more artifacts.

    This is a specialized version of ImageCompression configured for JPEG only.
    For more advanced use cases (e.g., other compression types, custom parameters),
    consider using ImageCompression directly.

    Args:
        jpeg_quality (tuple[int, int]): The range of compression rates to be applied. Range: [0, 100]
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        This transform is implemented as a subset of ImageCompression with fixed parameters:
        - JPEG compression only

        For more flexibility, including:
        - Other compression types (PNG, WebP)
        - Custom compression parameters
        Consider using albumentations.ImageCompression directly.

    References:
        - Kornia implementation: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomJPEG
    """

    class InitSchema(BaseTransformInitSchema):
        jpeg_quality: Annotated[tuple[float, float], AfterValidator(check_0plus), AfterValidator(nondecreasing)]

    def __init__(
        self,
        jpeg_quality: tuple[int, int] = (50, 50),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        warn(
            "RandomJPEG is a specialized version of ImageCompression. "
            "For more flexibility (other compression types, custom parameters), "
            "consider using ImageCompression directly from albumentations.ImageCompression.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(
            quality_range=jpeg_quality,
            compression_type="jpeg",
            always_apply=always_apply,
            p=p,
        )
        self.jpeg_quality = jpeg_quality

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("jpeg_quality",)
