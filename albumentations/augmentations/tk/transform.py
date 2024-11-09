from __future__ import annotations

from typing import Annotated
from warnings import warn

from pydantic import AfterValidator

from albumentations.augmentations.geometric.transforms import HorizontalFlip, VerticalFlip
from albumentations.augmentations.transforms import ImageCompression
from albumentations.core.pydantic import check_0plus, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.types import Targets

__all__ = ["RandomJPEG", "RandomHorizontalFlip", "RandomVerticalFlip"]


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
        jpeg_quality: Annotated[tuple[int, int], AfterValidator(check_0plus), AfterValidator(nondecreasing)]

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


class RandomHorizontalFlip(HorizontalFlip):
    """Horizontally flip the input randomly with a given probability.

    This transform is an alias for HorizontalFlip, provided for compatibility with
    torchvision and Kornia APIs. For new code, it is recommended to use
    albumentations.HorizontalFlip directly.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        This is a direct alias for albumentations.HorizontalFlip transform.
        It is provided to make migration from torchvision and Kornia easier by
        maintaining API compatibility.

    Example:
        >>> transform = A.RandomHorizontalFlip(p=0.5)
        >>> # Consider using instead:
        >>> transform = A.HorizontalFlip(p=0.5)

    References:
        - torchvision: https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomHorizontalFlip
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(
        self,
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        warn(
            "RandomHorizontalFlip is an alias for HorizontalFlip transform. "
            "Consider using HorizontalFlip directly from albumentations.HorizontalFlip. ",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(p=p, always_apply=always_apply)


class RandomVerticalFlip(VerticalFlip):
    """Vertically flip the input randomly with a given probability.

    This transform is an alias for VerticalFlip, provided for compatibility with
    torchvision and Kornia APIs. For new code, it is recommended to use
    albumentations.VerticalFlip directly.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        This is a direct alias for albumentations.VerticalFlip transform.
        It is provided to make migration from torchvision and Kornia easier by
        maintaining API compatibility.

    Example:
        >>> transform = A.RandomVerticalFlip(p=0.5)
        >>> # Consider using instead:
        >>> transform = A.VerticalFlip(p=0.5)

    References:
        - torchvision: https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomVerticalFlip
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomVerticalFlip
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(
        self,
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        warn(
            "RandomVerticalFlip is an alias for VerticalFlip transform. "
            "Consider using VerticalFlip directly from albumentations.VerticalFlip. ",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(p=p, always_apply=always_apply)
