from __future__ import annotations

from typing import Annotated
from warnings import warn

from pydantic import AfterValidator

from albumentations.augmentations.geometric.transforms import HorizontalFlip, VerticalFlip
from albumentations.augmentations.transforms import ImageCompression, ToGray
from albumentations.core.pydantic import check_0plus, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.types import Targets

__all__ = ["RandomJPEG", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomGrayscale"]


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
        - torchvision: https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomHorizontalFlip.html
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
        - torchvision: https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomVerticalFlip.html
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


class RandomGrayscale(ToGray):
    """Randomly convert the input image to grayscale with a given probability.

    This transform is an alias for ToGray, provided for compatibility with
    torchvision and Kornia APIs. For new code, it is recommended to use
    albumentations.ToGray directly.

    Uses ITU-R 601-2 luma transform: grayscale = 0.299R + 0.587G + 0.114B
    (same as torchvision.transforms.RandomGrayscale).

    Args:
        p (float): probability that image should be converted to grayscale. Default: 0.1.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        This is a direct alias for albumentations.ToGray transform with method="weighted_average".
        It is provided to make migration from torchvision and Kornia easier by
        maintaining API compatibility.

        For more flexibility, consider using albumentations.ToGray directly, which supports:
        - Multiple grayscale conversion methods ("weighted_average", "from_lab", "desaturation", etc.)
        - Some methods that work with any number of channels ("desaturation", "average", "max", "pca")
        - Different perceptual and performance trade-offs

        This transform specifically:
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b
        - Unlike torchvision, single-channel inputs are not supported
        - Uses the same ITU-R 601-2 weights (0.299, 0.587, 0.114) as torchvision

    Example:
        >>> transform = A.RandomGrayscale(p=0.1)
        >>> # Consider using instead:
        >>> transform = A.ToGray(p=0.1, method="weighted_average")

    References:
        - torchvision: https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomGrayscale.html
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomGrayscale
        - ITU-R BT.601: https://en.wikipedia.org/wiki/Rec._601
    """

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(
        self,
        p: float = 0.1,
        always_apply: bool | None = None,
    ):
        warn(
            "RandomGrayscale is an alias for ToGray transform. "
            "Consider using ToGray directly from albumentations.ToGray.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(p=p, always_apply=always_apply)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()
