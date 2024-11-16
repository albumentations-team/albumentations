from __future__ import annotations

from numbers import Real
from typing import Annotated, cast
from warnings import warn

import cv2
from pydantic import AfterValidator, Field, field_validator
from typing_extensions import Literal

from albumentations.augmentations.dropout.coarse_dropout import Erasing
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.geometric.transforms import Affine, HorizontalFlip, Perspective, VerticalFlip
from albumentations.augmentations.transforms import ImageCompression, InvertImg, ToGray
from albumentations.core.pydantic import InterpolationType, check_0plus, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.types import PAIR, ColorType, ScaleFloatType, Targets

__all__ = [
    "RandomJPEG",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomGrayscale",
    "RandomPerspective",
    "RandomAffine",
    "RandomRotation",
    "RandomErasing",
    "RandomInvert",
]


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
        It is provided for compatibility with torchvision and Kornia APIs to make
        it easier to use Albumentations alongside these libraries.

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
        It is provided for compatibility with torchvision and Kornia APIs to make
        it easier to use Albumentations alongside these libraries.

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
        super().__init__(p=p)


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
        It is provided for compatibility with torchvision and Kornia APIs to make
        it easier to use Albumentations alongside these libraries.

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
        super().__init__(p=p)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


class RandomPerspective(Perspective):
    """Perform a random perspective transformation with a given probability.

    This transform is an alias for Perspective, provided for compatibility with
    torchvision and Kornia APIs. For new code, it is recommended to use
    albumentations.Perspective directly.

    Args:
        distortion_scale (float): argument to control the degree of distortion.
            Range: [0, 1]. Default: 0.5.
        interpolation (int): interpolation method. Default: cv2.INTER_LINEAR.
        fill (int | float | list[int] | list[float]): padding value if border_mode is cv2.BORDER_CONSTANT.
            Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    Note:
        This is a direct alias for albumentations.Perspective transform.
        It is provided for compatibility with torchvision and Kornia APIs to make
        it easier to use Albumentations alongside these libraries.

        For more flexibility, consider using albumentations.Perspective directly, which supports:
        - Additional border modes
        - Different interpolation methods
        - Different fill values
        - Fill value for mask
        - Different interpolation methods for mask
        - Scale and aspect ratio adjustments
        - Fit output options

    Example:
        >>> transform = A.RandomPerspective(distortion_scale=0.5, p=0.5)
        >>> # Consider using instead:
        >>> transform = A.Perspective(scale=(0.05, 0.5), p=0.5)

    References:
        - torchvision: https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomPerspective.html
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomPerspective
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        distortion_scale: float = Field(ge=0, le=1)
        fill: ColorType
        interpolation: InterpolationType

    def __init__(
        self,
        distortion_scale: float = 0.5,
        interpolation: int = cv2.INTER_LINEAR,
        fill: ColorType = 0,
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        warn(
            "RandomPerspective is an alias for Perspective transform. "
            "Consider using Perspective directly from albumentations.Perspective.",
            UserWarning,
            stacklevel=2,
        )
        # Convert distortion_scale to scale range expected by Perspective
        super().__init__(
            scale=(distortion_scale, distortion_scale),
            interpolation=interpolation,
            pad_val=fill,
            p=p,
        )
        self.distortion_scale = distortion_scale
        self.interpolation = interpolation
        self.fill = fill

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "distortion_scale", "interpolation", "fill"


class RandomAffine(Affine):
    """Perform a random affine transformation with center invariance.

    This transform is an alias for Affine, provided for compatibility with
    torchvision and Kornia APIs. For new code, it is recommended to use
    albumentations.Affine directly.

    Args:
        degrees (float | tuple[float, float]): Range of degrees to select from.
            If degrees is a single number, the range will be (-degrees, +degrees).
            Set to None to deactivate rotations. Default: 0.
        translate (tuple[float, float]): Tuple of maximum absolute fraction for horizontal
            and vertical translations (dx, dy). For example (0.3, 0.3) means translation is randomly
            sampled from (-0.3 * width, 0.3 * width) and (-0.3 * height, 0.3 * height).
            Default: (0, 0).
        scale (tuple[float, float]): Scaling factor interval, e.g (0.8, 1.2).
            Scale is randomly sampled from the range. Default: (1, 1).
        shear (float | tuple[float, float] | tuple[float, float, float, float]):
            Range of degrees for shear transformation. If a single number, shear parallel
            to x axis in range (-shear, +shear). If 2 values, x-axis shear in (shear[0], shear[1]).
            If 4 values, x-axis and y-axis shear in (shear[0], shear[1]) and (shear[2], shear[3]).
            Default: 0.
        interpolation (int): interpolation method. Default: cv2.INTER_LINEAR.
        fill (int | float | list[int] | list[float]): padding value if border_mode is cv2.BORDER_CONSTANT.
            Default: 0.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    Note:
        This transform is implemented as a subset of Affine with fixed parameters:
        - No scale per axis
        - No additional transform parameters

        It is provided for compatibility with torchvision and Kornia APIs to make
        it easier to use Albumentations alongside these libraries.

        For more flexibility, consider using albumentations.Affine directly, which supports:
        - Additional border modes
        - Different interpolation methods
        - Independent control of each transformation parameter
        - Scale per axis
        - Additional transform parameters
        - Mask fill value
        - Different interpolation methods for mask
        - Different padding modes

    Example:
        >>> transform = A.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1))
        >>> # Consider using instead:
        >>> transform = A.Affine(rotate=(-30, 30), translate_percent=(-0.1, 0.1), scale=(0.9, 1.1))

    References:
        - torchvision: https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomAffine.html
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomAffine
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        degrees: ScaleFloatType
        translate: tuple[float, float]
        scale: tuple[float, float] | fgeometric.XYFloatDict
        shear: ScaleFloatType | tuple[float, float, float, float] | fgeometric.XYFloatDict
        interpolation: InterpolationType
        fill: ColorType

        @field_validator("shear", mode="after")
        @classmethod
        def process_shear(
            cls,
            value: ScaleFloatType | tuple[float, float, float, float],
        ) -> fgeometric.XYFloatDict:
            """Convert shear parameter to internal format."""
            if isinstance(value, Real):
                return {"x": (-value, value), "y": (-value, value)}
            if isinstance(value, (tuple, list)):
                if len(value) == PAIR:
                    return {"x": (-value[0], value[1]), "y": (-value[0], value[1])}
                return {"x": (value[0], value[1]), "y": (value[2], value[3])}  # type: ignore[misc]
            if isinstance(value, dict):
                return value
            raise TypeError(f"Invalid shear value: {value}")

    def __init__(
        self,
        degrees: float | tuple[float, float] = 0,
        translate: tuple[float, float] = (0, 0),
        scale: tuple[float, float] | fgeometric.XYFloatDict = (1, 1),
        shear: float | tuple[float, float] | tuple[float, float, float, float] | fgeometric.XYFloatDict = 0,
        interpolation: int = cv2.INTER_LINEAR,
        fill: ColorType = 0,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        warn(
            "RandomAffine is an alias for Affine transform. "
            "Consider using Affine directly from albumentations.Affine.",
            UserWarning,
            stacklevel=2,
        )

        self.degrees = degrees
        self.translate = translate
        self.fill = fill
        self.shear = cast(fgeometric.XYFloatDict, shear)
        self.interpolation = interpolation
        self.scale = cast(fgeometric.XYFloatDict, scale)

        # Convert torchvision parameters to Albumentations format
        rotate = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        translate_percent = {"x": (-translate[0], translate[0]), "y": (-translate[1], translate[1])}

        super().__init__(
            rotate=rotate,
            translate_percent=cast(fgeometric.XYFloatScale, translate_percent),
            scale=cast(fgeometric.XYFloatScale, scale),
            shear=cast(fgeometric.XYFloatScale, shear),
            interpolation=interpolation,
            cval=fill,
            p=p,
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "degrees", "translate", "scale", "shear", "interpolation", "fill"


class RandomRotation(Affine):
    """Rotate the input randomly by an angle.

    This transform is an alias for Affine, provided for compatibility with
    torchvision and Kornia APIs. For new code, it is recommended to use
    albumentations.Affine directly.

    Args:
        degrees (float | tuple[float, float]): Range of degrees to select from.
            If degrees is a single number, the range will be (-degrees, +degrees).
            Default: 0.
        interpolation (int): interpolation method. Default: cv2.INTER_LINEAR.
        expand (bool): If True, expands the output image to make it large enough to hold
            the entire rotated image. If False, make the output image the same size
            as the input. Default: False.
        fill (int | float | list[int] | list[float]): padding value if border_mode is cv2.BORDER_CONSTANT.
            Default: 0.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        This is a direct alias for albumentations.Affine transform with rotation only.
        It is provided for compatibility with torchvision and Kornia APIs to make
        it easier to use Albumentations alongside these libraries.

        For more flexibility, consider using albumentations.Affine directly, which supports:
        - Additional border modes
        - Different interpolation methods
        - Independent control of each transformation parameter
        - Scale per axis
        - Additional transform parameters
        - Mask fill value
        - Different interpolation methods for mask
        - Different padding modes

    Example:
        >>> transform = A.RandomRotation(degrees=30)
        >>> # Consider using instead:
        >>> transform = A.Affine(rotate=(-30, 30))

    References:
        - torchvision: https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomRotation.html
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomRotate
    """

    class InitSchema(BaseTransformInitSchema):
        degrees: ScaleFloatType
        interpolation: InterpolationType
        expand: bool
        fill: ColorType

    def __init__(
        self,
        degrees: float | tuple[float, float] = 0,
        interpolation: int = cv2.INTER_LINEAR,
        expand: bool = False,
        fill: ColorType = 0,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        warn(
            "RandomRotation is an alias for Affine transform. "
            "Consider using Affine directly from albumentations.Affine.",
            UserWarning,
            stacklevel=2,
        )

        self.degrees = degrees
        self.fill = fill
        self.interpolation = interpolation
        self.expand = expand

        super().__init__(
            rotate=degrees,
            interpolation=interpolation,
            cval=fill,
            fit_output=expand,
            p=p,
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "degrees", "interpolation", "expand", "fill"


class RandomErasing(Erasing):
    """Randomly select a rectangle region in the input image and erase its pixels.
    This is an alias for Erasing transform with torchvision-compatible parameters.

    Args:
        p (float): Probability of applying the transform. Default: 0.5.
        scale (tuple[float, float]): Range of proportion of erased area against input image.
            Default: (0.02, 0.33).
        ratio (tuple[float, float]): Range of aspect ratio of erased area.
            Default: (0.3, 3.3).
        value (float | tuple[float, float, float] | Literal["random"]): Erasing value.
            - If float: Used to erase all pixels
            - If tuple: Used to erase R, G, B channels respectively
            - If "random": Erase each pixel with random values
            Default: 0.0

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        This is a direct alias for albumentations.Erasing transform with torchvision-compatible parameters.
        It is provided for compatibility with torchvision API to make it easier to use Albumentations
        alongside torchvision.

        For more flexibility, consider using albumentations.Erasing directly, which supports:
        - Additional erasing modes
        - Per-channel erasing values
        - Custom fill values for masks
        - Different erasing patterns
        - Independent control of erasing parameters
        - Additional transform parameters

    Example:
        >>> import albumentations as A
        >>> transform = A.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        >>> transformed = transform(image=image)
    """

    class InitSchema(BaseTransformInitSchema):
        scale: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_0plus)]
        ratio: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_0plus)]
        value: float | tuple[float, ...] | Literal["random"]

    def __init__(
        self,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (0.3, 3.3),
        value: float | tuple[float, ...] | Literal["random"] = 0.0,
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        warn(
            "RandomErasing is an alias for Erasing transform. "
            "Consider using Erasing directly from albumentations.Erasing.",
            UserWarning,
            stacklevel=2,
        )
        # Convert torchvision parameters to Erasing parameters
        super().__init__(
            p=p,
            scale=scale,
            ratio=ratio,
            fill_value=value,
        )
        self.value = value

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "scale", "ratio", "value"


class RandomInvert(InvertImg):
    """Invert the input image values randomly with a given probability.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        This is a direct alias for albumentations.Erasing transform with torchvision-compatible parameters.
        It is provided for compatibility with torchvision and kornia API to make it easier to use Albumentations
        alongside torchvision.

    Example:
        >>> transform = A.RandomInvert(p=0.5)
        >>> # Consider using instead:
        >>> transform = A.InvertImg(p=0.5)

    References:
        - torchvision: https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomInvert.html
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomInvert
    """

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(
        self,
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        warn(
            "RandomInvert is an alias for InvertImg transform. "
            "Consider using InvertImg directly from albumentations.InvertImg.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(p=p)
