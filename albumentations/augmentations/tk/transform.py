from __future__ import annotations

from numbers import Real
from typing import Annotated, cast
from warnings import warn

import cv2
from pydantic import AfterValidator, Field, field_validator
from typing_extensions import Literal

from albumentations.augmentations.blur.transforms import GaussianBlur, MedianBlur
from albumentations.augmentations.dropout.channel_dropout import ChannelDropout
from albumentations.augmentations.dropout.coarse_dropout import Erasing
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.geometric.transforms import Affine, HorizontalFlip, Perspective, VerticalFlip
from albumentations.augmentations.transforms import (
    CLAHE,
    ColorJitter,
    Equalize,
    ImageCompression,
    InvertImg,
    PlanckianJitter,
    Posterize,
    RandomBrightnessContrast,
    Solarize,
    ToGray,
)
from albumentations.core.pydantic import (
    InterpolationType,
    check_0plus,
    check_01,
    check_1plus,
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.types import PAIR, ColorType, ScaleFloatType, ScaleIntType, Targets

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
    "RandomHue",
    "RandomClahe",
    "RandomContrast",
    "RandomBrightness",
    "RandomChannelDropout",
    "RandomEqualize",
    "RandomGaussianBlur",
    "RandomPlanckianJitter",
    "RandomMedianBlur",
    "RandomSolarize",
    "RandomPosterize",
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
        >>> transform = A.Compose([A.RandomRotation(degrees=30)])
        >>> # Consider using instead:
        >>> transform = A.Compose([A.Affine(rotate=(-30, 30))])

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
        >>> transform = A.Compose([A.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))])
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

    Number of channels:
        Any

    Note:
        This is a direct alias for albumentations.InvertImg transform.
        It is provided for compatibility with torchvision and Kornia API to make it easier to use Albumentations
        alongside torchvision.

    Example:
        >>> transform = A.Compose([A.RandomInvert(p=0.5)])
        >>> # Consider using instead:
        >>> transform = A.Compose([A.InvertImg(p=0.5)])

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


class RandomHue(ColorJitter):
    """Randomly adjust the hue of the input image.

    This transform is a specialized version of ColorJitter that only adjusts hue.
    For more flexibility (brightness, contrast, saturation), consider using
    ColorJitter directly.

    Args:
        hue (tuple[float, float]): Range for changing hue.
            Values should be in the range [-0.5, 0.5]. Default: (0, 0)
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        1, 3

    Note:
        This is a specialized version of albumentations.ColorJitter transform.
        It is provided for compatibility with Kornia APIs.
        For more flexibility, including brightness, contrast, and saturation adjustments,
        consider using ColorJitter directly.

    Example:
        >>> transform = A.Compose([A.RandomHue(hue=(-0.2, 0.2), p=0.5)])
        >>> # Consider using instead:
        >>> transform = A.Compose([A.ColorJitter(hue=(-0.2, 0.2), p=0.5)])

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomHue
    """

    class InitSchema(BaseTransformInitSchema):
        hue: Annotated[tuple[float, float], AfterValidator(nondecreasing)]

    def __init__(
        self,
        hue: tuple[float, float] = (0, 0),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomHue is a specialized version of ColorJitter. "
            "For more flexibility (brightness, contrast, saturation adjustments), "
            "consider using ColorJitter directly from albumentations.ColorJitter.",
            UserWarning,
            stacklevel=2,
        )

        super().__init__(
            hue=hue,
            brightness=(1, 1),  # No brightness change
            contrast=(1, 1),  # No contrast change
            saturation=(1, 1),  # No saturation change
            p=p,
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("hue",)


class RandomClahe(CLAHE):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) randomly.

    This transform is an alias for CLAHE, provided for compatibility with
    Kornia API. For new code, it is recommended to use albumentations.CLAHE directly.

    Args:
        clip_limit (tuple[float, float]): Clipping limit. Higher values give more contrast. Default: (1, 4).
        tile_grid_size (tuple[int, int]): Size of grid for histogram equalization.
            Larger grid size gives more contrast. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        1, 3

    Note:
        This is a direct alias for albumentations.CLAHE transform.
        It is provided for compatibility with Kornia API to make it easier
        to use Albumentations alongside Kornia.

    Example:
        >>> transform = A.Compose([A.RandomClahe(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5)])
        >>> # Consider using instead:
        >>> transform = A.Compose([A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5)])

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomClahe
    """

    class InitSchema(BaseTransformInitSchema):
        clip_limit: Annotated[tuple[float, float], AfterValidator(nondecreasing)]
        tile_grid_size: Annotated[tuple[int, int], AfterValidator(check_1plus)]

    def __init__(
        self,
        clip_limit: float | tuple[float, float] = (1, 4),
        tile_grid_size: tuple[int, int] = (8, 8),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomClahe is an alias for CLAHE transform. Consider using CLAHE directly from albumentations.CLAHE.",
            UserWarning,
            stacklevel=2,
        )

        if isinstance(clip_limit, (int, float)):
            clip_limit = (0, clip_limit)

        super().__init__(
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            p=p,
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("clip_limit", "tile_grid_size")


class RandomContrast(RandomBrightnessContrast):
    """Randomly adjust the contrast of the input image.

    This transform is a specialized version of RandomBrightnessContrast that only adjusts contrast.

    Args:
        contrast (tuple[float, float]): Factor range for changing contrast.
            - A factor of 0.0 gives black image
            - A factor of 1.0 gives the original image
            - A factor > 1.0 increases contrast
            - A factor < 1.0 decreases contrast
            Default: (1, 1)
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        This is a specialized version of albumentations.RandomBrightnessContrast transform.
        It is provided for compatibility with Kornia APIs.
        For more flexibility, including brightness adjustments, consider using
        RandomBrightnessContrast directly.

    Example:
        >>> transform = A.Compose([A.RandomContrast(contrast=(0.8, 1.2), p=0.5)])
        >>> # Consider using instead:
        >>> transform = A.Compose([A.RandomBrightnessContrast(contrast=(0.8, 1.2), brightness_limit=0, p=0.5)])

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomContrast
    """

    class InitSchema(BaseTransformInitSchema):
        contrast: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_0plus)]

    def __init__(
        self,
        contrast: tuple[float, float] = (1, 1),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomContrast is a specialized version of RandomBrightnessContrast. "
            "For more flexibility (including brightness adjustments), "
            "consider using RandomBrightnessContrast directly from albumentations.RandomBrightnessContrast.",
            UserWarning,
            stacklevel=2,
        )

        super().__init__(
            contrast_limit=(contrast[0] - 1, contrast[1] - 1),
            brightness_limit=0,  # No brightness change
            p=p,
        )

        self.contrast = contrast

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("contrast",)


class RandomBrightness(RandomBrightnessContrast):
    """Randomly adjust the brightness of the input image.

    This transform is a specialized version of RandomBrightnessContrast that only adjusts brightness.

    Args:
        brightness (tuple[float, float]): Range for changing brightness.
            - A factor of 0.0 gives a black image
            - A factor of 1.0 gives the original image
            - A factor > 1.0 increases brightness
            - A factor < 1.0 decreases brightness
            Default: (0.8, 1.2)
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        This is a specialized version of albumentations.RandomBrightnessContrast transform.
        For more flexibility (including contrast adjustments), consider using
        RandomBrightnessContrast directly.

    Example:
        >>> transform = A.Compose([A.RandomBrightness(brightness=(0.8, 1.2), p=0.5)])
        >>> # Consider using instead:
        >>> transform = A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.8, 1.2), contrast_limit=0, p=0.5)])

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomBrightness
    """

    class InitSchema(BaseTransformInitSchema):
        brightness: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_0plus)]

    def __init__(
        self,
        brightness: tuple[float, float] = (1, 1),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomBrightness is a specialized version of RandomBrightnessContrast. "
            "For more flexibility (including contrast adjustments), "
            "consider using RandomBrightnessContrast directly from albumentations.RandomBrightnessContrast.",
            UserWarning,
            stacklevel=2,
        )

        super().__init__(
            brightness_limit=brightness,
            contrast_limit=0,  # No contrast change
            p=p,
        )

        self.brightness = brightness

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("brightness",)


class RandomChannelDropout(ChannelDropout):
    """Randomly drop channels in the input image.

    This transform is an alias for ChannelDropout, provided for compatibility with
    Kornia API. For new code, it is recommended to use albumentations.ChannelDropout directly.

    Args:
        num_drop_channels (int): Number of channels to drop randomly. Default: 1.
        fill_value (float): Value to fill the dropped channels. Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        This transform is a specialized version of ChannelDropout provided primarily for compatibility
        with Kornia's API.

        Consider using ChannelDropout directly for:
         - the ability to sample a number of channels to drop from a range


    Example:
        >>> # RandomChannelDropout way (Kornia compatibility)
        >>> transform = A.Compose([A.RandomChannelDropout(num_drop_channels=2, fill_value=128)])
        >>> # Preferred ChannelDropout way
        >>> transform = A.Compose([A.ChannelDropout(channel_drop_range=(2, 2), fill_value=128)])

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomChannelDropout
    """

    class InitSchema(BaseTransformInitSchema):
        num_drop_channels: int
        fill_value: float

    def __init__(
        self,
        num_drop_channels: int = 1,
        fill_value: float = 0,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomChannelDropout is an alias for ChannelDropout transform. "
            "Consider using ChannelDropout directly from albumentations.ChannelDropout.",
            UserWarning,
            stacklevel=2,
        )

        super().__init__(
            channel_drop_range=(num_drop_channels, num_drop_channels),
            fill_value=fill_value,
            p=p,
        )
        self.num_drop_channels = num_drop_channels

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "num_drop_channels", "fill_value"


class RandomEqualize(Equalize):
    """Equalize given image using histogram equalization.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        This transform is a specialized case of Equalize with fixed parameters:
        - Uses OpenCV's equalization method (mode='cv')
        - Applies equalization to each channel independently (by_channels=True)
        - Does not support masking

        For more flexibility, consider using Equalize directly which provides:
        - Choice between OpenCV and Pillow equalization methods
        - Option to equalize luminance channel only
        - Support for masked equalization
        - Additional parameters for fine-tuning the equalization process

    Example:
        >>> # RandomEqualize way (Kornia compatibility)
        >>> transform = A.Compose([A.RandomEqualize(p=0.5)])
        >>> # Equivalent Equalize way with full functionality
        >>> transform = A.Compose([A.Equalize(mode='cv', by_channels=True, p=0.5)])

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomEqualize
    """

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(
        self,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomEqualize is a specialized version of Equalize transform. "
            "Consider using Equalize directly from albumentations.Equalize for more flexibility.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(
            mode="cv",
            by_channels=True,
            mask=None,
            mask_params=(),
            p=p,
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


class RandomGaussianBlur(GaussianBlur):
    """Apply Gaussian blur to the input image.

    This transform is an alias for GaussianBlur, provided for compatibility with
    Kornia API. For new code, it is recommended to use albumentations.GaussianBlur directly.

    Args:
        kernel_size (int | tuple[int, int]): Gaussian kernel size.
            - If a single int is provided, kernel_size will be (kernel_size, kernel_size).
            - If a tuple of two ints is provided, it defines the kernel size range.
            Must be odd and greater than or equal to 3. Default: (3, 7).
        sigma (float | tuple[float, float]): Gaussian kernel standard deviation.
            - If a single float is provided, sigma will be (sigma, sigma).
            - If a tuple of two floats is provided, it defines the sigma range.
            If 0, it will be computed as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8.
            Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        This transform is a direct alias for GaussianBlur with identical functionality.
        The only difference is in parameter naming to match Kornia's API:
        - 'kernel_size' instead of 'blur_limit'
        - 'sigma' instead of 'sigma_limit'

        For new projects, it is recommended to use GaussianBlur directly as it
        provides a more consistent interface within the Albumentations ecosystem.

    Example:
        >>> # RandomGaussianBlur way (Kornia compatibility)
        >>> transform = A.RandomGaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2))
        >>> # Preferred GaussianBlur way
        >>> transform = A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2))

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomGaussianBlur
    """

    class InitSchema(BaseTransformInitSchema):
        kernel_size: ScaleIntType
        sigma: ScaleFloatType

    def __init__(
        self,
        kernel_size: ScaleIntType = (3, 7),
        sigma: ScaleFloatType = 0,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomGaussianBlur is an alias for GaussianBlur transform. "
            "Consider using GaussianBlur directly from albumentations.GaussianBlur.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(
            blur_limit=kernel_size,
            sigma_limit=sigma,
            p=p,
            always_apply=always_apply,
        )
        self.kernel_size = kernel_size
        self.sigma = sigma

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "kernel_size", "sigma"


class RandomPlanckianJitter(PlanckianJitter):
    """Apply Planckian Jitter to simulate realistic illumination changes.

    This transform is a specialized version of PlanckianJitter that uses blackbody
    radiation model to create physically accurate color temperature variations.
    It is provided for compatibility with Kornia's API.

    Args:
        mode (Literal["blackbody", "cied"]): The illuminant model to use.
            - "blackbody": Uses Planckian blackbody radiation model
            - "cied": Uses CIE standard illuminant D series
            Default: "blackbody"
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        3

    Note:
        This transform is a specialized version of PlanckianJitter with fixed parameters:
        - Uses uniform sampling method
        - Default temperature ranges:
            * For blackbody mode: [3000K, 15000K]
            * For CIED mode: [4000K, 15000K]

        For more flexibility, consider using PlanckianJitter directly which provides:
        - Custom temperature ranges
        - Choice of sampling methods (uniform or gaussian)
        - More control over the color temperature distribution

    Example:
        >>> # RandomPlanckianJitter way (Kornia compatibility)
        >>> transform = A.RandomPlanckianJitter(mode="blackbody")
        >>> # Equivalent PlanckianJitter way with full functionality
        >>> transform = A.PlanckianJitter(mode="blackbody", sampling_method="uniform")

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomPlanckianJitter
        - [ZBTvdW22] Zhu, M., et al. "Simple and effective temperature-based color augmentation."
          arXiv preprint arXiv:2211.05108 (2022).
    """

    class InitSchema(BaseTransformInitSchema):
        mode: Literal["blackbody", "cied"]

    def __init__(
        self,
        mode: Literal["blackbody", "cied"] = "blackbody",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomPlanckianJitter is a specialized version of PlanckianJitter transform. "
            "Consider using PlanckianJitter directly from albumentations.PlanckianJitter for more flexibility.",
            UserWarning,
            stacklevel=2,
        )
        # Use default temperature ranges based on mode
        temperature_range = None  # Will use defaults from PlanckianJitter

        super().__init__(
            mode=mode,
            temperature_limit=temperature_range,
            sampling_method="uniform",
            p=p,
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("mode",)


class RandomMedianBlur(MedianBlur):
    """Apply median blur to the input image.

    This transform is a specialized version of MedianBlur that uses fixed kernel size.
    It is provided for compatibility with Kornia API. For new code, it is recommended
    to use albumentations.MedianBlur directly.

    Args:
        kernel_size (tuple[int, int]): Aperture linear size.
            Must be odd and greater than 1. Default: (3, 3).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        This transform is a specialized version of MedianBlur with fixed parameters:
        - Uses fixed kernel size instead of sampling from a range
        - No support for dynamic kernel size selection

        For more flexibility, consider using MedianBlur directly which provides:
        - Ability to sample kernel size from a range
        - Support for both fixed and dynamic kernel sizes
        - More consistent interface within Albumentations

    Example:
        >>> # RandomMedianBlur way (Kornia compatibility)
        >>> transform = A.RandomMedianBlur(kernel_size=(3, 3))
        >>> # Preferred MedianBlur way with sampling
        >>> transform = A.MedianBlur(blur_limit=(3, 7))
        >>> # Or with fixed size
        >>> transform = A.MedianBlur(blur_limit=3)

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomMedianBlur
    """

    class InitSchema(BaseTransformInitSchema):
        kernel_size: Annotated[tuple[int, int], AfterValidator(check_1plus)]

    def __init__(
        self,
        kernel_size: tuple[int, int] = (3, 3),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomMedianBlur is a specialized version of MedianBlur with a probability parameter. "
            "Consider using MedianBlur directly from albumentations.MedianBlur.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(
            blur_limit=kernel_size,
            p=p,
        )
        self.kernel_size = kernel_size

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("kernel_size",)


class RandomSolarize(Solarize):
    """Invert all pixel values above a threshold.

    This transform is an alias for Solarize, provided for compatibility with
    Kornia API naming convention, but using Albumentations' parameter format.

    Args:
        thresholds (tuple[float, float]): Range for solarizing threshold as a fraction
            of maximum value. The thresholds should be in the range [0, 1] and will be multiplied by the
            maximum value of the image type (255 for uint8 images or 1.0 for float images).
            Default: (0.1, 0.1).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        This transform differs from Kornia's RandomSolarize in parameter format:
        - Uses normalized thresholds [0, 1] for both uint8 and float32 images
        - No support for post-solarization brightness addition

        For brightness adjustment, use composition with RandomBrightness:
        ```python
        A.Compose([
            A.RandomSolarize(thresholds=0.1),
            A.RandomBrightness(limit=0.1)
        ])
        ```

    Example:
        >>> # RandomSolarize with fixed threshold at 10% of max value
        >>> transform = A.RandomSolarize(thresholds=0.1)  # 25.5 for uint8, 0.1 for float32
        >>> # RandomSolarize with threshold range
        >>> transform = A.RandomSolarize(thresholds=(0.1, 0.1))

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomSolarize
    """

    class InitSchema(BaseTransformInitSchema):
        thresholds: Annotated[tuple[float, float], AfterValidator(check_01), AfterValidator(nondecreasing)]

    def __init__(
        self,
        thresholds: tuple[float, float] = (0.1, 0.1),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomSolarize is an alias for Solarize transform. "
            "Consider using Solarize directly from albumentations.Solarize. "
            "Note: parameter format differs from Kornia's implementation."
            "For brightness addition, use composition with RandomBrightness.",
            UserWarning,
            stacklevel=2,
        )

        super().__init__(
            threshold_range=thresholds,
            p=p,
        )
        self.thresholds = thresholds

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("thresholds",)


class RandomPosterize(Posterize):
    """Reduce the number of bits for each color channel.

    This transform is an alias for Posterize, provided for compatibility with
    Kornia API. For new code, it is recommended to use albumentations.Posterize directly.

    Args:
        num_bits (tuple[int, int]): Range for number of bits to keep for each channel.
            Values should be in range [0, 8] for uint8 images.
            Default: (3, 3).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        This transform is a direct alias for Posterize with identical functionality.
        For new projects, it is recommended to use Posterize directly as it
        provides a more consistent interface within the Albumentations ecosystem.

        For float32 images:
        1. Image is converted to uint8 (multiplied by 255 and clipped)
        2. Posterization is applied
        3. Image is converted back to float32 (divided by 255)

    Example:
        >>> # RandomPosterize way (Kornia compatibility)
        >>> transform = A.RandomPosterize(num_bits=(3, 3))  # Fixed 3 bits per channel
        >>> transform = A.RandomPosterize(num_bits=(3, 5))  # Random from 3 to 5 bits
        >>> # Preferred Posterize way
        >>> transform = A.Posterize(bits=(3, 3))
        >>> transform = A.Posterize(bits=(3, 5))

    References:
        - Kornia: https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomPosterize
    """

    class InitSchema(BaseTransformInitSchema):
        num_bits: Annotated[tuple[int, int], AfterValidator(check_range_bounds(0, 8)), AfterValidator(nondecreasing)]

    def __init__(
        self,
        num_bits: tuple[int, int] = (3, 3),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        warn(
            "RandomPosterize is an alias for Posterize transform. "
            "Consider using Posterize directly from albumentations.Posterize.",
            UserWarning,
            stacklevel=2,
        )

        super().__init__(
            num_bits=num_bits,
            p=p,
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("num_bits",)
