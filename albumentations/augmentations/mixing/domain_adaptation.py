"""Domain adaptation transforms for image augmentation.

This module provides transformations designed to bridge the domain gap between
datasets by adapting the style of an input image to match that of reference images
from a target domain. Adaptations are based on matching statistical properties
like histograms, frequency spectra, or overall pixel distributions.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Annotated, Any, Callable, Literal, cast

import cv2
import numpy as np
from pydantic import AfterValidator, field_validator, model_validator
from typing_extensions import Self

from albumentations.augmentations.mixing.domain_adaptation_functional import (
    adapt_pixel_distribution,
    apply_histogram,
    fourier_domain_adaptation,
)
from albumentations.augmentations.utils import read_rgb_image
from albumentations.core.pydantic import ZeroOneRangeType, check_range_bounds, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform

__all__ = [
    "FDA",
    "HistogramMatching",
    "PixelDistributionAdaptation",
]

MAX_BETA_LIMIT = 0.5


# Base class for Domain Adaptation Init Schema
class BaseDomainAdaptationInitSchema(BaseTransformInitSchema):
    reference_images: Sequence[Any] | None = None
    read_fn: Callable[[Any], np.ndarray] | None = None
    metadata_key: str

    @model_validator(mode="after")
    def _check_deprecated_args(self) -> Self:
        if self.reference_images is not None:
            warnings.warn(
                "'reference_images' and 'read_fn' arguments are deprecated. "
                "Please pass pre-loaded reference images "
                f"using the '{self.metadata_key}' key in the input data dictionary.",
                DeprecationWarning,
                stacklevel=3,  # Adjust stacklevel as needed
            )

            if self.read_fn is None:
                msg = "read_fn cannot be None when using the deprecated 'reference_images' argument."
                raise ValueError(msg)

        return self


# Base class for Domain Adaptation Transforms
class BaseDomainAdaptation(ImageOnlyTransform):
    # Pydantic schema for initialization arguments
    InitSchema: type[BaseDomainAdaptationInitSchema]

    def __init__(
        self,
        reference_images: Sequence[Any] | None,
        read_fn: Callable[[Any], np.ndarray] | None,
        metadata_key: str,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.metadata_key = metadata_key

    @property
    def targets_as_params(self) -> list[str]:
        return [self.metadata_key]

    def _get_reference_image(self, data: dict[str, Any]) -> np.ndarray:
        """Retrieves the reference image from metadata or deprecated arguments."""
        reference_image = None

        if metadata_images := data.get(self.metadata_key):
            if not isinstance(metadata_images, Sequence) or not metadata_images:
                raise ValueError(
                    f"Metadata key '{self.metadata_key}' should contain a non-empty sequence of numpy arrays.",
                )
            if not isinstance(metadata_images[0], np.ndarray):
                raise ValueError(
                    f"Images in metadata key '{self.metadata_key}' should be numpy arrays.",
                )
            reference_image = self.py_random.choice(metadata_images)

            if self.reference_images is not None:
                warnings.warn(
                    f"Both 'reference_images' (deprecated constructor argument) and metadata via "
                    f"'{self.metadata_key}' were provided. Prioritizing metadata.",
                    UserWarning,
                    stacklevel=3,  # Adjust stacklevel as needed
                )

        elif self.reference_images is not None:
            # Deprecation warning is handled by the InitSchema validator
            if self.read_fn is None:
                # This case should ideally be caught by InitSchema, but safety check
                msg = "read_fn cannot be None when using the deprecated 'reference_images' argument."
                raise ValueError(msg)
            ref_source = self.py_random.choice(self.reference_images)
            reference_image = self.read_fn(ref_source)
        else:
            raise ValueError(
                f"{self.__class__.__name__} requires reference images. Provide them via the `metadata_key` "
                f"'{self.metadata_key}' in the input data, or use the deprecated 'reference_images' argument.",
            )

        if reference_image is None:
            # Should not happen if logic above is correct, but safety check
            msg = "Could not obtain a reference image."
            raise RuntimeError(msg)

        return reference_image

    def to_dict_private(self) -> dict[str, Any]:
        """Convert the transform to a dictionary for serialization.

        Raises:
            NotImplementedError: Domain adaptation transforms cannot be reliably serialized
                                 when using metadata key or deprecated arguments.

        """
        if self.reference_images is not None:
            msg = (
                f"{self.__class__.__name__} cannot be reliably serialized when using the deprecated 'reference_images'."
            )
            raise NotImplementedError(msg)

        msg = (
            f"{self.__class__.__name__} cannot be reliably serialized due to its dependency "
            "on external data via metadata."
        )
        raise NotImplementedError(msg)


class HistogramMatching(BaseDomainAdaptation):
    """Adjust the pixel value distribution of an input image to match a reference image.

    This transform modifies the pixel intensities of the input image so that its histogram
    matches the histogram of a provided reference image. This process is applied independently
    to each channel of the image if it is multi-channel.

    Why use Histogram Matching?

    **Domain Adaptation:** Helps bridge the gap between images from different sources
    (e.g., different cameras, lighting conditions, synthetic vs. real data) by aligning
    their overall intensity and contrast characteristics.

    *Use Case Example:* Imagine you have labeled training images from one source (e.g., daytime photos,
    medical scans from Hospital A) but expect your model to work on images from a different
    source at test time (e.g., nighttime photos, scans from Hospital B). You might only have
    unlabeled images from the target (test) domain. HistogramMatching can be used to make your
    labeled training images resemble the *style* (intensity and contrast distribution) of the
    unlabeled target images. By training on these adapted images, your model may generalize
    better to the target domain without needing labels for it.

    How it works:
    The core idea is to map the pixel values of the input image such that its cumulative
    distribution function (CDF) matches the CDF of the reference image. This effectively
    reshapes the input image's histogram to resemble the reference's histogram.

    Args:
        metadata_key (str): Key in the input `data` dictionary to retrieve the reference image(s).
            The value should be a sequence (e.g., list) of numpy arrays (pre-loaded images).
            Default: "hm_metadata".
        blend_ratio (tuple[float, float]): Range for the blending factor between the original
            and the histogram-matched image. A value of 0 means the original image is returned,
            1 means the fully matched image is returned. A random value within this range [min, max]
            is sampled for each application. This allows for varying degrees of adaptation.
            Default: (0.5, 1.0).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - Requires at least one reference image to be provided via the `metadata_key` argument.
        - The `reference_images` and `read_fn` constructor arguments are deprecated.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> reference_image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> # Initialize transform using default metadata_key="hm_metadata"
        >>> transform = A.HistogramMatching(blend_ratio=(0.5, 1.0), p=1)
        >>> # Pass the reference image via the default metadata key
        >>> result = transform(image=image, hm_metadata=[reference_image])
        >>> matched_image = result["image"]

    References:
        Histogram Matching in scikit-image:
            https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html

    """

    class InitSchema(BaseDomainAdaptationInitSchema):
        blend_ratio: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 1)),
        ]

    def __init__(
        self,
        reference_images: Sequence[Any] | None = None,
        blend_ratio: tuple[float, float] = (0.5, 1.0),
        read_fn: Callable[[Any], np.ndarray] | None = read_rgb_image,
        metadata_key: str = "hm_metadata",
        p: float = 0.5,
    ):
        super().__init__(reference_images=reference_images, read_fn=read_fn, metadata_key=metadata_key, p=p)
        self.blend_ratio = blend_ratio

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Generate parameters for the transform based on input data.

        Args:
            params (dict[str, Any]): Parameters from the previous transform in the pipeline
            data (dict[str, Any]): Input data dictionary containing the image and metadata

        Returns:
            dict[str, Any]: Dictionary containing the reference image and blend ratio

        """
        reference_image = self._get_reference_image(data)
        return {
            "reference_image": reference_image,
            "blend_ratio": self.py_random.uniform(*self.blend_ratio),
        }

    def apply(
        self,
        img: np.ndarray,
        reference_image: np.ndarray,
        blend_ratio: float,
        **params: Any,
    ) -> np.ndarray:
        """Apply histogram matching to the input image.

        Args:
            img (np.ndarray): Input image to be transformed
            reference_image (np.ndarray): Reference image for histogram matching
            blend_ratio (float): Blending factor between the original and matched image
            **params (Any): Additional parameters

        Returns:
            np.ndarray: Transformed image with histogram matched to the reference image

        """
        return apply_histogram(img, reference_image, blend_ratio)


class FDA(BaseDomainAdaptation):
    """Fourier Domain Adaptation (FDA).

    Adapts the style of the input image to match the style of a reference image
    by manipulating their frequency components in the Fourier domain. This is
    particularly useful for unsupervised domain adaptation (UDA).

    Why use FDA?

    **Domain Adaptation:** FDA helps bridge the domain gap between source and target
    datasets (e.g., synthetic vs. real, day vs. night) by aligning their low-frequency
    Fourier spectrum components. This can improve model performance on the target domain
    without requiring target labels.

    *Use Case Example:* Imagine you have labeled training data acquired under certain conditions
    (e.g., images from Hospital A using a specific scanner) but need your model to perform well
    on data from a different distribution (e.g., unlabeled images from Hospital B with a different scanner).
    FDA can adapt the labeled source images to match the *style* (frequency characteristics)
    of the unlabeled target images, potentially improving the model's generalization to the
    target domain at test time.

    How it works:
    FDA operates in the frequency domain. It replaces the low-frequency components
    of the source image's Fourier transform with the low-frequency components from the
    reference (target domain) image's Fourier transform. The `beta_limit` parameter
    controls the size of the frequency window being swapped.

    Args:
        metadata_key (str): Key in the input `data` dictionary to retrieve the reference image(s).
            The value should be a sequence (e.g., list) of numpy arrays (pre-loaded images).
            Default: "fda_metadata".
        beta_limit (tuple[float, float] | float): Controls the extent of the low-frequency
            spectrum swap. A larger beta means more components are swapped. Corresponds to the L
            parameter in the original paper. Should be in the range [0, 0.5]. Sampling is uniform
            within the provided range [min, max]. Default: (0, 0.1).
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - Requires at least one reference image to be provided via the `metadata_key` argument.
        - The `reference_images` and `read_fn` constructor arguments are deprecated.

    References:
        - FDA: https://github.com/YanchaoYang/FDA
        - FDA: https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> reference_image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> # Initialize transform using default metadata_key="fda_metadata"
        >>> aug = A.Compose([A.FDA(p=1)])
        >>> # Pass the reference image via the default metadata key
        >>> result = aug(image=image, fda_metadata=[reference_image])

    """

    class InitSchema(BaseDomainAdaptationInitSchema):
        beta_limit: ZeroOneRangeType

        @field_validator("beta_limit")
        @classmethod
        def _check_ranges(cls, value: tuple[float, float]) -> tuple[float, float]:
            bounds = 0, MAX_BETA_LIMIT
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError(f"Values should be in the range {bounds} got {value} ")
            return value

    def __init__(
        self,
        reference_images: Sequence[Any] | None = None,
        beta_limit: tuple[float, float] | float = (0, 0.1),
        read_fn: Callable[[Any], np.ndarray] | None = read_rgb_image,
        metadata_key: str = "fda_metadata",
        p: float = 0.5,
    ):
        super().__init__(reference_images=reference_images, read_fn=read_fn, metadata_key=metadata_key, p=p)
        self.beta_limit = cast("tuple[float, float]", beta_limit)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Generate parameters for the transform based on input data."""
        target_image = self._get_reference_image(data)
        height, width = params["shape"][:2]

        # Resize the target image to match the input image dimensions
        target_image_resized = cv2.resize(target_image, dsize=(width, height))

        return {"target_image": target_image_resized, "beta": self.py_random.uniform(*self.beta_limit)}

    def apply(
        self,
        img: np.ndarray,
        target_image: np.ndarray,
        beta: float,
        **params: Any,
    ) -> np.ndarray:
        """Apply Fourier Domain Adaptation to the input image.

        Args:
            img (np.ndarray): Input image to be transformed
            target_image (np.ndarray): Target domain image for adaptation
            beta (float): Coefficient controlling the extent of frequency component swapping
            **params (Any): Additional parameters

        Returns:
            np.ndarray: Transformed image with adapted frequency components

        """
        return fourier_domain_adaptation(img, target_image, beta)


class PixelDistributionAdaptation(BaseDomainAdaptation):
    """Adapts the pixel value distribution of an input image to match a reference image
    using statistical transformations (PCA, StandardScaler, or MinMaxScaler).

    This transform aims to harmonize images from different domains by aligning their pixel-level
    statistical properties.

    Why use Pixel Distribution Adaptation?
    **Domain Adaptation:** Useful for aligning images across domains with differing pixel statistics
    (e.g., caused by different sensors, lighting, or post-processing).

    *Use Case Example:* Consider having labeled data from Scanner A and needing the model to perform
    well on unlabeled data from Scanner B, where images might have different overall brightness,
    contrast, or color biases. This transform can adapt the labeled images from Scanner A to
    mimic the pixel distribution *style* of the images from Scanner B, potentially improving
    generalization without needing labels for Scanner B data.

    How it works:
    1. A chosen statistical transform (`transform_type`) is fitted to both the input (source) image
       and the reference (target) image separately.
    2. The input image is transformed using the transform fitted on it (moving it to a standardized space).
    3. The inverse transform *fitted on the reference image* is applied to the result from step 2
       (moving the standardized input into the reference image's statistical space).
    4. The result is optionally blended with the original input image using `blend_ratio`.

    Args:
        metadata_key (str): Key in the input `data` dictionary to retrieve the reference image(s).
            The value should be a sequence (e.g., list) of numpy arrays (pre-loaded images).
            Default: "pda_metadata".
        blend_ratio (tuple[float, float]): Specifies the minimum and maximum blend ratio for mixing
            the adapted image with the original. A value of 0 means the original image is returned,
            1 means the fully adapted image is returned. A random value within this range [min, max]
            is sampled for each application. Default: (0.25, 1.0).
        transform_type (Literal["pca", "standard", "minmax"]): Specifies the type of statistical
            transformation to apply:
            - "pca": Principal Component Analysis.
            - "standard": StandardScaler (zero mean, unit variance).
            - "minmax": MinMaxScaler (scales to [0, 1] range).
            Default: "pca".
        p (float): The probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - Requires at least one reference image to be provided via the `metadata_key` argument.
        - The `reference_images` and `read_fn` constructor arguments are deprecated.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> reference_image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> # Initialize transform using default metadata_key="pda_metadata"
        >>> transform = A.PixelDistributionAdaptation(
        ...     blend_ratio=(0.5, 1.0),
        ...     transform_type="standard",
        ...     p=1.0
        ... )
        >>> # Pass the reference image via the default metadata key
        >>> result = transform(image=image, pda_metadata=[reference_image])
        >>> adapted_image = result["image"]

    References:
        Qudida: https://github.com/arsenyinfo/qudida

    """

    class InitSchema(BaseDomainAdaptationInitSchema):
        blend_ratio: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 1)),
        ]
        transform_type: Literal["pca", "standard", "minmax"]

    def __init__(
        self,
        reference_images: Sequence[Any] | None = None,
        blend_ratio: tuple[float, float] = (0.25, 1.0),
        read_fn: Callable[[Any], np.ndarray] | None = read_rgb_image,
        transform_type: Literal["pca", "standard", "minmax"] = "pca",
        metadata_key: str = "pda_metadata",
        p: float = 0.5,
    ):
        super().__init__(reference_images=reference_images, read_fn=read_fn, metadata_key=metadata_key, p=p)
        self.blend_ratio = blend_ratio
        self.transform_type = transform_type

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Get parameters for the transform."""
        reference_image = self._get_reference_image(data)
        return {
            "reference_image": reference_image,
            "blend_ratio": self.py_random.uniform(*self.blend_ratio),
        }

    def apply(self, img: np.ndarray, reference_image: np.ndarray, blend_ratio: float, **params: Any) -> np.ndarray:
        """Apply pixel distribution adaptation to the input image.

        Args:
            img (np.ndarray): Input image to be transformed
            reference_image (np.ndarray): Reference image for distribution adaptation
            blend_ratio (float): Blending factor between the original and adapted image
            **params (Any): Additional parameters

        Returns:
            np.ndarray: Transformed image with pixel distribution adapted to the reference image

        """
        return adapt_pixel_distribution(
            img,
            ref=reference_image,
            weight=blend_ratio,
            transform_type=self.transform_type,
        )
