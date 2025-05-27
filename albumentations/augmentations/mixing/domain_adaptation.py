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
    reference_images: Sequence[Any] | None
    read_fn: Callable[[Any], np.ndarray] | None
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


class BaseDomainAdaptation(ImageOnlyTransform):
    """Base class for domain adaptation transforms.

    Domain adaptation transforms modify source images to match the characteristics of a target domain.
    These transforms typically require an additional reference image or dataset from the target domain
    to extract style information or domain-specific features.

    This base class provides the framework for implementing various domain adaptation techniques such as
    color transfer, style transfer, frequency domain adaptation, or histogram matching.

    Args:
        reference_images (Sequence[Any] | None): Deprecated. Sequence of references to images from the target
            domain. Should be used with read_fn to load actual images. Prefer passing pre-loaded images via
            metadata_key.
        read_fn (Callable[[Any], np.ndarray] | None): Deprecated. Function to read an image from a reference.
            Should be used with reference_images.
        metadata_key (str): Key in the input data dictionary that contains pre-loaded target domain images.
        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Notes:
        - Subclasses should implement the `apply` method to perform the actual adaptation.
        - Use `targets_as_params` property to define what additional data your transform needs.
        - Override `get_params_dependent_on_data` to extract the target domain data.
        - Domain adaptation often requires per-sample auxiliary data, which should be passed
          through the main data dictionary rather than at initialization time.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Implement a simple color transfer domain adaptation transform
        >>> class SimpleColorTransfer(A.BaseDomainAdaptation):
        ...     class InitSchema(A.BaseTransformInitSchema):
        ...         intensity: float = Field(gt=0, le=1)
        ...         reference_key: str
        ...
        ...     def __init__(
        ...         self,
        ...         intensity: float = 0.5,
        ...         reference_key: str = "target_image",
        ...         p: float = 1.0
        ...     ):
        ...         super().__init__(p=p)
        ...         self.intensity = intensity
        ...         self.reference_key = reference_key
        ...
        ...     @property
        ...     def targets_as_params(self) -> list[str]:
        ...         return [self.reference_key]  # We need target domain image
        ...
        ...     def get_params_dependent_on_data(
        ...         self,
        ...         params: dict[str, Any],
        ...         data: dict[str, Any]
        ...     ) -> dict[str, Any]:
        ...         target_image = data.get(self.reference_key)
        ...         if target_image is None:
        ...             # Fallback if target image is not provided
        ...             return {"target_image": None}
        ...         return {"target_image": target_image}
        ...
        ...     def apply(
        ...         self,
        ...         img: np.ndarray,
        ...         target_image: np.ndarray = None,
        ...         **params
        ...     ) -> np.ndarray:
        ...         if target_image is None:
        ...             return img
        ...
        ...         # Simple color transfer implementation
        ...         # Calculate mean and std of source and target images
        ...         src_mean = np.mean(img, axis=(0, 1))
        ...         src_std = np.std(img, axis=(0, 1))
        ...         tgt_mean = np.mean(target_image, axis=(0, 1))
        ...         tgt_std = np.std(target_image, axis=(0, 1))
        ...
        ...         # Normalize source image
        ...         normalized = (img - src_mean) / (src_std + 1e-7)
        ...
        ...         # Scale by target statistics and blend with original
        ...         transformed = normalized * tgt_std + tgt_mean
        ...         transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        ...
        ...         # Blend the result based on intensity
        ...         result = cv2.addWeighted(img, 1 - self.intensity, transformed, self.intensity, 0)
        ...         return result
        >>>
        >>> # Usage example with a target image from a different domain
        >>> source_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> target_image = np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)  # Different domain image
        >>>
        >>> # Create the transform with the pipeline
        >>> transform = A.Compose([
        ...     SimpleColorTransfer(intensity=0.7, reference_key="target_img", p=1.0),
        ... ])
        >>>
        >>> # Apply the transform with the target image passed in the data dictionary
        >>> result = transform(image=source_image, target_img=target_image)
        >>> adapted_image = result["image"]  # Image with characteristics transferred from target domain

    """

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create sample images for demonstration
        >>> # Source image: dark image with low contrast
        >>> source_image = np.ones((100, 100, 3), dtype=np.uint8) * 50  # Dark gray image
        >>> source_image[30:70, 30:70] = 100  # Add slightly brighter square in center
        >>>
        >>> # Target image: higher brightness and contrast
        >>> target_image = np.ones((100, 100, 3), dtype=np.uint8) * 150  # Bright image
        >>> target_image[20:80, 20:80] = 200  # Add even brighter square
        >>>
        >>> # Initialize the histogram matching transform with custom settings
        >>> transform = A.Compose([
        ...     A.HistogramMatching(
        ...         blend_ratio=(0.7, 0.9),  # Control the strength of histogram matching
        ...         metadata_key="reference_imgs",  # Custom metadata key
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> # Apply the transform
        >>> result = transform(
        ...     image=source_image,
        ...     reference_imgs=[target_image]  # Pass reference image via metadata key
        ... )
        >>>
        >>> # Get the histogram-matched image
        >>> matched_image = result["image"]
        >>>
        >>> # The matched_image will have brightness and contrast similar to target_image
        >>> # while preserving the content of source_image
        >>>
        >>> # Multiple reference images can be provided:
        >>> ref_imgs = [
        ...     target_image,
        ...     np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)  # Another reference image
        ... ]
        >>> multiple_refs_result = transform(image=source_image, reference_imgs=ref_imgs)
        >>> # A random reference image from the list will be chosen for each transform application

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create sample images for demonstration
        >>> # Source image: synthetic or simulated image (e.g., from a rendered game environment)
        >>> source_img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> # Create a pattern in the source image
        >>> source_img[20:80, 20:80, 0] = 200  # Red square
        >>> source_img[40:60, 40:60, 1] = 200  # Green inner square
        >>>
        >>> # Target domain image: real-world image with different texture/frequency characteristics
        >>> # For this example, we'll create an image with different frequency patterns
        >>> target_img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> for i in range(100):
        ...     for j in range(100):
        ...         # Create a high-frequency pattern
        ...         target_img[i, j, 0] = ((i + j) % 8) * 30
        ...         target_img[i, j, 1] = ((i - j) % 8) * 30
        ...         target_img[i, j, 2] = ((i * j) % 8) * 30
        >>>
        >>> # Example 1: FDA with minimal adaptation (small beta value)
        >>> # This will subtly adjust the frequency characteristics
        >>> minimal_fda = A.Compose([
        ...     A.FDA(
        ...         beta_limit=(0.01, 0.05),  # Small beta range for subtle adaptation
        ...         metadata_key="target_domain",  # Custom metadata key
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> # Apply the transform with minimal adaptation
        >>> minimal_result = minimal_fda(
        ...     image=source_img,
        ...     target_domain=[target_img]  # Pass reference image via custom metadata key
        ... )
        >>> minimal_adapted_img = minimal_result["image"]
        >>>
        >>> # Example 2: FDA with moderate adaptation (medium beta value)
        >>> moderate_fda = A.Compose([
        ...     A.FDA(
        ...         beta_limit=(0.1, 0.2),  # Medium beta range
        ...         metadata_key="target_domain",
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> moderate_result = moderate_fda(image=source_img, target_domain=[target_img])
        >>> moderate_adapted_img = moderate_result["image"]
        >>>
        >>> # Example 3: FDA with strong adaptation (larger beta value)
        >>> strong_fda = A.Compose([
        ...     A.FDA(
        ...         beta_limit=(0.3, 0.5),  # Larger beta range (upper limit is MAX_BETA_LIMIT)
        ...         metadata_key="target_domain",
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> strong_result = strong_fda(image=source_img, target_domain=[target_img])
        >>> strong_adapted_img = strong_result["image"]
        >>>
        >>> # Example 4: Using multiple target domain images
        >>> # Creating a list of target domain images with different characteristics
        >>> target_imgs = [target_img]
        >>>
        >>> # Add another target image with different pattern
        >>> another_target = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> for i in range(100):
        ...     for j in range(100):
        ...         another_target[i, j, 0] = (i // 10) * 25
        ...         another_target[i, j, 1] = (j // 10) * 25
        ...         another_target[i, j, 2] = ((i + j) // 10) * 25
        >>> target_imgs.append(another_target)
        >>>
        >>> # Using default FDA settings with multiple target images
        >>> multi_target_fda = A.Compose([
        ...     A.FDA(p=1.0)  # Using default settings with default metadata_key="fda_metadata"
        ... ])
        >>>
        >>> # A random target image will be selected from the list for each application
        >>> multi_target_result = multi_target_fda(image=source_img, fda_metadata=target_imgs)
        >>> adapted_image = multi_target_result["image"]

    References:
        - FDA: https://github.com/YanchaoYang/FDA
        - FDA: https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create sample images for demonstration
        >>> # Source image: simulated image from domain A (e.g., medical scan from one scanner)
        >>> source_image = np.random.normal(100, 20, (100, 100, 3)).clip(0, 255).astype(np.uint8)
        >>>
        >>> # Reference image: image from domain B with different statistical properties
        >>> # (e.g., scan from a different scanner with different intensity distribution)
        >>> reference_image = np.random.normal(150, 30, (100, 100, 3)).clip(0, 255).astype(np.uint8)
        >>>
        >>> # Example 1: Using PCA transformation (default)
        >>> pca_transform = A.Compose([
        ...     A.PixelDistributionAdaptation(
        ...         transform_type="pca",
        ...         blend_ratio=(0.8, 1.0),  # Strong adaptation
        ...         metadata_key="reference_images",
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> # Apply the transform with the reference image
        >>> pca_result = pca_transform(
        ...     image=source_image,
        ...     reference_images=[reference_image]
        ... )
        >>>
        >>> # Get the adapted image
        >>> pca_adapted_image = pca_result["image"]
        >>>
        >>> # Example 2: Using StandardScaler transformation
        >>> standard_transform = A.Compose([
        ...     A.PixelDistributionAdaptation(
        ...         transform_type="standard",
        ...         blend_ratio=(0.5, 0.7),  # Moderate adaptation
        ...         metadata_key="reference_images",
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> standard_result = standard_transform(
        ...     image=source_image,
        ...     reference_images=[reference_image]
        ... )
        >>> standard_adapted_image = standard_result["image"]
        >>>
        >>> # Example 3: Using MinMaxScaler transformation
        >>> minmax_transform = A.Compose([
        ...     A.PixelDistributionAdaptation(
        ...         transform_type="minmax",
        ...         blend_ratio=(0.3, 0.5),  # Subtle adaptation
        ...         metadata_key="reference_images",
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> minmax_result = minmax_transform(
        ...     image=source_image,
        ...     reference_images=[reference_image]
        ... )
        >>> minmax_adapted_image = minmax_result["image"]
        >>>
        >>> # Example 4: Using multiple reference images
        >>> # When multiple reference images are provided, one is randomly selected for each transformation
        >>> multiple_references = [
        ...     reference_image,
        ...     np.random.normal(180, 25, (100, 100, 3)).clip(0, 255).astype(np.uint8),
        ...     np.random.normal(120, 40, (100, 100, 3)).clip(0, 255).astype(np.uint8)
        ... ]
        >>>
        >>> multi_ref_transform = A.Compose([
        ...     A.PixelDistributionAdaptation(p=1.0)  # Using default settings
        ... ])
        >>>
        >>> # Each time the transform is applied, it randomly selects one of the reference images
        >>> multi_ref_result = multi_ref_transform(
        ...     image=source_image,
        ...     pda_metadata=multiple_references  # Using the default metadata key
        ... )
        >>> adapted_image = multi_ref_result["image"]

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
