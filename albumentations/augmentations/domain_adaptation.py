from __future__ import annotations

import random
from typing import Any, Callable, Literal, Sequence, Tuple, cast

import cv2
import numpy as np
from albucore.functions import to_float
from albucore.utils import clip, is_grayscale_image, is_multispectral_image
from pydantic import AfterValidator, field_validator
from typing_extensions import Annotated

from albumentations.augmentations.domain_adaptation_functional import (
    adapt_pixel_distribution,
    apply_histogram,
    fourier_domain_adaptation,
)
from albumentations.augmentations.utils import read_rgb_image
from albumentations.core.pydantic import ZeroOneRangeType, check_01, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform
from albumentations.core.types import ScaleFloatType

__all__ = [
    "HistogramMatching",
    "FDA",
    "PixelDistributionAdaptation",
]

MAX_BETA_LIMIT = 0.5


class HistogramMatching(ImageOnlyTransform):
    """Adjust the pixel values of an input image to match the histogram of a reference image.

    This transform applies histogram matching, a technique that modifies the distribution of pixel
    intensities in the input image to closely resemble that of a reference image. This process is
    performed independently for each channel in multi-channel images, provided both the input and
    reference images have the same number of channels.

    Histogram matching is particularly useful for:
    - Normalizing images from different sources or captured under varying conditions.
    - Preparing images for feature matching or other computer vision tasks where consistent
      tone and contrast are important.
    - Simulating different lighting or camera conditions in a controlled manner.

    Args:
        reference_images (Sequence[Any]): A sequence of reference image sources. These can be
            file paths, URLs, or any objects that can be converted to images by the `read_fn`.
        blend_ratio (tuple[float, float]): Range for the blending factor between the original
            and the matched image. Must be two floats between 0 and 1, where:
            - 0 means no blending (original image is returned)
            - 1 means full histogram matching
            A random value within this range is chosen for each application.
            Default: (0.5, 1.0)
        read_fn (Callable[[Any], np.ndarray]): A function that takes an element from
            `reference_images` and returns a numpy array representing the image.
            Default: read_rgb_image (reads image file from disk)
        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - This transform cannot be directly serialized due to its dependency on external image data.
        - The effectiveness of the matching depends on the similarity between the input and reference images.
        - For best results, choose reference images that represent the desired tone and contrast.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> reference_image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> transform = A.HistogramMatching(
        ...     reference_images=[reference_image],
        ...     blend_ratio=(0.5, 1.0),
        ...     read_fn=lambda x: x,
        ...     p=1
        ... )
        >>> result = transform(image=image)
        >>> matched_image = result["image"]

    References:
        - Histogram Matching in scikit-image:
          https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html
    """

    class InitSchema(BaseTransformInitSchema):
        reference_images: Sequence[Any]
        blend_ratio: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_01)]
        read_fn: Callable[[Any], np.ndarray]

    def __init__(
        self,
        reference_images: Sequence[Any],
        blend_ratio: tuple[float, float] = (0.5, 1.0),
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.blend_ratio = blend_ratio

    def apply(
        self: np.ndarray,
        img: np.ndarray,
        reference_image: np.ndarray,
        blend_ratio: float,
        **params: Any,
    ) -> np.ndarray:
        return apply_histogram(img, reference_image, blend_ratio)

    def get_params(self) -> dict[str, np.ndarray]:
        return {
            "reference_image": self.read_fn(random.choice(self.reference_images)),
            "blend_ratio": random.uniform(*self.blend_ratio),
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "reference_images", "blend_ratio", "read_fn"

    def to_dict_private(self) -> dict[str, Any]:
        msg = "HistogramMatching can not be serialized."
        raise NotImplementedError(msg)


class FDA(ImageOnlyTransform):
    """Fourier Domain Adaptation (FDA) for simple "style transfer" in the context of unsupervised domain adaptation
    (UDA). FDA manipulates the frequency components of images to reduce the domain gap between source
    and target datasets, effectively adapting images from one domain to closely resemble those from another without
    altering their semantic content.

    This transform is particularly beneficial in scenarios where the training (source) and testing (target) images
    come from different distributions, such as synthetic versus real images, or day versus night scenes.
    Unlike traditional domain adaptation methods that may require complex adversarial training, FDA achieves domain
    alignment by swapping low-frequency components of the Fourier transform between the source and target images.
    This technique has shown to improve the performance of models on the target domain, particularly for tasks
    like semantic segmentation, without additional training for domain invariance.

    The 'beta_limit' parameter controls the extent of frequency component swapping, with lower values preserving more
    of the original image's characteristics and higher values leading to more pronounced adaptation effects.
    It is recommended to use beta values less than 0.3 to avoid introducing artifacts.

    Args:
        reference_images (Sequence[Any]): Sequence of objects to be converted into images by `read_fn`. This typically
            involves paths to images that serve as target domain examples for adaptation.
        beta_limit (tuple[float, float] | float): Coefficient beta from the paper, controlling the swapping extent of
            frequency components. If one value is provided beta will be sampled from uniform
            distribution [0, beta_limit]. Values should be less than 0.5.
        read_fn (Callable): User-defined function for reading images. It takes an element from `reference_images` and
            returns a numpy array of image pixels. By default, it is expected to take a path to an image and return a
            numpy array.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        - https://github.com/YanchaoYang/FDA
        - https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> target_image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> aug = A.Compose([A.FDA([target_image], p=1, read_fn=lambda x: x)])
        >>> result = aug(image=image)

    Note:
        FDA is a powerful tool for domain adaptation, particularly in unsupervised settings where annotated target
        domain samples are unavailable. It enables significant improvements in model generalization by aligning
        the low-level statistics of source and target images through a simple yet effective Fourier-based method.
    """

    class InitSchema(BaseTransformInitSchema):
        reference_images: Sequence[Any]
        read_fn: Callable[[Any], np.ndarray]
        beta_limit: ZeroOneRangeType

        @field_validator("beta_limit")
        @classmethod
        def check_ranges(cls, value: tuple[float, float]) -> tuple[float, float]:
            bounds = 0, MAX_BETA_LIMIT
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError(f"Values should be in the range {bounds} got {value} ")
            return value

    def __init__(
        self,
        reference_images: Sequence[Any],
        beta_limit: ScaleFloatType = (0, 0.1),
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.beta_limit = cast(Tuple[float, float], beta_limit)

    def apply(
        self,
        img: np.ndarray,
        target_image: np.ndarray,
        beta: float,
        **params: Any,
    ) -> np.ndarray:
        return fourier_domain_adaptation(img, target_image, beta)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
        target_img = self.read_fn(random.choice(self.reference_images))
        target_img = cv2.resize(target_img, dsize=(params["cols"], params["rows"]))

        return {"target_image": target_img}

    def get_params(self) -> dict[str, float]:
        return {"beta": random.uniform(*self.beta_limit)}

    def get_transform_init_args_names(self) -> tuple[str, str, str]:
        return "reference_images", "beta_limit", "read_fn"

    def to_dict_private(self) -> dict[str, Any]:
        msg = "FDA can not be serialized."
        raise NotImplementedError(msg)


class PixelDistributionAdaptation(ImageOnlyTransform):
    """Performs pixel-level domain adaptation by aligning the pixel value distribution of an input image
    with that of a reference image. This process involves fitting a simple statistical transformation
    (such as PCA, StandardScaler, or MinMaxScaler) to both the original and the reference images,
    transforming the original image with the transformation trained on it, and then applying the inverse
    transformation using the transform fitted on the reference image. The result is an adapted image
    that retains the original content while mimicking the pixel value distribution of the reference domain.

    The process can be visualized as two main steps:
    1. Adjusting the original image to a standard distribution space using a selected transform.
    2. Moving the adjusted image into the distribution space of the reference image by applying the inverse
       of the transform fitted on the reference image.

    This technique is especially useful in scenarios where images from different domains (e.g., synthetic
    vs. real images, day vs. night scenes) need to be harmonized for better consistency or performance in
    image processing tasks.

    Args:
        reference_images (Sequence[Any]): A sequence of objects (typically image paths) that will be
            converted into images by `read_fn`. These images serve as references for the domain adaptation.
        blend_ratio (tuple[float, float]): Specifies the minimum and maximum blend ratio for mixing
            the adapted image with the original, enhancing the diversity of the output images.
        read_fn (Callable): A user-defined function for reading and converting the objects in
            `reference_images` into numpy arrays. By default, it assumes these objects are image paths.
        transform_type (str): Specifies the type of statistical transformation to apply. Supported values
            are "pca" for Principal Component Analysis, "standard" for StandardScaler, and "minmax" for
            MinMaxScaler.
        p (float): The probability of applying the transform to any given image. Default is 1.0.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        For more information on the underlying approach, see: https://github.com/arsenyinfo/qudida

    Note:
        The PixelDistributionAdaptation transform is a novel way to perform domain adaptation at the pixel level,
        suitable for adjusting images across different conditions without complex modeling. It is effective
        for preparing images before more advanced processing or analysis.
    """

    class InitSchema(BaseTransformInitSchema):
        reference_images: Sequence[Any]
        blend_ratio: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_01)] = (
            0.25,
            1.0,
        )
        read_fn: Callable[[Any], np.ndarray]
        transform_type: Literal["pca", "standard", "minmax"]

    def __init__(
        self,
        reference_images: Sequence[Any],
        blend_ratio: tuple[float, float] = (0.25, 1.0),
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        transform_type: Literal["pca", "standard", "minmax"] = "pca",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.blend_ratio = blend_ratio
        self.transform_type = transform_type

    @staticmethod
    def _validate_shape(img: np.ndarray) -> None:
        if is_grayscale_image(img) or is_multispectral_image(img):
            raise ValueError(
                f"Unexpected image shape: expected 3 dimensions, got {len(img.shape)}."
                f"Is it a grayscale or multispectral image? It's not supported for now.",
            )

    def ensure_uint8(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        if img.dtype == np.float32:
            if img.min() < 0 or img.max() > 1:
                message = (
                    "PixelDistributionAdaptation uses uint8 under the hood, so float32 should be converted,"
                    "Can not do it automatically when the image is out of [0..1] range."
                )
                raise TypeError(message)
            return clip(img * 255, np.uint8), True
        return img, False

    def apply(self, img: np.ndarray, reference_image: np.ndarray, blend_ratio: float, **params: Any) -> np.ndarray:
        self._validate_shape(img)
        reference_image, _ = self.ensure_uint8(reference_image)
        img, needs_reconvert = self.ensure_uint8(img)

        adapted = adapt_pixel_distribution(
            img,
            ref=reference_image,
            weight=blend_ratio,
            transform_type=self.transform_type,
        )

        return to_float(adapted) if needs_reconvert else adapted

    def get_params(self) -> dict[str, Any]:
        return {
            "reference_image": self.read_fn(random.choice(self.reference_images)),
            "blend_ratio": random.uniform(self.blend_ratio[0], self.blend_ratio[1]),
        }

    def get_transform_init_args_names(self) -> tuple[str, str, str, str]:
        return "reference_images", "blend_ratio", "read_fn", "transform_type"

    def to_dict_private(self) -> dict[str, Any]:
        msg = "PixelDistributionAdaptation can not be serialized."
        raise NotImplementedError(msg)
