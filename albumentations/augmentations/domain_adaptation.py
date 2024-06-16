import random
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, cast

import cv2
import numpy as np
from albucore.utils import clip, is_grayscale_image, is_multispectral_image
from pydantic import field_validator

from albumentations.augmentations.domain_adaptation_functional import (
    adapt_pixel_distribution,
    apply_histogram,
    fourier_domain_adaptation,
)
from albumentations.augmentations.utils import read_rgb_image
from albumentations.core.pydantic import NonNegativeFloatRangeType, ZeroOneRangeType
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform
from albumentations.core.types import ScaleFloatType

__all__ = [
    "HistogramMatching",
    "FDA",
    "PixelDistributionAdaptation",
]

MAX_BETA_LIMIT = 0.5


class HistogramMatching(ImageOnlyTransform):
    """Implements histogram matching, a technique that adjusts the pixel values of an input image
    to match the histogram of a reference image. This adjustment ensures that the output image
    has a similar tone and contrast to the reference. The process is applied independently to
    each channel of multi-channel images, provided both the input and reference images have the
    same number of channels.

    Histogram matching serves as an effective normalization method in image processing tasks such
    as feature matching. It is particularly useful when images originate from varied sources or are
    captured under different lighting conditions, helping to standardize the images' appearance
    before further processing.

    Args:
        reference_images (Sequence[Any]): A sequence of objects to be converted into images by `read_fn`.
            Typically, this is a sequence of image paths.
        blend_ratio (Tuple[float, float]): Specifies the minimum and maximum blend ratio for blending the matched
            image with the original image. A random blend factor within this range is chosen for each image to
            increase the diversity of the output images.
        read_fn (Callable[[Any], np.ndarray]): A user-defined function for reading images, which accepts an
            element from `reference_images` and returns a numpy array of image pixels. By default, this is expected
            to take a file path and return an image as a numpy array.
        p (float): The probability of applying the transform to any given image. Defaults to 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        This class cannot be serialized directly due to its dynamic nature and dependency on external image data.
        An attempt to serialize it will raise a NotImplementedError.

    Reference:
        https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> target_image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> aug = A.Compose([A.HistogramMatching([target_image], p=1, read_fn=lambda x: x)])
        >>> result = aug(image=image)
    """

    class InitSchema(BaseTransformInitSchema):
        reference_images: Sequence[Any]
        blend_ratio: ZeroOneRangeType = (0.5, 1.0)
        read_fn: Callable[[Any], np.ndarray]

    def __init__(
        self,
        reference_images: Sequence[Any],
        blend_ratio: Tuple[float, float] = (0.5, 1.0),
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        always_apply: Optional[bool] = None,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
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

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "reference_image": self.read_fn(random.choice(self.reference_images)),
            "blend_ratio": random.uniform(self.blend_ratio[0], self.blend_ratio[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("reference_images", "blend_ratio", "read_fn")

    def to_dict_private(self) -> Dict[str, Any]:
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
        beta_limit (float or tuple of float): Coefficient beta from the paper, controlling the swapping extent of
            frequency components. Values should be less than 0.5.
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
        beta_limit: NonNegativeFloatRangeType = (0, 0.1)

        @field_validator("beta_limit")
        @classmethod
        def check_ranges(cls, value: Tuple[float, float]) -> Tuple[float, float]:
            bounds = 0, MAX_BETA_LIMIT
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError(f"Values should be in the range {bounds} got {value} ")
            return value

    def __init__(
        self,
        reference_images: Sequence[Any],
        beta_limit: ScaleFloatType = (0, 0.1),
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        always_apply: Optional[bool] = None,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
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

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        img = params["image"]
        target_img = self.read_fn(random.choice(self.reference_images))
        target_img = cv2.resize(target_img, dsize=(img.shape[1], img.shape[0]))

        return {"target_image": target_img}

    def get_params(self) -> Dict[str, float]:
        return {"beta": random.uniform(self.beta_limit[0], self.beta_limit[1])}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return "reference_images", "beta_limit", "read_fn"

    def to_dict_private(self) -> Dict[str, Any]:
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
        blend_ratio (Tuple[float, float]): Specifies the minimum and maximum blend ratio for mixing
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
        blend_ratio: ZeroOneRangeType = (0.25, 1.0)
        read_fn: Callable[[Any], np.ndarray]
        transform_type: Literal["pca", "standard", "minmax"]

    def __init__(
        self,
        reference_images: Sequence[Any],
        blend_ratio: Tuple[float, float] = (0.25, 1.0),
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        transform_type: Literal["pca", "standard", "minmax"] = "pca",
        always_apply: Optional[bool] = None,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
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

    def ensure_uint8(self, img: np.ndarray) -> Tuple[np.ndarray, bool]:
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
        if needs_reconvert:
            adapted = adapted.astype("float32") * (1 / 255)
        return adapted

    def get_params(self) -> Dict[str, Any]:
        return {
            "reference_image": self.read_fn(random.choice(self.reference_images)),
            "blend_ratio": random.uniform(self.blend_ratio[0], self.blend_ratio[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return "reference_images", "blend_ratio", "read_fn", "transform_type"

    def to_dict_private(self) -> Dict[str, Any]:
        msg = "PixelDistributionAdaptation can not be serialized."
        raise NotImplementedError(msg)
