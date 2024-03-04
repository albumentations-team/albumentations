import random
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

from albumentations.augmentations.domain_adaptation_functional import (
    adapt_pixel_distribution,
    apply_histogram,
    fourier_domain_adaptation,
)
from albumentations.augmentations.utils import (
    is_grayscale_image,
    is_multispectral_image,
    read_rgb_image,
)
from albumentations.core.transforms_interface import ImageOnlyTransform, to_tuple
from albumentations.core.types import ScaleFloatType

__all__ = [
    "HistogramMatching",
    "FDA",
    "PixelDistributionAdaptation",
]


class HistogramMatching(ImageOnlyTransform):
    """Apply histogram matching. It manipulates the pixels of an input image so that its histogram matches
    the histogram of the reference image. If the images have multiple channels, the matching is done independently
    for each channel, as long as the number of channels is equal in the input image and the reference.

    Histogram matching can be used as a lightweight normalization for image processing,
    such as feature matching, especially in circumstances where the images have been taken from different
    sources or in different conditions (i.e. lighting).

    See:
        https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html

    Args:
        reference_images (Sequence[Any]): Sequence of objects that will be converted to images by `read_fn`. By default,
        it expects a sequence of paths to images.
        blend_ratio: Tuple of min and max blend ratio. Matched image will be blended with original
            with random blend factor for increased diversity of generated images.
        read_fn (Callable): Used-defined function to read image. Function should get an element of `reference_images`
        and return numpy array of image pixels. Default: takes as input a path to an image and returns a numpy array.
        p: probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, uint16, float32

    """

    def __init__(
        self,
        reference_images: Sequence[Any],
        blend_ratio: Tuple[float, float] = (0.5, 1.0),
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.blend_ratio = blend_ratio

    def apply(
        self: np.ndarray,
        img: np.ndarray,
        reference_image: Optional[np.ndarray] = None,
        blend_ratio: float = 0.5,
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
    """Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA
    Simple "style transfer".

    Args:
        reference_images (Sequence[Any]): Sequence of objects that will be converted to images by `read_fn`. By default,
        it expects a sequence of paths to images.
        beta_limit (float or tuple of float): coefficient beta from paper. Recommended less 0.3.
        read_fn (Callable): Used-defined function to read image. Function should get an element of `reference_images`
        and return numpy array of image pixels. Default: takes as input a path to an image and returns a numpy array.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://github.com/YanchaoYang/FDA
        https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> target_image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> aug = A.Compose([A.FDA([target_image], p=1, read_fn=lambda x: x)])
        >>> result = aug(image=image)

    """

    def __init__(
        self,
        reference_images: Sequence[np.ndarray],
        beta_limit: ScaleFloatType = 0.1,
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.beta_limit = to_tuple(beta_limit, low=0)

    def apply(
        self, img: np.ndarray, target_image: Optional[np.ndarray] = None, beta: float = 0.1, **params: Any
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
    """Another naive and quick pixel-level domain adaptation. It fits a simple transform (such as PCA, StandardScaler
    or MinMaxScaler) on both original and reference image, transforms original image with transform trained on this
    image and then performs inverse transformation using transform fitted on reference image.

    Args:
        reference_images (Sequence[Any]): Sequence of objects that will be converted to images by `read_fn`. By default,
        it expects a sequence of paths to images.
        blend_ratio (float, float): Tuple of min and max blend ratio. Matched image will be blended with original
            with random blend factor for increased diversity of generated images.
        read_fn (Callable): Used-defined function to read image. Function should get an element of `reference_images`
        and return numpy array of image pixels. Default: takes as input a path to an image and returns a numpy array.
        transform_type (str): type of transform; "pca", "standard", "minmax" are allowed.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, float32

    See also: https://github.com/arsenyinfo/qudida

    """

    def __init__(
        self,
        reference_images: Sequence[Any],
        blend_ratio: Tuple[float, float] = (0.25, 1.0),
        read_fn: Callable[[Any], np.ndarray] = read_rgb_image,
        transform_type: Literal["pca", "standard", "minmax"] = "pca",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.blend_ratio = blend_ratio
        expected_transformers = ("pca", "standard", "minmax")
        if transform_type not in expected_transformers:
            raise ValueError(f"Got unexpected transform_type {transform_type}. Expected one of {expected_transformers}")
        self.transform_type = transform_type

    @staticmethod
    def _validate_shape(img: np.ndarray) -> None:
        if is_grayscale_image(img) or is_multispectral_image(img):
            raise ValueError(
                f"Unexpected image shape: expected 3 dimensions, got {len(img.shape)}."
                f"Is it a grayscale or multispectral image? It's not supported for now."
            )

    def ensure_uint8(self, img: np.ndarray) -> Tuple[np.ndarray, bool]:
        if img.dtype == np.float32:
            if img.min() < 0 or img.max() > 1:
                message = (
                    "PixelDistributionAdaptation uses uint8 under the hood, so float32 should be converted,"
                    "Can not do it automatically when the image is out of [0..1] range."
                )
                raise TypeError(message)
            return (img * 255).astype("uint8"), True
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
