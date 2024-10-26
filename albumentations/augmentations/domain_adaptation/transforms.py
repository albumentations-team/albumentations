from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Callable, Literal, cast

import cv2
import numpy as np
from albucore import add_weighted, get_num_channels
from pydantic import AfterValidator, Field, field_validator

import albumentations.augmentations.geometric.functional as fgeometric
from albumentations.augmentations.domain_adaptation.functional import (
    adapt_pixel_distribution,
    apply_histogram,
    fourier_domain_adaptation,
)
from albumentations.augmentations.utils import read_rgb_image
from albumentations.core.composition import Compose
from albumentations.core.pydantic import ZeroOneRangeType, check_01, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, BasicTransform, ImageOnlyTransform
from albumentations.core.types import ScaleFloatType

__all__ = [
    "HistogramMatching",
    "FDA",
    "PixelDistributionAdaptation",
    "TemplateTransform",
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
            "reference_image": self.read_fn(self.py_random.choice(self.reference_images)),
            "blend_ratio": self.py_random.uniform(*self.blend_ratio),
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
        self.beta_limit = cast(tuple[float, float], beta_limit)

    def apply(
        self,
        img: np.ndarray,
        target_image: np.ndarray,
        beta: float,
        **params: Any,
    ) -> np.ndarray:
        return fourier_domain_adaptation(img, target_image, beta)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
        target_img = self.read_fn(self.py_random.choice(self.reference_images))
        target_img = cv2.resize(target_img, dsize=(params["cols"], params["rows"]))

        return {"target_image": target_img, "beta": self.py_random.uniform(*self.beta_limit)}

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
            the adapted image with the original. This enhances the diversity of the output images.
            Values should be in the range [0, 1]. Default: (0.25, 1.0)
        read_fn (Callable): A user-defined function for reading and converting the objects in
            `reference_images` into numpy arrays. By default, it assumes these objects are image paths.
        transform_type (Literal["pca", "standard", "minmax"]): Specifies the type of statistical
            transformation to apply.
            - "pca": Principal Component Analysis
            - "standard": StandardScaler (zero mean and unit variance)
            - "minmax": MinMaxScaler (scales to a fixed range, usually [0, 1])
            Default: "pca"
        p (float): The probability of applying the transform to any given image. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The effectiveness of the adaptation depends on the similarity between the input and reference domains.
        - PCA transformation may alter color relationships more significantly than other methods.
        - StandardScaler and MinMaxScaler preserve color relationships better but may provide less dramatic adaptations.
        - The blend_ratio parameter allows for a smooth transition between the original and fully adapted image.
        - This transform cannot be directly serialized due to its dependency on external image data.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> reference_image = np.random.randint(0, 256, [100, 100, 3], dtype=np.uint8)
        >>> transform = A.PixelDistributionAdaptation(
        ...     reference_images=[reference_image],
        ...     blend_ratio=(0.5, 1.0),
        ...     transform_type="standard",
        ...     read_fn=lambda x: x,
        ...     p=1.0
        ... )
        >>> result = transform(image=image)
        >>> adapted_image = result["image"]

    References:
        - https://github.com/arsenyinfo/qudida
        - https://arxiv.org/abs/1911.11483
    """

    class InitSchema(BaseTransformInitSchema):
        reference_images: Sequence[Any]
        blend_ratio: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_01)]
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

    def apply(self, img: np.ndarray, reference_image: np.ndarray, blend_ratio: float, **params: Any) -> np.ndarray:
        return adapt_pixel_distribution(
            img,
            ref=reference_image,
            weight=blend_ratio,
            transform_type=self.transform_type,
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "reference_image": self.read_fn(self.py_random.choice(self.reference_images)),
            "blend_ratio": self.py_random.uniform(*self.blend_ratio),
        }

    def get_transform_init_args_names(self) -> tuple[str, str, str, str]:
        return "reference_images", "blend_ratio", "read_fn", "transform_type"

    def to_dict_private(self) -> dict[str, Any]:
        msg = "PixelDistributionAdaptation can not be serialized."
        raise NotImplementedError(msg)


class TemplateTransform(ImageOnlyTransform):
    """Apply blending of input image with specified templates.

    This transform overlays one or more template images onto the input image using alpha blending.
    It allows for creating complex composite images or simulating various visual effects.

    Args:
        templates (numpy array | list[np.ndarray]): Images to use as templates for the transform.
            If a single numpy array is provided, it will be used as the only template.
            If a list of numpy arrays is provided, one will be randomly chosen for each application.

        img_weight (tuple[float, float]  | float): Weight of the original image in the blend.
            If a single float, that value will always be used.
            If a tuple (min, max), the weight will be randomly sampled from the range [min, max) for each application.
            To use a fixed weight, use (weight, weight).
            Default: (0.5, 0.5).

        template_transform (A.Compose | None): A composition of Albumentations transforms to apply to the template
            before blending.
            This should be an instance of A.Compose containing one or more Albumentations transforms.
            Default: None.

        name (str | None): Name of the transform instance. Used for serialization purposes.
            Default: None.

        p (float): Probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The template(s) must have the same number of channels as the input image or be single-channel.
        - If a single-channel template is used with a multi-channel image, the template will be replicated across
          all channels.
        - The template(s) will be resized to match the input image size if they differ.
        - To make this transform serializable, provide a name when initializing it.

    Mathematical Formulation:
        Given:
        - I: Input image
        - T: Template image
        - w_i: Weight of input image (sampled from img_weight)

        The blended image B is computed as:

        B = w_i * I + (1 - w_i) * T

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> template = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Apply template transform with a single template
        >>> transform = A.TemplateTransform(templates=template, name="my_template_transform", p=1.0)
        >>> blended_image = transform(image=image)['image']

        # Apply template transform with multiple templates and custom weights
        >>> templates = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        >>> transform = A.TemplateTransform(
        ...     templates=templates,
        ...     img_weight=(0.3, 0.7),
        ...     name="multi_template_transform",
        ...     p=1.0
        ... )
        >>> blended_image = transform(image=image)['image']

        # Apply template transform with additional transforms on the template
        >>> template_transform = A.Compose([A.RandomBrightnessContrast(p=1)])
        >>> transform = A.TemplateTransform(
        ...     templates=template,
        ...     img_weight=0.6,
        ...     template_transform=template_transform,
        ...     name="transformed_template",
        ...     p=1.0
        ... )
        >>> blended_image = transform(image=image)['image']

    References:
        - Alpha compositing: https://en.wikipedia.org/wiki/Alpha_compositing
        - Image blending: https://en.wikipedia.org/wiki/Image_blending
    """

    class InitSchema(BaseTransformInitSchema):
        templates: np.ndarray | Sequence[np.ndarray]
        img_weight: ZeroOneRangeType
        template_weight: ZeroOneRangeType | None = Field(
            deprecated="Template_weight is deprecated. Computed automatically as (1 - img_weight)",
        )
        template_transform: Compose | BasicTransform | None = None
        name: str | None

        @field_validator("templates")
        @classmethod
        def validate_templates(cls, v: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
            if isinstance(v, np.ndarray):
                return [v]
            if isinstance(v, list):
                if not all(isinstance(item, np.ndarray) for item in v):
                    msg = "All templates must be numpy arrays."
                    raise ValueError(msg)
                return v
            msg = "Templates must be a numpy array or a list of numpy arrays."
            raise TypeError(msg)

    def __init__(
        self,
        templates: np.ndarray | list[np.ndarray],
        img_weight: ScaleFloatType = (0.5, 0.5),
        template_weight: None = None,
        template_transform: Compose | BasicTransform | None = None,
        name: str | None = None,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.templates = templates
        self.img_weight = cast(tuple[float, float], img_weight)
        self.template_transform = template_transform
        self.name = name

    def apply(
        self,
        img: np.ndarray,
        template: np.ndarray,
        img_weight: float,
        **params: Any,
    ) -> np.ndarray:
        if img_weight == 0:
            return template
        if img_weight == 1:
            return img

        return add_weighted(img, img_weight, template, 1 - img_weight)

    def get_params(self) -> dict[str, float]:
        return {
            "img_weight": self.py_random.uniform(*self.img_weight),
        }

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        img = data["image"] if "image" in data else data["images"][0]

        template = self.py_random.choice(self.templates)

        if self.template_transform is not None:
            template = self.template_transform(image=template)["image"]

        if get_num_channels(template) not in [1, get_num_channels(img)]:
            msg = (
                "Template must be a single channel or "
                "has the same number of channels as input "
                f"image ({get_num_channels(img)}), got {get_num_channels(template)}"
            )
            raise ValueError(msg)

        if template.dtype != img.dtype:
            msg = "Image and template must be the same image type"
            raise ValueError(msg)

        if img.shape[:2] != template.shape[:2]:
            template = fgeometric.resize(template, img.shape[:2], interpolation=cv2.INTER_AREA)

        if get_num_channels(template) == 1 and get_num_channels(img) > 1:
            template = np.stack((template,) * get_num_channels(img), axis=-1)

        # in order to support grayscale image with dummy dim
        template = template.reshape(img.shape)

        return {"template": template}

    @classmethod
    def is_serializable(cls) -> bool:
        return False

    def to_dict_private(self) -> dict[str, Any]:
        if self.name is None:
            msg = (
                "To make a TemplateTransform serializable you should provide the `name` argument, "
                "e.g. `TemplateTransform(name='my_transform', ...)`."
            )
            raise ValueError(msg)
        return {"__class_fullname__": self.get_class_fullname(), "__name__": self.name}
