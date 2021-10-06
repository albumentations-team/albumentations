import random
from typing import List, Union, Tuple, Callable

import cv2
import numpy as np
from qudida import DomainAdapter
from skimage.exposure import match_histograms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .functional import clipped, preserve_shape, is_grayscale_image, is_multispectral_image
from .utils import read_rgb_image
from ..core.transforms_interface import ImageOnlyTransform, to_tuple

__all__ = [
    "HistogramMatching",
    "FDA",
    "PixelDistributionAdaptation",
    "fourier_domain_adaptation",
    "apply_histogram",
    "adapt_pixel_distribution",
]


@clipped
@preserve_shape
def fourier_domain_adaptation(img: np.ndarray, target_img: np.ndarray, beta: float) -> np.ndarray:
    """
    Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA

    Args:
        img:  source image
        target_img:  target image for domain adaptation
        beta: coefficient from source paper

    Returns:
        transformed image

    """

    img = np.squeeze(img)
    target_img = np.squeeze(target_img)

    if target_img.shape != img.shape:
        raise ValueError(
            "The source and target images must have the same shape,"
            " but got {} and {} respectively.".format(img.shape, target_img.shape)
        )

    # get fft of both source and target
    fft_src = np.fft.fft2(img.astype(np.float32), axes=(0, 1))
    fft_trg = np.fft.fft2(target_img.astype(np.float32), axes=(0, 1))

    # extract amplitude and phase of both fft-s
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    amplitude_trg = np.abs(fft_trg)

    # mutate the amplitude part of source with target
    amplitude_src = np.fft.fftshift(amplitude_src, axes=(0, 1))
    amplitude_trg = np.fft.fftshift(amplitude_trg, axes=(0, 1))
    height, width = amplitude_src.shape[:2]
    border = np.floor(min(height, width) * beta).astype(int)
    center_y, center_x = np.floor([height / 2.0, width / 2.0]).astype(int)

    y1, y2 = center_y - border, center_y + border + 1
    x1, x2 = center_x - border, center_x + border + 1

    amplitude_src[y1:y2, x1:x2] = amplitude_trg[y1:y2, x1:x2]
    amplitude_src = np.fft.ifftshift(amplitude_src, axes=(0, 1))

    # get mutated image
    src_image_transformed = np.fft.ifft2(amplitude_src * np.exp(1j * phase_src), axes=(0, 1))
    src_image_transformed = np.real(src_image_transformed)

    return src_image_transformed


@preserve_shape
def apply_histogram(img, reference_image, blend_ratio):
    reference_image = cv2.resize(reference_image, dsize=(img.shape[1], img.shape[0]))
    matched = match_histograms(np.squeeze(img), np.squeeze(reference_image), multichannel=True)
    img = cv2.addWeighted(matched, blend_ratio, img, 1 - blend_ratio, 0)
    return img


@preserve_shape
def adapt_pixel_distribution(
    img: np.ndarray, ref: np.ndarray, transform_type: str = "pca", weight: float = 0.5
) -> np.ndarray:
    initial_type = img.dtype
    transformer = {"pca": PCA, "standard": StandardScaler, "minmax": MinMaxScaler}[transform_type]()
    adapter = DomainAdapter(transformer=transformer, ref_img=ref)
    result = adapter(img).astype("float32")
    blended = (img.astype("float32") * (1 - weight) + result * weight).astype(initial_type)
    return blended


class HistogramMatching(ImageOnlyTransform):
    """
    Apply histogram matching. It manipulates the pixels of an input image so that its histogram matches
    the histogram of the reference image. If the images have multiple channels, the matching is done independently
    for each channel, as long as the number of channels is equal in the input image and the reference.

    Histogram matching can be used as a lightweight normalisation for image processing,
    such as feature matching, especially in circumstances where the images have been taken from different
    sources or in different conditions (i.e. lighting).

    See:
        https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html

    Args:
        reference_images (List[str] or List(np.ndarray)): List of file paths for reference images
            or list of reference images.
        blend_ratio (float, float): Tuple of min and max blend ratio. Matched image will be blended with original
            with random blend factor for increased diversity of generated images.
        read_fn (Callable): Used-defined function to read image. Function should get image path and return numpy
            array of image pixels.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        uint8, uint16, float32
    """

    def __init__(
        self,
        reference_images: List[Union[str, np.ndarray]],
        blend_ratio=(0.5, 1.0),
        read_fn=read_rgb_image,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.blend_ratio = blend_ratio

    def apply(self, img, reference_image=None, blend_ratio=0.5, **params):
        return apply_histogram(img, reference_image, blend_ratio)

    def get_params(self):
        return {
            "reference_image": self.read_fn(random.choice(self.reference_images)),
            "blend_ratio": random.uniform(self.blend_ratio[0], self.blend_ratio[1]),
        }

    def get_transform_init_args_names(self):
        return ("reference_images", "blend_ratio", "read_fn")

    def _to_dict(self):
        raise NotImplementedError("HistogramMatching can not be serialized.")


class FDA(ImageOnlyTransform):
    """
    Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA
    Simple "style transfer".

    Args:
        reference_images (List[str] or List(np.ndarray)): List of file paths for reference images
            or list of reference images.
        beta_limit (float or tuple of float): coefficient beta from paper. Recommended less 0.3.
        read_fn (Callable): Used-defined function to read image. Function should get image path and return numpy
            array of image pixels.

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
        reference_images: List[Union[str, np.ndarray]],
        beta_limit=0.1,
        read_fn=read_rgb_image,
        always_apply=False,
        p=0.5,
    ):
        super(FDA, self).__init__(always_apply=always_apply, p=p)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.beta_limit = to_tuple(beta_limit, low=0)

    def apply(self, img, target_image=None, beta=0.1, **params):
        return fourier_domain_adaptation(img=img, target_img=target_image, beta=beta)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        target_img = self.read_fn(random.choice(self.reference_images))
        target_img = cv2.resize(target_img, dsize=(img.shape[1], img.shape[0]))

        return {"target_image": target_img}

    def get_params(self):
        return {"beta": random.uniform(self.beta_limit[0], self.beta_limit[1])}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("reference_images", "beta_limit", "read_fn")

    def _to_dict(self):
        raise NotImplementedError("FDA can not be serialized.")


class PixelDistributionAdaptation(ImageOnlyTransform):
    """
    Another naive and quick pixel-level domain adaptation. It fits a simple transform (such as PCA, StandardScaler
    or MinMaxScaler) on both original and reference image, transforms original image with transform trained on this
    image and then performs inverse transformation using transform fitted on reference image.

    Args:
        reference_images (List[str] or List(np.ndarray)): List of file paths for reference images
            or list of reference images.
        blend_ratio (float, float): Tuple of min and max blend ratio. Matched image will be blended with original
            with random blend factor for increased diversity of generated images.
        read_fn (Callable): Used-defined function to read image. Function should get image path and return numpy
            array of image pixels. Usually it's default `read_rgb_image` when images paths are used as reference,
            otherwise it could be identity function `lambda x: x` if reference images have been read in advance.
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
        reference_images: List[Union[str, np.ndarray]],
        blend_ratio: Tuple[float, float] = (0.25, 1.0),
        read_fn: Callable[[Union[str, np.ndarray]], np.ndarray] = read_rgb_image,
        transform_type: str = "pca",
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.blend_ratio = blend_ratio
        expected_transformers = ("pca", "standard", "minmax")
        if transform_type not in expected_transformers:
            raise ValueError(
                f"Got unexpected transform_type {transform_type}. Expected one of {expected_transformers}"
            )
        self.transform_type = transform_type

    @staticmethod
    def _validate_shape(img: np.ndarray):
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

    def apply(self, img, reference_image, blend_ratio, **params):
        self._validate_shape(img)
        reference_image, _ = self.ensure_uint8(reference_image)
        img, needs_reconvert = self.ensure_uint8(img)

        adapted = adapt_pixel_distribution(
            img=img,
            ref=reference_image,
            weight=blend_ratio,
            transform_type=self.transform_type,
        )
        if needs_reconvert:
            adapted = adapted.astype("float32") * (1 / 255)
        return adapted

    def get_params(self):
        return {
            "reference_image": self.read_fn(random.choice(self.reference_images)),
            "blend_ratio": random.uniform(self.blend_ratio[0], self.blend_ratio[1]),
        }

    def get_transform_init_args_names(self):
        return ("reference_images", "blend_ratio", "read_fn", "transform_type")

    def _to_dict(self):
        raise NotImplementedError("PixelDistributionAdaptation can not be serialized.")
