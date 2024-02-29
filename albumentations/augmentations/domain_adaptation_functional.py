import abc
from copy import deepcopy
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.exposure import match_histograms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing_extensions import Protocol

from albumentations.augmentations.utils import (
    clipped,
    get_opencv_dtype_from_numpy,
    preserve_shape,
)

NON_GRAY_IMAGE_SHAPE = 3
RGB_NUM_CHANNELS = 3

__all__ = [
    "fourier_domain_adaptation",
    "apply_histogram",
    "adapt_pixel_distribution",
]


class TransformerInterface(Protocol):
    @abc.abstractmethod
    def inverse_transform(self, x: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray: ...

    @abc.abstractmethod
    def transform(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray: ...


class DomainAdapter:
    """Source: https://github.com/arsenyinfo/qudida by Arseny Kravchenko"""

    def __init__(
        self,
        transformer: TransformerInterface,
        ref_img: np.ndarray,
        color_conversions: Tuple[None, None] = (None, None),
    ):
        self.color_in, self.color_out = color_conversions
        self.source_transformer = deepcopy(transformer)
        self.target_transformer = transformer
        self.target_transformer.fit(self.flatten(ref_img))

    def to_colorspace(self, img: np.ndarray) -> np.ndarray:
        return img if self.color_in is None else cv2.cvtColor(img, self.color_in)

    def from_colorspace(self, img: np.ndarray) -> np.ndarray:
        if self.color_out is None:
            return img
        return cv2.cvtColor(img.astype("uint8"), self.color_out)

    def flatten(self, img: np.ndarray) -> np.ndarray:
        img = self.to_colorspace(img)
        img = img.astype("float32") / 255.0
        return img.reshape(-1, 3)

    def reconstruct(self, pixels: np.ndarray, height: int, width: int) -> np.ndarray:
        pixels = (np.clip(pixels, 0, 1) * 255).astype("uint8")
        return self.from_colorspace(pixels.reshape(height, width, 3))

    @staticmethod
    def _pca_sign(x: np.ndarray) -> np.ndarray:
        return np.sign(np.trace(x.components_))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        pixels = self.flatten(image)
        self.source_transformer.fit(pixels)

        # dirty hack to make sure colors are not inverted
        if (
            hasattr(self.target_transformer, "components_")
            and hasattr(self.source_transformer, "components_")
            and self._pca_sign(self.target_transformer) != self._pca_sign(self.source_transformer)
        ):
            self.target_transformer.components_ *= -1

        representation = self.source_transformer.transform(pixels)
        result = self.target_transformer.inverse_transform(representation)
        return self.reconstruct(result, height, width)


@preserve_shape
def adapt_pixel_distribution(
    img: np.ndarray, ref: np.ndarray, transform_type: str = "pca", weight: float = 0.5
) -> np.ndarray:
    initial_type = img.dtype
    transformer = {"pca": PCA, "standard": StandardScaler, "minmax": MinMaxScaler}[transform_type]()
    adapter = DomainAdapter(transformer=transformer, ref_img=ref)
    result = adapter(img).astype("float32")
    return (img.astype("float32") * (1 - weight) + result * weight).astype(initial_type)


@clipped
@preserve_shape
def fourier_domain_adaptation(img: np.ndarray, target_img: np.ndarray, beta: float) -> np.ndarray:
    img = np.squeeze(img)
    target_img = np.squeeze(target_img)
    # Ensure input images have the same shape
    if img.shape != target_img.shape:
        raise ValueError(
            f"The source and target images must have the same shape, but got {img.shape} and {target_img.shape} "
            "respectively."
        )

    # Convert images to float32 if not already to avoid unnecessary conversions
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if target_img.dtype != np.float32:
        target_img = target_img.astype(np.float32)

    # Compute FFT of both source and target images
    fft_src = np.fft.fft2(img, axes=(0, 1))
    fft_trg = np.fft.fft2(target_img, axes=(0, 1))

    # Extract amplitude and phase
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    amplitude_trg = np.abs(fft_trg)

    # Compute border for amplitude substitution
    height, width = img.shape[:2]
    border = int(np.floor(min(height, width) * beta))
    center_y, center_x = height // 2, width // 2

    # Define region for amplitude substitution
    y1, y2 = center_y - border, center_y + border
    x1, x2 = center_x - border, center_x + border

    # Directly mutate the amplitude part of the source with the target in the specified region
    amplitude_src[y1:y2, x1:x2] = amplitude_trg[y1:y2, x1:x2]

    # Reconstruct the source image from its mutated amplitude and original phase
    src_image_transformed = np.fft.ifft2(amplitude_src * np.exp(1j * phase_src), axes=(0, 1))

    # Return the real part of the transformed image
    return np.real(src_image_transformed)


@preserve_shape
def apply_histogram(img: np.ndarray, reference_image: np.ndarray, blend_ratio: float) -> np.ndarray:
    # Ensure the data types match
    if img.dtype != reference_image.dtype:
        raise RuntimeError(
            f"Dtype of image and reference image must be the same. Got {img.dtype} and {reference_image.dtype}."
        )

    # Resize reference image only if necessary
    if img.shape[:2] != reference_image.shape[:2]:
        reference_image = cv2.resize(reference_image, dsize=(img.shape[1], img.shape[0]))

    img, reference_image = np.squeeze(img), np.squeeze(reference_image)

    # Determine if the images are multi-channel based on a predefined condition or shape analysis
    is_multichannel = img.ndim == NON_GRAY_IMAGE_SHAPE and img.shape[2] == RGB_NUM_CHANNELS

    # Match histograms between the images
    matched = match_histograms(img, reference_image, channel_axis=2 if is_multichannel else None)

    # Blend the original image and the matched image
    return cv2.addWeighted(matched, blend_ratio, img, 1 - blend_ratio, 0, dtype=get_opencv_dtype_from_numpy(img.dtype))
