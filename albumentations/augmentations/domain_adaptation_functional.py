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

GRAYSCALE_IMAGE_SHAPE = 2
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


def low_freq_mutate(amp_src: np.ndarray, amp_trg: np.ndarray, beta: float) -> np.ndarray:
    height, width = amp_src.shape[:2]
    border = int(np.floor(min(height, width) * beta))
    center_y, center_h = height // 2, width // 2
    h1, h2 = max(0, center_y - border), min(center_y + border, height - 1)
    w1, w2 = max(0, center_h - border), min(center_h + border, width - 1)
    amp_src[h1:h2, w1:w2] = amp_trg[h1:h2, w1:w2]
    return amp_src


@clipped
@preserve_shape
def fourier_domain_adaptation(img: np.ndarray, target_img: np.ndarray, beta: float) -> np.ndarray:
    src_img = img.astype(np.float32)
    trg_img = target_img.astype(np.float32)

    if len(src_img.shape) == GRAYSCALE_IMAGE_SHAPE:
        src_img = np.expand_dims(src_img, axis=-1)
    if len(trg_img.shape) == GRAYSCALE_IMAGE_SHAPE:
        trg_img = np.expand_dims(trg_img, axis=-1)

    num_channels = src_img.shape[-1]

    # Prepare container for the output image
    src_in_trg = np.zeros_like(src_img)

    for channel_id in range(num_channels):
        # Perform FFT on each channel
        fft_src = np.fft.fft2(src_img[:, :, channel_id])
        fft_trg = np.fft.fft2(trg_img[:, :, channel_id])

        # Shift the zero frequency component to the center
        fft_src_shifted = np.fft.fftshift(fft_src)
        fft_trg_shifted = np.fft.fftshift(fft_trg)

        # Extract amplitude and phase
        amp_src, pha_src = np.abs(fft_src_shifted), np.angle(fft_src_shifted)
        amp_trg = np.abs(fft_trg_shifted)

        # Mutate the amplitude part of the source with the target

        mutated_amp = low_freq_mutate(amp_src.copy(), amp_trg, beta)

        # Combine the mutated amplitude with the original phase
        fft_src_mutated = np.fft.ifftshift(mutated_amp * np.exp(1j * pha_src))

        # Perform inverse FFT
        src_in_trg_channel = np.fft.ifft2(fft_src_mutated)

        # Store the result in the corresponding channel of the output image
        src_in_trg[:, :, channel_id] = np.real(src_in_trg_channel)

    return src_in_trg


@preserve_shape
def apply_histogram(img: np.ndarray, reference_image: np.ndarray, blend_ratio: float) -> np.ndarray:
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
