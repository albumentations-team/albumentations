from __future__ import annotations

import abc
from copy import deepcopy

import cv2
import numpy as np
from albucore.functions import add_weighted
from albucore.utils import clip, clipped, is_multispectral_image, preserve_channel_dim
from skimage.exposure import match_histograms
from typing_extensions import Protocol

from albumentations.augmentations.functional import center
from albumentations.core.types import MONO_CHANNEL_DIMENSIONS
import albumentations.augmentations.functional as fmain

__all__ = [
    "fourier_domain_adaptation",
    "apply_histogram",
    "adapt_pixel_distribution",
]


class BaseScaler:
    def __init__(self) -> None:
        self.data_min: np.ndarray | None = None
        self.data_max: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MinMaxScaler(BaseScaler):
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        super().__init__()
        self.min: float = feature_range[0]
        self.max: float = feature_range[1]
        self.data_range: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        self.data_min = np.min(x, axis=0)
        self.data_max = np.max(x, axis=0)
        self.data_range = self.data_max - self.data_min
        # Handle case where data_min equals data_max
        self.data_range[self.data_range == 0] = 1

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_max is None or self.data_range is None:
            raise ValueError(
                "This MinMaxScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )
        x_std = (x - self.data_min) / self.data_range
        return x_std * (self.max - self.min) + self.min

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.data_min is None or self.data_max is None or self.data_range is None:
            raise ValueError(
                "This MinMaxScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )
        x_std = (x - self.min) / (self.max - self.min)
        return x_std * self.data_range + self.data_min


class StandardScaler(BaseScaler):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x: np.ndarray) -> None:
        self.mean = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.scale = np.sqrt(self.var)
        # Handle case where variance is zero
        self.scale[self.scale == 0] = 1

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise ValueError(
                "This StandardScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )
        return (x - self.mean) / self.scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise ValueError(
                "This StandardScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )
        return (x * self.scale) + self.mean


class PCA:
    def __init__(self, n_components: int | None = None) -> None:
        if n_components is not None and n_components <= 0:
            raise ValueError("Number of components must be greater than zero.")
        self.n_components = n_components
        self.mean: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        x = x.astype(np.float64)
        n_samples, n_features = x.shape

        # Determine the number of components if not set
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        self.mean, eigenvectors, eigenvalues = cv2.PCACompute2(x, mean=None, maxComponents=self.n_components)
        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues.flatten()

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise ValueError(
                "This PCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )
        x = x.astype(np.float64)
        return cv2.PCAProject(x, self.mean, self.components_)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise ValueError(
                "This PCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.",
            )
        return cv2.PCABackProject(x, self.mean, self.components_)

    def explained_variance_ratio(self) -> np.ndarray:
        if self.explained_variance_ is None:
            raise ValueError(
                "This PCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method.",
            )
        total_variance = np.sum(self.explained_variance_)
        return self.explained_variance_ / total_variance

    def cumulative_explained_variance_ratio(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio())


class TransformerInterface(Protocol):
    @abc.abstractmethod
    def inverse_transform(self, x: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray: ...

    @abc.abstractmethod
    def transform(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray: ...


class DomainAdapter:
    """Source: https://github.com/arsenyinfo/qudida by Arseny Kravchenko"""

    def __init__(
        self,
        transformer: TransformerInterface,
        ref_img: np.ndarray,
        color_conversions: tuple[None, None] = (None, None),
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
        return cv2.cvtColor(clip(img, np.uint8), self.color_out)

    def flatten(self, img: np.ndarray) -> np.ndarray:
        img = self.to_colorspace(img)
        img = fmain.to_float(img)
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


@clipped
@preserve_channel_dim
def adapt_pixel_distribution(
    img: np.ndarray,
    ref: np.ndarray,
    transform_type: str = "pca",
    weight: float = 0.5,
) -> np.ndarray:
    transformer = {"pca": PCA, "standard": StandardScaler, "minmax": MinMaxScaler}[transform_type]()
    adapter = DomainAdapter(transformer=transformer, ref_img=ref)
    result = adapter(img).astype(np.float32)
    return img.astype(np.float32) * (1 - weight) + result * weight


def low_freq_mutate(amp_src: np.ndarray, amp_trg: np.ndarray, beta: float) -> np.ndarray:
    height, width = amp_src.shape[:2]
    border = int(np.floor(min(height, width) * beta))

    center_x, center_y = center(width, height)

    h1, h2 = max(0, int(center_y - border)), min(int(center_y + border), height)
    w1, w2 = max(0, int(center_x - border)), min(int(center_x + border), width)
    amp_src[h1:h2, w1:w2] = amp_trg[h1:h2, w1:w2]
    return amp_src


@clipped
@preserve_channel_dim
def fourier_domain_adaptation(img: np.ndarray, target_img: np.ndarray, beta: float) -> np.ndarray:
    src_img = img.astype(np.float32)
    trg_img = target_img.astype(np.float32)

    if len(src_img.shape) == MONO_CHANNEL_DIMENSIONS:
        src_img = np.expand_dims(src_img, axis=-1)
    if len(trg_img.shape) == MONO_CHANNEL_DIMENSIONS:
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


@clipped
@preserve_channel_dim
def apply_histogram(img: np.ndarray, reference_image: np.ndarray, blend_ratio: float) -> np.ndarray:
    # Resize reference image only if necessary
    if img.shape[:2] != reference_image.shape[:2]:
        reference_image = cv2.resize(reference_image, dsize=(img.shape[1], img.shape[0]))

    img = np.squeeze(img)
    reference_image = np.squeeze(reference_image)

    # Determine if the images are multi-channel based on a predefined condition or shape analysis
    is_multichannel = is_multispectral_image(img)

    # Match histograms between the images
    matched = match_histograms(img, reference_image, channel_axis=2 if is_multichannel else None)

    # Blend the original image and the matched image
    return add_weighted(matched, blend_ratio, img, 1 - blend_ratio)
