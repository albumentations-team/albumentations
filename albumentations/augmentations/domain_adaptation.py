from typing import List
import random

import cv2
import numpy as np

from ..core.transforms_interface import ImageOnlyTransform, to_tuple
from .functional import clipped

__all__ = ["HistogramMatching", "FDA", "fourier_domain_adaptation"]


@clipped
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
    center_x, center_y = np.floor([height / 2.0, width / 2.0]).astype(int)

    y1, y2 = center_y - border, center_y + border + 1
    x1, x2 = center_x - border, center_x + border + 1

    amplitude_src[y1:y2, x1:x2] = amplitude_trg[y1:y2, x1:x2]
    amplitude_src = np.fft.ifftshift(amplitude_src, axes=(0, 1))

    # get mutated image
    src_image_transformed = np.fft.ifft2(amplitude_src * np.exp(1j * phase_src), axes=(0, 1))
    src_image_transformed = np.real(src_image_transformed)

    return src_image_transformed


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
        reference_images (List[str]): List of file paths for reference images.
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

    def __init__(self, reference_images: List[str], blend_ratio=(0.5, 1.0), read_fn=cv2.imread, p=0.5):
        super().__init__(p=p)
        self.reference_images = reference_images
        self.read_fn = read_fn
        self.blend_ratio = blend_ratio

    def apply(self, img, reference_image=None, blend_ratio=0.5, **params):
        from skimage.exposure import match_histograms

        if random.random() < self.p:
            reference_image = cv2.resize(reference_image, dsize=(img.shape[1], img.shape[0]))
            matched = match_histograms(img, reference_image, multichannel=True)
            img = cv2.addWeighted(matched, blend_ratio, img, 1 - blend_ratio, 0)

        return img

    def get_params(self):
        return {
            "reference_image": self.read_fn(random.choice(self.reference_images)),
            "blend_ratio": random.uniform(self.blend_ratio[0], self.blend_ratio[1]),
        }

    def get_transform_init_args_names(self):
        return ("blend_ratio",)


class FDA(ImageOnlyTransform):
    """
    Fourier Domain Adaptation from https://github.com/YanchaoYang/FDA
    Simple "style transfer".
    Important: you need to pass target image as a parameter `target_image` in __call__, see example

    Args:
        beta_limit (float or tuple of float): coefficient beta from paper. Recommended less 0.3.

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
        >>> aug = A.Compose([A.FDA(p=1)])
        >>> result = aug(image=image, target_image=target_image)

    """

    def __init__(self, beta_limit=0.1, always_apply=False, p=0.5):
        super(FDA, self).__init__(always_apply=always_apply, p=p)
        self.beta_limit = to_tuple(beta_limit, low=0)

    def apply(self, img, target_image=None, beta=0.1, **params):
        return fourier_domain_adaptation(img=img, target_img=target_image, beta=beta)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        target_img = params["target_image"]
        target_img = cv2.resize(target_img, dsize=(img.shape[1], img.shape[0]))

        if target_img.shape != img.shape:
            raise ValueError(
                "The source and target images must contain the same shape,"
                " but got {} and {} respectively.".format(img.shape, target_img.shape)
            )

        return {"target_image": target_img}

    def get_params(self):
        return {"beta": random.uniform(self.beta_limit[0], self.beta_limit[1])}

    @property
    def targets_as_params(self):
        return ["image", "target_image"]

    def get_transform_init_args_names(self):
        return ("beta_limit",)
