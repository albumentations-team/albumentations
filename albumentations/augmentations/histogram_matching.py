from typing import List
import random

import cv2

from ..core.transforms_interface import ImageOnlyTransform

__all__ = ["HistogramMatching"]


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
