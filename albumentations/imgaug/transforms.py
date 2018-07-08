from imgaug import augmenters as iaa

from ..core.transforms_interface import BasicTransform, DualTransform, ImageOnlyTransform

__all__ = ['BasicIAATransform', 'DualIAATransform', 'ImageOnlyIAATransform', 'IAAEmboss', 'IAASuperpixels',
           'IAASharpen', 'IAAAdditiveGaussianNoise', 'IAAPiecewiseAffine', 'IAAPerspective']


class BasicIAATransform(BasicTransform):

    def __init__(self, p=0.5):
        super(BasicIAATransform, self).__init__(p)
        self.processor = iaa.Noop()
        self.deterministic_processor = iaa.Noop()

    def __call__(self, **kwargs):
        self.deterministic_processor = self.processor.to_deterministic()
        return super(BasicIAATransform, self).__call__(**kwargs)

    def apply(self, img, **params):
        return self.deterministic_processor.augment_image(img)


class DualIAATransform(DualTransform, BasicIAATransform):
    pass


class ImageOnlyIAATransform(ImageOnlyTransform, BasicIAATransform):
    pass


class IAAEmboss(ImageOnlyIAATransform):
    """Embosses the input image and overlays the result with the original image.

    Args:
        alpha ((float, float)): range to choose the visibility of the embossed image. At 0, only the original image is
            visible,at 1.0 only its embossed version is visible. Default: (0.2, 0.5).
        strength ((float, float)): strength range of the embossing. Default: (0.2, 0.7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5):
        super(IAAEmboss, self).__init__(p)
        self.processor = iaa.Emboss(alpha, strength)


class IAASuperpixels(ImageOnlyIAATransform):
    """Completely or partially transform the input image to its superpixel representation. Uses skimage's version
    of the SLIC algorithm. May be slow.

    Args:
        p_replace (float): defines the probability of any superpixel area being replaced by the superpixel, i.e. by
            the average pixel color within its area. Default: 0.1.
        n_segments (int): target number of superpixels to generate. Default: 100.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, p_replace=0.1, n_segments=100, p=0.5):
        super(IAASuperpixels, self).__init__(p)
        self.processor = iaa.Superpixels(p_replace=p_replace, n_segments=n_segments)


class IAASharpen(ImageOnlyIAATransform):
    """Sharpens the input image and overlays the result with the original image.

    Args:
        alpha ((float, float)): range to choose the visibility of the sharpened image. At 0, only the original image is
            visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).
        lightness ((float, float)): range to choose the lightness of the sharpened image. Default: (0.5, 1.0).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, alpha=(0.2, 0.5), lightness=(0.5, 1.), p=0.5):
        super(IAASharpen, self).__init__(p)
        self.processor = iaa.Sharpen(alpha, lightness)


class IAAAdditiveGaussianNoise(ImageOnlyIAATransform):
    """Adds gaussian noise to the input image.

    Args:
        loc (int): mean of the normal distribution that generates the noise. Default: 0.
        scale ((float, float)): standard deviation of the normal distribution that generates the noise.
            Default: (0.01 * 255, 0.05 * 255).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, loc=0, scale=(0.01 * 255, 0.05 * 255), p=0.5):
        super(IAAAdditiveGaussianNoise, self).__init__(p)
        self.processor = iaa.AdditiveGaussianNoise(loc, scale)


class IAAPiecewiseAffine(DualIAATransform):
    """Places a regular grid of points on the input and randomly moves the neighbourhood of these point around
    via affine transformations.

    Args:
        scale ((float, float): factor range that determines how far each point is moved. Default: (0.03, 0.05).
        nb_rows (int): number of rows of points that the regular grid should have. Default: 4.
        nb_columns (int): number of columns of points that the regular grid should have. Default: 4.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=.5):
        super(IAAPiecewiseAffine, self).__init__(p)
        self.processor = iaa.PiecewiseAffine(scale, nb_rows, nb_cols)


class IAAPerspective(DualIAATransform):
    """Performs a random four point perspective transform of the input.

    Args:
        scale ((float, float): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners. Default: (0.05, 0.1).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, scale=(0.05, 0.1), p=.5):
        super(IAAPerspective, self).__init__(p)
        self.processor = iaa.PerspectiveTransform(scale)
