import imgaug as ia
from imgaug import augmenters as iaa

from ..augmentations.bbox_utils import convert_bboxes_from_albumentations, \
    convert_bboxes_to_albumentations
from ..core.transforms_interface import BasicTransform, DualTransform, ImageOnlyTransform

__all__ = ['BasicIAATransform', 'DualIAATransform', 'ImageOnlyIAATransform', 'IAAEmboss', 'IAASuperpixels',
           'IAASharpen', 'IAAAdditiveGaussianNoise', 'IAACropAndPad', 'IAAFliplr', 'IAAFlipud', 'IAAAffine',
           'IAAPiecewiseAffine', 'IAAPerspective']


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
    def __init__(self, p):
        super(DualIAATransform, self).__init__(p)

    def apply_to_bboxes(self, bboxes, rows=0, cols=0, **params):
        if len(bboxes):
            bboxes = convert_bboxes_from_albumentations(bboxes, 'pascal_voc', rows=rows, cols=cols)

            bboxes_t = ia.BoundingBoxesOnImage([ia.BoundingBox(*bbox[:4]) for bbox in bboxes], (rows, cols))
            bboxes_t = self.deterministic_processor.augment_bounding_boxes([bboxes_t])[0].bounding_boxes
            bboxes_t = [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] + list(bbox_orig[4:]) for (bbox, bbox_orig) in
                        zip(bboxes_t, bboxes)]

            bboxes = convert_bboxes_to_albumentations(bboxes_t, 'pascal_voc', rows=rows, cols=cols)
        return bboxes


class ImageOnlyIAATransform(ImageOnlyTransform, BasicIAATransform):
    pass


class IAACropAndPad(DualIAATransform):
    def __init__(self, px=None, percent=None, pad_mode='constant', pad_cval=0, keep_size=True, p=1):
        super(IAACropAndPad, self).__init__(p)
        self.processor = iaa.CropAndPad(px, percent, pad_mode, pad_cval, keep_size)


class IAAFliplr(DualIAATransform):
    def __init__(self, p=0.5):
        super(IAAFliplr, self).__init__(1)
        self.processor = iaa.Fliplr(p)


class IAAFlipud(DualIAATransform):
    def __init__(self, p=0.5):
        super(IAAFlipud, self).__init__(1)
        self.processor = iaa.Flipud(p)


class IAAEmboss(ImageOnlyIAATransform):
    """Emboss the input image and overlays the result with the original image.

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
    """Sharpen the input image and overlays the result with the original image.

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
    """Add gaussian noise to the input image.

    Args:
        loc (int): mean of the normal distribution that generates the noise. Default: 0.
        scale ((float, float)): standard deviation of the normal distribution that generates the noise.
            Default: (0.01 * 255, 0.05 * 255).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, loc=0, scale=(0.01 * 255, 0.05 * 255), per_channel=False, p=0.5):
        super(IAAAdditiveGaussianNoise, self).__init__(p)
        self.processor = iaa.AdditiveGaussianNoise(loc, scale, per_channel)


class IAAPiecewiseAffine(DualIAATransform):
    """Place a regular grid of points on the input and randomly move the neighbourhood of these point around
    via affine transformations.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        scale ((float, float): factor range that determines how far each point is moved. Default: (0.03, 0.05).
        nb_rows (int): number of rows of points that the regular grid should have. Default: 4.
        nb_columns (int): number of columns of points that the regular grid should have. Default: 4.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, scale=0, nb_rows=4, nb_cols=4, order=1, cval=0, mode='constant', p=.5):
        super(IAAPiecewiseAffine, self).__init__(p)
        self.processor = iaa.PiecewiseAffine(scale, nb_rows, nb_cols, order, cval, mode)


class IAAAffine(DualIAATransform):
    """Place a regular grid of points on the input and randomly move the neighbourhood of these point around
    via affine transformations.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, scale=1.0, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0, order=1, cval=0,
                 mode='reflect', p=0.5):
        super(IAAAffine, self).__init__(p)
        self.processor = iaa.Affine(scale, translate_percent, translate_px, rotate, shear, order, cval, mode)


class IAAPerspective(DualIAATransform):
    """Perform a random four point perspective transform of the input.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        scale ((float, float): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners. Default: (0.05, 0.1).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, scale=(0.05, 0.1), keep_size=True, p=.5):
        super(IAAPerspective, self).__init__(p)
        self.processor = iaa.PerspectiveTransform(scale, keep_size)
