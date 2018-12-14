import imgaug as ia
from imgaug import augmenters as iaa

from ..augmentations.bbox_utils import convert_bboxes_from_albumentations, \
    convert_bboxes_to_albumentations
from ..augmentations.keypoints_utils import convert_keypoints_from_albumentations, \
    convert_keypoints_to_albumentations
from ..core.transforms_interface import BasicTransform, DualTransform, ImageOnlyTransform, to_tuple

__all__ = ['BasicIAATransform', 'DualIAATransform', 'ImageOnlyIAATransform', 'IAAEmboss', 'IAASuperpixels',
           'IAASharpen', 'IAAAdditiveGaussianNoise', 'IAACropAndPad', 'IAAFliplr', 'IAAFlipud', 'IAAAffine',
           'IAAPiecewiseAffine', 'IAAPerspective']


class BasicIAATransform(BasicTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(BasicIAATransform, self).__init__(always_apply, p)

    @property
    def processor(self):
        return iaa.Noop()

    def __call__(self, **kwargs):
        return super(BasicIAATransform, self).__call__(**kwargs)

    def update_params(self, params, **kwargs):
        params = super(BasicIAATransform, self).update_params(params, **kwargs)
        params['deterministic_processor'] = self.processor.to_deterministic()
        return params

    def apply(self, img, deterministic_processor=None, **params):
        return deterministic_processor.augment_image(img)


class DualIAATransform(DualTransform, BasicIAATransform):
    def apply_to_bboxes(self, bboxes, deterministic_processor=None, rows=0, cols=0, **params):
        if len(bboxes):
            bboxes = convert_bboxes_from_albumentations(bboxes, 'pascal_voc', rows=rows, cols=cols)

            bboxes_t = ia.BoundingBoxesOnImage([ia.BoundingBox(*bbox[:4]) for bbox in bboxes], (rows, cols))
            bboxes_t = deterministic_processor.augment_bounding_boxes([bboxes_t])[0].bounding_boxes
            bboxes_t = [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] + list(bbox_orig[4:]) for (bbox, bbox_orig) in
                        zip(bboxes_t, bboxes)]

            bboxes = convert_bboxes_to_albumentations(bboxes_t, 'pascal_voc', rows=rows, cols=cols)
        return bboxes

    """Applies transformation to keypoints.
    Notes:
        Since IAA supports only xy keypoints, scale and orientation will remain unchanged.
    TODO:
        Emit a warning message if child classes of DualIAATransform are instantiated
        inside Compose with keypoints format other than 'xy'.
    """

    def apply_to_keypoints(self, keypoints, deterministic_processor=None, rows=0, cols=0, **params):
        if len(keypoints):
            keypoints = convert_keypoints_from_albumentations(keypoints, 'xy', rows=rows, cols=cols)
            keypoints_t = ia.KeypointsOnImage([ia.Keypoint(*kp[:2]) for kp in keypoints], (rows, cols))
            keypoints_t = deterministic_processor.augment_keypoints([keypoints_t])[0].keypoints

            bboxes_t = [[kp.x, kp.y] + list(kp_orig[2:]) for (kp, kp_orig) in zip(keypoints_t, keypoints)]

            keypoints = convert_keypoints_to_albumentations(bboxes_t, 'xy', rows=rows, cols=cols)
        return keypoints


class ImageOnlyIAATransform(ImageOnlyTransform, BasicIAATransform):
    pass


class IAACropAndPad(DualIAATransform):
    def __init__(self, px=None, percent=None, pad_mode='constant', pad_cval=0, keep_size=True,
                 always_apply=False, p=1):
        super(IAACropAndPad, self).__init__(always_apply, p)
        self.px = px
        self.percent = percent
        self.pad_mode = pad_mode
        self.pad_cval = pad_cval
        self.keep_size = keep_size

    @property
    def processor(self):
        return iaa.CropAndPad(self.px, self.percent, self.pad_mode, self.pad_cval, self.keep_size)


class IAAFliplr(DualIAATransform):
    @property
    def processor(self):
        return iaa.Fliplr(1)


class IAAFlipud(DualIAATransform):
    @property
    def processor(self):
        return iaa.Flipud(1)


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

    def __init__(self, alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5):
        super(IAAEmboss, self).__init__(always_apply, p)
        self.alpha = to_tuple(alpha, 0.)
        self.strength = to_tuple(strength, 0.)

    @property
    def processor(self):
        return iaa.Emboss(self.alpha, self.strength)


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

    def __init__(self, p_replace=0.1, n_segments=100, always_apply=False, p=0.5):
        super(IAASuperpixels, self).__init__(always_apply, p)
        self.p_replace = p_replace
        self.n_segments = n_segments

    @property
    def processor(self):
        return iaa.Superpixels(p_replace=self.p_replace, n_segments=self.n_segments)


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

    def __init__(self, alpha=(0.2, 0.5), lightness=(0.5, 1.), always_apply=False, p=0.5):
        super(IAASharpen, self).__init__(always_apply, p)
        self.alpha = to_tuple(alpha, 0)
        self.lightness = to_tuple(lightness, 0)

    @property
    def processor(self):
        return iaa.Sharpen(self.alpha, self.lightness)


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

    def __init__(self, loc=0, scale=(0.01 * 255, 0.05 * 255), per_channel=False, always_apply=False, p=0.5):
        super(IAAAdditiveGaussianNoise, self).__init__(always_apply, p)
        self.loc = loc
        self.scale = to_tuple(scale, 0.)
        self.per_channel = per_channel

    @property
    def processor(self):
        return iaa.AdditiveGaussianNoise(self.loc, self.scale, self.per_channel)


class IAAPiecewiseAffine(DualIAATransform):
    """Place a regular grid of points on the input and randomly move the neighbourhood of these point around
    via affine transformations.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        scale ((float, float): factor range that determines how far each point is moved. Default: (0.03, 0.05).
        nb_rows (int): number of rows of points that the regular grid should have. Default: 4.
        nb_cols (int): number of columns of points that the regular grid should have. Default: 4.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, scale=(0.03, 0.05), nb_rows=4, nb_cols=4, order=1, cval=0, mode='constant',
                 always_apply=False, p=.5):
        super(IAAPiecewiseAffine, self).__init__(always_apply, p)
        self.scale = to_tuple(scale, 0.)
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.order = order
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.PiecewiseAffine(self.scale, self.nb_rows, self.nb_cols, self.order, self.cval, self.mode)


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
                 mode='reflect', always_apply=False, p=0.5):
        super(IAAAffine, self).__init__(always_apply, p)
        self.scale = to_tuple(scale, 1.0)
        self.translate_percent = to_tuple(translate_percent, 0)
        self.translate_px = to_tuple(translate_px, 0)
        self.rotate = to_tuple(rotate)
        self.shear = to_tuple(shear)
        self.order = order
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.Affine(self.scale, self.translate_percent, self.translate_px, self.rotate, self.shear,
                          self.order, self.cval, self.mode)


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

    def __init__(self, scale=(0.05, 0.1), keep_size=True, always_apply=False, p=.5):
        super(IAAPerspective, self).__init__(always_apply, p)
        self.scale = to_tuple(scale, 1.)
        self.keep_size = keep_size

    @property
    def processor(self):
        return iaa.PerspectiveTransform(self.scale, self.keep_size)
