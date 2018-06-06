from imgaug import augmenters as iaa


from ..core.transforms_interface import BasicTransform, DualTransform, ImageOnlyTransform


__all__ = ['BasicIAATransform', 'DualIAATransform', 'ImageOnlyIAATransform',
           'IAAEmboss', 'IAASuperpixels', 'IAASharpen',
           'IAAAdditiveGaussianNoise', 'IAAPiecewiseAffine', 'IAAPerspective']


class BasicIAATransform(BasicTransform):
    def __init__(self, p=0.5):
        super().__init__(p)
        self.processor = iaa.Noop()
        self.deterministic_processor = iaa.Noop()

    def __call__(self, **kwargs):
        self.deterministic_processor = self.processor.to_deterministic()
        return super().__call__(**kwargs)

    def apply(self, img, **params):
        return self.deterministic_processor.augment_image(img)


class DualIAATransform(DualTransform, BasicIAATransform):
    pass


class ImageOnlyIAATransform(ImageOnlyTransform, BasicIAATransform):
    pass


class IAAEmboss(ImageOnlyIAATransform):
    def __init__(self, alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5):
        super().__init__(p)
        self.processor = iaa.Emboss(alpha, strength)


class IAASuperpixels(ImageOnlyIAATransform):
    '''
    may be slow
    '''

    def __init__(self, p_replace=0.1, n_segments=100, p=0.5):
        super().__init__(p)
        self.processor = iaa.Superpixels(p_replace=p_replace, n_segments=n_segments)


class IAASharpen(ImageOnlyIAATransform):
    def __init__(self, alpha=(0.2, 0.5), lightness=(0.5, 1.), p=0.5):
        super().__init__(p)
        self.processor = iaa.Sharpen(alpha, lightness)


class IAAAdditiveGaussianNoise(ImageOnlyIAATransform):
    def __init__(self, loc=0, scale=(0.01 * 255, 0.05 * 255), p=0.5):
        super().__init__(p)
        self.processor = iaa.AdditiveGaussianNoise(loc, scale)


class IAAPiecewiseAffine(DualIAATransform):
    def __init__(self, scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=.5):
        super().__init__(p)
        self.processor = iaa.PiecewiseAffine(scale, nb_rows, nb_cols)


class IAAPerspective(DualIAATransform):
    def __init__(self, scale=(0.05, 0.1), p=.5):
        super().__init__(p)
        self.processor = iaa.PerspectiveTransform(scale)
