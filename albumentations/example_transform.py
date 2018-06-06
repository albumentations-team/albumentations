from .composition import Compose, OneOf
from .transforms import *
from .iaa_transforms import *


def augment_flips_color(p=.5):
    return Compose([
        # CLAHE(),
        RandomRotate90(),
        Transpose(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        # Blur(blur_limit=3),
        # Distort1(),
        # Distort2(),
        HueSaturationValue()
    ], p=p)


def strong_aug(p=.5):
    return Compose([
        # ToGray(p=0.3),
        # InvertImg(p=0.1),
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            Distort1(p=0.3),
            Distort2(p=.1),
            IAAPiecewiseAffine(p=0.3),
            # ElasticTransform(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clipLimit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        # FixMasks(1.),
        # AddChannel(1.)
    ], p=p)