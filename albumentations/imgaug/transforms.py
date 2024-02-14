try:
    import imgaug as ia
except ImportError as e:
    raise ImportError(
        "You are trying to import an augmentation that depends on the imgaug library, but imgaug is not installed. To "
        "install a version of Albumentations that contains imgaug please run 'pip install -U albumentations[imgaug]'"
    ) from e

try:
    from imgaug import augmenters as iaa
except ImportError:
    import imgaug.imgaug.augmenters as iaa

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Type, cast

import numpy as np

from albumentations.core.bbox_utils import (
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
)
from albumentations.core.keypoints_utils import (
    convert_keypoints_from_albumentations,
    convert_keypoints_to_albumentations,
)

from ..augmentations import Perspective
from ..core.transforms_interface import (
    BasicTransform,
    DualTransform,
    ImageOnlyTransform,
    to_tuple,
)
from ..core.types import BoxType, KeypointType, ScaleFloatType

__all__ = [
    "BasicIAATransform",
    "DualIAATransform",
    "ImageOnlyIAATransform",
    "IAAEmboss",
    "IAASuperpixels",
    "IAASharpen",
    "IAAAdditiveGaussianNoise",
    "IAACropAndPad",
    "IAAFliplr",
    "IAAFlipud",
    "IAAAffine",
    "IAAPiecewiseAffine",
    "IAAPerspective",
]


class BasicIAATransform(BasicTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    @property
    def processor(self) -> Type[iaa.Augmenter]:
        return iaa.Noop()

    def update_params(self, params: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        params = super().update_params(params, **kwargs)
        params["deterministic_processor"] = self.processor.to_deterministic()
        return params

    def apply(self, img: np.ndarray, deterministic_processor: Any, **params: Any) -> np.ndarray:
        return deterministic_processor.augment_image(img)


class DualIAATransform(DualTransform, BasicIAATransform):
    def apply_to_bboxes(
        self, bboxes: Sequence[BoxType], deterministic_processor: Any, rows: int = 0, cols: int = 0, **params: Any
    ) -> Sequence[BoxType]:
        if len(bboxes) > 0:
            bboxes = convert_bboxes_from_albumentations(bboxes, "pascal_voc", rows=rows, cols=cols)

            bboxes_t = ia.BoundingBoxesOnImage([ia.BoundingBox(*bbox[:4]) for bbox in bboxes], (rows, cols))
            bboxes_t = deterministic_processor.augment_bounding_boxes([bboxes_t])[0].bounding_boxes
            bboxes_t = [
                [bbox.x1, bbox.y1, bbox.x2, bbox.y2] + list(bbox_orig[4:])
                for (bbox, bbox_orig) in zip(bboxes_t, bboxes)
            ]

            bboxes = convert_bboxes_to_albumentations(bboxes_t, "pascal_voc", rows=rows, cols=cols)
        return bboxes

    """Applies transformation to keypoints.
    Notes:
        Since IAA supports only xy keypoints, scale and orientation will remain unchanged.
    TODO:
        Emit a warning message if child classes of DualIAATransform are instantiated
        inside Compose with keypoints format other than 'xy'.
    """

    def apply_to_keypoints(
        self,
        keypoints: Sequence[KeypointType],
        deterministic_processor: Any,
        rows: int = 0,
        cols: int = 0,
        **params: Any,
    ) -> Sequence[KeypointType]:
        if len(keypoints) > 0:
            keypoints = convert_keypoints_from_albumentations(keypoints, "xy", rows=rows, cols=cols)
            keypoints_t = ia.KeypointsOnImage([ia.Keypoint(*kp[:2]) for kp in keypoints], (rows, cols))
            keypoints_t = deterministic_processor.augment_keypoints([keypoints_t])[0].keypoints

            bboxes_t = [[kp.x, kp.y] + list(kp_orig[2:]) for (kp, kp_orig) in zip(keypoints_t, keypoints)]

            keypoints = convert_keypoints_to_albumentations(
                cast(Sequence[KeypointType], bboxes_t), "xy", rows=rows, cols=cols
            )
        return keypoints


class ImageOnlyIAATransform(ImageOnlyTransform, BasicIAATransform):
    pass


class IAACropAndPad(DualIAATransform):
    """This augmentation is deprecated. Please use CropAndPad instead."""

    def __init__(
        self,
        px: Optional[float] = None,
        percent: Optional[float] = None,
        pad_mode: str = "constant",
        pad_cval: float = 0,
        keep_size: bool = True,
        always_apply: bool = False,
        p: float = 1,
    ):
        super().__init__(always_apply, p)
        self.px = px
        self.percent = percent
        self.pad_mode = pad_mode
        self.pad_cval = pad_cval
        self.keep_size = keep_size
        warnings.warn("IAACropAndPad is deprecated. Please use CropAndPad instead", FutureWarning)

    @property
    def processor(self) -> np.ndarray:
        return iaa.CropAndPad(self.px, self.percent, self.pad_mode, self.pad_cval, self.keep_size)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("px", "percent", "pad_mode", "pad_cval", "keep_size")


class IAAFliplr(DualIAATransform):
    """This augmentation is deprecated. Please use HorizontalFlip instead."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        warnings.warn("IAAFliplr is deprecated. Please use HorizontalFlip instead.", FutureWarning)

    @property
    def processor(self) -> np.ndarray:
        return iaa.Fliplr(1)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()


class IAAFlipud(DualIAATransform):
    """This augmentation is deprecated. Please use VerticalFlip instead."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        warnings.warn("IAAFlipud is deprecated. Please use VerticalFlip instead.", FutureWarning)

    @property
    def processor(self) -> np.ndarray:
        return iaa.Flipud(1)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()


class IAAEmboss(ImageOnlyIAATransform):
    """Emboss the input image and overlays the result with the original image.
    This augmentation is deprecated. Please use Emboss instead.

    Args:
        alpha ((float, float)): range to choose the visibility of the embossed image. At 0, only the original image is
            visible,at 1.0 only its embossed version is visible. Default: (0.2, 0.5).
        strength ((float, float)): strength range of the embossing. Default: (0.2, 0.7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(
        self,
        alpha: Tuple[float, float] = (0.2, 0.5),
        strength: Tuple[float, float] = (0.2, 0.7),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.alpha = to_tuple(alpha, 0.0)
        self.strength = to_tuple(strength, 0.0)
        warnings.warn("This augmentation is deprecated. Please use Emboss instead", FutureWarning)

    @property
    def processor(self) -> np.ndarray:
        return iaa.Emboss(self.alpha, self.strength)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "alpha", "strength"


class IAASuperpixels(ImageOnlyIAATransform):
    """Completely or partially transform the input image to its superpixel representation. Uses skimage's version
    of the SLIC algorithm. May be slow.

    This augmentation is deprecated. Please use Superpixels instead.

    Args:
        p_replace (float): defines the probability of any superpixel area being replaced by the superpixel, i.e. by
            the average pixel color within its area. Default: 0.1.
        n_segments (int): target number of superpixels to generate. Default: 100.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(self, p_replace: float = 0.1, n_segments: int = 100, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.p_replace = p_replace
        self.n_segments = n_segments
        warnings.warn("IAASuperpixels is deprecated. Please use Superpixels instead.", FutureWarning)

    @property
    def processor(self) -> np.ndarray:
        return iaa.Superpixels(p_replace=self.p_replace, n_segments=self.n_segments)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "p_replace", "n_segments"


class IAASharpen(ImageOnlyIAATransform):
    """Sharpen the input image and overlays the result with the original image.
    This augmentation is deprecated. Please use Sharpen instead
    Args:
        alpha ((float, float)): range to choose the visibility of the sharpened image. At 0, only the original image is
            visible, at 1.0 only its sharpened version is visible. Default: (0.2, 0.5).
        lightness ((float, float)): range to choose the lightness of the sharpened image. Default: (0.5, 1.0).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(
        self,
        alpha: Tuple[float, float] = (0.2, 0.5),
        lightness: Tuple[float, float] = (0.5, 1.0),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.alpha = to_tuple(alpha, 0)
        self.lightness = to_tuple(lightness, 0)
        warnings.warn("IAASharpen is deprecated. Please use Sharpen instead", FutureWarning)

    @property
    def processor(self) -> np.ndarray:
        return iaa.Sharpen(self.alpha, self.lightness)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return "alpha", "lightness"


class IAAAdditiveGaussianNoise(ImageOnlyIAATransform):
    """Add gaussian noise to the input image.

    This augmentation is deprecated. Please use GaussNoise instead.

    Args:
        loc: mean of the normal distribution that generates the noise. Default: 0.
        scale: standard deviation of the normal distribution that generates the noise.
            Default: (0.01 * 255, 0.05 * 255).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image
    """

    def __init__(
        self,
        loc: int = 0,
        scale: Tuple[float, float] = (0.01 * 255, 0.05 * 255),
        per_channel: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.loc = loc
        self.scale = to_tuple(scale, 0.0)
        self.per_channel = per_channel
        warnings.warn("IAAAdditiveGaussianNoise is deprecated. Please use GaussNoise instead", FutureWarning)

    @property
    def processor(self) -> np.ndarray:
        return iaa.AdditiveGaussianNoise(self.loc, self.scale, self.per_channel)

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return "loc", "scale", "per_channel"


class IAAPiecewiseAffine(DualIAATransform):
    """Place a regular grid of points on the input and randomly move the neighborhood of these point around
    via affine transformations.

    This augmentation is deprecated. Please use PiecewiseAffine instead.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        scale: factor range that determines how far each point is moved. Default: (0.03, 0.05).
        nb_rows: number of rows of points that the regular grid should have. Default: 4.
        nb_cols: number of columns of points that the regular grid should have. Default: 4.
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(
        self,
        scale: Tuple[float, float] = (0.03, 0.05),
        nb_rows: int = 4,
        nb_cols: int = 4,
        order: int = 1,
        cval: int = 0,
        mode: str = "constant",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.scale = to_tuple(scale, 0.0)
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.order = order
        self.cval = cval
        self.mode = mode
        warnings.warn("This IAAPiecewiseAffine is deprecated. Please use PiecewiseAffine instead", FutureWarning)

    @property
    def processor(self) -> np.ndarray:
        return iaa.PiecewiseAffine(self.scale, self.nb_rows, self.nb_cols, self.order, self.cval, self.mode)

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str, str, str]:
        return ("scale", "nb_rows", "nb_cols", "order", "cval", "mode")


class IAAAffine(DualIAATransform):
    """Place a regular grid of points on the input and randomly move the neighborhood of these point around
    via affine transformations.

    This augmentation is deprecated. Please use Affine instead.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(
        self,
        scale: float,
        translate_percent: float,
        translate_px: float,
        rotate: float = 0.0,
        shear: float = 0.0,
        order: int = 1,
        cval: float = 0,
        mode: str = "reflect",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.scale = to_tuple(scale, 1.0)
        self.translate_percent = to_tuple(translate_percent, 0)
        self.translate_px = to_tuple(translate_px, 0)
        self.rotate = to_tuple(rotate)
        self.shear = to_tuple(shear)
        self.order = order
        self.cval = cval
        self.mode = mode
        warnings.warn("This IAAAffine is deprecated. Please use Affine instead", FutureWarning)

    @property
    def processor(self) -> np.ndarray:
        return iaa.Affine(
            self.scale,
            self.translate_percent,
            self.translate_px,
            self.rotate,
            self.shear,
            self.order,
            self.cval,
            self.mode,
        )

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("scale", "translate_percent", "translate_px", "rotate", "shear", "order", "cval", "mode")


class IAAPerspective(Perspective):
    """Perform a random four point perspective transform of the input.
    This augmentation is deprecated. Please use Perspective instead.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        scale: standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners. Default: (0.05, 0.1).
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(
        self,
        scale: ScaleFloatType = (0.05, 0.1),
        keep_size: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)  # type: ignore[arg-type]
        self.scale = to_tuple(scale, 1.0)
        self.keep_size = keep_size
        warnings.warn("This IAAPerspective is deprecated. Please use Perspective instead", FutureWarning)

    @property
    def processor(self) -> np.ndarray:
        return iaa.PerspectiveTransform(self.scale, keep_size=self.keep_size)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return "scale", "keep_size"
