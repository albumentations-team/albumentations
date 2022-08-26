import numbers
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from albumentations.augmentations.color import functional as F
from albumentations.augmentations.utils import (
    from_float,
    is_grayscale_image,
    is_rgb_image,
    to_float,
)
from albumentations.core.transforms_interface import (
    ImageColorType,
    ImageOnlyTransform,
    ScaleFloatType,
    ScaleIntType,
    to_tuple,
)

__all__ = [
    "RandomGamma",
    "Normalize",
    "HueSaturationValue",
    "RGBShift",
    "RandomBrightness",
    "RandomContrast",
    "RandomBrightnessContrast",
    "CLAHE",
    "ChannelShuffle",
    "InvertImg",
    "ToFloat",
    "FromFloat",
    "ToGray",
    "ToSepia",
    "Solarize",
    "Equalize",
    "Posterize",
    "ColorJitter",
    "FancyPCA",
]


class RandomGamma(ImageOnlyTransform):
    """
    Args:
        gamma_limit (float or (float, float)): If gamma_limit is a single float value,
            the range will be (-gamma_limit, gamma_limit). Default: (80, 120).
        eps: Deprecated.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self, gamma_limit: ScaleFloatType = (80, 120), eps: None = None, always_apply: bool = False, p: float = 0.5
    ):
        super().__init__(always_apply, p)
        self.gamma_limit = to_tuple(gamma_limit)
        self.eps = eps

    def apply(self, img: np.ndarray, gamma: float = 1, **params) -> np.ndarray:
        return F.gamma_transform(img, gamma=gamma)

    def get_params(self) -> Dict[str, float]:
        return {"gamma": random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("gamma_limit", "eps")


class Normalize(ImageOnlyTransform):
    """Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`

    Args:
        mean (float, list of float): mean values
        std  (float, list of float): std values
        max_pixel_value (float): maximum possible pixel value

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        mean: ImageColorType = (0.485, 0.456, 0.406),
        std: ImageColorType = (0.229, 0.224, 0.225),
        max_pixel_value: float = 255.0,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.normalize(img, self.mean, self.std, self.max_pixel_value)

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("mean", "std", "max_pixel_value")


class HueSaturationValue(ImageOnlyTransform):
    """Randomly change hue, saturation and value of the input image.

    Args:
        hue_shift_limit ((float, float) or float): range for changing hue. If hue_shift_limit is a single float,
            the range will be (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
        sat_shift_limit ((float, float) or float): range for changing saturation. If sat_shift_limit is a single float,
            the range will be (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
        val_shift_limit ((float, float) or float): range for changing value. If val_shift_limit is a single float,
            the range will be (-val_shift_limit, val_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        hue_shift_limit: ScaleFloatType = 20,
        sat_shift_limit: ScaleFloatType = 30,
        val_shift_limit: ScaleFloatType = 20,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(
        self, img: np.ndarray, hue_shift: float = 0, sat_shift: float = 0, val_shift: float = 0, **params
    ) -> np.ndarray:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            raise TypeError("HueSaturationValue transformation expects 1-channel or 3-channel images.")
        return F.shift_hsv(img, hue_shift, sat_shift, val_shift)

    def get_params(self) -> Dict[str, float]:
        return {
            "hue_shift": random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1]),
            "sat_shift": random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1]),
            "val_shift": random.uniform(self.val_shift_limit[0], self.val_shift_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("hue_shift_limit", "sat_shift_limit", "val_shift_limit")


class RGBShift(ImageOnlyTransform):
    """Randomly shift values for each channel of the input RGB image.

    Args:
        r_shift_limit ((float, float) or float): range for changing values for the red channel.
            If r_shift_limit is a single float, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).
        g_shift_limit ((float, float) or float): range for changing values for the green channel. If g_shift_limit is a
            single float, the range  will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).
        b_shift_limit ((float, float) or float): range for changing values for the blue channel.
            If b_shift_limit is a single float, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        r_shift_limit: ScaleFloatType = 20,
        g_shift_limit: ScaleFloatType = 20,
        b_shift_limit: ScaleFloatType = 20,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def apply(
        self, img: np.ndarray, r_shift: float = 0, g_shift: float = 0, b_shift: float = 0, **params
    ) -> np.ndarray:
        if not is_rgb_image(img):
            raise TypeError("RGBShift transformation expects 3-channel images.")
        return F.shift_rgb(img, r_shift, g_shift, b_shift)

    def get_params(self) -> Dict[str, float]:
        return {
            "r_shift": random.uniform(self.r_shift_limit[0], self.r_shift_limit[1]),
            "g_shift": random.uniform(self.g_shift_limit[0], self.g_shift_limit[1]),
            "b_shift": random.uniform(self.b_shift_limit[0], self.b_shift_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("r_shift_limit", "g_shift_limit", "b_shift_limit")


class RandomBrightnessContrast(ImageOnlyTransform):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        brightness_limit: ScaleFloatType = 0.2,
        contrast_limit: ScaleFloatType = 0.2,
        brightness_by_max: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.brightness_limit = to_tuple(brightness_limit)
        self.contrast_limit = to_tuple(contrast_limit)
        self.brightness_by_max = brightness_by_max

    def apply(self, img: np.ndarray, alpha: float = 1.0, beta: float = 0.0, **params) -> np.ndarray:
        return F.brightness_contrast_adjust(img, alpha, beta, self.brightness_by_max)

    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("brightness_limit", "contrast_limit", "brightness_by_max")


class RandomContrast(RandomBrightnessContrast):
    """Randomly change contrast of the input image.

    Args:
        limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, limit: ScaleFloatType = 0.2, always_apply: bool = False, p: float = 0.5):
        super().__init__(brightness_limit=0, contrast_limit=limit, always_apply=always_apply, p=p)
        warnings.warn(
            f"{self.__class__.__name__} has been deprecated. Please use RandomBrightnessContrast",
            FutureWarning,
        )

    def get_transform_init_args(self) -> Dict[str, Any]:
        return {"limit": self.contrast_limit}


class RandomBrightness(RandomBrightnessContrast):
    """Randomly change brightness of the input image.

    Args:
        limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, limit: ScaleFloatType = 0.2, always_apply: bool = False, p: float = 0.5):
        super().__init__(brightness_limit=limit, contrast_limit=0, always_apply=always_apply, p=p)
        warnings.warn(
            "This class has been deprecated. Please use RandomBrightnessContrast",
            FutureWarning,
        )

    def get_transform_init_args(self) -> Dict[str, Any]:
        return {"limit": self.brightness_limit}


class CLAHE(ImageOnlyTransform):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input image.

    Args:
        clip_limit (float or (float, float)): upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be (1, clip_limit). Default: (1, 4).
        tile_grid_size ((int, int)): size of grid for histogram equalization. Default: (8, 8).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self,
        clip_limit: ScaleFloatType = 4.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.clip_limit = to_tuple(clip_limit, 1)
        self.tile_grid_size = cast(Tuple[int, int], tuple(tile_grid_size))

    def apply(self, img: np.ndarray, clip_limit: float = 2, **params) -> np.ndarray:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            raise TypeError("CLAHE transformation expects 1-channel or 3-channel images.")

        return F.clahe(img, clip_limit, self.tile_grid_size)

    def get_params(self) -> Dict[str, float]:
        return {"clip_limit": random.uniform(self.clip_limit[0], self.clip_limit[1])}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("clip_limit", "tile_grid_size")


class ChannelShuffle(ImageOnlyTransform):
    """Randomly rearrange channels of the input RGB image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]

    def apply(self, img: np.ndarray, channels_shuffled: Sequence[int] = (0, 1, 2), **params) -> np.ndarray:
        return F.channel_shuffle(img, channels_shuffled)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, List[int]]:
        img = params["image"]
        ch_arr = list(range(img.shape[2]))
        random.shuffle(ch_arr)
        return {"channels_shuffled": ch_arr}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()


class InvertImg(ImageOnlyTransform):
    """Invert the input image by subtracting pixel values from 255.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.invert(img)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()


class ToFloat(ImageOnlyTransform):
    """Divide pixel values by `max_value` to get a float32 output array where all values lie in the range [0, 1.0].
    If `max_value` is None the transform will try to infer the maximum value by inspecting the data type of the input
    image.

    See Also:
        :class:`~albumentations.augmentations.transforms.FromFloat`

    Args:
        max_value (float): maximum possible input value. Default: None.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        any type

    """

    def __init__(self, max_value: Optional[float] = None, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return to_float(img, self.max_value)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("max_value",)


class FromFloat(ImageOnlyTransform):
    """Take an input array where all values should lie in the range [0, 1.0], multiply them by `max_value` and then
    cast the resulted value to a type specified by `dtype`. If `max_value` is None the transform will try to infer
    the maximum value for the data type from the `dtype` argument.

    This is the inverse transform for :class:`~albumentations.augmentations.transforms.ToFloat`.

    Args:
        max_value (float): maximum possible input value. Default: None.
        dtype (string or numpy data type): data type of the output. See the `'Data types' page from the NumPy docs`_.
            Default: 'uint16'.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image

    Image types:
        float32

    .. _'Data types' page from the NumPy docs:
       https://docs.scipy.org/doc/numpy/user/basics.types.html
    """

    def __init__(
        self,
        dtype: Union[str, np.dtype] = "uint16",
        max_value: Optional[float] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.dtype = np.dtype(dtype)
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return from_float(img, self.dtype, self.max_value)

    def get_transform_init_args(self) -> Dict[str, Any]:
        return {"dtype": self.dtype.name, "max_value": self.max_value}


class ToGray(ImageOnlyTransform):
    """Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater
    than 127, invert the resulting grayscale image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if is_grayscale_image(img):
            warnings.warn("The image is already gray.")
            return img
        if not is_rgb_image(img):
            raise TypeError("ToGray transformation expects 3-channel images.")

        return F.to_gray(img)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()


class ToSepia(ImageOnlyTransform):
    """Applies sepia filter to the input RGB image

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.sepia_transformation_matrix = np.matrix(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
        )

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if not is_rgb_image(img):
            raise TypeError("ToSepia transformation expects 3-channel images.")
        return F.linear_transformation_rgb(img, self.sepia_transformation_matrix)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ()


class Solarize(ImageOnlyTransform):
    """Invert all pixel values above a threshold.

    Args:
        threshold ((int, int) or int, or (float, float) or float): range for solarizing threshold.
            If threshold is a single value, the range will be [threshold, threshold]. Default: 128.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        any
    """

    def __init__(self, threshold: ScaleFloatType = 128, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

        if isinstance(threshold, (int, float)):
            self.threshold = to_tuple(threshold, low=threshold)
        else:
            self.threshold = to_tuple(threshold, low=0)

    def apply(self, img: np.ndarray, threshold: float = 0, **params) -> np.ndarray:
        return F.solarize(img, threshold)

    def get_params(self) -> Dict[str, float]:
        return {"threshold": random.uniform(self.threshold[0], self.threshold[1])}

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("threshold",)


class Equalize(ImageOnlyTransform):
    """Equalize the image histogram.

    Args:
        mode (str): {'cv', 'pil'}. Use OpenCV or Pillow equalization method.
        by_channels (bool): If True, use equalization by channels separately,
            else convert image to YCbCr representation and use equalization by `Y` channel.
        mask (np.ndarray, callable): If given, only the pixels selected by
            the mask are included in the analysis. Maybe 1 channel or 3 channel array or callable.
            Function signature must include `image` argument.
        mask_params (sequence of str): Params for mask function.

    Targets:
        image

    Image types:
        uint8
    """

    def __init__(
        self,
        mode: str = "cv",
        by_channels: bool = True,
        mask: Optional[Union[np.ndarray, Callable[..., np.ndarray]]] = None,
        mask_params: Sequence[str] = (),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        modes = ["cv", "pil"]
        if mode not in modes:
            raise ValueError(f"Unsupported equalization mode. Supports: {modes}. Got: {mode}")

        super().__init__(always_apply, p)
        self.mode = mode
        self.by_channels = by_channels
        self.mask = mask
        self.mask_params = mask_params

    def apply(self, img: np.ndarray, mask: Optional[np.ndarray] = None, **params) -> np.ndarray:
        return F.equalize(img, mode=self.mode, by_channels=self.by_channels, mask=mask)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if not callable(self.mask):
            return {"mask": self.mask}

        return {"mask": self.mask(**params)}

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"] + list(self.mask_params)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("mode", "by_channels")


class Posterize(ImageOnlyTransform):
    """Reduce the number of bits for each color channel.

    Args:
        num_bits ((int, int) or int,
                  or list of ints [r, g, b],
                  or list of ints [[r1, r1], [g1, g2], [b1, b2]]): number of high bits.
            If num_bits is a single value, the range will be [num_bits, num_bits].
            Must be in range [0, 8]. Default: 4.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
    image

    Image types:
        uint8
    """

    def __init__(self, num_bits: ScaleIntType = 4, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

        if isinstance(num_bits, (list, tuple)):
            if len(num_bits) == 3:
                self.num_bits = [to_tuple(i, 0) for i in num_bits]
            else:
                self.num_bits = to_tuple(num_bits, 0)
        else:
            self.num_bits = to_tuple(num_bits, num_bits)

    def apply(self, img: np.ndarray, num_bits: Union[int, Sequence[int]] = 1, **params) -> np.ndarray:
        return F.posterize(img, num_bits)

    def get_params(self) -> Dict[str, Union[int, Sequence[int]]]:
        if len(self.num_bits) == 3:
            return {"num_bits": [random.randint(i[0], i[1]) for i in self.num_bits]}
        return {"num_bits": random.randint(self.num_bits[0], self.num_bits[1])}

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("num_bits",)


class ColorJitter(ImageOnlyTransform):
    """Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision,
    this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in
    Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8
    overflow, but we use value saturation.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(
        self,
        brightness: ScaleFloatType = 0.2,
        contrast: ScaleFloatType = 0.2,
        saturation: ScaleFloatType = 0.2,
        hue: ScaleFloatType = 0.2,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        self.brightness = self.__check_values(brightness, "brightness")
        self.contrast = self.__check_values(contrast, "contrast")
        self.saturation = self.__check_values(saturation, "saturation")
        self.hue = self.__check_values(hue, "hue", offset=0, bounds=(-0.5, 0.5), clip=False)

        self.transforms: List[Callable[[np.ndarray, float], np.ndarray]] = [
            F.adjust_brightness_torchvision,
            F.adjust_contrast_torchvision,
            F.adjust_saturation_torchvision,
            F.adjust_hue_torchvision,
        ]

    @staticmethod
    def __check_values(
        value: ScaleFloatType,
        name: str,
        offset: float = 1,
        bounds: Tuple[float, float] = (0, float("inf")),
        clip: bool = True,
    ) -> Tuple[float, float]:
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [offset - value, offset + value]
            if clip:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError(f"{name} values should be between {bounds}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        return value

    def get_params(self) -> Dict[str, Any]:
        brightness = random.uniform(self.brightness[0], self.brightness[1])
        contrast = random.uniform(self.contrast[0], self.contrast[1])
        saturation = random.uniform(self.saturation[0], self.saturation[1])
        hue = random.uniform(self.hue[0], self.hue[1])

        order = [0, 1, 2, 3]
        random.shuffle(order)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "order": tuple(order),
        }

    def apply(
        self,
        img: np.ndarray,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        hue: float = 0,
        order: Tuple[int, int, int, int] = (0, 1, 2, 3),
        **params
    ) -> np.ndarray:
        if not is_rgb_image(img) and not is_grayscale_image(img):
            raise TypeError("ColorJitter transformation expects 1-channel or 3-channel images.")
        call_params = [brightness, contrast, saturation, hue]
        for i in order:
            img = self.transforms[i](img, call_params[i])
        return img

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str]:
        return ("brightness", "contrast", "saturation", "hue")


class FancyPCA(ImageOnlyTransform):
    """Augment RGB image using FancyPCA from Krizhevsky's paper
    "ImageNet Classification with Deep Convolutional Neural Networks"

    Args:
        alpha (float):  how much to perturb/scale the eigen vecs and vals.
            scale is samples from gaussian distribution (mu=0, sigma=alpha)

    Targets:
        image

    Image types:
        3-channel uint8 images only

    Credit:
        http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
        https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
    """

    def __init__(self, alpha: float = 0.1, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, img: np.ndarray, alpha: float = 0.1, **params) -> np.ndarray:
        img = F.fancy_pca(img, alpha)
        return img

    def get_params(self) -> Dict[str, float]:
        return {"alpha": random.gauss(0, self.alpha)}

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("alpha",)
