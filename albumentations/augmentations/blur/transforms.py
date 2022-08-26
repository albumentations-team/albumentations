import random
import warnings
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from albumentations import random_utils
from albumentations.augmentations import functional as FMain
from albumentations.augmentations.blur import functional as F
from albumentations.core.transforms_interface import (
    ImageOnlyTransform,
    ScaleFloatType,
    ScaleIntType,
    to_tuple,
)

__all__ = ["Blur", "MotionBlur", "GaussianBlur", "GlassBlur", "AdvancedBlur", "MedianBlur", "Defocus", "ZoomBlur"]


class Blur(ImageOnlyTransform):
    """Blur the input image using a random-sized kernel.

    Args:
        blur_limit (int, (int, int)): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit: ScaleIntType = 7, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)

    def apply(self, img: np.ndarray, ksize: int = 3, **params) -> np.ndarray:
        return F.blur(img, ksize)

    def get_params(self) -> Dict[str, Any]:
        return {"ksize": int(random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2)))}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("blur_limit",)


class MotionBlur(Blur):
    """Apply motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        allow_shifted (bool): if set to true creates non shifted kernels only,
            otherwise creates randomly shifted kernels. Default: True.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: ScaleIntType = 7,
        allow_shifted: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(blur_limit=blur_limit, always_apply=always_apply, p=p)
        self.allow_shifted = allow_shifted

        if not allow_shifted and self.blur_limit[0] % 2 != 1 or self.blur_limit[1] % 2 != 1:
            raise ValueError(f"Blur limit must be odd when centered=True. Got: {self.blur_limit}")

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return super().get_transform_init_args_names() + ("allow_shifted",)

    def apply(self, img: np.ndarray, kernel: np.ndarray = None, **params) -> np.ndarray:  # type: ignore
        return FMain.convolve(img, kernel=kernel)

    def get_params(self) -> Dict[str, Any]:
        ksize = random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        if ksize <= 2:
            raise ValueError("ksize must be > 2. Got: {}".format(ksize))
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        x1, x2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        if x1 == x2:
            y1, y2 = random.sample(range(ksize), 2)
        else:
            y1, y2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)

        def make_odd_val(v1, v2):
            len_v = abs(v1 - v2) + 1
            if len_v % 2 != 1:
                if v2 > v1:
                    v2 -= 1
                else:
                    v1 -= 1
            return v1, v2

        if not self.allow_shifted:
            x1, x2 = make_odd_val(x1, x2)
            y1, y2 = make_odd_val(y1, y2)

            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2

            center = ksize / 2 - 0.5
            dx = xc - center
            dy = yc - center
            x1, x2 = [int(i - dx) for i in [x1, x2]]
            y1, y2 = [int(i - dy) for i in [y1, y2]]

        cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)

        # Normalize kernel
        return {"kernel": kernel.astype(np.float32) / np.sum(kernel)}


class MedianBlur(Blur):
    """Blur the input image using a median filter with a random aperture linear size.

    Args:
        blur_limit (int): maximum aperture linear size for blurring the input image.
            Must be odd and in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit: ScaleIntType = 7, always_apply: bool = False, p: float = 0.5):
        super().__init__(blur_limit, always_apply, p)

        if self.blur_limit[0] % 2 != 1 or self.blur_limit[1] % 2 != 1:
            raise ValueError("MedianBlur supports only odd blur limits.")

    def apply(self, img: np.ndarray, ksize: int = 3, **params) -> np.ndarray:
        return F.median_blur(img, ksize)


class GaussianBlur(ImageOnlyTransform):
    """Blur the input image using a Gaussian filter with a random kernel size.

    Args:
        blur_limit (int, (int, int)): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: ScaleIntType = (3, 7),
        sigma_limit: ScaleFloatType = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 0)
        self.sigma_limit = to_tuple(sigma_limit if sigma_limit is not None else 0, 0)

        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            warnings.warn(
                "blur_limit and sigma_limit minimum value can not be both equal to 0. "
                "blur_limit minimum value changed to 3."
            )

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("GaussianBlur supports only odd blur limits.")

    def apply(self, img: np.ndarray, ksize: int = 3, sigma: float = 0, **params) -> np.ndarray:
        return F.gaussian_blur(img, ksize, sigma=sigma)

    def get_params(self) -> Dict[str, float]:
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1)
        if ksize != 0 and ksize % 2 != 1:
            ksize = (ksize + 1) % (self.blur_limit[1] + 1)

        return {"ksize": ksize, "sigma": random.uniform(*self.sigma_limit)}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("blur_limit", "sigma_limit")


class GlassBlur(Blur):
    """Apply glass noise to the input image.

    Args:
        sigma (float): standard deviation for Gaussian kernel.
        max_delta (int): max distance between pixels which are swapped.
        iterations (int): number of repeats.
            Should be in range [1, inf). Default: (2).
        mode (str): mode of computation: fast or exact. Default: "fast".
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1903.12261
    |  https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
    """

    def __init__(
        self,
        sigma: float = 0.7,
        max_delta: int = 4,
        iterations: int = 2,
        always_apply: bool = False,
        mode: str = "fast",
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        if iterations < 1:
            raise ValueError(f"Iterations should be more or equal to 1, but we got {iterations}")

        if mode not in ["fast", "exact"]:
            raise ValueError(f"Mode should be 'fast' or 'exact', but we got {mode}")

        self.sigma = sigma
        self.max_delta = max_delta
        self.iterations = iterations
        self.mode = mode

    def apply(self, img: np.ndarray, dxy: np.ndarray = None, **params) -> np.ndarray:  # type: ignore
        assert dxy is not None
        return F.glass_blur(img, self.sigma, self.max_delta, self.iterations, dxy, self.mode)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        img = params["image"]

        # generate array containing all necessary values for transformations
        width_pixels = img.shape[0] - self.max_delta * 2
        height_pixels = img.shape[1] - self.max_delta * 2
        total_pixels = width_pixels * height_pixels
        dxy = random_utils.randint(-self.max_delta, self.max_delta, size=(total_pixels, self.iterations, 2))

        return {"dxy": dxy}

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("sigma", "max_delta", "iterations")

    @property
    def targets_as_params(self) -> List[str]:
        return ["image"]


class AdvancedBlur(ImageOnlyTransform):
    """Blur the input image using a Generalized Normal filter with a randomly selected parameters.
        This transform also adds multiplicative noise to generated kernel before convolution.

    Args:
        blur_limit: maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigmaX_limit: Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigmaX_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        sigmaY_limit: Same as `sigmaY_limit` for another dimension.
        rotate_limit: Range from which a random angle used to rotate Gaussian kernel is picked.
            If limit is a single int an angle is picked from (-rotate_limit, rotate_limit). Default: (-90, 90).
        beta_limit: Distribution shape parameter, 1 is the normal distribution. Values below 1.0 make distribution
            tails heavier than normal, values above 1.0 make it lighter than normal. Default: (0.5, 8.0).
        noise_limit: Multiplicative factor that control strength of kernel noise. Must be positive and preferably
            centered around 1.0. If set single value `noise_limit` will be in range (0, noise_limit).
            Default: (0.75, 1.25).
        p (float): probability of applying the transform. Default: 0.5.

    Reference:
        https://arxiv.org/abs/2107.10833

    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: ScaleIntType = (3, 7),
        sigmaX_limit: ScaleFloatType = (0.2, 1.0),
        sigmaY_limit: ScaleFloatType = (0.2, 1.0),
        rotate_limit: ScaleIntType = 90,
        beta_limit: ScaleFloatType = (0.5, 8.0),
        noise_limit: ScaleFloatType = (0.9, 1.1),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)
        self.sigmaX_limit = self.__check_values(to_tuple(sigmaX_limit, 0.0), name="sigmaX_limit")
        self.sigmaY_limit = self.__check_values(to_tuple(sigmaY_limit, 0.0), name="sigmaY_limit")
        self.rotate_limit = to_tuple(rotate_limit)
        self.beta_limit = to_tuple(beta_limit, low=0.0)
        self.noise_limit = self.__check_values(to_tuple(noise_limit, 0.0), name="noise_limit")

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("AdvancedBlur supports only odd blur limits.")

        if self.sigmaX_limit[0] == 0 and self.sigmaY_limit[0] == 0:
            raise ValueError("sigmaX_limit and sigmaY_limit minimum value can not be both equal to 0.")

        if not (self.beta_limit[0] < 1.0 < self.beta_limit[1]):
            raise ValueError("Beta limit is expected to include 1.0")

    @staticmethod
    def __check_values(
        value: Sequence[float], name: str, bounds: Tuple[float, float] = (0, float("inf"))
    ) -> Sequence[float]:
        if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
            raise ValueError(f"{name} values should be between {bounds}")
        return value

    def apply(self, img: np.ndarray, kernel: np.ndarray = None, **params) -> np.ndarray:
        return FMain.convolve(img, kernel=kernel)

    def get_params(self) -> Dict[str, np.ndarray]:
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        sigmaX = random.uniform(*self.sigmaX_limit)
        sigmaY = random.uniform(*self.sigmaY_limit)
        angle = np.deg2rad(random.uniform(*self.rotate_limit))

        # Split into 2 cases to avoid selection of narrow kernels (beta > 1) too often.
        if random.random() < 0.5:
            beta = random.uniform(self.beta_limit[0], 1)
        else:
            beta = random.uniform(1, self.beta_limit[1])

        noise_matrix = random_utils.uniform(self.noise_limit[0], self.noise_limit[1], size=[ksize, ksize])

        # Generate mesh grid centered at zero.
        ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
        # Shape (ksize, ksize, 2)
        grid = np.stack(np.meshgrid(ax, ax), axis=-1)

        # Calculate rotated sigma matrix
        d_matrix = np.array([[sigmaX**2, 0], [0, sigmaY**2]])
        u_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        sigma_matrix = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

        inverse_sigma = np.linalg.inv(sigma_matrix)
        # Described in "Parameter Estimation For Multivariate Generalized Gaussian Distributions"
        kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
        # Add noise
        kernel = kernel * noise_matrix

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        return {"kernel": kernel}

    def get_transform_init_args_names(self) -> Tuple[str, str, str, str, str, str]:
        return (
            "blur_limit",
            "sigmaX_limit",
            "sigmaY_limit",
            "rotate_limit",
            "beta_limit",
            "noise_limit",
        )


class Defocus(ImageOnlyTransform):
    """
    Apply defocus transform. See https://arxiv.org/abs/1903.12261.

    Args:
        radius ((int, int) or int): range for radius of defocusing.
            If limit is a single int, the range will be [1, limit]. Default: (3, 10).
        alias_blur ((float, float) or float): range for alias_blur of defocusing (sigma of gaussian blur).
            If limit is a single float, the range will be (0, limit). Default: (0.1, 0.5).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        Any
    """

    def __init__(
        self,
        radius: ScaleIntType = (3, 10),
        alias_blur: ScaleFloatType = (0.1, 0.5),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.radius = to_tuple(radius, low=1)
        self.alias_blur = to_tuple(alias_blur, low=0)

        if self.radius[0] <= 0:
            raise ValueError("Parameter radius must be positive")

        if self.alias_blur[0] < 0:
            raise ValueError("Parameter alias_blur must be non-negative")

    def apply(self, img: np.ndarray, radius: int = 3, alias_blur: float = 0.5, **params) -> np.ndarray:
        return F.defocus(img, radius, alias_blur)

    def get_params(self) -> Dict[str, Any]:
        return {
            "radius": random_utils.randint(self.radius[0], self.radius[1] + 1),
            "alias_blur": random_utils.uniform(self.alias_blur[0], self.alias_blur[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("radius", "alias_blur")


class ZoomBlur(ImageOnlyTransform):
    """
    Apply zoom blur transform. See https://arxiv.org/abs/1903.12261.

    Args:
        max_factor ((float, float) or float): range for max factor for blurring.
            If max_factor is a single float, the range will be (1, limit). Default: (1, 1.31).
            All max_factor values should be larger than 1.
        step_factor ((float, float) or float): If single float will be used as step parameter for np.arange.
            If tuple of float step_factor will be in range `[step_factor[0], step_factor[1])`. Default: (0.01, 0.03).
            All step_factor values should be positive.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        Any
    """

    def __init__(
        self,
        max_factor: ScaleFloatType = 1.31,
        step_factor: ScaleFloatType = (0.01, 0.03),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.max_factor = to_tuple(max_factor, low=1.0)
        self.step_factor = to_tuple(step_factor, step_factor)

        if self.max_factor[0] < 1:
            raise ValueError("Max factor must be larger or equal 1")
        if self.step_factor[0] <= 0:
            raise ValueError("Step factor must be positive")

    def apply(self, img: np.ndarray, zoom_factors: np.ndarray = None, **params) -> np.ndarray:
        assert zoom_factors is not None
        return F.zoom_blur(img, zoom_factors)

    def get_params(self) -> Dict[str, Any]:
        max_factor = random.uniform(self.max_factor[0], self.max_factor[1])
        step_factor = random.uniform(self.step_factor[0], self.step_factor[1])
        return {"zoom_factors": np.arange(1.0, max_factor, step_factor)}

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("max_factor", "step_factor")
