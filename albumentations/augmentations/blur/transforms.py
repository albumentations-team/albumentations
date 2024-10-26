from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import cv2
import numpy as np
from pydantic import Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from albumentations.augmentations import functional as fmain
from albumentations.augmentations.utils import check_range
from albumentations.core.pydantic import (
    NonNegativeFloatRangeType,
    OnePlusFloatRangeType,
    OnePlusIntRangeType,
    SymmetricRangeType,
)
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform
from albumentations.core.types import ScaleFloatType, ScaleIntType
from albumentations.core.utils import to_tuple

from . import functional as fblur

__all__ = ["Blur", "MotionBlur", "GaussianBlur", "GlassBlur", "AdvancedBlur", "MedianBlur", "Defocus", "ZoomBlur"]


HALF = 0.5
TWO = 2


def process_blur_limit(value: ScaleIntType, info: ValidationInfo, min_value: float = 0) -> tuple[int, int]:
    bounds = 0, float("inf")
    result = to_tuple(value, min_value)
    check_range(result, *bounds, info.field_name)

    for v in result:
        if v != 0 and v % 2 != 1:
            raise ValueError(f"Blur limit must be 0 or odd. Got: {result}")
    return result


class BlurInitSchema(BaseTransformInitSchema):
    blur_limit: ScaleIntType

    @field_validator("blur_limit")
    @classmethod
    def process_blur(cls, value: ScaleIntType, info: ValidationInfo) -> tuple[int, int]:
        return process_blur_limit(value, info, min_value=3)


class Blur(ImageOnlyTransform):
    """Apply uniform box blur to the input image using a randomly sized square kernel.

    This transform uses OpenCV's cv2.blur function, which performs a simple box filter blur.
    The size of the blur kernel is randomly selected for each application, allowing for
    varying degrees of blur intensity.

    Args:
        blur_limit (tuple[int, int] | int): Controls the range of the blur kernel size.
            - If a single int is provided, the kernel size will be randomly chosen
              between 3 and that value.
            - If a tuple of two ints is provided, it defines the inclusive range
              of possible kernel sizes.
            The kernel size must be odd and greater than or equal to 3.
            Larger kernel sizes produce stronger blur effects.
            Default: (3, 7)

        p (float): Probability of applying the transform. Default: 0.5

    Notes:
        - The blur kernel is always square (same width and height).
        - Only odd kernel sizes are used to ensure the blur has a clear center pixel.
        - Box blur is faster than Gaussian blur but may produce less natural results.
        - This blur method averages all pixels under the kernel area, which can
          reduce noise but also reduce image detail.

    Targets:
        image

    Image types:
        uint8, float32

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Blur(blur_limit=(3, 7), p=1.0)
        >>> result = transform(image=image)
        >>> blurred_image = result["image"]
    """

    class InitSchema(BlurInitSchema):
        pass

    def __init__(self, blur_limit: ScaleIntType = (3, 7), p: float = 0.5, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)
        self.blur_limit = cast(tuple[int, int], blur_limit)

    def apply(self, img: np.ndarray, kernel: int, **params: Any) -> np.ndarray:
        return fblur.blur(img, kernel)

    def get_params(self) -> dict[str, Any]:
        return {"kernel": self.random_generator.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2)))}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("blur_limit",)


class MotionBlur(Blur):
    """Apply motion blur to the input image using a random-sized kernel.

    This transform simulates the effect of camera or object motion during image capture,
    creating a directional blur. It uses a line-shaped kernel with random orientation
    to achieve this effect.

    Args:
        blur_limit (int | tuple[int, int]): Maximum kernel size for blurring the input image.
            Should be in range [3, inf).
            - If a single int is provided, the kernel size will be randomly chosen
              between 3 and that value.
            - If a tuple of two ints is provided, it defines the inclusive range
              of possible kernel sizes.
            Default: (3, 7)

        allow_shifted (bool): If set to True, allows the motion blur kernel to be
            randomly shifted from the center. If False, the kernel will always be
            centered. Default: True

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The blur kernel is always a straight line, simulating linear motion.
        - The angle of the motion blur is randomly chosen for each application.
        - Larger kernel sizes result in more pronounced motion blur effects.
        - When `allow_shifted` is True, the blur effect can appear more natural and varied,
          as it simulates motion that isn't perfectly centered in the frame.
        - This transform is particularly useful for:
          * Simulating camera shake or motion blur in action scenes
          * Data augmentation for object detection or tracking tasks
          * Creating more challenging inputs for image stabilization algorithms

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.MotionBlur(blur_limit=7, allow_shifted=True, p=0.5)
        >>> result = transform(image=image)
        >>> motion_blurred_image = result["image"]

    References:
        - Motion blur: https://en.wikipedia.org/wiki/Motion_blur
        - OpenCV filter2D (used internally):
          https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    """

    class InitSchema(BaseTransformInitSchema):
        allow_shifted: bool
        blur_limit: ScaleIntType

        @model_validator(mode="after")
        def process_blur(self) -> Self:
            self.blur_limit = to_tuple(self.blur_limit, 3)

            if self.allow_shifted and isinstance(self.blur_limit, tuple) and any(x % 2 != 1 for x in self.blur_limit):
                raise ValueError(f"Blur limit must be odd when centered=True. Got: {self.blur_limit}")

            return self

    def __init__(
        self,
        blur_limit: ScaleIntType = 7,
        allow_shifted: bool = True,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(blur_limit=blur_limit, p=p, always_apply=always_apply)
        self.allow_shifted = allow_shifted
        self.blur_limit = cast(tuple[int, int], blur_limit)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (*super().get_transform_init_args_names(), "allow_shifted")

    def apply(self, img: np.ndarray, kernel: np.ndarray, **params: Any) -> np.ndarray:
        return fmain.convolve(img, kernel=kernel)

    def get_params(self) -> dict[str, Any]:
        ksize = self.py_random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2)))
        if ksize <= TWO:
            raise ValueError(f"ksize must be > 2. Got: {ksize}")
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        x1, x2 = self.py_random.randint(0, ksize - 1), self.py_random.randint(0, ksize - 1)
        if x1 == x2:
            y1, y2 = self.py_random.sample(range(ksize), 2)
        else:
            y1, y2 = self.py_random.randint(0, ksize - 1), self.py_random.randint(0, ksize - 1)

        def make_odd_val(v1: int, v2: int) -> tuple[int, int]:
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
            x1, x2 = (int(i - dx) for i in [x1, x2])
            y1, y2 = (int(i - dy) for i in [y1, y2])

        cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)

        # Normalize kernel
        return {"kernel": kernel.astype(np.float32) / np.sum(kernel)}


class MedianBlur(Blur):
    """Apply median blur to the input image.

    This transform uses a median filter to blur the input image. Median filtering is particularly
    effective at removing salt-and-pepper noise while preserving edges, making it a popular choice
    for noise reduction in image processing.

    Args:
        blur_limit (int | tuple[int, int]): Maximum aperture linear size for blurring the input image.
            Must be odd and in the range [3, inf).
            - If a single int is provided, the kernel size will be randomly chosen
              between 3 and that value.
            - If a tuple of two ints is provided, it defines the inclusive range
              of possible kernel sizes.
            Default: (3, 7)

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The kernel size (aperture linear size) must always be odd and greater than 1.
        - Unlike mean blur or Gaussian blur, median blur uses the median of all pixels under
          the kernel area, making it more robust to outliers.
        - This transform is particularly useful for:
          * Removing salt-and-pepper noise
          * Preserving edges while smoothing images
          * Pre-processing images for edge detection algorithms
        - For color images, the median is calculated independently for each channel.
        - Larger kernel sizes result in stronger blurring effects but may also remove
          fine details from the image.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.MedianBlur(blur_limit=(3, 7), p=0.5)
        >>> result = transform(image=image)
        >>> blurred_image = result["image"]

    References:
        - Median filter: https://en.wikipedia.org/wiki/Median_filter
        - OpenCV medianBlur: https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9
    """

    def __init__(self, blur_limit: ScaleIntType = 7, p: float = 0.5, always_apply: bool | None = None):
        super().__init__(blur_limit=blur_limit, p=p, always_apply=always_apply)

    def apply(self, img: np.ndarray, kernel: int, **params: Any) -> np.ndarray:
        return fblur.median_blur(img, kernel)


class GaussianBlur(ImageOnlyTransform):
    """Apply Gaussian blur to the input image using a randomly sized kernel.

    This transform blurs the input image using a Gaussian filter with a random kernel size
    and sigma value. Gaussian blur is a widely used image processing technique that reduces
    image noise and detail, creating a smoothing effect.

    Args:
        blur_limit (tuple[int, int] | int): Controls the range of the Gaussian kernel size.
            - If a single int is provided, the kernel size will be randomly chosen
              between 0 and that value.
            - If a tuple of two ints is provided, it defines the inclusive range
              of possible kernel sizes.
            Must be zero or odd and in range [0, inf). If set to 0, it will be computed
            from sigma as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            Larger kernel sizes produce stronger blur effects.
            Default: (3, 7)

        sigma_limit (tuple[float, float] | float): Range for the Gaussian kernel standard
            deviation (sigma). Must be in range [0, inf).
            - If a single float is provided, sigma will be randomly chosen
              between 0 and that value.
            - If a tuple of two floats is provided, it defines the inclusive range
              of possible sigma values.
            If set to 0, sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`.
            Larger sigma values produce stronger blur effects.
            Default: 0

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - The relationship between kernel size and sigma affects the blur strength:
          larger kernel sizes allow for stronger blurring effects.
        - When both blur_limit and sigma_limit are set to ranges starting from 0,
          the blur_limit minimum is automatically set to 3 to ensure a valid kernel size.
        - For uint8 images, the computation might be faster than for floating-point images.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2), p=1)
        >>> result = transform(image=image)
        >>> blurred_image = result["image"]
    """

    class InitSchema(BlurInitSchema):
        sigma_limit: NonNegativeFloatRangeType

        @field_validator("blur_limit")
        @classmethod
        def process_blur(cls, value: ScaleIntType, info: ValidationInfo) -> tuple[int, int]:
            return process_blur_limit(value, info, min_value=0)

        @model_validator(mode="after")
        def validate_limits(self) -> Self:
            if (
                isinstance(self.blur_limit, (tuple, list))
                and self.blur_limit[0] == 0
                and isinstance(self.sigma_limit, (tuple, list))
                and self.sigma_limit[0] == 0
            ):
                self.blur_limit = 3, max(3, self.blur_limit[1])
                warnings.warn(
                    "blur_limit and sigma_limit minimum value can not be both equal to 0. "
                    "blur_limit minimum value changed to 3.",
                    stacklevel=2,
                )

            if isinstance(self.blur_limit, tuple):
                for v in self.blur_limit:
                    if v != 0 and v % 2 != 1:
                        raise ValueError(f"Blur limit must be 0 or odd. Got: {self.blur_limit}")

            return self

    def __init__(
        self,
        blur_limit: ScaleIntType = (3, 7),
        sigma_limit: ScaleFloatType = 0,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p, always_apply)
        self.blur_limit = cast(tuple[int, int], blur_limit)
        self.sigma_limit = cast(tuple[float, float], sigma_limit)

    def apply(self, img: np.ndarray, ksize: int, sigma: float, **params: Any) -> np.ndarray:
        return fblur.gaussian_blur(img, ksize, sigma=sigma)

    def get_params(self) -> dict[str, float]:
        ksize = self.py_random.randrange(self.blur_limit[0], self.blur_limit[1] + 1)
        if ksize != 0 and ksize % 2 != 1:
            ksize = (ksize + 1) % (self.blur_limit[1] + 1)

        return {"ksize": ksize, "sigma": self.py_random.uniform(*self.sigma_limit)}

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return "blur_limit", "sigma_limit"


class GlassBlur(ImageOnlyTransform):
    """Apply a glass blur effect to the input image.

    This transform simulates the effect of looking through textured glass by locally
    shuffling pixels in the image. It creates a distorted, frosted glass-like appearance.

    Args:
        sigma (float): Standard deviation for the Gaussian kernel used in the process.
            Higher values increase the blur effect. Must be non-negative.
            Default: 0.7

        max_delta (int): Maximum distance in pixels for shuffling.
            Determines how far pixels can be moved. Larger values create more distortion.
            Must be a positive integer.
            Default: 4

        iterations (int): Number of times to apply the glass blur effect.
            More iterations create a stronger effect but increase computation time.
            Must be a positive integer.
            Default: 2

        mode (Literal["fast", "exact"]): Mode of computation. Options are:
            - "fast": Uses a faster but potentially less accurate method.
            - "exact": Uses a slower but more precise method.
            Default: "fast"

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - This transform is particularly effective for creating a 'looking through
          glass' effect or simulating the view through a frosted window.
        - The 'fast' mode is recommended for most use cases as it provides a good
          balance between effect quality and computation speed.
        - Increasing 'iterations' will strengthen the effect but also increase the
          processing time linearly.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.GlassBlur(sigma=0.7, max_delta=4, iterations=3, mode="fast", p=1)
        >>> result = transform(image=image)
        >>> glass_blurred_image = result["image"]

    References:
        - This implementation is based on the technique described in:
          "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness"
          https://arxiv.org/abs/1903.12261
        - Original implementation:
          https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
    """

    class InitSchema(BaseTransformInitSchema):
        sigma: float = Field(ge=0)
        max_delta: int = Field(ge=1)
        iterations: int = Field(ge=1)
        mode: Literal["fast", "exact"]

    def __init__(
        self,
        sigma: float = 0.7,
        max_delta: int = 4,
        iterations: int = 2,
        mode: Literal["fast", "exact"] = "fast",
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.sigma = sigma
        self.max_delta = max_delta
        self.iterations = iterations
        self.mode = mode

    def apply(self, img: np.ndarray, *args: Any, dxy: np.ndarray, **params: Any) -> np.ndarray:
        return fblur.glass_blur(img, self.sigma, self.max_delta, self.iterations, dxy, self.mode)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
        height, width = params["shape"][:2]

        # generate array containing all necessary values for transformations
        width_pixels = height - self.max_delta * 2
        height_pixels = width - self.max_delta * 2
        total_pixels = int(width_pixels * height_pixels)
        dxy = self.random_generator.integers(-self.max_delta, self.max_delta, size=(total_pixels, self.iterations, 2))

        return {"dxy": dxy}

    def get_transform_init_args_names(self) -> tuple[str, str, str, str]:
        return "sigma", "max_delta", "iterations", "mode"


class AdvancedBlur(ImageOnlyTransform):
    """Applies a Generalized Gaussian blur to the input image with randomized parameters for advanced data augmentation.

    This transform creates a custom blur kernel based on the Generalized Gaussian distribution,
    which allows for a wide range of blur effects beyond standard Gaussian blur. It then applies
    this kernel to the input image through convolution. The transform also incorporates noise
    into the kernel, resulting in a unique combination of blurring and noise injection.

    Key features of this augmentation:

    1. Generalized Gaussian Kernel: Uses a generalized normal distribution to create kernels
       that can range from box-like blurs to very peaked blurs, controlled by the beta parameter.

    2. Anisotropic Blurring: Allows for different blur strengths in horizontal and vertical
       directions (controlled by sigma_x and sigma_y), and rotation of the kernel.

    3. Kernel Noise: Adds multiplicative noise to the kernel before applying it to the image,
       creating more diverse and realistic blur effects.

    Implementation Details:
        The kernel is generated using a 2D Generalized Gaussian function. The process involves:
        1. Creating a 2D grid based on the kernel size
        2. Applying rotation to this grid
        3. Calculating the kernel values using the Generalized Gaussian formula
        4. Adding multiplicative noise to the kernel
        5. Normalizing the kernel

        The resulting kernel is then applied to the image using convolution.

    Args:
        blur_limit (tuple[int, int] | int, optional): Controls the size of the blur kernel. If a single int
            is provided, the kernel size will be randomly chosen between 3 and that value.
            Must be odd and ≥ 3. Larger values create stronger blur effects.
            Default: (3, 7)

        sigma_x_limit (tuple[float, float] | float): Controls the spread of the blur in the x direction.
            Higher values increase blur strength.
            If a single float is provided, the range will be (0, limit).
            Default: (0.2, 1.0)

        sigma_y_limit (tuple[float, float] | float): Controls the spread of the blur in the y direction.
            Higher values increase blur strength.
            If a single float is provided, the range will be (0, limit).
            Default: (0.2, 1.0)

        rotate_limit (tuple[int, int] | int): Range of angles (in degrees) for rotating the kernel.
            This rotation allows for diagonal blur directions. If limit is a single int, an angle is picked
            from (-rotate_limit, rotate_limit).
            Default: (-90, 90)

        beta_limit (tuple[float, float] | float): Shape parameter of the Generalized Gaussian distribution.
            - beta = 1 gives a standard Gaussian distribution
            - beta < 1 creates heavier tails, resulting in more uniform, box-like blur
            - beta > 1 creates lighter tails, resulting in more peaked, focused blur
            Default: (0.5, 8.0)

        noise_limit (tuple[float, float] | float): Controls the strength of multiplicative noise
            applied to the kernel. Values around 1.0 keep the original kernel mostly intact,
            while values further from 1.0 introduce more variation.
            Default: (0.75, 1.25)

        p (float): Probability of applying the transform. Default: 0.5

    Notes:
        - This transform is particularly useful for simulating complex, real-world blur effects
          that go beyond simple Gaussian blur.
        - The combination of blur and noise can help in creating more robust models by simulating
          a wider range of image degradations.
        - Extreme values, especially for beta and noise, may result in unrealistic effects and
          should be used cautiously.

    Reference:
        This transform is inspired by techniques described in:
        "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
        https://arxiv.org/abs/2107.10833

    Targets:
        image

    Image types:
        uint8, float32
    """

    class InitSchema(BlurInitSchema):
        sigma_x_limit: NonNegativeFloatRangeType
        sigma_y_limit: NonNegativeFloatRangeType
        beta_limit: NonNegativeFloatRangeType
        noise_limit: NonNegativeFloatRangeType
        rotate_limit: SymmetricRangeType

        @field_validator("beta_limit")
        @classmethod
        def check_beta_limit(cls, value: ScaleFloatType) -> tuple[float, float]:
            result = to_tuple(value, low=0)
            if not (result[0] < 1.0 < result[1]):
                msg = "beta_limit is expected to include 1.0."
                raise ValueError(msg)
            return result

        @model_validator(mode="after")
        def validate_limits(self) -> Self:
            if (
                isinstance(self.sigma_x_limit, (tuple, list))
                and self.sigma_x_limit[0] == 0
                and isinstance(self.sigma_y_limit, (tuple, list))
                and self.sigma_y_limit[0] == 0
            ):
                msg = "sigma_x_limit and sigma_y_limit minimum value cannot be both equal to 0."
                raise ValueError(msg)
            return self

    def __init__(
        self,
        blur_limit: ScaleIntType = (3, 7),
        sigma_x_limit: ScaleFloatType = (0.2, 1.0),
        sigma_y_limit: ScaleFloatType = (0.2, 1.0),
        sigmaX_limit: ScaleFloatType | None = None,  # noqa: N803
        sigmaY_limit: ScaleFloatType | None = None,  # noqa: N803
        rotate_limit: ScaleIntType = (-90, 90),
        beta_limit: ScaleFloatType = (0.5, 8.0),
        noise_limit: ScaleFloatType = (0.9, 1.1),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        if sigmaX_limit is not None:
            warnings.warn("sigmaX_limit is deprecated; use sigma_x_limit instead.", DeprecationWarning, stacklevel=2)
            sigma_x_limit = sigmaX_limit

        if sigmaY_limit is not None:
            warnings.warn("sigmaY_limit is deprecated; use sigma_y_limit instead.", DeprecationWarning, stacklevel=2)
            sigma_y_limit = sigmaY_limit

        self.blur_limit = cast(tuple[int, int], blur_limit)
        self.sigma_x_limit = cast(tuple[float, float], sigma_x_limit)
        self.sigma_y_limit = cast(tuple[float, float], sigma_y_limit)
        self.rotate_limit = cast(tuple[int, int], rotate_limit)
        self.beta_limit = cast(tuple[float, float], beta_limit)
        self.noise_limit = cast(tuple[float, float], noise_limit)

    def apply(self, img: np.ndarray, kernel: np.ndarray, **params: Any) -> np.ndarray:
        return fmain.convolve(img, kernel=kernel)

    def get_params(self) -> dict[str, np.ndarray]:
        ksize = self.py_random.randrange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
        sigma_x = self.py_random.uniform(*self.sigma_x_limit)
        sigma_y = self.py_random.uniform(*self.sigma_y_limit)
        angle = np.deg2rad(self.py_random.uniform(*self.rotate_limit))

        # Split into 2 cases to avoid selection of narrow kernels (beta > 1) too often.
        beta = (
            self.py_random.uniform(self.beta_limit[0], 1)
            if self.py_random.random() < HALF
            else self.py_random.uniform(1, self.beta_limit[1])
        )

        noise_matrix = self.random_generator.uniform(*self.noise_limit, size=(ksize, ksize))

        # Generate mesh grid centered at zero.
        ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
        # > Shape (ksize, ksize, 2)
        grid = np.stack(np.meshgrid(ax, ax), axis=-1)

        # Calculate rotated sigma matrix
        d_matrix = np.array([[sigma_x**2, 0], [0, sigma_y**2]])
        u_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        sigma_matrix = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

        inverse_sigma = np.linalg.inv(sigma_matrix)
        # Described in "Parameter Estimation For Multivariate Generalized Gaussian Distributions"
        kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
        # Add noise
        kernel *= noise_matrix

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        return {"kernel": kernel}

    def get_transform_init_args_names(self) -> tuple[str, str, str, str, str, str]:
        return (
            "blur_limit",
            "sigma_x_limit",
            "sigma_y_limit",
            "rotate_limit",
            "beta_limit",
            "noise_limit",
        )


class Defocus(ImageOnlyTransform):
    """Apply defocus blur to the input image.

    This transform simulates the effect of an out-of-focus camera by applying a defocus blur
    to the image. It uses a combination of disc kernels and Gaussian blur to create a realistic
    defocus effect.

    Args:
        radius (tuple[int, int] | int): Range for the radius of the defocus blur.
            If a single int is provided, the range will be [1, radius].
            Larger values create a stronger blur effect.
            Default: (3, 10)

        alias_blur (tuple[float, float] | float): Range for the standard deviation of the Gaussian blur
            applied after the main defocus blur. This helps to reduce aliasing artifacts.
            If a single float is provided, the range will be (0, alias_blur).
            Larger values create a smoother, more aliased effect.
            Default: (0.1, 0.5)

        p (float): Probability of applying the transform. Should be in the range [0, 1].
            Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - The defocus effect is created using a disc kernel, which simulates the shape of a camera's aperture.
        - The additional Gaussian blur (alias_blur) helps to soften the edges of the disc kernel, creating a
          more natural-looking defocus effect.
        - Larger radius values will create a stronger, more noticeable defocus effect.
        - The alias_blur parameter can be used to fine-tune the appearance of the defocus, with larger values
          creating a smoother, potentially more realistic effect.

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4), always_apply=True)
        >>> result = transform(image=image)
        >>> defocused_image = result['image']

    References:
        - https://en.wikipedia.org/wiki/Defocus_aberration
        - https://www.researchgate.net/publication/261311609_Realistic_Defocus_Blur_for_Multiplane_Computer-Generated_Holography
    """

    class InitSchema(BaseTransformInitSchema):
        radius: OnePlusIntRangeType
        alias_blur: NonNegativeFloatRangeType

    def __init__(
        self,
        radius: ScaleIntType = (3, 10),
        alias_blur: ScaleFloatType = (0.1, 0.5),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.radius = cast(tuple[int, int], radius)
        self.alias_blur = cast(tuple[float, float], alias_blur)

    def apply(self, img: np.ndarray, radius: int, alias_blur: float, **params: Any) -> np.ndarray:
        return fblur.defocus(img, radius, alias_blur)

    def get_params(self) -> dict[str, Any]:
        return {
            "radius": self.py_random.randint(*self.radius),
            "alias_blur": self.py_random.uniform(*self.alias_blur),
        }

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return ("radius", "alias_blur")


class ZoomBlur(ImageOnlyTransform):
    """Apply zoom blur transform.

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
        unit8, float32

    Reference:
        https://arxiv.org/abs/1903.12261
    """

    class InitSchema(BaseTransformInitSchema):
        max_factor: OnePlusFloatRangeType
        step_factor: NonNegativeFloatRangeType

    def __init__(
        self,
        max_factor: ScaleFloatType = (1, 1.31),
        step_factor: ScaleFloatType = (0.01, 0.03),
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.max_factor = cast(tuple[float, float], max_factor)
        self.step_factor = cast(tuple[float, float], step_factor)

    def apply(self, img: np.ndarray, zoom_factors: np.ndarray, **params: Any) -> np.ndarray:
        return fblur.zoom_blur(img, zoom_factors)

    def get_params(self) -> dict[str, Any]:
        step_factor = self.py_random.uniform(*self.step_factor)
        max_factor = max(1 + step_factor, self.py_random.uniform(*self.max_factor))
        return {"zoom_factors": np.arange(1.0, max_factor, step_factor)}

    def get_transform_init_args_names(self) -> tuple[str, str]:
        return ("max_factor", "step_factor")
