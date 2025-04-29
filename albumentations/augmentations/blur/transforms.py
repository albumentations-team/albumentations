"""Transform classes for applying various blur operations to images.

This module contains transform classes that implement different blur effects including
standard blur, motion blur, median blur, Gaussian blur, glass blur, advanced blur, defocus,
and zoom blur. These transforms are designed to work within the albumentations pipeline
and support parameters for controlling the intensity and properties of the blur effects.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

import numpy as np
from pydantic import (
    AfterValidator,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from albumentations.augmentations import functional as fmain
from albumentations.core.pydantic import (
    NonNegativeFloatRangeType,
    OnePlusFloatRangeType,
    OnePlusIntRangeType,
    SymmetricRangeType,
    check_range_bounds,
    convert_to_0plus_range,
    nondecreasing,
    process_non_negative_range,
)
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)
from albumentations.core.utils import to_tuple

from . import functional as fblur

__all__ = [
    "AdvancedBlur",
    "Blur",
    "Defocus",
    "GaussianBlur",
    "GlassBlur",
    "MedianBlur",
    "MotionBlur",
    "ZoomBlur",
]


HALF = 0.5
TWO = 2


class BlurInitSchema(BaseTransformInitSchema):
    blur_limit: tuple[int, int] | int

    @field_validator("blur_limit")
    @classmethod
    def process_blur(cls, value: tuple[int, int] | int, info: ValidationInfo) -> tuple[int, int]:
        return fblur.process_blur_limit(value, info, min_value=3)


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

    def __init__(
        self,
        blur_limit: tuple[int, int] | int = (3, 7),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.blur_limit = cast("tuple[int, int]", blur_limit)

    def apply(self, img: np.ndarray, kernel: int, **params: Any) -> np.ndarray:
        """Apply blur to the input image.

        Args:
            img (np.ndarray): Image to blur.
            kernel (int): Size of the kernel for blur.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Blurred image.

        """
        return fblur.box_blur(img, kernel)

    def get_params(self) -> dict[str, Any]:
        """Get parameters for the transform.

        Returns:
            dict[str, Any]: Dictionary with parameters.

        """
        kernel = fblur.sample_odd_from_range(
            self.py_random,
            self.blur_limit[0],
            self.blur_limit[1],
        )
        return {"kernel": kernel}


class MotionBlur(Blur):
    """Apply motion blur to the input image using a directional kernel.

    This transform simulates motion blur effects that occur during image capture,
    such as camera shake or object movement. It creates a directional blur using
    a line-shaped kernel with controllable angle, direction, and position.

    Args:
        blur_limit (int | tuple[int, int]): Maximum kernel size for blurring.
            Should be in range [3, inf).
            - If int: kernel size will be randomly chosen from [3, blur_limit]
            - If tuple: kernel size will be randomly chosen from [min, max]
            Larger values create stronger blur effects.
            Default: (3, 7)

        angle_range (tuple[float, float]): Range of possible angles in degrees.
            Controls the rotation of the motion blur line:
            - 0°: Horizontal motion blur →
            - 45°: Diagonal motion blur ↗
            - 90°: Vertical motion blur ↑
            - 135°: Diagonal motion blur ↖
            Default: (0, 360)

        direction_range (tuple[float, float]): Range for motion bias.
            Controls how the blur extends from the center:
            - -1.0: Blur extends only backward (←)
            -  0.0: Blur extends equally in both directions (←→)
            -  1.0: Blur extends only forward (→)
            For example, with angle=0:
            - direction=-1.0: ←•
            - direction=0.0:  ←•→
            - direction=1.0:   •→
            Default: (-1.0, 1.0)

        allow_shifted (bool): Allow random kernel position shifts.
            - If True: Kernel can be randomly offset from center
            - If False: Kernel will always be centered
            Default: True

        p (float): Probability of applying the transform. Default: 0.5

    Examples of angle vs direction:
        1. Horizontal motion (angle=0°):
           - direction=0.0:   ←•→   (symmetric blur)
           - direction=1.0:    •→   (forward blur)
           - direction=-1.0:  ←•    (backward blur)

        2. Vertical motion (angle=90°):
           - direction=0.0:   ↑•↓   (symmetric blur)
           - direction=1.0:    •↑   (upward blur)
           - direction=-1.0:  ↓•    (downward blur)

        3. Diagonal motion (angle=45°):
           - direction=0.0:   ↙•↗   (symmetric blur)
           - direction=1.0:    •↗   (forward diagonal blur)
           - direction=-1.0:  ↙•    (backward diagonal blur)

    Note:
        - angle controls the orientation of the motion line
        - direction controls the distribution of the blur along that line
        - Together they can simulate various motion effects:
          * Camera shake: Small angle range + direction near 0
          * Object motion: Specific angle + direction=1.0
          * Complex motion: Random angle + random direction

    Example:
        >>> import albumentations as A
        >>> # Horizontal camera shake (symmetric)
        >>> transform = A.MotionBlur(
        ...     angle_range=(-5, 5),      # Near-horizontal motion
        ...     direction_range=(0, 0),    # Symmetric blur
        ...     p=1.0
        ... )
        >>>
        >>> # Object moving right
        >>> transform = A.MotionBlur(
        ...     angle_range=(0, 0),        # Horizontal motion
        ...     direction_range=(0.8, 1.0), # Strong forward bias
        ...     p=1.0
        ... )

    References:
        - Motion blur fundamentals:
          https://en.wikipedia.org/wiki/Motion_blur

        - Directional blur kernels:
          https://www.sciencedirect.com/topics/computer-science/directional-blur

        - OpenCV filter2D (used for convolution):
          https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04

        - Research on motion blur simulation:
          "Understanding and Evaluating Blind Deconvolution Algorithms" (CVPR 2009)
          https://doi.org/10.1109/CVPR.2009.5206815

        - Motion blur in photography:
          "The Manual of Photography", Chapter 7: Motion in Photography
          ISBN: 978-0240520377

        - Kornia's implementation (similar approach):
          https://kornia.readthedocs.io/en/latest/augmentation.html#kornia.augmentation.RandomMotionBlur

    See Also:
        - GaussianBlur: For uniform blur effects
        - MedianBlur: For noise reduction while preserving edges
        - RandomRain: Another motion-based effect
        - Perspective: For geometric motion-like distortions

    """

    class InitSchema(BlurInitSchema):
        allow_shifted: bool
        angle_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(0, 360)),
        ]
        direction_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_range_bounds(min_val=-1.0, max_val=1.0)),
        ]

    def __init__(
        self,
        blur_limit: tuple[int, int] | int = (3, 7),
        allow_shifted: bool = True,
        angle_range: tuple[float, float] = (0, 360),
        direction_range: tuple[float, float] = (-1.0, 1.0),
        p: float = 0.5,
    ):
        super().__init__(blur_limit=blur_limit, p=p)
        self.allow_shifted = allow_shifted
        self.blur_limit = cast("tuple[int, int]", blur_limit)
        self.angle_range = angle_range
        self.direction_range = direction_range

    def apply(self, img: np.ndarray, kernel: np.ndarray, **params: Any) -> np.ndarray:
        """Apply motion blur to the input image.

        Args:
            img (np.ndarray): Image to blur.
            kernel (np.ndarray): Kernel for motion blur.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Motion blurred image.

        """
        return fmain.convolve(img, kernel=kernel)

    def get_params(self) -> dict[str, Any]:
        """Get parameters for the transform.

        Returns:
            dict[str, Any]: Dictionary with parameters.

        """
        ksize = fblur.sample_odd_from_range(
            self.py_random,
            self.blur_limit[0],
            self.blur_limit[1],
        )

        angle = self.py_random.uniform(*self.angle_range)
        direction = self.py_random.uniform(*self.direction_range)

        # Create motion blur kernel
        kernel = fblur.create_motion_kernel(
            ksize,
            angle,
            direction,
            allow_shifted=self.allow_shifted,
            random_state=self.py_random,
        )

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

    def __init__(
        self,
        blur_limit: tuple[int, int] | int = (3, 7),
        p: float = 0.5,
    ):
        super().__init__(blur_limit=blur_limit, p=p)

    def apply(self, img: np.ndarray, kernel: int, **params: Any) -> np.ndarray:
        """Apply median blur to the input image.

        Args:
            img (np.ndarray): Image to blur.
            kernel (int): Size of the kernel for blur.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Median blurred image.

        """
        return fblur.median_blur(img, kernel)


class GaussianBlur(ImageOnlyTransform):
    """Apply Gaussian blur to the input image using a randomly sized kernel.

    This transform blurs the input image using a Gaussian filter with a random kernel size
    and sigma value. Gaussian blur is a widely used image processing technique that reduces
    image noise and detail, creating a smoothing effect.

    Args:
        sigma_limit (tuple[float, float] | float): Range for the Gaussian kernel standard
            deviation (sigma). Must be more or equal than 0.
            - If a single float is provided, sigma will be randomly chosen
              between 0 and that value.
            - If a tuple of two floats is provided, it defines the inclusive range
              of possible sigma values.
            Default: (0.5, 3.0)

        blur_limit (tuple[int, int] | int): Controls the range of the Gaussian kernel size.
            - If a single int is provided, the kernel size will be randomly chosen
              between 0 and that value.
            - If a tuple of two ints is provided, it defines the inclusive range
              of possible kernel sizes.
            Must be zero or odd and in range [0, inf). If set to 0 (default), the kernel size
            will be computed from sigma as `int(sigma * 3.5) * 2 + 1` to exactly match PIL's
            implementation.
            Default: 0

        p (float): Probability of applying the transform. Default: 0.5

    Targets:
        image

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        - When blur_limit=0 (default), this implementation exactly matches PIL's
          GaussianBlur behavior:
          * Kernel size is computed as int(sigma * 3.5) * 2 + 1
          * Gaussian values are computed using the standard formula
          * Kernel is normalized to preserve image luminance
        - When blur_limit is specified, the kernel size is randomly sampled from that range
          regardless of sigma, which might result in inconsistent blur effects.
        - The default sigma range (0.5, 3.0) provides a good balance between subtle
          and strong blur effects:
          * sigma=0.5 results in a 3x3 kernel
          * sigma=1.0 results in a 7x7 kernel
          * sigma=2.0 results in a 15x15 kernel
          * sigma=3.0 results in a 21x21 kernel

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> # Default behavior: matches PIL's GaussianBlur
        >>> transform = A.GaussianBlur(p=1.0, sigma_limit=(0.5, 3.0))
        >>> # Or manual kernel size range
        >>> transform = A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 3.0), p=1.0)
        >>> result = transform(image=image)
        >>> blurred_image = result["image"]

    """

    class InitSchema(BaseTransformInitSchema):
        sigma_limit: Annotated[
            tuple[float, float] | float,
            AfterValidator(process_non_negative_range),
            AfterValidator(nondecreasing),
        ]
        blur_limit: Annotated[
            tuple[int, int] | int,
            AfterValidator(convert_to_0plus_range),
            AfterValidator(nondecreasing),
        ]

    def __init__(
        self,
        blur_limit: tuple[int, int] | int = 0,
        sigma_limit: tuple[float, float] | float = (0.5, 3.0),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.blur_limit = cast("tuple[int, int]", blur_limit)
        self.sigma_limit = cast("tuple[float, float]", sigma_limit)

    def apply(
        self,
        img: np.ndarray,
        kernel: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply Gaussian blur to the input image.

        Args:
            img (np.ndarray): Image to blur.
            kernel (np.ndarray): Kernel for Gaussian blur.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Gaussian blurred image.

        """
        return fmain.separable_convolve(img, kernel=kernel)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, float]:
        """Get parameters that depend on input data.

        Args:
            params (dict[str, Any]): Parameters.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, float]: Dictionary with parameters.

        """
        sigma = self.py_random.uniform(*self.sigma_limit)
        ksize = self.py_random.randint(*self.blur_limit)
        return {"kernel": fblur.create_gaussian_kernel_1d(sigma, ksize)}


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
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.sigma = sigma
        self.max_delta = max_delta
        self.iterations = iterations
        self.mode = mode

    def apply(
        self,
        img: np.ndarray,
        *args: Any,
        dxy: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply glass blur effect to the input image.

        Args:
            img (np.ndarray): Image to blur.
            *args (Any): Additional positional arguments.
            dxy (np.ndarray): Displacement map.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Image with glass blur effect.

        """
        return fblur.glass_blur(
            img,
            self.sigma,
            self.max_delta,
            self.iterations,
            dxy,
            self.mode,
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Get parameters that depend on input data.

        Args:
            params (dict[str, Any]): Parameters.
            data (dict[str, Any]): Input data.

        Returns:
            dict[str, np.ndarray]: Dictionary with parameters.

        """
        height, width = params["shape"][:2]
        # generate array containing all necessary values for transformations
        width_pixels = height - self.max_delta * 2
        height_pixels = width - self.max_delta * 2
        total_pixels = int(width_pixels * height_pixels)
        dxy = self.random_generator.integers(
            -self.max_delta,
            self.max_delta,
            size=(total_pixels, self.iterations, 2),
        )

        return {"dxy": dxy}


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
        def _check_beta_limit(cls, value: tuple[float, float] | float) -> tuple[float, float]:
            result = to_tuple(value, low=0)
            if not (result[0] < 1.0 < result[1]):
                raise ValueError(
                    f"Beta limit should include 1.0, got {result}",
                )
            return result

        @model_validator(mode="after")
        def _validate_limits(self) -> Self:
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
        blur_limit: tuple[int, int] | int = (3, 7),
        sigma_x_limit: tuple[float, float] | float = (0.2, 1.0),
        sigma_y_limit: tuple[float, float] | float = (0.2, 1.0),
        rotate_limit: tuple[int, int] | int = (-90, 90),
        beta_limit: tuple[float, float] | float = (0.5, 8.0),
        noise_limit: tuple[float, float] | float = (0.9, 1.1),
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.blur_limit = cast("tuple[int, int]", blur_limit)
        self.sigma_x_limit = cast("tuple[float, float]", sigma_x_limit)
        self.sigma_y_limit = cast("tuple[float, float]", sigma_y_limit)
        self.rotate_limit = cast("tuple[int, int]", rotate_limit)
        self.beta_limit = cast("tuple[float, float]", beta_limit)
        self.noise_limit = cast("tuple[float, float]", noise_limit)

    def apply(self, img: np.ndarray, kernel: np.ndarray, **params: Any) -> np.ndarray:
        """Apply advanced blur to the input image.

        Args:
            img (np.ndarray): Image to blur.
            kernel (np.ndarray): Kernel for blur.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Blurred image.

        """
        return fmain.convolve(img, kernel=kernel)

    def get_params(self) -> dict[str, np.ndarray]:
        """Get parameters for the transform.

        Returns:
            dict[str, np.ndarray]: Dictionary with parameters.

        """
        ksize = fblur.sample_odd_from_range(
            self.py_random,
            self.blur_limit[0],
            self.blur_limit[1],
        )
        sigma_x = self.py_random.uniform(*self.sigma_x_limit)
        sigma_y = self.py_random.uniform(*self.sigma_y_limit)
        angle = np.deg2rad(self.py_random.uniform(*self.rotate_limit))

        # Split into 2 cases to avoid selection of narrow kernels (beta > 1) too often.
        beta = (
            self.py_random.uniform(self.beta_limit[0], 1)
            if self.py_random.random() < HALF
            else self.py_random.uniform(1, self.beta_limit[1])
        )

        noise_matrix = self.random_generator.uniform(
            *self.noise_limit,
            size=(ksize, ksize),
        )

        # Generate mesh grid centered at zero.
        ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
        # > Shape (ksize, ksize, 2)
        grid = np.stack(np.meshgrid(ax, ax), axis=-1)

        # Calculate rotated sigma matrix
        d_matrix = np.array([[sigma_x**2, 0], [0, sigma_y**2]])
        u_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        )
        sigma_matrix = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

        inverse_sigma = np.linalg.inv(sigma_matrix)
        # Described in "Parameter Estimation For Multivariate Generalized Gaussian Distributions"
        kernel = np.exp(
            -0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta),
        )
        # Add noise
        kernel *= noise_matrix

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        return {"kernel": kernel}


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
        >>> transform = A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4))
        >>> result = transform(image=image)
        >>> defocused_image = result['image']

    References:
        Defocus aberration: https://en.wikipedia.org/wiki/Defocus_aberration

    """

    class InitSchema(BaseTransformInitSchema):
        radius: OnePlusIntRangeType
        alias_blur: NonNegativeFloatRangeType

    def __init__(
        self,
        radius: tuple[int, int] | int = (3, 10),
        alias_blur: tuple[float, float] | float = (0.1, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.radius = cast("tuple[int, int]", radius)
        self.alias_blur = cast("tuple[float, float]", alias_blur)

    def apply(
        self,
        img: np.ndarray,
        radius: int,
        alias_blur: float,
        **params: Any,
    ) -> np.ndarray:
        """Apply defocus blur to the input image.

        Args:
            img (np.ndarray): Image to blur.
            radius (int): Radius of the defocus blur.
            alias_blur (float): Standard deviation of the Gaussian blur.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Defocused image.

        """
        return fblur.defocus(img, radius, alias_blur)

    def get_params(self) -> dict[str, Any]:
        """Get parameters for the transform.

        Returns:
            dict[str, Any]: Dictionary with parameters.

        """
        return {
            "radius": self.py_random.randint(*self.radius),
            "alias_blur": self.py_random.uniform(*self.alias_blur),
        }


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
        Zoom Blur: https://arxiv.org/abs/1903.12261

    """

    class InitSchema(BaseTransformInitSchema):
        max_factor: OnePlusFloatRangeType
        step_factor: NonNegativeFloatRangeType

    def __init__(
        self,
        max_factor: tuple[float, float] | float = (1, 1.31),
        step_factor: tuple[float, float] | float = (0.01, 0.03),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.max_factor = cast("tuple[float, float]", max_factor)
        self.step_factor = cast("tuple[float, float]", step_factor)

    def apply(
        self,
        img: np.ndarray,
        zoom_factors: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply zoom blur to the input image.

        Args:
            img (np.ndarray): Image to blur.
            zoom_factors (np.ndarray): Array of zoom factors.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Zoom blurred image.

        """
        return fblur.zoom_blur(img, zoom_factors)

    def get_params(self) -> dict[str, Any]:
        """Get parameters for the transform.

        Returns:
            dict[str, Any]: Dictionary with parameters.

        """
        step_factor = self.py_random.uniform(*self.step_factor)
        max_factor = max(1 + step_factor, self.py_random.uniform(*self.max_factor))
        return {"zoom_factors": np.arange(1.0, max_factor, step_factor)}
