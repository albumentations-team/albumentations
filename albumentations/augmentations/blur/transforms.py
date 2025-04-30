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

from albumentations.augmentations.pixel import functional as fpixel
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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image for demonstration
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize blur effects
        >>> cv2.rectangle(image, (50, 50), (250, 250), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (150, 150), 60, (0, 255, 0), -1)  # Green circle
        >>> cv2.line(image, (50, 150), (250, 150), (0, 0, 255), 5)  # Blue line
        >>>
        >>> # Example 1: Basic usage with default parameters
        >>> transform = A.Compose([
        ...     A.Blur(p=1.0)  # Always apply with default blur_limit=(3, 7)
        ... ])
        >>>
        >>> result = transform(image=image)
        >>> blurred_image = result["image"]
        >>> # The image will have a random blur with kernel size between 3 and 7
        >>>
        >>> # Example 2: Using a fixed blur kernel size
        >>> fixed_transform = A.Compose([
        ...     A.Blur(blur_limit=5, p=1.0)  # Always use kernel size 5x5
        ... ])
        >>>
        >>> fixed_result = fixed_transform(image=image)
        >>> fixed_blurred_image = fixed_result["image"]
        >>> # The image will have a consistent 5x5 kernel blur
        >>>
        >>> # Example 3: Using a custom range for blur kernel sizes
        >>> strong_transform = A.Compose([
        ...     A.Blur(blur_limit=(7, 13), p=1.0)  # Use larger kernel for stronger blur
        ... ])
        >>>
        >>> strong_result = strong_transform(image=image)
        >>> strong_blurred = strong_result["image"]
        >>> # The image will have a stronger blur with kernel size between 7 and 13
        >>>
        >>> # Example 4: As part of a pipeline with other transforms
        >>> pipeline = A.Compose([
        ...     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        ...     A.Blur(blur_limit=(3, 5), p=0.5),  # 50% chance of applying blur
        ...     A.HorizontalFlip(p=0.5)
        ... ])
        >>>
        >>> pipeline_result = pipeline(image=image)
        >>> transformed_image = pipeline_result["image"]
        >>> # The image may or may not be blurred depending on the random probability

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image for demonstration
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize motion blur effects
        >>> cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (150, 150), 30, (0, 255, 0), -1)  # Green circle
        >>> cv2.putText(image, "Motion Blur", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        >>>
        >>> # Example 1: Horizontal camera shake (symmetric)
        >>> horizontal_shake = A.Compose([
        ...     A.MotionBlur(
        ...         blur_limit=(10, 12),     # Strong blur
        ...         angle_range=(-5, 5),     # Near-horizontal motion (±5°)
        ...         direction_range=(0, 0),  # Symmetric blur (equally in both directions)
        ...         p=1.0                    # Always apply
        ...     )
        ... ])
        >>>
        >>> horizontal_result = horizontal_shake(image=image)
        >>> horizontal_blur = horizontal_result["image"]
        >>> # The image will have a horizontal camera shake effect, blurring equally in both directions
        >>>
        >>> # Example 2: Object moving right (directional motion)
        >>> rightward_motion = A.Compose([
        ...     A.MotionBlur(
        ...         blur_limit=(7, 9),         # Medium blur
        ...         angle_range=(0, 0),        # Exactly horizontal motion (0°)
        ...         direction_range=(0.8, 1.0), # Strong forward bias (mostly rightward)
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> rightward_result = rightward_motion(image=image)
        >>> rightward_blur = rightward_result["image"]
        >>> # The image will simulate an object moving rightward, with blur mostly to the right
        >>>
        >>> # Example 3: Object moving diagonally down-right
        >>> diagonal_motion = A.Compose([
        ...     A.MotionBlur(
        ...         blur_limit=(9, 11),       # Stronger blur
        ...         angle_range=(135, 135),   # 135° motion (down-right diagonal)
        ...         direction_range=(0.7, 0.9), # Forward bias
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> diagonal_result = diagonal_motion(image=image)
        >>> diagonal_blur = diagonal_result["image"]
        >>> # The image will simulate diagonal motion down and to the right
        >>>
        >>> # Example 4: Vertical motion (up-down)
        >>> vertical_motion = A.Compose([
        ...     A.MotionBlur(
        ...         blur_limit=9,             # Fixed kernel size
        ...         angle_range=(90, 90),     # Vertical motion (90°)
        ...         direction_range=(-0.2, 0.2), # Near-symmetric (slight bias)
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> vertical_result = vertical_motion(image=image)
        >>> vertical_blur = vertical_result["image"]
        >>> # The image will simulate vertical motion blur
        >>>
        >>> # Example 5: Random motion blur (can be in any direction)
        >>> random_motion = A.Compose([
        ...     A.MotionBlur(
        ...         blur_limit=(5, 12),       # Variable strength
        ...         angle_range=(0, 360),     # Any angle
        ...         direction_range=(-1.0, 1.0), # Any direction bias
        ...         allow_shifted=True,       # Allow kernel to be shifted from center
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> random_result = random_motion(image=image)
        >>> random_blur = random_result["image"]
        >>> # The image will have a random motion blur in any direction
        >>>
        >>> # Example 6: Multiple random parameters with kernel centered (not shifted)
        >>> centered_motion = A.Compose([
        ...     A.MotionBlur(
        ...         blur_limit=(5, 9),
        ...         angle_range=(0, 360),
        ...         direction_range=(-1.0, 1.0),
        ...         allow_shifted=False,      # Kernel will always be centered
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> centered_result = centered_motion(image=image)
        >>> centered_blur = centered_result["image"]
        >>> # The image will have motion blur with the kernel centered (not shifted)
        >>>
        >>> # Example 7: In a composition with other transforms
        >>> pipeline = A.Compose([
        ...     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ...     A.MotionBlur(                                   # 30% chance of applying motion blur
        ...         blur_limit=(3, 7),
        ...         angle_range=(0, 180),                       # Only horizontal to vertical
        ...         direction_range=(-0.5, 0.5),                # Moderate direction bias
        ...         p=0.3
        ...     ),
        ...     A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3)
        ... ])
        >>>
        >>> pipeline_result = pipeline(image=image)
        >>> transformed_image = pipeline_result["image"]
        >>> # The image may have motion blur applied with 30% probability along with other effects

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
        return fpixel.convolve(img, kernel=kernel)

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image for demonstration
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize blur effects
        >>> cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (150, 150), 30, (0, 255, 0), -1)  # Green circle
        >>> cv2.putText(image, "Sample Text", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        >>>
        >>> # Add salt and pepper noise to demonstrate median blur's noise removal capability
        >>> noise = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> noise_points = np.random.random((300, 300)) > 0.95  # 5% of pixels as noise
        >>> image[noise_points] = 255  # White noise (salt)
        >>> noise_points = np.random.random((300, 300)) > 0.95  # Another 5% of pixels
        >>> image[noise_points] = 0    # Black noise (pepper)
        >>>
        >>> # Example 1: Minimal median blur (3x3 kernel)
        >>> minimal_blur = A.Compose([
        ...     A.MedianBlur(
        ...         blur_limit=3,  # Fixed 3x3 kernel
        ...         p=1.0          # Always apply
        ...     )
        ... ])
        >>>
        >>> minimal_result = minimal_blur(image=image)
        >>> minimal_blurred = minimal_result["image"]
        >>> # The image will have minimal median blur, removing most salt and pepper noise
        >>> # while preserving edges and details
        >>>
        >>> # Example 2: Medium median blur
        >>> medium_blur = A.Compose([
        ...     A.MedianBlur(
        ...         blur_limit=5,  # Fixed 5x5 kernel
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> medium_result = medium_blur(image=image)
        >>> medium_blurred = medium_result["image"]
        >>> # The image will have a medium median blur, removing noise and small details
        >>> # while still preserving major edges
        >>>
        >>> # Example 3: Strong median blur
        >>> strong_blur = A.Compose([
        ...     A.MedianBlur(
        ...         blur_limit=9,  # Fixed 9x9 kernel
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> strong_result = strong_blur(image=image)
        >>> strong_blurred = strong_result["image"]
        >>> # The image will have a strong median blur, potentially removing smaller
        >>> # features while still preserving major edges better than other blur types
        >>>
        >>> # Example 4: Random kernel size range
        >>> random_kernel = A.Compose([
        ...     A.MedianBlur(
        ...         blur_limit=(3, 9),  # Kernel size between 3x3 and 9x9
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> random_result = random_kernel(image=image)
        >>> random_blurred = random_result["image"]
        >>> # The image will have a random median blur strength
        >>>
        >>> # Example 5: In a pipeline for noise reduction
        >>> pipeline = A.Compose([
        ...     A.GaussNoise(var_limit=(10, 50), p=0.5),        # Possibly add some noise
        ...     A.MedianBlur(blur_limit=(3, 5), p=0.7),         # 70% chance of applying median blur
        ...     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3)
        ... ])
        >>>
        >>> pipeline_result = pipeline(image=image)
        >>> processed_image = pipeline_result["image"]
        >>> # The image may have been denoised with the median blur (70% probability)

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
          * sigma=0.5 results in a subtle blur
          * sigma=3.0 results in a stronger blur

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image for demonstration
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize blur effects
        >>> cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (150, 150), 30, (0, 255, 0), -1)  # Green circle
        >>> cv2.putText(image, "Sample Text", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        >>>
        >>> # Example 1: Default Gaussian blur (automatic kernel size)
        >>> default_blur = A.Compose([
        ...     A.GaussianBlur(p=1.0)  # Using default parameters
        ... ])
        >>>
        >>> default_result = default_blur(image=image)
        >>> default_blurred = default_result["image"]
        >>> # The image will have a medium Gaussian blur with sigma between 0.5 and 3.0
        >>>
        >>> # Example 2: Light Gaussian blur
        >>> light_blur = A.Compose([
        ...     A.GaussianBlur(
        ...         sigma_limit=(0.2, 0.5),  # Small sigma for subtle blur
        ...         blur_limit=0,            # Auto-compute kernel size
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> light_result = light_blur(image=image)
        >>> light_blurred = light_result["image"]
        >>> # The image will have a subtle Gaussian blur effect
        >>>
        >>> # Example 3: Strong Gaussian blur
        >>> strong_blur = A.Compose([
        ...     A.GaussianBlur(
        ...         sigma_limit=(3.0, 7.0),  # Larger sigma for stronger blur
        ...         blur_limit=0,            # Auto-compute kernel size
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> strong_result = strong_blur(image=image)
        >>> strong_blurred = strong_result["image"]
        >>> # The image will have a strong Gaussian blur effect
        >>>
        >>> # Example 4: Fixed kernel size
        >>> fixed_kernel = A.Compose([
        ...     A.GaussianBlur(
        ...         sigma_limit=(0.5, 2.0),
        ...         blur_limit=(9, 9),       # Fixed 9x9 kernel size
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> fixed_result = fixed_kernel(image=image)
        >>> fixed_kernel_blur = fixed_result["image"]
        >>> # The image will have Gaussian blur with a fixed 9x9 kernel
        >>>
        >>> # Example 5: Random kernel size range
        >>> random_kernel = A.Compose([
        ...     A.GaussianBlur(
        ...         sigma_limit=(1.0, 2.0),
        ...         blur_limit=(5, 9),       # Kernel size between 5x5 and 9x9
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> random_result = random_kernel(image=image)
        >>> random_kernel_blur = random_result["image"]
        >>> # The image will have Gaussian blur with a kernel size between 5x5 and 9x9
        >>>
        >>> # Example 6: In an augmentation pipeline
        >>> pipeline = A.Compose([
        ...     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ...     A.GaussianBlur(sigma_limit=(0.5, 1.5), p=0.3),  # 30% chance of applying
        ...     A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3)
        ... ])
        >>>
        >>> pipeline_result = pipeline(image=image)
        >>> transformed_image = pipeline_result["image"]
        >>> # The image may have Gaussian blur applied with 30% probability along with other effects

    References:
        - OpenCV Gaussian Blur: https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
        - PIL GaussianBlur: https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur

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
        return fpixel.separable_convolve(img, kernel=kernel)

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image for demonstration
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize glass blur effects
        >>> cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (150, 150), 30, (0, 255, 0), -1)  # Green circle
        >>> cv2.putText(image, "Text Sample", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        >>>
        >>> # Example 1: Subtle glass effect (light frosting)
        >>> subtle_transform = A.Compose([
        ...     A.GlassBlur(
        ...         sigma=0.4,           # Lower sigma for gentler blur
        ...         max_delta=2,         # Small displacement
        ...         iterations=1,        # Single iteration
        ...         mode="fast",
        ...         p=1.0                # Always apply
        ...     )
        ... ])
        >>>
        >>> subtle_result = subtle_transform(image=image)
        >>> subtle_glass = subtle_result["image"]
        >>> # The image will have a subtle glass-like distortion, like light frosting
        >>>
        >>> # Example 2: Medium glass effect (typical frosted glass)
        >>> medium_transform = A.Compose([
        ...     A.GlassBlur(
        ...         sigma=0.7,           # Default sigma
        ...         max_delta=4,         # Default displacement
        ...         iterations=2,        # Default iterations
        ...         mode="fast",
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> medium_result = medium_transform(image=image)
        >>> medium_glass = medium_result["image"]
        >>> # The image will have a moderate glass-like effect, similar to standard frosted glass
        >>>
        >>> # Example 3: Strong glass effect (heavy distortion)
        >>> strong_transform = A.Compose([
        ...     A.GlassBlur(
        ...         sigma=1.0,           # Higher sigma for stronger blur
        ...         max_delta=6,         # Larger displacement
        ...         iterations=3,        # More iterations
        ...         mode="fast",
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> strong_result = strong_transform(image=image)
        >>> strong_glass = strong_result["image"]
        >>> # The image will have a strong glass-like distortion, heavily obscuring details
        >>>
        >>> # Example 4: Using exact mode for higher quality
        >>> exact_transform = A.Compose([
        ...     A.GlassBlur(
        ...         sigma=0.7,
        ...         max_delta=4,
        ...         iterations=2,
        ...         mode="exact",        # More precise but slower
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> exact_result = exact_transform(image=image)
        >>> exact_glass = exact_result["image"]
        >>> # The image will have a similar effect to medium, but with potentially better quality
        >>>
        >>> # Example 5: In a pipeline with other transforms
        >>> pipeline = A.Compose([
        ...     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
        ...     A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=0.5),  # 50% chance of applying
        ...     A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3)
        ... ])
        >>>
        >>> pipeline_result = pipeline(image=image)
        >>> transformed_image = pipeline_result["image"]
        >>> # The image may have glass blur applied with 50% probability along with other effects

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image for demonstration
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize blur effects
        >>> cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (150, 150), 30, (0, 255, 0), -1)  # Green circle
        >>> cv2.putText(image, "Text Example", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        >>> cv2.line(image, (50, 250), (250, 250), (0, 0, 255), 3)  # Blue line
        >>>
        >>> # Example 1: Gaussian-like blur (beta = 1)
        >>> gaussian_like = A.Compose([
        ...     A.AdvancedBlur(
        ...         blur_limit=5,
        ...         sigma_x_limit=(0.5, 0.5),
        ...         sigma_y_limit=(0.5, 0.5),
        ...         rotate_limit=0,
        ...         beta_limit=(1.0, 1.0),  # Standard Gaussian (beta = 1)
        ...         noise_limit=(1.0, 1.0),  # No noise
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> gaussian_result = gaussian_like(image=image)
        >>> gaussian_image = gaussian_result["image"]
        >>> # The image will have a standard Gaussian blur applied
        >>>
        >>> # Example 2: Box-like blur (beta < 1)
        >>> box_like = A.Compose([
        ...     A.AdvancedBlur(
        ...         blur_limit=(7, 9),
        ...         sigma_x_limit=(0.6, 0.8),
        ...         sigma_y_limit=(0.6, 0.8),
        ...         rotate_limit=0,
        ...         beta_limit=(0.5, 0.7),  # Box-like blur (beta < 1)
        ...         noise_limit=(0.9, 1.1),  # Slight noise
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> box_result = box_like(image=image)
        >>> box_image = box_result["image"]
        >>> # The image will have a more box-like blur with heavier tails
        >>>
        >>> # Example 3: Peaked blur (beta > 1)
        >>> peaked = A.Compose([
        ...     A.AdvancedBlur(
        ...         blur_limit=(7, 9),
        ...         sigma_x_limit=(0.6, 0.8),
        ...         sigma_y_limit=(0.6, 0.8),
        ...         rotate_limit=0,
        ...         beta_limit=(3.0, 6.0),  # Peaked blur (beta > 1)
        ...         noise_limit=(0.9, 1.1),  # Slight noise
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> peaked_result = peaked(image=image)
        >>> peaked_image = peaked_result["image"]
        >>> # The image will have a more focused, peaked blur with lighter tails
        >>>
        >>> # Example 4: Anisotropic blur (directional)
        >>> directional = A.Compose([
        ...     A.AdvancedBlur(
        ...         blur_limit=(9, 11),
        ...         sigma_x_limit=(0.8, 1.0),    # Stronger x blur
        ...         sigma_y_limit=(0.2, 0.3),    # Weaker y blur
        ...         rotate_limit=(0, 0),         # No rotation
        ...         beta_limit=(1.0, 2.0),
        ...         noise_limit=(0.9, 1.1),
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> directional_result = directional(image=image)
        >>> directional_image = directional_result["image"]
        >>> # The image will have a horizontal directional blur
        >>>
        >>> # Example 5: Rotated directional blur
        >>> rotated = A.Compose([
        ...     A.AdvancedBlur(
        ...         blur_limit=(9, 11),
        ...         sigma_x_limit=(0.8, 1.0),    # Stronger x blur
        ...         sigma_y_limit=(0.2, 0.3),    # Weaker y blur
        ...         rotate_limit=(45, 45),       # 45 degree rotation
        ...         beta_limit=(1.0, 2.0),
        ...         noise_limit=(0.9, 1.1),
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> rotated_result = rotated(image=image)
        >>> rotated_image = rotated_result["image"]
        >>> # The image will have a diagonal directional blur
        >>>
        >>> # Example 6: Noisy blur
        >>> noisy = A.Compose([
        ...     A.AdvancedBlur(
        ...         blur_limit=(5, 7),
        ...         sigma_x_limit=(0.4, 0.6),
        ...         sigma_y_limit=(0.4, 0.6),
        ...         rotate_limit=(-30, 30),
        ...         beta_limit=(0.8, 1.2),
        ...         noise_limit=(0.7, 1.3),      # Strong noise variation
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> noisy_result = noisy(image=image)
        >>> noisy_image = noisy_result["image"]
        >>> # The image will have blur with significant noise in the kernel
        >>>
        >>> # Example 7: Random parameters (for general augmentation)
        >>> random_blur = A.Compose([
        ...     A.AdvancedBlur(p=0.5)  # Using default parameter ranges
        ... ])
        >>>
        >>> random_result = random_blur(image=image)
        >>> random_image = random_result["image"]
        >>> # The image may have a random advanced blur applied with 50% probability

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
        return fpixel.convolve(img, kernel=kernel)

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

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image for demonstration
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize defocus effects
        >>> cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (150, 150), 30, (0, 255, 0), -1)  # Green circle
        >>> cv2.putText(image, "Sharp Text", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        >>>
        >>> # Example 1: Subtle defocus effect (small aperture)
        >>> subtle_transform = A.Compose([
        ...     A.Defocus(
        ...         radius=(2, 3),           # Small defocus radius
        ...         alias_blur=(0.1, 0.2),   # Minimal aliasing
        ...         p=1.0                    # Always apply
        ...     )
        ... ])
        >>>
        >>> subtle_result = subtle_transform(image=image)
        >>> subtle_defocus = subtle_result["image"]
        >>> # The image will have a subtle defocus effect, with just slight blurring
        >>>
        >>> # Example 2: Moderate defocus effect (medium aperture)
        >>> moderate_transform = A.Compose([
        ...     A.Defocus(
        ...         radius=(4, 6),           # Medium defocus radius
        ...         alias_blur=(0.2, 0.3),   # Moderate aliasing
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> moderate_result = moderate_transform(image=image)
        >>> moderate_defocus = moderate_result["image"]
        >>> # The image will have a noticeable defocus effect, similar to a poorly focused camera
        >>>
        >>> # Example 3: Strong defocus effect (large aperture)
        >>> strong_transform = A.Compose([
        ...     A.Defocus(
        ...         radius=(8, 12),          # Large defocus radius
        ...         alias_blur=(0.4, 0.6),   # Strong aliasing
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> strong_result = strong_transform(image=image)
        >>> strong_defocus = strong_result["image"]
        >>> # The image will have a strong defocus effect, heavily blurring the details
        >>>
        >>> # Example 4: Using in a pipeline with other transforms
        >>> pipeline = A.Compose([
        ...     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
        ...     A.Defocus(radius=(3, 8), alias_blur=0.3, p=0.5),  # 50% chance of applying defocus
        ...     A.GaussNoise(var_limit=(10, 30), p=0.3)           # Possible noise after defocus
        ... ])
        >>>
        >>> pipeline_result = pipeline(image=image)
        >>> transformed_image = pipeline_result["image"]
        >>> # The image may have defocus blur applied with 50% probability

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

    This transform simulates the effect of zooming during exposure, creating a dynamic radial blur.
    It works by averaging multiple versions of the image at different zoom levels, creating
    a smooth transition from the center outward.

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
        uint8, float32

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> import cv2
        >>>
        >>> # Create a sample image for demonstration
        >>> image = np.zeros((300, 300, 3), dtype=np.uint8)
        >>> # Add some shapes to visualize zoom blur effects
        >>> cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Red square
        >>> cv2.circle(image, (150, 150), 30, (0, 255, 0), -1)  # Green circle
        >>> cv2.line(image, (50, 150), (250, 150), (0, 0, 255), 5)  # Blue line
        >>>
        >>> # Example 1: Subtle zoom blur
        >>> subtle_transform = A.Compose([
        ...     A.ZoomBlur(
        ...         max_factor=(1.05, 1.10),  # Small zoom range
        ...         step_factor=0.01,         # Fine steps
        ...         p=1.0                     # Always apply
        ...     )
        ... ])
        >>>
        >>> subtle_result = subtle_transform(image=image)
        >>> subtle_blur = subtle_result["image"]
        >>> # The image will have a subtle zoom blur effect, simulating a slight zoom during exposure
        >>>
        >>> # Example 2: Moderate zoom blur
        >>> moderate_transform = A.Compose([
        ...     A.ZoomBlur(
        ...         max_factor=(1.15, 1.25),  # Medium zoom range
        ...         step_factor=0.02,         # Medium steps
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> moderate_result = moderate_transform(image=image)
        >>> moderate_blur = moderate_result["image"]
        >>> # The image will have a more noticeable zoom blur effect
        >>>
        >>> # Example 3: Strong zoom blur
        >>> strong_transform = A.Compose([
        ...     A.ZoomBlur(
        ...         max_factor=(1.3, 1.5),    # Large zoom range
        ...         step_factor=(0.03, 0.05), # Larger steps (randomly chosen)
        ...         p=1.0
        ...     )
        ... ])
        >>>
        >>> strong_result = strong_transform(image=image)
        >>> strong_blur = strong_result["image"]
        >>> # The image will have a strong zoom blur effect, simulating fast zooming
        >>>
        >>> # Example 4: In a pipeline with other transforms
        >>> pipeline = A.Compose([
        ...     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        ...     A.ZoomBlur(max_factor=(1.1, 1.3), step_factor=0.02, p=0.5),
        ...     A.HorizontalFlip(p=0.5)
        ... ])
        >>>
        >>> pipeline_result = pipeline(image=image)
        >>> transformed_image = pipeline_result["image"]
        >>> # The image may have zoom blur applied with 50% probability

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
