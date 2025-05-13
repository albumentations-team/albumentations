"""Transforms for type conversion between float and other data types.

This module provides transform classes for converting image data types, primarily
for converting between floating point and integer representations. These transforms
are useful for preprocessing before neural network input (ToFloat) and for converting
network outputs back to standard image formats (FromFloat).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from albucore import (
    clip,
    from_float,
    get_max_value,
    to_float,
)
from pydantic import (
    model_validator,
)
from typing_extensions import Literal, Self

from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    ImageOnlyTransform,
)

__all__ = [
    "FromFloat",
    "ToFloat",
]


class ToFloat(ImageOnlyTransform):
    """Convert the input image to a floating-point representation.

    This transform divides pixel values by `max_value` to get a float32 output array
    where all values lie in the range [0, 1.0]. It's useful for normalizing image data
    before feeding it into neural networks or other algorithms that expect float input.

    Args:
        max_value (float | None): The maximum possible input value. If None, the transform
            will try to infer the maximum value by inspecting the data type of the input image:
            - uint8: 255
            - uint16: 65535
            - uint32: 4294967295
            - float32: 1.0
            Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, volume

    Image types:
        uint8, uint16, uint32, float32

    Returns:
        np.ndarray: Image in floating point representation, with values in range [0, 1.0].

    Note:
        - If the input image is already float32 with values in [0, 1], it will be returned unchanged.
        - For integer types (uint8, uint16, uint32), the function will scale the values to [0, 1] range.
        - The output will always be float32, regardless of the input type.
        - This transform is often used as a preprocessing step before applying other transformations
          or feeding the image into a neural network.

    Raises:
        TypeError: If the input image data type is not supported.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>>
        # Convert uint8 image to float
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ToFloat(max_value=None)
        >>> float_image = transform(image=image)['image']
        >>> assert float_image.dtype == np.float32
        >>> assert 0 <= float_image.min() <= float_image.max() <= 1.0
        >>>
        # Convert uint16 image to float with custom max_value
        >>> image = np.random.randint(0, 4096, (100, 100, 3), dtype=np.uint16)
        >>> transform = A.ToFloat(max_value=4095)
        >>> float_image = transform(image=image)['image']
        >>> assert float_image.dtype == np.float32
        >>> assert 0 <= float_image.min() <= float_image.max() <= 1.0

    See Also:
        FromFloat: The inverse operation, converting from float back to the original data type.

    """

    class InitSchema(BaseTransformInitSchema):
        max_value: float | None

    def __init__(
        self,
        max_value: float | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the ToFloat transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the ToFloat transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied ToFloat transform.

        """
        return to_float(img, self.max_value)


class FromFloat(ImageOnlyTransform):
    """Convert an image from floating point representation to the specified data type.

    This transform is designed to convert images from a normalized floating-point representation
    (typically with values in the range [0, 1]) to other data types, scaling the values appropriately.

    Args:
        dtype (str): The desired output data type. Supported types include 'uint8', 'uint16',
                     'uint32'. Default: 'uint8'.
        max_value (float | None): The maximum value for the output dtype. If None, the transform
                                  will attempt to infer the maximum value based on the dtype.
                                  Default: None.
        p (float): Probability of applying the transform. Default: 1.0.

    Targets:
        image, volume

    Image types:
        float32, float64

    Note:
        - This is the inverse transform for ToFloat.
        - Input images are expected to be in floating point format with values in the range [0, 1].
        - For integer output types (uint8, uint16, uint32), the function will scale the values
          to the appropriate range (e.g., 0-255 for uint8).
        - For float output types (float32, float64), the values will remain in the [0, 1] range.
        - The transform uses the `from_float` function internally, which ensures output values
          are within the valid range for the specified dtype.

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> transform = A.FromFloat(dtype='uint8', max_value=None, p=1.0)
        >>> image = np.random.rand(100, 100, 3).astype(np.float32)  # Float image in [0, 1] range
        >>> result = transform(image=image)
        >>> uint8_image = result['image']
        >>> assert uint8_image.dtype == np.uint8
        >>> assert uint8_image.min() >= 0 and uint8_image.max() <= 255

    """

    class InitSchema(BaseTransformInitSchema):
        dtype: Literal["uint8", "uint16", "uint32"]
        max_value: float | None

        @model_validator(mode="after")
        def _update_max_value(self) -> Self:
            if self.max_value is None:
                self.max_value = get_max_value(np.dtype(self.dtype))

            return self

    def __init__(
        self,
        dtype: Literal["uint8", "uint16", "uint32"] = "uint8",
        max_value: float | None = None,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.dtype = dtype
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the FromFloat transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the FromFloat transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied FromFloat transform.

        """
        return from_float(img, np.dtype(self.dtype), self.max_value)

    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the FromFloat transform to the input images.

        Args:
            images (np.ndarray): The input images to apply the FromFloat transform to.
            **params (Any): Additional parameters (not used in this transform).

        """
        return clip(np.rint(images * self.max_value), np.dtype(self.dtype), inplace=True)

    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the FromFloat transform to the input volume.

        Args:
            volume (np.ndarray): The input volume to apply the FromFloat transform to.
            **params (Any): Additional parameters (not used in this transform).

        """
        return self.apply_to_images(volume, **params)

    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the FromFloat transform to the input volumes.

        Args:
            volumes (np.ndarray): The input volumes to apply the FromFloat transform to.
            **params (Any): Additional parameters (not used in this transform).

        """
        return self.apply_to_images(volumes, **params)
