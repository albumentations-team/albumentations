from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

import numpy as np
from albucore import get_num_channels
from pydantic import AfterValidator, Field

from albumentations.core.pydantic import check_1plus
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform

from .functional import channel_dropout

__all__ = ["ChannelDropout"]

MIN_DROPOUT_CHANNEL_LIST_LENGTH = 2


class ChannelDropout(ImageOnlyTransform):
    """Randomly drop channels in the input image.

    This transform randomly selects a number of channels to drop from the input image
    and replaces them with a specified fill value. This can improve model robustness
    to missing or corrupted channels.

    The technique is conceptually similar to:
    - Dropout layers in neural networks, which randomly set input units to 0 during training.
    - CoarseDropout augmentation, which drops out regions in the spatial dimensions of the image.

    However, ChannelDropout operates on the channel dimension, effectively "dropping out"
    entire color channels or feature maps.

    Args:
        channel_drop_range (tuple[int, int]): Range from which to choose the number
            of channels to drop. The actual number will be randomly selected from
            the inclusive range [min, max]. Default: (1, 1).
        fill_value (float): Pixel value used to fill the dropped channels.
            Default: 0.
        p (float): Probability of applying the transform. Must be in the range
            [0, 1]. Default: 0.5.

    Raises:
        NotImplementedError: If the input image has only one channel.
        ValueError: If the upper bound of channel_drop_range is greater than or
            equal to the number of channels in the input image.

    Targets:
        image

    Image types:
        uint8, float32

    Example:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> transform = A.ChannelDropout(channel_drop_range=(1, 2), fill_value=128, p=1.0)
        >>> result = transform(image=image)
        >>> dropped_image = result['image']
        >>> assert dropped_image.shape == image.shape
        >>> assert np.any(dropped_image != image)  # Some channels should be different

    Note:
        - The number of channels to drop is randomly chosen within the specified range.
        - Channels are randomly selected for dropping.
        - This transform is not applicable to single-channel (grayscale) images.
        - The transform will raise an error if it's not possible to drop the specified
          number of channels (e.g., trying to drop 3 channels from an RGB image).
        - This augmentation can be particularly useful for training models to be robust
          against missing or corrupted channel data in multi-spectral or hyperspectral imagery.

    """

    class InitSchema(BaseTransformInitSchema):
        channel_drop_range: Annotated[tuple[int, int], AfterValidator(check_1plus)]
        fill_value: Annotated[float, Field(description="Pixel value for the dropped channel.")]

    def __init__(
        self,
        channel_drop_range: tuple[int, int] = (1, 1),
        fill_value: float = 0,
        always_apply: bool | None = None,
        p: float = 0.5,
    ):
        super().__init__(p=p, always_apply=always_apply)

        self.channel_drop_range = channel_drop_range
        self.fill_value = fill_value

    def apply(self, img: np.ndarray, channels_to_drop: tuple[int, ...], **params: Any) -> np.ndarray:
        return channel_dropout(img, channels_to_drop, self.fill_value)

    def get_params_dependent_on_data(self, params: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
        image = data["image"] if "image" in data else data["images"][0]
        num_channels = get_num_channels(image)

        if num_channels == 1:
            msg = "Images has one channel. ChannelDropout is not defined."
            raise NotImplementedError(msg)

        if self.channel_drop_range[1] >= num_channels:
            msg = "Can not drop all channels in ChannelDropout."
            raise ValueError(msg)

        num_drop_channels = self.py_random.randint(*self.channel_drop_range)

        channels_to_drop = self.py_random.sample(range(num_channels), k=num_drop_channels)

        return {"channels_to_drop": channels_to_drop}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "channel_drop_range", "fill_value"
