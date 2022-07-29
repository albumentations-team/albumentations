import random
from typing import Any, Mapping, Tuple, Union

import numpy as np

from albumentations.core.transforms_interface import ImageOnlyTransform

from .functional import channel_dropout

__all__ = ["ChannelDropout"]


class ChannelDropout(ImageOnlyTransform):
    """Randomly Drop Channels in the input Image.

    Args:
        channel_drop_range (int, int): range from which we choose the number of channels to drop.
        fill_value (int, float): pixel value for the dropped channel.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, unit32, float32
    """

    def __init__(
        self,
        channel_drop_range: Tuple[int, int] = (1, 1),
        fill_value: Union[int, float] = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(ChannelDropout, self).__init__(always_apply, p)

        self.channel_drop_range = channel_drop_range

        self.min_channels = channel_drop_range[0]
        self.max_channels = channel_drop_range[1]

        if not 1 <= self.min_channels <= self.max_channels:
            raise ValueError("Invalid channel_drop_range. Got: {}".format(channel_drop_range))

        self.fill_value = fill_value

    def apply(self, img: np.ndarray, channels_to_drop: Tuple[int, ...] = (0,), **params) -> np.ndarray:
        return channel_dropout(img, channels_to_drop, self.fill_value)

    def get_params_dependent_on_targets(self, params: Mapping[str, Any]):
        img = params["image"]

        num_channels = img.shape[-1]

        if len(img.shape) == 2 or num_channels == 1:
            raise NotImplementedError("Images has one channel. ChannelDropout is not defined.")

        if self.max_channels >= num_channels:
            raise ValueError("Can not drop all channels in ChannelDropout.")

        num_drop_channels = random.randint(self.min_channels, self.max_channels)

        channels_to_drop = random.sample(range(num_channels), k=num_drop_channels)

        return {"channels_to_drop": channels_to_drop}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "channel_drop_range", "fill_value"

    @property
    def targets_as_params(self):
        return ["image"]
