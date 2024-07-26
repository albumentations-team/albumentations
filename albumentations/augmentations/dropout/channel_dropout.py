from __future__ import annotations

import random
from typing import Any, Mapping

from typing_extensions import Annotated

from albucore import get_num_channels
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform

from .functional import channel_dropout


import numpy as np
from pydantic import Field

from albumentations.core.pydantic import OnePlusIntRangeType
from albumentations.core.types import ColorType

__all__ = ["ChannelDropout"]

MIN_DROPOUT_CHANNEL_LIST_LENGTH = 2


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

    class InitSchema(BaseTransformInitSchema):
        channel_drop_range: OnePlusIntRangeType = (1, 1)
        fill_value: Annotated[ColorType, Field(description="Pixel value for the dropped channel.")]

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

        num_drop_channels = random.randint(self.channel_drop_range[0], self.channel_drop_range[1])

        channels_to_drop = random.sample(range(num_channels), k=num_drop_channels)

        return {"channels_to_drop": channels_to_drop}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "channel_drop_range", "fill_value"
