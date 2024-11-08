from __future__ import annotations

from albumentations.augmentations.geometric.transforms import HorizontalFlip
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.types import Targets

__all__ = [
    "TimeReverse",
]


class TimeReverse(HorizontalFlip):
    """Reverse the time axis of a spectrogram image, also known as time inversion.

    Time inversion of a spectrogram is analogous to the random flip of an image,
    an augmentation technique widely used in the visual domain. This can be relevant
    in the context of audio classification tasks when working with spectrograms.
    The technique was successfully applied in the AudioCLIP paper, which extended
    CLIP to handle image, text, and audio inputs.

    This transform is implemented as a subclass of HorizontalFlip since reversing
    time in a spectrogram is equivalent to flipping the image horizontally.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        This transform is functionally identical to HorizontalFlip but provides
        a more semantically meaningful name when working with spectrograms and
        other time-series visualizations.

    References:
        - AudioCLIP paper: https://arxiv.org/abs/2106.13043
        - Audiomentations: https://iver56.github.io/audiomentations/waveform_transforms/reverse/
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.BBOXES, Targets.KEYPOINTS)

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(
        self,
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
