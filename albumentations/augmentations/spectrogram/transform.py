"""Transforms for spectrogram augmentation.

This module provides transforms specifically designed for augmenting spectrograms
in audio processing tasks. Includes time reversal, time masking, and frequency
masking transforms commonly used in audio machine learning applications.
"""

from __future__ import annotations

from warnings import warn

from pydantic import Field

from albumentations.augmentations.dropout.xy_masking import XYMasking
from albumentations.augmentations.geometric.flip import HorizontalFlip
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.type_definitions import ALL_TARGETS

__all__ = [
    "FrequencyMasking",
    "TimeMasking",
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
        image, mask, bboxes, keypoints, volume, mask3d

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

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        pass

    def __init__(
        self,
        p: float = 0.5,
    ):
        warn(
            "TimeReverse is an alias for HorizontalFlip transform. "
            "Consider using HorizontalFlip directly from albumentations.HorizontalFlip. ",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(p=p)


class TimeMasking(XYMasking):
    """Apply masking to a spectrogram in the time domain.

    This transform masks random segments along the time axis of a spectrogram,
    implementing the time masking technique proposed in the SpecAugment paper.
    Time masking helps in training models to be robust against temporal variations
    and missing information in audio signals.

    This is a specialized version of XYMasking configured for time masking only.
    For more advanced use cases (e.g., multiple masks, frequency masking, or custom
    fill values), consider using XYMasking directly.

    Args:
        time_mask_param (int): Maximum possible length of the mask in the time domain.
            Must be a positive integer. Length of the mask is uniformly sampled from (0, time_mask_param).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        This transform is implemented as a subset of XYMasking with fixed parameters:
        - Single horizontal mask (num_masks_x=1)
        - No vertical masks (num_masks_y=0)
        - Zero fill value
        - Random mask length up to time_mask_param

        For more flexibility, including:
        - Multiple masks
        - Custom fill values
        - Frequency masking
        - Combined time-frequency masking
        Consider using albumentations.XYMasking directly.

    References:
        - SpecAugment paper: https://arxiv.org/abs/1904.08779
        - Original implementation: https://pytorch.org/audio/stable/transforms.html#timemask

    """

    class InitSchema(BaseTransformInitSchema):
        time_mask_param: int = Field(gt=0)

    def __init__(
        self,
        time_mask_param: int = 40,
        p: float = 0.5,
    ):
        warn(
            "TimeMasking is a specialized version of XYMasking. "
            "For more flexibility (multiple masks, custom fill values, frequency masking), "
            "consider using XYMasking directly from albumentations.XYMasking.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(
            num_masks_x=1,
            num_masks_y=0,
            mask_x_length=(0, time_mask_param),
            fill=0,
            fill_mask=0,
            p=p,
        )
        self.time_mask_param = time_mask_param


class FrequencyMasking(XYMasking):
    """Apply masking to a spectrogram in the frequency domain.

    This transform masks random segments along the frequency axis of a spectrogram,
    implementing the frequency masking technique proposed in the SpecAugment paper.
    Frequency masking helps in training models to be robust against frequency variations
    and missing spectral information in audio signals.

    This is a specialized version of XYMasking configured for frequency masking only.
    For more advanced use cases (e.g., multiple masks, time masking, or custom
    fill values), consider using XYMasking directly.

    Args:
        freq_mask_param (int): Maximum possible length of the mask in the frequency domain.
            Must be a positive integer. Length of the mask is uniformly sampled from (0, freq_mask_param).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Number of channels:
        Any

    Note:
        This transform is implemented as a subset of XYMasking with fixed parameters:
        - Single vertical mask (num_masks_y=1)
        - No horizontal masks (num_masks_x=0)
        - Zero fill value
        - Random mask length up to freq_mask_param

        For more flexibility, including:
        - Multiple masks
        - Custom fill values
        - Time masking
        - Combined time-frequency masking
        Consider using albumentations.XYMasking directly.

    References:
        - SpecAugment paper: https://arxiv.org/abs/1904.08779
        - Original implementation: https://pytorch.org/audio/stable/transforms.html#freqmask

    """

    class InitSchema(BaseTransformInitSchema):
        freq_mask_param: int = Field(gt=0)

    def __init__(
        self,
        freq_mask_param: int = 30,
        p: float = 0.5,
    ):
        warn(
            "FrequencyMasking is a specialized version of XYMasking. "
            "For more flexibility (multiple masks, custom fill values, time masking), "
            "consider using XYMasking directly from albumentations.XYMasking.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(
            p=p,
            fill=0,
            fill_mask=0,
            mask_y_length=(0, freq_mask_param),
            num_masks_x=0,
            num_masks_y=1,
        )
        self.freq_mask_param = freq_mask_param
