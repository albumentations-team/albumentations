from typing import Any

__all__ = [
    "IAAEmboss",
    "IAASuperpixels",
    "IAASharpen",
    "IAAAdditiveGaussianNoise",
    "IAACropAndPad",
    "IAAFliplr",
    "IAAFlipud",
    "IAAAffine",
    "IAAPiecewiseAffine",
    "IAAPerspective",
]


class IAAStub:
    def __init__(self, *args: Any, **kwargs: Any):
        cls_name = self.__class__.__name__
        # Use getattr to safely access subclass attributes with a default value
        doc_link = "https://albumentations.ai/docs/api_reference/augmentations" + getattr(
            self, "doc_link", "/default_doc_link"
        )
        alternative = getattr(self, "alternative", "DefaultAlternative")
        raise RuntimeError(
            f"You are trying to use a deprecated augmentation '{cls_name}' which depends on the imgaug library, "
            f"but imgaug is not installed.\n\n"
            "There are two options to fix this error:\n"
            "1. [Recommended]. Switch to the Albumentations' implementation of the augmentation with the same API: "
            f"{alternative} - {doc_link}\n"
            "2. Install a version of Albumentations that contains imgaug by running "
            "'pip install -U albumentations[imgaug]'."
        )


class IAACropAndPad(IAAStub):
    alternative = "CropAndPad"
    doc_link = "/crops/transforms/#albumentations.augmentations.crops.transforms.CropAndPad"


class IAAFliplr(IAAStub):
    alternative = "HorizontalFlip"
    doc_link = "/transforms/#albumentations.augmentations.transforms.HorizontalFlip"


class IAAFlipud(IAAStub):
    alternative = "VerticalFlip"
    doc_link = "/transforms/#albumentations.augmentations.transforms.VerticalFlip"


class IAAEmboss(IAAStub):
    alternative = "Emboss"
    doc_link = "/transforms/#albumentations.augmentations.transforms.Emboss"


class IAASuperpixels(IAAStub):
    alternative = "Superpixels"
    doc_link = "/transforms/#albumentations.augmentations.transforms.Superpixels"


class IAASharpen(IAAStub):
    alternative = "Sharpen"
    doc_link = "/transforms/#albumentations.augmentations.transforms.Sharpen"


class IAAAdditiveGaussianNoise(IAAStub):
    alternative = "GaussNoise"
    doc_link = "/transforms/#albumentations.augmentations.transforms.GaussNoise"


class IAAPiecewiseAffine(IAAStub):
    alternative = "PiecewiseAffine"
    doc_link = "/geometric/transforms/#albumentations.augmentations.geometric.transforms.PiecewiseAffine"


class IAAAffine(IAAStub):
    alternative = "Affine"
    doc_link = "/geometric/transforms/#albumentations.augmentations.geometric.transforms.Affine"


class IAAPerspective(IAAStub):
    alternative = "Perspective"
    doc_link = "/geometric/transforms/#albumentations.augmentations.geometric.transforms.Perspective"
