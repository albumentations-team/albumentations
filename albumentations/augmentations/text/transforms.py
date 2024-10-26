from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import AfterValidator

import albumentations.augmentations.text.functional as ftext
from albumentations.core.bbox_utils import check_bboxes, denormalize_bboxes
from albumentations.core.pydantic import check_01, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform
from albumentations.core.types import ColorType

__all__ = ["TextImage"]


class TextImage(ImageOnlyTransform):
    """Apply text rendering transformations on images.

    This class supports rendering text directly onto images using a variety of configurations,
    such as custom fonts, font sizes, colors, and augmentation methods. The text can be placed
    inside specified bounding boxes.

    Args:
        font_path (str | Path): Path to the font file to use for rendering text.
        stopwords (list[str] | None): List of stopwords for text augmentation.
        augmentations (tuple[str | None, ...] | list[str | None]): List of text augmentations to apply.
            None: text is printed as is
            "insertion": insert random stop words into the text.
            "swap": swap random words in the text.
            "deletion": delete random words from the text.
        fraction_range (tuple[float, float]): Range for selecting a fraction of bounding boxes to modify.
        font_size_fraction_range (tuple[float, float]): Range for selecting the font size as a fraction of
            bounding box height.
        font_color (list[str] | str): List of possible font colors or a single font color.
        clear_bg (bool): Whether to clear the background before rendering text.
        metadata_key (str): Key to access metadata in the parameters.
        p (float): Probability of applying the transform.

    Targets:
        image

    Image types:
        uint8, float32

    Reference:
        https://github.com/danaaubakirova/doc-augmentation

    Examples:
        >>> import albumentations as A
        >>> transform = A.Compose([
            A.TextImage(
                font_path=Path("/path/to/font.ttf"),
                stopwords=["the", "is", "in"],
                augmentations=("insertion", "deletion"),
                fraction_range=(0.5, 1.0),
                font_size_fraction_range=(0.5, 0.9),
                font_color=["red", "green", "blue"],
                metadata_key="text_metadata",
                p=0.5
            )
        ])
        >>> transformed = transform(image=my_image, text_metadata=my_metadata)
        >>> image = transformed['image']
        # This will render text on `my_image` based on the metadata provided in `my_metadata`.
    """

    class InitSchema(BaseTransformInitSchema):
        font_path: str | Path
        stopwords: tuple[str, ...]
        augmentations: tuple[str | None, ...] | list[str | None]
        fraction_range: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_01)]
        font_size_fraction_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_01),
        ]
        font_color: list[ColorType | str] | ColorType | str
        clear_bg: bool
        metadata_key: str

    def __init__(
        self,
        font_path: str | Path,
        stopwords: tuple[str, ...] = ("the", "is", "in", "at", "of"),
        augmentations: tuple[Literal["insertion", "swap", "deletion"] | None] = (None,),
        fraction_range: tuple[float, float] = (1.0, 1.0),
        font_size_fraction_range: tuple[float, float] = (0.8, 0.9),
        font_color: list[ColorType | str] | ColorType | str = "black",
        clear_bg: bool = False,
        metadata_key: str = "textimage_metadata",
        always_apply: bool | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p, always_apply=always_apply)
        self.metadata_key = metadata_key
        self.font_path = font_path
        self.fraction_range = fraction_range
        self.stopwords = stopwords
        self.augmentations = list(augmentations)
        self.font_size_fraction_range = font_size_fraction_range
        self.font_color = font_color
        self.clear_bg = clear_bg

    @property
    def targets_as_params(self) -> list[str]:
        return [self.metadata_key]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "font_path",
            "stopwords",
            "augmentations",
            "fraction_range",
            "font_size_fraction_range",
            "font_color",
            "metadata_key",
            "clear_bg",
        )

    def random_aug(
        self,
        text: str,
        fraction: float,
        choice: Literal["insertion", "swap", "deletion"],
    ) -> str:
        words = [word for word in text.strip().split() if word]
        num_words = len(words)
        num_words_to_modify = max(1, int(fraction * num_words))

        if choice == "insertion":
            result_sentence = ftext.insert_random_stopwords(words, num_words_to_modify, self.stopwords, self.py_random)
        elif choice == "swap":
            result_sentence = ftext.swap_random_words(words, num_words_to_modify, self.py_random)
        elif choice == "deletion":
            result_sentence = ftext.delete_random_words(words, num_words_to_modify, self.py_random)
        else:
            raise ValueError("Invalid choice. Choose from 'insertion', 'swap', or 'deletion'.")

        result_sentence = re.sub(" +", " ", result_sentence).strip()
        return result_sentence if result_sentence != text else ""

    def preprocess_metadata(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
        text: str,
        bbox_index: int,
    ) -> dict[str, Any]:
        try:
            from PIL import ImageFont
        except ImportError as err:
            raise ImportError(
                "ImageFont from PIL is required to use TextImage transform. Install it with `pip install Pillow`.",
            ) from err
        check_bboxes(np.array([bbox]))
        denormalized_bbox = denormalize_bboxes(np.array([bbox]), image.shape[:2])[0]

        x_min, y_min, x_max, y_max = (int(x) for x in denormalized_bbox[:4])
        bbox_height = y_max - y_min

        font_size_fraction = self.py_random.uniform(*self.font_size_fraction_range)

        font = ImageFont.truetype(str(self.font_path), int(font_size_fraction * bbox_height))

        if not self.augmentations or self.augmentations is None:
            augmented_text = text
        else:
            augmentation = self.py_random.choice(self.augmentations)

            augmented_text = text if augmentation is None else self.random_aug(text, 0.5, choice=augmentation)

        font_color = self.py_random.choice(self.font_color) if isinstance(self.font_color, list) else self.font_color

        return {
            "bbox_coords": (x_min, y_min, x_max, y_max),
            "bbox_index": bbox_index,
            "original_text": text,
            "text": augmented_text,
            "font": font,
            "font_color": font_color,
        }

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image = data["image"]

        metadata = data[self.metadata_key]

        if metadata == []:
            return {
                "overlay_data": [],
            }

        if isinstance(metadata, dict):
            metadata = [metadata]

        fraction = self.py_random.uniform(*self.fraction_range)

        num_bboxes_to_modify = int(len(metadata) * fraction)

        bbox_indices_to_update = self.py_random.sample(range(len(metadata)), num_bboxes_to_modify)

        overlay_data = [
            self.preprocess_metadata(image, metadata[bbox_index]["bbox"], metadata[bbox_index]["text"], bbox_index)
            for bbox_index in bbox_indices_to_update
        ]

        return {
            "overlay_data": overlay_data,
        }

    def apply(
        self,
        img: np.ndarray,
        overlay_data: list[dict[str, Any]],
        **params: Any,
    ) -> np.ndarray:
        return ftext.render_text(img, overlay_data, clear_bg=self.clear_bg)

    def apply_with_params(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        res = super().apply_with_params(params, *args, **kwargs)
        res["overlay_data"] = [
            {
                "bbox_coords": overlay["bbox_coords"],
                "text": overlay["text"],
                "original_text": overlay["original_text"],
                "bbox_index": overlay["bbox_index"],
                "font_color": overlay["font_color"],
            }
            for overlay in params["overlay_data"]
        ]

        return res
