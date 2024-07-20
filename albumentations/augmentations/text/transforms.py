from __future__ import annotations

from typing import Any, Callable, Literal, cast
import random
import numpy as np
from PIL import ImageFont
from pydantic import AfterValidator, model_validator
from albucore.utils import get_num_channels

import albumentations.augmentations.text.functional as ftext
import re
from typing_extensions import Annotated, Self

from albumentations.core.bbox_utils import check_bbox, denormalize_bbox
from albumentations.core.pydantic import check_01, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform
from albumentations.core.types import BoxType, ColorType

try:
    from rake_nltk import Rake

    RAKE_AVAILABLE = True
except ImportError:
    RAKE_AVAILABLE = False


class TextImage(ImageOnlyTransform):
    class InitSchema(BaseTransformInitSchema):
        font_path: str
        stopwords: list[str] | None
        pos_tagger: Callable[[str], list[str]] | None
        augmentations: tuple[str, ...] | list[str] | None
        fraction_range: Annotated[tuple[float, float], AfterValidator(nondecreasing), AfterValidator(check_01)]
        font_size_fraction_range: Annotated[
            tuple[float, float],
            AfterValidator(nondecreasing),
            AfterValidator(check_01),
        ]
        font_color: list[ColorType | str] | ColorType | str
        metadata_key: str

        @model_validator(mode="after")
        def validate_input(self) -> Self:
            if self.augmentations is None:
                self.augmentations = ()
            if not self.stopwords:
                self.augmentations = [aug for aug in self.augmentations if aug not in {"kreplacement", "insertion"}]

            if not self.pos_tagger:
                self.augmentations = [aug for aug in self.augmentations if aug != "kreplacement"]

            self.stopwords = self.stopwords or ["the", "is", "in", "at", "of"]

            return self

    def __init__(
        self,
        font_path: str,
        stopwords: list[str] | None = None,
        pos_tagger: Callable[[str], list[str]] | None = None,
        augmentations: tuple[str, ...] = (),
        fraction_range: tuple[float, float] = (1.0, 1.0),
        font_size_fraction_range: tuple[float, float] = (0.8, 0.9),
        font_color: list[ColorType | str] | ColorType | str = "black",
        metadata_key: str = "textimage_metadata",
        always_apply: bool | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p, always_apply=always_apply)
        self.metadata_key = metadata_key
        self.font_path = font_path
        self.fraction_range = fraction_range
        self.stopwords = stopwords
        self.pos_tagger = pos_tagger
        self.augmentations = list(augmentations)
        self.font_size_fraction_range = font_size_fraction_range
        self.font_color = font_color

        if RAKE_AVAILABLE:
            self.rake = Rake(self.stopwords)
        else:
            self.rake = None

    @property
    def targets_as_params(self) -> list[str]:
        return [self.metadata_key, "image"]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "font_path",
            "stopwords",
            "pos_tagger",
            "augmentations",
            "fraction_range",
            "font_size_fraction_range",
            "font_color",
            "metadata_key",
        )

    def random_aug(
        self,
        text: str,
        fraction: float,
        choice: Literal["insertion", "swap", "deletion", "kreplacement"],
    ) -> str:
        words = [word for word in text.strip().split() if word]
        num_words = len(words)
        num_words_to_modify = max(1, int(fraction * num_words))

        if choice == "insertion":
            augmented_words = ftext.insert_random_stopwords(words, num_words_to_modify, self.stopwords)
        elif choice == "swap":
            augmented_words = ftext.swap_random_words(words, num_words_to_modify)
        elif choice == "deletion":
            augmented_words = ftext.delete_random_words(words, num_words_to_modify)
        elif choice == "kreplacement":
            return ftext.augment_text_with_synonyms(text, [3], self.pos_tagger, self.rake)
        else:
            raise ValueError("Invalid choice. Choose from 'insertion', 'kreplacement', 'swap', or 'deletion'.")

        result_sentence = " ".join(augmented_words)
        result_sentence = re.sub(" +", " ", result_sentence).strip()
        return result_sentence if result_sentence != text else ""

    def preprocess_metadata(self, image: np.ndarray, bbox: BoxType, text: str) -> dict[str, Any]:
        num_channels = get_num_channels(image)

        if num_channels not in {1, 3}:
            raise TypeError("Image must be either 1 or 3 channels.")

        image_height, image_width = image.shape[:2]

        check_bbox(bbox)
        denormalized_bbox = denormalize_bbox(bbox[:4], rows=image_height, cols=image_width)

        x_min, y_min, x_max, y_max = (int(x) for x in denormalized_bbox[:4])
        bbox_height = y_max - y_min

        font_size_fraction = random.uniform(*self.font_size_fraction_range)

        font = ImageFont.truetype(str(self.font_path), int(font_size_fraction * bbox_height))

        if not self.augmentations:
            augmented_text = text
        else:
            augmentation = random.choice(self.augmentations)
            augmented_text = self.random_aug(
                text,
                0.5,
                choice=cast(Literal["insertion", "swap", "deletion", "kreplacement"], augmentation),
            )

        font_color = random.choice(self.font_color) if isinstance(self.font_color, list) else self.font_color

        return {
            "bbox_coords": (x_min, y_min, x_max, y_max),
            "text": augmented_text,
            "font": font,
            "font_color": font_color,
        }

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        metadata = params[self.metadata_key]

        if metadata == []:
            return {
                "overlay_data": [],
            }

        if isinstance(metadata, dict):
            metadata = [metadata]

        image = params["image"]

        fraction = random.uniform(*self.fraction_range)

        num_bboxes_to_modify = int(len(metadata) * fraction)

        bbox_indices_to_update = random.sample(range(len(metadata)), num_bboxes_to_modify)

        overlay_data = [
            self.preprocess_metadata(image, metadata[index]["bbox"], metadata[index]["text"])
            for index in bbox_indices_to_update
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
        return ftext.render_text(img, overlay_data)
