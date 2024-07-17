from __future__ import annotations

from typing import Any, Callable, Literal, cast, TYPE_CHECKING
import random
import numpy as np
from PIL import ImageFont
from pydantic import AfterValidator, model_validator

from albumentations.augmentations.mixing.functional import copy_and_paste_blend
import albumentations.augmentations.text.functional as ftext
import re
from typing_extensions import Annotated, Self

from albumentations.core.bbox_utils import check_bbox, denormalize_bbox
from albumentations.core.pydantic import check_01, nondecreasing
from albumentations.core.transforms_interface import BaseTransformInitSchema, ImageOnlyTransform
from albumentations.core.types import BoxType, SizeType

if TYPE_CHECKING:
    from pathlib import Path


try:
    from rake_nltk import Rake

    RAKE_AVAILABLE = True
except ImportError:
    RAKE_AVAILABLE = False


class TextAugmenter(ImageOnlyTransform):
    class InitSchema(BaseTransformInitSchema):
        font_path: Path
        stopwords: list[str] | None = None
        pos_tagger: Callable[[str], list[str]] | None = None
        augmentations: tuple[str, ...] | list[str] = ("insertion", "swap", "deletion", "kreplacement")
        fraction_range: Annotated[tuple[int, int], AfterValidator(nondecreasing), AfterValidator(check_01)]
        font_size_fraction_range: Annotated[tuple[int, int], AfterValidator(nondecreasing), AfterValidator(check_01)]

        metadata_key: str = "textaug_metadata"

        @model_validator(mode="after")
        def validate_input(self) -> Self:
            if not self.stopwords:
                self.augmentations = [aug for aug in self.augmentations if aug not in {"kreplacement", "insertion"}]

            if not self.pos_tagger:
                self.augmentations = [aug for aug in self.augmentations if aug != "kreplacement"]

            self.stopwords = self.stopwords or []

            return self

    def __init__(
        self,
        font_path: Path,
        stopwords: list[str] | None = None,
        pos_tagger: Callable[[str], list[str]] | None = None,
        augmentations: tuple[str, ...] = ("insertion", "swap", "deletion", "kreplacement"),
        fraction_range: tuple[float, float] = (0.1, 0.9),
        font_size_fraction_range: tuple[float, float] = (0.8, 0.9),
        metadata_key: str = "textaug_metadata",
        alway_apply: bool | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p, always_apply=alway_apply)
        self.metadata_key = metadata_key
        self.font_path = font_path
        self.fraction_range = fraction_range
        self.stopwords = stopwords
        self.pos_tagger = pos_tagger
        self.augmentations = list(augmentations)
        self.font_size_fraction_range = font_size_fraction_range

        if RAKE_AVAILABLE:
            if self.stopwords:
                self.rake = Rake(self.stopwords)
            else:
                self.rake = Rake()
        else:
            self.rake = None

    @property
    def targets_as_params(self) -> list[str]:
        return [self.metadata_key, "image"]

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

    def preprocess_metadata(self, img_shape: SizeType, bbox: BoxType, text: str) -> dict[str, Any]:
        image_height, image_width = img_shape[:2]

        check_bbox(bbox)
        denormalized_bbox = denormalize_bbox(bbox[:4], rows=image_height, cols=image_width)

        x_min, y_min, x_max, y_max = (int(x) for x in denormalized_bbox[:4])
        bbox_height = y_max - y_min
        bbox_width = x_max - x_min

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

        overlay_image = ftext.render_text((bbox_height, bbox_width), augmented_text, font)
        offset = (y_min, x_min)

        return {
            "overlay_image": overlay_image,
            "offset": offset,
        }

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        metadata = params[self.metadata_key]
        img_shape = params["rows"], params["cols"]
        bboxes = metadata["bboxes"]
        texts = metadata["texts"]
        fraction = random.uniform(*self.fraction_range)

        num_lines_to_modify = int(len(texts) * fraction)
        bbox_indices_to_update = random.sample(range(len(texts)), num_lines_to_modify)

        overlay_data = [
            self.preprocess_metadata(img_shape, bboxes[index], texts[index]) for index in bbox_indices_to_update
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
        for data in overlay_data:
            overlay_image = data["overlay_image"]
            offset = data["offset"]
            img = copy_and_paste_blend(img, overlay_image, offset=offset)
        return img
