from __future__ import annotations
import random
from typing import TYPE_CHECKING, Any
from albucore.utils import preserve_channel_dim, MONO_CHANNEL_DIMENSIONS, NUM_MULTI_CHANNEL_DIMENSIONS, NUM_RGB_CHANNELS
import cv2

import re

import numpy as np

from albumentations.core.types import PAIR

import albumentations.augmentations.functional as fmain

# Importing wordnet and other dependencies only for type checking
if TYPE_CHECKING:
    from nltk.tag import StanfordPOSTagger
    from rake_nltk import Rake
    from PIL import Image


# Try to import wordnet, handle if not available
try:
    from nltk.corpus import wordnet

    nltk_available = True
except ImportError:
    nltk_available = False


def delete_random_words(words: list[str], num_words: int) -> str:
    if num_words >= len(words):
        return ""

    indices_to_delete = random.sample(range(len(words)), num_words)
    new_words = [word for idx, word in enumerate(words) if idx not in indices_to_delete]
    return " ".join(new_words)


def swap_random_words(words: list[str], num_words: int = 1) -> str:
    if num_words == 0 or len(words) < PAIR:
        return " ".join(words)

    words = words.copy()

    for _ in range(num_words):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)


def insert_random_stopwords(words: list[str], num_insertions: int = 1, stopwords: list[str] | None = None) -> str:
    if stopwords is None:
        stopwords = ["and", "the", "is", "in", "at", "of"]  # Default stopwords if none provided

    for _ in range(num_insertions):
        idx = random.randint(0, len(words))
        words.insert(idx, random.choice(stopwords))
    return " ".join(words)


def extract_keywords_and_pos(text: str, pos_tagger: StanfordPOSTagger, rake: Rake) -> dict[str, str]:
    """Extract keywords and their POS tags from the prompt."""
    pos_dict = {}
    try:
        tagged_prompt = pos_tagger.tag(text.split())
    except Exception as e:
        raise RuntimeError(f"Error processing prompt '{text}': {e}") from e

    pos_dict = dict(tagged_prompt)

    keywords_dict = {}
    keywords = rake.run(text)
    for pair in keywords:
        words = pair[0].split()
        for word in words:
            if word in pos_dict:
                keywords_dict[word] = pos_dict[word]

    return keywords_dict


def get_synonyms(word: str, part_of_speech: str) -> list[str]:
    """Get synonyms for a given word and part of speech using wordnet."""
    if not nltk_available:
        return []

    synsets = wordnet.synsets(word, part_of_speech)

    return list({lemma.name().lower() for syn in synsets for lemma in syn.lemmas() if lemma.name().lower() != word})


def select_and_replace_keywords(
    keywords_lst: list[str],
    keywords_dict: dict[str, str],
    chosen_nums: list[int],
) -> tuple[list[str], list[str]]:
    """Select and replace keywords with synonyms."""
    counter = 1
    chosen_keywords_lst = []
    chosen_replacements_lst = []
    for keyword in keywords_lst:
        if counter <= max(chosen_nums):
            part_of_speech = keywords_dict[keyword][0].lower()
            if part_of_speech == "j":  # Adjust part_of_speech tag if necessary
                part_of_speech = "a"  # Example: 'j' for adjective to 'a'
            candidates = get_synonyms(keyword, part_of_speech)
            if candidates:
                counter += 1
                chosen_keywords_lst.append(keyword)
                chosen_replacement = random.choice(candidates)
                chosen_replacements_lst.append(chosen_replacement)
        else:
            break
    return chosen_keywords_lst, chosen_replacements_lst


def augment_text_with_synonyms(
    text: str,
    nums_lst: list[int],
    pos_tagger: StanfordPOSTagger,
    rake: Rake,
) -> str:
    """Generate a new text by replacing chosen keywords with synonyms."""
    synonyms_text_str = ""
    keywords_dict = extract_keywords_and_pos(text, pos_tagger, rake)
    if not keywords_dict:
        return ""
    keywords_lst = list(keywords_dict.keys())
    chosen_keywords, chosen_synonyms = select_and_replace_keywords(
        keywords_lst,
        keywords_dict,
        nums_lst,
    )
    for chosen_word, chosen_synonym in zip(chosen_keywords, chosen_synonyms):
        text = re.sub(rf"\b{chosen_word}\b", chosen_synonym, text)
        if chosen_keywords.index(chosen_word) + 1 in nums_lst:
            synonyms_text_str += re.sub("_", " ", text) + " "
    return synonyms_text_str.strip()


def convert_image_to_pil(image: np.ndarray) -> Image:
    """Convert a NumPy array image to a PIL image."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is not installed") from ImportError

    if len(image.shape) == MONO_CHANNEL_DIMENSIONS:  # (height, width)
        return Image.fromarray(image)
    if len(image.shape) == NUM_MULTI_CHANNEL_DIMENSIONS and image.shape[2] == 1:  # (height, width, 1)
        return Image.fromarray(image[:, :, 0], mode="L")
    if len(image.shape) == NUM_MULTI_CHANNEL_DIMENSIONS and image.shape[2] == NUM_RGB_CHANNELS:  # (height, width, 3)
        return Image.fromarray(image)

    raise TypeError(f"Unsupported image shape: {image.shape}")


def draw_text_on_pil_image(pil_image: Image, metadata_list: list[dict[str, Any]]) -> Image:
    """Draw text on a PIL image using metadata information."""
    try:
        from PIL import ImageDraw
    except ImportError:
        raise ImportError("Pillow is not installed") from ImportError

    draw = ImageDraw.Draw(pil_image)
    for metadata in metadata_list:
        bbox_coords = metadata["bbox_coords"]
        text = metadata["text"]
        font = metadata["font"]
        font_color = metadata["font_color"]
        position = bbox_coords[:2]
        draw.text(position, text, font=font, fill=font_color)
    return pil_image


def draw_text_on_multi_channel_image(image: np.ndarray, metadata_list: list[dict[str, Any]]) -> np.ndarray:
    """Draw text on a multi-channel image with more than three channels."""
    try:
        from PIL import ImageDraw, Image
    except ImportError:
        raise ImportError("Pillow is not installed") from ImportError

    channels = [Image.fromarray(image[:, :, i]) for i in range(image.shape[2])]
    pil_images = [ImageDraw.Draw(channel) for channel in channels]

    for metadata in metadata_list:
        bbox_coords = metadata["bbox_coords"]
        text = metadata["text"]
        font = metadata["font"]
        font_color = metadata["font_color"]
        position = bbox_coords[:2]

        for channel_id, pil_image in enumerate(pil_images):
            pil_image.text(position, text, font=font, fill=font_color[channel_id])

    return np.stack([np.array(channel) for channel in channels], axis=2)


@preserve_channel_dim
def render_text(image: np.ndarray, metadata_list: list[dict[str, Any]], clear_bg: bool) -> np.ndarray:
    original_dtype = image.dtype

    if original_dtype == np.float32:
        image = fmain.from_float(image, dtype=np.uint8)

    # First clean background under boxes using seamless clone if clear_bg is True
    if clear_bg:
        image = seamless_clone_text_background(image, metadata_list)

    if len(image.shape) == MONO_CHANNEL_DIMENSIONS or (
        len(image.shape) == NUM_MULTI_CHANNEL_DIMENSIONS and image.shape[2] in {1, NUM_RGB_CHANNELS}
    ):
        pil_image = convert_image_to_pil(image)
        pil_image = draw_text_on_pil_image(pil_image, metadata_list)
        result = np.array(pil_image)
    else:
        result = draw_text_on_multi_channel_image(image, metadata_list)

    return fmain.to_float(result) if original_dtype == np.float32 else result


def seamless_clone_text_background(image: np.ndarray, metadata_list: list[dict[str, Any]]) -> np.ndarray:
    result_image = image.copy()

    for metadata in metadata_list:
        x_min, y_min, x_max, y_max = metadata["bbox_coords"]

        # Create a mask for the text area
        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        mask.fill(255)

        # Coordinates for seamless cloning
        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

        # Extract the region from the original image
        region_to_clone = result_image[y_min:y_max, x_min:x_max]

        # Perform seamless cloning
        cloned_region = cv2.seamlessClone(region_to_clone, result_image, mask, center, cv2.NORMAL_CLONE)

        # Place the cloned region back into the result image
        result_image[y_min:y_max, x_min:x_max] = cloned_region

    return result_image
