from __future__ import annotations
import random
from typing import TYPE_CHECKING

import re

import numpy as np

from albumentations.core.types import NUM_RGB_CHANNELS, PAIR, ColorType

# Importing wordnet and other dependencies only for type checking
if TYPE_CHECKING:
    from nltk.tag import StanfordPOSTagger
    from rake_nltk import Rake
    from PIL import ImageFont


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


def render_text(
    bbox_shape: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    font_color: ColorType | str,
    num_channels: int,
) -> np.ndarray:
    # Check if Pillow is available
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError(
            "Pillow is not installed. Please install it to use the render_text function.",
        ) from ImportError

    bbox_height, bbox_width = bbox_shape

    background_color: int | str

    # Determine the image mode based on the number of channels
    if num_channels == 1:
        mode = "L"  # Grayscale image
        background_color = 255  # White background for grayscale
    elif num_channels == NUM_RGB_CHANNELS:
        mode = "RGB"  # RGB image
        background_color = "white"  # White background for RGB
    else:
        raise ValueError("num_channels must be either 1 (grayscale) or 3 (RGB).")

    # Create an empty image with the specified mode and background color
    bbox_img = Image.new(mode, (bbox_width, bbox_height), color=background_color)
    draw = ImageDraw.Draw(bbox_img)

    # Draw the text with the specified color
    draw.text((0, 0), text, fill=font_color, font=font)

    return np.array(bbox_img)
