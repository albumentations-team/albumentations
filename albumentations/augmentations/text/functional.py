from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from albucore import (
    MONO_CHANNEL_DIMENSIONS,
    NUM_MULTI_CHANNEL_DIMENSIONS,
    NUM_RGB_CHANNELS,
    preserve_channel_dim,
    uint8_io,
)

from albumentations.core.types import PAIR

# Importing wordnet and other dependencies only for type checking
if TYPE_CHECKING:
    from PIL import Image


def delete_random_words(words: list[str], num_words: int, py_random: random.Random) -> str:
    if num_words >= len(words):
        return ""

    indices_to_delete = py_random.sample(range(len(words)), num_words)
    new_words = [word for idx, word in enumerate(words) if idx not in indices_to_delete]
    return " ".join(new_words)


def swap_random_words(words: list[str], num_words: int, py_random: random.Random) -> str:
    if num_words == 0 or len(words) < PAIR:
        return " ".join(words)

    words = words.copy()

    for _ in range(num_words):
        idx1, idx2 = py_random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)


def insert_random_stopwords(
    words: list[str],
    num_insertions: int,
    stopwords: tuple[str, ...] | None,
    py_random: random.Random,
) -> str:
    if stopwords is None:
        stopwords = ("and", "the", "is", "in", "at", "of")  # Default stopwords if none provided

    for _ in range(num_insertions):
        idx = py_random.randint(0, len(words))
        words.insert(idx, py_random.choice(stopwords))
    return " ".join(words)


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
        if isinstance(font_color, (list, tuple)):
            font_color = tuple(int(c) for c in font_color)
        elif isinstance(font_color, float):
            font_color = int(font_color)
        position = bbox_coords[:2]
        draw.text(position, text, font=font, fill=font_color)
    return pil_image


def draw_text_on_multi_channel_image(image: np.ndarray, metadata_list: list[dict[str, Any]]) -> np.ndarray:
    """Draw text on a multi-channel image with more than three channels."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("Pillow is not installed") from ImportError

    channels = [Image.fromarray(image[:, :, i]) for i in range(image.shape[2])]
    pil_images = [ImageDraw.Draw(channel) for channel in channels]

    for metadata in metadata_list:
        bbox_coords = metadata["bbox_coords"]
        text = metadata["text"]
        font = metadata["font"]
        font_color = metadata["font_color"]

        # Handle different font_color types
        if isinstance(font_color, str):
            # If it's a string, use it as is for all channels
            font_color = [font_color] * image.shape[2]
        elif isinstance(font_color, (int, float)):
            # If it's a single number, convert to int and use for all channels
            font_color = [int(font_color)] * image.shape[2]
        elif isinstance(font_color, Sequence):
            # If it's a sequence, ensure it has the right length and convert to int
            if len(font_color) != image.shape[2]:
                raise ValueError(
                    f"font_color sequence length ({len(font_color)}) "
                    f"must match the number of image channels ({image.shape[2]})",
                )
            font_color = [int(c) for c in font_color]
        else:
            raise TypeError(f"Unsupported font_color type: {type(font_color)}")

        position = bbox_coords[:2]

        for channel_id, pil_image in enumerate(pil_images):
            pil_image.text(position, text, font=font, fill=font_color[channel_id])

    return np.stack([np.array(channel) for channel in channels], axis=2)


@uint8_io
@preserve_channel_dim
def render_text(image: np.ndarray, metadata_list: list[dict[str, Any]], clear_bg: bool) -> np.ndarray:
    # First clean background under boxes using seamless clone if clear_bg is True
    if clear_bg:
        image = inpaint_text_background(image, metadata_list)

    if len(image.shape) == MONO_CHANNEL_DIMENSIONS or (
        len(image.shape) == NUM_MULTI_CHANNEL_DIMENSIONS and image.shape[2] in {1, NUM_RGB_CHANNELS}
    ):
        pil_image = convert_image_to_pil(image)
        pil_image = draw_text_on_pil_image(pil_image, metadata_list)
        return np.array(pil_image)

    return draw_text_on_multi_channel_image(image, metadata_list)


def inpaint_text_background(
    image: np.ndarray,
    metadata_list: list[dict[str, Any]],
    method: int = cv2.INPAINT_TELEA,
) -> np.ndarray:
    result_image = image.copy()
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for metadata in metadata_list:
        x_min, y_min, x_max, y_max = metadata["bbox_coords"]

        # Black out the region
        result_image[y_min:y_max, x_min:x_max] = 0

        # Update the mask to indicate the region to inpaint
        mask[y_min:y_max, x_min:x_max] = 255

    # Inpaint the blacked-out regions
    return cv2.inpaint(result_image, mask, inpaintRadius=3, flags=method)
