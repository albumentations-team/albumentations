import pytest
import albumentations.augmentations.text.functional as ftext
from tests.utils import set_seed
from albucore.utils import get_num_channels
from PIL import Image, ImageFont
import numpy as np
import cv2


@pytest.mark.parametrize(
    "sentence, num_words, expected_length",
    [
        ("The quick brown fox jumps over the lazy dog", 0, 9),  # num_words=0 should delete 0 words
        ("The quick brown fox jumps over the lazy dog", 9, 0),  # num_words=9 should delete all words
        ("The quick brown fox jumps over the lazy dog", 4, 5),  # num_words=4 should delete 4 words
        ("Hello world", 1, 1),  # num_words=1 should delete 1 word
        ("Hello", 1, 0),  # Single word sentence should be deleted
    ]
)
def test_delete_random_words(sentence, num_words, expected_length):
    words = sentence.split()
    result = ftext.delete_random_words(words, num_words)
    result_length = len(result.split())
    assert expected_length == result_length
    if num_words == 0:
        assert result == sentence  # No words should be deleted
    if num_words >= len(sentence.split()):
        assert result == ""  # All words should be deleted


@pytest.mark.parametrize(
    "sentence, num_words",
    [
        ("The quick brown fox jumps over the lazy dog", 0),  # No swaps
        ("The quick brown fox jumps over the lazy dog", 1),   # One swap
        ("The quick brown fox jumps over the lazy dog", 3),  # Three swaps
        ("Hello world", 1),  # Single swap for two words
        ("Hello", 1),  # Single word should remain unchanged
    ]
)
def test_swap_random_words(sentence, num_words):
    set_seed(42)  # Set seed for reproducibility

    words_in_sentence = sentence.split(" ")

    result = ftext.swap_random_words(words_in_sentence, num_words)
    words_in_result = result.split(" ")

    # Handle single word case
    if len(words_in_sentence) == 1:
        assert result == sentence, "Single word input should remain unchanged"
    else:
        assert words_in_result != words_in_sentence or num_words == 0, f"Result should be different from input for n={num_words}"
        assert len(words_in_result) == len(words_in_sentence), "Result should have the same number of words as input"
        assert sorted(words_in_result) == sorted(words_in_sentence), "Result should contain the same words as input"


@pytest.mark.parametrize(
    "sentence, num_insertions, stopwords, expected_length_range",
    [
        ("The quick brown fox jumps over the lazy dog", 0, None, (9, 9)),  # No insertions
        ("The quick brown fox jumps over the lazy dog", 1, None, (10, 10)),  # One insertion with default stopwords
        ("The quick brown fox jumps over the lazy dog", 3, None, (12, 12)),  # Three insertions with default stopwords
        ("The quick brown fox jumps over the lazy dog", 1, ["a", "b", "c"], (10, 10)),  # One insertion with custom stopwords
        ("Hello world", 1, None, (3, 3)),  # Single insertion for two words
        ("Hello", 1, None, (2, 2)),  # Single insertion for a single word
    ]
)
def test_insert_random_stopwords(sentence, num_insertions, stopwords, expected_length_range):
    set_seed(42)
    words = sentence.split()
    result = ftext.insert_random_stopwords(words, num_insertions, stopwords)
    result_length = len(result.split())

    # Ensure the result length is within the expected range
    assert expected_length_range[0] <= result_length <= expected_length_range[1], f"Result length {result_length} not in expected range {expected_length_range} for input '{sentence}' with num_insertions={num_insertions}"

    # Check if the number of words increased correctly
    assert result_length == len(sentence.split()) + num_insertions, "The number of words in the result should be the original number plus the number of insertions"

    # Ensure all inserted words are from the stopwords list
    if stopwords is None:
        stopwords = ["and", "the", "is", "in", "at", "of"]
    inserted_words = [word for word in result.split() if word not in sentence.split()]
    assert all(word in stopwords for word in inserted_words), "All inserted words should be from the stopwords list"


@pytest.mark.parametrize(
    "image_shape",
    [
        (100, 100),  # Grayscale image
        (100, 100, 1),  # Single channel image
        (100, 100, 3),  # RGB image
    ]
)
def test_convert_image_to_pil(image_shape):
    image = np.random.randint(0, 255, size=image_shape, dtype=np.uint8)
    pil_image = ftext.convert_image_to_pil(image)
    assert isinstance(pil_image, Image.Image)

font = ImageFont.truetype("./tests/files/LiberationSerif-Bold.ttf", size=12)

dummy_metadata = {
    "bbox_coords": (10, 10, 100, 50),
    "text": "Test",
    "font": font,
    "font_color": (255, 0, 0)  # Red color
}

@pytest.mark.parametrize(
    "image_shape, metadata_list",
    [((100, 100), [{
    "bbox_coords": (10, 10, 100, 50),
    "text": "Test",
    "font": font,
    "font_color": 127  # Red color
}]),  # Grayscale image
    ((100, 100, 1), [{
    "bbox_coords": (10, 10, 100, 50),
    "text": "Test",
    "font": font,
    "font_color": 127  # Red color
}]),  # Single channel image
    ((100, 100, 3), [{
    "bbox_coords": (10, 10, 100, 50),
    "text": "Test",
    "font": font,
    "font_color": (127, 127, 127)  # Red color
}, {
    "bbox_coords": (20, 20, 110, 60),
    "text": "Test",
    "font": font,
    "font_color": "red"  # Red color
}]),  # RGB image with tuple and string font color
    ((100, 100, 5), [{
    "bbox_coords": (20, 20, 110, 60),
    "text": "Test",
    "font": font,
    "font_color": (127, 127, 127, 127, 127)
}])])
def test_draw_text_on_pil_image(image_shape, metadata_list):
    image = np.random.randint(0, 255, size=image_shape, dtype=np.uint8)

    if get_num_channels(image) in {1, 3}:
        pil_image = ftext.convert_image_to_pil(image)
        result = ftext.draw_text_on_pil_image(pil_image, metadata_list)
        assert isinstance(result, Image.Image)
    else:
        result = ftext.draw_text_on_multi_channel_image(image, metadata_list)
        assert isinstance(result, np.ndarray)
