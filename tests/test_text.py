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
    "prompt, pos_tags, rake_keywords, expected_output",
    [
        (
            "The quick brown fox",
            [("The", "DT"), ("quick", "JJ"), ("brown", "JJ"), ("fox", "NN")],
            [("quick brown", 1.0), ("fox", 0.8)],
            {"quick": "JJ", "brown": "JJ", "fox": "NN"}
        ),
        (
            "Hello world",
            [("Hello", "UH"), ("world", "NN")],
            [("Hello world", 1.0)],
            {"Hello": "UH", "world": "NN"}
        ),
        (
            "Python is great for machine learning",
            [("Python", "NNP"), ("is", "VBZ"), ("great", "JJ"), ("for", "IN"), ("machine", "NN"), ("learning", "NN")],
            [("Python", 1.0), ("machine learning", 0.9)],
            {"Python": "NNP", "machine": "NN", "learning": "NN"}
        ),
        (
            "Data science is fun",
            [("Data", "NN"), ("science", "NN"), ("is", "VBZ"), ("fun", "JJ")],
            [("Data science", 1.0), ("fun", 0.8)],
            {"Data": "NN", "science": "NN", "fun": "JJ"}
        ),
        (
            "Natural language processing",
            [("Natural", "JJ"), ("language", "NN"), ("processing", "NN")],
            [("Natural language processing", 1.0)],
            {"Natural": "JJ", "language": "NN", "processing": "NN"}
        ),
    ]
)
def test_extract_keywords_and_pos(mocker, prompt, pos_tags, rake_keywords, expected_output):
    # Mock the StanfordPOSTagger
    pos_tagger = mocker.Mock()
    pos_tagger.tag.return_value = pos_tags

    # Mock the Rake
    rake = mocker.Mock()
    rake.run.return_value = rake_keywords

    result = ftext.extract_keywords_and_pos(prompt, pos_tagger, rake)
    assert result == expected_output, f"For prompt '{prompt}', expected {expected_output} but got {result}"


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


@pytest.fixture
def test_image():
    # Create a simple image for testing
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
    cv2.rectangle(image, (30, 30), (70, 70), (0, 0, 255), -1)  # Red square in the center
    return image

@pytest.mark.parametrize("metadata_list, expected_unchanged_regions", [
    ([{"bbox_coords": (30, 30, 70, 70)}], [(slice(None, 30), slice(None)), (slice(70, None), slice(None)), (slice(None), slice(None, 30)), (slice(None), slice(70, None))]),
    ([], [slice(None)]),
    ([{"bbox_coords": (30, 30, 70, 70)}, {"bbox_coords": (10, 10, 20, 20)}], [(slice(None, 10), slice(None)), (slice(20, 30), slice(None)), (slice(70, None), slice(None)), (slice(None), slice(None, 10)), (slice(None), slice(20, 30)), (slice(None), slice(70, None))])
])
def test_seamless_clone(test_image, metadata_list, expected_unchanged_regions):
    original_image = test_image.copy()

    # Perform seamless cloning
    result_image = ftext.seamless_clone_text_background(test_image, metadata_list)

    # Check that the regions specified in metadata_list have been cloned seamlessly
    for metadata in metadata_list:
        x_min, y_min, x_max, y_max = metadata["bbox_coords"]
        region_before = original_image[y_min:y_max, x_min:x_max]
        region_after = result_image[y_min:y_max, x_min:x_max]
        assert not np.array_equal(region_before, region_after), "Region should have been cloned but it is not."

    # Check that the rest of the image remains unchanged
    for unchanged_region in expected_unchanged_regions:
        assert np.array_equal(original_image[unchanged_region], result_image[unchanged_region]), "Region should remain unchanged but it is not."
