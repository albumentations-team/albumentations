from __future__ import annotations
import random
from typing import Callable, TYPE_CHECKING

import re

from albumentations.core.types import PAIR

# Importing wordnet and other dependencies only for type checking
if TYPE_CHECKING:
    from nltk.corpus.reader.wordnet import Synset
    from nltk.tag import StanfordPOSTagger
    from rake_nltk import Rake


def delete_random_words(sentence: str, p: float = 0.2) -> str:
    words = sentence.split()
    if len(words) <= 1:
        return sentence
    new_words = [word for word in words if random.random() > p]
    return " ".join(new_words)


def swap_random_words(sentence: str, n: int = 1) -> str:
    words = sentence.split()
    for _ in range(n):
        if len(words) < PAIR:
            break
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)


def insert_random_stopwords(sentence: str, num_insertions: int = 1, stopwords: list[str] | None = None) -> str:
    if stopwords is None:
        stopwords = ["and", "the", "is", "in", "at", "of"]  # Default stopwords if none provided

    words = sentence.split()
    for _ in range(num_insertions):
        idx = random.randint(0, len(words))
        words.insert(idx, random.choice(stopwords))
    return " ".join(words)


def extract_keywords_and_pos(prompt: str, pos_tagger: StanfordPOSTagger, rake: Rake) -> dict[str, str]:
    """Extract keywords and their POS tags from the prompt."""
    pos_dict = {}
    try:
        tagged_prompt = pos_tagger.tag(prompt.split())
    except Exception as e:
        raise RuntimeError(f"Error processing prompt '{prompt}': {e}") from e

    pos_dict = dict(tagged_prompt)

    keywords_dict = {}
    keywords = rake.run(prompt)
    for pair in keywords:
        words = pair[0].split()
        for word in words:
            if word in pos_dict:
                keywords_dict[word] = pos_dict[word]

    return keywords_dict


def get_synonyms(word: str, part_of_speech: str, synsets_fn: Callable[..., list[Synset]]) -> list[str]:
    """Get synonyms for a given word and part of speech using the provided synsets function."""
    try:
        # Try fetching synsets with part_of_speech if available
        synsets = synsets_fn(word, part_of_speech)
    except TypeError:
        # If synsets_fn does not accept part_of_speech, call without it
        synsets = []

    # If no synsets found with part_of_speech or TypeError was raised, try without part_of_speech
    if not synsets:
        synsets = synsets_fn(word)

    return list({lemma.name().lower() for syn in synsets for lemma in syn.lemmas() if lemma.name().lower() != word})


def select_and_replace_keywords(
    keywords_lst: list[str],
    keywords_dict: dict[str, str],
    get_synonyms_fn: Callable[[str, str, Callable[[str, str], list[Synset]]], list[str]],
    chosen_nums: list[int],
    synsets_fn: Callable[[str, str], list[Synset]],
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
            candidates = get_synonyms_fn(keyword, part_of_speech, synsets_fn)
            if candidates:
                counter += 1
                chosen_keywords_lst.append(keyword)
                chosen_replacement = random.choice(candidates)
                chosen_replacements_lst.append(chosen_replacement)
        else:
            break
    return chosen_keywords_lst, chosen_replacements_lst


def augment_sentence_with_synonyms(
    prompt: str,
    nums_lst: list[int],
    pos_tagger: StanfordPOSTagger,
    rake: Rake,
    synsets_fn: Callable[[str, str], list[Synset]],
) -> str:
    """Generate a new sentence by replacing chosen keywords with synonyms."""
    synonyms_prompt_str = ""
    keywords_dict = extract_keywords_and_pos(prompt, pos_tagger, rake)
    if not keywords_dict:
        return ""
    keywords_lst = list(keywords_dict.keys())
    chosen_keywords, chosen_synonyms = select_and_replace_keywords(
        keywords_lst,
        keywords_dict,
        get_synonyms,
        nums_lst,
        synsets_fn,
    )
    for chosen_word, chosen_synonym in zip(chosen_keywords, chosen_synonyms):
        prompt = re.sub(rf"\b{chosen_word}\b", chosen_synonym, prompt)
        if chosen_keywords.index(chosen_word) + 1 in nums_lst:
            synonyms_prompt_str += re.sub("_", " ", prompt) + " "
    return synonyms_prompt_str.strip()
