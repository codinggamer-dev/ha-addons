"""Tests for spoken number-word normalisation."""

from __future__ import annotations

import pytest

from custom_components.cactus_home.numbers import fold_numbers, normalize_spoken_numbers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("fünfzig Prozent", "50 Prozent"),
        ("mach das Licht auf fünfzig Prozent", "mach das Licht auf 50 Prozent"),
        ("dreißig", "30"),
        ("fünfzehn", "15"),
        ("eins", "1"),
        ("null", "0"),
        ("einundzwanzig", "21"),
        ("fünfundvierzig Prozent", "45 Prozent"),
        ("siebenundsiebzig", "77"),
        ("hundert", "100"),
        ("einhundert", "100"),
        ("zweihundert", "200"),
        ("einhundertfünfzig", "150"),
        ("zwölf Grad", "12 Grad"),
        ("fuenfzig prozent", "50 prozent"),
        ("dreissig", "30"),
    ],
)
def test_german_numbers(text, expected):
    assert normalize_spoken_numbers(text, "de") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("fifty percent", "50 percent"),
        ("twenty five percent", "25 percent"),
        ("one hundred", "100"),
        ("eighty", "80"),
        ("twelve degrees", "12 degrees"),
        ("zero", "0"),
    ],
)
def test_english_numbers(text, expected):
    assert normalize_spoken_numbers(text, "en") == expected


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("fünfzig", "de-DE"),
        ("fünfzig", "de_AT"),
        ("fifty", "en-US"),
        ("fifty", "en_GB"),
    ],
)
def test_regional_tags_are_understood(text, language):
    assert normalize_spoken_numbers(text, language) == "50"


def test_language_scoping_is_respected():
    """A German word is not converted when the language is English."""
    assert normalize_spoken_numbers("fünfzig", "en") == "fünfzig"
    assert normalize_spoken_numbers("fifty", "de") == "fifty"


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("turn on ein Licht", "de"),
        ("turn on one light", "en"),
        ("one of them", "en"),
    ],
)
def test_articles_are_left_alone(text, language):
    """A bare article is not a quantity: "ein Licht" means "a light"."""
    assert normalize_spoken_numbers(text, language) == text


def test_a_quantity_word_makes_even_a_weak_word_a_quantity():
    assert normalize_spoken_numbers("ein Prozent", "de") == "1 Prozent"
    assert normalize_spoken_numbers("one percent", "en") == "1 percent"


@pytest.mark.parametrize("language", ["pt", "fr", "zh", None, ""])
def test_unsupported_languages_are_untouched(language):
    """No guessing: an unsupported language passes through byte for byte."""
    assert normalize_spoken_numbers("fünfzig Prozent", language) == "fünfzig Prozent"


def test_fold_numbers_covers_both_languages():
    assert fold_numbers("Licht drei") == "Licht 3"
    assert fold_numbers("light three") == "light 3"
    assert fold_numbers("Licht 3") == "Licht 3"


def test_digits_and_punctuation_survive():
    assert normalize_spoken_numbers("50 Prozent", "de") == "50 Prozent"
    assert normalize_spoken_numbers("Licht 2 auf fünfzig", "de") == "Licht 2 auf 50"
    assert normalize_spoken_numbers("fünfzig Prozent!", "de") == "50 Prozent!"
    assert normalize_spoken_numbers("Zimmer zwei, bitte", "de") == "Zimmer 2, bitte"


def test_repeated_numbers_all_convert():
    assert normalize_spoken_numbers("von dreißig auf sechzig", "de") == "von 30 auf 60"


def test_empty_text_is_safe():
    assert normalize_spoken_numbers("", "de") == ""
    assert fold_numbers("") == ""
