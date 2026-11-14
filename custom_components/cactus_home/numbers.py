"""Turn spoken number words into digits.

Speech recognition returns numbers as words. Whistle writes "50%" as
"fünfzig Prozent", and Needle grounds a numeric argument only when the digits
appear in the request, so "fünfzig Prozent" reached the tool call with no
usable quantity. Converting the words to digits before the agent sees the text
fixes that without touching the schemas.

Supported: German and English, 0 to 999, including German compounds
("einundzwanzig", "einhundertfünfzig"). Other languages pass through unchanged
rather than being guessed at.

Safety: a bare article such as "ein" or "one" is left alone, because "ein Licht"
means "a light", not "one light". A word is only converted on its own when it is
unambiguous ("eins", "fünfzig", "eighty"), when it forms a compound, or when a
quantity word follows it ("Prozent", "Grad", "degrees").

One asymmetry is deliberate. German separates the article "ein" from the numeral
"eins", so "eins" converts on its own. English uses "one" for both the numeral
and the pronoun, so a bare "one" stays a word and only converts with a unit word
("one percent") or another number beside it ("twenty one"). Treating it as
unambiguous would turn "one of the lights" into "1 of the lights".
"""

from __future__ import annotations

import re
from typing import Final, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------
# German
# --------------------------------------------------------------------------
_DE_ZERO: Final = {"null": 0}

_DE_UNITS: Final = {
    "eins": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "fünf": 5,
    "fuenf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
}

# Articles and case forms of "one". Converted only with context.
_DE_ONE: Final = {"ein", "eine", "einer", "einem", "einen"}

_DE_TEENS: Final = {
    "zehn": 10,
    "elf": 11,
    "zwölf": 12,
    "zwoelf": 12,
    "dreizehn": 13,
    "vierzehn": 14,
    "fünfzehn": 15,
    "fuenfzehn": 15,
    "sechzehn": 16,
    "siebzehn": 17,
    "achtzehn": 18,
    "neunzehn": 19,
}

_DE_TENS: Final = {
    "zwanzig": 20,
    "dreißig": 30,
    "dreissig": 30,
    "vierzig": 40,
    "fünfzig": 50,
    "fuenfzig": 50,
    "sechzig": 60,
    "siebzig": 70,
    "achtzig": 80,
    "neunzig": 90,
}

_DE_SIMPLE: Final = {**_DE_ZERO, **_DE_UNITS, **_DE_TEENS, **_DE_TENS}

# --------------------------------------------------------------------------
# English
# --------------------------------------------------------------------------
_EN_UNITS: Final = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

_EN_TEENS: Final = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_EN_TENS: Final = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_EN_SIMPLE: Final = {**_EN_UNITS, **_EN_TEENS, **_EN_TENS}

_EN_HUNDRED: Final = {"hundred": 100, "hundered": 100}

# Words that alone are not a quantity ("a light", "one of them").
_WEAK: Final[Mapping[str, frozenset[str]]] = {
    "de": frozenset(_DE_ONE),
    "en": frozenset({"one", "a", "an"}),
}

_WEAK_ALL: Final = _WEAK["de"] | _WEAK["en"]

# A following quantity word makes even a weak number word a real quantity.
_QUANTITY_WORDS: Final = frozenset(
    {"prozent", "percent", "grad", "degree", "degrees", "celsius", "%"}
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

_LANGUAGE_TAGS: Final[Mapping[str, str]] = {
    "de": "de",
    "de-de": "de",
    "de-at": "de",
    "de-ch": "de",
    "en": "en",
    "en-gb": "en",
    "en-us": "en",
}


def _language_key(language: str | None) -> str | None:
    if not language:
        return None
    tag = language.replace("_", "-").casefold()
    if tag in _LANGUAGE_TAGS:
        return _LANGUAGE_TAGS[tag]
    return _LANGUAGE_TAGS.get(tag.split("-", 1)[0])


def _parse_de(word: str) -> int | None:
    """Parse one German number word, including compounds, into 0..999."""
    if word in _DE_SIMPLE:
        return _DE_SIMPLE[word]
    if word in _DE_ONE:
        return 1

    if "hundert" in word:
        head, _, tail = word.partition("hundert")
        if head in ("", "ein", "eine", "eins"):
            head_value = 1
        elif head in _DE_UNITS:
            head_value = _DE_UNITS[head]
        else:
            return None
        if not tail:
            return head_value * 100
        tail_value = _parse_de(tail)
        if tail_value is None:
            return None
        return head_value * 100 + tail_value

    if "und" in word:
        unit, _, tens = word.partition("und")
        if unit in _DE_ONE or unit == "eins":
            unit_value = 1
        elif unit in _DE_UNITS:
            unit_value = _DE_UNITS[unit]
        else:
            return None
        if tens not in _DE_TENS:
            return None
        return _DE_TENS[tens] + unit_value

    return None


def _parse_en_span(words: Sequence[str], start: int) -> tuple[int, int] | None:
    """Parse an English number across consecutive words, e.g. "twenty five"."""
    current = 0
    consumed = 0

    for word in words[start:]:
        if word in _EN_SIMPLE:
            current += _EN_SIMPLE[word]
        elif word in _EN_HUNDRED:
            current = max(1, current) * 100
        else:
            break
        consumed += 1
        if current > 999:
            return None

    if consumed == 0:
        return None
    return current, consumed


def _next_word(words: Sequence[str], index: int) -> str | None:
    if index < len(words):
        return words[index]
    return None


def _convert(text: str, languages: Iterable[str]) -> str:
    matches = list(_WORD.finditer(text))
    if not matches:
        return text

    lowered = [match.group(0).casefold() for match in matches]
    out: list[str] = []
    cursor = 0
    index = 0

    while index < len(matches):
        value: int | None = None
        consumed = 1

        for language in languages:
            if language == "de":
                value = _parse_de(lowered[index])
            elif language == "en":
                parsed = _parse_en_span(lowered, index)
                if parsed is not None:
                    value, consumed = parsed
            if value is not None:
                break

        if value is None:
            index += 1
            continue

        span = lowered[index : index + consumed]
        strong = any(word not in _WEAK_ALL for word in span)
        followed = _next_word(lowered, index + consumed)

        if not strong and consumed == 1 and followed not in _QUANTITY_WORDS:
            index += consumed
            continue

        start = matches[index].start()
        end = matches[index + consumed - 1].end()
        out.append(text[cursor:start])
        out.append(str(value))
        cursor = end
        index += consumed

    if not out:
        return text
    out.append(text[cursor:])
    return "".join(out)


def normalize_spoken_numbers(text: str, language: str | None = None) -> str:
    """Replace spoken number words with digits.

    Only German ("de") and English ("en") are converted. Any other language
    returns the text unchanged, rather than risking a wrong guess.
    """
    if not text:
        return text
    key = _language_key(language)
    if key is None:
        return text
    return _convert(text, (key,))


def fold_numbers(text: str) -> str:
    """Fold strong number words in both supported languages to digits.

    Used for name comparison, where the language is unknown and where the two
    sides of a comparison must end up identical: an entity alias spelled
    "Licht drei" then matches an entity named "Licht 3".
    """
    if not text:
        return text
    return _convert(text, ("de", "en"))
