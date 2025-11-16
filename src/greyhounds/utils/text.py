from __future__ import annotations

import re
import unicodedata

_COUNTRY_SUFFIX_RE = re.compile(r"\s*\(([A-Z]{2,3})\)\s*$")
_APOSTROPHES_RE = re.compile(r"[\u2019\u2018\']+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^0-9A-Za-z\s]+")
_WHITESPACE_RE = re.compile(r"\s+")
_PARENTHESIS_CONTENT_RE = re.compile(r"\s*\([^\)]*\)")


def normalize_spaces(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_country_suffix(text: str) -> str:
    return _COUNTRY_SUFFIX_RE.sub("", text)


def remove_apostrophes(text: str) -> str:
    return _APOSTROPHES_RE.sub("", text)


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_horse_name(raw_name: str) -> str:
    name = strip_country_suffix(raw_name or "")
    name = normalize_spaces(name)
    name = remove_apostrophes(name)
    name = strip_accents(name)
    name = _NON_ALNUM_SPACE_RE.sub(" ", name)
    name = normalize_spaces(name)
    return name.title()


def normalize_track_name(raw_name: str) -> str:
    name = raw_name or ""
    name = _PARENTHESIS_CONTENT_RE.sub("", name)
    name = remove_apostrophes(name)
    name = strip_accents(name)
    name = re.sub(r"\bStadium\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\bGreyhound Stadium\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\bRacecourse\b", "", name, flags=re.IGNORECASE)
    name = _NON_ALNUM_SPACE_RE.sub(" ", name)
    name = normalize_spaces(name).title()
    return name