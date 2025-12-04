from __future__ import annotations

import re
import unicodedata

_COUNTRY_SUFFIX_RE = re.compile(r"\s*\(([A-Z]{2,3})\)\s*$")
_APOSTROPHES_RE = re.compile(r"[\u2019\u2018\']+")
_NON_ALNUM_SPACE_RE = re.compile(r"[^0-9A-Za-z\s]+")
_WHITESPACE_RE = re.compile(r"\s+")
_PARENTHESIS_CONTENT_RE = re.compile(r"\s*\([^\)]*\)")
_PROVIDER_PREFIX_RE = re.compile(
    r"^(?:"
    r"SIS(?:\s+TV)?"
    r"|TRP"
    r"|RPGTV"
    r"|SKY\s+SPORTS(?:\s+RACING)?"
    r"|SPORTY\s+STUFF"
    r"|PREM\.?\s*GH(?:\s*RACING)?"
    r"|PREMIER\s+GREYHOUNDS"
    r"|RACING\s+POST"
    r"|TIMEFORM\s+TV"
    r"|IGOBF"
    r"|ISGB"
    r"|BAGS"
    r"|VC"
    r"|RCE"
    r")\s*(?:-|/)?\s*",
    re.IGNORECASE,
)
_DATE_TOKEN_RE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\b", re.IGNORECASE)
_MONTH_TOKEN_RE = re.compile(
    r"\b(?:"
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    r")\b",
    re.IGNORECASE,
)
_SESSION_TOKEN_RE = re.compile(
    r"\b(?:Matinee|Morning|Early|Late|Afternoon|Evening|Midnight|Night|Eve)\b",
    re.IGNORECASE,
)
_DAY_TOKEN_RE = re.compile(
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)(?:day)?\b",
    re.IGNORECASE,
)
_COUNTRY_PREFIX_RE = re.compile(
    r"^(?:Aus|Australia|Ire|Ireland|Nz|New\s+Zealand|Uk|United\s+Kingdom)\b\s*",
    re.IGNORECASE,
)
_TRAILING_DOGS_RE = re.compile(r"\b(Dogs?|Dg)\b", re.IGNORECASE)
_EMBEDDED_DAY_SUFFIX_RE = re.compile(r"(\d{1,2})(st|nd|rd|th)", re.IGNORECASE)
_NUMERIC_CAMEL_RE = re.compile(r"(\D)(\d)")
_VALLEY_TYPO_RE = re.compile(r"\bValey\b", re.IGNORECASE)
_CANONICAL_OVERRIDES = {
    "Shelbourne": "Shelbourne Park",
    "Shelbourn": "Shelbourne Park",
}


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
    name = normalize_spaces(str(raw_name or ""))
    if not name:
        return ""

    name = name.replace("/", " ").replace("\\", " ").replace("-", " ")
    name = _EMBEDDED_DAY_SUFFIX_RE.sub(r"\1", name)
    name = _NUMERIC_CAMEL_RE.sub(r"\1 \2", name)
    name = _PROVIDER_PREFIX_RE.sub("", name)
    name = _PROVIDER_PREFIX_RE.sub("", name)  # segunda passada para casos encadeados
    name = _COUNTRY_PREFIX_RE.sub("", name)
    name = _PARENTHESIS_CONTENT_RE.sub("", name)
    name = _DATE_TOKEN_RE.sub(" ", name)
    name = _MONTH_TOKEN_RE.sub(" ", name)
    name = _DAY_TOKEN_RE.sub(" ", name)
    name = _SESSION_TOKEN_RE.sub(" ", name)
    name = re.sub(r"\b\d{4}\b", " ", name)
    name = _TRAILING_DOGS_RE.sub(" ", name)
    name = re.sub(r"^The\s+", "", name, flags=re.IGNORECASE)
    name = normalize_spaces(name)
    name = remove_apostrophes(name)
    name = strip_accents(name)
    name = re.sub(r"\bStadium\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\bGreyhound Stadium\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\bRacecourse\b", "", name, flags=re.IGNORECASE)
    name = _NON_ALNUM_SPACE_RE.sub(" ", name)
    name = normalize_spaces(name).title()
    name = _VALLEY_TYPO_RE.sub("Valley", name)
    name = _CANONICAL_OVERRIDES.get(name, name)
    if not name:
        return normalize_spaces(str(raw_name or "")).title()
    return name