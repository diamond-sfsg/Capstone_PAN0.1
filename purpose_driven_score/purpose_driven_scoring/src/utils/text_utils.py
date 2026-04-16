"""Shared text utility functions."""

from __future__ import annotations

import re
from typing import Iterable

from config.scoring_config import CORPORATE_PURPOSE_POSITIVE_PATTERNS, FALSE_PURPOSE_PATTERNS, PURPOSE_KEYWORDS


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace in a string."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace for matching."""
    return normalize_whitespace(text).lower()


def split_sentences(text: str) -> list[str]:
    """Split text into rough sentence units."""
    text = normalize_whitespace(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    """Count keyword occurrences in text."""
    haystack = normalize_text(text)
    total = 0
    for keyword in keywords:
        total += haystack.count(keyword.lower())
    return total


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    """Return True when any keyword exists in text."""
    haystack = normalize_text(text)
    return any(keyword.lower() in haystack for keyword in keywords)


def unique_keywords_found(text: str, keywords: Iterable[str]) -> list[str]:
    """Return unique keywords found in text."""
    haystack = normalize_text(text)
    found = []
    for keyword in keywords:
        lowered = keyword.lower()
        if lowered in haystack and lowered not in found:
            found.append(lowered)
    return found


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return 0.0 when denominator is zero."""
    if not denominator:
        return 0.0
    return numerator / denominator


def keyword_density_score(text: str, keywords: Iterable[str]) -> int:
    """Score a text chunk by unique keyword matches and raw hits."""
    normalized = normalize_text(text)
    raw_hits = 0
    unique_hits = 0
    for keyword in keywords:
        lowered = keyword.lower()
        count = normalized.count(lowered)
        raw_hits += count
        if count:
            unique_hits += 1
    return raw_hits + (unique_hits * 2)


def is_false_purpose_context(text: str) -> bool:
    """Detect product-usage or technical contexts that misuse the word purpose."""
    lowered = normalize_text(text)
    return any(pattern in lowered for pattern in FALSE_PURPOSE_PATTERNS)


def is_corporate_purpose_sentence(text: str) -> bool:
    """Detect sentences that likely express company-level purpose rather than product usage."""
    lowered = normalize_text(text)
    has_positive_pattern = any(pattern in lowered for pattern in CORPORATE_PURPOSE_POSITIVE_PATTERNS)
    has_stakeholder = contains_any(lowered, PURPOSE_KEYWORDS["stakeholders"])
    has_impact = contains_any(lowered, PURPOSE_KEYWORDS["impact"])
    if is_false_purpose_context(lowered) and not (has_stakeholder and has_impact):
        return False
    if has_positive_pattern:
        return True
    if "purpose" in lowered and has_stakeholder and has_impact:
        return True
    if "mission" in lowered and has_stakeholder and has_impact:
        return True
    return False
