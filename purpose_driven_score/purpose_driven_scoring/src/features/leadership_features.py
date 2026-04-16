"""Leadership-related feature extraction."""

from __future__ import annotations

from config.scoring_config import PURPOSE_KEYWORDS
from utils.text_utils import contains_any, count_keyword_hits, is_corporate_purpose_sentence, split_sentences


def extract_leadership_features(record):
    """Extract leadership-related features from text."""
    combined = " ".join([record.get("title", ""), record.get("headings", ""), record.get("text", "")])
    sentences = split_sentences(combined)
    co_mentions = 0
    for sentence in sentences:
        if contains_any(sentence, PURPOSE_KEYWORDS["leadership"]) and (
            is_corporate_purpose_sentence(sentence)
            or contains_any(sentence, PURPOSE_KEYWORDS["stakeholders"] + PURPOSE_KEYWORDS["impact"] + PURPOSE_KEYWORDS["strategy"])
        ):
            co_mentions += 1
    return {
        "leadership_hits": count_keyword_hits(combined, PURPOSE_KEYWORDS["leadership"]),
        "leadership_purpose_co_mentions": co_mentions,
    }
