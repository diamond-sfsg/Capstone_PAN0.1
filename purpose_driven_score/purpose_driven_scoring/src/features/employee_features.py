"""Employee-related feature extraction."""

from __future__ import annotations

from config.scoring_config import PURPOSE_KEYWORDS
from utils.text_utils import contains_any, count_keyword_hits, is_corporate_purpose_sentence, split_sentences


def extract_employee_features(record):
    """Extract employee-related features from text."""
    combined = " ".join([record.get("title", ""), record.get("headings", ""), record.get("text", "")])
    sentences = split_sentences(combined)
    embedded_sentences = 0
    for sentence in sentences:
        if contains_any(sentence, PURPOSE_KEYWORDS["employees"]) and (
            is_corporate_purpose_sentence(sentence)
            or contains_any(sentence, PURPOSE_KEYWORDS["stakeholders"] + PURPOSE_KEYWORDS["impact"] + PURPOSE_KEYWORDS["strategy"])
        ):
            embedded_sentences += 1
    return {
        "employee_hits": count_keyword_hits(combined, PURPOSE_KEYWORDS["employees"]),
        "employee_embedding_sentences": embedded_sentences,
    }
