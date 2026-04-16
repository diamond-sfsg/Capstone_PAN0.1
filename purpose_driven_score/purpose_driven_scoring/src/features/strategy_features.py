"""Strategy-related feature extraction."""

from __future__ import annotations

from config.scoring_config import PURPOSE_KEYWORDS
from utils.text_utils import contains_any, count_keyword_hits, is_corporate_purpose_sentence, split_sentences


def extract_strategy_features(record):
    """Extract strategy-related features from text."""
    combined = " ".join(
        [
            record.get("title", ""),
            record.get("meta_description", ""),
            record.get("headings", ""),
            record.get("text", ""),
        ]
    )
    sentences = split_sentences(combined)
    purpose_strategy_sentences = 0
    purpose_capital_sentences = 0
    decision_sentences = 0
    for sentence in sentences:
        if is_corporate_purpose_sentence(sentence) or (
            contains_any(sentence, PURPOSE_KEYWORDS["stakeholders"]) and contains_any(sentence, PURPOSE_KEYWORDS["impact"])
        ):
            if contains_any(sentence, PURPOSE_KEYWORDS["strategy"]):
                purpose_strategy_sentences += 1
            if contains_any(sentence, PURPOSE_KEYWORDS["capital"]):
                purpose_capital_sentences += 1
            if contains_any(sentence, PURPOSE_KEYWORDS["execution"]):
                decision_sentences += 1
    return {
        "strategy_hits": count_keyword_hits(combined, PURPOSE_KEYWORDS["strategy"]),
        "capital_hits": count_keyword_hits(combined, PURPOSE_KEYWORDS["capital"]),
        "purpose_strategy_sentences": purpose_strategy_sentences,
        "purpose_capital_sentences": purpose_capital_sentences,
        "purpose_decision_sentences": decision_sentences,
    }
