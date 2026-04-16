"""Execution-related feature extraction."""

from __future__ import annotations

from config.scoring_config import PURPOSE_KEYWORDS
from utils.text_utils import count_keyword_hits, unique_keywords_found


def extract_execution_features(record):
    """Extract execution-related features from text."""
    combined = " ".join(
        [
            record.get("title", ""),
            record.get("meta_description", ""),
            record.get("headings", ""),
            record.get("text", ""),
        ]
    )
    industries = unique_keywords_found(combined, PURPOSE_KEYWORDS["industries"])
    return {
        "execution_hits": count_keyword_hits(combined, PURPOSE_KEYWORDS["execution"]),
        "measurement_hits_execution": count_keyword_hits(combined, PURPOSE_KEYWORDS["measurement"]),
        "capability_hits": count_keyword_hits(combined, PURPOSE_KEYWORDS["capability"]),
        "industry_hits": len(industries),
        "industries_found": industries,
    }

