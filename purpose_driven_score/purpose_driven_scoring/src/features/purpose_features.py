"""Purpose-related feature extraction."""

from __future__ import annotations

from config.scoring_config import PURPOSE_KEYWORDS
from utils.date_utils import extract_years
from utils.text_utils import (
    count_keyword_hits,
    is_corporate_purpose_sentence,
    is_false_purpose_context,
    split_sentences,
    unique_keywords_found,
)


def extract_purpose_features(record):
    """Extract purpose-related features from text."""
    combined = " ".join(
        [
            record.get("title", ""),
            record.get("meta_description", ""),
            record.get("headings", ""),
            record.get("text", ""),
        ]
    )
    sentences = split_sentences(combined)
    purpose_sentences = [sentence for sentence in sentences if is_corporate_purpose_sentence(sentence)]
    false_purpose_sentences = [sentence for sentence in sentences if is_false_purpose_context(sentence)]
    stakeholder_sentences = [
        sentence for sentence in sentences if any(term in sentence.lower() for term in PURPOSE_KEYWORDS["stakeholders"])
    ]
    human_evidence_sentences = []
    for sentence in purpose_sentences + stakeholder_sentences:
        cleaned = sentence.strip()
        lowered = cleaned.lower()
        if len(cleaned.split()) < 8:
            continue
        if is_false_purpose_context(lowered) and cleaned not in purpose_sentences:
            continue
        if cleaned not in human_evidence_sentences:
            human_evidence_sentences.append(cleaned)
    return {
        "purpose_hits": count_keyword_hits(" ".join(purpose_sentences), PURPOSE_KEYWORDS["purpose_core"]),
        "stakeholder_hits": count_keyword_hits(combined, PURPOSE_KEYWORDS["stakeholders"]),
        "impact_hits": count_keyword_hits(combined, PURPOSE_KEYWORDS["impact"]),
        "branding_hits": count_keyword_hits(combined, PURPOSE_KEYWORDS["branding"]),
        "measurement_hits": count_keyword_hits(combined, PURPOSE_KEYWORDS["measurement"]),
        "purpose_sentence_count": len(purpose_sentences),
        "purpose_sentences": purpose_sentences[:5],
        "stakeholder_sentences": stakeholder_sentences[:5],
        "human_evidence_sentences": human_evidence_sentences[:8],
        "purpose_terms_found": unique_keywords_found(" ".join(purpose_sentences), PURPOSE_KEYWORDS["purpose_core"]),
        "false_purpose_sentence_count": len(false_purpose_sentences),
        "stakeholder_sentence_count": len(stakeholder_sentences),
        "years_found": extract_years(combined),
    }
