"""Articulation scoring logic."""

from __future__ import annotations


def compute_articulation_score(summary):
    """Compute the articulation score from extracted features."""
    presence = _purpose_presence(summary)
    clarity = _clarity(summary)
    consistency = _historical_consistency(summary)
    leadership = _leadership_centrality(summary)
    branding = _distinction_from_branding(summary)
    dimensions = {
        "purpose_presence": presence,
        "clarity": clarity,
        "historical_consistency": consistency,
        "leadership_centrality": leadership,
        "distinction_from_branding": branding,
    }
    overall = round(sum(dimensions.values()) / len(dimensions), 2)
    return {"overall": overall, "dimensions": dimensions}


def _purpose_presence(summary):
    explicit_ratio = summary["explicit_purpose_page_ratio"]
    purpose_ratio = summary["purpose_page_ratio"]
    stakeholder_ratio = summary["stakeholder_sentence_page_ratio"]
    source_coverage = summary["source_coverage_count"]
    false_ratio = summary["false_purpose_page_ratio"]
    if purpose_ratio < 0.03:
        return 0
    if explicit_ratio < 0.03:
        return 1
    if explicit_ratio < 0.06 or stakeholder_ratio < 0.06:
        return 2
    if explicit_ratio < 0.1 or source_coverage < 2 or false_ratio > 0.25:
        return 3
    if explicit_ratio < 0.16 or stakeholder_ratio < 0.12 or false_ratio > 0.12:
        return 4
    return 5


def _clarity(summary):
    explicit_ratio = summary["explicit_purpose_page_ratio"]
    accountability_ratio = summary["accountability_page_ratio"]
    high_signal_ratio = summary["high_signal_page_ratio"]
    stakeholder_ratio = summary["stakeholder_sentence_page_ratio"]
    false_ratio = summary["false_purpose_page_ratio"]
    if explicit_ratio < 0.03 and accountability_ratio < 0.03:
        return 0
    if explicit_ratio < 0.05 and high_signal_ratio < 0.06:
        return 1
    if explicit_ratio < 0.08 or stakeholder_ratio < 0.04:
        return 2
    if explicit_ratio < 0.12 or accountability_ratio < 0.08 or false_ratio > 0.18:
        return 3
    if explicit_ratio < 0.18 or high_signal_ratio < 0.15 or stakeholder_ratio < 0.1 or false_ratio > 0.08:
        return 4
    return 5


def _historical_consistency(summary):
    sec_years = len([year for year in summary["sec_filing_years"] if 2021 <= year <= 2025])
    if sec_years <= 1:
        return 1
    if sec_years == 2:
        return 2
    if sec_years == 3:
        return 3
    if sec_years == 4:
        return 4
    return 5


def _leadership_centrality(summary):
    leadership_ratio = summary["leadership_page_ratio"]
    explicit_ratio = summary["explicit_purpose_page_ratio"]
    if leadership_ratio == 0:
        return 0
    if leadership_ratio < 0.03:
        return 1
    if leadership_ratio < 0.06:
        return 2
    if leadership_ratio < 0.1 or explicit_ratio < 0.06:
        return 3
    if leadership_ratio < 0.16:
        return 4
    return 5


def _distinction_from_branding(summary):
    strategy_ratio = summary["strategy_page_ratio"]
    accountability_ratio = summary["accountability_page_ratio"]
    explicit_ratio = summary["explicit_purpose_page_ratio"]
    false_ratio = summary["false_purpose_page_ratio"]
    if strategy_ratio < 0.03 and accountability_ratio < 0.03:
        return 1
    if strategy_ratio < 0.05 and accountability_ratio < 0.05:
        return 2
    if strategy_ratio < 0.1 or false_ratio > 0.18:
        return 3
    if strategy_ratio >= 0.1 and accountability_ratio >= 0.08 and explicit_ratio >= 0.08 and false_ratio <= 0.1:
        return 4
    return 5
