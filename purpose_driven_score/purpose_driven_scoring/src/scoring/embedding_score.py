"""Embedding-based scoring logic."""

from __future__ import annotations


def compute_embedding_score(summary):
    """Compute the embedding score from extracted features."""
    dimensions = {
        "corporate_strategy_integration": _strategy_integration(summary),
        "capital_and_performance_alignment": _capital_alignment(summary),
        "employee_embedding": _employee_embedding(summary),
        "strategic_decision_justification": _decision_justification(summary),
        "execution_outcome_accountability": _accountability(summary),
    }
    overall = round(sum(dimensions.values()) / len(dimensions), 2)
    return {"overall": overall, "dimensions": dimensions}


def _strategy_integration(summary):
    ratio = summary["strategy_page_ratio"]
    source_coverage = summary["source_coverage_count"]
    explicit_ratio = summary["explicit_purpose_page_ratio"]
    if ratio == 0:
        return 0
    if ratio < 0.05:
        return 1
    if ratio < 0.1:
        return 2
    if ratio < 0.16 or source_coverage < 2 or explicit_ratio < 0.06:
        return 3
    if ratio < 0.25:
        return 4
    return 5


def _capital_alignment(summary):
    ratio = summary["capital_page_ratio"]
    sec_coverage = summary["sec_year_coverage_ratio"]
    explicit_ratio = summary["explicit_purpose_page_ratio"]
    if ratio == 0:
        return 0
    if ratio < 0.05:
        return 1
    if ratio < 0.1:
        return 2
    if ratio < 0.16 or explicit_ratio < 0.06:
        return 3
    if ratio < 0.25 or sec_coverage < 0.4:
        return 4
    return 5


def _employee_embedding(summary):
    ratio = summary["employee_page_ratio"]
    stakeholder_ratio = summary["stakeholder_sentence_page_ratio"]
    if ratio == 0:
        return 0
    if ratio < 0.04:
        return 1
    if ratio < 0.08:
        return 2
    if ratio < 0.14 or stakeholder_ratio < 0.05:
        return 3
    if ratio < 0.22:
        return 4
    return 5


def _decision_justification(summary):
    ratio = summary["decision_page_ratio"]
    explicit_ratio = summary["explicit_purpose_page_ratio"]
    if ratio == 0:
        return 0
    if ratio < 0.03:
        return 1
    if ratio < 0.06:
        return 2
    if ratio < 0.1 or explicit_ratio < 0.08:
        return 3
    if ratio < 0.16:
        return 4
    return 5


def _accountability(summary):
    ratio = summary["accountability_page_ratio"]
    sec_coverage = summary["sec_year_coverage_ratio"]
    if ratio == 0:
        return 0
    if ratio < 0.05:
        return 1
    if ratio < 0.1:
        return 2
    if ratio < 0.18:
        return 3
    if ratio < 0.28 or sec_coverage < 0.4:
        return 4
    return 5
