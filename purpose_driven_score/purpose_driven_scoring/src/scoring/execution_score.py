"""Execution scoring logic."""

from __future__ import annotations


def compute_execution_score(summary):
    """Compute the execution score from extracted features."""
    dimensions = {
        "corporate_strategy_integration": _strategy_integration(summary),
        "capital_and_performance_alignment": _capital_alignment(summary),
        "execution_consistency": _execution_consistency(summary),
        "core_capability_reinforcement": _capability_reinforcement(summary),
        "industry_impact_breadth": _industry_breadth(summary),
    }
    overall = round(sum(dimensions.values()) / len(dimensions), 2)
    return {"overall": overall, "dimensions": dimensions}


def _strategy_integration(summary):
    ratio = summary["strategy_page_ratio"]
    if ratio == 0:
        return 0
    if ratio < 0.06:
        return 1
    if ratio < 0.12:
        return 2
    if ratio < 0.2:
        return 3
    if ratio < 0.3:
        return 4
    return 5


def _capital_alignment(summary):
    ratio = summary["capital_page_ratio"]
    if ratio == 0:
        return 0
    if ratio < 0.06:
        return 1
    if ratio < 0.12:
        return 2
    if ratio < 0.2:
        return 3
    if ratio < 0.3:
        return 4
    return 5


def _execution_consistency(summary):
    ratio = summary["execution_page_ratio"]
    sec_coverage = summary["sec_year_coverage_ratio"]
    if ratio == 0:
        return 0
    if ratio < 0.08:
        return 1
    if ratio < 0.14:
        return 2
    if ratio < 0.22:
        return 3
    if ratio < 0.32 or sec_coverage < 0.4:
        return 4
    return 5


def _capability_reinforcement(summary):
    ratio = summary["capability_page_ratio"]
    high_signal_ratio = summary["high_signal_page_ratio"]
    if ratio == 0:
        return 0
    if ratio < 0.05:
        return 1
    if ratio < 0.1:
        return 2
    if ratio < 0.18:
        return 3
    if ratio < 0.28 or high_signal_ratio < 0.12:
        return 4
    return 5


def _industry_breadth(summary):
    ratio = summary["industry_page_ratio"]
    if ratio == 0:
        return 0
    if ratio < 0.05:
        return 1
    if ratio < 0.1:
        return 2
    if ratio < 0.18:
        return 3
    if ratio < 0.28:
        return 4
    return 5
