# src/history_consistency/__init__.py

"""
History Consistency

Company-level scoring pipeline for evaluating whether a company has maintained
a consistent purpose narrative across the past ten years of disclosures.
"""

from history_consistency.hc_config import (
    DIMENSION_ID,
    DIMENSION_LABEL,
    HC_QUESTION_ID,
    HC_QUESTION_TEXT,
    HC_RUBRIC,
    HC_QUESTION_CONFIG,
    HC_SCORING_CONFIG,
    validate_hc_config,
)

__all__ = [
    "DIMENSION_ID",
    "DIMENSION_LABEL",
    "HC_QUESTION_ID",
    "HC_QUESTION_TEXT",
    "HC_RUBRIC",
    "HC_QUESTION_CONFIG",
    "HC_SCORING_CONFIG",
    "validate_hc_config",
]
