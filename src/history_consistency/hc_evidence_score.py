# src/history_consistency/hc_evidence_score.py

from __future__ import annotations

from typing import Any

import pandas as pd

from history_consistency.hc_config import (
    EVIDENCE_QUALITY_FACTOR_MIN,
    EVIDENCE_QUALITY_FACTOR_SCALE,
    HC_SCORE_MIN,
    HC_SCORE_MAX,
)


def clamp(value: Any, min_value: float, max_value: float) -> float:
    """
    Safely clamp numeric values to a score range.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = min_value

    if pd.isna(numeric):
        numeric = min_value

    return float(max(min_value, min(max_value, numeric)))


def evidence_quality_factor(base_evidence_score_0_1: Any) -> float:
    """
    Convert retrieval base evidence score into a mild quality factor.
    """
    base = clamp(base_evidence_score_0_1, 0.0, 1.0)
    return float(EVIDENCE_QUALITY_FACTOR_MIN + EVIDENCE_QUALITY_FACTOR_SCALE * base)


def fallback_llm_score_from_base(base_evidence_score_0_1: Any) -> float:
    """
    Deterministic fallback for local runs without LLM review.

    The fallback keeps the pipeline runnable and is flagged downstream as
    fallback_from_base_score.
    """
    return clamp(clamp(base_evidence_score_0_1, 0.0, 1.0) * HC_SCORE_MAX, HC_SCORE_MIN, HC_SCORE_MAX)


def _safe_factor(row: pd.Series, column: str, default: float = 1.0) -> float:
    if column not in row.index:
        return default
    return clamp(row.get(column), 0.0, 10.0)


def compute_hc_evidence_contribution(
    row: pd.Series,
    *,
    llm_score_col: str = "hc_llm_score_0_5",
    allow_fallback_llm_score: bool = True,
) -> tuple[float, str, float]:
    """
    Compute one evidence row contribution in 0-5 space.

    Returns:
        contribution_0_5, score_source, quality_factor
    """
    base_score = row.get("hc_base_evidence_score_0_1", 0.0)
    quality_factor = evidence_quality_factor(base_score)
    redundancy_factor = _safe_factor(row, "hc_redundancy_factor", default=1.0)

    llm_score = row.get(llm_score_col) if llm_score_col in row.index else None
    llm_score_missing = llm_score is None or pd.isna(llm_score)

    if llm_score_missing:
        if allow_fallback_llm_score:
            llm_score_0_5 = fallback_llm_score_from_base(base_score)
            score_source = "fallback_from_base_score"
        else:
            llm_score_0_5 = 0.0
            score_source = "missing_llm_score"
    else:
        llm_score_0_5 = clamp(llm_score, HC_SCORE_MIN, HC_SCORE_MAX)
        score_source = "llm"

    contribution = clamp(
        llm_score_0_5 * quality_factor * redundancy_factor,
        HC_SCORE_MIN,
        HC_SCORE_MAX,
    )

    return contribution, score_source, quality_factor


def score_hc_evidence_dataframe(
    evidence_df: pd.DataFrame,
    *,
    llm_score_col: str = "hc_llm_score_0_5",
    allow_fallback_llm_score: bool = True,
) -> pd.DataFrame:
    """
    Add HC evidence contribution columns expected by the aggregator.
    """
    out = evidence_df.copy()

    if out.empty:
        out["hc_evidence_quality_factor"] = []
        out["hc_llm_score_source"] = []
        out["hc_evidence_contribution_0_5"] = []
        return out

    if "hc_base_evidence_score_0_1" not in out.columns:
        raise ValueError("Evidence dataframe missing hc_base_evidence_score_0_1.")

    if "hc_redundancy_factor" not in out.columns:
        out["hc_redundancy_factor"] = 1.0

    contributions = out.apply(
        lambda row: compute_hc_evidence_contribution(
            row,
            llm_score_col=llm_score_col,
            allow_fallback_llm_score=allow_fallback_llm_score,
        ),
        axis=1,
    )

    out["hc_evidence_contribution_0_5"] = [item[0] for item in contributions]
    out["hc_llm_score_source"] = [item[1] for item in contributions]
    out["hc_evidence_quality_factor"] = [item[2] for item in contributions]

    return out
