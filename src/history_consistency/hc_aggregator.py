# src/history_consistency/hc_aggregator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from history_consistency.hc_config import (
    HC_QUESTION_ID,
    HC_QUESTION_TEXT,
    HC_AGGREGATION_WEIGHTS,
    SINGLE_YEAR_SCORE_CAP,
    HC_SCORE_MIN,
    HC_SCORE_MAX,
    MIN_EVIDENCE_PER_COMPANY,
    MIN_YEARS_FOR_FULL_HC,
)
from history_consistency.hc_bonus import (
    compute_hc_history_bonus,
    bonus_result_to_dict,
)
from history_consistency.hc_evidence_score import clamp


@dataclass(frozen=True)
class HCAggregationResult:
    """
    Final company-level HC score result.
    """

    company: str
    hc_question_id: str
    hc_question_text: str
    hc_final_score_0_5: float
    hc_score_0_100: float
    hc_base_score_0_5: float
    hc_history_bonus: float
    distinct_year_count: int
    evidence_count: int
    best_evidence_score: float
    best_distinct_year_evidence_score: float
    mean_top_evidence_by_year: float
    needs_human_review: bool
    review_reason: str


def score_to_100(score_0_5: float) -> float:
    """
    Convert 0–5 score to 0–100 score.
    """
    score = clamp(score_0_5, HC_SCORE_MIN, HC_SCORE_MAX)
    return float(score / 5.0 * 100.0)


def get_best_evidence_score(
    df: pd.DataFrame,
    contribution_col: str = "hc_evidence_contribution_0_5",
) -> float:
    """
    Return best evidence contribution.
    """
    if df.empty or contribution_col not in df.columns:
        return 0.0

    values = pd.to_numeric(df[contribution_col], errors="coerce").dropna()

    if values.empty:
        return 0.0

    return float(values.max())


def get_best_distinct_year_evidence_score(
    df: pd.DataFrame,
    contribution_col: str = "hc_evidence_contribution_0_5",
) -> float:
    """
    Return best evidence contribution from a year different from the top evidence year.

    If no second distinct year exists, return 0.
    """
    if (
        df.empty
        or contribution_col not in df.columns
        or "year" not in df.columns
    ):
        return 0.0

    temp = df.copy()
    temp[contribution_col] = pd.to_numeric(
        temp[contribution_col],
        errors="coerce",
    )
    temp["year_numeric"] = pd.to_numeric(temp["year"], errors="coerce")

    temp = temp.dropna(subset=[contribution_col, "year_numeric"])

    if temp.empty:
        return 0.0

    temp = temp.sort_values(contribution_col, ascending=False)
    top_year = int(temp.iloc[0]["year_numeric"])

    other_years = temp[temp["year_numeric"].astype(int) != top_year]

    if other_years.empty:
        return 0.0

    return float(other_years[contribution_col].max())


def get_mean_top_evidence_by_year(
    df: pd.DataFrame,
    contribution_col: str = "hc_evidence_contribution_0_5",
) -> float:
    """
    Compute mean of each year's top evidence contribution.

    This rewards distributed multi-year evidence instead of many chunks
    from the same year.
    """
    if (
        df.empty
        or contribution_col not in df.columns
        or "year" not in df.columns
    ):
        return 0.0

    temp = df.copy()
    temp[contribution_col] = pd.to_numeric(
        temp[contribution_col],
        errors="coerce",
    )
    temp["year_numeric"] = pd.to_numeric(temp["year"], errors="coerce")

    temp = temp.dropna(subset=[contribution_col, "year_numeric"])

    if temp.empty:
        return 0.0

    top_by_year = temp.groupby("year_numeric")[contribution_col].max()

    if top_by_year.empty:
        return 0.0

    return float(top_by_year.mean())


def compute_hc_base_score(
    best_evidence_score: float,
    best_distinct_year_evidence_score: float,
    mean_top_evidence_by_year: float,
) -> float:
    """
    Compute HC base score before history bonus.

    Formula:
    HC_Q_base_score =
        0.50 × best_evidence
        + 0.30 × best_distinct_year_evidence
        + 0.20 × mean_top_evidence_by_year
    """
    w = HC_AGGREGATION_WEIGHTS

    score = (
        w["best_evidence"] * best_evidence_score
        + w["best_distinct_year_evidence"] * best_distinct_year_evidence_score
        + w["mean_top_evidence_by_year"] * mean_top_evidence_by_year
    )

    return clamp(score, HC_SCORE_MIN, HC_SCORE_MAX)


def maybe_apply_single_year_cap(
    base_score_0_5: float,
    distinct_year_count: int,
) -> float:
    """
    Cap HC score if evidence only comes from one year.

    Rationale:
    One-year evidence can support a history narrative,
    but it cannot fully prove ten-year consistency.
    """
    if distinct_year_count <= 1:
        return float(min(base_score_0_5, SINGLE_YEAR_SCORE_CAP))

    return base_score_0_5


def build_review_reason(
    evidence_count: int,
    distinct_year_count: int,
    used_fallback_llm: bool,
) -> Tuple[bool, str]:
    """
    Determine whether company-level HC score needs human review.
    """
    reasons: List[str] = []

    if evidence_count < MIN_EVIDENCE_PER_COMPANY:
        reasons.append(
            f"Insufficient evidence: {evidence_count} evidence rows found."
        )

    if distinct_year_count < MIN_YEARS_FOR_FULL_HC:
        reasons.append(
            f"Limited year coverage: {distinct_year_count} distinct years found."
        )

    if used_fallback_llm:
        reasons.append(
            "LLM score fallback was used; replace with rubric-based LLM score before final reporting."
        )

    needs_review = len(reasons) > 0
    review_reason = " | ".join(reasons) if reasons else ""

    return needs_review, review_reason


def aggregate_company_hc_score(
    company: str,
    evidence_df: pd.DataFrame,
    contribution_col: str = "hc_evidence_contribution_0_5",
) -> HCAggregationResult:
    """
    Aggregate selected evidence into final company-level HC score.
    """
    if evidence_df.empty:
        return HCAggregationResult(
            company=company,
            hc_question_id=HC_QUESTION_ID,
            hc_question_text=HC_QUESTION_TEXT,
            hc_final_score_0_5=0.0,
            hc_score_0_100=0.0,
            hc_base_score_0_5=0.0,
            hc_history_bonus=0.0,
            distinct_year_count=0,
            evidence_count=0,
            best_evidence_score=0.0,
            best_distinct_year_evidence_score=0.0,
            mean_top_evidence_by_year=0.0,
            needs_human_review=True,
            review_reason="No HC evidence found.",
        )

    evidence_count = len(evidence_df)

    bonus_result = compute_hc_history_bonus(evidence_df)
    bonus_dict = bonus_result_to_dict(bonus_result)

    distinct_year_count = int(bonus_dict["distinct_year_count"])

    best_evidence_score = get_best_evidence_score(
        evidence_df,
        contribution_col=contribution_col,
    )
    best_distinct_year_evidence_score = get_best_distinct_year_evidence_score(
        evidence_df,
        contribution_col=contribution_col,
    )
    mean_top_evidence_by_year = get_mean_top_evidence_by_year(
        evidence_df,
        contribution_col=contribution_col,
    )

    base_score = compute_hc_base_score(
        best_evidence_score=best_evidence_score,
        best_distinct_year_evidence_score=best_distinct_year_evidence_score,
        mean_top_evidence_by_year=mean_top_evidence_by_year,
    )

    base_score = maybe_apply_single_year_cap(
        base_score_0_5=base_score,
        distinct_year_count=distinct_year_count,
    )

    history_bonus = float(bonus_dict["hc_history_bonus"])

    final_score = clamp(
        base_score + history_bonus,
        HC_SCORE_MIN,
        HC_SCORE_MAX,
    )

    score_100 = score_to_100(final_score)

    used_fallback_llm = False
    if "hc_llm_score_source" in evidence_df.columns:
        used_fallback_llm = bool(
            (evidence_df["hc_llm_score_source"] == "fallback_from_base_score").any()
        )

    needs_review, review_reason = build_review_reason(
        evidence_count=evidence_count,
        distinct_year_count=distinct_year_count,
        used_fallback_llm=used_fallback_llm,
    )

    return HCAggregationResult(
        company=company,
        hc_question_id=HC_QUESTION_ID,
        hc_question_text=HC_QUESTION_TEXT,
        hc_final_score_0_5=final_score,
        hc_score_0_100=score_100,
        hc_base_score_0_5=base_score,
        hc_history_bonus=history_bonus,
        distinct_year_count=distinct_year_count,
        evidence_count=evidence_count,
        best_evidence_score=best_evidence_score,
        best_distinct_year_evidence_score=best_distinct_year_evidence_score,
        mean_top_evidence_by_year=mean_top_evidence_by_year,
        needs_human_review=needs_review,
        review_reason=review_reason,
    )


def aggregation_result_to_dict(result: HCAggregationResult) -> Dict[str, object]:
    """
    Convert aggregation result to dict.
    """
    return {
        "company": result.company,
        "hc_question_id": result.hc_question_id,
        "hc_question_text": result.hc_question_text,
        "hc_final_score_0_5": result.hc_final_score_0_5,
        "hc_score_0_100": result.hc_score_0_100,
        "hc_base_score_0_5": result.hc_base_score_0_5,
        "hc_history_bonus": result.hc_history_bonus,
        "distinct_year_count": result.distinct_year_count,
        "evidence_count": result.evidence_count,
        "best_evidence_score": result.best_evidence_score,
        "best_distinct_year_evidence_score": result.best_distinct_year_evidence_score,
        "mean_top_evidence_by_year": result.mean_top_evidence_by_year,
        "needs_human_review": result.needs_human_review,
        "review_reason": result.review_reason,
    }


def aggregate_all_company_hc_scores(
    evidence_df: pd.DataFrame,
    contribution_col: str = "hc_evidence_contribution_0_5",
) -> pd.DataFrame:
    """
    Aggregate HC scores for all companies in evidence dataframe.
    """
    if evidence_df.empty:
        return pd.DataFrame()

    if "company" not in evidence_df.columns:
        raise ValueError("Evidence dataframe must contain 'company' column.")

    rows: List[Dict[str, object]] = []

    for company, group in evidence_df.groupby("company", dropna=False):
        company_name = str(company).strip()
        result = aggregate_company_hc_score(
            company=company_name,
            evidence_df=group.copy(),
            contribution_col=contribution_col,
        )
        rows.append(aggregation_result_to_dict(result))

    return pd.DataFrame(rows).sort_values(
        ["needs_human_review", "hc_final_score_0_5", "company"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "company": ["A", "A", "A", "B"],
            "chunk_id": ["a1", "a2", "a3", "b1"],
            "year": [2018, 2021, 2024, 2024],
            "hc_evidence_contribution_0_5": [4.2, 3.8, 4.0, 4.5],
            "hc_redundancy_type": [
                "cross_year_recurring",
                "cross_year_recurring",
                "unique_evidence",
                "unique_evidence",
            ],
            "hc_llm_score_source": ["llm", "llm", "llm", "fallback_from_base_score"],
        }
    )

    scores = aggregate_all_company_hc_scores(sample)
    print(scores.to_string(index=False))
