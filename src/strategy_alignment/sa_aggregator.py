# src/strategy_alignment/sa_aggregator.py

"""
Aggregation logic for Strategy & Source Alignment.

The aggregation follows the PA/HC new-rule pattern:

1. LLM score is the primary judgment.
2. Evidence quality factor mildly adjusts the LLM score.
3. Overlap factor mildly controls duplicated evidence.
4. Question score is aggregated from top evidence contributions.
5. Final SA score averages the two question scores.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .sa_config import (
    QUESTION_ORDER,
    SA_QUESTIONS,
    QUESTION_AGGREGATION_WEIGHTS,
    SA_FINAL_SCORE_WEIGHTS,
    MIN_EVIDENCE_PER_QUESTION,
    MIN_PURPOSE_CONFIDENCE_FOR_AUTO_SCORE,
    MIN_FINAL_SCORE,
    MAX_FINAL_SCORE,
)
from .sa_evidence_score import compute_evidence_contribution, clip


# ============================================================
# Helpers
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]

        return [text]

    return [str(value).strip()]


def _clip_score_0_5(value: Any) -> float:
    return clip(_safe_float(value, 0.0), MIN_FINAL_SCORE, MAX_FINAL_SCORE)


# ============================================================
# Evidence contribution assignment
# ============================================================

def attach_llm_result_to_evidence(
    candidates_df: pd.DataFrame,
    llm_result: Dict[str, Any],
) -> pd.DataFrame:
    """
    Attach one question-level LLM result to candidate evidence rows.

    In this first design, the LLM scores the question based on the evidence pack.
    The same LLM question score is applied to candidate evidence rows, then
    adjusted by each row's evidence quality and overlap factors.

    Later, if you switch to evidence-by-evidence LLM scoring, this function can
    be replaced without changing the aggregator interface.
    """
    if candidates_df.empty:
        return candidates_df.copy()

    out = candidates_df.copy()

    llm_score = _clip_score_0_5(llm_result.get("llm_score_0_5", 0.0))

    out["llm_score_0_5"] = llm_score
    out["llm_reasoning"] = _safe_str(llm_result.get("alignment_summary", ""))
    out["purpose_connection_type"] = _safe_str(
        llm_result.get("purpose_connection_type", "")
    )

    best_ids = set(_ensure_list(llm_result.get("best_supporting_evidence_ids", [])))
    weak_ids = set(_ensure_list(llm_result.get("contradictory_or_weak_evidence_ids", [])))

    out["llm_marked_best_support"] = out["chunk_id"].astype(str).isin(best_ids)
    out["llm_marked_weak_or_contradictory"] = out["chunk_id"].astype(str).isin(weak_ids)

    # Slightly reward rows explicitly selected by LLM and penalize weak rows.
    # This is still mild and cannot create a score above the LLM score.
    out["llm_selection_factor"] = 1.00
    out.loc[out["llm_marked_best_support"], "llm_selection_factor"] = 1.05
    out.loc[out["llm_marked_weak_or_contradictory"], "llm_selection_factor"] = 0.70

    out["evidence_contribution_0_5"] = out.apply(
        lambda row: compute_evidence_contribution(
            llm_score_0_5=row["llm_score_0_5"],
            evidence_quality_factor=_safe_float(
                row.get("evidence_quality_factor", 0.70),
                0.70,
            )
            * _safe_float(row.get("llm_selection_factor", 1.0), 1.0),
            overlap_factor=_safe_float(row.get("overlap_factor", 1.0), 1.0),
        ),
        axis=1,
    )

    out["llm_needs_human_review"] = bool(
        llm_result.get("needs_human_review", False)
    )
    out["llm_human_review_reason"] = _safe_str(
        llm_result.get("human_review_reason", "")
    )
    out["score_label"] = _safe_str(llm_result.get("score_label", ""))

    return out


def attach_llm_results_to_all_evidence(
    candidate_map: Dict[str, pd.DataFrame],
    llm_results: Dict[str, Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    """
    Attach LLM results to each question's candidate DataFrame.
    """
    output = {}

    for question_id, candidates_df in candidate_map.items():
        llm_result = llm_results.get(question_id, {})
        output[question_id] = attach_llm_result_to_evidence(
            candidates_df=candidates_df,
            llm_result=llm_result,
        )

    return output


# ============================================================
# Question-level aggregation
# ============================================================

def aggregate_question_score(
    evidence_df: pd.DataFrame,
    question_id: str,
) -> Dict[str, Any]:
    """
    Aggregate one SA question score from candidate evidence contributions.

    Formula:
    SA_Q_score =
        0.70 * best_evidence_contribution
      + 0.30 * second_best_evidence_contribution

    If only one evidence row is available, use that one contribution directly.
    """
    if evidence_df.empty:
        return {
            "question_id": question_id,
            "question_name": SA_QUESTIONS.get(question_id, ""),
            "question_score_0_5": 0.0,
            "num_evidence": 0,
            "best_evidence_id": "",
            "second_best_evidence_id": "",
            "needs_human_review": True,
            "review_reason": "no_candidate_evidence",
        }

    working = evidence_df.copy()

    if "evidence_contribution_0_5" not in working.columns:
        raise ValueError(
            "evidence_df must contain evidence_contribution_0_5 before aggregation"
        )

    working = working.sort_values(
        by=["evidence_contribution_0_5", "rank_score"],
        ascending=False,
    ).reset_index(drop=True)

    num_evidence = len(working)

    best = working.iloc[0]
    best_score = _clip_score_0_5(best["evidence_contribution_0_5"])
    best_id = _safe_str(best.get("chunk_id", ""))

    if num_evidence >= 2:
        second = working.iloc[1]
        second_score = _clip_score_0_5(second["evidence_contribution_0_5"])
        second_id = _safe_str(second.get("chunk_id", ""))

        q_score = (
            QUESTION_AGGREGATION_WEIGHTS["best_evidence"] * best_score
            + QUESTION_AGGREGATION_WEIGHTS["second_best_evidence"] * second_score
        )
    else:
        second_score = 0.0
        second_id = ""
        q_score = best_score

    needs_review = bool(working.get("llm_needs_human_review", pd.Series([False])).any())
    review_reasons = []

    if num_evidence < MIN_EVIDENCE_PER_QUESTION:
        needs_review = True
        review_reasons.append("insufficient_candidate_evidence")

    llm_reasons = (
        working.get("llm_human_review_reason", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.strip()
    )
    llm_reasons = [x for x in llm_reasons.unique().tolist() if x]

    review_reasons.extend(llm_reasons)

    return {
        "question_id": question_id,
        "question_name": SA_QUESTIONS.get(question_id, ""),
        "question_score_0_5": _clip_score_0_5(q_score),
        "num_evidence": num_evidence,
        "best_evidence_id": best_id,
        "second_best_evidence_id": second_id,
        "best_evidence_contribution_0_5": best_score,
        "second_best_evidence_contribution_0_5": second_score,
        "needs_human_review": needs_review,
        "review_reason": ";".join(sorted(set(review_reasons))),
    }


def aggregate_all_question_scores(
    evidence_map: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate all SA question scores.
    """
    output = {}

    for question_id in QUESTION_ORDER:
        evidence_df = evidence_map.get(question_id, pd.DataFrame())
        output[question_id] = aggregate_question_score(
            evidence_df=evidence_df,
            question_id=question_id,
        )

    return output


# ============================================================
# Final SA aggregation
# ============================================================

def aggregate_sa_final_score(
    company: str,
    purpose_reference: Dict[str, Any],
    question_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate final SA score from SA_Q1 and SA_Q2.

    First version:
    SA_final_score_0_5 =
        0.50 * SA_Q1
      + 0.50 * SA_Q2
    """
    sa_q1 = _clip_score_0_5(
        question_results.get("SA_Q1", {}).get("question_score_0_5", 0.0)
    )
    sa_q2 = _clip_score_0_5(
        question_results.get("SA_Q2", {}).get("question_score_0_5", 0.0)
    )

    final_score = (
        SA_FINAL_SCORE_WEIGHTS["SA_Q1"] * sa_q1
        + SA_FINAL_SCORE_WEIGHTS["SA_Q2"] * sa_q2
    )

    final_score = _clip_score_0_5(final_score)
    score_0_100 = final_score / 5.0 * 100.0

    purpose_confidence = _safe_float(
        purpose_reference.get("purpose_confidence_0_1", 0.0),
        0.0,
    )

    review_reasons = []

    if purpose_confidence < MIN_PURPOSE_CONFIDENCE_FOR_AUTO_SCORE:
        review_reasons.append("low_purpose_confidence")

    for question_id, result in question_results.items():
        if bool(result.get("needs_human_review", False)):
            reason = _safe_str(result.get("review_reason", ""))
            if reason:
                review_reasons.append(f"{question_id}:{reason}")
            else:
                review_reasons.append(f"{question_id}:needs_review")

    needs_review = len(review_reasons) > 0

    return {
        "company": company,
        "extracted_purpose": _safe_str(
            purpose_reference.get("extracted_purpose", "")
            or purpose_reference.get("purpose_statement_normalized", "")
        ),
        "purpose_statement_normalized": _safe_str(
            purpose_reference.get("purpose_statement_normalized", "")
        ),
        "purpose_statement_raw": _safe_str(
            purpose_reference.get("purpose_statement_raw", "")
        ),
        "purpose_confidence_0_1": purpose_confidence,
        "sa_q1_score_0_5": sa_q1,
        "sa_q2_score_0_5": sa_q2,
        "sa_final_score_0_5": final_score,
        "sa_score_0_100": score_0_100,
        "sa_needs_human_review": needs_review,
        "sa_review_reason": ";".join(sorted(set(review_reasons))),
    }


# ============================================================
# Full company aggregation wrapper
# ============================================================

def aggregate_company_sa_result(
    company: str,
    purpose_reference: Dict[str, Any],
    candidate_map: Dict[str, pd.DataFrame],
    llm_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Full aggregation for one company.

    Returns:
    - final_score_row
    - question_results
    - evidence_map_with_scores
    """
    evidence_map = attach_llm_results_to_all_evidence(
        candidate_map=candidate_map,
        llm_results=llm_results,
    )

    question_results = aggregate_all_question_scores(evidence_map)

    final_score_row = aggregate_sa_final_score(
        company=company,
        purpose_reference=purpose_reference,
        question_results=question_results,
    )

    return {
        "final_score_row": final_score_row,
        "question_results": question_results,
        "evidence_map": evidence_map,
    }


# ============================================================
# Export helper
# ============================================================

def flatten_evidence_map(
    company: str,
    purpose_reference: Dict[str, Any],
    evidence_map: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Convert question-level evidence map into one long DataFrame.
    """
    frames = []

    for question_id, df in evidence_map.items():
        if df.empty:
            continue

        temp = df.copy()
        temp["company"] = company
        temp["question_id"] = question_id
        temp["question_name"] = SA_QUESTIONS.get(question_id, "")
        temp["purpose_statement_normalized"] = _safe_str(
            purpose_reference.get("purpose_statement_normalized", "")
        )
        temp["extracted_purpose"] = _safe_str(
            purpose_reference.get("extracted_purpose", "")
            or purpose_reference.get("purpose_statement_normalized", "")
        )
        temp["purpose_statement_raw"] = _safe_str(
            purpose_reference.get("purpose_statement_raw", "")
        )
        temp["purpose_confidence_0_1"] = _safe_float(
            purpose_reference.get("purpose_confidence_0_1", 0.0),
            0.0,
        )

        temp["needs_human_review"] = temp.get(
            "llm_needs_human_review",
            False,
        )

        frames.append(temp)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
