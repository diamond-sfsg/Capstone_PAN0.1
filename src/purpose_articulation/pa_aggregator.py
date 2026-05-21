from __future__ import annotations

import json
from typing import Any

import pandas as pd

from purpose_articulation.pa_config import (
    Q3_SET_QUALITY_MIN,
    Q3_SET_QUALITY_RANGE,
    TOP1_WEIGHT,
    TOP2_WEIGHT,
)
from purpose_articulation.pa_evidence_score import clamp
from purpose_articulation.pa_rubric import PA_QUESTIONS, QUESTION_ORDER


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _serialize_flags(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return "|".join(str(item) for item in value)

    if isinstance(value, str):
        return value

    return str(value)


def compute_pa_evidence_contribution(row: pd.Series) -> float:
    """
    Compute final contribution for one Q1/Q2 evidence row.

    Formula:
        contribution =
            llm_score_0_5
            × evidence_quality_factor
            × overlap_factor
            + pa_tone_bonus

    Then clamp to [0, 5].

    Notes:
    - source prior is already used in retrieval/ranking.
    - source prior is not directly multiplied here to avoid double boosting.
    """
    llm_score = _safe_float(row.get("llm_score_0_5"), 0.0)
    evidence_quality = _safe_float(row.get("evidence_quality_factor"), 0.70)
    overlap_factor = _safe_float(row.get("overlap_factor"), 1.00)
    tone_bonus = _safe_float(row.get("pa_tone_bonus"), 0.0)

    contribution = (
        llm_score
        * evidence_quality
        * overlap_factor
        + tone_bonus
    )

    return clamp(contribution, 0.0, 5.0)


def aggregate_evidence_question_score(
    evidence_df: pd.DataFrame,
    question_id: str,
) -> dict:
    """
    Aggregate evidence-level rows into a Q1/Q2 question score.

    Formula:
        if only one evidence:
            question_score = top1
        if two or more:
            question_score = 0.70 * top1 + 0.30 * top2

    This avoids rewarding a company just because it has many repetitive chunks.
    """
    if question_id not in PA_QUESTIONS:
        raise ValueError(f"Unknown question_id: {question_id}")

    if evidence_df is None or evidence_df.empty:
        return {
            "question_id": question_id,
            "question_name": PA_QUESTIONS[question_id].name,
            "question_text": PA_QUESTIONS[question_id].question,
            "scoring_type": "evidence",
            "question_score_0_5": 0.0,
            "question_score_0_100": 0.0,
            "top_evidence_chunk_ids": "",
            "evidence_count_used": 0,
            "needs_human_review": True,
            "aggregation_note": "No evidence available for this question.",
        }

    q_df = evidence_df[evidence_df["question_id"] == question_id].copy()

    if q_df.empty:
        return {
            "question_id": question_id,
            "question_name": PA_QUESTIONS[question_id].name,
            "question_text": PA_QUESTIONS[question_id].question,
            "scoring_type": "evidence",
            "question_score_0_5": 0.0,
            "question_score_0_100": 0.0,
            "top_evidence_chunk_ids": "",
            "evidence_count_used": 0,
            "needs_human_review": True,
            "aggregation_note": "No evidence available for this question.",
        }

    if "pa_evidence_contribution" not in q_df.columns:
        q_df["pa_evidence_contribution"] = q_df.apply(
            compute_pa_evidence_contribution,
            axis=1,
        )

    q_df = q_df.sort_values(
        by=[
            "pa_evidence_contribution",
            "llm_score_0_5",
            "base_evidence_score_0_1",
        ],
        ascending=False,
    ).reset_index(drop=True)

    top1 = _safe_float(q_df.iloc[0].get("pa_evidence_contribution"), 0.0)
    top2 = (
        _safe_float(q_df.iloc[1].get("pa_evidence_contribution"), 0.0)
        if len(q_df) > 1
        else None
    )

    if top2 is None:
        question_score = top1
        aggregation_note = "Only one evidence row available; score equals top evidence contribution."
    else:
        question_score = TOP1_WEIGHT * top1 + TOP2_WEIGHT * top2
        aggregation_note = f"Weighted top-2 aggregation: {TOP1_WEIGHT:.2f} * top1 + {TOP2_WEIGHT:.2f} * top2."

    question_score = clamp(question_score, 0.0, 5.0)

    top_ids = q_df.head(2)["chunk_id"].astype(str).tolist()

    return {
        "question_id": question_id,
        "question_name": PA_QUESTIONS[question_id].name,
        "question_text": PA_QUESTIONS[question_id].question,
        "scoring_type": "evidence",
        "question_score_0_5": question_score,
        "question_score_0_100": question_score / 5.0 * 100.0,
        "top_evidence_chunk_ids": "|".join(top_ids),
        "evidence_count_used": int(len(q_df)),
        "needs_human_review": bool(len(q_df) < 1),
        "aggregation_note": aggregation_note,
    }


def aggregate_q3_score(
    company: str,
    llm_set_result: dict,
    set_quality: dict,
    evidence_set_df: pd.DataFrame,
) -> dict:
    """
    Aggregate Q3 evidence-set LLM result into a question score.

    Formula:
        Q3_score =
            llm_set_score_0_5
            × evidence_set_quality_factor

    where:
        evidence_set_quality_factor = 0.75 + 0.25 * evidence_set_quality

    Q3 is evidence-set scoring because it evaluates whether purpose appears
    beyond branding / marketing context.
    """
    llm_set_score = _safe_float(llm_set_result.get("llm_set_score_0_5"), 0.0)

    evidence_set_quality = _safe_float(set_quality.get("evidence_set_quality"), 0.0)
    evidence_set_quality_factor = (
        Q3_SET_QUALITY_MIN
        + Q3_SET_QUALITY_RANGE * evidence_set_quality
    )

    q3_score = clamp(
        llm_set_score * evidence_set_quality_factor,
        0.0,
        5.0,
    )

    if evidence_set_df is None or evidence_set_df.empty:
        top_ids = []
        evidence_count_used = 0
    else:
        top_ids = evidence_set_df.head(5)["chunk_id"].astype(str).tolist()
        evidence_count_used = int(len(evidence_set_df))

    return {
        "company": company,
        "question_id": "PA_Q3",
        "question_name": PA_QUESTIONS["PA_Q3"].name,
        "question_text": PA_QUESTIONS["PA_Q3"].question,
        "scoring_type": "evidence_set",
        "question_score_0_5": q3_score,
        "question_score_0_100": q3_score / 5.0 * 100.0,
        "top_evidence_chunk_ids": "|".join(top_ids),
        "evidence_count_used": evidence_count_used,
        "needs_human_review": bool(evidence_count_used == 0),
        "llm_set_score_0_5": llm_set_score,
        "evidence_set_quality": evidence_set_quality,
        "evidence_set_quality_factor": evidence_set_quality_factor,
        "source_diversity": _safe_float(set_quality.get("source_diversity"), 0.0),
        "formal_document_presence": _safe_float(set_quality.get("formal_document_presence"), 0.0),
        "strategic_section_presence": _safe_float(set_quality.get("strategic_section_presence"), 0.0),
        "llm_reason": str(llm_set_result.get("reason", "")),
        "risk_flags": _serialize_flags(llm_set_result.get("risk_flags")),
        "aggregation_note": "Q3 uses evidence-set scoring with evidence-set quality adjustment.",
    }


def aggregate_company_pa_score(
    company: str,
    question_rows: list[dict],
) -> dict:
    """
    Aggregate Q1/Q2/Q3 into company-level PA score.

    Formula:
        PA_score_0_5 =
            average(PA_Q1, PA_Q2, PA_Q3)

        PA_score_0_100 =
            PA_score_0_5 / 5 * 100
    """
    row_by_question = {
        row.get("question_id"): row
        for row in question_rows
    }

    scores: dict[str, float] = {}

    for question_id in QUESTION_ORDER:
        row = row_by_question.get(question_id)
        if row is None:
            scores[question_id] = 0.0
        else:
            scores[question_id] = _safe_float(row.get("question_score_0_5"), 0.0)

    pa_score_0_5 = sum(scores.values()) / len(QUESTION_ORDER)
    pa_score_0_100 = pa_score_0_5 / 5.0 * 100.0

    needs_human_review = any(
        bool(row.get("needs_human_review", False))
        for row in question_rows
    )

    return {
        "company": company,
        "PA_Q1_purpose_presence_score": scores.get("PA_Q1", 0.0),
        "PA_Q2_clarity_score": scores.get("PA_Q2", 0.0),
        "PA_Q3_distinction_from_branding_score": scores.get("PA_Q3", 0.0),
        "PA_score_0_5": pa_score_0_5,
        "PA_score_0_100": pa_score_0_100,
        "needs_human_review": needs_human_review,
    }


def attach_company_score_metadata(
    company_score: dict,
    year_stats: dict,
    source_mix: str,
) -> dict:
    """
    Add diagnostic metadata to company-level PA score.
    """
    result = dict(company_score)

    result.update(year_stats)
    result["source_mix"] = source_mix

    limitation_parts = []

    year_note = year_stats.get("year_coverage_note")
    if year_note:
        limitation_parts.append(str(year_note))

    if source_mix:
        try:
            source_obj = json.loads(source_mix)
            if "official_web" not in source_obj:
                limitation_parts.append("No official_web evidence in scored evidence pool.")
            if "edgar" not in source_obj:
                limitation_parts.append("No EDGAR/formal evidence in scored evidence pool.")
        except json.JSONDecodeError:
            pass

    result["limitation_note"] = " ".join(limitation_parts).strip()

    return result
