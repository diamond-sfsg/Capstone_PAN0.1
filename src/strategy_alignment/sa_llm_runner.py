# src/strategy_alignment/sa_llm_runner.py

"""
LLM runner and response parser for Strategy & Source Alignment.

This module is deliberately client-agnostic.
You can plug in Gemini, OpenAI, or another model by passing a callable:

    llm_client(prompt: str) -> str

The rest of the pipeline only depends on the normalized JSON result.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional

import pandas as pd

from .sa_config import MIN_LLM_SCORE, MAX_LLM_SCORE


LLMClient = Callable[[str], str]


# ============================================================
# Helpers
# ============================================================

def clip_score(value: Any, lower: float = MIN_LLM_SCORE, upper: float = MAX_LLM_SCORE) -> float:
    try:
        value = float(value)
    except Exception:
        value = lower

    return max(lower, min(upper, value))


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _ensure_list(value: Any) -> list:
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


# ============================================================
# JSON extraction
# ============================================================

def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON object from LLM response.

    Handles:
    - pure JSON
    - fenced ```json blocks
    - extra text before/after JSON
    """
    text = _safe_str(response_text)

    if not text:
        raise ValueError("Empty LLM response")

    # Remove fenced code block if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # Try direct JSON first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: extract first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {response_text[:300]}")

    json_text = text[start : end + 1]

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON from LLM response: {exc}. "
            f"Raw response head: {response_text[:300]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Parsed LLM response is not a JSON object")

    return parsed


# ============================================================
# Normalization
# ============================================================

def normalize_sa_llm_result(
    raw_result: Dict[str, Any],
    expected_question_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize LLM result into stable fields.

    Expected output fields:
    - question_id
    - llm_score_0_5
    - score_label
    - alignment_summary
    - purpose_connection_type
    - best_supporting_evidence_ids
    - contradictory_or_weak_evidence_ids
    - needs_human_review
    - human_review_reason
    """
    question_id = _safe_str(raw_result.get("question_id", ""))

    if expected_question_id:
        if not question_id:
            question_id = expected_question_id

    llm_score = clip_score(raw_result.get("llm_score_0_5", 0.0))

    purpose_connection_type = _safe_str(
        raw_result.get("purpose_connection_type", "insufficient_evidence")
    )

    allowed_connection_types = {
        "explicit_alignment",
        "reasonable_semantic_alignment",
        "weak_or_incidental_alignment",
        "no_alignment",
        "insufficient_evidence",
    }

    if purpose_connection_type not in allowed_connection_types:
        purpose_connection_type = "insufficient_evidence"

    best_ids = _ensure_list(raw_result.get("best_supporting_evidence_ids", []))
    weak_ids = _ensure_list(raw_result.get("contradictory_or_weak_evidence_ids", []))

    needs_review = _safe_bool(raw_result.get("needs_human_review", False))
    review_reason = _safe_str(raw_result.get("human_review_reason", ""))

    if llm_score <= 1 and not best_ids:
        needs_review = True
        if not review_reason:
            review_reason = "low_score_no_supporting_evidence"

    if purpose_connection_type == "insufficient_evidence":
        needs_review = True
        if not review_reason:
            review_reason = "insufficient_evidence"

    return {
        "question_id": question_id,
        "llm_score_0_5": llm_score,
        "score_label": _safe_str(raw_result.get("score_label", "")),
        "alignment_summary": _safe_str(raw_result.get("alignment_summary", "")),
        "purpose_connection_type": purpose_connection_type,
        "best_supporting_evidence_ids": best_ids,
        "contradictory_or_weak_evidence_ids": weak_ids,
        "needs_human_review": needs_review,
        "human_review_reason": review_reason,
        "raw_llm_result": raw_result,
    }


# ============================================================
# LLM call wrapper
# ============================================================

def run_sa_question_llm(
    prompt: str,
    question_id: str,
    llm_client: LLMClient,
) -> Dict[str, Any]:
    """
    Run LLM scoring for one SA question.
    """
    if llm_client is None:
        raise ValueError("llm_client must be provided")

    response_text = llm_client(prompt)
    raw_json = extract_json_from_response(response_text)

    return normalize_sa_llm_result(
        raw_json,
        expected_question_id=question_id,
    )


def run_sa_batch_llm(
    prompts: Dict[str, str],
    llm_client: LLMClient,
) -> Dict[str, Dict[str, Any]]:
    """
    Run LLM scoring for all SA question prompts.
    """
    results = {}

    for question_id, prompt in prompts.items():
        try:
            results[question_id] = run_sa_question_llm(
                prompt=prompt,
                question_id=question_id,
                llm_client=llm_client,
            )
        except Exception as exc:
            results[question_id] = build_failed_llm_result(
                question_id=question_id,
                error=exc,
            )

    return results


def build_failed_llm_result(
    question_id: str,
    error: Exception,
) -> Dict[str, Any]:
    """
    Stable fallback result when LLM call or parsing fails.
    """
    return {
        "question_id": question_id,
        "llm_score_0_5": 0.0,
        "score_label": "LLM scoring failed",
        "alignment_summary": "",
        "purpose_connection_type": "insufficient_evidence",
        "best_supporting_evidence_ids": [],
        "contradictory_or_weak_evidence_ids": [],
        "needs_human_review": True,
        "human_review_reason": f"llm_error: {type(error).__name__}: {str(error)}",
        "raw_llm_result": {},
    }
