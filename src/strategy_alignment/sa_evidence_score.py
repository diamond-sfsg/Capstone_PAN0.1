# src/strategy_alignment/sa_evidence_score.py

"""
Evidence scoring functions for Strategy & Source Alignment.

The evidence score is not the final SA score.
It only adjusts LLM rubric scoring mildly.

Main formula:
base_evidence_score_0_1 =
    0.40 * keyword_relevance
  + 0.35 * context_completeness
  + 0.25 * rag_similarity

evidence_quality_factor =
    0.70 + 0.30 * base_evidence_score_0_1

evidence_contribution_0_5 =
    llm_score_0_5 * evidence_quality_factor * overlap_factor
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .sa_config import (
    EVIDENCE_BASE_SCORE_WEIGHTS,
    EVIDENCE_QUALITY_BASE,
    EVIDENCE_QUALITY_SPAN,
    SA_QUESTION_KEYWORDS,
    SA_OVERLAP_FACTOR,
    MIN_LLM_SCORE,
    MAX_LLM_SCORE,
)


# ============================================================
# Basic helpers
# ============================================================

def clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        if value is None or math.isnan(float(value)):
            return lower
        return max(lower, min(upper, float(value)))
    except Exception:
        return lower


def normalize_text(text: Any) -> str:
    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def tokenize(text: Any) -> List[str]:
    normalized = normalize_text(text)
    return re.findall(r"[a-zA-Z][a-zA-Z\-']+", normalized)


def phrase_count(text: str, phrase: str) -> int:
    """
    Count phrase matches using a conservative regex boundary.

    This avoids most substring false positives while still allowing phrases.
    """
    if not text or not phrase:
        return 0

    text_norm = normalize_text(text)
    phrase_norm = normalize_text(phrase)

    if not phrase_norm:
        return 0

    pattern = r"(?<![a-zA-Z])" + re.escape(phrase_norm) + r"(?![a-zA-Z])"
    return len(re.findall(pattern, text_norm))


def any_phrase_match(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase_count(text, phrase) > 0 for phrase in phrases)


def count_unique_phrase_hits(text: str, phrases: Iterable[str]) -> int:
    return sum(1 for phrase in phrases if phrase_count(text, phrase) > 0)


# ============================================================
# Purpose reference helpers
# ============================================================

def extract_purpose_terms(purpose_reference: Optional[Dict[str, Any]]) -> List[str]:
    """
    Extract meaningful terms from PA purpose reference.

    These terms are not used to override LLM judgment.
    They only help determine whether the evidence is contextually connected
    to the stated purpose.
    """
    if not purpose_reference:
        return []

    values = []

    for key in [
        "extracted_purpose",
        "purpose_statement_normalized",
        "purpose_statement_raw",
        "reason_for_existence",
    ]:
        value = purpose_reference.get(key, "")
        if isinstance(value, str) and value.strip():
            values.append(value)

    for key in ["served_stakeholders", "intended_impact"]:
        value = purpose_reference.get(key, [])
        if isinstance(value, list):
            values.extend([str(x) for x in value if str(x).strip()])
        elif isinstance(value, str) and value.strip():
            values.append(value)

    tokens = []
    for value in values:
        tokens.extend(tokenize(value))

    stopwords = {
        "the", "and", "for", "with", "that", "this", "our", "their", "its",
        "from", "into", "are", "was", "were", "has", "have", "had", "will",
        "can", "may", "all", "more", "new", "business", "company", "purpose",
        "mission", "vision", "value", "values",
    }

    useful = [
        token
        for token in tokens
        if len(token) >= 4 and token not in stopwords
    ]

    # Deduplicate while preserving order.
    seen = set()
    output = []
    for token in useful:
        if token not in seen:
            seen.add(token)
            output.append(token)

    return output[:30]


def purpose_term_overlap_score(
    text: str,
    purpose_reference: Optional[Dict[str, Any]],
) -> float:
    """
    Lightweight lexical overlap between evidence and extracted purpose.

    This is intentionally weak. Semantic alignment is mainly judged by LLM.
    """
    terms = extract_purpose_terms(purpose_reference)
    if not terms:
        return 0.0

    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0

    hits = sum(1 for term in terms if term in text_tokens)

    # Cap at 5 meaningful hits.
    return clip(hits / 5.0, 0.0, 1.0)


# ============================================================
# Keyword relevance
# ============================================================

def keyword_relevance_score(text: str, question_id: str) -> float:
    """
    Compute question-specific keyword relevance.

    Core terms carry the most weight.
    Support terms broaden recall.
    Purpose-alignment terms detect explicit linkage to purpose.
    """
    if question_id not in SA_QUESTION_KEYWORDS:
        raise KeyError(f"Unknown SA question_id: {question_id}")

    config = SA_QUESTION_KEYWORDS[question_id]

    core_terms = config.get("core", [])
    support_terms = config.get("support", [])
    alignment_terms = config.get("purpose_alignment_terms", [])

    core_hits = count_unique_phrase_hits(text, core_terms)
    support_hits = count_unique_phrase_hits(text, support_terms)
    alignment_hits = count_unique_phrase_hits(text, alignment_terms)

    core_score = clip(core_hits / 3.0, 0.0, 1.0)
    support_score = clip(support_hits / 4.0, 0.0, 1.0)
    alignment_score = clip(alignment_hits / 2.0, 0.0, 1.0)

    return clip(
        0.50 * core_score
        + 0.25 * support_score
        + 0.25 * alignment_score,
        0.0,
        1.0,
    )


# ============================================================
# Context completeness
# ============================================================

def context_completeness_score(
    text: str,
    question_id: str,
    purpose_reference: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Estimate whether the chunk contains enough context to be scored by LLM.

    SA_Q1 looks for:
    - resource allocation signal
    - strategic/investment target
    - purpose linkage
    - explicit rationale or justification

    SA_Q2 looks for:
    - operational decision signal
    - operating area / function
    - purpose linkage
    - governance / tracking / framework signal
    """
    text_norm = normalize_text(text)

    purpose_overlap = purpose_term_overlap_score(text_norm, purpose_reference)

    if question_id == "SA_Q1":
        allocation_signal = any_phrase_match(
            text_norm,
            [
                "r&d", "research and development", "capital expenditure",
                "capex", "capital allocation", "investment", "investments",
                "resource allocation", "growth initiative", "strategic investment",
            ],
        )

        strategic_target_signal = any_phrase_match(
            text_norm,
            [
                "innovation", "technology", "pipeline", "platform", "capacity",
                "infrastructure", "product", "portfolio", "acquisition",
                "commercialization", "expansion",
            ],
        )

        purpose_link_signal = any_phrase_match(
            text_norm,
            [
                "purpose", "mission", "vision", "values", "impact",
                "stakeholder", "customer", "community", "patient",
                "sustainability", "long-term value",
            ],
        ) or purpose_overlap >= 0.20

        rationale_signal = any_phrase_match(
            text_norm,
            [
                "because", "in order to", "so that", "designed to",
                "intended to", "enables us to", "supports our",
                "aligned with", "consistent with", "driven by",
            ],
        )

        components = [
            allocation_signal,
            strategic_target_signal,
            purpose_link_signal,
            rationale_signal,
        ]

    elif question_id == "SA_Q2":
        operation_signal = any_phrase_match(
            text_norm,
            [
                "operation", "operational", "decision-making", "decision making",
                "supplier", "supply chain", "product development",
                "market entry", "go-to-market", "operating model",
                "business unit", "procurement", "manufacturing",
            ],
        )

        function_signal = any_phrase_match(
            text_norm,
            [
                "supplier", "vendor", "partner", "product", "market",
                "customer", "distribution", "quality", "service",
                "employee", "training", "compliance", "business unit",
            ],
        )

        purpose_link_signal = any_phrase_match(
            text_norm,
            [
                "purpose", "mission", "vision", "values", "impact",
                "stakeholder", "customer", "community", "patient",
                "sustainability", "long-term value",
            ],
        ) or purpose_overlap >= 0.20

        framework_signal = any_phrase_match(
            text_norm,
            [
                "framework", "policy", "criteria", "standard", "tracked",
                "reported", "measured", "evaluated", "governance",
                "accountability", "independently", "verified",
            ],
        )

        components = [
            operation_signal,
            function_signal,
            purpose_link_signal,
            framework_signal,
        ]

    else:
        raise KeyError(f"Unknown SA question_id: {question_id}")

    binary_score = sum(1 for item in components if item) / len(components)

    # Purpose lexical overlap provides a small boost but cannot dominate.
    return clip(0.85 * binary_score + 0.15 * purpose_overlap, 0.0, 1.0)


# ============================================================
# Base evidence score and quality factor
# ============================================================

def compute_base_evidence_score(
    keyword_relevance: float,
    context_completeness: float,
    rag_similarity: float,
) -> float:
    return clip(
        EVIDENCE_BASE_SCORE_WEIGHTS["keyword_relevance"] * clip(keyword_relevance)
        + EVIDENCE_BASE_SCORE_WEIGHTS["context_completeness"] * clip(context_completeness)
        + EVIDENCE_BASE_SCORE_WEIGHTS["rag_similarity"] * clip(rag_similarity),
        0.0,
        1.0,
    )


def compute_evidence_quality_factor(base_evidence_score_0_1: float) -> float:
    """
    Mild adjustment factor in [0.70, 1.00].
    """
    return clip(
        EVIDENCE_QUALITY_BASE
        + EVIDENCE_QUALITY_SPAN * clip(base_evidence_score_0_1),
        EVIDENCE_QUALITY_BASE,
        EVIDENCE_QUALITY_BASE + EVIDENCE_QUALITY_SPAN,
    )


# ============================================================
# Overlap factor
# ============================================================

def infer_overlap_label(row: pd.Series) -> str:
    """
    Infer an overlap label from available duplicate fields.

    This function intentionally keeps cross-year recurrence unpenalized.
    """
    similarity_scope = str(row.get("similarity_scope", "")).lower()
    duplicate_group = str(row.get("duplicate_group", "")).lower()

    combined = f"{similarity_scope} {duplicate_group}"

    if "same_year_exact" in combined or "exact_same_year" in combined:
        return "same_year_exact_duplicate"

    if "same_year_near" in combined or "near_same_year" in combined:
        return "same_year_near_duplicate"

    if "cross_year" in combined or "recurring" in combined:
        return "cross_year_recurring"

    return "first_use"


def compute_overlap_factor(
    overlap_label: Optional[str] = None,
    row: Optional[pd.Series] = None,
) -> float:
    if overlap_label is None and row is not None:
        overlap_label = infer_overlap_label(row)

    if not overlap_label:
        overlap_label = "unknown"

    return float(SA_OVERLAP_FACTOR.get(overlap_label, SA_OVERLAP_FACTOR["unknown"]))


# ============================================================
# Evidence contribution
# ============================================================

def compute_evidence_contribution(
    llm_score_0_5: float,
    evidence_quality_factor: float,
    overlap_factor: float,
) -> float:
    llm_score = clip(llm_score_0_5, MIN_LLM_SCORE, MAX_LLM_SCORE)

    contribution = (
        llm_score
        * clip(evidence_quality_factor, 0.0, 1.0)
        * clip(overlap_factor, 0.0, 1.0)
    )

    return clip(contribution, 0.0, 5.0)


# ============================================================
# DataFrame scoring
# ============================================================

def add_evidence_scores(
    df: pd.DataFrame,
    question_id: str,
    rag_scores: Optional[List[float]] = None,
    purpose_reference: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Add SA evidence scoring columns to a candidate DataFrame.

    LLM score is not available at this stage, so this function only computes:
    - keyword_relevance
    - context_completeness
    - rag_similarity
    - base_evidence_score_0_1
    - evidence_quality_factor
    - overlap_factor
    """
    if "text_clean" not in df.columns:
        raise ValueError("Input DataFrame must contain column: text_clean")

    out = df.copy()

    if rag_scores is None:
        rag_scores = [0.0] * len(out)

    if len(rag_scores) != len(out):
        raise ValueError(
            f"rag_scores length mismatch: expected {len(out)}, got {len(rag_scores)}"
        )

    out["question_id"] = question_id

    out["keyword_relevance"] = out["text_clean"].map(
        lambda text: keyword_relevance_score(text, question_id)
    )

    out["context_completeness"] = out["text_clean"].map(
        lambda text: context_completeness_score(
            text,
            question_id,
            purpose_reference=purpose_reference,
        )
    )

    out["rag_similarity"] = [clip(x, 0.0, 1.0) for x in rag_scores]

    out["base_evidence_score_0_1"] = out.apply(
        lambda row: compute_base_evidence_score(
            row["keyword_relevance"],
            row["context_completeness"],
            row["rag_similarity"],
        ),
        axis=1,
    )

    out["evidence_quality_factor"] = out["base_evidence_score_0_1"].map(
        compute_evidence_quality_factor
    )

    out["overlap_label"] = out.apply(infer_overlap_label, axis=1)
    out["overlap_factor"] = out["overlap_label"].map(
        lambda label: compute_overlap_factor(overlap_label=label)
    )

    return out
