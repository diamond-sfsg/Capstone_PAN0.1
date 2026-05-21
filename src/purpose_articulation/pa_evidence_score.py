from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

from purpose_articulation.pa_config import (
    BOILERPLATE_TERMS,
    CONTEXT_WEIGHT,
    DEFAULT_SOURCE_PRIOR,
    EVIDENCE_QUALITY_MIN,
    EVIDENCE_QUALITY_RANGE,
    GENERIC_BRANDING_PHRASES,
    KEYWORD_WEIGHT,
    PA_CORE_TERMS,
    PA_SOURCE_PRIOR_Q1_Q2,
    PA_SOURCE_PRIOR_Q3,
    PA_SUPPORT_TERMS,
    PA_TONE_BONUS_MAX,
    PA_TONE_BONUS_MIN,
    RAG_WEIGHT,
)


TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z\-]+\b")


def clamp(value: float | int | None, lower: float = 0.0, upper: float = 1.0) -> float:
    """
    Clamp numeric value into [lower, upper].
    """
    if value is None:
        return lower

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return lower

    if pd.isna(value_float):
        return lower

    return max(lower, min(upper, value_float))


def normalized_text(text: str | None) -> str:
    """
    Lowercase and normalize whitespace.
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str | None) -> list[str]:
    """
    Simple English tokenization for lexical matching.
    """
    if text is None or pd.isna(text):
        return []

    return TOKEN_RE.findall(str(text).lower())


def _phrase_or_token_hit(term: str, norm_text: str, token_counts: Counter) -> bool:
    term_norm = term.lower().strip()

    if not term_norm:
        return False

    if " " in term_norm:
        return term_norm in norm_text

    return token_counts.get(term_norm, 0) > 0


def keyword_relevance_score(text: str | None) -> float:
    """
    Compute 0-1 PA keyword relevance.

    This is not the final PA score.
    It is used as part of evidence ranking / evidence quality control.
    """
    norm = normalized_text(text)
    tokens = tokenize(text)

    if not tokens:
        return 0.0

    token_counts = Counter(tokens)

    core_unique_hits = 0
    support_unique_hits = 0
    core_raw_hits = 0
    support_raw_hits = 0
    phrase_hits = 0.0

    for term in PA_CORE_TERMS:
        term_norm = term.lower().strip()

        if not term_norm:
            continue

        if " " in term_norm:
            if term_norm in norm:
                phrase_hits += 1.0
                core_unique_hits += 1
        else:
            count = token_counts.get(term_norm, 0)
            if count > 0:
                core_unique_hits += 1
                core_raw_hits += count

    for term in PA_SUPPORT_TERMS:
        term_norm = term.lower().strip()

        if not term_norm:
            continue

        if " " in term_norm:
            if term_norm in norm:
                phrase_hits += 0.5
                support_unique_hits += 1
        else:
            count = token_counts.get(term_norm, 0)
            if count > 0:
                support_unique_hits += 1
                support_raw_hits += count

    raw_score = (
        2.5 * phrase_hits
        + 1.5 * core_unique_hits
        + 0.8 * support_unique_hits
        + 0.15 * core_raw_hits
        + 0.05 * support_raw_hits
    )

    # Smooth saturation so repeated generic terms do not dominate.
    score = 1 - math.exp(-raw_score / 6.0)

    return clamp(score, 0.0, 1.0)


def context_completeness_score(text: str | None, section: str | None = "") -> float:
    """
    Compute 0-1 score for whether the text is readable, complete, and usable.

    This roughly checks:
    - not too short
    - not too long / noisy
    - enough alphabetic content
    - sentence-like structure
    - not obvious boilerplate
    - section context is not obviously bad
    """
    if text is None or pd.isna(text):
        return 0.0

    raw = str(text).strip()

    if not raw:
        return 0.0

    norm = normalized_text(raw)
    tokens = tokenize(raw)

    token_count = len(tokens)

    if token_count < 8:
        return 0.05

    score = 0.50

    # Length quality.
    if 40 <= token_count <= 350:
        score += 0.20
    elif 20 <= token_count < 40:
        score += 0.10
    elif 350 < token_count <= 500:
        score += 0.02
    elif token_count > 500:
        score -= 0.18

    # Alphabetic ratio.
    alpha_chars = sum(ch.isalpha() for ch in raw)
    total_chars = max(len(raw), 1)
    alpha_ratio = alpha_chars / total_chars

    if alpha_ratio >= 0.65:
        score += 0.10
    elif alpha_ratio >= 0.50:
        score += 0.03
    else:
        score -= 0.20

    # Sentence structure.
    if any(punct in raw for punct in [".", ";", ":"]):
        score += 0.07

    # Corporate/purpose subject context.
    subject_markers = [
        "we ",
        "our ",
        "company ",
        "customers",
        "communities",
        "patients",
        "people",
        "stakeholders",
        "society",
    ]

    if any(marker in norm for marker in subject_markers):
        score += 0.08

    # Boilerplate penalty.
    if any(term in norm for term in BOILERPLATE_TERMS):
        score -= 0.25

    # Section-based adjustment.
    section_norm = normalized_text(section)

    if section_norm:
        helpful_sections = [
            "mission",
            "purpose",
            "about",
            "values",
            "overview",
            "strategy",
            "business",
            "letter",
            "sustainability",
        ]

        weak_sections = [
            "risk factor",
            "legal",
            "table of contents",
            "signature",
            "exhibit",
        ]

        if any(term in section_norm for term in helpful_sections):
            score += 0.08

        if any(term in section_norm for term in weak_sections):
            score -= 0.08

    return clamp(score, 0.0, 1.0)


def generic_branding_penalty(text: str | None) -> float:
    """
    Detect generic branding / slogan language.

    Returns 0-1. Higher means more generic branding risk.
    """
    norm = normalized_text(text)

    if not norm:
        return 0.0

    hits = sum(1 for phrase in GENERIC_BRANDING_PHRASES if phrase in norm)

    return clamp(hits / 3.0, 0.0, 1.0)


def rule_based_pa_tone_bonus(text: str | None) -> float:
    """
    Small rule-based fallback tone bonus.

    The LLM will also return pa_tone_bonus. This function provides:
    - fallback if LLM is unavailable
    - diagnostic signal
    """
    norm = normalized_text(text)

    if not norm:
        return 0.0

    bonus = 0.0

    explicit_patterns = [
        "our purpose is",
        "our mission is",
        "we exist to",
        "exists to",
        "our purpose",
        "our mission",
        "our vision is",
    ]

    if any(pattern in norm for pattern in explicit_patterns):
        bonus += 0.10

    stakeholder_terms = [
        "customers",
        "patients",
        "communities",
        "people",
        "stakeholders",
        "society",
        "employees",
        "families",
    ]

    if any(term in norm for term in stakeholder_terms):
        bonus += 0.06

    impact_terms = [
        "impact",
        "improve",
        "enable",
        "access",
        "affordable",
        "safe",
        "healthy",
        "sustainable",
        "opportunity",
        "quality of life",
    ]

    if any(term in norm for term in impact_terms):
        bonus += 0.06

    if "beyond profit" in norm or "not only" in norm or "more than profit" in norm:
        bonus += 0.04

    # Penalize generic slogan-like language.
    bonus -= 0.08 * generic_branding_penalty(norm)

    return clamp(bonus, PA_TONE_BONUS_MIN, PA_TONE_BONUS_MAX)


def get_source_prior(question_id: str, normalized_source: str | None) -> float:
    """
    Get source prior for a given PA question.

    Q1/Q2:
        official_web boosted.

    Q3:
        edgar boosted because the question tests whether purpose goes beyond branding.
    """
    source = str(normalized_source or "unknown").strip().lower()

    if not source:
        source = "unknown"

    if question_id in {"PA_Q1", "PA_Q2"}:
        return PA_SOURCE_PRIOR_Q1_Q2.get(source, DEFAULT_SOURCE_PRIOR)

    if question_id == "PA_Q3":
        return PA_SOURCE_PRIOR_Q3.get(source, DEFAULT_SOURCE_PRIOR)

    return DEFAULT_SOURCE_PRIOR


def compute_base_evidence_score(
    keyword_score: float | int | None,
    context_score: float | int | None,
    rag_similarity: float | int | None,
) -> float:
    """
    Combine evidence relevance / readability signals into one 0-1 score.
    """
    score = (
        KEYWORD_WEIGHT * clamp(keyword_score)
        + CONTEXT_WEIGHT * clamp(context_score)
        + RAG_WEIGHT * clamp(rag_similarity)
    )

    return clamp(score, 0.0, 1.0)


def evidence_quality_factor(base_evidence_score: float | int | None) -> float:
    """
    Convert base evidence score into mild multiplicative factor.

    Example:
        base = 0.0 -> factor = 0.70
        base = 0.5 -> factor = 0.85
        base = 1.0 -> factor = 1.00
    """
    return EVIDENCE_QUALITY_MIN + EVIDENCE_QUALITY_RANGE * clamp(base_evidence_score)


def compute_q3_evidence_set_quality(evidence_df: pd.DataFrame) -> dict:
    """
    Compute Q3 evidence-set quality based on:
    - source diversity
    - formal document presence
    - strategic / operational section presence

    This is not the LLM score. It is a mild evidence-set quality adjustment.
    """
    if evidence_df is None or evidence_df.empty:
        return {
            "source_diversity": 0.0,
            "formal_document_presence": 0.0,
            "strategic_section_presence": 0.0,
            "evidence_set_quality": 0.0,
            "evidence_set_quality_factor": 0.75,
        }

    sources = set(
        evidence_df.get("normalized_source", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .str.lower()
        .tolist()
    )

    source_diversity = clamp(len(sources) / 3.0, 0.0, 1.0)

    formal_document_presence = 1.0 if "edgar" in sources else 0.0

    section_text = " ".join(
        evidence_df.get("section", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.lower()
        .tolist()
    )

    strategic_terms = [
        "strategy",
        "strategic",
        "business",
        "operations",
        "operating",
        "capital",
        "allocation",
        "risk",
        "priorities",
        "performance",
        "management",
        "annual report",
        "sustainability",
    ]

    strategic_section_presence = (
        1.0 if any(term in section_text for term in strategic_terms) else 0.0
    )

    evidence_set_quality = clamp(
        0.40 * source_diversity
        + 0.30 * formal_document_presence
        + 0.30 * strategic_section_presence,
        0.0,
        1.0,
    )

    evidence_set_quality_factor_value = 0.75 + 0.25 * evidence_set_quality

    return {
        "source_diversity": source_diversity,
        "formal_document_presence": formal_document_presence,
        "strategic_section_presence": strategic_section_presence,
        "evidence_set_quality": evidence_set_quality,
        "evidence_set_quality_factor": evidence_set_quality_factor_value,
    }
