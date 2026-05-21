# src/strategy_alignment/sa_prompt_builder.py

"""
Prompt builder for Strategy & Source Alignment.

SA depends on PA output. The LLM must compare strategy / operational evidence
against the extracted purpose reference, not evaluate strategy quality in general.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from .sa_config import SA_PROMPT_CORE_INSTRUCTION
from .sa_rubric import get_sa_question


# ============================================================
# Helpers
# ============================================================

def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _format_list(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return "; ".join([str(x).strip() for x in value if str(x).strip()])

    text = str(value).strip()
    return text


def _truncate_text(text: str, max_chars: int = 1800) -> str:
    text = _safe_str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


# ============================================================
# Purpose reference formatting
# ============================================================

def format_purpose_reference(purpose_reference: Dict[str, Any]) -> str:
    """
    Format PA extracted purpose as the comparison anchor for SA.
    """
    purpose_statement = _safe_str(
        purpose_reference.get("extracted_purpose", "")
        or purpose_reference.get("purpose_statement_normalized", "")
    )
    purpose_raw = _safe_str(
        purpose_reference.get("purpose_statement_raw", "")
    )
    stakeholders = _format_list(
        purpose_reference.get("served_stakeholders", "")
    )
    intended_impact = _format_list(
        purpose_reference.get("intended_impact", "")
    )
    reason = _safe_str(
        purpose_reference.get("reason_for_existence", "")
    )
    confidence = _safe_float(
        purpose_reference.get("purpose_confidence_0_1", 0.0)
    )

    return f"""
Extracted Purpose Reference from PA Module:
- Extracted purpose: {purpose_statement}
- Normalized purpose statement: {_safe_str(purpose_reference.get("purpose_statement_normalized", ""))}
- Raw purpose statement: {purpose_raw}
- Served stakeholders: {stakeholders}
- Intended impact: {intended_impact}
- Reason for existence: {reason}
- Purpose extraction confidence: {confidence:.2f}
""".strip()


# ============================================================
# Rubric formatting
# ============================================================

def format_question_rubric(question_id: str) -> str:
    """
    Format one SA question rubric into prompt-ready text.
    """
    question = get_sa_question(question_id)
    descriptions = question["score_descriptions"]

    score_lines = []
    for score in range(0, 6):
        score_lines.append(f"Score {score}: {descriptions[score]}")

    return f"""
Question ID: {question["question_id"]}
Question Name: {question["question_name"]}
Question: {question["question_text"]}

Rubric:
{chr(10).join(score_lines)}
""".strip()


# ============================================================
# Evidence formatting
# ============================================================

def format_evidence_rows(
    evidence_df: pd.DataFrame,
    max_evidence: Optional[int] = None,
    max_chars_per_evidence: int = 1800,
) -> str:
    """
    Format candidate evidence rows for prompt.
    """
    if evidence_df.empty:
        return "No candidate evidence was retrieved."

    working = evidence_df.copy()

    if "rank" in working.columns:
        working = working.sort_values("rank")

    if max_evidence is not None:
        working = working.head(max_evidence)

    blocks = []

    for idx, row in working.iterrows():
        evidence_id = _safe_str(row.get("chunk_id", f"row_{idx}"))
        year = _safe_str(row.get("year", ""))
        source = _safe_str(row.get("source", ""))
        section = _safe_str(row.get("section", ""))
        rank = _safe_str(row.get("rank", ""))
        base_score = _safe_float(row.get("base_evidence_score_0_1", 0.0))
        quality_factor = _safe_float(row.get("evidence_quality_factor", 0.70))
        overlap_factor = _safe_float(row.get("overlap_factor", 1.0))
        text = _truncate_text(
            _safe_str(row.get("text_clean", "")),
            max_chars=max_chars_per_evidence,
        )

        block = f"""
Evidence {rank}
- evidence_id: {evidence_id}
- year: {year}
- source: {source}
- section: {section}
- base_evidence_score_0_1: {base_score:.3f}
- evidence_quality_factor: {quality_factor:.3f}
- overlap_factor: {overlap_factor:.3f}
- text:
{text}
""".strip()

        blocks.append(block)

    return "\n\n".join(blocks)


# ============================================================
# JSON schema instruction
# ============================================================

def get_sa_llm_json_schema_instruction() -> str:
    """
    Fixed JSON schema for SA LLM scoring.
    """
    schema = {
        "question_id": "SA_Q1 or SA_Q2",
        "llm_score_0_5": "number from 0 to 5",
        "score_label": "short label explaining score level",
        "alignment_summary": "brief explanation of how evidence aligns or does not align with stated purpose",
        "purpose_connection_type": (
            "one of: explicit_alignment, reasonable_semantic_alignment, "
            "weak_or_incidental_alignment, no_alignment, insufficient_evidence"
        ),
        "best_supporting_evidence_ids": ["chunk_id_1", "chunk_id_2"],
        "contradictory_or_weak_evidence_ids": ["chunk_id_3"],
        "needs_human_review": "boolean",
        "human_review_reason": "string",
    }

    return f"""
Return only valid JSON. Do not include markdown.

Required JSON schema:
{json.dumps(schema, indent=2)}
""".strip()


# ============================================================
# Main prompt builder
# ============================================================

def build_sa_question_prompt(
    company: str,
    question_id: str,
    purpose_reference: Dict[str, Any],
    evidence_df: pd.DataFrame,
) -> str:
    """
    Build LLM prompt for one company and one SA question.

    This prompt asks the LLM to score one SA question using retrieved evidence.
    The model must compare the evidence against the PA-extracted purpose reference.
    """
    purpose_block = format_purpose_reference(purpose_reference)
    rubric_block = format_question_rubric(question_id)
    evidence_block = format_evidence_rows(evidence_df)
    json_schema_instruction = get_sa_llm_json_schema_instruction()

    prompt = f"""
You are evaluating Strategy & Source Alignment for a purpose-driven company scoring system.

Company:
{company}

Core instruction:
{SA_PROMPT_CORE_INSTRUCTION}

Purpose reference:
{purpose_block}

Scoring rubric:
{rubric_block}

Candidate evidence:
{evidence_block}

Scoring rules:
1. Use only the evidence provided above.
2. The extracted purpose reference is the comparison anchor.
3. Do not reward generic strategy, growth, innovation, market leadership, or operational excellence unless it is connected to the stated purpose.
4. Exact wording is not required. Semantic alignment is acceptable.
5. If evidence is relevant to strategy but does not connect to purpose, give a low or moderate score.
6. If evidence is too generic, boilerplate, or only financial-performance oriented, do not score above 2.
7. If public disclosures explicitly justify R&D, capital, resource allocation, operations, product decisions, supplier choices, or market entry through the stated purpose, score higher.
8. If evidence is insufficient, set needs_human_review to true.

{json_schema_instruction}
""".strip()

    return prompt


def build_sa_batch_prompts(
    company: str,
    purpose_reference: Dict[str, Any],
    candidate_map: Dict[str, pd.DataFrame],
) -> Dict[str, str]:
    """
    Build prompts for all SA questions for one company.
    """
    prompts = {}

    for question_id, evidence_df in candidate_map.items():
        prompts[question_id] = build_sa_question_prompt(
            company=company,
            question_id=question_id,
            purpose_reference=purpose_reference,
            evidence_df=evidence_df,
        )

    return prompts
