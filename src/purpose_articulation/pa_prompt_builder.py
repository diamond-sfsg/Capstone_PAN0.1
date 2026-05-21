from __future__ import annotations

import json

import pandas as pd

from purpose_articulation.pa_rubric import PA_QUESTIONS, format_rubric


def _safe_year(value) -> int | None:
    if value is None or pd.isna(value):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_pa_evidence_prompt(question_id: str, evidence_row: dict) -> str:
    """
    Build prompt for Q1/Q2 evidence-level scoring.
    """
    if question_id not in PA_QUESTIONS:
        raise ValueError(f"Unknown PA question_id: {question_id}")

    question = PA_QUESTIONS[question_id]
    rubric_text = format_rubric(question_id)

    evidence_payload = {
        "company": evidence_row.get("company"),
        "question_id": question_id,
        "question_name": question.name,
        "source": evidence_row.get("source"),
        "normalized_source": evidence_row.get("normalized_source"),
        "section": evidence_row.get("section"),
        "subsection": evidence_row.get("subsection"),
        "year": _safe_year(evidence_row.get("year")),
        "evidence_text": evidence_row.get("text_clean"),
    }

    prompt = f"""
You are evaluating Purpose Articulation evidence for a company.

Task:
Score how strongly the evidence supports the following rubric question.

Question:
{question.question}

Rubric:
{rubric_text}

Important rules:
- Use only the evidence text and metadata provided.
- Do not use outside knowledge.
- Give a score from 0 to 5.
- The score should evaluate support for this specific question only.
- Extract the company's stated or implied purpose from the evidence in one concise sentence when possible.
- If the evidence does not contain a purpose statement or purpose-like claim, set extracted_purpose to an empty string.
- The extracted_purpose field is for evidence retention only and must not influence the score.
- Also provide a small PA tone bonus from 0.00 to 0.30.
- PA tone bonus rewards explicit, firm, clear, specific purpose language.
- Do not give tone bonus for generic marketing slogans.
- If the evidence is incomplete, boilerplate, or mostly marketing language, reflect that in the score and risk flags.

Evidence payload:
{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}

Return valid JSON only with this exact schema:
{{
  "llm_score_0_5": 0,
  "pa_tone_bonus": 0.0,
  "extracted_purpose": "one concise sentence, or empty string if none is present",
  "support_level": "none | weak | moderate | strong | very_strong",
  "reason": "brief explanation",
  "risk_flags": ["generic_language", "marketing_only", "incomplete_context"]
}}
""".strip()

    return prompt


def build_pa_evidence_set_prompt(
    question_id: str,
    evidence_set_df: pd.DataFrame,
    set_quality: dict,
) -> str:
    """
    Build prompt for Q3 evidence-set scoring.
    """
    if question_id not in PA_QUESTIONS:
        raise ValueError(f"Unknown PA question_id: {question_id}")

    question = PA_QUESTIONS[question_id]
    rubric_text = format_rubric(question_id)

    evidence_items = []

    if evidence_set_df is not None and not evidence_set_df.empty:
        for _, row in evidence_set_df.iterrows():
            evidence_items.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "source": row.get("source"),
                    "normalized_source": row.get("normalized_source"),
                    "section": row.get("section"),
                    "subsection": row.get("subsection"),
                    "year": _safe_year(row.get("year")),
                    "evidence_text": row.get("text_clean"),
                }
            )

    evidence_payload = {
        "question_id": question_id,
        "question_name": question.name,
        "evidence_set_quality": set_quality,
        "evidence_items": evidence_items,
    }

    prompt = f"""
You are evaluating Purpose Articulation at the evidence-set level.

Task:
Score whether the company's purpose is positioned within a strategic or operational context,
rather than appearing primarily in marketing or promotional materials.

Question:
{question.question}

Rubric:
{rubric_text}

Important rules:
- Use only the provided evidence set and metadata.
- Do not use outside knowledge.
- This is an evidence-set judgment, not a single-evidence judgment.
- Pay attention to source diversity, formal documents, strategic/operational sections, and whether marketing language dominates.
- A company should not receive a high score if purpose only appears in homepage, slogans, brand campaigns, or promotional language.
- Return a score from 0 to 5.
- Extract the clearest company purpose conveyed by the evidence set in one concise sentence when possible.
- If the evidence set does not contain a purpose statement or purpose-like claim, set extracted_purpose to an empty string.
- The extracted_purpose field is for evidence retention only and must not influence the score.

Evidence set payload:
{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}

Return valid JSON only with this exact schema:
{{
  "llm_set_score_0_5": 0,
  "extracted_purpose": "one concise sentence, or empty string if none is present",
  "support_level": "none | weak | moderate | strong | very_strong",
  "reason": "brief explanation",
  "risk_flags": ["marketing_only", "insufficient_formal_documents", "weak_strategy_context"]
}}
""".strip()

    return prompt
