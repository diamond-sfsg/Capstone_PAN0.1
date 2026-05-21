# src/history_consistency/hc_prompt_builder.py

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from history_consistency.hc_config import (
    DIMENSION_LABEL,
    HC_QUESTION_ID,
    HC_QUESTION_TEXT,
    HC_RUBRIC,
    PROMPT_OUTPUT_SCHEMA,
)


@dataclass(frozen=True)
class HCEvidenceRecord:
    """
    Prompt-ready evidence record.
    """

    chunk_id: str
    year: Optional[int]
    source: str
    section: str
    text: str
    rank_score: Optional[float]
    base_evidence_score: Optional[float]


def safe_int(value: object) -> Optional[int]:
    """
    Safely convert value to int.
    """
    if value is None or pd.isna(value):
        return None

    try:
        return int(value)
    except Exception:
        return None


def safe_float(value: object) -> Optional[float]:
    """
    Safely convert value to float.
    """
    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def truncate_text(text: str, max_chars: int = 1400) -> str:
    """
    Truncate evidence text to keep prompt length controlled.
    """
    text = str(text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + " ... [TRUNCATED]"


def evidence_dataframe_to_records(
    evidence_df: pd.DataFrame,
    max_chars_per_evidence: int = 1400,
) -> List[HCEvidenceRecord]:
    """
    Convert selected evidence dataframe into prompt-ready records.
    """
    records: List[HCEvidenceRecord] = []

    if evidence_df.empty:
        return records

    df = evidence_df.copy()

    if "hc_selected_rank" in df.columns:
        df = df.sort_values("hc_selected_rank", ascending=True)
    elif "hc_rank_score" in df.columns:
        df = df.sort_values("hc_rank_score", ascending=False)

    for _, row in df.iterrows():
        records.append(
            HCEvidenceRecord(
                chunk_id=str(row.get("chunk_id", "")),
                year=safe_int(row.get("year")),
                source=str(row.get("source", "") or ""),
                section=str(row.get("section", "") or ""),
                text=truncate_text(
                    str(row.get("text_clean", "") or ""),
                    max_chars=max_chars_per_evidence,
                ),
                rank_score=safe_float(row.get("hc_rank_score")),
                base_evidence_score=safe_float(
                    row.get("hc_base_evidence_score_0_1")
                ),
            )
        )

    return records


def format_hc_rubric() -> str:
    """
    Format the HC rubric exactly from config.
    """
    lines = ["Score | Description"]

    for score in sorted(HC_RUBRIC.keys()):
        lines.append(f"{score} | {HC_RUBRIC[score]}")

    return "\n".join(lines)


def format_output_schema() -> str:
    """
    Format expected JSON schema for prompt.
    """
    return json.dumps(PROMPT_OUTPUT_SCHEMA, indent=2)


def format_evidence_records(records: List[HCEvidenceRecord]) -> str:
    """
    Format evidence records for prompt.
    """
    if not records:
        return "No evidence was retrieved."

    blocks: List[str] = []

    for i, record in enumerate(records, start=1):
        year_text = str(record.year) if record.year is not None else "Unknown"
        rank_score = (
            f"{record.rank_score:.4f}"
            if record.rank_score is not None
            else "N/A"
        )
        base_score = (
            f"{record.base_evidence_score:.4f}"
            if record.base_evidence_score is not None
            else "N/A"
        )

        block = f"""
Evidence {i}
chunk_id: {record.chunk_id}
year: {year_text}
source: {record.source}
section: {record.section}
retrieval_rank_score: {rank_score}
base_evidence_score_0_1: {base_score}
text:
{record.text}
""".strip()

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def build_hc_prompt(
    company: str,
    evidence_df: pd.DataFrame,
    max_chars_per_evidence: int = 1400,
) -> str:
    """
    Build company-level HC scoring prompt.

    HC has only one question. The LLM should evaluate the evidence pack as a set,
    not score each evidence independently.
    """
    records = evidence_dataframe_to_records(
        evidence_df=evidence_df,
        max_chars_per_evidence=max_chars_per_evidence,
    )

    evidence_text = format_evidence_records(records)

    prompt = f"""
You are evaluating a company's History Consistency for a purpose-driven business scoring system.

Company:
{company}

Dimension:
{DIMENSION_LABEL}

Question ID:
{HC_QUESTION_ID}

Question:
{HC_QUESTION_TEXT}

Rubric:
{format_hc_rubric()}

Evidence:
{evidence_text}

Instructions:
1. Use only the provided evidence.
2. Evaluate whether the company has maintained a consistent purpose narrative over time.
3. Pay attention to whether the central purpose remains stable across years and disclosures.
4. Do not reward generic repetition unless it supports a stable purpose narrative.
5. Do not penalize wording changes if the central purpose remains the same.
6. Penalize contradictions, frequent narrative shifts, or evidence that only supports a single-year claim.
7. Return only valid JSON. Do not include markdown.

Expected JSON schema:
{format_output_schema()}

JSON:
""".strip()

    return prompt


def build_empty_hc_response(company: str) -> Dict[str, object]:
    """
    Return deterministic empty response when no evidence exists.
    """
    return {
        "company": company,
        "hc_score_0_5": 0,
        "rationale": "No evidence was retrieved for History Consistency scoring.",
        "evidence_used": [],
        "confidence": "low",
        "needs_human_review": True,
    }


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "chunk_id": ["c1", "c2"],
            "year": [2018, 2024],
            "source": ["edgar", "edgar"],
            "section": ["letter to shareholders", "business"],
            "text_clean": [
                "Since our founding, our purpose has been to improve access to technology for communities.",
                "We remain committed to improving access to technology and supporting communities globally.",
            ],
            "hc_rank_score": [0.91, 0.88],
            "hc_base_evidence_score_0_1": [0.83, 0.79],
        }
    )

    print(build_hc_prompt("Sample Company", sample))
