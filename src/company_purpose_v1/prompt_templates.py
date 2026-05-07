from __future__ import annotations

import json
from typing import Any

from company_purpose_v1.rubric_config import PURPOSE_RUBRIC


SYSTEM_PROMPT = """
You are a business analytics evaluator scoring whether a company is purpose-driven in a given year based on retrieved evidence and a detailed rubric.

You must score a company-year using only the provided evidence.
Do not use outside knowledge.
Do not over-reward generic mission language.
A high score requires evidence quality, relevance, and alignment with the rubric.

Return valid JSON only.
Do not include markdown.
""".strip()


def _safe(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _format_evidence_pack(evidence_pack: dict[str, list[dict[str, Any]]]) -> str:
    sections = []

    for dimension, rows in evidence_pack.items():
        rubric = PURPOSE_RUBRIC[dimension]
        question_labels = ", ".join(
            question["score_field"] for question in rubric["questions"]
        )
        sections.append(f"\n## {dimension} - {rubric['label']} ({question_labels})")

        if not rows:
            sections.append("No retrieved evidence.")
            continue

        for idx, row in enumerate(rows, start=1):
            text = _safe(row.get("text_for_prompt"))[:900]
            sections.append(
                f"""
Evidence {idx}
chunk_id: {_safe(row.get("chunk_id"))}
source: {_safe(row.get("source"))}
year: {_safe(row.get("year"))}
section: {_safe(row.get("section"))}
rag_weight: {_safe(row.get("rag_weight"))}
bucket_score: {_safe(row.get("bucket_score"))}
llm_relevance: {_safe(row.get("llm_bucket_relevance_score"))}
llm_credibility: {_safe(row.get("llm_credibility_score"))}
llm_review_reason: {_safe(row.get("llm_review_reason"))}
text:
{text}
""".strip()
            )

    return "\n\n".join(sections)


def build_company_purpose_prompt(
    company: str,
    year: str | int,
    evidence_pack: dict[str, list[dict[str, Any]]],
) -> str:
    rubric_text = {}

    for dimension, rubric in PURPOSE_RUBRIC.items():
        rubric_text[dimension] = {
            "label": rubric["label"],
            "score_field": rubric["score_field"],
            "questions": [
                {
                    "label": question["label"],
                    "score_field": question["score_field"],
                    "question": question["question"],
                    "criteria": question.get("criteria", []),
                    "score_guide": question["score_guide"],
                }
                for question in rubric["questions"]
            ],
        }

    evidence_text = _format_evidence_pack(evidence_pack)

    expected_schema = {
        "company": "string",
        "year": "string or integer",
        "pa_Q1": "integer 0-5",
        "pa_Q2": "integer 0-5",
        "pa_Q3": "integer 0-5",
        "pa_final_score": "integer 0-5",
        "hc_Q1": "integer 0-5",
        "hc_final_score": "integer 0-5",
        "sa_Q1": "integer 0-5",
        "sa_Q2": "integer 0-5",
        "sa_final_score": "integer 0-5",
        "company_purpose_score_0_100": "float 0-100",
        "purpose_driven_label": "boolean",
        "confidence": "float 0-1",
        "pa_reason": "string, maximum 60 words",
        "hc_reason": "string, maximum 60 words",
        "sa_reason": "string, maximum 60 words",
        "overall_reason": "string, maximum 80 words",
        "key_supporting_chunk_ids": "list of strings",
        "weak_or_contradictory_chunk_ids": "list of strings",
        "needs_human_review": "boolean",
    }

    prompt = f"""
Score the following company-year using the Sample Work Purpose Rubric.

Company: {company}
Year: {year}

Rubric:
{json.dumps(rubric_text, ensure_ascii=False, indent=2)}

Retrieved evidence:
{evidence_text}

Rules:
1. Score each rubric question from 0 to 5.
2. Use only retrieved evidence.
3. Penalize vague, boilerplate, or purely promotional language.
4. Purpose Articulation evaluates what the company says.
5. History Consistency evaluates continuity over time.
6. Strategy Alignment evaluates whether purpose connects to strategy, resources, KPIs, or outcomes.
7. If one dimension has insufficient evidence, score that dimension conservatively.
8. pa_final_score should be the average of pa_Q1, pa_Q2, and pa_Q3.
9. hc_final_score should equal hc_Q1.
10. sa_final_score should be the average of sa_Q1 and sa_Q2.
11. company_purpose_score_0_100 should be the weighted average of PA, HC, and SA final scores converted to 0-100.
12. purpose_driven_label should be true only if the overall score is strong and no major dimension is unsupported.

Return JSON using exactly this schema:
{json.dumps(expected_schema, ensure_ascii=False, indent=2)}
""".strip()

    return prompt
