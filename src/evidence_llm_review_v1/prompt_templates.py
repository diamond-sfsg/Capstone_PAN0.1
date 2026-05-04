from __future__ import annotations

import json
from typing import Any

try:
    from .llm_review_config import (
        DIMENSION_DEFINITIONS,
        COMMON_FIELD_DESCRIPTIONS,
        PA_ONLY_FIELD_DESCRIPTIONS,
        COMMON_LLM_REVIEW_FIELDS,
        PA_ONLY_LLM_FIELDS,
    )
except ImportError:
    from evidence_llm_review_v1.llm_review_config import (
        DIMENSION_DEFINITIONS,
        COMMON_FIELD_DESCRIPTIONS,
        PA_ONLY_FIELD_DESCRIPTIONS,
        COMMON_LLM_REVIEW_FIELDS,
        PA_ONLY_LLM_FIELDS,
    )


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
You are an evidence quality reviewer for a business analytics capstone project.

The project evaluates whether companies demonstrate purpose-driven behavior.
Your job is NOT to produce the final company purpose score.
Your job is to review one evidence chunk at a time and judge whether it is useful,
credible, specific, and relevant to its assigned evidence bucket.

Be conservative.
Do not reward vague corporate language.
Do not infer more than the evidence supports.
If the evidence is generic, boilerplate, purely promotional, or weakly related,
assign lower scores and mark human review when appropriate.

Return valid JSON only.
Do not include markdown.
Do not include commentary outside the JSON object.
""".strip()


# =============================================================================
# JSON SCHEMAS
# =============================================================================

COMMON_JSON_SCHEMA = {
    "llm_bucket_relevance_score": "integer 0-5",
    "llm_evidence_specificity_score": "integer 0-5",
    "llm_source_context_score": "integer 0-5",
    "llm_boilerplate_risk_score": "integer 0-5",
    "llm_credibility_score": "integer 0-5",
    "llm_confidence": "float 0-1",
    "llm_needs_human_review": "boolean",
    "llm_review_reason": "string, maximum 40 words",
}


PA_ONLY_JSON_SCHEMA = {
    "llm_purpose_clarity_score": "integer 0-5",
    "llm_beyond_profit_score": "integer 0-5",
    "llm_commitment_tone_score": "integer 0-5",
    "llm_authenticity_score": "integer 0-5",
    "llm_pa_tone_context_score": "integer 0-5",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and str(value) == "nan":
        return ""
    return str(value)


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _format_field_descriptions(field_descriptions: dict[str, str]) -> str:
    return "\n".join(
        f"- {field}: {description}"
        for field, description in field_descriptions.items()
    )


def _json_schema_for_bucket(bucket_name: str) -> dict[str, str]:
    schema = dict(COMMON_JSON_SCHEMA)

    if bucket_name == "purpose_articulation":
        schema.update(PA_ONLY_JSON_SCHEMA)

    return schema


def get_text_from_row(row: dict[str, Any]) -> str:
    for col in ["text_clean", "text_raw", "text"]:
        value = row.get(col)
        if value is not None and str(value).strip():
            return str(value)

    return ""


def get_dimension_definition(bucket_name: str) -> dict[str, Any]:
    if bucket_name not in DIMENSION_DEFINITIONS:
        raise ValueError(f"Unknown bucket_name: {bucket_name}")

    return DIMENSION_DEFINITIONS[bucket_name]


# =============================================================================
# COMMON REVIEW PROMPT
# =============================================================================

def build_common_review_prompt(row: dict[str, Any], bucket_name: str) -> str:
    """
    Build the user prompt for common evidence review.
    This applies to all three evidence pools:
    - purpose_articulation
    - history_consistency
    - strategy_alignment
    """

    dim = get_dimension_definition(bucket_name)
    text = get_text_from_row(row)

    metadata = {
        "chunk_id": _safe_str(row.get("chunk_id")),
        "company": _safe_str(row.get("company")),
        "year": _safe_str(row.get("year")),
        "source": _safe_str(row.get("source")),
        "section": _safe_str(row.get("section")),
        "subsection": _safe_str(row.get("subsection")),
        "evidence_bucket": bucket_name,
        "bucket_score": _safe_str(row.get("bucket_score")),
        "bucket_threshold": _safe_str(row.get("bucket_threshold")),
        "bucket_overlap_type": _safe_str(row.get("bucket_overlap_type")),
        "top_score_margin": _safe_str(row.get("top_score_margin")),
        "phase2_review_flag": _safe_str(row.get("review_flag")),
    }

    schema = _json_schema_for_bucket(bucket_name)

    prompt = f"""
Review this evidence chunk for the assigned evidence bucket.

Assigned evidence bucket:
{bucket_name} - {dim["label"]}

Bucket definition:
{dim["definition"]}

Strong evidence examples:
{_format_list(dim["strong_evidence"])}

Weak evidence examples:
{_format_list(dim["weak_evidence"])}

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Evidence text:
\"\"\"
{text}
\"\"\"

Common review fields:
{_format_field_descriptions(COMMON_FIELD_DESCRIPTIONS)}

Scoring guidance:
- 5 = very strong
- 4 = strong
- 3 = usable but mixed
- 2 = weak
- 1 = very weak
- 0 = not relevant or unusable

Boilerplate risk is reversed:
- 5 = very high boilerplate risk
- 0 = no boilerplate risk

Human review should be true if:
- the evidence appears misclassified
- relevance is low
- credibility is low
- boilerplate risk is high
- the text is ambiguous
- metadata suggests overlap or low margin

Return JSON using exactly this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()

    return prompt


# =============================================================================
# PA-ONLY TONE / CONTEXT PROMPT
# =============================================================================

def build_pa_tone_review_prompt(row: dict[str, Any]) -> str:
    """
    Build the user prompt for PA-only tone/context review.
    This should only be applied to purpose_articulation evidence.
    """

    bucket_name = "purpose_articulation"
    dim = get_dimension_definition(bucket_name)
    text = get_text_from_row(row)

    metadata = {
        "chunk_id": _safe_str(row.get("chunk_id")),
        "company": _safe_str(row.get("company")),
        "year": _safe_str(row.get("year")),
        "source": _safe_str(row.get("source")),
        "section": _safe_str(row.get("section")),
        "subsection": _safe_str(row.get("subsection")),
        "evidence_bucket": bucket_name,
        "bucket_score": _safe_str(row.get("bucket_score")),
        "bucket_threshold": _safe_str(row.get("bucket_threshold")),
    }

    prompt = f"""
Review this purpose articulation evidence for tone and context quality.

This PA-only review evaluates HOW the company articulates purpose.
Do not judge history consistency or strategy execution here.

Purpose Articulation definition:
{dim["definition"]}

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Evidence text:
\"\"\"
{text}
\"\"\"

PA-only review fields:
{_format_field_descriptions(PA_ONLY_FIELD_DESCRIPTIONS)}

Scoring guidance:
- llm_purpose_clarity_score:
  5 = clearly states why the company exists
  0 = no clear purpose articulation

- llm_beyond_profit_score:
  5 = clearly goes beyond profit or shareholder return
  0 = only financial or generic business goal

- llm_commitment_tone_score:
  5 = durable, serious, long-term commitment
  0 = casual, vague, or throwaway language

- llm_authenticity_score:
  5 = company-specific and authentic
  0 = generic slogan or marketing boilerplate

- llm_pa_tone_context_score:
  Overall tone/context quality for purpose articulation.

Return JSON using exactly this schema:
{json.dumps(PA_ONLY_JSON_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()

    return prompt


# =============================================================================
# COMBINED PROMPT
# =============================================================================

def build_review_prompt(row: dict[str, Any], bucket_name: str) -> str:
    """
    Build one combined prompt.

    For HC and SA:
        returns common review prompt only.

    For PA:
        returns common review requirements + PA-only tone/context requirements
        in a single JSON output.
    """

    if bucket_name != "purpose_articulation":
        return build_common_review_prompt(row=row, bucket_name=bucket_name)

    dim = get_dimension_definition(bucket_name)
    text = get_text_from_row(row)

    metadata = {
        "chunk_id": _safe_str(row.get("chunk_id")),
        "company": _safe_str(row.get("company")),
        "year": _safe_str(row.get("year")),
        "source": _safe_str(row.get("source")),
        "section": _safe_str(row.get("section")),
        "subsection": _safe_str(row.get("subsection")),
        "evidence_bucket": bucket_name,
        "bucket_score": _safe_str(row.get("bucket_score")),
        "bucket_threshold": _safe_str(row.get("bucket_threshold")),
        "bucket_overlap_type": _safe_str(row.get("bucket_overlap_type")),
        "top_score_margin": _safe_str(row.get("top_score_margin")),
        "phase2_review_flag": _safe_str(row.get("review_flag")),
    }

    combined_field_descriptions = dict(COMMON_FIELD_DESCRIPTIONS)
    combined_field_descriptions.update(PA_ONLY_FIELD_DESCRIPTIONS)

    combined_schema = dict(COMMON_JSON_SCHEMA)
    combined_schema.update(PA_ONLY_JSON_SCHEMA)

    prompt = f"""
Review this evidence chunk for the assigned evidence bucket.

Assigned evidence bucket:
purpose_articulation - {dim["label"]}

Bucket definition:
{dim["definition"]}

Strong evidence examples:
{_format_list(dim["strong_evidence"])}

Weak evidence examples:
{_format_list(dim["weak_evidence"])}

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Evidence text:
\"\"\"
{text}
\"\"\"

You must complete two review layers:

Layer 1 - Common evidence review:
Evaluate whether this evidence is relevant, specific, credible, and useful.

Layer 2 - Purpose Articulation tone/context review:
Evaluate whether the company clearly articulates a purpose, whether it goes beyond profit,
whether the language shows durable commitment, and whether it sounds company-specific
rather than generic marketing language.

Fields:
{_format_field_descriptions(combined_field_descriptions)}

Scoring guidance:
- 5 = very strong
- 4 = strong
- 3 = usable but mixed
- 2 = weak
- 1 = very weak
- 0 = not relevant or unusable

Boilerplate risk is reversed:
- 5 = very high boilerplate risk
- 0 = no boilerplate risk

Human review should be true if:
- the evidence appears misclassified
- relevance is low
- credibility is low
- boilerplate risk is high
- the purpose articulation is vague or ambiguous
- metadata suggests overlap or low margin

Return JSON using exactly this schema:
{json.dumps(combined_schema, ensure_ascii=False, indent=2)}
""".strip()

    return prompt


# =============================================================================
# RESPONSE VALIDATION HELPERS
# =============================================================================

def expected_fields_for_bucket(bucket_name: str) -> list[str]:
    if bucket_name == "purpose_articulation":
        return COMMON_LLM_REVIEW_FIELDS + PA_ONLY_LLM_FIELDS

    return COMMON_LLM_REVIEW_FIELDS


def empty_review_result(bucket_name: str) -> dict[str, Any]:
    result = {}

    for field in expected_fields_for_bucket(bucket_name):
        if field == "llm_needs_human_review":
            result[field] = True
        elif field == "llm_review_reason":
            result[field] = "Missing or invalid LLM response."
        elif field == "llm_confidence":
            result[field] = 0.0
        else:
            result[field] = None

    return result

