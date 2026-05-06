from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE2_INPUT_DIR = PROJECT_ROOT / "data" / "phase2" / "evidence_buckets_v1"
PHASE3_OUTPUT_DIR = PROJECT_ROOT / "data" / "phase3" / "evidence_llm_review_v1"

LLM_PROVIDER = "openai"
LLM_MODEL_ENV_VAR = "OPENAI_MODEL"
LLM_API_KEY_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

RUN_VERSION = "evidence_llm_review_v1_newdata"
TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 500
MAX_ROWS_PER_BUCKET = None
SAVE_EVERY_N_ROWS = 25

ID_COL = "chunk_id"
TEXT_COL_CANDIDATES = ["text_clean", "text_raw", "text"]

REQUIRED_BASE_COLS = [
    "evidence_bucket",
    "bucket_score",
    "bucket_threshold",
    "company",
    "source",
    "text_clean",
]

OPTIONAL_CONTEXT_COLS = [
    "doc_id",
    "year",
    "source_file",
    "section",
    "subsection",
    "quality_flag",
    "is_duplicate_like",
    "duplicate_group",
    "bucket_overlap_count",
    "bucket_overlap_type",
    "top_bucket_by_score",
    "top_score",
    "second_score",
    "top_score_margin",
    "needs_overlap_review",
    "needs_margin_review",
    "review_flag",
]

EVIDENCE_POOL_CONFIG = {
    "purpose_articulation": {
        "input_csv": "purpose_articulation_evidence_v1_newdata.csv",
    },
    "history_consistency": {
        "input_csv": "history_consistency_evidence_v1_newdata.csv",
    },
    "strategy_alignment": {
        "input_csv": "strategy_alignment_evidence_v1_newdata.csv",
    },
}

OUTPUT_FILES = {
    "purpose_articulation": "purpose_articulation_llm_review_v1_newdata.csv",
    "history_consistency": "history_consistency_llm_review_v1_newdata.csv",
    "strategy_alignment": "strategy_alignment_llm_review_v1_newdata.csv",
    "all_reviews": "all_evidence_llm_reviews_v1_newdata.csv",
    "human_review_queue": "human_review_queue_v1_newdata.csv",
    "diagnostics": "evidence_llm_review_diagnostics_v1_newdata.txt",
}

COMMON_LLM_REVIEW_FIELDS = [
    "llm_bucket_relevance_score",
    "llm_evidence_specificity_score",
    "llm_source_context_score",
    "llm_boilerplate_risk_score",
    "llm_credibility_score",
    "llm_confidence",
    "llm_needs_human_review",
    "llm_review_reason",
]

PA_ONLY_LLM_FIELDS = [
    "llm_purpose_clarity_score",
    "llm_beyond_profit_score",
    "llm_commitment_tone_score",
    "llm_authenticity_score",
    "llm_pa_tone_context_score",
]

ALL_LLM_REVIEW_FIELDS = COMMON_LLM_REVIEW_FIELDS + PA_ONLY_LLM_FIELDS

COMMON_FIELD_DESCRIPTIONS = {
    "llm_bucket_relevance_score": "How relevant the chunk is to the assigned evidence bucket.",
    "llm_evidence_specificity_score": "How concrete, company-specific, and non-generic the evidence is.",
    "llm_source_context_score": "How useful the source, section, and metadata context are for interpretation.",
    "llm_boilerplate_risk_score": "Risk that the chunk is generic boilerplate, legal language, or weak marketing copy.",
    "llm_credibility_score": "How credible and decision-useful the evidence is for later human analysis.",
    "llm_confidence": "Reviewer confidence in the judgment, from 0 to 1.",
    "llm_needs_human_review": "Whether a human should inspect this chunk before final use.",
    "llm_review_reason": "Brief reason for the review judgment.",
}

PA_ONLY_FIELD_DESCRIPTIONS = {
    "llm_purpose_clarity_score": "Whether the chunk clearly states why the company exists.",
    "llm_beyond_profit_score": "Whether the purpose goes beyond profit or shareholder return.",
    "llm_commitment_tone_score": "Whether the language signals durable commitment rather than a one-off slogan.",
    "llm_authenticity_score": "Whether the purpose language feels company-specific and authentic.",
    "llm_pa_tone_context_score": "Overall tone and context quality for purpose articulation evidence.",
}

DIMENSION_DEFINITIONS = {
    "purpose_articulation": {
        "label": "Purpose Articulation",
        "definition": (
            "Evidence that the company clearly describes its mission, reason for existence, "
            "or purpose, especially when it goes beyond narrow financial outcomes."
        ),
        "strong_evidence": [
            "A direct statement of purpose, mission, or reason for existence.",
            "Company-specific values tied to customers, employees, communities, or society.",
            "Language that connects purpose to identity or long-term value creation.",
        ],
        "weak_evidence": [
            "Generic marketing slogans without concrete meaning.",
            "Purely financial performance claims.",
            "Legal disclaimers, risk factors, or table-of-contents text.",
        ],
    },
    "history_consistency": {
        "label": "History Consistency",
        "definition": (
            "Evidence that purpose, mission, values, or strategic commitments are sustained "
            "across time rather than appearing as isolated statements."
        ),
        "strong_evidence": [
            "Explicit continuity language such as remain committed or continue to.",
            "Recurring purpose or strategy statements across years.",
            "Long-term commitments linked to historical identity or durable priorities.",
        ],
        "weak_evidence": [
            "One-year announcements with no continuity signal.",
            "Temporary initiatives or short-term performance commentary.",
            "Repeated legal boilerplate that is similar across filings but not substantive.",
        ],
    },
    "strategy_alignment": {
        "label": "Strategy Alignment",
        "definition": (
            "Evidence that stated purpose or mission is connected to strategy, investments, "
            "resource allocation, operating priorities, or concrete business decisions."
        ),
        "strong_evidence": [
            "Purpose or mission linked to strategic priorities.",
            "Capital allocation, investment, or operating choices aligned with stated purpose.",
            "Concrete initiatives that show execution rather than aspiration only.",
        ],
        "weak_evidence": [
            "Strategy language with no connection to purpose or stakeholder value.",
            "Broad innovation or growth claims without specific actions.",
            "Financial or risk-management text unrelated to purpose alignment.",
        ],
    },
}

HUMAN_REVIEW_RULES = {
    "low_bucket_relevance_threshold": 2,
    "low_credibility_threshold": 2,
    "high_boilerplate_risk_threshold": 4,
    "problematic_phase2_review_flags": [
        "overlap_review",
        "margin_review",
        "overlap_and_margin_review",
        "needs_overlap_review",
        "needs_margin_review",
    ],
}
