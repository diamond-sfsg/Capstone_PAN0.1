"""
Company Purpose Scoring Configuration

This module contains configuration constants for Phase 4 company purpose scoring,
including file paths, LLM settings, scoring parameters, dimension weights, and
evidence selection thresholds.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASE3_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "phase3"
    / "evidence_llm_review_v1"
    / "all_evidence_llm_reviews_v1_newdata.csv"
)

PHASE4_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "phase4"
    / "company_purpose_score_v1"
)

DEFAULT_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

OPENAI_MODEL_ENV_VAR = "OPENAI_MODEL"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
ANTHROPIC_MODEL_ENV_VAR = "ANTHROPIC_MODEL"
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 1200

TARGET_LEVEL = "company_year"

COMPANY_COL = "company"
YEAR_COL = "year"
BUCKET_COL = "evidence_bucket"
TEXT_COL_CANDIDATES = ["text_clean", "text_raw", "text"]
ID_COL = "chunk_id"

DIMENSIONS = [
    "purpose_articulation",
    "history_consistency",
    "strategy_alignment",
]

DIMENSION_TO_SCORE_FIELD = {
    "purpose_articulation": "pa_final_score",
    "history_consistency": "hc_final_score",
    "strategy_alignment": "sa_final_score",
}

DIMENSION_WEIGHTS = {
    "purpose_articulation": 1 / 3,
    "history_consistency": 1 / 3,
    "strategy_alignment": 1 / 3,
}

TOP_K_EVIDENCE_PER_DIMENSION = 8
MIN_EVIDENCE_PER_DIMENSION = 2
HISTORY_LOOKBACK_YEARS = 5

PURPOSE_DRIVEN_THRESHOLD_0_100 = 70

# Whether to exclude Phase 3 human-review evidence entirely.
# False = keep but penalize.
EXCLUDE_HUMAN_REVIEW_QUEUE = False

SOURCE_WEIGHTS = {
    "edgar": 1.00,
    "official_web": 0.90,
    "linkedin": 0.70,
}

DEFAULT_SOURCE_WEIGHT = 0.75

OUTPUT_FILE_TEMPLATES = {
    "company_year_scores": "company_year_purpose_score_{run_version}.csv",
    "company_scores": "company_purpose_score_{run_version}.csv",
    "evidence_pack": "company_purpose_evidence_pack_{run_version}.csv",
    "diagnostics": "company_purpose_score_diagnostics_{run_version}.txt",
}
