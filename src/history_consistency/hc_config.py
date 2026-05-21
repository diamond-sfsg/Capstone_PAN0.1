# src/history_consistency/hc_config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DATA_DIR = DATA_DIR / "clean_2.0"
OUTPUT_DIR = DATA_DIR / "history_consistency"

INPUT_CHUNKS_CSV = CLEAN_DATA_DIR / "unified_chunks_final_v4.csv"

HC_COMPANY_SCORE_OUTPUT = OUTPUT_DIR / "hc_company_score_v1.csv"
HC_EVIDENCE_OUTPUT = OUTPUT_DIR / "hc_evidence_details_v1.csv"
HC_DIAGNOSTICS_OUTPUT = OUTPUT_DIR / "hc_diagnostics_v1.txt"


# =============================================================================
# Target Definition
# =============================================================================

DIMENSION_ID = "history_consistency"
DIMENSION_LABEL = "History Consistency"

TARGET_LEVEL = "company"
YEAR_WINDOW = 10

MIN_YEARS_FOR_FULL_HC = 3
MIN_EVIDENCE_PER_COMPANY = 2
TOP_K_EVIDENCE = 12
TOP_K_EVIDENCE_PER_YEAR = 3


# =============================================================================
# HC Question — DO NOT MODIFY WORDING
# =============================================================================

HC_QUESTION_ID = "HC_Q1"

HC_QUESTION_TEXT = (
    "Historical Consistency: To what extent has the company maintained a "
    "consistent purpose narrative over the past ten years across annual reports, "
    "CEO letters, and corporate disclosures?"
)


# =============================================================================
# HC Rubric — DO NOT MODIFY WORDING
# =============================================================================

HC_RUBRIC: Dict[int, str] = {
    0: "Narrative frequently changes or contains contradictions.",
    1: "Core theme shifts often, making it difficult to identify a consistent narrative.",
    2: "Some recurring themes exist, but the narrative frequently drifts.",
    3: "The core idea remains consistent, but wording differs significantly across documents.",
    4: "Core narrative remains largely consistent across major disclosures.",
    5: (
        "Core narrative remains almost unchanged across major disclosures, or any changes "
        "are clearly explainable without altering the central purpose."
    ),
}


# =============================================================================
# Retrieval Query / Keywords
# =============================================================================

HC_DIMENSION_QUERY = (
    "consistent purpose narrative over the past ten years across annual reports, "
    "CEO letters, corporate disclosures, recurring mission themes, long-term purpose, "
    "continued commitment, sustained purpose, historical continuity"
)

HC_KEYWORDS_CORE: List[str] = [
    "consistent purpose",
    "purpose narrative",
    "long-term purpose",
    "continued commitment",
    "sustained commitment",
    "historical consistency",
    "consistent narrative",
    "core narrative",
    "core purpose",
    "central purpose",
    "recurring theme",
    "same purpose",
]

HC_KEYWORDS_TEMPORAL: List[str] = [
    "over the past ten years",
    "over time",
    "for decades",
    "since",
    "from the beginning",
    "historically",
    "long-standing",
    "longstanding",
    "continued",
    "continue to",
    "remains committed",
    "we remain committed",
    "we have always",
    "has always been",
]

HC_KEYWORDS_DOCUMENT_CONTEXT: List[str] = [
    "annual report",
    "CEO letter",
    "letter to shareholders",
    "corporate disclosure",
    "10-k",
    "proxy statement",
    "shareholder letter",
    "company history",
    "our history",
    "heritage",
    "legacy",
]

HC_NEGATIVE_TERMS: List[str] = [
    "risk factors",
    "forward-looking statements",
    "safe harbor",
    "legal proceedings",
    "market risk",
    "accounting policies",
    "tax matters",
]


# =============================================================================
# Source / Section Priors
# =============================================================================

SOURCE_PRIOR: Dict[str, float] = {
    "edgar": 1.20,
    "official_web": 1.00,
    "linkedin": 0.70,
}

PREFERRED_SECTIONS: List[str] = [
    "ceo letter",
    "letter to shareholders",
    "business",
    "overview",
    "mission",
    "about",
    "company history",
    "strategy",
    "annual report",
    "proxy",
    "sustainability",
]

SECTION_PRIOR: Dict[str, float] = {
    "ceo letter": 1.25,
    "letter to shareholders": 1.25,
    "company history": 1.20,
    "mission": 1.15,
    "about": 1.10,
    "business": 1.05,
    "overview": 1.05,
    "strategy": 1.05,
    "risk factors": 0.60,
    "legal proceedings": 0.50,
}


# =============================================================================
# Evidence Base Score Weights
# Same logic as PA: retrieval quality controls evidence contribution.
# =============================================================================

BASE_EVIDENCE_SCORE_WEIGHTS: Dict[str, float] = {
    "keyword_relevance": 0.40,
    "context_completeness": 0.35,
    "rag_similarity": 0.25,
}

EVIDENCE_QUALITY_FACTOR_MIN = 0.70
EVIDENCE_QUALITY_FACTOR_SCALE = 0.30


# =============================================================================
# HC Redundancy / Overlap Logic
# Relaxed version of PA overlap control.
# Cross-year recurrence is NOT penalized because it is an HC signal.
# =============================================================================

HC_REDUNDANCY_FACTOR: Dict[str, float] = {
    "unique_evidence": 1.00,
    "cross_year_recurring": 1.00,
    "same_year_near_duplicate": 0.85,
    "same_year_exact_duplicate": 0.65,
    "same_doc_repeated_boilerplate": 0.75,
    "unknown": 1.00,
}

DUPLICATE_SCOPE_COLUMN = "similarity_scope"
DUPLICATE_GROUP_COLUMN = "duplicate_group"

CROSS_YEAR_RECURRING_VALUES: Tuple[str, ...] = (
    "cross_year_recurring",
    "cross_year_similar",
)

SAME_YEAR_DUPLICATE_VALUES: Tuple[str, ...] = (
    "exact_same_year",
    "near_same_year",
    "same_year_duplicate",
)


# =============================================================================
# HC Bonus — Old Framework Logic
# Rewards cross-year consistency / duplicate-year recurrence.
# This replaces PA tone bonus.
# =============================================================================

HC_HISTORY_BONUS_ENABLED = True
HC_HISTORY_BONUS_MAX = 0.50

HC_HISTORY_BONUS_WEIGHTS: Dict[str, float] = {
    "multi_year_coverage": 0.15,
    "three_plus_year_coverage": 0.15,
    "cross_year_recurring_theme": 0.15,
    "historical_and_recent_coverage": 0.05,
}

HISTORICAL_RECENT_SPLIT_YEARS = 5


# =============================================================================
# Aggregation Weights
# HC has only one Q, but evidence must come from multiple years where possible.
# =============================================================================

HC_AGGREGATION_WEIGHTS: Dict[str, float] = {
    "best_evidence": 0.50,
    "best_distinct_year_evidence": 0.30,
    "mean_top_evidence_by_year": 0.20,
}

SINGLE_YEAR_SCORE_CAP = 3.0


# =============================================================================
# Score Bounds
# =============================================================================

LLM_SCORE_MIN = 0
LLM_SCORE_MAX = 5

HC_SCORE_MIN = 0
HC_SCORE_MAX = 5

HC_SCORE_100_MIN = 0
HC_SCORE_100_MAX = 100


# =============================================================================
# Column Names
# =============================================================================

REQUIRED_INPUT_COLUMNS: List[str] = [
    "chunk_id",
    "company",
    "year",
    "source",
    "section",
    "text_clean",
]

OPTIONAL_INPUT_COLUMNS: List[str] = [
    "doc_id",
    "source_file",
    "subsection",
    "token_count",
    "char_count",
    "quality_flag",
    "duplicate_group",
    "similarity_scope",
]

OUTPUT_COLUMNS: List[str] = [
    "company",
    "hc_question_id",
    "hc_question_text",
    "hc_final_score_0_5",
    "hc_score_0_100",
    "hc_base_score_0_5",
    "hc_history_bonus",
    "distinct_year_count",
    "evidence_count",
    "needs_human_review",
    "review_reason",
]


# =============================================================================
# Prompt Control
# =============================================================================

LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3

PROMPT_OUTPUT_SCHEMA = {
    "hc_score_0_5": "integer from 0 to 5",
    "rationale": "short explanation using only provided evidence",
    "evidence_used": "list of chunk_ids",
    "confidence": "low | medium | high",
    "needs_human_review": "boolean",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class HCQuestionConfig:
    question_id: str
    question_text: str
    rubric: Dict[int, str]


@dataclass(frozen=True)
class HCScoringConfig:
    base_evidence_score_weights: Dict[str, float]
    evidence_quality_factor_min: float
    evidence_quality_factor_scale: float
    redundancy_factor: Dict[str, float]
    aggregation_weights: Dict[str, float]
    history_bonus_weights: Dict[str, float]
    history_bonus_max: float
    single_year_score_cap: float


HC_QUESTION_CONFIG = HCQuestionConfig(
    question_id=HC_QUESTION_ID,
    question_text=HC_QUESTION_TEXT,
    rubric=HC_RUBRIC,
)

HC_SCORING_CONFIG = HCScoringConfig(
    base_evidence_score_weights=BASE_EVIDENCE_SCORE_WEIGHTS,
    evidence_quality_factor_min=EVIDENCE_QUALITY_FACTOR_MIN,
    evidence_quality_factor_scale=EVIDENCE_QUALITY_FACTOR_SCALE,
    redundancy_factor=HC_REDUNDANCY_FACTOR,
    aggregation_weights=HC_AGGREGATION_WEIGHTS,
    history_bonus_weights=HC_HISTORY_BONUS_WEIGHTS,
    history_bonus_max=HC_HISTORY_BONUS_MAX,
    single_year_score_cap=SINGLE_YEAR_SCORE_CAP,
)


# =============================================================================
# Validation
# =============================================================================

def validate_hc_config() -> None:
    """Validate static HC config values before running the pipeline."""

    if set(HC_RUBRIC.keys()) != {0, 1, 2, 3, 4, 5}:
        raise ValueError("HC_RUBRIC must contain score levels 0–5.")

    if not abs(sum(BASE_EVIDENCE_SCORE_WEIGHTS.values()) - 1.0) < 1e-6:
        raise ValueError("BASE_EVIDENCE_SCORE_WEIGHTS must sum to 1.0.")

    if not abs(sum(HC_AGGREGATION_WEIGHTS.values()) - 1.0) < 1e-6:
        raise ValueError("HC_AGGREGATION_WEIGHTS must sum to 1.0.")

    if HC_HISTORY_BONUS_MAX < 0:
        raise ValueError("HC_HISTORY_BONUS_MAX must be non-negative.")

    if SINGLE_YEAR_SCORE_CAP > HC_SCORE_MAX:
        raise ValueError("SINGLE_YEAR_SCORE_CAP cannot exceed HC_SCORE_MAX.")
