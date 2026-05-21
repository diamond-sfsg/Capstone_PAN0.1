from __future__ import annotations

import os
from pathlib import Path


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "clean_2.0"
    / "unified_chunks_final_v4.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "phase_pa"

PA_COMPANY_SCORE_PATH = OUTPUT_DIR / "pa_company_score_v1.csv"
PA_QUESTION_SCORE_PATH = OUTPUT_DIR / "pa_question_score_v1.csv"
PA_EVIDENCE_DETAIL_PATH = OUTPUT_DIR / "pa_evidence_detail_v1.csv"
PA_EVIDENCE_LIBRARY_PATH = OUTPUT_DIR / "pa_evidence_library_v1.csv"
PA_LLM_RAW_OUTPUT_PATH = OUTPUT_DIR / "pa_llm_raw_outputs_v1.jsonl"
PA_DIAGNOSTICS_PATH = OUTPUT_DIR / "pa_diagnostics_v1.txt"


# =============================================================================
# Pipeline identity
# =============================================================================

DIMENSION_NAME = "purpose_articulation"
PIPELINE_VERSION = "pa_company_level_pooled_v1"


# =============================================================================
# Retrieval and selection parameters
# =============================================================================

# For each company and question, retrieve this many candidates before LLM scoring.
TOP_K_CANDIDATES_PER_QUESTION = 25

# Q1/Q2 are evidence-level questions.
# Only this many top evidence chunks are sent to the LLM per company-question.
TOP_K_LLM_EVIDENCE_Q1_Q2 = 5

# Q3 is evidence-set scoring.
# Select several evidence chunks per source to show cross-source distribution.
TOP_K_EVIDENCE_PER_SOURCE_Q3 = 3
TOP_K_GLOBAL_EVIDENCE_Q3 = 8


# =============================================================================
# Evidence score parameters
# =============================================================================

# base_evidence_score_0_1 =
#   keyword + context completeness + RAG / TF-IDF similarity
KEYWORD_WEIGHT = 0.40
CONTEXT_WEIGHT = 0.35
RAG_WEIGHT = 0.25

# Convert base evidence score into a mild quality factor.
# This prevents keyword/rag scoring from overpowering LLM rubric scoring.
#
# evidence_quality_factor = 0.70 + 0.30 * base_evidence_score_0_1
EVIDENCE_QUALITY_MIN = 0.70
EVIDENCE_QUALITY_RANGE = 0.30


# =============================================================================
# Aggregation parameters
# =============================================================================

# For Q1/Q2, top evidence matters most.
# This avoids rewarding companies just because they have more repeated evidence.
TOP1_WEIGHT = 0.70
TOP2_WEIGHT = 0.30

# Q3 evidence-set quality factor:
# evidence_set_quality_factor = 0.75 + 0.25 * evidence_set_quality
Q3_SET_QUALITY_MIN = 0.75
Q3_SET_QUALITY_RANGE = 0.25


# =============================================================================
# PA tone bonus
# =============================================================================

# Tone bonus is intentionally small because the Clarity rubric already captures
# concrete and specific language. This bonus is only a tie-breaker.
PA_TONE_BONUS_MIN = 0.00
PA_TONE_BONUS_MAX = 0.30


# =============================================================================
# Source prior
# =============================================================================

# Q1/Q2: Purpose Presence and Clarity.
# Official website gets higher priority because PA evaluates whether the company
# actively and clearly articulates its purpose.
PA_SOURCE_PRIOR_Q1_Q2 = {
    "official_web": 1.20,
    "edgar": 1.00,
    "linkedin": 0.80,
    "unknown": 1.00,
}

# Q3: Distinction from Branding.
# EDGAR / formal documents get higher priority because this question asks whether
# purpose appears beyond marketing or promotional contexts.
PA_SOURCE_PRIOR_Q3 = {
    "edgar": 1.20,
    "official_web": 0.90,
    "linkedin": 0.70,
    "unknown": 1.00,
}

DEFAULT_SOURCE_PRIOR = 1.00


# =============================================================================
# Overlap penalty
# =============================================================================

# Same evidence reused across PA questions receives decreasing weight.
# This prevents one strong statement from inflating all question scores.
OVERLAP_FACTORS = {
    1: 1.00,
    2: 0.70,
    3: 0.50,
}

OVERLAP_FACTOR_4_PLUS = 0.30


# =============================================================================
# Section / PA-specific lexical configuration
# =============================================================================

# Sections that should be excluded from purpose retrieval (likely not discussing purpose)
SECTION_BLACKLIST = [
    "risk factor",
    "risk factors",
    "legal proceedings",
    "regulation",
    "regulatory",
    "accounting policies",
    "notes to financial statements",
    "tax",
    "market risk",
    "quantitative and qualitative disclosures",
    "controls and procedures",
]

# PA-specific positive indicator phrases (presence suggests purpose language)
PA_POSITIVE_TERMS = [
    "our purpose",
    "purpose is",
    "our mission",
    "mission is",
    "we exist to",
    "why we exist",
    "reason for existence",
    "we serve",
    "serve our",
    "we help",
    "we enable",
    "we empower",
    "we are committed to",
    "improve lives",
    "create impact",
    "make a difference",
    "customers",
    "communities",
    "stakeholders",
    "employees",
    "society",
]

# PA-specific negative/boilerplate terms (presence suggests non-purpose / noisy text)
PA_NEGATIVE_TERMS = [
    "brexit",
    "sanctions",
    "penalties",
    "regulatory requirements",
    "risk factors",
    "interest expense",
    "hedging instruments",
    "borrowing",
    "tax rates",
    "revenue recognition",
    "accounting policies",
    "derivatives",
    "securities act",
    "legal proceedings",
]


# =============================================================================
# Candidate ranking weights for PA pipeline
# =============================================================================

# Weights used to compute pa_candidate_rank_score (final RAG-like ranking)
PA_CANDIDATE_WEIGHTS = {
    "keyword": 0.35,
    "rag_norm": 0.25,
    "source_section_prior": 0.25,
    "context": 0.15,
}

# Penalty applied when negative boilerplate is present (subtracted)
NEGATIVE_BOILERPLATE_PENALTY = 0.30



# =============================================================================
# Year diagnostics
# =============================================================================

# Year stats are only diagnostic metadata.
# They do not directly affect PA score.
RECENT_YEAR_WINDOW = 2


# =============================================================================
# Source normalization aliases
# =============================================================================

SOURCE_ALIASES = {
    "official_web": [
        "official_web",
        "official web",
        "official website",
        "website",
        "web",
        "company website",
        "corporate website",
        "official",
    ],
    "edgar": [
        "edgar",
        "sec",
        "10-k",
        "10k",
        "10-q",
        "10q",
        "annual report",
        "filing",
        "form 10-k",
        "form 10-q",
    ],
    "linkedin": [
        "linkedin",
        "linked in",
        "li",
    ],
}


# =============================================================================
# Input column aliases
# =============================================================================

COLUMN_ALIASES = {
    "company": [
        "company",
        "company_name",
        "name",
        "ticker_company",
    ],
    "year": [
        "year",
        "filing_year",
        "report_year",
        "fiscal_year",
    ],
    "source": [
        "source",
        "data_source",
        "document_source",
    ],
    "section": [
        "section",
        "section_name",
        "heading",
        "source_section",
    ],
    "subsection": [
        "subsection",
        "sub_section",
        "subheading",
    ],
    "chunk_id": [
        "chunk_id",
        "id",
        "chunk_index",
    ],
    "doc_id": [
        "doc_id",
        "document_id",
        "filing_id",
    ],
    "text_clean": [
        "text_clean",
        "clean_text",
        "text",
        "chunk_text",
    ],
    "text_raw": [
        "text_raw",
        "raw_text",
        "original_text",
    ],
}


# =============================================================================
# LLM settings
# =============================================================================

# Default mock mode allows local end-to-end testing without an API key.
#
# To use OpenAI:
#   set PA_LLM_PROVIDER=openai
#   set OPENAI_API_KEY=your_key
#   optional: set OPENAI_MODEL=gpt-4o-mini
#
# To use Gemini:
#   set PA_LLM_PROVIDER=gemini
#   set GEMINI_API_KEY=your_key
#   optional: set GEMINI_MODEL=gemini-2.5-flash
#
# To use Claude:
#   set PA_LLM_PROVIDER=claude
#   set ANTHROPIC_API_KEY=your_key
#   optional: set CLAUDE_MODEL=claude-opus-4-1-20250805

LLM_PROVIDER = os.getenv("PA_LLM_PROVIDER", "mock").strip().lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-1-20250805")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

LLM_TEMPERATURE = float(os.getenv("PA_LLM_TEMPERATURE", "0.0"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("PA_LLM_MAX_OUTPUT_TOKENS", "800"))
LLM_MAX_RETRIES = int(os.getenv("PA_LLM_MAX_RETRIES", "2"))
LLM_FALLBACK_TO_MOCK = (
    os.getenv("PA_LLM_FALLBACK_TO_MOCK", "true").lower() == "true"
)


# =============================================================================
# PA keyword sets
# =============================================================================

PA_CORE_TERMS = [
    "purpose",
    "mission",
    "why we exist",
    "reason for existence",
    "reason we exist",
    "exist to",
    "we exist to",
    "our purpose",
    "our mission",
    "beyond profit",
    "serve",
    "serving",
    "stakeholders",
    "customers",
    "communities",
    "patients",
    "people",
    "society",
    "impact",
    "positive impact",
    "improve lives",
    "empower",
    "enable",
]

PA_SUPPORT_TERMS = [
    "vision",
    "values",
    "commitment",
    "long-term value",
    "sustainable",
    "responsibility",
    "trusted",
    "innovation",
    "access",
    "affordable",
    "healthy",
    "safe",
    "inclusive",
    "opportunity",
    "environment",
    "world",
    "future",
]

GENERIC_BRANDING_PHRASES = [
    "make the world a better place",
    "create value",
    "drive innovation",
    "best in class",
    "world class",
    "leading provider",
    "trusted partner",
    "delight customers",
    "empower people",
    "change the world",
    "unlock potential",
]

BOILERPLATE_TERMS = [
    "forward-looking statements",
    "risk factors",
    "safe harbor",
    "trademark",
    "copyright",
    "all rights reserved",
    "table of contents",
    "page intentionally left blank",
]
