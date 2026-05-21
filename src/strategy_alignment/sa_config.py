# src/strategy_alignment/sa_config.py

"""
Configuration for Strategy & Source Alignment scoring.

SA follows the same new scoring logic as PA and HC:
- LLM rubric score is the primary score.
- Evidence quality factor is used as a mild adjustment.
- Overlap factor prevents duplicated evidence from inflating the score.
- The prompt must compare strategy evidence against the purpose extracted from PA.
"""

from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DATA_DIR = DATA_DIR / "clean_2.0"

PHASE_OUTPUT_DIR = DATA_DIR / "phase_sa"
PHASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CHUNKS_PATH = CLEAN_DATA_DIR / "unified_chunks_final_v4.csv"

# This is produced by the PA pipeline and now includes extracted_purpose.
PA_PURPOSE_REFERENCE_PATH = (
    DATA_DIR
    / "phase_pa"
    / "pa_evidence_detail_v1.csv"
)

SA_EVIDENCE_OUTPUT_PATH = PHASE_OUTPUT_DIR / "sa_evidence_details_v1.csv"
SA_SCORE_OUTPUT_PATH = PHASE_OUTPUT_DIR / "company_sa_score_v1.csv"
SA_DIAGNOSTICS_OUTPUT_PATH = PHASE_OUTPUT_DIR / "sa_diagnostics_v1.txt"


# ============================================================
# Dimension metadata
# ============================================================

DIMENSION_ID = "strategy_alignment"
DIMENSION_NAME = "Strategy & Source Alignment"

SA_QUESTIONS = {
    "SA_Q1": "R&D & Capital Alignment",
    "SA_Q2": "Operational Decision-Making",
}

QUESTION_ORDER = ["SA_Q1", "SA_Q2"]


# ============================================================
# Required input columns
# ============================================================

REQUIRED_CHUNK_COLUMNS = [
    "chunk_id",
    "company",
    "year",
    "source",
    "section",
    "text_clean",
]

OPTIONAL_CHUNK_COLUMNS = [
    "doc_id",
    "source_file",
    "subsection",
    "token_count",
    "char_count",
    "quality_flag",
    "duplicate_group",
    "similarity_scope",
]

# These fields are the normalized SA purpose-reference interface.
# They can be loaded from an older purpose-reference file or derived from
# PA evidence detail rows that contain extracted_purpose.
REQUIRED_PURPOSE_REFERENCE_COLUMNS = [
    "company",
    "purpose_statement_normalized",
    "purpose_confidence_0_1",
]

OPTIONAL_PURPOSE_REFERENCE_COLUMNS = [
    "extracted_purpose",
    "purpose_statement_raw",
    "served_stakeholders",
    "intended_impact",
    "reason_for_existence",
    "supporting_evidence_ids",
]


# ============================================================
# Evidence retrieval settings
# ============================================================

TOP_K_EVIDENCE_PER_QUESTION = 8
MIN_EVIDENCE_PER_QUESTION = 2

# SA is company-level for the new framework.
# We pool all available years for a company, then ask whether strategy and operations
# align with the company's stated purpose.
POOL_YEARS_BY_COMPANY = True

# If PA purpose extraction is weak, SA should be marked for review.
MIN_PURPOSE_CONFIDENCE_FOR_AUTO_SCORE = 0.50


# ============================================================
# Evidence base score formula
# ============================================================

# Same logic as PA/HC:
# base_evidence_score_0_1 =
#   0.40 * keyword_relevance
# + 0.35 * context_completeness
# + 0.25 * rag_similarity

EVIDENCE_BASE_SCORE_WEIGHTS = {
    "keyword_relevance": 0.40,
    "context_completeness": 0.35,
    "rag_similarity": 0.25,
}

# evidence_quality_factor = 0.70 + 0.30 * base_evidence_score_0_1
EVIDENCE_QUALITY_BASE = 0.70
EVIDENCE_QUALITY_SPAN = 0.30


# ============================================================
# SA overlap factor
# ============================================================

# SA has two questions, so overlap should not be deleted.
# But it should be softer than a hard evidence-exclusion rule because one
# strategy paragraph may reasonably support both strategic direction and
# operational embedding.
SA_OVERLAP_FACTOR = {
    "first_use": 1.00,
    "same_chunk_used_for_second_question": 0.75,
    "same_doc_different_chunk": 0.95,
    "same_year_near_duplicate": 0.80,
    "same_year_exact_duplicate": 0.65,
    "cross_year_recurring": 1.00,
    "unknown": 1.00,
}


# ============================================================
# Source prior for evidence ranking only
# ============================================================

# Source prior should affect retrieval ranking, not final LLM score.
# Strategy/capital evidence is more credible in filings and formal disclosures.
SA_SOURCE_PRIOR = {
    "edgar": 1.20,
    "official_web": 1.00,
    "linkedin": 0.70,
    "unknown": 1.00,
}


# ============================================================
# Question-specific keywords and retrieval queries
# ============================================================

SA_QUESTION_KEYWORDS = {
    "SA_Q1": {
        "core": [
            "research and development",
            "R&D",
            "capital allocation",
            "capital expenditure",
            "capex",
            "investment",
            "growth initiative",
            "strategic investment",
            "resource allocation",
            "innovation investment",
            "product investment",
            "technology investment",
            "long-term investment",
        ],
        "support": [
            "funding",
            "portfolio",
            "priority",
            "pipeline",
            "platform",
            "infrastructure",
            "capacity expansion",
            "acquisition",
            "venture",
            "partnership",
            "commercialization",
        ],
        "purpose_alignment_terms": [
            "aligned with our purpose",
            "supports our mission",
            "advance our mission",
            "deliver on our purpose",
            "consistent with our purpose",
            "reinforce our purpose",
            "create impact",
            "serve customers",
            "serve communities",
            "long-term value",
        ],
    },
    "SA_Q2": {
        "core": [
            "operational decision",
            "decision-making",
            "supplier selection",
            "supply chain",
            "product development",
            "market entry",
            "go-to-market",
            "business unit",
            "operating model",
            "operational framework",
            "process",
            "governance",
        ],
        "support": [
            "procurement",
            "vendor",
            "partner",
            "product design",
            "customer needs",
            "market expansion",
            "distribution",
            "quality control",
            "manufacturing",
            "service delivery",
            "employee training",
            "compliance",
        ],
        "purpose_alignment_terms": [
            "aligned with our purpose",
            "guided by our purpose",
            "supports our mission",
            "mission-driven",
            "purpose-led",
            "values-based decision",
            "stakeholder impact",
            "customer impact",
            "community impact",
            "measured and reported",
        ],
    },
}


SA_QUESTION_QUERIES = {
    "SA_Q1": (
        "Evidence that the company's R&D investment, capital expenditure, "
        "growth initiatives, strategic investments, or resource allocation "
        "reflect and reinforce its stated purpose."
    ),
    "SA_Q2": (
        "Evidence that the company's operational decisions, including supplier "
        "selection, product development, market entry, business operations, or "
        "decision-making frameworks, are aligned with its stated purpose."
    ),
}


# ============================================================
# Section priors for ranking
# ============================================================

SA_SECTION_PRIOR = {
    "strategy": 1.20,
    "business": 1.15,
    "management discussion": 1.15,
    "capital allocation": 1.20,
    "research and development": 1.20,
    "operations": 1.15,
    "products": 1.10,
    "supply chain": 1.10,
    "sustainability": 1.00,
    "ceo letter": 1.00,
    "about": 0.90,
    "risk factors": 0.65,
    "legal proceedings": 0.50,
    "unknown": 1.00,
}


# ============================================================
# Q-level aggregation
# ============================================================

# SA_Q_score =
#   0.70 * best_evidence_contribution
# + 0.30 * second_best_evidence_contribution

QUESTION_AGGREGATION_WEIGHTS = {
    "best_evidence": 0.70,
    "second_best_evidence": 0.30,
}

# First version: equal weight across two questions.
SA_FINAL_SCORE_WEIGHTS = {
    "SA_Q1": 0.50,
    "SA_Q2": 0.50,
}


# ============================================================
# Score bounds
# ============================================================

MIN_LLM_SCORE = 0.0
MAX_LLM_SCORE = 5.0

MIN_FINAL_SCORE = 0.0
MAX_FINAL_SCORE = 5.0


# ============================================================
# Prompt constraints
# ============================================================

SA_PROMPT_CORE_INSTRUCTION = """
You are not evaluating whether the company has a strong strategy in general.
You are evaluating whether the strategy or operational evidence is aligned with
the company's stated purpose.

Use the extracted purpose statement as the reference point. Compare the evidence
against the stakeholders served, the intended impact, and the reason the company
claims to exist.

Do not give a high score for generic strategy language such as growth, innovation,
market leadership, customer focus, or operational excellence unless the evidence
clearly connects those actions to the stated purpose.

Semantic alignment is acceptable; exact wording is not required.
If the evidence only supports financial performance, competitive positioning,
or brand image without connecting to the stated purpose, the score should remain
low or moderate.
""".strip()


# ============================================================
# Output schema
# ============================================================

SA_EVIDENCE_OUTPUT_COLUMNS = [
    "company",
    "question_id",
    "question_name",
    "chunk_id",
    "year",
    "source",
    "section",
    "text_clean",
    "keyword_relevance",
    "context_completeness",
    "rag_similarity",
    "base_evidence_score_0_1",
    "evidence_quality_factor",
    "overlap_factor",
    "llm_score_0_5",
    "evidence_contribution_0_5",
    "llm_reasoning",
    "extracted_purpose",
    "purpose_statement_normalized",
    "purpose_statement_raw",
    "purpose_confidence_0_1",
    "needs_human_review",
]

SA_SCORE_OUTPUT_COLUMNS = [
    "company",
    "extracted_purpose",
    "purpose_statement_normalized",
    "purpose_statement_raw",
    "purpose_confidence_0_1",
    "sa_q1_score_0_5",
    "sa_q2_score_0_5",
    "sa_final_score_0_5",
    "sa_score_0_100",
    "sa_needs_human_review",
    "sa_review_reason",
]
