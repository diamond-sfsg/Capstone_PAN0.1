from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "evidence_score1.0"
DATA_ROOT = PROJECT_ROOT / "data"
CLEAN_DATA_ROOT = DATA_ROOT / "clean_2.0"
PHASE2_OUTPUT_ROOT = DATA_ROOT / "phase2"

INPUT_CSV = CLEAN_DATA_ROOT / "unified_chunks_v3.csv"

# Create output dir if needed
PHASE2_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# General runtime config
# ------------------------------------------------------------------------------

TEXT_COL = "text_clean"
RAW_TEXT_COL = "text_raw"

MIN_TOKEN_COUNT = 20

# Drop rules for retrieval stage
DROP_QUALITY_FLAGS = {"garbled_text"}
DROP_TOO_LONG = True
TOO_LONG_FLAG = "too_long"

# Optional safeguard threshold if token_count exists
MAX_TOKEN_COUNT = 340

# Keep duplicates for now; only preserve the flag for later analysis
KEEP_DUPLICATES = True

# ------------------------------------------------------------------------------
# Required columns from unified_chunks_v3.csv
# ------------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "chunk_id",
    "doc_id",
    "company",
    "year",
    "source",
    "source_file",
    "section",
    "subsection",
    "text_raw",
    "text_clean",
    "token_count",
    "char_count",
    "is_short_text",
    "is_duplicate_like",
    "duplicate_group",
    "quality_flag",
    "normalize_version",
]

# ------------------------------------------------------------------------------
# Output schema for purpose_articulation evidence retrieval
# ------------------------------------------------------------------------------

PURPOSE_OUTPUT_COLUMNS = [
    "chunk_id",
    "doc_id",
    "company",
    "year",
    "source",
    "source_file",
    "section",
    "subsection",
    "token_count",
    "quality_flag",
    "is_duplicate_like",
    "retrieval_status",
    "text_clean",
    "lexical_score",
    "tfidf_score",
    "embedding_score",
    "metadata_score",
    "lexical_rank",
    "tfidf_rank",
    "embedding_rank",
    "metadata_rank",
]

# ------------------------------------------------------------------------------
# Embedding config
# ------------------------------------------------------------------------------

# Keep this here so later you can swap providers without touching other modules.
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"

# Whether current run should actually call embeddings.
# For scaffold stage, this can remain False.
ENABLE_EMBEDDING = False