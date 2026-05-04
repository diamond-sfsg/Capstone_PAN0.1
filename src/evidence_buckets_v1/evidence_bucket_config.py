from pathlib import Path

# config.py is located at:
# project_root/src/evidence_buckets_v1/evidence_bucket_config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "phase2" / "evidence_score_v1.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "phase2"
    / "evidence_buckets_v1"
)

BUCKET_CONFIG = {
    "purpose_articulation": {
        "short": "pa",
        "score_col": "pa_sum_score",
        "threshold": 0.90,
        "output_csv": "purpose_articulation_evidence_v1.csv",
        "output_db": "purpose_articulation_evidence_v1.db",
    },
    "history_consistency": {
        "short": "hc",
        "score_col": "hc_sum_score",
        "threshold": 0.95,
        "output_csv": "history_consistency_evidence_v1.csv",
        "output_db": "history_consistency_evidence_v1.db",
    },
    "strategy_alignment": {
        "short": "sa",
        "score_col": "sa_sum_score",
        "threshold": 0.85,
        "output_csv": "strategy_alignment_evidence_v1.csv",
        "output_db": "strategy_alignment_evidence_v1.db",
    },
}

AMBIGUITY_MARGIN = 0.08

ID_COL = "chunk_id"
DUPLICATE_GROUP_COL = "duplicate_group"

BASE_KEEP_COLS = [
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
    "quality_flag",
    "is_duplicate_like",
    "duplicate_group",
]

SCORE_KEEP_COLS = [
    "pa_sum_score",
    "hc_sum_score",
    "sa_sum_score",
    "pa_lexical_score",
    "pa_tfidf_score",
    "pa_embedding_score",
    "pa_metadata_score",
    "pa_prompt_score",
    "hc_lexical_score",
    "hc_tfidf_score",
    "hc_embedding_score",
    "hc_metadata_score",
    "hc_prompt_score",
    "hc_history_bonus_score",
    "sa_lexical_score",
    "sa_tfidf_score",
    "sa_embedding_score",
    "sa_metadata_score",
    "sa_prompt_score",
]