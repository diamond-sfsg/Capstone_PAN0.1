from pathlib import Path

# 项目根目录（自动推）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 数据路径
DATA_CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DATA_CLEAN_V2_DIR = PROJECT_ROOT / "data" / "clean_2.0"

# 文件路径
EDGAR_CLEAN_PATH = DATA_CLEAN_DIR / "edgar_clean.jsonl"
LINKEDIN_CLEAN_PATH = DATA_CLEAN_DIR / "linkedin_clean.jsonl"
OFFICIAL_WEB_CLEAN_PATH = DATA_CLEAN_DIR / "official_web_clean.jsonl"

EDGAR_CLEAN_V2_PATH = DATA_CLEAN_V2_DIR / "edgar_clean.jsonl"
LINKEDIN_CLEAN_V2_PATH = DATA_CLEAN_V2_DIR / "linkedin_clean.jsonl"
OFFICIAL_WEB_CLEAN_V2_PATH = DATA_CLEAN_V2_DIR / "official_web_clean.jsonl"

NORMALIZE_VERSION = "v2.0"

TARGET_CHUNK_TOKENS = 220
MIN_CHUNK_TOKENS = 80
MAX_CHUNK_TOKENS = 350
SHORT_TEXT_THRESHOLD = 80

STAND_COLUMNS = [
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
    "is_exact_duplicate",
    "is_same_year_duplicate_like",
    "is_cross_year_similar",
    "is_duplicate_like",
    "duplicate_group",
    "similarity_scope",
    "quality_flag",
    "normalize_version",
]