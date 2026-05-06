from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assign_bin.chunk_loader import load_all_sources
from assign_bin.chunk_splitter import split_records_into_chunks
from assign_bin.config import DATA_CLEAN_V2_DIR, STAND_COLUMNS
from assign_bin.postprocess import tag_similarity_relations
from assign_bin.report import print_chunk_report

from clean import edgar_clean, linkedin_clean, official_web_clean


BASE_OUTPUT = DATA_CLEAN_V2_DIR / "unified_chunks_v4_base.csv"
FINAL_OUTPUT = DATA_CLEAN_V2_DIR / "unified_chunks_v4.csv"


def run_clean_sources() -> None:
    print("[INFO] Cleaning EDGAR from data/edgar_by_type...")
    edgar_clean.run_pipeline()

    print("[INFO] Cleaning LinkedIn from data/raw/linkedin...")
    linkedin_clean.run_pipeline()

    print("[INFO] Cleaning official web from data/raw/official_web...")
    official_web_clean.run_pipeline()


def build_base_chunks(limit: int | None = None) -> pd.DataFrame:
    print("[INFO] Loading cleaned JSONL sources...")
    records = load_all_sources(use_v2=False)

    if limit is not None:
        records = records[:limit]
        print(f"[INFO] Debug limit applied: {limit} source records")

    print("[INFO] Splitting records into normalized chunks...")
    chunks = split_records_into_chunks(records)
    df = pd.DataFrame(chunks)

    if df.empty:
        raise ValueError("No chunks generated.")

    df = df.reindex(columns=STAND_COLUMNS)
    BASE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASE_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved base v4 chunks: {BASE_OUTPUT} ({len(df):,} rows)")
    return df


def tag_and_save_similarity(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Tagging duplicate/similarity relations...")
    tagged = tag_similarity_relations(df)

    extra_cols = [
        "is_exact_duplicate",
        "is_same_year_duplicate_like",
        "is_cross_year_similar",
        "similarity_scope",
    ]
    ordered_cols = STAND_COLUMNS + [c for c in extra_cols if c not in STAND_COLUMNS]
    tagged = tagged.reindex(columns=ordered_cols)
    tagged.to_csv(FINAL_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved final v4 chunks: {FINAL_OUTPUT} ({len(tagged):,} rows)")
    return tagged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild unified_chunks_v4.csv from edgar_by_type, linkedin, and official_web."
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Reuse existing data/clean/*.jsonl files instead of re-running source cleaners.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional source-record limit for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_clean:
        run_clean_sources()

    base_df = build_base_chunks(limit=args.limit)
    final_df = tag_and_save_similarity(base_df)

    print("[INFO] v4 corpus report:")
    print_chunk_report(final_df)


if __name__ == "__main__":
    main()
