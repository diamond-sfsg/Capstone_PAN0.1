from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 让脚本直接运行时也能找到 assign_bin
CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assign_bin.chunk_loader import load_all_sources
from assign_bin.chunk_splitter import split_records_into_chunks
from assign_bin.config import DATA_CLEAN_V2_DIR, STAND_COLUMNS
from assign_bin.postprocess import tag_similarity_relations


BASE_FILENAME = "unified_chunks_v3_base.csv"
FINAL_FILENAME = "unified_chunks_v3.csv"


def run_base(use_v2: bool = False, limit: int | None = None) -> Path:
    print("[INFO] Step = base")
    print("[INFO] Loading source documents...")
    records = load_all_sources(use_v2=use_v2)

    if limit is not None:
        records = records[:limit]
        print(f"[INFO] Debug limit applied: {limit} source records")

    print("[INFO] Splitting documents into chunks...")
    chunks = split_records_into_chunks(records)

    df = pd.DataFrame(chunks)
    if df.empty:
        raise ValueError("No chunks generated in base step.")

    # base 阶段只保证标准列存在
    df = df.reindex(columns=STAND_COLUMNS)

    out_path = DATA_CLEAN_V2_DIR / BASE_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Exported base chunks: {len(df)}")
    print(f"[INFO] Saved to: {out_path}")
    return out_path


def run_similarity() -> Path:
    print("[INFO] Step = similarity")

    in_path = DATA_CLEAN_V2_DIR / BASE_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(
            f"Base corpus not found: {in_path}\n"
            "Run with --step base first."
        )

    print(f"[INFO] Reading base corpus: {in_path}")
    df = pd.read_csv(in_path)

    print("[INFO] Tagging similarity relations...")
    df = tag_similarity_relations(df)

    extra_cols = [
        "is_exact_duplicate",
        "is_same_year_duplicate_like",
        "is_cross_year_similar",
        "similarity_scope",
    ]
    ordered_cols = STAND_COLUMNS + [c for c in extra_cols if c not in STAND_COLUMNS]

    df = df.reindex(columns=ordered_cols)

    out_path = DATA_CLEAN_V2_DIR / FINAL_FILENAME
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Saved tagged corpus: {out_path}")
    print(f"[INFO] Total rows: {len(df)}")
    return out_path


def run_report(final: bool = True) -> None:
    from assign_bin.report import print_chunk_report

    print("[INFO] Step = report")

    filename = FINAL_FILENAME if final else BASE_FILENAME
    path = DATA_CLEAN_V2_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {path}\n"
            "Run the required previous step first."
        )

    print(f"[INFO] Reading: {path}")
    df = pd.read_csv(path)
    print_chunk_report(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 pipeline for chunk normalization and similarity tagging."
    )
    parser.add_argument(
        "--step",
        choices=["base", "similarity", "report", "all"],
        required=True,
        help="Which step to run.",
    )
    parser.add_argument(
        "--use-v2",
        action="store_true",
        help="Load input files from clean_2.0 paths if supported by chunk_loader.",
    )
    parser.add_argument(
        "--report-on",
        choices=["base", "final"],
        default="final",
        help="For --step report only: choose which file to report on.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of source records for debugging (base/all only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.step == "base":
        run_base(use_v2=args.use_v2, limit=args.limit)

    elif args.step == "similarity":
        run_similarity()

    elif args.step == "report":
        run_report(final=(args.report_on == "final"))

    elif args.step == "all":
        run_base(use_v2=args.use_v2, limit=args.limit)
        run_similarity()
        run_report(final=True)


if __name__ == "__main__":
    main()