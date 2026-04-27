from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_cleaning.cleaners.edgar import clean_edgar_files_by_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean EDGAR files by type while preserving original filenames.")
    parser.add_argument("--input", default="data/10Ks/edgar", help="EDGAR input directory.")
    parser.add_argument("--output", default="output/cleaned/edgar_by_type", help="Categorized cleaned output directory.")
    parser.add_argument("--progress", action="store_true", help="Show file-by-file progress percentage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = clean_edgar_files_by_type(Path(args.input), Path(args.output), show_progress=args.progress)
    print(f"Metadata JSON files: {summary.metadata_json_files}")
    print(f"Submissions JSON files: {summary.submissions_json_files}")
    print(f"HTML files: {summary.html_files}")
    print(f"TXT files: {summary.txt_files}")
    print(f"Output files: {summary.output_files}")
    print(f"Removed URLs: {summary.removed_url_count}")
    print(f"Removed XBRL/noise tokens: {summary.removed_xbrl_token_count}")


if __name__ == "__main__":
    main()
