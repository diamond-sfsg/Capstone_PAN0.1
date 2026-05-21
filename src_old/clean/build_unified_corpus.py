from __future__ import annotations

import argparse
import importlib
import json
import sqlite3
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List
import re
import xml.etree.ElementTree as ET

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SRC_OLD_ROOT = PROJECT_ROOT / "src_old"
for path in [SRC_OLD_ROOT, SRC_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from assign_bin.chunk_splitter import split_records_into_chunks
from assign_bin.config import DATA_CLEAN_V2_DIR, STAND_COLUMNS
from assign_bin.postprocess import tag_similarity_relations

VERSION = "v4"
CLEAN_DIR = DATA_CLEAN_V2_DIR
MAP_XLSX = PROJECT_ROOT / "data" / "company_map_seed_v4_with_tickers.xlsx"
MAP_CSV = CLEAN_DIR / "company_map_seed_v4.csv"

SOURCE_MODULES = [
    "clean.edgar_clean",
    "clean.linkedin_clean",
    "clean.official_web_clean",
]

INPUT_FILES = [
    CLEAN_DIR / f"edgar_clean_{VERSION}.jsonl",
    CLEAN_DIR / f"linkedin_clean_{VERSION}.jsonl",
    CLEAN_DIR / f"official_web_clean_{VERSION}.jsonl",
]

BASE_CSV = CLEAN_DIR / f"unified_chunks_final_base_{VERSION}.csv"
FINAL_CSV = CLEAN_DIR / f"unified_chunks_final_{VERSION}.csv"
FINAL_DB = CLEAN_DIR / f"unified_chunks_final_{VERSION}.db"


def _clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _xlsx_cell_text(cell: ET.Element, shared_strings: List[str], ns: Dict[str, str]) -> str:
    cell_type = cell.get("t")

    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", ns))

    value = cell.find("a:v", ns)
    if value is None or value.text is None:
        return ""

    text = value.text
    if cell_type == "s":
        try:
            return shared_strings[int(text)]
        except (ValueError, IndexError):
            return ""

    return text


def read_xlsx_table(path: Path) -> pd.DataFrame:
    """
    Read a simple one-sheet xlsx without requiring openpyxl.
    """
    if not path.exists():
        raise FileNotFoundError(f"Company map not found: {path}")

    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(path) as archive:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: List[Dict[int, str]] = []
    for row in sheet.findall(".//a:sheetData/a:row", ns):
        values: Dict[int, str] = {}
        for cell in row.findall("a:c", ns):
            ref = cell.get("r", "")
            match = re.match(r"([A-Z]+)", ref)
            if not match:
                continue

            col_idx = 0
            for char in match.group(1):
                col_idx = col_idx * 26 + (ord(char) - ord("A") + 1)

            values[col_idx - 1] = _xlsx_cell_text(cell, shared_strings, ns)
        rows.append(values)

    if not rows:
        return pd.DataFrame()

    max_col = max((max(row.keys()) for row in rows if row), default=-1)
    matrix = [[row.get(i, "") for i in range(max_col + 1)] for row in rows]
    header = [_clean_cell_text(value) for value in matrix[0]]
    return pd.DataFrame(matrix[1:], columns=header)


def normalize_company_key(value: Any) -> str:
    text = _clean_cell_text(value).lower()
    text = text.replace("_", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_company_ticker_map() -> Dict[str, str]:
    company_map = read_xlsx_table(MAP_XLSX)
    required = {"company", "canonical_company", "ticker"}
    missing = required.difference(company_map.columns)
    if missing:
        raise ValueError(f"Company map missing required columns: {sorted(missing)}")

    company_map = company_map.copy()
    for col in company_map.columns:
        company_map[col] = company_map[col].map(_clean_cell_text)

    MAP_CSV.parent.mkdir(parents=True, exist_ok=True)
    company_map.to_csv(MAP_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] Company map CSV refreshed from xlsx: {MAP_CSV} ({len(company_map)} rows)")

    aliases: Dict[str, str] = {}
    skipped = 0
    for _, row in company_map.iterrows():
        ticker = _clean_cell_text(row.get("ticker")).upper()
        if not ticker:
            skipped += 1
            continue

        for alias_col in ["company", "canonical_company", "ticker"]:
            alias = _clean_cell_text(row.get(alias_col))
            key = normalize_company_key(alias)
            if key:
                aliases[key] = ticker

    print(f"[INFO] Built {len(aliases)} company/ticker aliases; skipped {skipped} rows without ticker")
    return aliases


def map_company_to_ticker(company: Any, alias_map: Dict[str, str]) -> str | None:
    key = normalize_company_key(company)
    if not key:
        return None
    return alias_map.get(key)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Missing input file: {path}")
        return

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Bad JSON in {path.name}:{line_no}: {exc}")


def first_present(record: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def normalize_source_record(raw: Dict[str, Any], source_file: Path) -> Dict[str, Any]:
    text_raw = first_present(raw, ["text", "text_raw", "content", "body"])
    if text_raw is None and isinstance(raw.get("paragraphs"), list):
        text_raw = "\n".join(raw["paragraphs"])

    return {
        "doc_id": first_present(raw, ["doc_id", "id", "document_id"]),
        "company": first_present(raw, ["company", "ticker", "org", "name"]),
        "year": first_present(raw, ["year", "fiscal_year", "report_year"]),
        "source": first_present(raw, ["source"], source_file.stem.replace(f"_clean_{VERSION}", "")),
        "source_file": source_file.name,
        "section": first_present(raw, ["section", "heading", "title"]),
        "subsection": first_present(raw, ["subsection", "subheading", "doc_type"]),
        "text_raw": text_raw,
    }


def rebuild_source_files() -> None:
    for module_name in SOURCE_MODULES:
        print(f"[INFO] Running source cleaner: {module_name}")
        module = importlib.import_module(module_name)
        module.run_pipeline()


def load_source_records(alias_map: Dict[str, str], drop_unmapped: bool = True) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    unmapped_counts: Dict[str, int] = {}

    for path in INPUT_FILES:
        source_records = []
        for raw in read_jsonl(path):
            record = normalize_source_record(raw, path)
            original_company = record.get("company")
            ticker = map_company_to_ticker(original_company, alias_map)

            if ticker:
                record["company"] = ticker
            else:
                company_key = _clean_cell_text(original_company) or "<blank>"
                unmapped_counts[company_key] = unmapped_counts.get(company_key, 0) + 1
                if drop_unmapped:
                    continue

            source_records.append(record)

        source_records = [r for r in source_records if str(r.get("text_raw") or "").strip()]
        records.extend(source_records)
        print(f"[INFO] Loaded {len(source_records)} records from {path.name}")

    print(f"[INFO] Total source records: {len(records)}")
    if unmapped_counts:
        top_unmapped = sorted(unmapped_counts.items(), key=lambda item: item[1], reverse=True)[:20]
        print(f"[WARN] Unmapped source records skipped: {sum(unmapped_counts.values())}")
        for company, count in top_unmapped:
            print(f"       {company}: {count}")
    return records


def write_sqlite(df: pd.DataFrame) -> None:
    with sqlite3.connect(FINAL_DB) as conn:
        df.to_sql("chunks", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_company ON chunks(company)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(year)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")

    print(f"[DONE] SQLite saved to {FINAL_DB}")


def build_unified_chunks(limit: int | None = None, drop_unmapped: bool = True) -> pd.DataFrame:
    alias_map = load_company_ticker_map()
    records = load_source_records(alias_map=alias_map, drop_unmapped=drop_unmapped)
    if limit is not None:
        records = records[:limit]
        print(f"[INFO] Debug limit applied: {limit} source records")

    if not records:
        raise ValueError("No source records found. Rebuild source files first.")

    print("[INFO] Splitting records into normalized chunks...")
    chunks = split_records_into_chunks(records)
    base = pd.DataFrame(chunks)
    if base.empty:
        raise ValueError("No chunks generated from source records.")

    base = base.reindex(columns=STAND_COLUMNS)
    BASE_CSV.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(BASE_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] Base chunks saved to {BASE_CSV} ({len(base)} rows)")

    print("[INFO] Applying existing similarity tagging logic...")
    final = tag_similarity_relations(base)
    final = final.reindex(columns=STAND_COLUMNS)
    final.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] Final chunks saved to {FINAL_CSV} ({len(final)} rows)")

    write_sqlite(final)
    return final


def print_summary(df: pd.DataFrame) -> None:
    print("\n===== SUMMARY =====")
    print(f"Total chunks: {len(df)}")
    print("\nBy source:")
    print(df["source"].value_counts(dropna=False))
    print("\nBy year:")
    print(df["year"].value_counts(dropna=False).sort_index().tail(12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild source clean files and merge them into unified_chunks_final_v4."
    )
    parser.add_argument(
        "--skip-source-clean",
        action="store_true",
        help="Use existing *_clean_v4.jsonl files instead of rebuilding them from data/raw.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional source-record limit for a quick debug run.",
    )
    parser.add_argument(
        "--keep-unmapped",
        action="store_true",
        help="Keep records whose company cannot be mapped to a ticker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_source_clean:
        rebuild_source_files()

    df = build_unified_chunks(limit=args.limit, drop_unmapped=not args.keep_unmapped)
    print_summary(df)


if __name__ == "__main__":
    main()
