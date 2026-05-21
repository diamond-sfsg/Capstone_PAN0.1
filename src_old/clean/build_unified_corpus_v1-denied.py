import json
import sqlite3
from pathlib import Path
from typing import List, Dict

import pandas as pd

CLEAN_DIR = Path("data/clean")
OUTPUT_CSV = CLEAN_DIR / "unified_corpus.csv"
OUTPUT_DB = CLEAN_DIR / "unified_corpus.db"

INPUT_FILES = [
    CLEAN_DIR / "edgar_clean.jsonl",
    CLEAN_DIR / "linkedin_clean.jsonl",
    CLEAN_DIR / "official_web_clean.jsonl",
]

TARGET_COLUMNS = [
    "doc_id",
    "chunk_id",
    "company",
    "ticker",
    "year",
    "source",
    "source_type",
    "doc_type",
    "section",
    "page_title",
    "url",
    "text",
    "metadata_json",
]


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"[INFO] Loaded {len(rows)} rows from {path.name}")
    return rows


def normalize_record(record: Dict) -> Dict:
    metadata = record.get("metadata", {})

    normalized = {
        "doc_id": record.get("doc_id"),
        "chunk_id": record.get("chunk_id"),
        "company": record.get("company"),
        "ticker": record.get("ticker"),
        "year": record.get("year"),
        "source": record.get("source"),
        "source_type": record.get("source_type"),
        "doc_type": record.get("doc_type"),
        "section": record.get("section"),
        "page_title": record.get("page_title"),
        "url": record.get("url"),
        "text": record.get("text"),
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
    }
    return normalized


def build_dataframe() -> pd.DataFrame:
    all_rows = []

    for path in INPUT_FILES:
        rows = read_jsonl(path)
        all_rows.extend([normalize_record(r) for r in rows])

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("[WARN] No records found across input files.")
        return df

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[TARGET_COLUMNS]

    # remove exact duplicate chunk_id if any
    if "chunk_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["chunk_id"])
        after = len(df)
        if before != after:
            print(f"[INFO] Dropped {before - after} duplicate chunk_id rows")

    return df


def save_csv(df: pd.DataFrame):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] CSV saved to {OUTPUT_CSV}")


def save_sqlite(df: pd.DataFrame):
    conn = sqlite3.connect(OUTPUT_DB)
    cur = conn.cursor()

    cur.execute("""
    DROP TABLE IF EXISTS documents
    """)

    cur.execute("""
    CREATE TABLE documents (
        doc_id TEXT,
        chunk_id TEXT PRIMARY KEY,
        company TEXT,
        ticker TEXT,
        year INTEGER,
        source TEXT,
        source_type TEXT,
        doc_type TEXT,
        section TEXT,
        page_title TEXT,
        url TEXT,
        text TEXT,
        metadata_json TEXT
    )
    """)

    df.to_sql("documents", conn, if_exists="append", index=False)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_company ON documents(company)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON documents(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_year ON documents(year)")

    conn.commit()
    conn.close()

    print(f"[DONE] SQLite DB saved to {OUTPUT_DB}")


def print_summary(df: pd.DataFrame):
    print("\n===== SUMMARY =====")
    print(f"Total rows: {len(df)}")

    if not df.empty:
        print("\nBy source:")
        print(df["source"].value_counts(dropna=False))

        print("\nBy company:")
        print(df["company"].value_counts(dropna=False))

        print("\nSample columns:")
        print(df.head(3)[["company", "source", "doc_type", "page_title", "year"]])


def run():
    df = build_dataframe()
    if df.empty:
        return

    save_csv(df)
    save_sqlite(df)
    print_summary(df)


if __name__ == "__main__":
    run()